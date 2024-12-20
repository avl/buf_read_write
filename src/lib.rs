//! # bufstream2
//!
//! This crate contains the [`BufStream`] struct, a combination of [`std::io::BufReader`]
//! and [`std::io::BufWriter`].
//!
//! # Motivation
//!
//! When reading or writing files in rust, it's absolutely essential to wrap [`std::fs::File`]
//! in `BufReader` or `BufWriter`. Failure to do this can cause poor performance, at least if
//! data is written in small chunks. This is because each individual write becomes an operating
//! system call.
//!
//! Sone applications need to both read and write to the same file. Unfortunately, `BufReader`
//! only supports reading, and `BufWriter` only supports writing. The two cannot be easily
//! combined.
//!
//! This crate attempts to resolve this, by introducing a [`BufStream`] construct that allows
//! both buffered reading and writing.
//!
//! # Design decisions
//!
//! The following design decisions have been made for this crate:
//!
//! * It requires the underlying object to implement [`std::io::Read`], [`std::io::Write`],
//!   and [`std::io::Seek`]. The motivation for this is that reading and writing to the same
//!   file is mostly only useful together with seeking, and requiring this simplifies the
//!   design.
//!
//! * It shares the buffer between both reading and writing. This means that reads of
//!   data that has just previously been written will be satisfied directly from the buffer.
//!   It also means that writing one place in the file, then moving to a different place and
//!   reading, will invalidate the buffer (writing it back correctly to the backing
//!   implementation).
//!
//! * Bufstream2 is not a disk cache. Reads and writes larger than the buffer size will
//!   be satisfied by bypassing the buffer. The purpose of Bufstream2 is only to provide
//!   acceptable performance when doing small reads/writes.
//!
//! * Buffered reads assume the file is being traversed forward. Reading position 2000
//!   with a buffer size of 1000, will result in a call to the backing implementation of
//!   bytes `2000..3000`.
//!
//! * All writes behave like [`std::io::Write::write_all`]. This simplifies the implementation,
//!   and is often what you want for disk io (the main use case for this library). Even
//!   [`std::io::BufWriter`] will effectively do this when it is flushing its IO buffer.
//!
//! * Seeks are not always immediately passed on to the backing implementation. Instead, before
//!   each read, a seek is issued if required. This makes sense, since when the buffer needs
//!   to be flushed, extra seeks might otherwise be needed.
//!   NOTE! SeekFrom::End() *does* cause a flush and an immediate call to the backing
//!   implementation. This is due to the need for seeking to determine the end of the stream.
//!
//! * This crate does not attempt to support files larger than 2^64 bytes. Seeking this far
//!   is always impossible because of type ranges. But this crate additionally does not support
//!   writing beyond the end of this limit, even if no seeks occur. Because of how large 2^64
//!   is, this is unlikely to be a problem in practice.
//!
//! # Implementation
//!
//! * An extensive test suite exists, including automatic chaos testing, exhaustive testing
//!   for simple cases, and 'cargo mutants'-testing.
//!
//! * No unsafe code is used
//!
//! * Bufstream2 has no dependencies (apart from dev-dependencies)
//!
#![deny(missing_docs)]
#![deny(unsafe_code)]
extern crate core;

use std::cell::RefCell;
use std::io::{BufRead, Read, Seek, SeekFrom, Write};
use std::mem::forget;
use std::ops::Range;


const DEFAULT_BUF_SIZE: usize = 8192;


#[derive(Clone, Debug)]
struct MovingBuffer {
    offset: u64,
    data: Vec<u8>,
}

#[allow(unused)]
#[cfg(debug_assertions)]
macro_rules! debug_println {
    ($f:expr, $($a:expr),+) => {{
        println!($f, $($a),+ );
    }};
}

#[allow(unused)]
#[cfg(not(debug_assertions))]
macro_rules! debug_println {
    ($f:expr, $($a:expr),+) => {{}};
}

fn overlap(range1: Range<u64>, range2: Range<u64>) -> Option<Range<u64>> {
    if range1.end <= range2.start {
        return None;
    }
    if range2.end <= range1.start {
        return None;
    }
    Some(range1.start.max(range2.start)..range1.end.min(range2.end))
}

#[inline(always)]
fn checked_add(position: u64, size: usize) -> std::io::Result<u64> {
    position.checked_add(size.try_into()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Arithmetic overflow"))?)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Arithmetic overflow"))
}
#[inline(always)]
fn checked_add_usize(position: usize, size:usize) -> std::io::Result<usize> {
    position.checked_add(size)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Arithmetic overflow"))
}
#[inline(always)]
fn checked_sub_u64(position: u64, size:u64) -> std::io::Result<u64> {
    position.checked_sub(size)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Arithmetic underflow"))
}

#[inline]
fn to_usize(value: u64) -> std::io::Result<usize> {
    value.try_into()
        .map_err(|_err| std::io::Error::new(std::io::ErrorKind::InvalidData, "Arithmetic overflow"))
}
#[inline]
fn to_u64(value: usize) -> std::io::Result<u64> {
    value.try_into()
        .map_err(|_err| std::io::Error::new(std::io::ErrorKind::InvalidData, "Arithmetic overflow"))
}

struct DropGuard<'a>(&'a mut Vec<u8>);
impl Drop for DropGuard<'_> {
    fn drop(&mut self) {
        self.0.clear();
    }
}


impl MovingBuffer {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            offset: 0,
        }
    }

    #[inline]
    fn flush(
        &mut self,
        flusher: &mut impl FnMut(u64, &[u8]) -> Result<(), std::io::Error>,
    ) -> Result<(), std::io::Error> {
        if !self.data.is_empty() {
            flusher(self.offset, &self.data)?;
        }
        self.data.clear();
        Ok(())
    }
    #[inline(always)]
    fn end(&self) -> Result<u64, std::io::Error> {
        checked_add(self.offset, self.data.len())
    }
    #[inline(always)]
    fn write_at(
        &mut self,
        position: u64,
        data: &[u8],
        write_at: &mut impl FnMut(u64, &[u8]) -> Result<(), std::io::Error>,
    ) -> Result<(), std::io::Error> {
        // The following cannot overflow, because of how Vec works.
        let free_capacity = self.data.capacity() - self.data.len();

        if position == self.end()? && free_capacity >= data.len() {
            self.data.extend(data);
            Ok(())
        } else if position >= self.offset && checked_add(position, data.len())? <= self.end()? {
            let relative_offset = to_usize(checked_sub_u64(position, self.offset)?)?;
            self.data[relative_offset..checked_add_usize(relative_offset,data.len())?].copy_from_slice(data);

            Ok(())
        } else {
            self.flush(write_at)?;
            self.data.clear();
            if data.len() < self.data.capacity() {
                self.offset = position;
                self.data.extend(data);
                Ok(())
            } else {
                write_at(position, data)?;
                Ok(())
            }
        }
    }

    #[inline(always)]
    fn read_at<
        R: FnMut(u64, &mut [u8]) -> std::io::Result<usize>,
        W: FnMut(u64, &[u8]) -> std::io::Result<()>,
    >(
        &mut self,
        position: u64,
        buf: &mut [u8],
        read_at: &mut R,
        write_at: &mut W,
    ) -> std::io::Result<usize> {
        if buf.len() > self.data.capacity() {
            self.flush(write_at)?;
            return read_at(position, buf);
        }
        _ = checked_add(position, buf.len())?;

        #[inline(never)]
        fn inner_read_at<
            F: FnMut(u64, &mut [u8]) -> std::io::Result<usize>,
            W: FnMut(u64, &[u8]) -> std::io::Result<()>,
        >(
            position: u64,
            buf: &mut [u8],
            tself: &mut MovingBuffer,
            read_at: &mut F,
            write_at: &mut W,
        ) -> std::io::Result<usize> {
            if buf.is_empty() {
                return Ok(0);
            }

            tself.flush(write_at)?;
            let cap = tself.data.capacity();
            _ = checked_add(position, cap)?;
            tself.data.resize(cap, 0);
            tself.offset = position;

            let mut dropguard = DropGuard(&mut tself.data);
            let got = read_at(position, &mut dropguard.0)?;
            dropguard.0.truncate(got);
            forget(dropguard);
            let curgot = got.min(buf.len());
            buf[..curgot].copy_from_slice(&tself.data[0..curgot]);
            Ok(curgot)
        }

        let read_range = position..checked_add(position, buf.len())?;
        let buffered_range = self.offset..self.end()?;

        let buflen = buf.len();

        if read_range.end <= buffered_range.start {
            return inner_read_at(read_range.start, buf, self, read_at, write_at);
        }
        if read_range.start >= buffered_range.end {
            return inner_read_at(read_range.start, buf, self, read_at, write_at);
        }

        let mut got = 0;
        if read_range.start < buffered_range.start {
            let len = to_usize(buffered_range.start - read_range.start)?.min(buflen);
            got = read_at(read_range.start, &mut buf[0..len])?;
            if got < len {
                return Ok(got);
            }
        }

        if let Some(overlap) = overlap(read_range.clone(), buffered_range.clone()) {
            let overlapping_src_slice =
                &self.data[to_usize(overlap.start - self.offset)?..to_usize(overlap.end - self.offset)?];
            buf[to_usize(overlap.start - position)?..to_usize(overlap.end - position)?]
                .copy_from_slice(overlapping_src_slice);
            got = checked_add_usize(got, overlapping_src_slice.len())?;
        }

        if read_range.end > buffered_range.end {
            let got2 = inner_read_at(
                buffered_range.end,
                &mut buf[buflen - to_usize(read_range.end - buffered_range.end)?..],
                self,
                read_at,
                write_at,
            )?;
            got = checked_add_usize(got, got2)?;
        }
        Ok(got)
    }
}

/// Buffering reader/writer
///
/// See crate documentation for more details!
#[derive(Debug)]
pub struct BufStream<T> {
    buffer: MovingBuffer,
    position: u64,
    inner_position: u64,
    inner: T,
}

impl<T> BufStream<T> {
    #[cfg(test)]
    pub(crate) fn clone(&self) -> Self
    where
        T: Clone,
    {
        BufStream {
            buffer: self.buffer.clone(),
            position: self.position,
            inner: self.inner.clone(),
            inner_position: self.inner_position,
        }
    }
}


#[inline]
fn obtain_stream_position<T:Seek>(inner: &mut T, inner_position: &mut u64) -> std::io::Result<u64> {
    if *inner_position != u64::MAX {
        debug_assert_eq!(*inner_position, inner.stream_position()?);
        return Ok(*inner_position);
    }
    *inner_position = inner.stream_position()?;
    Ok(*inner_position)
}

impl<T:Read+Write+Seek+std::fmt::Debug> BufRead for BufStream<T> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        let buf_end = checked_add(self.buffer.offset, self.buffer.data.len())?;
        if self.position >= self.buffer.offset && self.position < buf_end {
            let usable = to_usize(buf_end - self.position)?;
            debug_assert!(usable > 0);
            let buf_offset = to_usize(checked_sub_u64(self.position, self.buffer.offset)?)?;
            return Ok(&self.buffer.data[buf_offset..checked_add_usize(buf_offset, usable)?]);
        }
        self.flush_write()?;
        let cap = self.buffer.data.capacity();
        self.buffer.data.resize(cap, 0);
        self.buffer.offset = self.position;
        debug_assert!(self.buffer.data.len() > 0);
        let mut dropguard = DropGuard(&mut self.buffer.data);


        if obtain_stream_position(&mut self.inner, &mut self.inner_position)? != self.position {
            self.inner.seek(SeekFrom::Start(self.position))?;
        }
        self.inner_position = u64::MAX;

        let got = self.inner.read(&mut dropguard.0)?;
        dropguard.0.truncate(got);
        self.inner_position = checked_add(self.position, got)?;
        forget(dropguard);
        dbg!(got);
        Ok(&self.buffer.data)
    }

    fn consume(&mut self, amt: usize) {
        self.position = checked_add(self.position, amt)
            .expect("u64::MAX offset cannot be exceeded");
    }
}

impl <T: Write + Seek> BufStream<T> {
    #[cold]
    #[inline(never)]
    fn write_cold(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buffer.write_at(self.position, buf,

                             &mut |pos, data| {
                                 if
                                 obtain_stream_position(&mut self.inner, &mut self.inner_position)? != pos as u64 {
                                     let t = self.inner.seek(SeekFrom::Start(pos as u64));
                                     t?;
                                 }
                                 self.inner_position = u64::MAX;
                                 let t = self.inner.write_all(data);
                                 t?;
                                 self.inner_position = checked_add(pos, data.len())?;
                                 Ok(())
                             })?;

        self.position = self.position.checked_add(to_u64(buf.len())?)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Arithmetic overflow"))?;

        Ok(buf.len())
    }
}

impl<T: Write + Seek> Write for BufStream<T> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let free_capacity = self.buffer.data.capacity() - self.buffer.data.len();
        if self.position == self.buffer.offset+self.buffer.data.len() as u64 && free_capacity >= buf.len() {
            self.buffer.data.extend(buf);
            self.position = checked_add(self.position, buf.len())?;
            return Ok(buf.len());
        }
        self.write_cold(buf)
    }


    fn flush(&mut self) -> std::io::Result<()> {
        self.flush_write()?;
        self.inner.flush()
    }
}

impl<T: Write + Seek> BufStream<T> {
    fn flush_write(&mut self) -> Result<(), std::io::Error> {
        let t = self.buffer.flush(&mut |offset, data| {
            if offset != obtain_stream_position(&mut self.inner, &mut self.inner_position)?
            {
                self.inner.seek(SeekFrom::Start(offset as u64))?;
            }
            self.inner_position = u64::MAX;

            self.inner.write_all(data)?;
            self.inner_position = checked_add(offset, data.len())?;
            Ok(())
        });
        t?;
        Ok(())
    }

    /// Crate a new instance, with the given buffer size
    pub fn with_capacity(inner: T, capacity: usize) -> Self {
        Self {
            buffer: MovingBuffer::with_capacity(capacity),
            position: 0,
            inner_position: u64::MAX,
            inner,
        }
    }

    /// Crate an instance with a default buffer size
    pub fn new(inner: T) -> Self {
        Self::with_capacity(inner, DEFAULT_BUF_SIZE)
    }
}

impl<T: Write+Seek> Seek for BufStream<T> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match pos {
            SeekFrom::Start(pos) => {
                self.position = pos;
            }
            SeekFrom::End(e) => {
                self.flush_write()?;
                let pos = self.inner.seek(SeekFrom::End(e))?;
                self.inner_position = pos;
                self.position = pos;
            }
            SeekFrom::Current(delta) => {
                self.position = self.position.checked_add_signed(delta )
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "Seek index out of range"))?;
            }
        }
        Ok(self.position as u64)
    }
}
impl<T: Read + Seek + Write> BufStream<T> {
    #[cold]
    #[inline(never)]
    fn read_cold(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let inner = RefCell::new(&mut self.inner);
        let inner_position = RefCell::new(&mut self.inner_position);
        let got = self.buffer.read_at(
            self.position,
            buf,
            &mut |pos, data| {
                let mut inner = inner.borrow_mut();
                if inner.stream_position()? != pos as u64 {
                    inner.seek(SeekFrom::Start(pos as u64))?;
                }
                **inner_position.borrow_mut() = u64::MAX;
                let got = inner.read(data)?;
                **inner_position.borrow_mut() = checked_add(pos, got)?;
                debug_assert!(got <= data.len());
                Ok(got)
            },
            &mut |offset, data| {
                let mut inner = inner.borrow_mut();
                if offset != inner.stream_position()? {
                    inner.seek(SeekFrom::Start(offset as u64))?;
                }
                **inner_position.borrow_mut() = u64::MAX;

                inner.write_all(data)?;

                **inner_position.borrow_mut() = checked_add(offset, data.len())?;
                Ok(())
            },
        )?;
        debug_assert!(got <= buf.len());
        self.position = checked_add(self.position,got)?;
        Ok(got)
    }

}

impl<T: Read + Seek + Write> Read for BufStream<T> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if let Some(offset) = self.position.checked_sub(self.buffer.offset) {
            if (offset as u64 ) < self.buffer.data.len().saturating_sub(buf.len()) as u64 {
                let offset = offset as usize;
                buf.copy_from_slice(&self.buffer.data[offset..offset+buf.len()]);

                self.position = checked_add(self.position, buf.len())?;

                return Ok(buf.len());
            }
        }
        self.read_cold(buf)
    }
}

#[cfg(test)]
mod tests {
    use std::io::ErrorKind;
    use super::*;
    use rand::{Rng, RngCore};
    use std::panic;
    use std::panic::AssertUnwindSafe;

    #[derive(Default, PartialEq, Eq, Debug, Clone)]
    struct FakeStream {
        buf: Vec<u8>,
        position: usize,
        short_read_by: usize,
        panic_after: usize,
        err_after: usize,
    }
    impl FakeStream {
        fn repair(&mut self) {
            self.short_read_by = 0;
            self.panic_after = 0;
            self.err_after = 0;
        }
        fn maybe_panic(&mut self) -> std::io::Result<()> {
            if self.panic_after >= 1 {
                self.panic_after -= 1;
                if self.panic_after == 0 {
                    panic!("Panic")
                }
            }
            if self.err_after >= 1 {
                self.err_after -= 1;
                if self.err_after == 0 {
                    return Err(std::io::Error::new(ErrorKind::Other, "Error"));
                }
            }
            Ok(())
        }
    }
    impl Read for FakeStream {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.maybe_panic()?;

            let mut to_read = buf.len();

            if to_read > 1 && self.short_read_by > 0 {
                to_read = (to_read - self.short_read_by).max(1);
            }
            let end = (self.position + to_read).min(self.buf.len());

            let got = end.saturating_sub(self.position);
            if got == 0 {
                return Ok(0);
            }
            buf[0..got].copy_from_slice(&self.buf[self.position..self.position + got]);
            self.position += got;
            Ok(got)
        }
    }
    impl Write for FakeStream {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.maybe_panic()?;
            for b in buf {
                if self.position >= self.buf.len() {
                    assert!(self.position <= self.buf.len());
                    self.buf.push(*b);
                } else {
                    self.buf[self.position] = *b;
                }
                self.position += 1;
            }
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.maybe_panic()?;
            Ok(())
        }
    }
    impl Seek for FakeStream {
        fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
            self.maybe_panic()?;
            match pos {
                SeekFrom::Start(s) => {
                    self.position = s as usize;
                }
                SeekFrom::End(e) => {
                    self.position = self.buf.len() - e as usize;
                }
                SeekFrom::Current(c) => {
                    self.position = self.position.checked_add_signed(c as isize).unwrap();
                }
            }
            Ok(self.position as u64)
        }
    }

    fn run_exhaustive_conf(bufsize: usize, ops: &[(usize, usize)], mut databyte: u8) {
        let mut good = FakeStream::default();
        let cut_inner = FakeStream::default();

        let mut cut = BufStream::with_capacity(cut_inner, bufsize);

        for (op, param) in ops.iter().copied() {
            match op {
                0 if good.buf.len() > 0 => {
                    let seek_to = param;
                    debug_println!("==SEEK to {}", seek_to);
                    good.seek(SeekFrom::Start(seek_to as u64)).unwrap();
                    cut.seek(SeekFrom::Start(seek_to as u64)).unwrap();
                }
                1 => {
                    let read_bytes = param / 2;
                    let short_read = param % 2;
                    debug_println!("==READ {}", read_bytes);

                    let mut goodbuf = vec![0u8; read_bytes];
                    let short_read = if good.position + 1 < good.buf.len() && read_bytes > 1 {
                        short_read
                    } else {
                        0
                    }; //Can't have a short read when at the end! It's by definition not a short read if there actually wasn't anything to read.
                    good.short_read_by = short_read;
                    let good_position = good.position;
                    let goodgot = good.read(&mut goodbuf).unwrap();

                    if good_position + goodbuf.len() <= good.buf.len() {
                        assert_eq!(goodgot + short_read, goodbuf.len());
                    }

                    let mut cutbuf = vec![0u8; read_bytes];
                    cut.inner.short_read_by = short_read;
                    let cutgot = cut.read(&mut cutbuf).unwrap();

                    let gotmin = cutgot.min(goodgot);
                    assert_eq!(&goodbuf[..gotmin], &cutbuf[..gotmin]);
                    debug_println!("did READ {} -> {:?}", read_bytes, cutbuf);
                    if cutgot != goodgot {
                        good.position = cut.position as usize;
                    }
                }
                0 | 2 => {
                    let write_bytes = param;

                    let mut buf = vec![0u8; write_bytes];
                    for i in 0..write_bytes {
                        buf[i] = databyte;
                        databyte = databyte.wrapping_add(17);
                    }
                    debug_println!("==WRITE {} {:?}", buf.len(), buf);
                    let goodgot = good.write(&buf).unwrap();
                    let cutgot = cut.write(&buf).unwrap();
                    assert_eq!(goodgot, cutgot);
                    assert_eq!(goodgot, buf.len());
                }
                _ => unreachable!(),
            }
        }

        cut.flush().unwrap();
        assert_eq!(cut.buffer.data.capacity(), bufsize);
        assert_eq!(&good.buf, &cut.inner.buf);
        assert_eq!(&good.position, &(cut.position as usize));
    }

    #[test]
    fn exhaustive() {
        let mut databyte = 0;
        for bufsize in [1, 3, 7] {
            for first_op in 0..3 {
                let first_op_param_options = if first_op != 0 { 12 } else { 2 };
                for first_op_param in 0..first_op_param_options {
                    for second_op in 0..3 {
                        let second_op_param_options = if second_op != 0 { 12 } else { 2 };
                        for second_op_param in 0..second_op_param_options {
                            for third_op in 0..3 {
                                let third_op_param_options = if third_op != 0 { 12 } else { 2 };
                                for third_op_param in 0..third_op_param_options {
                                    for fourth_op in 0..3 {
                                        let fourth_op_param_options =
                                            if fourth_op != 0 { 12 } else { 2 };
                                        for fourth_op_param in 0..fourth_op_param_options {
                                            debug_println!(
                                                "\n\n========Iteration {} {} {} {} {} {} {} {} {} {}===========",
                                                bufsize,
                                                first_op,
                                                first_op_param,
                                                second_op,
                                                second_op_param,
                                                third_op,
                                                third_op_param,
                                                fourth_op,
                                                fourth_op_param,
                                                databyte
                                            );
                                            run_exhaustive_conf(
                                                bufsize,
                                                &[
                                                    (first_op, first_op_param),
                                                    (second_op, second_op_param),
                                                    (third_op, third_op_param),
                                                    (fourth_op, fourth_op_param),
                                                ],
                                                databyte,
                                            );
                                            databyte = databyte.wrapping_add(1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn exhaustive_regress() {
        let case = "7 2 6 1 6 0 1 1 6 13";

        let mut items = case.split(" ").map(|x| x.parse::<usize>().unwrap());
        let mut n = move || items.next().unwrap();

        run_exhaustive_conf(
            n(),
            &[(n(), n()), (n(), n()), (n(), n()), (n(), n())],
            n() as u8,
        );
    }

    #[test]
    fn fuzz_many() {
        for i in 0..1000000 {
            fuzz(i, Some(3), Some(1), false);
            fuzz(i, Some(1), Some(3), false);
            fuzz(i, Some(10), Some(15), false);
            fuzz(i, Some(15), Some(10), false);
            fuzz(i, None, None, true);
        }
    }

    #[test]
    fn regression() {
        fuzz(0, Some(15), Some(10), false);
    }

    #[test]
    fn writes_are_buffered() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, 100);
        cut.write(&[1, 2, 3]).unwrap();
        assert!(cut.inner.buf.is_empty());
    }

    #[test]
    fn writes_are_buffered_after_seek() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, 100);
        cut.write(&[1, 2, 3]).unwrap();
        cut.seek(SeekFrom::Start(10)).unwrap();
        cut.write(&[4, 5, 6]).unwrap();
        assert_eq!(cut.inner.buf, [1, 2, 3]); //This should have been flushed

        assert_eq!(cut.buffer.data, [4, 5, 6]); //This should have been flushed
        assert_eq!(cut.buffer.offset, 10); //This should have been flushed
    }

    #[test]
    fn reads_are_buffered() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, 100);
        cut.write(&[1, 2, 3, 4, 5]).unwrap();
        cut.seek(SeekFrom::Start(0)).unwrap();
        let mut temp = [0, 0, 0];
        cut.read(&mut temp).unwrap();
        assert_eq!(temp, [1, 2, 3]);

        cut.inner.buf.clear(); //Clear the inner buf

        cut.seek(SeekFrom::Start(1)).unwrap();
        let mut temp2 = [0, 0];
        cut.read(&mut temp2).unwrap(); //This can now only succeed from the buffer

        assert_eq!(temp2, [2, 3]); //If this succeeds, reads are buffered
    }

    #[test]
    fn seek_plus_read_will_still_use_buffer() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, 3);
        cut.write(&[1, 2, 3, 4, 5, 7, 8, 9, 10]).unwrap();

        cut.seek(SeekFrom::Start(7)).unwrap();
        cut.buffer.data.clear();
        cut.read(&mut [0, 0, 0]).unwrap();
        assert_eq!(cut.buffer.offset, 7);

        cut.seek(SeekFrom::Start(0)).unwrap();
        cut.inner.short_read_by = 1;

        let mut buf = [0, 0, 0];
        let got = cut.read(&mut buf).unwrap();
        assert_eq!(got, 2); //Short read
        assert_eq!(cut.buffer.data, [1, 2]); //New read should have populated cache
        assert_eq!(cut.buffer.offset, 0);
    }

    fn catch<R>(f: &mut dyn FnMut() -> std::io::Result<R>) -> std::io::Result<R> {
        let f = AssertUnwindSafe(f);
        match panic::catch_unwind(|| f()) {
            Ok(ok) => ok,
            Err(panic) => Result::Err(std::io::Error::new(
                ErrorKind::Other,
                format!("panic: {:?}", panic),
            )),
        }
    }

    fn recreate_from(cut: &BufStream<FakeStream>) -> FakeStream {
        let mut temp = cut.inner.clone();

        for (i, b) in cut.buffer.data.iter().enumerate() {
            let i = i + cut.buffer.offset as usize;
            if temp.buf.len() <= i {
                temp.buf.resize(i + 1, 0);
            }
            temp.buf[i] = *b;
        }
        temp.position = cut.position as usize;
        temp
    }

    fn fuzz(seed: u64, buffer_size: Option<usize>, write_sizes: Option<usize>, bufread: bool) {
        let mut small_rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let buffer_size = buffer_size.unwrap_or(small_rng.gen_range(1..10));
        let write_sizes = write_sizes.unwrap_or(small_rng.gen_range(1..10));
        let mut good = FakeStream::default();
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, buffer_size);
        use rand::SeedableRng;
        debug_println!(
            "\n\n==== Seed: {}, buffer: {}, write size: {} ====",
            seed,
            buffer_size,
            write_sizes
        );
        for _ in 0..7 {
            let mut panic_or_err = false;
            if small_rng.gen_bool(0.05) {
                good.panic_after = 1;
                cut.inner.panic_after = 1;
                panic_or_err = true;
            } else if small_rng.gen_bool(0.05) {
                good.err_after = 1;
                cut.inner.err_after = 1;
                panic_or_err = true;
            }
            match small_rng.gen_range(0..3) {
                0 if good.buf.len() > 0 => {
                    let seek_to = small_rng.gen_range(0..good.buf.len());

                    debug_println!("==SEEK to {}", seek_to);
                    let re_good = catch(&mut || good.seek(SeekFrom::Start(seek_to as u64)));
                    let re_cut = catch(&mut || cut.seek(SeekFrom::Start(seek_to as u64)));

                    good.repair();
                    cut.inner.repair();
                    match (re_good, re_cut) {
                        (Ok(g), Ok(c)) => {
                            assert_eq!(c, g);
                        }
                        (_, Ok(_)) | (Err(_), Err(_)) if panic_or_err => {
                            good.seek(SeekFrom::Start(cut.position as u64)).unwrap();
                        }
                        (g, c) => {
                            panic!(
                                "Unexpected results: {:?}, {:?} (panicking: {:?})",
                                g, c, panic_or_err
                            );
                        }
                    }
                }
                1 => {
                    //Read
                    let read_bytes = small_rng.gen_range(0..write_sizes);
                    let short_read = small_rng.gen_bool(0.3) as usize;
                    debug_println!("==READ {}", read_bytes);
                    let mut goodbuf = vec![0u8; read_bytes];
                    good.short_read_by = short_read;
                    let good_got = catch(&mut || good.read(&mut goodbuf));

                    let mut cutbuf = vec![0u8; read_bytes];
                    cut.inner.short_read_by = short_read;

                    let cut_got;
                    if bufread {
                        cut_got = catch(&mut || {
                            let cutbuflen = cutbuf.len();
                            dbg!(&cut);
                            dbg!(cutbuflen);
                            let fillbuf = cut.fill_buf()?;
                            let data = &fillbuf[0..cutbuflen.min(fillbuf.len())];
                            cutbuf[0..data.len()].copy_from_slice(data);
                            let len = data.len();
                            cut.consume(len);
                            Ok(len)
                        });

                    } else {
                        cut_got = catch(&mut || cut.read(&mut cutbuf));
                    }

                    debug_println!(
                        "did READ {:?}/{} -> {:?} (short-read: {})",
                        cut_got,
                        read_bytes,
                        cutbuf,
                        short_read
                    );
                    match (good_got, cut_got) {
                        (Ok(good_got), Ok(cut_got)) => {
                            if good_got > 0 {
                                assert!(cut_got > 0);
                            }
                            if short_read == 0 && good.position + read_bytes <= good.buf.len() && !bufread {
                                assert_eq!(cut_got, good_got);
                                assert_eq!(cut_got, read_bytes);
                            }
                            let mingot = cut_got.min(good_got);
                            if cut_got != good_got {
                                good.position = cut.position as usize;
                            }
                            assert_eq!(goodbuf[0..mingot], cutbuf[0..mingot]);
                        }
                        (_, Ok(_)) | (Err(_), Err(_)) if panic_or_err => {
                            good.seek(SeekFrom::Start(cut.position as u64)).unwrap();
                        }
                        (g, c) => {
                            panic!(
                                "Unexpected read results: {:?}, {:?} (panicking: {:?})",
                                g, c, panic_or_err
                            );
                        }
                    }
                }
                0 | 2 => {
                    // Write
                    let write_bytes = small_rng.gen_range(0..write_sizes);
                    let mut buf = vec![0u8; write_bytes];
                    small_rng.fill_bytes(&mut buf);
                    debug_println!("==WRITE {} {:?}", buf.len(), buf);
                    let good_got = catch(&mut || good.write(&buf));
                    let cut_got = catch(&mut || cut.write(&buf));
                    match (good_got, cut_got) {
                        (Ok(good_got), Ok(cut_got)) => {
                            assert_eq!(good_got, cut_got);
                            assert_eq!(good_got, write_bytes);
                        }
                        (_, Ok(_)) | (Err(_), Err(_)) if panic_or_err => {
                            good = recreate_from(&cut);
                        }
                        (g, c) => {
                            panic!(
                                "Unexpected write results: {:?}, {:?} (panicking: {:?})",
                                g, c, panic_or_err
                            );
                        }
                    }
                }
                _ => unreachable!(),
            }
            cut.inner.repair();
            good.repair();
            debug_println!("Good state: {:?}", good);
            debug_println!("Cut state: {:?}", cut);
            let mut cut_cloned = cut.clone();
            cut_cloned.flush().unwrap();
            assert_eq!(&good.buf, &cut_cloned.inner.buf);
            assert_eq!(&good.position, &(cut_cloned.position as usize));
        }
    }
}
