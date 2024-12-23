//! # buf_read_write
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
//! * buf_read_write is not a disk cache. Reads and writes larger than the buffer size will
//!   be satisfied by bypassing the buffer. The purpose of buf_read_write is only to provide
//!   acceptable performance when doing small reads/writes.
//!
//! * Buffered reads assume the file is being traversed forward. Reading position 2000
//!   with a buffer size of 1000, will result in a call to the backing implementation of
//!   bytes `2000..3000`. Reading 1000 bytes at position 2000, then immediately after reading
//!   1000 bytes at position 1999, will result in two reads of 1000 bytes. Note, the results
//!   will still be correct for any seek pattern, it's just that forward operation is faster.
//!
//! * All writes behave like [`std::io::Write::write_all`]. This simplifies the implementation,
//!   and is often what you want for disk io (the main use case for this library).
//!   ([`std::io::BufWriter`] also effectively does this when it is flushing its IO buffer).
//!
//! * Seeks are not always immediately passed on to the backing implementation. Instead, before
//!   each read, a seek is issued if required. This makes sense, since when the buffer needs
//!   to be flushed, extra seeks might otherwise be needed.
//!   NOTE! SeekFrom::End() *does* cause a flush and an immediate call to the backing
//!   implementation. This is due to the need for seeking to determine the end of the stream.
//!
//! * This crate does not attempt to support files larger than 2^64 bytes. Seeking directly this
//!   far is always impossible because of type ranges. But this crate additionally does not support
//!   writing beyond the end of this limit, even if no seeks occur. Because of how large 2^64
//!   is, this is unlikely to be a problem in practice.
//!
//! * This crate does not panic itself, but the backing implementation may panic. Such panics
//!   are handled gracefully. In general, the particular effect of any panicking operation is
//!   that is may have completed to some arbitrary degree, but cannot have 'unexpected' effects.
//!   I.e, panics do not uncover unsound behaviour in this library.
//!
//! * The underlying IO operations can fail. If this happens, naturally, IO may not be written
//!   properly. Retrying [`Write::flush`] is supported, and if the backing implementation recovers,
//!   so will BufStream. I.e, the internal state of BufStream is not corrupted by the backing
//!   implementation failing or panicking.
//!
//! * This crate relies on unwinding for correctness. Unwinding is guaranteed by rust, so this
//!   is not a limitation. Calling [`std::process::abort`] will lead to data loss, buffers will
//!   not be flushed in this case.
//!
//! # Implementation
//!
//! * An extensive test suite exists, including automatic chaos testing, exhaustive testing
//!   for simple cases, and 'cargo mutants'-testing.
//!
//! * No unsafe code is used
//!
//! * buf_read_write has no dependencies (apart from dev-dependencies)
//!
//! * Note that when mixing writes, reads and seeks, the buffer will be reused.
//!   The dirty region of the buffer is tracked using a simple range. A consequence of this
//!   is that if a large chunk is read, and a single byte is modified at the head and tail
//!   of this chunk, when the buffer is flushed, the entire buffer will be written to the
//!   backing implementation.
//!   For disk IO, this can be acceptable, since writing a whole buffer may be equally
//!   fast as writing two smaller buffers. If this behavior is not desired, consider
//!   flushing the buffer between such writes.
//!
#![deny(missing_docs)]
#![deny(unsafe_code)]
extern crate core;

use std::cell::RefCell;
use std::fs::File;
use std::io::{BufRead, Read, Seek, SeekFrom, Write};
use std::mem::forget;
use std::ops::Range;

const DEFAULT_BUF_SIZE: usize = 8192;

#[derive(Clone, Debug)]
struct MovingBuffer {
    offset: u64,
    dirty: Range<usize>,
    data: Vec<u8>,
}

#[allow(unused)]
#[cfg(debug_assertions)]
macro_rules! debug_println {
    ($f:expr, $($a:expr),+) => {{
        //Enable this if you need to debug
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
    position
        .checked_add(size.try_into().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Arithmetic overflow")
        })?)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Arithmetic overflow"))
}

#[inline(always)]
fn checked_add_usize(position: usize, size: usize) -> std::io::Result<usize> {
    position
        .checked_add(size)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Arithmetic overflow"))
}
#[inline(always)]
fn checked_sub_u64(position: u64, size: u64) -> std::io::Result<u64> {
    position
        .checked_sub(size)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Arithmetic underflow"))
}

#[inline]
fn to_usize(value: u64) -> std::io::Result<usize> {
    value
        .try_into()
        .map_err(|_err| std::io::Error::new(std::io::ErrorKind::InvalidData, "Arithmetic overflow"))
}
#[inline]
fn to_u64(value: usize) -> std::io::Result<u64> {
    value
        .try_into()
        .map_err(|_err| std::io::Error::new(std::io::ErrorKind::InvalidData, "Arithmetic overflow"))
}

struct DropGuard<'a>(&'a mut Vec<u8>, usize);
impl Drop for DropGuard<'_> {
    fn drop(&mut self) {
        self.0.truncate(self.1);
    }
}

fn union(range: &mut Range<usize>, rhs: Range<usize>) {
    if range.start == range.end {
        *range = rhs;
    } else {
        *range = range.start.min(rhs.start)..range.end.max(rhs.end);
    }
}

impl MovingBuffer {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            dirty: 0..0,
            offset: 0,
        }
    }

    #[inline]
    fn flush(
        &mut self,
        flusher: &mut impl FnMut(u64, &[u8]) -> Result<(), std::io::Error>,
    ) -> Result<(), std::io::Error> {
        if !self.dirty.is_empty() {
            flusher(
                self.offset + self.dirty.start as u64,
                &self.data[self.dirty.clone()],
            )?;
        }
        self.dirty = 0..0;
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
        write_back: &mut impl FnMut(u64, &[u8]) -> Result<(), std::io::Error>,
    ) -> Result<(), std::io::Error> {
        // The following cannot overflow, because of how Vec works.
        let free_capacity = self.data.capacity() - self.data.len();
        if position == self.end()? && free_capacity >= data.len() {
            union(
                &mut self.dirty,
                self.data.len()..self.data.len() + data.len(),
            );
            self.data.extend(data);
            Ok(())
        } else if position >= self.offset && checked_add(position, data.len())? <= self.end()? {
            let relative_offset = to_usize(checked_sub_u64(position, self.offset)?)?;
            let end = checked_add_usize(relative_offset, data.len())?;
            self.data[relative_offset..end].copy_from_slice(data);
            union(&mut self.dirty, relative_offset..end);
            Ok(())
        } else {
            self.flush(write_back)?;
            self.data.clear();
            if data.len() < self.data.capacity() {
                self.offset = position;
                union(&mut self.dirty, 0..data.len());
                self.data.extend(data);
                Ok(())
            } else {
                write_back(position, data)?;
                Ok(())
            }
        }
    }
    #[inline(always)]
    fn write_zeroes_at(
        &mut self,
        position: u64,
        zeroes: usize,
        write_back: &mut impl FnMut(u64, &[u8]) -> Result<(), std::io::Error>,
    ) -> Result<(), std::io::Error> {
        // The following cannot overflow, because of how Vec works.
        let free_capacity = self.data.capacity() - self.data.len();
        if position == self.end()? && free_capacity >= zeroes {
            union(&mut self.dirty, self.data.len()..self.data.len() + zeroes);
            let oldlen = self.data.len();
            debug_assert!(oldlen + zeroes <= self.data.capacity());
            self.data.resize(oldlen + zeroes, 0);
            Ok(())
        } else if position >= self.offset && checked_add(position, zeroes)? <= self.end()? {
            let relative_offset = to_usize(checked_sub_u64(position, self.offset)?)?;
            let end = checked_add_usize(relative_offset, zeroes)?;
            self.data[relative_offset..end].fill(0);
            union(&mut self.dirty, relative_offset..end);
            Ok(())
        } else {
            self.flush(write_back)?;
            self.data.clear();
            if zeroes < self.data.capacity() {
                self.offset = position;
                union(&mut self.dirty, 0..zeroes);
                let oldlen = self.data.len();
                debug_assert!(oldlen + zeroes <= self.data.capacity());
                self.data.resize(oldlen + zeroes, 0);
                Ok(())
            } else {
                let zerobuf = [0u8; 1024];
                let mut to_write = zeroes;
                let mut curpos = position;
                while to_write > 0 {
                    let write_now = to_write.min(zerobuf.len());
                    write_back(curpos, &zerobuf[0..write_now])?;
                    decrement_remaining(&mut to_write, write_now);
                    curpos += write_now as u64;
                }
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

            let dropguard = DropGuard(&mut tself.data, 0);
            let got = read_at(position, dropguard.0)?;
            dropguard.0.truncate(got);
            forget(dropguard);
            let curgot = got.min(buf.len());
            buf[..curgot].copy_from_slice(&tself.data[0..curgot]);
            Ok(curgot)
        }

        let read_range = position..checked_add(position, buf.len())?;
        let buffered_range = self.offset..self.end()?;

        //let buflen = buf.len();

        if read_range.end <= buffered_range.start {
            return inner_read_at(read_range.start, buf, self, read_at, write_at);
        }
        if read_range.start > buffered_range.end {
            return inner_read_at(read_range.start, buf, self, read_at, write_at);
        }

        if read_range.start >= buffered_range.start
            && read_range.start <= buffered_range.end
            && read_range.end > buffered_range.end
        {
            let complementary_bytes = to_usize(read_range.end - buffered_range.end)?;
            if self.data.len() + complementary_bytes > self.data.capacity() {
                return inner_read_at(read_range.start, buf, self, read_at, write_at);
            } else {
                let start = self.data.len();
                let dropguard = DropGuard(&mut self.data, start);
                let newsize = start + complementary_bytes;

                dropguard.0.resize(newsize, 0);
                let got = read_at(self.offset + start as u64, &mut dropguard.0[start..])?;
                dropguard.0.truncate(start + got);
                forget(dropguard);

                let buffered_range = self.offset..self.offset + self.data.len() as u64;
                let satisfaction = if read_range.end <= buffered_range.end {
                    read_range.end - read_range.start
                } else {
                    buffered_range.end - read_range.start
                } as usize;
                let read_offset = (read_range.start - buffered_range.start) as usize;
                buf[0..satisfaction]
                    .copy_from_slice(self.data[read_offset..read_offset + satisfaction].as_ref());
                debug_assert!(self.dirty.end <= self.data.len());

                return Ok(satisfaction);
            }
        }

        let mut got = 0;
        if read_range.start < buffered_range.start {
            return inner_read_at(read_range.start, buf, self, read_at, write_at);
        }

        if let Some(overlap) = overlap(read_range.clone(), buffered_range.clone()) {
            let overlapping_src_slice = &self.data
                [to_usize(overlap.start - self.offset)?..to_usize(overlap.end - self.offset)?];
            buf[to_usize(overlap.start - position)?..to_usize(overlap.end - position)?]
                .copy_from_slice(overlapping_src_slice);
            got = checked_add_usize(got, overlapping_src_slice.len())?;
        }

        debug_assert!(read_range.end <= buffered_range.end);
        /*if read_range.end > buffered_range.end {
            let got2 = inner_read_at(
                buffered_range.end,
                &mut buf[buflen - to_usize(read_range.end - buffered_range.end)?..],
                self,
                read_at,
                write_at,
            )?;
            got = checked_add_usize(got, got2)?;
        }*/
        Ok(got)
    }
}

/// Buffering reader/writer.
///
/// Note that T must implement both [`std::io::Seek`] and [`std::io::Write`]. The reason
/// for this is that this is needed so that Drop can flush any unwritten data.
///
/// See crate documentation for more details!
#[derive(Debug)]
pub struct BufStream<T>
where
    T: Seek + Write,
{
    buffer: MovingBuffer,
    position: u64,
    inner_position: u64,
    inner: T,

    /// Incremented on each write call to the backing implementation
    #[cfg(feature = "instrument")]
    pub count_write_calls: usize,
    /// Incremented on each read call to the backing implementation
    #[cfg(feature = "instrument")]
    pub count_read_calls: usize,
    /// Incremented on each seek call to the backing implementation
    #[cfg(feature = "instrument")]
    pub count_seek_calls: usize,
    /// Incremented on each flush call to the backing implementation
    #[cfg(feature = "instrument")]
    pub count_flush_calls: usize,
}

impl<T: Seek + Write> BufStream<T> {
    #[cfg(test)]
    pub(crate) fn clone(&self) -> Self
    where
        T: Clone,
    {
        let mut data = Vec::with_capacity(self.buffer.data.capacity());
        data.extend_from_slice(&self.buffer.data);
        let buffer = MovingBuffer {
            offset: self.buffer.offset,
            dirty: self.buffer.dirty.clone(),
            data,
        };
        BufStream {
            buffer,
            position: self.position,
            inner: self.inner.clone(),
            inner_position: self.inner_position,
            #[cfg(feature = "instrument")]
            count_write_calls: self.count_write_calls,
            #[cfg(feature = "instrument")]
            count_read_calls: self.count_read_calls,
            #[cfg(feature = "instrument")]
            count_seek_calls: self.count_seek_calls,
            #[cfg(feature = "instrument")]
            count_flush_calls: self.count_flush_calls,
        }
    }
}

#[inline]
fn check_stream_position(inner_position: u64, expected_position: u64) -> bool {
    if inner_position != u64::MAX && inner_position == expected_position {
        return true;
    }
    false
}

impl<T: Read + Write + Seek + std::fmt::Debug> BufRead for BufStream<T> {
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
        debug_assert!(!self.buffer.data.is_empty());
        let dropguard = DropGuard(&mut self.buffer.data, 0);

        if !check_stream_position(self.inner_position, self.position) {
            self.inner_position = u64::MAX;
            #[cfg(feature = "instrument")]
            {
                self.count_seek_calls += 1;
            }
            self.inner.seek(SeekFrom::Start(self.position))?;
        }
        self.inner_position = u64::MAX;

        #[cfg(feature = "instrument")]
        {
            self.count_read_calls += 1;
        }
        let got = self.inner.read(dropguard.0)?;
        dropguard.0.truncate(got);
        self.inner_position = checked_add(self.position, got)?;
        forget(dropguard);
        Ok(&self.buffer.data)
    }

    fn consume(&mut self, amt: usize) {
        self.position =
            checked_add(self.position, amt).expect("u64::MAX offset cannot be exceeded");
    }
}

impl<T: Write + Seek> BufStream<T> {
    /// This method can be used to establish a window to update in a file.
    ///
    /// Semantically, it writes all-zeroes to the provided range.
    ///
    /// Additionally, it flushes any existing buffer, and establishes a new, non flushed, all-zero,
    /// in-memory buffer covering the provided range.
    ///
    /// NOTE! If the provided range is larger than the buffer-capacity, this method does does a
    /// flush, and an immediate seek + write of the given number of zeroes, without initializing
    /// a new buffer.
    pub fn write_zeroes(&mut self, len: usize) -> std::io::Result<()> {
        let free_capacity = self.buffer.data.capacity() - self.buffer.data.len();
        if self.position == self.buffer.offset + self.buffer.data.len() as u64
            && free_capacity >= len
            && self.buffer.dirty.end == self.buffer.data.len()
        {
            let oldlen = self.buffer.data.len();
            debug_assert!(oldlen + len <= self.buffer.data.capacity());
            self.buffer.data.resize(oldlen + len, 0);
            self.buffer.dirty.end += len;
            self.position = checked_add(self.position, len)?;
            return Ok(());
        }
        self.write_zeroes_cold(len)?;
        Ok(())
    }
    fn write_zeroes_cold(&mut self, len: usize) -> std::io::Result<()> {
        self.buffer
            .write_zeroes_at(self.position, len, &mut |pos, data| {
                if !check_stream_position(self.inner_position, pos) {
                    self.inner_position = u64::MAX;
                    #[cfg(feature = "instrument")]
                    {
                        self.count_seek_calls += 1;
                    }
                    let t = self.inner.seek(SeekFrom::Start(pos));
                    t?;
                }
                self.inner_position = u64::MAX;
                #[cfg(feature = "instrument")]
                {
                    self.count_write_calls += 1;
                }
                let t = self.inner.write_all(data);
                t?;
                self.inner_position = checked_add(pos, data.len())?;
                Ok(())
            })?;

        self.position = self.position.checked_add(to_u64(len)?).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Arithmetic overflow")
        })?;

        Ok(())
    }

    #[cold]
    #[inline(never)]
    fn write_cold(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buffer.write_at(self.position, buf, &mut |pos, data| {
            if !check_stream_position(self.inner_position, pos) {
                self.inner_position = u64::MAX;
                #[cfg(feature = "instrument")]
                {
                    self.count_seek_calls += 1;
                }
                let t = self.inner.seek(SeekFrom::Start(pos));
                t?;
            }
            self.inner_position = u64::MAX;
            #[cfg(feature = "instrument")]
            {
                self.count_write_calls += 1;
            }
            let t = self.inner.write_all(data);
            t?;
            self.inner_position = checked_add(pos, data.len())?;
            Ok(())
        })?;

        self.position = self
            .position
            .checked_add(to_u64(buf.len())?)
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Arithmetic overflow")
            })?;

        Ok(buf.len())
    }
}

impl<T: Write + Seek> Write for BufStream<T> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let free_capacity = self.buffer.data.capacity() - self.buffer.data.len();
        if self.position == self.buffer.offset + self.buffer.data.len() as u64
            && free_capacity >= buf.len()
            && self.buffer.dirty.end == self.buffer.data.len()
        {
            debug_assert!(self.buffer.data.len() + buf.len() <= self.buffer.data.capacity());
            self.buffer.data.extend(buf);
            self.buffer.dirty.end += buf.len();
            self.position = checked_add(self.position, buf.len())?;
            return Ok(buf.len());
        }
        self.write_cold(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.flush_write()?;
        #[cfg(feature = "instrument")]
        {
            self.count_flush_calls += 1;
        }
        self.inner.flush()
    }
}

impl BufStream<File> {
    /// Flush the buffer, then call the underlying [`File::sync_all`].
    #[cfg_attr(test, mutants::skip)] // Very hard to test, sync_all is only needed in special cases
    pub fn sync_all(&mut self) -> std::io::Result<()> {
        self.flush()?;
        self.inner.sync_all()
    }
    /// Flush the buffer, then call the underlying [`File::set_len`].
    #[cfg_attr(test, mutants::skip)] // Very hard to test, sync_all is only needed in special cases
    pub fn set_len(&mut self, new_len: u64) -> std::io::Result<()> {
        self.flush()?;
        self.inner.set_len(new_len)
    }
}

impl<T: Write + Seek> Drop for BufStream<T> {
    fn drop(&mut self) {
        _ = self.flush_write();
    }
}

impl<T: Write + Seek> BufStream<T> {
    fn flush_write(&mut self) -> Result<(), std::io::Error> {
        let t = self.buffer.flush(&mut |offset, data| {
            if !check_stream_position(self.inner_position, offset) {
                self.inner_position = u64::MAX;
                #[cfg(feature = "instrument")]
                {
                    self.count_seek_calls += 1;
                }
                self.inner.seek(SeekFrom::Start(offset))?;
            }
            self.inner_position = u64::MAX;

            #[cfg(feature = "instrument")]
            {
                self.count_write_calls += 1;
            }
            self.inner.write_all(data)?;
            self.inner_position = checked_add(offset, data.len())?;
            Ok(())
        });
        t?;
        Ok(())
    }
}

impl<T: Seek + Write> BufStream<T> {
    /// Return a text-representation of the instrumentation metrics
    #[cfg(feature = "instrument")]
    #[cfg_attr(test, mutants::skip)]
    pub fn instrumentation(&self) -> String {
        format!(
            "BufStream(reads={}, writes={}, seeks={}, flushes={}",
            self.count_read_calls,
            self.count_write_calls,
            self.count_seek_calls,
            self.count_flush_calls
        )
    }

    /// This drops any buffered data, without writing it to the backing implementation.
    ///
    /// WARNING! This leads to data-loss. Mostly only usable for troubleshooting and very
    /// special requirements.
    #[cfg_attr(test, mutants::skip)]
    pub fn drop_buffer(&mut self) {
        self.buffer.data.clear();
        self.buffer.dirty = 0..0;
    }

    /// Conditionally flush the buffer.
    ///
    /// A flush is performed if this BufStream has more than 'buffer_size' bytes
    /// buffered for writing.
    ///
    /// The `flush_inner` parameter decides if a flush-call is emitted to the
    /// backing implementation.
    #[cfg_attr(test, mutants::skip)]
    pub fn flush_if(&mut self, buffer_size: usize, flush_inner: bool) -> std::io::Result<()> {
        if self.buffer.data.len() >= buffer_size {
            if flush_inner {
                self.flush()?;
            } else {
                self.flush_write()?;
            }
        }
        Ok(())
    }

    /// Crate a new instance, wrapping `inner`, with the given buffer size.
    ///
    /// Note:
    ///
    /// * To be able to write, `inner` must implement [`Write`] and [`Seek`].
    /// * To be able to read, `inner` must implement [`Read`], [`Write`] and [`Seek`].
    ///   The reason for this is that reading may require invalidating the buffer, which
    ///   may require flushing.
    pub fn with_capacity(mut inner: T, capacity: usize) -> std::io::Result<Self> {
        if to_u64(capacity).is_err() {
            panic!("Capacity cannot be larger than 2^64 -1 (u64::MAX)");
        }
        let pos = inner.stream_position()?;
        Ok(Self {
            buffer: MovingBuffer::with_capacity(capacity),
            position: pos,
            inner_position: pos,
            inner,

            #[cfg(feature = "instrument")]
            count_write_calls: 0,
            #[cfg(feature = "instrument")]
            count_read_calls: 0,
            #[cfg(feature = "instrument")]
            count_seek_calls: 1,
            #[cfg(feature = "instrument")]
            count_flush_calls: 0,
        })
    }

    /// Crate a new instance, wrapping `inner`, with a default buffer size.
    ///
    /// Note:
    ///
    /// * To be able to write, `inner` must implement [`Write`] and [`Seek`].
    /// * To be able to read, `inner` must implement [`Read`], [`Write`] and [`Seek`].
    ///   The reason for this is that reading may require invalidating the buffer, which
    ///   may require flushing.
    pub fn new(inner: T) -> std::io::Result<Self> {
        Self::with_capacity(inner, DEFAULT_BUF_SIZE)
    }
}

impl<T: Write + Seek> Seek for BufStream<T> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match pos {
            SeekFrom::Start(pos) => {
                self.position = pos;
            }
            SeekFrom::End(e) => {
                self.flush_write()?;
                #[cfg(feature = "instrument")]
                {
                    self.count_seek_calls += 1;
                }
                let pos = self.inner.seek(SeekFrom::End(e))?;
                self.inner_position = pos;
                self.position = pos;
            }
            SeekFrom::Current(delta) => {
                self.position = self.position.checked_add_signed(delta).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, "Seek index out of range")
                })?;
            }
        }
        Ok(self.position)
    }
}

#[inline(always)]
#[cfg_attr(test, mutants::skip)]
fn is_zero(value: usize) -> bool {
    value == 0
}

impl<T: Read + Seek + Write> BufStream<T> {
    /// Copy `count` bytes from this object to the `other` object.
    /// This will retry on short reads.
    #[cfg_attr(test, mutants::skip)]
    pub fn copy_to(&mut self, mut count: usize, mut other: impl Write) -> std::io::Result<()> {
        let cap = self.buffer.data.capacity();
        while count > 0 {
            let copynow = count.min(cap);
            self.with_bytes(copynow, |src_bytes| other.write_all(src_bytes))??;
            count -= copynow;
        }

        Ok(())
    }

    /// Call the given closure with 'count' bytes of from the file at the curren position.
    ///
    /// If possible, this request will be satisfied from bytes within the buffer.
    /// If that's not possible, this method will flush the buffer and read from the
    /// underlying storage. Note, this method will then allocate additional storage enough
    /// to hold the given range.
    ///
    /// If the read goes beyond the end of file, the buffer will be shorter than `count`.
    pub fn with_bytes<R>(
        &mut self,
        count: usize,
        mut f: impl FnMut(&[u8]) -> R,
    ) -> std::io::Result<R> {
        let u64count = to_u64(count)?;
        if self.position >= self.buffer.offset
            && self.position + u64count <= self.buffer.offset + to_u64(self.buffer.data.len())?
        {
            //Fits in buffer
            let ret = f(
                &self.buffer.data[to_usize(self.position - self.buffer.offset)?
                    ..to_usize(self.position + u64count - self.buffer.offset)?],
            );
            self.position = checked_add(self.position, count)?;
            Ok(ret)
        } else {
            self.flush_write()?;

            let mut temp = vec![0u8; count];
            if !check_stream_position(self.inner_position, self.position) {
                self.inner_position = u64::MAX;
                #[cfg(feature = "instrument")]
                {
                    self.count_seek_calls += 1;
                }
                self.inner.seek(SeekFrom::Start(self.position))?;
            }
            self.inner_position = u64::MAX;
            let mut curoff = 0;
            loop {
                #[cfg(feature = "instrument")]
                {
                    self.count_read_calls += 1;
                }
                let got = self.inner.read(&mut temp[curoff..])?;
                increment_pos_usize(&mut curoff, got);
                if is_zero(got) {
                    temp.truncate(curoff);
                    break;
                }
                if curoff == count {
                    break;
                }
            }
            self.position = checked_add(self.position, curoff)?;
            self.inner_position = self.position;
            Ok(f(&temp))
        }
    }

    #[cold]
    #[inline(never)]
    fn read_cold(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let inner = RefCell::new(&mut self.inner);
        let inner_position = RefCell::new(&mut self.inner_position);

        #[cfg(feature = "instrument")]
        let seek_calls = RefCell::new(&mut self.count_seek_calls);

        let got = self.buffer.read_at(
            self.position,
            buf,
            &mut |pos, data| {
                let mut inner = inner.borrow_mut();

                let mut inner_position = inner_position.borrow_mut();

                if !check_stream_position(**inner_position, pos) {
                    **inner_position = u64::MAX;
                    #[cfg(feature = "instrument")]
                    {
                        **seek_calls.borrow_mut() += 1;
                    }
                    inner.seek(SeekFrom::Start(pos))?;
                }
                **inner_position = u64::MAX;
                #[cfg(feature = "instrument")]
                {
                    self.count_read_calls += 1;
                }
                let got = inner.read(data)?;
                **inner_position = checked_add(pos, got)?;
                debug_assert!(got <= data.len());
                Ok(got)
            },
            &mut |offset, data| {
                let mut inner_position = inner_position.borrow_mut();
                let mut inner = inner.borrow_mut();
                if !check_stream_position(**inner_position, offset) {
                    **inner_position = u64::MAX;
                    #[cfg(feature = "instrument")]
                    {
                        **seek_calls.borrow_mut() += 1;
                    }
                    inner.seek(SeekFrom::Start(offset))?;
                }
                **inner_position = u64::MAX;

                #[cfg(feature = "instrument")]
                {
                    self.count_write_calls += 1;
                }
                inner.write_all(data)?;

                **inner_position = checked_add(offset, data.len())?;
                Ok(())
            },
        )?;
        debug_assert!(got <= buf.len());
        self.position = checked_add(self.position, got)?;
        Ok(got)
    }
}

#[inline(always)]
/// We need to skip this in mutants testing. Both arms of the if-statement do exactly
/// the same thing on machines where usize and u64 are the same size size.
#[cfg_attr(test, mutants::skip)]
fn increment_pos(position: &mut u64, buflen: usize) -> std::io::Result<()> {
    if std::mem::size_of::<usize>() > std::mem::size_of::<u64>() {
        *position += to_u64(buflen)?;
    } else {
        *position += buflen as u64;
    }
    Ok(())
}
#[inline(always)]
/// We need to skip this in mutants testing, because this is used in for loops
/// to make progress, causing them to never complete otherwise.
#[cfg_attr(test, mutants::skip)]
fn increment_pos_usize(position: &mut usize, buflen: usize) {
    *position += buflen;
}
#[inline(always)]
/// We need to skip this in mutants testing, because not decreasing the 'remaining' value
/// leads to infinite loops.
#[cfg_attr(test, mutants::skip)]
fn decrement_remaining(remaining: &mut usize, buflen: usize) {
    *remaining -= buflen;
}

impl<T: Read + Seek + Write> Read for BufStream<T> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let offset = self.position.wrapping_sub(self.buffer.offset);

        let buflen = buf.len();
        if offset < self.buffer.data.len().saturating_sub(buflen) as u64 {
            let offset = offset as usize;

            buf.copy_from_slice(&self.buffer.data[offset..offset + buflen]);

            increment_pos(&mut self.position, buflen)?;

            return Ok(buflen);
        }

        self.read_cold(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, RngCore};
    use std::io::ErrorKind;
    use std::panic;
    use std::panic::AssertUnwindSafe;

    #[derive(Default, PartialEq, Eq, Debug, Clone)]
    struct FakeStream {
        buf: Vec<u8>,
        position: usize,
        short_read_by: usize,
        panic_after: usize,
        err_after: usize,
        writes_have_occurred: bool,

        seek_calls: usize,
        write_calls: usize,
        read_calls: usize,
        flush_calls: usize,
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
            self.read_calls += 1;
            self.maybe_panic()?;

            let mut to_read = buf.len();

            if to_read > 1 && self.short_read_by > 0 {
                to_read = (to_read.saturating_sub(self.short_read_by)).max(1);
            }
            let end = (self.position + to_read).min(self.buf.len());

            let got = end.saturating_sub(self.position);
            if is_zero(got) {
                return Ok(0);
            }
            buf[0..got].copy_from_slice(&self.buf[self.position..self.position + got]);
            self.position += got;
            Ok(got)
        }
    }
    impl Write for FakeStream {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.write_calls += 1;
            self.writes_have_occurred = true;
            self.maybe_panic()?;
            for b in buf {
                if self.position >= self.buf.len() {
                    self.buf.resize(self.position, 0);
                    self.buf.push(*b);
                } else {
                    self.buf[self.position] = *b;
                }
                self.position += 1;
            }
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.flush_calls += 1;
            self.maybe_panic()?;
            Ok(())
        }
    }
    impl Seek for FakeStream {
        fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
            self.seek_calls += 1;
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

        let mut cut = BufStream::with_capacity(cut_inner, bufsize).unwrap();

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
        fuzz(8, Some(15), Some(10), false);
    }

    #[test]
    fn writes_are_buffered() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, 100).unwrap();
        cut.write(&[1, 2, 3]).unwrap();
        assert!(cut.inner.buf.is_empty());
    }

    #[test]
    fn zero_writes_are_buffered() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, 100).unwrap();
        cut.write_zeroes(3).unwrap();
        assert!(cut.inner.buf.is_empty());
    }

    #[test]
    #[cfg(feature = "instrument")]
    fn seeks_are_counted() {
        let mut cut_inner = FakeStream::default();
        cut_inner.buf = vec![1, 2, 3];
        let mut cut = BufStream::with_capacity(cut_inner, 100).unwrap();

        cut.seek(SeekFrom::Start(0)).unwrap();
        cut.seek(SeekFrom::Current(0)).unwrap();
        cut.seek(SeekFrom::End(1)).unwrap();

        assert_eq!(cut.count_seek_calls, 2, "only the SeekFrom::End should go through tot the underlying object, and then there's the initial seek");
    }

    #[test]
    fn writes_are_buffered_even_with_small_buffer() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, 3).unwrap();
        cut.write(&[1, 2, 3]).unwrap();
        assert!(cut.inner.buf.is_empty());
    }

    #[test]
    fn zero_writes_are_buffered_even_with_small_buffer() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, 3).unwrap();
        cut.write_zeroes(3).unwrap();
        assert!(cut.inner.buf.is_empty());
    }

    #[test]
    fn drop_implies_flush() {
        let mut cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(&mut cut_inner, 100).unwrap();
        cut.write(&[1, 2, 3]).unwrap();
        // Not yet flushed
        assert!(cut.inner.buf.is_empty());
        drop(cut);
        assert_eq!(cut_inner.buf, [1, 2, 3]);
    }

    #[test]
    fn can_seek_far_beyond_end() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, 100).unwrap();
        cut.seek(SeekFrom::Start(200)).unwrap();
        cut.write(&[1, 2, 3]).unwrap();
        cut.flush().unwrap();
        assert_eq!(cut.inner.buf.len(), 203);
    }

    #[test]
    fn can_write_beyond_end() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, 100).unwrap();
        cut.write(&[1, 2, 3]).unwrap();
        cut.flush().unwrap();
        cut.seek(SeekFrom::Start(3)).unwrap();
        cut.write(&[1, 2, 3]).unwrap();
        cut.flush().unwrap();
        assert_eq!(cut.inner.buf.len(), 6);
    }

    #[test]
    fn can_write_zero_beyond_end() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, 100).unwrap();
        cut.write_zeroes(3).unwrap();
        cut.flush().unwrap();
        cut.seek(SeekFrom::Start(3)).unwrap();
        cut.write_zeroes(3).unwrap();
        cut.flush().unwrap();
        assert_eq!(cut.inner.buf.len(), 6);
    }

    #[test]
    fn write_large_number_of_zeroes() {
        let mut cut_inner = FakeStream::default();
        cut_inner.buf.resize(4000, 42);
        let mut cut = BufStream::with_capacity(cut_inner, 100).unwrap();
        cut.seek(SeekFrom::Start(100)).unwrap();
        cut.write_zeroes(4000).unwrap();
        for x in cut.inner.buf.iter().take(100) {
            assert_eq!(*x, 42);
        }
        for x in cut.inner.buf.iter().skip(100) {
            assert_eq!(*x, 0);
        }
        assert_eq!(cut.inner.buf.len(), 4100);
    }

    #[test]
    fn reading_does_not_cause_writes() {
        let mut cut_inner = FakeStream::default();
        cut_inner.write(&[1, 2, 3, 4, 5]).unwrap();
        cut_inner.writes_have_occurred = false;
        let mut cut = BufStream::with_capacity(cut_inner, 100).unwrap();
        cut.seek(SeekFrom::Start(1)).unwrap();
        let mut buf = [0, 0, 0];
        cut.read(&mut buf).unwrap();
        assert_eq!(buf, [2, 3, 4]);
        cut.flush().unwrap();
        assert!(!cut.inner.writes_have_occurred);
    }

    #[test]
    fn can_seek_and_write_just_beyond_end() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, 100).unwrap();
        cut.write(&[1, 2, 3]).unwrap();
        cut.flush().unwrap();
        cut.seek(SeekFrom::Start(4)).unwrap();
        cut.write(&[1, 2, 3]).unwrap();
        cut.flush().unwrap();
        assert_eq!(cut.inner.buf, [1, 2, 3, 0, 1, 2, 3]);
        assert_eq!(cut.inner.buf.len(), 7);
    }
    #[test]
    fn stream_position_is_reloaded() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, 100).unwrap();
        cut.write(&[1]).unwrap();
        cut.inner_position = u64::MAX;
        let pos = check_stream_position(cut.inner_position, cut.position);
        assert!(!pos);
    }
    #[test]
    fn stream_position_logic() {
        assert_eq!(check_stream_position(1, 1), true);
        assert_eq!(check_stream_position(2, 1), false);
        assert_eq!(check_stream_position(1, u64::MAX), false);
        assert_eq!(check_stream_position(u64::MAX, u64::MAX), false);
        assert_eq!(check_stream_position(u64::MAX, u64::MAX - 1), false);
        assert_eq!(check_stream_position(u64::MAX - 1, u64::MAX - 1), true);
    }

    #[test]
    fn writes_are_buffered_after_seek() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, 100).unwrap();
        cut.write(&[1, 2, 3]).unwrap();
        cut.seek(SeekFrom::Start(10)).unwrap();
        cut.write(&[4, 5, 6]).unwrap();
        assert_eq!(cut.inner.buf, [1, 2, 3]); //This should have been flushed

        assert_eq!(cut.buffer.data, [4, 5, 6]); //This should have been flushed
        assert_eq!(cut.buffer.offset, 10); //This should have been flushed
    }

    #[test]
    fn zero_writes_are_buffered_after_seek() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, 100).unwrap();
        cut.write_zeroes(3).unwrap();
        cut.seek(SeekFrom::Start(10)).unwrap();
        cut.write_zeroes(3).unwrap();
        assert_eq!(cut.inner.buf, [0, 0, 0]); //This should have been flushed

        assert_eq!(cut.buffer.data, [0, 0, 0]); //This should have been flushed
        assert_eq!(cut.buffer.offset, 10); //This should have been flushed
    }

    #[test]
    fn big_read_followed_by_small_write_doesnt_write_everything() {
        let mut cut_inner = FakeStream::default();
        cut_inner.buf = vec![42u8; 20];
        let mut cut = BufStream::with_capacity(cut_inner, 20).unwrap();

        let mut buf = [0u8; 20];
        cut.read(&mut buf).unwrap();
        assert_eq!(buf, [42u8; 20]);

        cut.seek(SeekFrom::Start(10)).unwrap();
        cut.write(&[43]).unwrap();
        cut.seek(SeekFrom::Start(12)).unwrap();
        cut.write(&[43]).unwrap();

        cut.inner.buf = vec![1u8; 20];
        cut.flush().unwrap();

        assert_eq!(
            cut.inner.buf,
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 43, 42, 43, 1, 1, 1, 1, 1, 1, 1,]
        );
    }

    #[test]
    fn big_read_followed_by_small_zero_write_doesnt_write_everything() {
        let mut cut_inner = FakeStream::default();
        cut_inner.buf = vec![42u8; 20];
        let mut cut = BufStream::with_capacity(cut_inner, 20).unwrap();

        let mut buf = [0u8; 20];
        cut.read(&mut buf).unwrap();
        assert_eq!(buf, [42u8; 20]);

        cut.seek(SeekFrom::Start(10)).unwrap();
        cut.write_zeroes(1).unwrap();
        cut.seek(SeekFrom::Start(12)).unwrap();
        cut.write_zeroes(1).unwrap();

        cut.inner.buf = vec![1u8; 20];
        cut.flush().unwrap();

        assert_eq!(
            cut.inner.buf,
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 42, 0, 1, 1, 1, 1, 1, 1, 1,]
        );
    }

    #[test]
    fn reads_are_buffered() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, 100).unwrap();
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
        let mut cut = BufStream::with_capacity(cut_inner, 3).unwrap();
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
        let mut cut = BufStream::with_capacity(cut_inner, buffer_size).unwrap();
        use rand::SeedableRng;
        debug_println!(
            "\n\n==== Seed: {}, buffer: {}, write size: {} bufread: {:?} ====",
            seed,
            buffer_size,
            write_sizes,
            bufread
        );
        for _ in 0..7 {
            let mut panic_or_err = false;
            if small_rng.gen_bool(0.05) {
                good.panic_after = 1;
                cut.inner.panic_after = small_rng.gen_range(1..4);
                panic_or_err = true;
            } else if small_rng.gen_bool(0.05) {
                good.err_after = 1;
                cut.inner.err_after = small_rng.gen_range(1..4);
                panic_or_err = true;
            }
            match small_rng.gen_range(0..3) {
                0 if good.buf.len() > 0 => {
                    let seek_to = small_rng.gen_range(0..good.buf.len());

                    debug_println!("==SEEK to {} [{:?}]", seek_to, panic_or_err);
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
                    let mut goodbuf = vec![0u8; read_bytes];
                    good.short_read_by = short_read;
                    let good_got = catch(&mut || good.read(&mut goodbuf));

                    let mut cutbuf = vec![0u8; read_bytes];
                    cut.inner.short_read_by = short_read;
                    debug_println!("==READ {} [{:?}] {:?}", read_bytes, panic_or_err, bufread);

                    let cut_got;
                    if bufread {
                        if small_rng.gen_bool(0.5) {
                            cut_got = catch(&mut || {
                                let len = cut.with_bytes(read_bytes, |idata| {
                                    cutbuf[0..idata.len()].copy_from_slice(idata);
                                    idata.len()
                                })?;
                                Ok(len)
                            });
                        } else {
                            cut_got = catch(&mut || {
                                let cutbuflen = cutbuf.len();
                                let fillbuf = cut.fill_buf()?;
                                let data = &fillbuf[0..cutbuflen.min(fillbuf.len())];
                                cutbuf[0..data.len()].copy_from_slice(data);
                                let len = data.len();
                                cut.consume(len);
                                Ok(len)
                            });
                        }
                    } else {
                        cut_got = catch(&mut || cut.read(&mut cutbuf));
                    }

                    debug_println!(
                        "did READ {:?}/{} -> {:?} (short-read: {}) [{:?}]",
                        cut_got,
                        read_bytes,
                        cutbuf,
                        short_read,
                        panic_or_err
                    );
                    match (good_got, cut_got) {
                        (Ok(good_got), Ok(cut_got)) => {
                            if good_got > 0 {
                                assert!(cut_got > 0);
                            }
                            if short_read == 0
                                && good.position + read_bytes <= good.buf.len()
                                && !bufread
                            {
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
                    let zero_instead = small_rng.gen_bool(0.25);
                    let mut buf = vec![0u8; write_bytes];
                    if !zero_instead {
                        small_rng.fill_bytes(&mut buf);
                    }
                    debug_println!(
                        "==WRITE {} {:?} [{:?}] zero: {:?}",
                        buf.len(),
                        buf,
                        panic_or_err,
                        zero_instead
                    );
                    let good_got = catch(&mut || good.write(&buf));
                    let cut_got = catch(&mut || {
                        if zero_instead {
                            cut.write_zeroes(buf.len())?;
                            Ok(buf.len())
                        } else {
                            cut.write(&buf)
                        }
                    });
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
            assert_eq!(cut.buffer.data.capacity(), buffer_size);

            #[cfg(feature = "instrument")]
            {
                assert_eq!(cut.inner.write_calls, cut.count_write_calls, "write count");
                assert_eq!(cut.inner.read_calls, cut.count_read_calls, "read count");
                assert_eq!(cut.inner.seek_calls, cut.count_seek_calls, "seek count");
                assert_eq!(cut.inner.flush_calls, cut.count_flush_calls, "flush count");
            }
            let mut cut_cloned = cut.clone();
            cut_cloned.flush().unwrap();
            assert_eq!(&good.buf, &cut_cloned.inner.buf);
            assert_eq!(&good.position, &(cut_cloned.position as usize));

            assert_eq!(cut_cloned.buffer.data.capacity(), buffer_size);
        }
    }

    /// A gigantic stream. All values read as position%256, and if written, must
    /// be written to the same value.
    #[derive(Default)]
    struct SuperLargeStream {
        position: u128,
    }
    impl Write for SuperLargeStream {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            for b in buf {
                assert_eq!(*b, self.position as u8);
                self.position += 1;
            }
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl Read for SuperLargeStream {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            for b in buf.iter_mut() {
                *b = self.position as u8;
            }
            Ok(buf.len())
        }
    }
    impl Seek for SuperLargeStream {
        fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
            match pos {
                SeekFrom::Start(p) => {
                    self.position = p as u128;
                }
                SeekFrom::End(_e) => {
                    panic!("SeekFrom::End not supported");
                }
                SeekFrom::Current(d) => {
                    let new_position =
                        self.position.checked_add_signed(d.into()).ok_or_else(|| {
                            std::io::Error::new(std::io::ErrorKind::InvalidInput, "overflow")
                        })?;
                    if new_position > u64::MAX as u128 {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "overflow",
                        ));
                    }
                    self.position = new_position;
                }
            }
            Ok(self
                .position
                .try_into()
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "overflow"))?)
        }
    }

    #[test]
    fn test_extreme_size_handling() {
        let large_backing = SuperLargeStream::default();
        let mut large = BufStream::new(large_backing).unwrap();

        large.seek(SeekFrom::Start(u64::MAX)).unwrap(); // Should succeed

        large.write(&[u64::MAX as u8]).unwrap_err();
        large.seek(SeekFrom::Current(10)).unwrap_err();

        let mut buf = [0u8; 1];
        large.read(&mut buf).unwrap_err();

        large.seek(SeekFrom::Start(256 + 2)).unwrap();
        let mut buf = [0u8; 1];
        large.read(&mut buf).unwrap();
        assert_eq!(buf[0], 2);
    }

    #[test]
    fn test_dirty_buffer_beyond_u64_max() {
        let large_backing = SuperLargeStream::default();
        let mut large = BufStream::new(large_backing).unwrap();

        large.seek(SeekFrom::Start(u64::MAX)).unwrap(); // Should succeed
        large.read(&mut [0u8; 1024]).unwrap_err();
        large.write(&mut [255]).unwrap_err();
        large.flush().unwrap();
    }

    #[test]
    #[cfg(feature = "instrument")]
    fn instrument_counts_flushes() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::new(cut_inner).unwrap();
        for i in 1..6 {
            cut.flush().unwrap();
            assert_eq!(cut.count_flush_calls, i);
        }
    }

    #[test]
    #[cfg(feature = "instrument")]
    fn test_buffer_seek_only_when_needed() {
        let cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, 1000).unwrap();

        for _ in 0..3000 {
            let pos = cut.stream_position().unwrap();
            cut.write_zeroes(32).unwrap();

            cut.write(&[42u8; 200]).unwrap();
            let end = cut.stream_position().unwrap();
            cut.seek(SeekFrom::Start(pos)).unwrap();
            cut.write(&[43; 32]).unwrap();

            cut.seek(SeekFrom::Start(end)).unwrap();
            cut.flush_if(700, true).unwrap();
        }

        assert_eq!(cut.count_seek_calls, 1);
    }

    #[test]
    fn repeated_short_reads_are_handled() {
        let mut cut_inner = FakeStream::default();
        cut_inner.buf.resize(100, 42);
        cut_inner.short_read_by = 1000;
        let mut cut = BufStream::with_capacity(cut_inner, 1000).unwrap();
        let mut buf = [0u8; 100];
        cut.read_exact(&mut buf).unwrap();
        #[cfg(feature = "instrument")]
        {
            assert_eq!(cut.count_read_calls, 100);
        }
    }

    #[test]
    fn repeated_short_reads_are_handled_with_bytes() {
        let mut cut_inner = FakeStream::default();
        cut_inner.buf.resize(100, 42);
        cut_inner.short_read_by = 1000;
        let mut cut = BufStream::with_capacity(cut_inner, 1000).unwrap();
        cut.with_bytes(100, |_| {}).unwrap();
        #[cfg(feature = "instrument")]
        {
            assert_eq!(cut.count_read_calls, 100);
        }
    }

    #[test]
    fn repeated_short_reads_are_handled_with_bytes_big() {
        let mut cut_inner = FakeStream::default();
        cut_inner.buf.resize(200, 42);
        cut_inner.short_read_by = 1000;
        let mut cut = BufStream::with_capacity(cut_inner, 1).unwrap();
        cut.with_bytes(100, |_| {}).unwrap();
        #[cfg(feature = "instrument")]
        {
            assert_eq!(cut.count_read_calls, 100);
        }
    }
}
