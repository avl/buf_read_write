extern crate core;

use std::cell::RefCell;
use std::io::{BufWriter, ErrorKind, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::ptr::read;




#[derive(Clone, Debug)]
struct MovingBuffer {
    offset: usize,
    data: Vec<u8>,
}

#[cfg(debug_assertions)]
macro_rules! debug_println {
    ($f:expr, $($a:expr),+) => {{
        println!($f, $($a),+ );
    }};
}

#[cfg(not(debug_assertions))]
macro_rules! debug_println {
    ($f:expr, $($a:expr),+) => {{
    }};
}


fn overlap(range1: Range<usize>, range2: Range<usize>) -> Option<Range<usize>> {
    if range1.end <= range2.start {
        return None;
    }
    if range2.end <= range1.start {
        return None;
    }
    Some(range1.start.max(range2.start)..range1.end.min(range2.end))
}

impl MovingBuffer {

    fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            offset: 0,
        }
    }

    fn flush(&mut self, flusher: &mut impl FnMut(usize, &[u8]) -> Result<(), std::io::Error>)  -> Result<(), std::io::Error> {
        if !self.data.is_empty() {
            flusher(self.offset, &self.data)?;
        }
        self.data.clear();
        Ok(())
    }
    fn end(&self) -> usize {
        self.offset + self.data.len()
    }
    fn write_at(&mut self, position: usize, data: &[u8], write_at: &mut impl FnMut(usize,  &[u8]) -> Result<(), std::io::Error>) -> Result<(), std::io::Error> {
        let free_capacity = self.data.capacity() - self.data.len();

        if position == self.end() && free_capacity >= data.len() {
            self.data.extend(data);
            Ok(())
        } else if position >= self.offset && position + data.len() <= self.end() {
            let relative_offset = position - self.offset;
            self.data[relative_offset..relative_offset+data.len()].copy_from_slice(data);

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

    fn read_at<
        R: FnMut(usize,  &mut [u8]) -> std::io::Result<usize>,
        W: FnMut(usize,  &[u8]) -> std::io::Result<()>,
    >(&mut self, position: usize, buf: &mut [u8],
                                                                      read_at: &mut R,
                                                                      write_at: &mut W,
    ) -> std::io::Result<usize> {

        if buf.len() > self.data.capacity() {
            self.flush(write_at)?;
            return read_at(position, buf);
        }

        fn inner_read_at<
            F: FnMut(usize,  &mut [u8]) -> std::io::Result<usize>,
            W: FnMut(usize,  &[u8]) -> std::io::Result<()>,
        >(position: usize, buf: &mut [u8], tself: &mut MovingBuffer, read_at: &mut F, write_at: &mut W) -> std::io::Result<usize> {

            if buf.len() == 0 {
                return Ok(0);
            }
            tself.flush(write_at)?;
            let cap = tself.data.capacity();
            tself.data.resize(cap, 0);
            tself.offset = position;
            let got = read_at(position, &mut tself.data)?;
            tself.data.truncate(got);
            let curgot = got.min(buf.len());
            buf[..curgot].copy_from_slice(&tself.data[0..curgot]);
            Ok(got)
        }

        let read_range = position .. position+buf.len();
        let buffered_range = self.offset..self.end();

        let buflen = buf.len();

        if read_range.end <= buffered_range.start {
            return inner_read_at(read_range.start, buf, self, read_at, write_at);
        }
        if read_range.start >= buffered_range.end {
            return inner_read_at(read_range.start, buf, self, read_at, write_at);
        }

        let mut got =0;
        if read_range.start < buffered_range.start {
            if read_range.start + self.data.capacity() < buffered_range.start {
                // Buffer size is too small to reach to the already existing stuff, and
                // we don't split buffers.
                unreachable!();
            } else {
                let len = (buffered_range.start - read_range.start).min(buflen);
                got = read_at(read_range.start, &mut buf[0..len])?;
                if got < len {
                    return Ok(got);
                }
            }
        }

        if let Some(overlap) = overlap(read_range.clone(), buffered_range.clone()) {
            let overlapping_src_slice = &self.data[(overlap.start-self.offset)..(overlap.end-self.offset)];
            buf[overlap.start-position..overlap.end-position].copy_from_slice(overlapping_src_slice);
            got += overlapping_src_slice.len();
        }

        if read_range.end > buffered_range.end {
            let got2 = inner_read_at(buffered_range.end, &mut buf[buflen - (read_range.end - buffered_range.end)..], self, read_at, write_at)?;
            got += got2;
        }
        Ok(got)
    }
}


#[derive(Debug)]
struct BufStream<T> {
    buffer: MovingBuffer,
    position: usize,
    inner: T,
    poisoned: bool,
}

impl<T> BufStream<T> {
    pub(crate) fn clone(&self) -> Self where T: Clone{
        BufStream {
            buffer: self.buffer.clone(),
            position: self.position,
            inner: self.inner.clone(),
            poisoned: self.poisoned,
        }
    }
}

const DEFAULT_BUF_SIZE: usize = 8 * 1024;


impl<T:Write+ Seek> Write for BufStream<T> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.poisoned {
            panic!("Attempt to use poisoned Bufstream (i.e, one that was held while a write panicked)");
        }
        self.buffer.write_at(self.position, buf, &mut |pos,data|{
            if self.inner.stream_position()? != pos as u64 {
                let t = self.inner.seek(SeekFrom::Start(pos as u64));
                t?;
            }
            self.poisoned = true;
            let t = self.inner.write_all(data);
            self.poisoned = false;
            t?;
            Ok(())
        })?;

        self.position += buf.len();

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.flush_write()?;
        self.inner.flush()
    }
}

impl<T:Write+Seek> BufStream<T> {
    pub fn flush_write(&mut self) -> Result<(), std::io::Error> {
        self.poisoned = true;
        let t = self.buffer.flush(&mut |offset, data|{
            if offset != self.inner.stream_position()? as usize {
                self.inner.seek(SeekFrom::Start(offset as u64))?;
            }

            self.inner.write_all(data)?;
            Ok(())
        });
        self.poisoned = false;
        t?;
        Ok(())
    }
    pub fn with_capacity(inner: T, capacity: usize) -> Self {
        Self {
            buffer: MovingBuffer::with_capacity(capacity),
            position: 0,
            inner,
            poisoned: false,
        }
    }
    pub fn new(inner: T) -> Self {
        Self::with_capacity(inner, 8 * 1024)
    }
}

impl<T:Seek> Seek for BufStream<T> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match pos {
            SeekFrom::Start(pos) => {
                self.position = pos as usize;
            }
            SeekFrom::End(e) => {
                self.inner.seek(SeekFrom::End(e))?;
                self.position = self.inner.stream_position()? as usize;
            }
            SeekFrom::Current(delta) => {
                self.inner.seek(SeekFrom::Start((self.position as u64).checked_add_signed(delta)
                    .ok_or::<std::io::Error>(std::io::Error::new(ErrorKind::Other, "overflow"))
                    ?))?;
                self.position = self.inner.stream_position()? as usize;
            }
        }
        Ok(self.position as u64)
    }
}
impl<T:Read+Seek+Write> Read for BufStream<T> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut inner = RefCell::new(&mut self.inner);
        let got = self.buffer.read_at(self.position, buf, &mut |pos,data| {
            let mut inner = inner.borrow_mut();
            if inner.stream_position()? != pos as u64 {
                inner.seek(SeekFrom::Start(pos as u64))?;
            }
            let got = inner.read(data)?;
            Ok(got)
        },
        &mut |offset,data|{
            let mut inner = inner.borrow_mut();
            if offset != inner.stream_position()? as usize {
                inner.seek(SeekFrom::Start(offset as u64))?;
            }

            inner.write_all(data)?;
            Ok(())
        }

        )?;
        self.position += got;
        Ok(got)
    }
}


#[cfg(test)]
mod tests {
    use rand::{Rng, RngCore};
    use super::*;

    #[derive(Default, PartialEq, Eq, Debug, Clone)]
    struct FakeStream {
        buf: Vec<u8>,
        position: usize,
        short_read_by: usize,
    }
    impl Read for FakeStream {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let mut to_read = buf.len();

            if to_read > 1 && self.short_read_by > 0 {
                to_read = (to_read-self.short_read_by).max(1);
            }
            let end = (self.position+to_read).min(self.buf.len());

            let got = end.saturating_sub(self.position);
            if got == 0 {
                return Ok(0)
            }
            buf[0..got].copy_from_slice(&self.buf[self.position..self.position+got]);
            self.position += got;
            Ok(got)
        }
    }
    impl Write for FakeStream {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
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
            Ok(())
        }
    }
    impl Seek for FakeStream {
        fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
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


    fn run_exhaustive_conf(
        bufsize: usize,
        ops: &[(usize,usize)],
        mut databyte: u8,
    ) {
        let mut good = FakeStream::default();
        let mut cut_inner = FakeStream::default();

        let mut cut = BufStream::with_capacity(cut_inner, bufsize);

        for (op,param) in ops.iter().copied() {
            match op {
                0 if good.buf.len() > 0 => {
                    let seek_to = param;
                    debug_println!("==SEEK to {}", seek_to);
                    good.seek(SeekFrom::Start(seek_to as u64)).unwrap();
                    cut.seek(SeekFrom::Start(seek_to as u64)).unwrap();
                }
                1  => {
                    let read_bytes = param/2;
                    let short_read = param%2;
                    debug_println!("==READ {}",read_bytes);

                    let mut goodbuf = vec![0u8; read_bytes];
                    good.short_read_by = short_read;
                    let good_position = good.position;
                    let goodgot = good.read(&mut goodbuf).unwrap();

                    assert_eq!(goodgot +short_read, goodbuf.len().min(good.buf.len()-good_position));

                    let mut cutbuf = vec![0u8; read_bytes];
                    cut.inner.short_read_by = short_read;
                    let cutgot = cut.read(&mut cutbuf).unwrap();
                    assert_eq!(goodbuf, cutbuf);
                    debug_println!("did READ {} -> {:?}", read_bytes, cutbuf);
                    if cutgot != goodgot {
                        good.position = cut.position;
                    }
                }
                0|1|2 => {
                    let write_bytes = param;
                    let mut buf = vec![0u8; write_bytes];
                    for i in 0..write_bytes {
                        buf[i] = databyte;
                        databyte = databyte.wrapping_add(17);
                    }
                    debug_println!("==WRITE {} {:?}", buf.len(), buf);
                    let goodgot = good.write(&buf).unwrap();
                    let cutgot = cut.write(&buf).unwrap();
                }
                _ => unreachable!()
            }
        }
        let mut cut_cloned = cut.clone();
        cut_cloned.flush().unwrap();
        assert_eq!(&good.buf, &cut_cloned.inner.buf);
        assert_eq!(&good.position, &cut_cloned.position);

    }

    #[test]
    fn exhaustive() {
        let mut databyte = 0;
        for bufsize in [1,3,7] {
            for first_op in 0..3 {
                let first_op_param_options = if first_op != 0 {6} else {1};
                for first_op_param in 0..first_op_param_options {
                    for second_op in 0..3 {
                        let second_op_param_options = if second_op != 0 {6} else {1};
                        for third_op in 0..3 {
                            let third_op_param_options = if third_op != 0 { 6 } else { 1 };
                            for fourth_op in 0..3 {
                                let fourth_op_param_options = if fourth_op != 0 { 6 } else { 1 };

                                println!("\n\n========Iteration {} {} {} {} {} {} {} {} {} {}===========",
                                         bufsize,
                                         first_op, first_op_param_options,
                                         second_op, second_op_param_options,
                                         third_op, third_op_param_options,
                                         fourth_op, fourth_op_param_options,
                                         databyte
                                );
                                run_exhaustive_conf(
                                    bufsize,
                                    &[
                                    (first_op, first_op_param_options),
                                    (second_op, second_op_param_options),
                                    (third_op, third_op_param_options),
                                    (fourth_op, fourth_op_param_options)
                                ],
                                    databyte
                                );
                                databyte = databyte.wrapping_add(1);
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

        let mut items = case.split(" ").map(|x|x.parse::<usize>().unwrap());
        let mut n = move||items.next().unwrap();

        run_exhaustive_conf(
            n(),
            &[
                (n(),  n()),
                (n(),  n()),
                (n(),  n()),
                (n(),  n()),
            ],
            n() as u8
        );
    }

    #[test]
    fn fuzz_many() {
        for i in 0..10000000 {
            fuzz(i, Some(3), Some(1));
            fuzz(i, Some(1), Some(3));
            fuzz(i, Some(10), Some(15));
            fuzz(i, Some(15), Some(10));
            fuzz(i, None, None);
        }
    }

    #[test]
    fn regression() {

        fuzz(0, Some(15), Some(10));
    }

    fn fuzz(seed: u64, buffer_size: Option<usize>, write_sizes: Option<usize>) {
        let mut small_rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let buffer_size = buffer_size.unwrap_or(small_rng.gen_range(1..10));
        let write_sizes = write_sizes.unwrap_or(small_rng.gen_range(1..10));
        let mut good = FakeStream::default();
        let mut cut_inner = FakeStream::default();
        let mut cut = BufStream::with_capacity(cut_inner, buffer_size);
        use rand::SeedableRng;
        debug_println!("\n\n==== Seed: {}, buffer: {}, write size: {} ====", seed, buffer_size, write_sizes);
        for _ in 0..7 {
            match small_rng.gen_range(0..3) {
                0 if good.buf.len() > 0=> {
                    let seek_to = small_rng.gen_range(0..good.buf.len());
                    debug_println!("==SEEK to {}", seek_to);
                    good.seek(SeekFrom::Start(seek_to as u64)).unwrap();
                    cut.seek(SeekFrom::Start(seek_to as u64)).unwrap();
                }
                1  => {
                    let read_bytes = small_rng.gen_range(0..write_sizes);
                    let short_read = small_rng.gen_bool(0.3) as usize;
                    debug_println!("==READ {}",read_bytes);
                    let mut goodbuf = vec![0u8; read_bytes];
                    good.short_read_by = short_read;
                    let good_got = good.read(&mut goodbuf).unwrap();

                    println!("Good read got {:?}", goodbuf);

                    let mut cutbuf = vec![0u8; read_bytes];
                    cut.inner.short_read_by = short_read;
                    let cut_got = cut.read(&mut cutbuf).unwrap();
                    debug_println!("did READ {}/{} -> {:?} (short-read: {})", cut_got, read_bytes, cutbuf, short_read);
                    if good_got > 0 {
                        assert!(cut_got > 0);
                    }
                    let mingot = cut_got.min(good_got);
                    if cut_got != good_got {
                        good.position = cut.position;
                    }
                    assert_eq!(goodbuf[0..mingot], cutbuf[0..mingot]);
                }
                0|1|2 => {
                    let write_bytes = small_rng.gen_range(0..write_sizes);
                    let mut buf = vec![0u8; write_bytes];
                    small_rng.fill_bytes(&mut buf);
                    debug_println!("==WRITE {} {:?}", buf.len(), buf);
                    good.write(&buf).unwrap();
                    cut.write(&buf).unwrap();
                }
                _ => unreachable!()
            }
            debug_println!("Good state: {:?}", good);
            debug_println!("Cut state: {:?}", cut);
            let mut cut_cloned = cut.clone();
            cut_cloned.flush().unwrap();
            assert_eq!(&good.buf, &cut_cloned.inner.buf);
            assert_eq!(&good.position, &cut_cloned.position);

        }
    }
}
