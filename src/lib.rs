extern crate core;

use std::collections::VecDeque;
use std::io::{BufWriter, ErrorKind, Read, Seek, SeekFrom, Write};
use std::ops::Range;

#[derive(Clone, Debug)]
struct MovingBuffer {
    offset: usize,
    data: VecDeque<u8>,
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
            data: VecDeque::with_capacity(capacity),
            offset: 0,
        }
    }

    fn flush(&mut self, flusher: &mut impl FnMut(usize, &[u8]) -> Result<(), std::io::Error>)  -> Result<(), std::io::Error> {
        println!("Flushing {:?}", self);
        let slices = self.data.as_slices();
        dbg!(slices);
        println!("Caller flusher 1");
        flusher(self.offset, slices.0)?;
        println!("Caller flusher 1b");
        if !slices.1.is_empty() {
            println!("Caller flusher 2");
            flusher(self.offset+slices.0.len(), slices.1)?;
        }
        println!("Dat aclear");
        self.data.clear();
        Ok(())
    }
    fn end(&self) -> usize {
        self.offset + self.data.len()
    }
    fn write_at(&mut self, position: usize, data: &[u8], write_at: &mut impl FnMut(usize,  &[u8]) -> Result<(), std::io::Error>) -> Result<(), std::io::Error> {
        let free_capacity = self.data.capacity() - self.data.len();

        dbg!(position, self.end(), free_capacity, data.len());
        if position == self.end() && free_capacity >= data.len() {
            println!("Case 1 extend {:?} by {:?}", self.data, data);;
            self.data.extend(data);
            Ok(())
        } else if position >= self.offset && position + data.len() <= self.end() {
            let relative_offset = position - self.offset;
            for (dst,src) in self.data.range_mut(relative_offset..relative_offset+data.len()).zip(data) {
                *dst = *src;
            }
            println!("Case 2 slice overwrite {}..{} with {:?}", relative_offset, relative_offset+data.len(), data);
            Ok(())
        } else {
            println!("Restart-case, current: {:?}", self);
            self.flush(write_at)?;
            self.data.clear();
            if data.len() < self.data.capacity() {
                println!("Case 3 post flush new buf: {:?}", data);
                self.offset = position;
                self.data.extend(data);
                Ok(())
            } else {
                println!("Case 4 write-through: {:?} - {:?}", position, data);
                write_at(position, data)?;
                Ok(())
            }
        }
    }

    fn read_at(&mut self, position: usize, buf: &mut [u8], read_at: &mut impl FnMut(usize,  &mut [u8]) -> std::io::Result<()> ) -> std::io::Result<()> {
        let read_range = position .. position+buf.len();
        let buffered_range = self.offset..self.end();

        if read_range.start < buffered_range.start {
            read_at(read_range.start, &mut buf[0..buffered_range.start - read_range.start])?;
        }

        if let Some(overlap) = overlap(read_range.clone(), buffered_range.clone()) {
            let overlapping_src_slice = self.data.range((overlap.start-self.offset)..(overlap.end-self.offset));
            for (dst,src) in buf[overlap.start-position..overlap.end-position].iter_mut().zip(overlapping_src_slice) {
                *dst = *src;
            }
        }

        if read_range.end > buffered_range.end {
            read_at(buffered_range.end, &mut buf[0..read_range.end - buffered_range.end])?;
        }
        Ok(())
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
            println!("REAL WRITE writer at {} {:?} @ offset {}", offset, data, offset);
            if offset != self.inner.stream_position()? as usize {
                println!("Inner seek {}", offset);
                self.inner.seek(SeekFrom::Start(offset as u64))?;
            }

            println!("Inner write {:?}", data);
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
impl<T:Read+Seek> Read for BufStream<T> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.buffer.read_at(self.position, buf, &mut |pos,data|{
            if self.position != pos {
                self.inner.seek(SeekFrom::Start(pos as u64))?;
            }
            self.inner.read(data)?;
            Ok(())
        })?;
        Ok(buf.len())
    }
}


#[cfg(test)]
mod tests {
    use rand::{Rng, RngCore};
    use super::*;

    #[derive(Default, PartialEq, Eq, Debug, Clone)]
    struct FakeWrite {
        buf: Vec<u8>,
        position: usize,
    }
    impl Read for FakeWrite {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let end = (self.position+buf.len()).min(self.buf.len());
            let got = end-self.position;
            buf[0..got].copy_from_slice(&self.buf[self.position..self.position+got]);
            self.position += got;
            Ok(got)
        }
    }
    impl Write for FakeWrite {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if buf[0] == 159 && self.position != 0{
                println!("STop");
            }
            println!("ACtual fake write {:?} at {}", buf, self.position);
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
    impl Seek for FakeWrite {
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

    #[test]
    fn it_works() {
        for i in 0..1000 {
            fuzz(i);
        }
    }
    #[test]
    fn regression25() {

        fuzz(25);

    }
    fn fuzz(seed: u64) {
        let mut good = FakeWrite::default();
        let mut cut_inner = FakeWrite::default();
        let mut cut = BufStream::with_capacity(cut_inner, 16);
        use rand::SeedableRng;
        let mut small_rng = rand::rngs::SmallRng::seed_from_u64(seed);
        println!("Seed: {}", seed);
        for _ in 0..2 {
            match small_rng.gen_range(0..3) {
                0 if good.buf.len() > 0=> {
                    let seek_to = small_rng.gen_range(0..good.buf.len());
                    println!("Seek to {}", seek_to);
                    good.seek(SeekFrom::Start(seek_to as u64)).unwrap();
                    cut.seek(SeekFrom::Start(seek_to as u64)).unwrap();
                }
                1  if good.buf.len() - good.position > 0 => {
                    let read_bytes = small_rng.gen_range(0..(good.buf.len() - good.position) as usize);
                    println!("Read {}", read_bytes);
                    let mut goodbuf = vec![0u8; read_bytes];
                    good.read(&mut goodbuf).unwrap();

                    let mut cutbuf = vec![0u8; read_bytes];
                    cut.read(&mut cutbuf).unwrap();
                }
                0|1|2 => {
                    let write_bytes = small_rng.gen_range(0..10);
                    let mut buf = vec![0u8; write_bytes];
                    small_rng.fill_bytes(&mut buf);
                    println!("WRITE {:?}\nStates cut: {:?}, states good: {:?}", buf, cut, good);
                    good.write(&buf).unwrap();
                    cut.write(&buf).unwrap();
                }
                _ => unreachable!()
            }
            println!("Cut state: {:?}", cut);
            let mut cut_cloned = cut.clone();
            cut_cloned.flush().unwrap();
            assert_eq!(&good.buf, &cut_cloned.inner.buf);
            assert_eq!(&good.position, &cut_cloned.position);

        }
    }
}
