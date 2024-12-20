use buf_read_write::BufStream;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[test]
fn test_bufstream() {
    let my_data = vec![1, 2, 3, 4, 5];

    let mut stream = BufStream::new(Cursor::new(my_data));

    stream.seek(SeekFrom::Start(1)).unwrap();
    stream.write_all(&[42]).unwrap();

    let mut data = [0, 0];
    stream.seek(SeekFrom::Start(1)).unwrap();
    stream.read(&mut data).unwrap();
    assert_eq!(data, [42, 3]);
}

#[test]
fn test_seeks() {
    let my_data = vec![0, 1, 2, 3, 4];

    let mut stream = BufStream::new(Cursor::new(my_data));

    stream.seek(SeekFrom::Start(1)).unwrap();
    let mut buf = [0];
    stream.read(&mut buf).unwrap();
    assert_eq!(buf[0], 1);

    stream.seek(SeekFrom::Current(-1)).unwrap();
    let mut buf = [0];
    stream.read(&mut buf).unwrap();
    assert_eq!(buf[0], 1);

    stream.seek(SeekFrom::Current(0)).unwrap();
    let mut buf = [0];
    stream.read(&mut buf).unwrap();
    assert_eq!(buf[0], 2);

    stream.seek(SeekFrom::End(-1)).unwrap();
    let mut buf = [0];
    stream.read(&mut buf).unwrap();
    assert_eq!(buf[0], 4);

    stream.seek(SeekFrom::End(-5)).unwrap();
    stream.seek(SeekFrom::Current(i64::MIN)).unwrap_err();
    stream.seek(SeekFrom::End(-6)).unwrap_err();
}

#[derive(Default)]
struct Instrumentation {
    writes: AtomicUsize,
    reads: AtomicUsize,
    seeks: AtomicUsize,
    flushes: AtomicUsize,
}

#[derive(Default)]
struct InstrumentedCursor {
    cursor: Cursor<Vec<u8>>,
    instrumentation: Arc<Instrumentation>,
}

impl Read for InstrumentedCursor {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.instrumentation.reads.fetch_add(1, Ordering::Relaxed);
        self.cursor.read(buf)
    }
}
impl Write for InstrumentedCursor {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.instrumentation.writes.fetch_add(1, Ordering::Relaxed);
        self.cursor.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.instrumentation.flushes.fetch_add(1, Ordering::Relaxed);
        self.cursor.flush()
    }
}
impl Seek for InstrumentedCursor {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.instrumentation.seeks.fetch_add(1, Ordering::Relaxed);
        self.cursor.seek(pos)
    }
}

#[test]
fn test_writes_are_buffered() {
    let writer = InstrumentedCursor::default();
    let instrumentation = writer.instrumentation.clone();
    let mut bufstream = BufStream::new(writer);

    for i in 0..100 {
        bufstream.write_all(&[i]).unwrap();
    }
    assert_eq!(instrumentation.writes.load(Ordering::Relaxed), 0);
    bufstream.flush().unwrap();
    assert_eq!(instrumentation.writes.load(Ordering::Relaxed), 1);
}
#[test]
fn test_seeks_are_virtualized() {
    let writer = InstrumentedCursor::default();
    let instrumentation = writer.instrumentation.clone();
    let mut bufstream = BufStream::new(writer);

    for i in 0..100 {
        bufstream.seek(SeekFrom::Start(i)).unwrap();
    }
    assert_eq!(instrumentation.seeks.load(Ordering::Relaxed), 0);
    bufstream.flush().unwrap();
    assert_eq!(instrumentation.seeks.load(Ordering::Relaxed), 0);
}
#[test]
fn test_reads_are_buffered() {
    let cursor = InstrumentedCursor::default();
    let instrumentation = cursor.instrumentation.clone();
    let mut bufstream = BufStream::new(cursor);

    for i in 0..100 {
        bufstream.write_all(&[i]).unwrap();
    }
    bufstream.seek(SeekFrom::Start(0)).unwrap();
    for i in 0..100 {
        let mut buf = [0];
        bufstream.read(&mut buf).unwrap();
        assert_eq!(buf[0], i as u8);
    }

    assert_eq!(instrumentation.writes.load(Ordering::Relaxed), 0); //Buffered
    assert_eq!(instrumentation.reads.load(Ordering::Relaxed), 0); //Read from buffer
    assert_eq!(instrumentation.flushes.load(Ordering::Relaxed), 0);
    bufstream.flush().unwrap();
    assert_eq!(instrumentation.flushes.load(Ordering::Relaxed), 1);
    assert_eq!(instrumentation.writes.load(Ordering::Relaxed), 1); //Buffered
    assert_eq!(instrumentation.reads.load(Ordering::Relaxed), 0); //Read from buffer
}
