use std::io::{BufReader, BufWriter, Cursor, Read, Write};
use criterion::{criterion_group, criterion_main, Criterion};
use bufstream2::BufStream;

fn many_small_writes_bufstream() -> BufStream<Cursor<Vec<u8>>> {
    let mut v = Vec::new();
    let mut bufstream = BufStream::with_capacity(Cursor::new(v), 100);

    for i in 0..100 {
        bufstream.write(&[i]).unwrap();
    }
    bufstream
}

fn many_small_writes_bufwriter() -> BufWriter<Cursor<Vec<u8>>> {
    let mut v = Vec::new();
    let mut bufstream = BufWriter::new(Cursor::new(v));

    for i in 0..100 {
        bufstream.write(&[i]).unwrap();
    }
    bufstream
}

fn many_small_reads_bufstream() -> BufStream<Cursor<Vec<u8>>> {
    let mut v = vec![42u8;100];
    let mut bufstream = BufStream::with_capacity(Cursor::new(v),100);

    for i in 0..100 {
        bufstream.read(&mut [0]).unwrap();
    }
    bufstream
}

fn many_small_reads_bufreader() -> BufReader<Cursor<Vec<u8>>> {
    let mut v = vec![42u8;100];
    let mut bufstream = BufReader::new(Cursor::new(v));

    for i in 0..100 {
        bufstream.read(&mut [0]).unwrap();
    }
    bufstream
}



fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("many_small_writes_bufstream", |b| b.iter(|| many_small_writes_bufstream()));
    c.bench_function("many_small_writes_bufwriter", |b| b.iter(|| many_small_writes_bufwriter()));
    c.bench_function("many_small_reads_bufstream", |b| b.iter(|| many_small_reads_bufstream()));
    c.bench_function("many_small_reads_bufreader", |b| b.iter(|| many_small_reads_bufreader()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);