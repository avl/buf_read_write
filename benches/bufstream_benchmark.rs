use buf_read_write::BufStream;
use criterion::{criterion_group, criterion_main, Criterion};
use std::io::{BufReader, BufWriter, Cursor, Read, Write};

fn many_small_writes_bufstream() -> BufStream<Cursor<Vec<u8>>> {
    let v = Vec::new();
    let mut bufstream = BufStream::new(Cursor::new(v));

    for i in 0..100 {
        bufstream.write(&[i]).unwrap();
    }
    bufstream
}

fn many_small_writes_bufwriter() -> BufWriter<Cursor<Vec<u8>>> {
    let v = Vec::new();
    let mut bufstream = BufWriter::new(Cursor::new(v));

    for i in 0..100 {
        bufstream.write(&[i]).unwrap();
    }
    bufstream
}

fn many_small_reads_bufstream() -> BufStream<Cursor<Vec<u8>>> {
    let v = vec![42u8; 400];
    let mut bufstream = BufStream::new(Cursor::new(v));

    for _i in 0..100 {
        bufstream.read(&mut [0,1,2,3]).unwrap();
    }
    bufstream
}

fn many_small_reads_bufreader() -> BufReader<Cursor<Vec<u8>>> {
    let v = vec![42u8; 400];
    let mut bufstream = BufReader::new(Cursor::new(v));

    for _i in 0..100 {
        bufstream.read(&mut [0,1,2,3]).unwrap();
    }
    bufstream
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("many_small_writes_bufstream", |b| {
        b.iter(|| many_small_writes_bufstream())
    });
    c.bench_function("many_small_writes_bufwriter", |b| {
        b.iter(|| many_small_writes_bufwriter())
    });
    c.bench_function("many_small_reads_bufstream", |b| {
        b.iter(|| many_small_reads_bufstream())
    });
    c.bench_function("many_small_reads_bufreader", |b| {
        b.iter(|| many_small_reads_bufreader())
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
