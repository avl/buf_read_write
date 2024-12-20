# buf_read_write

[Repo](https://github.com/avl/buf_read_write)
[Docs](https://docs.rs/buf_read_write/latest/buf_read_write/)

![build](https://github.com/avl/arcshift/actions/workflows/rust.yml/badge.svg)
![build](https://github.com/avl/arcshift/actions/workflows/mutants.yml/badge.svg)
![build](https://github.com/avl/arcshift/actions/workflows/clippy.yml/badge.svg)

This crate contains a a buffered io implementation, a combination of `std::io::BufReader`
and `std::io::BufWriter`. This allows buffered input and output, for example with a `std::fs::File`.

The difference between this and regular BufWriter and BufReader is that a single instance can buffer
both reads and writes to the same backing object (for example a File).

This crate is meant to be used with file-like objects, and requires the backing implementation
to implement `std::io::Seek`.

## Advantages:

 * Seeking is supported (and required of the backing IO object), and does not invalidate the buffer.
 * Reads and writes share a single buffer. Reads can be satisfied from uncommitted writes.
 * No unsafe code
 * Extensive test suite
    * Custom random chaos testing
    * Exhaustive testing of all 4 item sequences of operations read, write and seek with limited offsets.
    * Cargo mutants testing
 * Performance overhead within 50% overhead of regular BufReader/BufWriter, for small reads/writes. 
 * Documented

## Use cases:

 * Manipulating data in on-disk formats, without doing unnecessary IO operations
 * General random access IO on files or file-like objects.

## Limitations:
 * buf_read_write is not a general disk caching library. The buffer is always contiguous.


## How to avoid unnecessary IO

Let's say you have a large data file stored on disk. The file contains many chunks, each with
its own header.

You wish to traverse the file, and rewrite a few fields in each header.

You can do the following sequence of operations:

1) Seek to the first header
2) Read this header entirely into memory
3) Seek to a sub-field in the header
4) Write this particular field (perhaps only a few bytes)
5) Flush the BufStream.

Only one IO-operation will be emitted (this is of course the point of buffered IO).
Note however, that this is true even if the written sub-fields are not adjacent - as long as they fit within the buffer
size.

If you do not first read the file, and then write non-contiguous regions, there will be an IO operation for
every such non-contiguous region. The reason is that `buf_read_write` does not support non-contiguous buffers,
and thus cannot represent a non-contiguous set of updates.

Note, a consequence of this is that the number of bytes actually written can be (much) larger than the number of
actually modified bytes. If this is undesirable, make sure to call 'flush' after every write.

