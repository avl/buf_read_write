# buf_read_write

[Repo](https://github.com/avl/buf_read_write)
[Docs](https://docs.rs/buf_read_write/latest/buf_read_write/)

This crate contains a a buffered io implementation, a combination of `std::io::BufReader`
and `std::io::BufWriter`. This allows buffered input and output, for example with a `std::fs::File`.

Advantages:

 * Seeking is supported, and does not invalidate the buffer.
 * Reads and writes share a single buffer. Reads can be satisfied from uncommitted writes.
 * No unsafe code
 * Extensive test suite
    * Custom random chaos testing
    * Exhaustive testing of all 4 item sequences of operations read, write and seek with limited offsets.
    * Cargo mutants testing
 * Performance overhead within 50% overhead of regular BufReader/BufWriter, for small reads/writes. 
 * Documented



