# Frequently Asked Questions (FAQ)

## General Questions

### What is SciCodec?
SciCodec is a comprehensive Python package designed for scientific data compression, providing a unified interface to various compression methods specifically tailored for scientific datasets.

### What types of data can I compress with SciCodec?
SciCodec is optimized for scientific data, particularly:
- Climate and weather data
- Scientific simulation outputs
- Large-scale numerical datasets
- NetCDF and HDF5 files

### Which compression methods are supported?
SciCodec supports multiple compression methods across three categories:
- **AI-Based Methods**: bmshj2018_factorized, mbt2018, cheng2020_anchor
- **Traditional Methods**: zlib, lz4, blosc, gzip, zstd, fpzip
- **Scientific Methods**: sz3, zfp, mgard

## Usage Questions

### How do I choose the right compression method?
The choice depends on your requirements:
- For highest compression ratio: AI-based methods
- For fastest compression: Traditional methods
- For error-bounded compression: Scientific methods

### What are the key parameters for compression?
Key parameters include:
- `abs_precision`: Absolute error bound
- `rel_precision`: Relative error bound
- `level`: Compression level (1-6)
- `bit_precision`: Bit precision (0-32)

### How do I handle batch compression?
Use the BatchCompressor class:
