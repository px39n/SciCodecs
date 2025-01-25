# SciCodec: Scientific Data Compression Toolkit

SciCodec is a comprehensive Python package designed for scientific data compression, providing a unified interface to various compression methods specifically tailored for scientific datasets.
https://px39n.gitbook.io/sci-codecs

## Key Features

Scientific data compression, especially for climate data, faces several key challenges:

1. Fragmented Interfaces: Different compression methods are implemented across multiple languages and specifications, making it difficult to use them consistently.

2. Limited Evaluation: Current research in scientific data compression is constrained by:
   - Inconsistent datasets used for testing different methods
   - Lack of standardized evaluation metrics and benchmarks
   - Results that are not directly comparable across studies

SciCodec addresses these challenges by providing:

1. Unified Interface: A consistent API that works across multiple compression methods and data formats
2. Standardized Evaluation: Common datasets and metrics for fair comparison of compression techniques
3. Comprehensive Testing: Support for evaluating compression across entire datasets rather than single variables

## Core Components

1. **Compressor**: Single compression task handler
   - Supports individual dataset compression
   - Configurable compression parameters
   - Error bound specification
2. **Datasets**: Scientific dataset handling module
   - API Benchmark datasets (WRF, NARR, etc.) for comparing compression performance
   - Support for NetCDF, HDF5, and other scientific data formats
   - Efficient data loading and preprocessing
   - Metadata preservation and management

3. **Evaluation Tools**: Comprehensive analysis utilities
   - Accuracy metrics calculation
   - Performance benchmarking
   - Compression ratio analysis

## Use Cases

SciCodec is particularly useful for:
- Climate and weather data compression
- Scientific simulation output compression
- Large-scale scientific data management
- Data archival with quality control
- Compression parameter optimization

The package aims to provide researchers and data scientists with efficient tools for managing large scientific datasets while maintaining data quality and accessibility.

## Installation

Basic installation:
```bash
pip install scicodec
```

Install with all basic dependencies:
```bash
pip install scicodec[basic]
```

For specific compression libraries:
```bash
# CompressAI support
pip install scicodec[compressai]

# Numcodecs support
pip install scicodec[numcodecs]
```

## Quick Start

**Dataset** (Popularbenchmark datasets)
```python
import scicodec as sc

# Download example dataset
ds = sc.dataset.download_benchmark("wrf_short")
sc.dataset.bitwise_analysis(ds)
sc.dataset.diagnostic(ds)
```

**Compression** （Support 30+ lossy methods)
```python
# Create compressor with desired method
compressor = sc.compression.Compressor(method="zfp", abs_precision=0.1)

# Compress data
compressed = compressor.encode(ds)

# Decompress data
decompressed = compressor.decode(compressed)

# Batch compression with multiple methods
batch_dict = {
    "T2": {"method": "zfp", "abs_precision": 0.1},
    "PSFC": {"method": "sz3", "abs_precision": 0.01}
    ,.....
}

batch_compressor = sc.compression.BatchCompressor(method=batch_dict)
batch_compressor.encode(ds, save_dir="./compressed")
```

**Evaluation** (Compression index, metric, statistics, and climate speicific evaluation)
```python
sc.evaluation.accuracy( "./compressed")
sc.evaluation.spatial( "./compressed")
sc.evaluation.efficiency(ds, "./compressed")
sc.evaluation.distribution(ds, "./compressed")
sc.evaluation.similarity(ds, "./compressed")
sc.evaluation.optimality(ds, "./compressed")
etc,.
```

## Requirements

- Python 3.7 or higher
- C++ compiler (for some compression libraries)
- CUDA toolkit (optional, for GPU acceleration)

For more detailed documentation and examples, visit our [documentation](https://px39n.gitbook.io/sci-codecs).
```
