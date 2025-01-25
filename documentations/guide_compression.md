
# Compression Guide

This guide demonstrates how to use the compression functionality in scicodec to compress and decompress data efficiently.

## 1. Basic Compression Concepts

### Available Compression Methods

scicodec supports multiple compression methods, categorized into three main types:



| Compression Method| Mode| Error Mode| Extended Package| Reference |
|---|---|---|---|---|
|SZ3|Lossy|abs_precision|libpressio|[code](https://github.com/szcompressor/sz3), [paper](https://dl.acm.org/doi/10.1145/3458817.3476165)|
|MGARD|Lossy|abs_precision|libpressio|[code](https://github.com/LLNL/mgard), [paper](https://ieeexplore.ieee.org/document/9355231)|
|ZFP|Lossy|fixed_ratio,bit_precision[0-64]|libpressio|[code](https://github.com/LLNL/zfp), [paper](https://ieeexplore.ieee.org/document/7760329)|
|MBT2018|Lossy|level[1-8]|compressai|[code](https://github.com/mbt2018/compression), [paper](https://arxiv.org/abs/1809.02736)|
|bmshj2018_factorized|Lossy|level[1-8]|compressai|[code](https://github.com/InterDigitalInc/CompressAI), [paper](https://arxiv.org/abs/1802.01436)|
|cheng2020_anchor|Lossy|level[1-6]|compressai|[code](https://github.com/cheng2020/compression), [paper](https://arxiv.org/abs/2002.02778)|
|Zlib|Lossless|level[0-9]|numcodecs|[code](https://github.com/madler/zlib)|
|Blosc|Lossless|level[0-9]|numcodecs|[code](https://github.com/Blosc/c-blosc)|
|LZ4|Lossless|level[0-12]|numcodecs|[code](https://github.com/lz4/lz4)|
|Zstd|Lossless|level[0-9]|numcodecs|[code](https://github.com/facebook/zstd)|
|GZip|Lossless|level[0-9]|numcodecs|[code](https://www.gnu.org/software/gzip/)|
|FPZip|Lossy|rel_precision,bit_precision[0-64]|FPZip,numcodecs|[code](https://github.com/LLNL/fpzip), [paper](https://ieeexplore.ieee.org/document/7516139)|

### Extended Compression Support

scicodec integrates several external packages to provide additional compression options:

1. **numcodecs** ([Documentation](https://numcodecs.readthedocs.io/))
   - Additional lossless codecs like Blosc, LZ4, Zstd
   - Filters and transforms
   - Custom codec development

2. **CompressAI** ([Documentation](https://interdigitalinc.github.io/CompressAI/))
   - Deep learning based compression models
   - Neural network architectures for learned compression
   - Training utilities and pre-trained models

3. **LibPressio** ([Documentation](https://libpressio.readthedocs.io/))
   - Interface to many scientific compressors
   - SZ, ZFP, MGARD integration
   - Compression metrics and analysis tools

4. **Customed** ([Custom Compressor Guide](custom_compressor.md))
- Implementing the Compressor interface
- Adding new error bounds
- Integration with scicodec's pipeline



### Unified Error Bounds

For lossy compression, different error bounds are available:

| Error Bound | Parameter | Description | Range |
|------------|-----------|-------------|--------|
| Absolute Error | abs_precision | max(|x - x'|) ≤ ε | 0 < ε < 1 |
| Relative Error | rel_precision | max(|x - x'|/|x|) ≤ ε | 0 < ε < 1 |
| Bit Precision | bit_precision | Fixed bit precision | 0 < bits < 64 |
| Fixed Ratio | fixed_ratio | size(x')/size(x) ≤ ratio | 0 < ratio < 1 |
| Level | level | Compression level | Varies by method |

## 2. Basic Usage

### Single Array Compression

```python
import scicodec as sc
import numpy as np

# Create sample data
data = np.random.rand(100, 255, 255).astype(np.float32)

# Initialize compressor
compressor = sc.compression.Compressor(method='zlib', level=5)

# Compress data
compressed = compressor.encode(data)

# Decompress data
decompressed = compressor.decode(compressed)
```

### Xarray Dataset Compression

```python
import scicodec as sc

# Load example dataset
ds = sc.dataset.download_benchmark("wrf_short")

# Define compression settings for each variable
encode_dict = {
    "T2": {"compressor": "zlib", "level": 1},
    "U10": {"compressor": "sz3", "abs_precision": 1},
    "V10": {"compressor": "fpzip", "rel_precision": 1e-4},
    "PSFC": {"compressor": "zlib", "level": 1}
}

# Initialize compressor with settings
compressor = sc.compression.Compressor(method=encode_dict)

# Compress dataset
compressed_dir = compressor.encode(ds, save_dir="./output")

# Decompress dataset
decompressed_ds = compressor.decode(compressed_dir)
```

## 3. Meta and Efficiency Information

scicodec automatically tracks and saves both compression performance metrics and metadata during compression. This information is stored in multiple locations for easy access:

### 3.1 Compression Performance Metrics

The compression performance metrics are saved in:

1. The compressor class as `compressor.efficiency_metrics` (pandas DataFrame)
2. The decoded dataset variable attributes
3. A CSV file "efficiency_metrics.csv" in the output directory

| Metric | Unit | Description |
|--------|------|-------------|
| Compression Ratio | >1 | Original size / Compressed size |
| Encoding Speed | MB/s | Compression speed |
| Decoding Speed | MB/s | Decompression speed |
| Original Bits | bytes | Size of original data |

```python
# Get compression metrics
metrics = compressor.efficiency_metrics
print(metrics)
print(decompressed_ds["T2"].attrs["Original Bits"])
```


### 3.2 Metadata Storage

For each compressed variable, scicodec saves essential metadata including:

1. Data characteristics:
   - Original min/max values
   - Data shape (h, w, l dimensions) 
   - Original data type
   - Original size in bits

2. Compression settings:
   - Compression method used
   - Parameter dictionary
   - Compression performance metrics

This metadata is stored in:
1. A netCDF file "meta_ds.nc" in the output directory
2. Variable attributes in the decoded dataset


```python
# Access metadata
meta_ds = xr.open_dataset("output/meta_ds.nc")
print(meta_ds)
print(decompressed_ds["T2"].attrs["method"])
```



## 4. Advanced Features

### Finding Optimal Compression Parameters

scicodec can automatically find optimal compression parameters based on target compression rate. The find_config_with_rate() method will:

1. Analyze your dataset characteristics
2. Test multiple compression configurations 
3. Find parameters that achieve the target compression rate while minimizing error
4. Support both single variable and multi-variable datasets
5. Return a configuration dictionary ready to use with the Compressor

This automated parameter tuning helps you achieve optimal compression without manual trial and error.

```python
# Find configuration for target compression rate
config = compressor.find_config_with_rate(dataset, compression_rate=100)
```

### Batch Processing

For processing multiple datasets with different compression methods, scicodec provides a BatchCompressor class. This allows you to:

1. Apply different compression configurations to multiple datasets
2. Process them in parallel for better performance 
3. Save all compressed outputs in an organized directory structure

For example, you can compress the same dataset with different methods to compare their performance:

```python


import scicodec as sc
example_ds = sc.dataset.download_benchmark("wrf")

# Compress the Xarray dataset.
sz3_dict={'T2': {'abs_precision': 0.72265625, 'compressor': 'sz3'}, 
...
zfp_dict={'T2': {'bit_precision': 6, 'compressor': 'zfp'}, 
...
batch_dict = {
    "work1": sz3_dict,
    'work2': {'compressor': "bmshj2018_factorized", 'level': 7},   
    'work3': zfp_dict,    
}

# Initialize batch compressor
batch_compressor = sc.compression.BatchCompressor(method=encode_dict)

# Batch compress
batch_compressor.encode(datasets, save_dir="./batch_output")

# Batch decompress
decompressed_datasets = batch_compressor.decode("./batch_output")
```



## 5. Output Directory Structure

When saving compressed datasets, the following directory structure is created:

```
saved_dir/
├── variable1/
│   └── compressed.bin
├── variable2/
│   └── compressed.bin
├── meta_ds.nc
├── efficiency_metrics.pkl
└── encoding_list.yaml
```
