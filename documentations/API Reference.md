









# SciCodec API Reference


## Compression Module API Reference

### Classes

#### `BatchCompressor`
Handles batch compression operations for multiple files or datasets.

#### Constructor
```python
BatchCompressor(method=None, **kwargs)
```
**Parameters:**
- `method`: Compression method specification
  - String: Single compression method (e.g., "sz3")
  - Dict: Mapping variables to compression settings
  - None: For decoding only
- `**kwargs`: Compression parameters
  - `abs_precision`: Absolute precision (e.g., 1e-3)
  - `rel_precision`: Relative precision

#### Methods
- `encode(data, save_dir=None)`: Batch encodes data to specified directory
- `decode(data, output_dir=None)`: Batch decodes compressed data

### `Compressor`
Core compression class handling individual compression tasks.

#### Constructor
```python
Compressor(method=None, **kwargs)
```
**Parameters:**
- `method`: Same as BatchCompressor
- `**kwargs`: Same as BatchCompressor

#### Methods

##### Compression Operations
```python
encode(data, method=None, para_dict=None, save_dir=None)
decode(data, output_dir=None)
```

##### Batch Operations
```python
batch_encode(data, save_dir=None)
batch_decode(data, output_dir=None)
```

##### Parameter Optimization
```python
find_config_with_rate(data, compression_rate=100, max_attempts=10, verbose=1)
```
Finds optimal compression parameters for target compression rate.

**Parameters:**
- `data`: Input data to compress
- `compression_rate`: Target compression ratio (default: 100)
- `max_attempts`: Maximum optimization attempts (default: 10)
- `verbose`: Verbosity level (default: 1)

**Returns:**
- Dictionary of optimal configurations per variable

##### Backend-Specific Methods

###### AI-Based Compression
```python
compressai_encode_array(data, method, para_dict)
compressai_decode_array(data, method, para_dict)
```
Handles AI-based compression methods (bmshj2018_factorized, mbt2018, cheng2020_anchor).

###### Traditional Compression
```python
numcodecs_encode_array(data, method, para_dict)
numcodecs_decode_array(data, method, para_dict)
```
Handles traditional compression methods (zlib, lz4, blosc, gzip, zstd, fpzip).

###### Scientific Data Compression
```python
libpressio_encode_array(data, method, para_dict)
libpressio_decode_array(data, method, para_dict)
```
Handles scientific compression methods (sz3, zfp, mgard).

##### Configuration Translators
```python
aicompressor_config_translator(method, para_dict)
numcodec_config_translator(method, para_dict)
libpressio_config_translator(method, para_dict)
```
Convert between different parameter formats for various backends.

## Supported Compression Methods

### AI-Based Methods
- `bmshj2018_factorized`
- `mbt2018`
- `cheng2020_anchor`

### Traditional Methods
- `zlib`
- `lz4`
- `blosc`
- `gzip`
- `zstd`
- `fpzip`

### Scientific Methods
- `sz3`
- `zfp`
- `mgard`

## Parameter Types

### General Parameters
- `abs_precision`: Absolute error bound
- `rel_precision`: Relative error bound
- `level`: Compression level (1-6)
- `bit_precision`: Bit precision (0-32)

### Method-Specific Parameters
- **AI Methods:**
  - `quality`: Quality level
  - `device`: Computing device ("cpu" or "cuda")

- **Traditional Methods:**
  - `clevel`: Compression level
  - `acceleration`: LZ4 acceleration factor

- **Scientific Methods:**
  - `mode`: Compression mode
  - `fixed_ratio`: Fixed compression ratio
  - `error_bound_type`: Error bound type for MGARD

## Usage Examples

### Basic Compression
```python
compressor = Compressor("sz3", abs_precision=1e-3)
compressed = compressor.encode(data)
decompressed = compressor.decode(compressed)
```

### Batch Compression
```python
batch_compressor = BatchCompressor({
    "workspace1": {"compressor": "sz3", "abs_precision": 1e-3},
    "workspace2": {"compressor": "zfp", "bit_precision": 16}
})
batch_compressor.encode(data, save_dir="output")
```

### Parameter Optimization
```python
compressor = Compressor("zfp")
optimal_config = compressor.find_config_with_rate(data, compression_rate=100)
```



## Evaluation Module (`scicodec.evaluation`)

### Objective Metrics

#### `efficiency(batch_dir)`
Calculates compression efficiency metrics for compressed datasets.

**Parameters:**
- `batch_dir` (list): List of paths to compressed NetCDF files

**Returns:**
- DataFrame with columns:
  - `encoding (MB/s)`: Compression speed
  - `decoding (MB/s)`: Decompression speed
  - `compression_ratio`: Original size / Compressed size
  - `original_bits`: Original data size in bits

**Example:**
```python
metrics = sc.evaluation.efficiency(batch_dir)
metrics.groupby(['var', 'method']).mean(numeric_only=1)
```

#### `accuracy(original_ds, batch_dir)`
Computes accuracy metrics between original and compressed datasets.

**Parameters:**
- `original_ds` (str): Path to original dataset
- `batch_dir` (list): List of paths to compressed datasets

**Returns:**
- DataFrame with accuracy metrics per variable and method

**Example:**
```python
acc_metrics = sc.evaluation.accuracy(original_ds, batch_dir)
acc_metrics.groupby(['var', 'method']).mean(numeric_only=1)
```

### Spatial Analysis

#### `plot_spatial_error(batch_dir, var_list, original_ds)`
Visualizes spatial error distribution between original and compressed data.

**Parameters:**
- `batch_dir` (list): List of paths to compressed files
- `var_list` (list): List of variables to analyze
- `original_ds` (str): Path to original dataset

**Example:**
```python
sc.evaluation.plot_spatial_error(batch_dir, var_list, original_ds)
```

### Temporal Analysis

#### `temporal(batch_dir, var_list, original_ds)`
Analyzes temporal characteristics and errors in compressed time series data.

**Parameters:**
- `batch_dir` (list): List of paths to compressed files
- `var_list` (list): List of variables to analyze
- `original_ds` (str): Path to original dataset

**Example:**
```python
sc.evaluation.temporal(batch_dir, var_list, original_ds)
```

### Distribution Analysis

#### `plot_distribution(batch_dir, var_list, original_ds)`
Creates box plots comparing data distributions between original and compressed datasets.

**Parameters:**
- `batch_dir` (list): List of paths to compressed files
- `var_list` (list): List of variables to analyze
- `original_ds` (str): Path to original dataset

**Example:**
```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6, 4.5))
sc.evaluation.plot_distribution(batch_dir, var_list, original_ds)
```

### Parameter Optimization

#### `plot_compression_ratios(df, method_list, var_list, x, y, level)`
Visualizes compression performance across different parameters.

**Parameters:**
- `df` (DataFrame): Compression results data
- `method_list` (list): Compression methods to compare
- `var_list` (list): Variables to analyze
- `x` (str): X-axis metric (e.g., "compression_ratio")
- `y` (str): Y-axis metric (e.g., "encoding (MB/s)")
- `level` (str): Parameter being tuned (e.g., "precision")

**Example:**
```python
plot_compression_ratios(results_df, ["zfp"], ["T2", "U10"], 
                       "compression_ratio", "encoding (MB/s)", "precision")
```

## Dataset Module (`scicodec.dataset`)

### `download_benchmark(name)`
Downloads benchmark dataset for compression testing.

**Parameters:**
- `name` (str): Name of benchmark dataset (e.g., "wrf")

**Returns:**
- xarray.Dataset: Loaded benchmark dataset

**Example:**
```python
example_ds = sc.dataset.download_benchmark("wrf")
```

### `plot_bit_information(dataset)`
Visualizes bit-level information about variables in a dataset.

**Parameters:**
- `dataset` (xarray.Dataset): Dataset to analyze

**Returns:**
- tuple: (figure, primary axis, secondary axis)

**Example:**
```python
fig, ax1, ax1right = sc.dataset.plot_bit_information(example_ds)
```

