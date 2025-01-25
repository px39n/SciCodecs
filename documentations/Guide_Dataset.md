

# Dataset Module Guide

The dataset module provides functionality for downloading benchmark datasets and standardizing data for compression testing.

## 1. Benchmark Datasets

### Available Datasets
```python
import scicodec as sc

# Available benchmark datasets:
# - "era5": ERA5 reanalysis data from ECMWF
# - "wrf_short": Short sample of WRF model output
# - "wrf": Full WRF model output dataset

# Download a benchmark dataset
ds = sc.dataset.download_benchmark("wrf")  # Returns xarray Dataset
```

## 2. Dataset Analysis

### Bit Information Analysis
Analyze and visualize the bit-level information of your dataset:

```python
from scicodec.dataset import plot_bit_information
import matplotlib.pyplot as plt

# Create visualization of bit-level information
fig, ax1, ax1right = plot_bit_information(ds)
```

This visualization helps understand:
- Data distribution across variables
- Bit patterns in the data
- Potential compression opportunities

## 3. Dataset Standardization

```python
# Standardize dataset for better compression
standardized_ds = sc.dataset.standardize_ds(ds)
```

The standardization process:
- Normalizes each variable to zero mean and unit variance
- Preserves dataset attributes and coordinates
- Improves compression efficiency for many algorithms


This guide focuses on:
1. Downloading benchmark datasets
2. Analyzing dataset characteristics
3. Dataset standardization functionality