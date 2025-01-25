I'll reorganize the guide to strictly follow your notebook structure:

# Evaluation Guide

## 1. Objective Indicators


| Type | Metric | Range | Threshold | Formula |
|------|---------|--------|-----------|----------|
| Error-based | RMSE | [0, ∞) | Application dependent | √(1/n∑(s̃ᵢ - sᵢ)²) |
| | RMSZ | [0,∞) | < 0.5 | √(1/N∑zᵢ²), where zᵢ=(sᵢ-s̃ᵢ)/σₛ |
| | NRMSE | [0, ∞) | Application dependent | √(∑(sᵢ - s̃ᵢ)²/n) / σₛ |
| | MAE | [0, ∞) | Application dependent | 1/n∑|s̃ᵢ - sᵢ| |
| | AE | (-∞, ∞) | ≈ 0 | 1/n∑(s̃ᵢ - sᵢ) |
| Signal-based | PSNR | [0, ∞) | ≥ 30 dB | 20log₁₀(max(S)/RMSE) |
| | SNR | [0, ∞) | Application dependent | 20log₁₀(σₛ/RMSE) |
| | BRI | [0, 1] | ≥ 0.99 | Iretained/Ioriginal |
| Error Bounds | MaxAE | [0, ∞) | Depends on user | max|s̃ᵢ - sᵢ| |
| | MRE | [0, 1] | < 5% | max|(s̃ᵢ - sᵢ)/sᵢ| |
| Similarity | SSIM | [0, 1] | ≥ 0.95 | (2μₛμₛ̃)(2σₛₛ̃)/((μₛ² + μₛ̃²)(σₛ² + σₛ̃²)) |
| | DSSIM | [0, 1] | ≥ 0.99919 | 1/M∑SSIM(xᵢ,yᵢ) |
| | PCC | [0, 1] | ≥ 0.99999 | Cov(S,S̃)/(σₛ·σₛ̃) |
| | CV | [0, ∞) | Application dependent | 1/n∑|∇s̃ᵢ - ∇sᵢ|² |
| Distribution | KS | [0,1] | ≤ 0.05 for N ∈ [10⁵, 10⁸] | supₓ|F(x) - G(x)| |
| | WD | [0,∞) | Application dependent | (∫₀¹|F⁻¹(t) - G⁻¹(t)|ᵖdt)^(1/p) |
| | KLD | [0,∞) | Application dependent | ∑P(i)log(P(i)/Q(i)) |
| | AD | [0,∞) | Application dependent | n∫(Fₙ(x)-F(x))²/F(x)(1-F(x))dF(x) |

Note: RMSE: Root Mean Square Error, RMSZ: Root Mean Square Z-score, NRMSE: Normalized Root Mean Square Error, MAE: Mean Absolute Error, AE: Average Error, PSNR: Peak Signal-to-Noise Ratio, SNR: Signal-to-Noise Ratio, MaxAE: Maximum Absolute Error, MRE: Maximum Relative Error, SSIM: Structural Similarity Index, DSSIM: Structural Dissimilarity, PCC: Pearson Correlation Coefficient, CV: Contrast Variance, KS: Kolmogorov-Smirnov, WD: Wasserstein Distance, KLD: Kullback-Leibler Divergence, AD: Anderson-Darling, BRI: Bit-wise Real Information


### 1.1 Efficiency Metrics
```python
import scicodec as sc

# Define paths and data
batch_list = ["work1", "work2", "work3"]
saved_dir = "./datasets/batch_output"
batch_dir = [f"{saved_dir}/{batch}.nc" for batch in batch_list]
original_ds = './datasets/wrf.nc'

# Calculate efficiency metrics
eff_metrics = sc.evaluation.efficiency(batch_dir)
eff_metrics.groupby(['var', 'method']).mean(numeric_only=1)
```

### 1.2 Accuracy Metrics
```python
# Calculate accuracy metrics
acc_metrics = sc.evaluation.accuracy(original_ds, batch_dir)
acc_metrics.groupby(['var', 'method']).mean(numeric_only=1)
```

## 2. Spatial Analysis
```python
import scicodec as sc

var_list = ["LAI", "T2", "U10", "PSFC"]
sc.evaluation.plot_spatial_error(batch_dir, var_list, original_ds)
```
![](https://i.imgur.com/ww45rZH.png)

## 3. Time Series Analysis
```python
# Temporal analysis
sc.evaluation.temporal(batch_dir, var_list, original_ds)
```

![](https://i.imgur.com/1kQvlom.png)

## 4. Data Distribution Analysis
```python
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import xarray as xr

# Create standardized box plots
fig = plt.figure(figsize=(6, 4.5))
ax = fig.add_subplot(111)

# Plot configuration
colors = ['#abd9e9', '#8ACEB9', '#C38081', '#85BFC3']
plt.ylabel('Standardized')
plt.grid(True, axis='y', alpha=0.2)
plt.ylim(-4, 4)
```

![](https://i.imgur.com/DvHYGg9.png)

## 5. Parameter Optimization
```python
plot_compression_ratios(df, method_list, var_list, x, y, level, x_lim=None, y_lim=None)

```


This structure exactly matches your notebook's organization:
1. Objective Indicators (Efficiency and Accuracy)
2. Spatial Analysis
3. Time Series Analysis
4. Data Distribution Analysis
5. Parameter Optimization

Each section includes the essential code examples from your notebook. Would you like me to expand on any particular section?
