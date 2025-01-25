import os
import xarray as xr
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance  # 新增
from tqdm import tqdm
def accuracy(original_path, batch_path):
    """
    Compute accuracy metrics between original and batch datasets.

    Parameters:
        original_path (list): List of file paths for the original datasets.
        batch_path (list): List of file paths for the batch datasets.

    Returns:
        pd.DataFrame: A DataFrame containing the accuracy metrics for each variable.
    """

    rows = []
    ds1 = xr.open_dataset(original_path)
    total_computation=len(batch_path)*len(ds1.data_vars)
    ds1.close()
    computation=0

    progress_bar = tqdm(total=total_computation, desc="Computing accuracy")
    for batch_file in batch_path:
        orig_file=original_path
        ds1 = xr.open_dataset(orig_file)
        ds2 = xr.open_dataset(batch_file)
        
        for var in ds1.data_vars:
            computation+=1
            progress_bar.update(1)
            if var in ds2.data_vars:
                x = ds1[var].values.flatten()
                r = ds2[var].values.flatten()
                # Compute accuracy metrics
                rmse = np.sqrt(np.mean((x - r) ** 2))
                srr = np.log2(np.std(x) / np.std(r)) if np.std(r) != 0 else np.inf
                mae = np.mean(np.abs(x - r))
                correlation = np.corrcoef(x, r)[0, 1] if len(x) > 1 else np.nan
                max_error = np.max(np.abs(x - r))
                mean_error = np.mean(x - r)
                peak_error = np.log2((np.max(x) - np.min(x)) / (2 * np.max(np.abs(r)))) if np.max(np.abs(r)) != 0 else np.inf

                # Additional metrics
                sigma_s = np.std(x)
                rmsz = np.sqrt(np.mean(((x - r) / sigma_s) ** 2)) if sigma_s != 0 else np.inf
                non_zero_mask = x != 0
                mre = np.max(np.abs((x[non_zero_mask] - r[non_zero_mask]) / x[non_zero_mask])) if np.any(non_zero_mask) else np.inf
                snr = 20 * np.log10(sigma_s / rmse) if rmse != 0 else np.inf
                data_range = np.max(x) - np.min(x)
                psnr = 20 * np.log10(data_range / rmse) if rmse != 0 else np.inf

                x_norm = (x - np.mean(x)) / np.std(x) if np.std(x) != 0 else x
                r_norm = (r - np.mean(r)) / np.std(r) if np.std(r) != 0 else r


                if x_norm.size > 100 and r_norm.size > 100:
                    indices = np.random.choice(len(x_norm), size=100, replace=False)
                    w_distance = wasserstein_distance(x_norm[indices], r_norm[indices])

                min_size = 7
                if x.size >= min_size**2 and r.size >= min_size**2:
                    side_length = int(np.sqrt(x.size))
                    x_2d = x[:side_length**2].reshape((side_length, side_length))
                    r_2d = r[:side_length**2].reshape((side_length, side_length))
                    data_range = np.max(x_2d) - np.min(x_2d)
                    ssim_index = ssim(x_2d, r_2d, data_range=data_range)
                else:
                    ssim_index = np.nan

                # Append metrics to rows
                rows.append({
                    'var': var,
                    'method': ds2[var].attrs.get('method', None),
                    'RMSE': rmse,
                    'SRR': srr,
                    'MAE': mae,
                    'Correlation Coefficient': correlation,
                    'Maximum Error': max_error,
                    'Mean Error': mean_error,
                    'Peak Error': peak_error,
                    'SSIM': ssim_index,
                    'RMSZ': rmsz,
                    'MRE': mre,
                    'SNR': snr,
                    'PSNR': psnr,
                    'Wasserstein Distance': w_distance
                })

    # Create a DataFrame from the collected rows
    df = pd.DataFrame(rows)
    return df
