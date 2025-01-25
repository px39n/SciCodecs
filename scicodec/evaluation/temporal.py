

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import netCDF4 as nc
from scipy import fft
from tqdm import tqdm
import xarray as xr


def temporal(batch_dir, var_list, original_ds):



    var_list = var_list# LAI
    model_list = batch_dir #zfpy, mbt2018,  #

    # Create figure and gridspec for time series, periodicity, and correlation plots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, len(var_list), figure=fig)
    # Pre-calculate averages for original data
    orig_data = {}
    orig_avgs = {}
    model_data = {}
    model_avgs = {}
    model_methods = {}

    # Load original data
    with nc.Dataset(original_ds) as ds:
        for var in tqdm(var_list):
            data = ds[var][:]
            orig_data[var] = data
            orig_avgs[var] = np.mean(data, axis=(1,2))[10:-10]

    # Load model data and methods
    for model_path in model_list:
        ds_model = xr.open_dataset(model_path)
        model_data[model_path] = {}
        model_avgs[model_path] = {}
        model_methods[model_path] = {}
        
        for var in var_list:
            data = ds_model[var].values
            model_data[model_path][var] = data
            model_avgs[model_path][var] = np.mean(data, axis=(1,2))[10:-10]
            model_methods[model_path][var] = ds_model[var].attrs.get('method')
        ds_model.close()

    # Loop through variables
    for j, var in enumerate(var_list):
        # First row - Original time series plot
        ax1 = fig.add_subplot(gs[0, j])
        
        # Plot original data
        ax1.plot(orig_avgs[var], label='Original', color='black', alpha=0.8)
        
        # Plot compressed data
        for model_path in model_list:
            method_name = model_methods[model_path][var]
            ax1.plot(model_avgs[model_path][var], label=method_name, alpha=0.8)
        
        ax1.set_title(var, fontsize=11, pad=10)
        if j == 0:  # Only show legend on first subplot
            ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Time Step', fontsize=10)
        if j == 0:
            ax1.set_ylabel('Average Value', fontsize=10)
        ax1.tick_params(axis='both', labelsize=8)
        
        # Second row - Periodicity analysis
        ax2 = fig.add_subplot(gs[1, j])
        
        for model_path in model_list:
            # Calculate error
            error = np.abs(model_data[model_path][var] - orig_data[var])
            avg_error = np.mean(error, axis=(1,2))[10:-10]
            
            # Calculate FFT
            signal = avg_error - np.mean(avg_error)
            freq = fft.fftfreq(len(signal))
            fft_vals = np.abs(fft.fft(signal))
            
            method_name = model_methods[model_path][var]
            ax2.plot(freq[1:len(freq)//2], fft_vals[1:len(freq)//2], 
                    label=method_name, alpha=0.8)
        
        ax2.set_title(f'{var} Frequency Analysis', fontsize=11, pad=10)
        if j == 0:
            ax2.legend(fontsize=8)
            ax2.set_ylabel('Magnitude', fontsize=10)
        ax2.set_xlabel('Frequency', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=8)
        ax2.set_yscale('log')

        # Third row - Lag correlation analysis
        ax3 = fig.add_subplot(gs[2, j])
        
        for model_path in model_list:
            # Calculate error time series
            error = model_data[model_path][var] - orig_data[var]
            error_mean = np.mean(error, axis=(1,2))[10:-10]
            
            # Calculate autocorrelation of error series
            max_lag = 10
            lags = range(0, max_lag+1)
            autocorr = []
            
            for lag in lags:
                if lag == 0:
                    corr = 1.0
                else:
                    corr = np.corrcoef(error_mean[lag:], error_mean[:-lag])[0,1]
                autocorr.append(corr)
                
            method_name = model_methods[model_path][var]
            ax3.plot(lags, autocorr, label=method_name, marker='o', alpha=0.8)
        
        ax3.set_title(f'{var} Error Autocorrelation', fontsize=11, pad=10)
        if j == 0:
            ax3.legend(fontsize=8)
            ax3.set_ylabel('Autocorrelation', fontsize=10)
        ax3.set_xlabel('Lag', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', labelsize=8)

    plt.tight_layout()
    plt.show()
