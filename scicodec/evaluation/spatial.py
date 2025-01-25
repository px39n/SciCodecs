import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import xarray as xr

def plot_spatial_error(model_list, var_list, original_ds):
    

    # Set style for academic aesthetic
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
    })


    # Create figure with GridSpec
    n_vars = len(var_list)
    n_models = len(model_list)
    fig = plt.figure(figsize=(4*n_vars, 3*n_models))
    gs = GridSpec(n_models*2, n_vars, figure=fig, hspace=0.1, wspace=-0.5)

    # Plot for each model and variable
    for i, model in enumerate(model_list):
        for j, var in enumerate(var_list):
            # Get data for model and variable
            ds_model = xr.open_dataset(model)
            ds_orig = xr.open_dataset(original_ds)
            
            data_model = ds_model[var].values
            data_orig = ds_orig[var].values
            
            # Calculate relative errors across all timesteps
            # When data_orig is 0, divide by the mean instead
            data_orig_mean = np.mean(data_orig[data_orig != 0])
            rel_errors = np.abs(data_model - data_orig) / np.where(data_orig == 0, data_orig_mean, np.abs(data_orig))
            
            # Calculate 99th percentile threshold of relative error
            threshold = np.percentile(rel_errors, 97.5)
            
            # Calculate error detection rate for each grid cell
            detection_rates = np.mean(rel_errors > threshold, axis=0) * 100
            
            # Calculate mean absolute error for second row plot
            mean_abs_error = np.mean(np.abs(data_model - data_orig), axis=0)
            
            # Calculate significant pixels (detection rate > 2%)
            significant_ratio = np.mean(detection_rates > 1) * 100
            
            # Plot relative error detection rates
            ax1 = fig.add_subplot(gs[i*2, j])
            im1 = ax1.imshow(detection_rates, cmap='RdYlBu_r', aspect='equal', origin='lower')
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.ax.tick_params(labelsize=8)
            
            # Add semi-transparent text box with threshold and significant ratio
            textstr = f'τ={threshold:.2e}\nSignificant: {significant_ratio:.1f}%'
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            ax1.text(0.98, 0.98, textstr,
                    transform=ax1.transAxes,
                    horizontalalignment='right',
                    verticalalignment='top',
                    fontsize=8,
                    bbox=props)
            
            if j == 0:
                model_name = ds_model[var].attrs.get('method', None)

                ax1.set_ylabel(f'{model_name}\nDetection Rate(%)', fontsize=10)
            if i == 0:
                ax1.set_title(var, fontsize=11, pad=10)
            ax1.set_xticks([])
            ax1.set_yticks([])
                
            # Plot mean absolute error
            ax2 = fig.add_subplot(gs[i*2+1, j])
            im2 = ax2.imshow(mean_abs_error, cmap='RdYlBu_r', aspect='equal', origin='lower')
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.ax.tick_params(labelsize=8)
            
            if j == 0:
                ax2.set_ylabel('MAE', fontsize=10)
            ax2.set_xticks([])
            ax2.set_yticks([])
    plt.tight_layout()
    plt.show()