import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
def calculate_spatial_metrics(ds1, ds2, var):
    original=ds1[var].mean(dim='time')
    mae = np.abs(ds1[var] - ds2[var]).mean(dim='time')
    max_error = np.abs(ds1[var] - ds2[var]).max(dim='time')
    correlation = xr.corr(ds1[var], ds2[var], dim='time')
    return original,mae, max_error, correlation


def spatial_accuracy_map(pds,aggregate=None):
    pds.sanity_check()
    ds1 = pds.xarray
    ds2 = pds.xarray_encoded
    var_list = pds.glob["var_list"]

    metrics = {}

    for var in var_list:
        original, mae, max_error, correlation = calculate_spatial_metrics(ds1, ds2, var)
        metrics[var] = xr.Dataset({
            'Value': original,
            'MAE': mae,
            'MaxError': max_error,
            'Correlation': correlation
        })

    # Return a dictionary with variable names as keys and corresponding metrics datasets as values
    return metrics
    
def time_series_accuracy(pds_list, legend_str):
    var_list = pds.glob["var_list"]
    global_df = pd.DataFrame()  # Initialize a global DataFrame to concatenate results

    for var in var_list:
        for pds in pds_list:
            pds.sanity_check()
            ds = pds.xarray_encoded
            legend_value = pds[var][legend_str]

            # Calculate mean and standard deviation along the specified dimensions
            mean_ds = ds.mean(dim=["XLONG", "XLAT"])
            std_ds = ds.std(dim=["XLONG", "XLAT"])

            # Convert to pandas DataFrame
            mean_df = mean_ds.to_dataframe(name='mean')
            std_df = std_ds.to_dataframe(name='std')

            # Add variable and legend information
            mean_df['variable'] = var
            mean_df[legend_str] = legend_value
            std_df['variable'] = var
            std_df[legend_str] = legend_value

            # Concatenate mean and std DataFrames
            result_df = pd.concat([mean_df, std_df], axis=1)

            # Append to global DataFrame
            global_df = pd.concat([global_df, result_df])

    return global_df

def spatial_accuracy_map(pds,aggregate=None):
    pds.sanity_check()
    ds1 = pds.xarray
    ds2 = pds.xarray_encoded
    var_list = pds.glob["var_list"]

    metrics = {}

    for var in var_list:
        original, mae, max_error, correlation = calculate_spatial_metrics(ds1, ds2, var)
        metrics[var] = xr.Dataset({
            'Value': original,
            'MAE': mae,
            'MaxError': max_error,
            'Correlation': correlation
        })

    # Return a dictionary with variable names as keys and corresponding metrics datasets as values
    return metrics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

def visualize_spatial_accuracy(pds_list, var_list=None, font_size=15):
    # Calculate spatial accuracy metrics for each pds
    plt.rcParams.update({'font.size': font_size})  # Adjust global font size as needed

    metrics_list = [spatial_accuracy_map(pds) for pds in pds_list]
    
    # Determine variables to plot
    var_list = var_list or list(metrics_list[0].keys())
    
    for var in var_list:
        num_metrics = len(metrics_list[0][var].data_vars)
        num_pds = len(pds_list)
        
        # Set up the GridSpec layout
        fig = plt.figure(figsize=(5 * num_metrics, 4 * num_pds))
        gs = gridspec.GridSpec(num_pds, num_metrics, figure=fig)
        
        for row_idx, (pds, metrics) in tqdm(enumerate(zip(pds_list, metrics_list))):
            if (metric_ds := metrics.get(var)):
                # Create the first subplot in the row to establish a shared y-axis
                axes = [fig.add_subplot(gs[row_idx, 0])]
                
                # Set initial titles and labels
                if row_idx == 0:
                    axes[0].text(-0.1, 1.05, 'XLAT', transform=axes[0].transAxes, fontsize=font_size,
                                 verticalalignment='top', horizontalalignment='center', rotation=0)
                
                # Create other subplots, sharing y-axis with the first in the row
                for i in range(1, num_metrics):
                    ax = fig.add_subplot(gs[row_idx, i], sharey=axes[0])
                    axes.append(ax)
                
                # Plot each metric
                for i, (ax, metric_name) in enumerate(zip(axes, metric_ds.data_vars)):
                    metric = metric_ds[metric_name]
                    aggregated_value = metric.mean().item()
                    std_value = metric.std().item()
                    im = metric.plot(ax=ax, cmap='jet', robust=True, add_colorbar=True)

                    # Format colorbar using ScalarFormatter
                    if im.colorbar:
                        im.colorbar.formatter = ScalarFormatter(useMathText=True)
                        im.colorbar.formatter.set_powerlimits((0, 2))
                        im.colorbar.update_ticks()
                        im.colorbar.set_label('')
                    
                    # Set titles and labels with appropriate font sizes
                    ax.set_title(f'{metric_name}' if row_idx == 0 else '', fontsize=font_size)
                    ax.set_xlabel('XLONG' if row_idx == num_pds - 1 else '', fontsize=font_size)
                    ax.xaxis.set_tick_params(labelbottom=(row_idx == num_pds - 1))
                    ax.set_ylabel(pds.workspace_name if i == 0 else '', fontsize=font_size)
                    ax.yaxis.set_tick_params(labelleft=(i == 0))
                    
                    # Annotation for average and standard deviation
                    ax.text(0.97, 0.12, f'aver.: {aggregated_value:.2e}', horizontalalignment='right',
                            verticalalignment='center', transform=ax.transAxes,
                            fontsize=font_size, color='white', bbox=dict(facecolor='black', alpha=0.6))
                    ax.text(0.97, 0.05, f'std: {std_value:.2e}', horizontalalignment='right',
                            verticalalignment='center', transform=ax.transAxes,
                            fontsize=font_size, color='white', bbox=dict(facecolor='black', alpha=0.6))
        
        # Adjust layout and display the plot
        plt.tight_layout(pad=-0.0)
        plt.show()

from matplotlib import pyplot as plt

def plot_compression_ratios(df, method_list, var_list, x, y, level,x_lim=None, y_lim=None):
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['Reds', 'Blues', 'Greens', 'Purples']  # Add more colors if needed
    markers = ['o', 's', 'D', '^']  # Add more markers if needed
    scatter_plots = []
    legend=[]
    for i, method in enumerate(method_list):
        df_method = df[df["method"] == method]
        color = colors[i % len(colors)]

        for j, var in enumerate(var_list):
            df_var = df_method[df_method["var"] == var]
            marker = markers[j % len(markers)]

            sc = ax.scatter(df_var[x], df_var[y], c=df_var[level],
                            s=50, cmap=color, edgecolor='k', marker=marker, 
                            label=f'{method} {var}')
            scatter_plots.append(sc)
            if i==1:
                legend.append(sc)
        plt.colorbar(sc, ax=ax, shrink=0.4, label=method+"_"+level)
        
    ax.legend(legend, var_list)
    ax.set_xlabel(x)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_ylabel(y)
    plt.grid()
    plt.show()

import pandas as pd
from tqdm import tqdm
import numpy as np
def time_series_accuracy(pds_list, legend_str):
    # Get the variable list from the first element in the list
    var_list = pds_list[0].glob["var_list"]
    global_df = pd.DataFrame()  # Initialize a global DataFrame to concatenate results
    ds_org=pds_list[0].xarray
    
    for pds in tqdm(pds_list):
        pds.sanity_check()  # Ensure data integrity
        ds = pds.xarray_encoded
        for var in var_list:
            ds_var = ds[var]
            legend_value = getattr(pds, var)[legend_str]
    
            # Calculate mean and standard deviation along the specified dimensions
            mean_ds = ds_var.mean(dim=["south_north", "west_east"])
            std_ds = ds_var.std(dim=["south_north", "west_east"])
            unit_df = ds_var.sel(south_north=160,west_east=160)
            mae_df = np.abs(ds_var-ds_org[var]).mean(dim=["south_north", "west_east"])
            
            # Convert to pandas DataFrame
            mean_df = mean_ds.to_dataframe(name='mean')
            std_df = std_ds.to_dataframe(name='std')
            unit_df = unit_df.to_dataframe(name='cell')  
            mae_df=mae_df.to_dataframe(name='mae')  
            
            # Concatenate mean and std DataFrames
            result_df = pd.concat([mean_df, std_df,unit_df,mae_df], axis=1)
            
            result_df['variable'] = var
            result_df[legend_str] = legend_value
            result_df['method'] = getattr(pds, var).get("method", "Unknown Method")
            # Append to global DataFrame
            global_df = pd.concat([global_df, result_df])
    return global_df
    

import matplotlib.pyplot as plt
import math
import matplotlib.dates as mdates
def visualize_time_series(global_df, y_str="mean", legend="tolerance", fontsize=15):
    # Update global font size
    plt.rcParams.update({'font.size': fontsize})
    
    # Extract unique variables from the DataFrame
    var_list = global_df["variable"].unique()
    num_vars = len(var_list)
    
    # Determine the number of rows and columns needed for the subplots
    cols = 3
    rows = math.ceil(num_vars / cols)
    
    # Create a grid for the subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows), squeeze=False)
    
    for idx, var in enumerate(var_list):
        df_var = global_df[global_df["variable"] == var]
        
        # Plotting in the appropriate subplot
        ax = axes[idx // cols, idx % cols]
        for legend_val in df_var[legend].unique():
            df_legend = df_var[df_var[legend] == legend_val]
            df_legend = df_legend.sort_index()  # Ensure the time series is sorted by the index
            ax.plot(df_legend.index, df_legend[y_str], label=f'{legend}: {legend_val}')
        
        ax.set_title(f'{var}')
        ax.set_xlabel('time')
        ax.set_ylabel(y_str)
        ax.set_yscale('log')  # Set y-axis to logarithmic scale
        ax.legend()
        
        # Adaptive adjustment of time labels
        fig.autofmt_xdate(rotation=45)
        # Set x-axis to display dates in a monthly format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
    # Remove empty subplots
    for i in range(num_vars, rows * cols):
        fig.delaxes(axes.flatten()[i])

    # Adjust layout for better readability
    plt.tight_layout()
    plt.show()
    