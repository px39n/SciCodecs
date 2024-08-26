import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
def calculate_spatial_metrics(ds1, ds2, var):
    original=ds1[var].mean(dim='XTIME')
    mae = np.abs(ds1[var] - ds2[var]).mean(dim='XTIME')
    max_error = np.abs(ds1[var] - ds2[var]).max(dim='XTIME')
    correlation = xr.corr(ds1[var], ds2[var], dim='XTIME')
    return original,mae, max_error, correlation


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

def visualize_spatial_accuracy(pds, var_list=None):
    # Calculate spatial accuracy metrics
    print(f"============================={pds.workspace_name}=============================")
    metrics = spatial_accuracy_map(pds)
    
    # Determine variables to plot
    var_list = var_list or list(metrics.keys())

    # Plotting each variable's metrics in a row
    for var in var_list:
        if var in metrics:
            metric_ds = metrics[var]
            num_metrics = len(metric_ds.data_vars)
            fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 4))
            
            # Plot each metric
            for i, metric_name in enumerate(metric_ds.data_vars):
                ax = axes[i] if num_metrics > 1 else axes  # Handle single metric case
                metric = metric_ds[metric_name]
                aggregated_value = metric.mean().item()
                metric.plot(ax=ax, cmap='viridis', robust=True)
                ax.set_title(f'{metric_name} ({aggregated_value:.2e}) for {var}')
                ax.set_xlabel('XLONG')
                ax.set_ylabel('XLAT')
                ax.set_aspect('equal')

            # Adjust layout and show plot
            plt.tight_layout()
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
            
            # Append to global DataFrame
            global_df = pd.concat([global_df, result_df])
    return global_df

import matplotlib.pyplot as plt
import math

def visualize_time_series(global_df, y_str="mean", legend="tolerance"):
    # Extract unique variables from the DataFrame
    var_list = global_df["variable"].unique()
    num_vars = len(var_list)
    
    # Determine the number of rows and columns needed for the subplots
    cols = 3
    rows = math.ceil(num_vars / cols)
    
    # Create a grid for the subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), squeeze=False)
    
    for idx, var in enumerate(var_list):
        df_var = global_df[global_df["variable"] == var]
        
        # Plotting in the appropriate subplot
        ax = axes[idx // cols, idx % cols]
        for legend_val in df_var[legend].unique():
            df_legend = df_var[df_var[legend] == legend_val]
            df_legend = df_legend.sort_index()  # Ensure the time series is sorted by the index
            ax.plot(df_legend.index, df_legend[y_str], label=f'{legend}: {legend_val}')
        
        ax.set_title(f'Variable: {var}')
        ax.set_xlabel('Time')
        ax.set_ylabel(y_str)
        ax.legend()
    
    # Remove empty subplots
    for i in range(num_vars, rows * cols):
        fig.delaxes(axes.flatten()[i])

    # Adjust layout for better readability
    plt.tight_layout()
    plt.show()