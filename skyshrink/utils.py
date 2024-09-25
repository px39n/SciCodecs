from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr
def get_directory_size(directory):
    total_size = 0
    path = Path(directory)
    
    if path.is_file():
        return path.stat().st_size  # Return size if it's a file
    
    for f in path.rglob('*'):
        if f.is_file():
            total_size += f.stat().st_size
    
    return total_size

def convert_bytes_to_mb(size_in_bytes):
    return size_in_bytes / (1024 * 1024)  # Convert bytes to MB

def convert_speed_to_mb_per_s(speed_in_bytes_per_s):
    return speed_in_bytes_per_s / (1024 * 1024)  # Convert bytes/s to MB/s

  

def format_df(df):
    # Function to format each cell to .2f if it's numeric
    def format_cell(x):
        if pd.notnull(x) and isinstance(x, (int, float)):
            return f"{x:.2f}"
        return x
    
    # Apply the formatting function to all cells in the DataFrame
    formatted_df = df.applymap(format_cell)
    
    return formatted_df

def add_pressure_coord(data):
    if isinstance(data, xr.DataArray):
        var_name = data.name if data.name else 'data'
        ds = data.to_dataset(name=var_name)
    elif isinstance(data, xr.Dataset):
        ds = data
    else:
        raise ValueError("Input must be an xarray DataArray or Dataset")

    if 'level' in ds.coords:
        return data  # If there's already a level coordinate, return the input as is
    
    # Create a new coordinate 'level' with values [1, 2]
    ds = ds.expand_dims(level=[10,50])
    
    # Add the 'level' coordinate to all variables
    for var in ds.data_vars:
        if 'level' not in ds[var].dims:
            ds[var] = ds[var].expand_dims('level')
    
    return ds[var_name] if isinstance(data, xr.DataArray) else ds


def remove_pressure_coord(data):
    if isinstance(data, xr.DataArray):
        var_name = data.name if data.name else 'data'
        ds = data.to_dataset(name=var_name)
    elif isinstance(data, xr.Dataset):
        ds = data
    else:
        raise ValueError("Input must be an xarray DataArray or Dataset")

    if 'level' not in ds.coords:
        return data  # If there's no level coordinate, return the input as is
    
    if len(ds.level) != 2:
        return data
    
    # Remove the 'level' coordinate from all variables
    for var in ds.data_vars:
        if 'level' in ds[var].dims:
            # Take the mean along the level dimension
            ds[var] = ds[var].isel(level=0)
    # Remove the 'level' coordinate from the dataset
    ds = ds.drop_vars('level')
    
    return ds[var_name] if isinstance(data, xr.DataArray) else ds

def ds_unified_coordinate(ds):
    longitude_candidates = ["lon", "longitude", "LON", "Longitude", "Xlon", "XLongitude", "XLON", "XLONG"]
    latitude_candidates = ["lat", "latitude", "LAT", "Latitude", "Xlat", "XLatitude", "XLAT", "XLATITUDE"]
    time_candidates = ["time", "Time", "Xtime", "XTIME"]

    def check_and_rename(ds, candidates, target_name):
        matches = [coord for coord in ds.coords if coord in candidates]
        if len(matches) > 1:
            raise ValueError(f"More than one coordinate found for {target_name}: {matches}")
        elif len(matches) == 1:
            if matches[0] != target_name:
                ds = ds.rename({matches[0]: target_name})
        return ds

    ds = check_and_rename(ds, longitude_candidates, "longitude")
    ds = check_and_rename(ds, latitude_candidates, "latitude")
    ds = check_and_rename(ds, time_candidates, "time")

    # Add units to longitude and latitude coordinates
    if 'longitude' in ds.coords:
        ds['longitude'].attrs['units'] = 'degrees_east'
    if 'latitude' in ds.coords:
        ds['latitude'].attrs['units'] = 'degrees_north'

    # Assign coordinates as index
    ds = ds.set_index(longitude="longitude", latitude="latitude", time="time")

    return ds
def adjust_coordinate_order(data):
    if isinstance(data, xr.DataArray):
        ds = data.to_dataset(name='data')
    elif isinstance(data, xr.Dataset):
        ds = data
    else:
        raise ValueError("Input must be an xarray DataArray or Dataset")
    if 'level' not in ds.coords:
        ds = ds.expand_dims(level=[1])
    new_order = ['time', 'level', 'latitude', 'longitude']
    for var in ds.data_vars:
        current_dims = ds[var].dims
        var_new_order = [dim for dim in new_order if dim in current_dims]
        var_new_order.extend([dim for dim in current_dims if dim not in var_new_order])
        ds[var] = ds[var].transpose(*var_new_order)
    if isinstance(data, xr.DataArray):
        return ds['data']
    else:
        return ds