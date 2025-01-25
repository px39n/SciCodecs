import os
import xarray as xr
import pandas as pd

def efficiency(batch_dir):
    """
    Create a table summarizing variable attributes from xarray datasets in a directory.

    Parameters:
        batch_dir (list): List of directories containing NetCDF files.

    Returns:
        pd.DataFrame: A DataFrame with columns 'var', 'method', 'encoding (MB/s)', 
                      'compression_ratio', 'original_bits', 'dtype', 'para_dict', 
                      'workspace_name'.
    """
    rows = []

    for file_path in batch_dir:

        # Extract workspace name from file name (remove .nc extension)
        workspace_name = os.path.basename(file_path).replace('.nc', '')
        
        # Open the dataset
        ds = xr.open_dataset(file_path)

        for var in ds.data_vars:
            # Extract variable attributes
            attrs = ds[var].attrs
            
            # Append the relevant data to the rows list
            rows.append({
                'var': var,
                'method': attrs.get('method', None),
                'encoding (MB/s)': attrs.get('encoding_speed', None),
                'decoding (MB/s)': attrs.get('encoding_speed', None),
                'compression_ratio': attrs.get('compression_ratio', None),
                'original_bits': attrs.get('original_bits', None),
                'dtype': attrs.get('dtype', None),
                'para_dict': attrs.get('para_dict', None),
                'workspace_name': workspace_name
            })

    # Create a DataFrame from the collected rows
    df = pd.DataFrame(rows)
    return df

# Example usage
# batch_dir = ["path_to_file1.nc", "path_to_file2.nc"]
# df = efficiency(batch_dir)
# print(df)
