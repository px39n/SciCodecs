from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
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
