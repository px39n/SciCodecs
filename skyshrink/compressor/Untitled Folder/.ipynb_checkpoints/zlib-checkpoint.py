import xarray as xr
from .base import CompressionMethod
import os
class ZlibCompressor(CompressionMethod):
    def __init__(self,  compression_level=5):
        super().__init__()
        # Store parameters in a dictionary for potential extensibility and debugging
        self.para_dict = {
            'compression_level': compression_level
        }
        self.name='zlib'
    def compress(self, original_path, zip_dir):
        # Open the dataset
        ds = xr.open_dataset(original_path)
        # Set up compression encoding using zlib
        encoding = {var: {'zlib': True, 'complevel': self.para_dict['compression_level']} for var in ds.variables}

                # Save the dataset with compression to the specified zip directory (zarr format)
        ds.to_netcdf(zip_dir, mode='w', encoding=encoding)
        ds.close()  # Close the dataset to free resources
        
        #print(f"Compression completed using {self.name} to {zip_dir}")
    
    def decompress(self, zip_dir, unzip_path):
        # Decompression simply involves opening the zarr format as it's inherently capable of decompressing the data
        ds = xr.open_dataset(zip_dir)
        # Optionally write back to NetCDF if needed, or directly use the data
        ds.to_netcdf(unzip_path)
        ds.close()  # Close the dataset to free resources
        
        #print(f"Decompression completed from {zip_dir} to {unzip_path}")
