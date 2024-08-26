import xarray as xr
from .base import CompressionMethod
import os
import zarr
class LZ4Compressor(CompressionMethod):
    def __init__(self,  compression_level=5):
        super().__init__()
        # Store parameters in a dictionary for potential extensibility and debugging
        self.para_dict = {
            #'compression_level': compression_level
        }
        self.name='LZ4'
    def compress(self, original_path, zip_dir):
        # Open the dataset
        ds = xr.open_dataset(original_path)
        compressor = zarr.LZ4() 
        # Set up compression encoding using zlib
        encoding = {var: {'compressor': compressor} for var in ds.data_vars}
        ds.to_zarr(zip_dir, mode='w', encoding=encoding)
        ds.close()  # Close the dataset to free resources
    
    def decompress(self, zip_dir, unzip_path):
        # Decompression simply involves opening the zarr format as it's inherently capable of decompressing the data
        ds = xr.open_zarr(zip_dir)
        # Optionally write back to NetCDF if needed, or directly use the data
        ds.to_netcdf(unzip_path)
        ds.close()  # Close the dataset to free resources

        #print(f"Decompression completed from {zip_dir} to {unzip_path}")
