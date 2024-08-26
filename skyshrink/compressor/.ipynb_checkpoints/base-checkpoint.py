import os
import time
import xarray as xr
import yaml
import zarr

 
class ZarrCompressor():
    def __init__(self):
        pass
        
    def compress(self,compressor, original_path, zip_dir):
        
        ds = xr.open_dataset(original_path) 
        if isinstance(compressor, dict):
            encoding = compressor
        else:
            encoding = {var: {'compressor': compressor} for var in var_list}
        # Open the dataset

        ds.to_zarr(zip_dir, mode='w', encoding=encoding)
        ds.close()  # Close the dataset to free resources 
    
    def decompress(self, zip_dir, unzip_path):
        # Decompression simply involves opening the Zarr format as it's inherently capable of decompressing the data
        ds = xr.open_zarr(zip_dir)
        # Optionally write back to NetCDF if needed, or directly use the data
        ds.to_netcdf(unzip_path)
        ds.close()  # Close the dataset to free resources
         

class CompressionMethod:
    def __init__(self ):
        self.name = name="BaseCompressor"
        self.para_dict=None 
    def compress(self, original_path, zip_dir):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def decompress(self, zip_dir, unzip_path):
        raise NotImplementedError("This method should be overridden by subclasses")
