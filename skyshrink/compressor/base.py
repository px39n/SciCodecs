import os
import time
import xarray as xr
import yaml
import zarr
import torch
import numpy as np
from .utils import rgb_to_numpy, gray_to_numpy,gray_numpy_to_rgb, compress_numpy,decompress_numpy, plot_compression_results, rgb_numpy_to_gray,gray_rgb_to_numpy,save_meta_cdf,nc_to_numpy
from tqdm import tqdm
import xarray as xr
import numpy as np

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


    def compress_compressai(self, compressor, original_path, zip_dir):   

        data_path = original_path
        mode = "stack"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Create a numpy array with shape (100, 255, 255, 3)
        ds = xr.open_dataset(data_path)
        net = compressor
        meta_ds = save_meta_cdf(ds, path=os.path.join(os.path.dirname(zip_dir), "meta_ds.nc"))  # Everything same but for each variable filled with NaN values.

        for variable in tqdm(ds.data_vars, desc="Processing variables"):
            numpy_gray=nc_to_numpy(ds, variable)
            numpy_gray_rgb= gray_numpy_to_rgb(numpy_gray, mode=mode)
            compress_numpy(numpy_gray_rgb, net, os.path.join(zip_dir,f"./{variable}/compressed_bitstrings.bin"), device=device,mode=mode)



    def decompress_compressai(self, zip_dir, unzip_path):   

        import pickle
        import os
        workspace_dir=os.path.dirname(zip_dir)
        with open(os.path.join(workspace_dir, "encoding.pkl"), "rb") as f:
            net = pickle.load(f)
        result_ds=xr.load_dataset(os.path.join(workspace_dir, "meta_ds.nc"))
        for variable in tqdm(result_ds.data_vars, desc="Decompressing variables"):
            compressed_gray_rgb, mode = decompress_numpy(net, os.path.join(zip_dir,f"./{variable}/compressed_bitstrings.bin"),verbose=False)
            compressed_gray = gray_rgb_to_numpy(compressed_gray_rgb, mode=mode)
            compressed_gray=rgb_numpy_to_gray(compressed_gray,mode)
            
            # Get the original min and max values for the variable from the result dataset attributes
            original_min = result_ds[variable].attrs['min']
            original_max = result_ds[variable].attrs['max']
            
            # Unnormalize the compressed_gray data
            unnormalized_compressed_gray = compressed_gray * (original_max - original_min) + original_min
            
            # Assign the unnormalized data to the result dataset
            result_ds[variable] = (result_ds[variable].dims, unnormalized_compressed_gray)


        result_ds.to_netcdf(unzip_path)
        result_ds.close()


         

class CompressionMethod:
    def __init__(self ):
        self.name = name="BaseCompressor"
        self.para_dict=None 
    def compress(self, original_path, zip_dir):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def decompress(self, zip_dir, unzip_path):
        raise NotImplementedError("This method should be overridden by subclasses")
