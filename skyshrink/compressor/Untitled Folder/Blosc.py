import xarray as xr
import zarr
from .base import CompressionMethod

class BloscCompressor(CompressionMethod):
    def __init__(self, compression_level=9, shuffle=zarr.Blosc.SHUFFLE):
        super().__init__()
        self.name = 'Blosc'
        self.para_dict = {
            'compression_level': compression_level,
            'shuffle': shuffle
        }
    
    def compress(self, original_path, zip_dir):
        # Open the dataset
        ds = xr.open_dataset(original_path)
        # Create a Blosc compressor with potential for lossy compression through high compression levels
        compressor = zarr.Blosc(cname='zstd', clevel=self.para_dict['compression_level'], shuffle=self.para_dict['shuffle'])
        # Set up compression encoding
        encoding = {var: {'compressor': compressor} for var in ds.data_vars}
        ds.to_zarr(zip_dir, mode='w', encoding=encoding)
        ds.close()  # Close the dataset to free resources 
    
    def decompress(self, zip_dir, unzip_path):
        # Decompression simply involves opening the Zarr format as it's inherently capable of decompressing the data
        ds = xr.open_zarr(zip_dir)
        # Optionally write back to NetCDF if needed, or directly use the data
        ds.to_netcdf(unzip_path)
        ds.close()  # Close the dataset to free resources
         
