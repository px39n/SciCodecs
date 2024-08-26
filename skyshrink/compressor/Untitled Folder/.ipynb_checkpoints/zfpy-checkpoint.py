import xarray as xr
import zarr
from .base import CompressionMethod

 
class ZfpyCompressor(CompressionMethod):
    def __init__(self, mode=4, tolerance=-1, rate=-1, precision=-1):
        super().__init__()
        self.name = 'zfpy'
        self.para_dict = {
            'mode': mode,
            'tolerance':tolerance,
            'rate':rate,
            'precision':precision
        }
    
    def compress(self, original_path, zip_dir):
        # Open the dataset
        ds = xr.open_dataset(original_path)
        # Create a Blosc compressor with potential for lossy compression through high compression levels
        # from numcodecs.registry import register_codec
        # register_codec(FPZip)
        compressor = zarr.ZFPY(**self.para_dict)
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
         
