import os
import time
import xarray as xr
import yaml
import zarr
from ..compressor.base import ZarrCompressor
from ..utils import get_directory_size
class Packarray:
    def __init__(self, workspace_name=None, original_path=None, log_path=None,verbose=1):
        
        self.glob={
        # Original    meta    level: original_meta
        "original_size": None,
        "var_list": [],
        "max": None,
        "min": None,
        # Compress    efficiency  level: compress_efficiency global
        "encoding_time": None,
        "compressed_size":None,
        "compression_ratio":None,
        "encoding_speed":None,
        # Decompress    efficiency level: decmpress_efficiency global
        "decompressed_size":None,
        "decoding_time":None,
        # accuracy
        # RMSE
        # MAE  
        }
        self.verbose=verbose
        self.encoding=None        
        self.workspace_name=workspace_name        
        self.report=None        
        self.unzip_path = None
        self.zip_dir = None        
        self.original_path = original_path
        self.compression_log = log_path 
        self.original_list=["original_size","var_list"]
        self.compress_list = ["encoding_speed", "compression_ratio", "compressed_precision", "compressed_size", "encoding_time"]
        self.decompress_list = ["decoding_speed","decoding_time"]
        self.accuracy_list = ["RMSE","SRR","MAE","Correlation Coefficient","Mean Error","Maximum Error","SSIM"]
        
        self.report={"original":False,"compress":False,"original_evaluate":False,"compress_evaluate":False,"decompress_evaluate":False}
        
        if log_path:
            self.load_yaml(log_path)

        self.sanity_check(["original"])
        self.update_parameter(["original"])

    def compress(self, encoding, zip_dir=None):

        self.zip_dir=zip_dir
        self.encoding=encoding
        self.detect_method()
        self.sanity_check(["original"])
        self.update_parameter(["compress_pre"])
        
        if not self.workspace_name:
            detect_workspace()
        if self.verbose:
            if os.path.exists(self.zip_dir):         
                print(f"[Skyshrink:{self.workspace_name}] Start Overwrite to ...{self.zip_dir[-30:]}")
            else:
                print(f"[Skyshrink:{self.workspace_name}] Start Compress to ...{self.zip_dir[-30:]}")

        start_time = time.time()
        ds = xr.open_dataset(self.original_path)
        
        if not isinstance (self.encoding, dict):
            ds.to_zarr(self.zip_dir, mode='w', encoding={var: {'compressor': self.encoding} for var in self.glob["var_list"]})
        else:
            ds.to_zarr(self.zip_dir, mode='w', encoding=self.encoding)
        
        ds.close()  # Close the dataset to free resources 
        self.glob["encoding_time"] = time.time() - start_time

        self.sanity_check()
        self.update_parameter(["compress"])
        self.update_parameter(["parameter"])
        
        self.save_yaml(self.compression_log)
 
    
    def decompress(self, unzip_path=None):
        if unzip_path:
            self.unzip_path=unzip_path
        self.sanity_check(["compress"]) 


        if not self.workspace_name:
            detect_workspace()  
        if self.verbose:    
            if os.path.exists(self.unzip_path): 
                print(f"[Skyshrink:{self.workspace_name}] Unzipping overwrite to ...{self.unzip_path[-30:]}")
            else:
                print(f"[Skyshrink:{self.workspace_name}] Unzipping to ...{self.unzip_path[-30:]}")
            

        start_time = time.time()
        z_c=ZarrCompressor()
        z_c.decompress(self.zip_dir, unzip_path=self.unzip_path)
        self.glob["decoding_time"] = time.time() - start_time
        self.update_parameter(["decompress"])
        self.save_yaml(self.compression_log) 

    
    def load_yaml(self, yaml_file):
        with open(yaml_file, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            # Only update attributes if they exist and are not None
            for key, value in data.items():
                if value is not None:
                    setattr(self, key, value)
                    
    def save_yaml(self, yaml_file):
        # Get all attributes of the class instance
        all_attributes = vars(self)
        # Prepare data dictionary for YAML serialization
        data = {}
        for attr, value in all_attributes.items():
            if attr !="encoding":
                data[attr] = value
        
        # Save data to YAML file
        with open(yaml_file, 'w') as file:
            yaml.dump(data, file)
            
    def __len__(self):
        ds = xr.open_dataset(self.original_path if self.original_path else self.unzip_path)
        return ds.dims

    def __repr__(self):
        ds = xr.open_dataset(self.original_path if self.original_path else self.unzip_path)
        return ds.__repr__()

    def ds_sel(self, **kwargs):
        ds = xr.open_dataset(self.original_path if self.original_path else self.unzip_path)
        return ds.sel(**kwargs)
    @property
    def xarray(self):
        return xr.open_dataset(self.original_path)
    @property    
    def xarray_encoded(self):
        return xr.open_dataset(self.unzip_path)
    

    def sanity_check(self,check_list=[]):

        # original Part
        # if not self.original_path.endswith('.nc'):
        #     raise ValueError("Only NetCDF (.nc) files are supported.")
        if self.original_path and os.path.exists(self.original_path):
            self.report["original"]=True
            
        # compress Part
        if self.zip_dir and self.compression_log and self.glob["var_list"]:
            self.report["compress"]=True

        # decompress Part
        if self.unzip_path and os.path.exists(self.unzip_path):
            self.report["decompress"]=True

        
        if self.glob:
            compress_list = self.original_list
            if all(self.glob.get(key) is not None for key in compress_list):
                self.report["original_evaluate"]=True

        # Initialize compress_evaluate to True
        flag = True
        for var in self.glob["var_list"]:
            if getattr(self, var):
                # Condition 2: Ensure no None values in each attribute for the current var
                compress_list = self.compress_list  
                if not all(getattr(self, var).get(key) is not None for key in compress_list):
                    flag = False
                    break
            else:
                flag = False
        self.report["compress_evaluate"] = flag

        # Initialize compress_evaluate to True
        flag = True
        for var in self.glob["var_list"]:
            if getattr(self, var):
                # Condition 2: Ensure no None values in each attribute for the current var
                compress_list = self.decompress_list  
                if not all(getattr(self, var).get(key) is not None for key in compress_list):
                    flag = False
                    break
            else:
                flag = False
        self.report["decompress_evaluate"] = flag

        # Initialize compress_evaluate to True
        flag = True
        for var in self.glob["var_list"]:
            if getattr(self, var):
                # Condition 2: Ensure no None values in each attribute for the current var
                compress_list = self.accuracy_list  
                if not all(getattr(self, var).get(key) is not None for key in compress_list):
                    flag = False
                    break
            else:
                flag = False
        self.report["accuracy"] = flag


        
        for check in check_list:
            if not self.report[check]:
                raise ValueError(f"{check} does not pass sanity check")


    def detect_method(self):
        if isinstance(self.encoding, dict):
            unique_codec_ids = list({value['compressor'].codec_id for value in self.encoding.values()})
            # Convert the list to a string with elements separated by commas
            if len(unique_codec_ids)>=1:
                unique_codec_ids_str = '_'.join(map(str, unique_codec_ids[:3]))
                self.glob["method"]="hybrid_"+unique_codec_ids_str
            else:
                self.glob["method"]=unique_codec_ids[0]
        else:
            self.glob["method"]=self.encoding.codec_id
        return self.glob["method"]
    
    def detect_workspace(self):
        self.workspace_name=self.glob["method"]
        return self.workspace_name

    
    def update_parameter(self, update_list, local=False):
        if "original" in update_list:
            if not self.report["original"]:
                raise ValueError("Update Failed")
            self.glob["original_size"] = get_directory_size(self.original_path)
            ds = xr.open_dataset(self.original_path)
            # Check if 'Time' is in the dataset dimensions and 'XTIME' is not already a coordinate
            if 'Time' in ds.dims and 'XTIME' not in ds.coords:
                ds = ds.assign_coords(XTIME=ds['Time'])

            self.glob["var_list"]=list(ds.data_vars)
            for var in self.glob["var_list"]:
                if not hasattr(self, var):
                    setattr(self, var, {})
                getattr(self, var)["original_size"] = ds[var].nbytes
                getattr(self, var)["min"] = ds[var].min().item()
                getattr(self, var)["max"] = ds[var].max().item()
                
            ds.close()  # Close the dataset to free resources  
            
        if "compress_pre" in update_list:
            if not self.report["original"]:
                raise ValueError("Update Failed")
            if self.zip_dir:
                base_path, _ = os.path.splitext(self.zip_dir)
                self.compression_log = base_path + '_log.yaml'
                
            else:
                # Using workspace
                if not self.workspace_name: 
                    self.detect_workspace()
                        
                temp_dir, filename = os.path.split(self.original_path)
                filename, ext = os.path.splitext(filename)
                workspace_dir = os.path.join(temp_dir, self.workspace_name)
                
                if not os.path.exists(workspace_dir):
                    os.makedirs(workspace_dir)
                    
                self.zip_dir = os.path.join(workspace_dir, filename + '_zipped')
                self.unzip_path = os.path.join(workspace_dir, filename + '_unzipped.nc')
                self.compression_log = os.path.join(workspace_dir, filename + '_log.yaml')

            
        if "compress" in update_list:
            if not self.report["original"] or not self.report["compress"]:
                raise ValueError("Update Failed")
                
            self.glob["compressed_size"] = get_directory_size(self.zip_dir)
            self.glob["compression_ratio"] = self.glob["compressed_size"] / self.glob["original_size"] 
            self.glob["encoding_speed"] =self.glob["original_size"] /self.glob["encoding_time"]
            
            ds = zarr.open(self.zip_dir , mode='r') 
            for var in self.glob["var_list"]:
                zarr_array = ds[var]
                if not hasattr(self, var):
                    setattr(self, var, {})
                compressed_size = get_directory_size(os.path.join(self.zip_dir, var))
                if not self.workspace_name:
                    detect_workspace()
                if isinstance(self.encoding, dict):
                    getattr(self, var)["method"] = self.encoding["var"]["compressor"].codec_id    
                else:
                    getattr(self, var)["method"] = self.glob["method"]
                getattr(self, var)["compressed_size"] = compressed_size
                getattr(self, var)["compressed_precision"] = zarr_array.dtype 
                getattr(self, var)["compression_ratio"] = compressed_size / getattr(self, var)["original_size"] if compressed_size > 0 else None
                getattr(self, var)["encoding_time"] = self.glob["encoding_time"]
                getattr(self, var)["encoding_speed"] = self.glob["encoding_speed"]
 
            
        if "decompress" in update_list:
            if not self.report["original"] or not self.report["compress"]:
                raise ValueError("Update Failed")
            
            self.glob["decompressed_size"] = get_directory_size(self.unzip_path) 
            self.glob["decoding_speed"] =self.glob["decompressed_size"] /self.glob["decoding_time"]
            
            ds = xr.open_dataset(self.unzip_path)
            for var in self.glob["var_list"]:
                if not hasattr(self, var):
                    setattr(self, var, {})
                #getattr(self, var)["unzipped_size"] = ds[var].nbytes
                getattr(self, var)["decoding_speed"] = self.glob["decoding_speed"]
                getattr(self, var)["decoding_time"] = self.glob["decoding_time"]
        
        if "parameter" in update_list:
            if not self.report["original"] or not self.report["compress"]:
                raise ValueError("Update Failed")
                
            #for each key, value in self.encoding:\
            for var in self.glob["var_list"]:
                
                if isinstance(self.encoding, dict):
                    compressor=self.encoding[var]["compressor"]
                else:
                    compressor=self.encoding
                    
                all_attributes = vars(compressor)
                for attr, value in all_attributes.items(): 
                    if attr=="clevel":
                        getattr(self, var)["level"]=value
                    else:
                        getattr(self, var)[attr]=value


        if "accuracy" in update_list:
            if not self.report["decompress"] or not self.report["original"]:
                raise ValueError("Update Failed")
            from skimage.metrics import structural_similarity as ssim
            import numpy as np
        
            # Load datasets
            ds1 = xr.open_dataset(self.original_path)
            ds2 = xr.open_dataset(self.unzip_path)
            
            # Initialize dictionary to store metrics for each variable
            for var in ds1.data_vars:
                if var in ds2.data_vars:
                    x = ds1[var].values.flatten()
                    r = ds2[var].values.flatten()
                    
                    # Calculate metrics
                    rmse = np.sqrt(np.mean((x - r) ** 2))
                    srr = np.log2(np.std(x) / np.std(r)) if np.std(r) != 0 else np.inf
                    mae = np.mean(np.abs(x - r))
                    correlation = np.corrcoef(x, r)[0, 1] if len(x) > 1 else np.nan
                    max_error = np.max(np.abs(x - r))
                    mean_error = np.mean(x - r)
                    peak_error = np.log2((np.max(x) - np.min(x)) / (2 * np.max(np.abs(r)))) if np.max(np.abs(r)) != 0 else np.inf
                    
                    # SSIM requires data range for floating point images
                    min_size = 7  # Minimum size for SSIM default window
                    if x.size >= min_size**2 and r.size >= min_size**2:
                        # Reshape into 2D if possible
                        side_length = int(np.sqrt(x.size))  # Find a suitable side length that is at least 7
                        x_2d = x[:side_length**2].reshape((side_length, side_length))
                        r_2d = r[:side_length**2].reshape((side_length, side_length))
                        data_range = np.max(x_2d) - np.min(x_2d)  # Determine the data range for SSIM
                        ssim_index = ssim(x_2d, r_2d, data_range=data_range)
                    else:
                        ssim_index = np.nan
                    
                    # Prepare data for DataFrame
                    getattr(self, var)["RMSE"]=rmse
                    getattr(self, var)["SRR"]=srr
                    getattr(self, var)["MAE"]=mae
                    getattr(self, var)["Correlation Coefficient"]=correlation
                    getattr(self, var)["Maximum Error"]=max_error
                    getattr(self, var)["Mean Error"]=mean_error 
                    getattr(self, var)["Peak Error"]=peak_error
                    getattr(self, var)["SSIM"]=ssim_index

            # Close datasets
            ds1.close()
            ds2.close()


 
