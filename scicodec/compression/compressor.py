
import math
import struct
import numpy as np
import time
import yaml
import xarray as xr
from tqdm import tqdm
import os
import pandas as pd
def count_bitlist_bytes(bitstrings):
    """Count total bytes in a bitlist.
    
    Args:
        bitstrings: List of dictionaries containing compressed data strings
        
    Returns:
        total_bytes: Total number of bytes in all strings
    """
    import libpressio
    total_bytes = 0
    libpressio
    # For each compressed chunk
    for chunk in bitstrings:
        # For each string in the chunk
        for string_list in chunk["strings"]:
            # Add length of the binary string
            total_bytes += len(string_list[0])
            
    return total_bytes

def save_bitlist(compressed_list, image_size, file_path, continue_write=False, mode="RGB"):
    mode_write = "rb+" if continue_write else "wb"
    with open(file_path, mode_write) as f:
        if continue_write:
            f.seek(0)
            current_num_images = struct.unpack('I', f.read(4))[0]
            f.seek(0)
            f.write(struct.pack('I', current_num_images + len(compressed_list)))
            f.seek(0, 2)
        else:
            f.write(struct.pack('I', len(compressed_list)))
            mode_bytes = mode.encode('utf-8')
            f.write(struct.pack('I', len(mode_bytes)))
            f.write(mode_bytes)
            h, w = image_size
            f.write(struct.pack('II', h, w))
        
        for bitstrings in compressed_list:
            # 写入字符串数量
            num_strings = len(bitstrings["strings"])
            f.write(struct.pack('I', num_strings))
            
            # 写入每个字符串
            for string_list in bitstrings["strings"]:
                data = string_list[0]
                f.write(struct.pack('I', len(data)))
                f.write(data)
            
            # 写入形状
            f.write(struct.pack('II', *bitstrings["shape"]))

def load_bitlist(file_path):
    import torch
    loaded_list = []
    with open(file_path, "rb") as f:
        num_images = struct.unpack('I', f.read(4))[0]
        mode_length = struct.unpack('I', f.read(4))[0]
        mode = f.read(mode_length).decode('utf-8')
        image_h, image_w = struct.unpack('II', f.read(8))
        
        for _ in range(num_images):
            # 读取字符串数量
            num_strings = struct.unpack('I', f.read(4))[0]
            
            # 读取所有字符串
            strings = []
            for _ in range(num_strings):
                data_length = struct.unpack('I', f.read(4))[0]
                data = f.read(data_length)
                strings.append([data])
            
            # 读取形状
            shape = struct.unpack('II', f.read(8))
            
            loaded_list.append({
                "strings": strings,
                "shape": torch.Size(shape)
            })
    
    return loaded_list, (image_h, image_w), mode
def nc_to_numpy(ds, variable):
    """
    Convert a variable from an xarray Dataset to a numpy array.
    
    Args:
    ds (xarray.Dataset): The input xarray Dataset
    variable (str): The name of the variable to extract
    
    Returns:
    numpy.ndarray: A numpy array containing the variable data
    """
    # Extract the variable data as a numpy array
    numpy_data = ds[variable].values
    
    # Ensure the data is 3D (time, latitude, longitude)
    if len(numpy_data.shape) == 2:
        raise ValueError("Input data must be 3D (time, latitude, longitude)")
    return numpy_data

def normalize_numpy(numpy_data):
    # Normalize the data to the range [0, 1]
    min_val = np.nanmin(numpy_data)
    max_val = np.nanmax(numpy_data)
    
    if min_val != max_val:
        numpy_data = (numpy_data - min_val) / (max_val - min_val)
    else:
        # If all values are the same, set them to 0.5
        numpy_data = np.full_like(numpy_data, 0.5)
    
    # Replace NaN values with 0
    numpy_data = np.nan_to_num(numpy_data, nan=0.0)
    
    return numpy_data

def gray_numpy_to_rgb(numpy_gray, mode="repeat"):
    if mode == "repeat":
        return np.repeat(numpy_gray[:, :, :, np.newaxis], 3, axis=3)
    elif mode == "stack":
        n = numpy_gray.shape[0]
        result_shape = ((n + 2) // 3, *numpy_gray.shape[1:], 3)
        result = np.zeros(result_shape, dtype=numpy_gray.dtype)
        for i in range(n):
            result_idx = i // 3
            channel_idx = i % 3
            result[result_idx, :, :, channel_idx] = numpy_gray[i]
        return result
    elif mode == "fill_zero":
        result = np.zeros((*numpy_gray.shape, 3), dtype=numpy_gray.dtype)
        result[:, :, :, 0] = numpy_gray
        return result
    else:
        raise ValueError("Invalid mode. Choose 'repeat', 'stack', or 'repeat 0'.")

def rgb_numpy_to_gray(gray_rgb_numpy, mode, length=None):
    if mode!="RGB":
        gray_rgb_numpy = np.concatenate([gray_rgb_numpy], axis=0)
    if mode=="stack":
        # Check last three frames
        if length is not None:
            gray_rgb_numpy = gray_rgb_numpy[:length]
        else:
            for i in range(1, 4):
                if np.average(gray_rgb_numpy[-i]) < 0.01:
                    gray_rgb_numpy = gray_rgb_numpy[:-i]
                else:
                    break
    return gray_rgb_numpy

def gray_rgb_to_numpy(compressed_rgb, mode="stack"):
    if mode == "stack":
        n, h, w, _ = compressed_rgb.shape
        result = np.zeros((n * 3, h, w), dtype=compressed_rgb.dtype)
        for i in range(n):
            for j in range(3):
                result[i * 3 + j] = compressed_rgb[i, :, :, j]
        return result
    elif mode == "fill_zero":
        return compressed_rgb[:, :, :, 0]
    elif mode == "repeat":
        return compressed_rgb[:, :, :, 0]
    else:
        raise ValueError("Invalid mode. Choose 'stack', 'fill_zero', or 'repeat'.")

def relative_to_precision(relative_error):
    """
    Convert relative error to precision (number of significant bits).
    If the relative error is 0, return 0.

    Parameters:
        relative_error (float): The relative error (0 <= relative_error <= 1).
    
    Returns:
        int: The precision (number of significant bits).
    """
    if relative_error == 0:
        return 0  # Special case for zero error
    elif relative_error > 0:
        return int(-math.log2(relative_error))  # Convert relative error to precision
    else:
        raise ValueError("Relative error must be a non-negative number.")

from .available import available_compressors,supported_error_bounds,supported_compressors
import numpy as np
import time



class BatchCompressor:
    def __init__(self, method=None,**kwargs):
        """Initialize compressor with specified method and parameters.
        
        Args:
            method: Can be either:
                - String specifying single compression method (e.g. "sz3") 
                - Dict mapping variables to compression settings
                - None for decoding.
            verbose: 0 for silent, 1 for verbose
            **kwargs: Single compression parameter like:
                - abs_precision: Absolute precision (e.g. 1e-3)
                - rel_precision: Relative precision
        """
        self.method = method
        self.params = kwargs
        self.shape = None
        self.dtype = None
        self.efficiency_metrics={}
        self.verbose=1

    def encode(self, data, save_dir=None):

        from tqdm import tqdm
        for workspace_name in tqdm(self.method.keys(),desc="Batch Encoding"):
            work_dir=os.path.join(save_dir,workspace_name)
            os.makedirs(work_dir,exist_ok=True)     

            if "compressor" in self.method[workspace_name]:
                initial_method=self.method[workspace_name]["compressor"]
                initial_para_dict=self.method[workspace_name].copy()
                del initial_para_dict["compressor"]
                compressor=Compressor(initial_method,**initial_para_dict)
            else:
                initial_method=self.method[workspace_name]
                initial_para_dict=None
                compressor=Compressor(initial_method)
            compressor.verbose=0
            compressor.encode(data,save_dir=work_dir)
        return save_dir
    

    def decode(self, data, output_dir=None):
        from tqdm import tqdm
        for workspace_name in tqdm(self.method.keys(),desc="Batch Decoding"):
            work_dir=os.path.join(data,workspace_name)

            
            if "compressor" in self.method[workspace_name]:
                initial_method=self.method[workspace_name]["compressor"]
                initial_para_dict=self.method[workspace_name].copy()
                del initial_para_dict["compressor"]
                compressor=Compressor(initial_method,**initial_para_dict)
            else:
                initial_method=self.method[workspace_name]
                initial_para_dict=None
                compressor=Compressor(initial_method)
            compressor.verbose=self.verbose
            ds=compressor.decode(work_dir)
            ds.to_netcdf(os.path.join(output_dir,workspace_name+".nc"))
        return output_dir





class Compressor:
    def __init__(self, method=None,**kwargs):
        """Initialize compressor with specified method and parameters.
        
        Args:
            method: Can be either:
                - String specifying single compression method (e.g. "sz3") 
                - Dict mapping variables to compression settings
                - None for decoding.
            verbose: 0 for silent, 1 for verbose
            **kwargs: Single compression parameter like:
                - abs_precision: Absolute precision (e.g. 1e-3)
                - rel_precision: Relative precision
        """
        self.method = method
        self.params = kwargs
        self.shape = None
        self.dtype = None
        self.efficiency_metrics={}
        self.verbose=1


 
        # Convert method to lowercase if it's a string
        if isinstance(self.method, str):
            self.method = self.method.lower()
        # If method is a dict, convert all keys to lowercase
        elif isinstance(self.method, dict):
            self.method = {k: v for k, v in self.method.items()}

        # Check Module and Package
        if len(kwargs) > 2:
            raise ValueError("Only 2 parameters allowed in kwargs" + "but got " + str(kwargs))
        
        self.check_input_available(method, kwargs)

        if method is None:
            pass
        elif isinstance(method, dict):
            # Check packages for all compression methods in dict
            for m in method.values():
                if m["compressor"] in available_compressors:
                    package = available_compressors[m["compressor"]]["package"]
                    if package:
                        self.check_package_installed(package)
        elif method in available_compressors:
            package = available_compressors[method]["package"]
            if package:
                self.check_package_installed(package)

    def check_package_installed(self, package):
        """Check if required package is installed and raise helpful error if not."""
        package_info = {
            "libpressio": {
                "import_name": "libpressio",
                "install_url": "https://robertu94.github.io/libpressio/",
                "install_note": "Using spack to install is recommended over pip"
            },
            "numcodecs": {
                "import_name": "numcodecs", 
                "install_url": "https://numcodecs.readthedocs.io/",
                "install_note": "Can be installed via pip install numcodecs"
            },
            "compressai": {
                "import_name": "compressai",
                "install_url": "https://github.com/InterDigitalInc/CompressAI",
                "install_note": "Can be installed via pip install compressai [need Microsoft C++ Build Tools]"
            }
        }        
        try:
            __import__(package_info[package]["import_name"])
        except ImportError:
            raise ImportError(
                f"{package} is required for this compressor but not installed.\n"
                f"Please visit {package_info[package]['install_url']} to install.\n"
                f"{package_info[package]['install_note']}"
            )

    def find_config_with_rate(self, data, compression_rate=100, max_attempts=10,verbose=1):
        """
        Find optimal compression parameters to achieve target compression rate.
        Returns a dict mapping variables to their optimal parameters.
        """
        if isinstance(data, str):
            ds = xr.open_dataset(data)
        elif isinstance(data, xr.Dataset):
            ds = data
        else:
            raise ValueError("Data must be a path to netCDF file or xarray Dataset")

        if not isinstance(self.method, dict):
            self.method = {var: {"compressor": self.method} | self.params 
                         for var in ds.data_vars}

        optimal_configs = {}
        
        for variable in tqdm(ds.data_vars, desc="Finding optimal configs", 
                           disable=self.verbose==0):
            data = nc_to_numpy(ds, variable)
            method = self.method[variable]["compressor"]
            params = {k:v for k,v in self.method[variable].items() 
                     if k != 'compressor'}

            cr=-100
            var_attempts=max_attempts
            # Binary search for optimal parameters
            if "rel_precision" in params:
                left, right = 0, 1
                while var_attempts > 0 and np.abs(cr - compression_rate) > 1:
                    mid = (left + right) / 2
                    params["rel_precision"] = mid
                    self.encode(data, method, params)
                    cr = self.efficiency_metrics["compression_ratio"]
                    if cr > compression_rate:
                        right= mid
                    else:
                        left  = mid
                    var_attempts -= 1
                    if verbose:
                        print(f"Variable: {variable}, Compression Rate: {cr}, Relative Precision: {mid}")
                optimal_configs[variable] = {"rel_precision": mid, "compression_rate": cr}

            elif "abs_precision" in params:
                left, right = 0, 10
                while var_attempts > 0 and np.abs(cr - compression_rate) > 1:
                    mid = (left + right) / 2
                    params["abs_precision"] = mid
                    self.encode(data, method, params)
                    cr = self.efficiency_metrics["compression_ratio"]
                    if cr > compression_rate:
                        right = mid
                    else:
                        left = mid
                    var_attempts -= 1
                    if verbose:
                        print(f"Variable: {variable}, Compression Rate: {cr}, Absolute Precision: {mid}")
                optimal_configs[variable] = {"abs_precision": mid, "compression_rate": cr}

            elif "level" in params:
                left, right = 1, 6
                while var_attempts > 0 and left <= right:
                    mid = (left + right) // 2
                    params["level"] = mid
                    self.encode(data, method, params)
                    cr = self.efficiency_metrics["compression_ratio"]
                    if cr > compression_rate:
                        left = mid + 1
                    else:
                        right = mid - 1
                    var_attempts -= 1
                    if verbose:
                        print(f"Variable: {variable}, Compression Rate: {cr}, Level: {mid}")
                optimal_configs[variable] = {"level": mid, "compression_rate": cr}
            elif "bit_precision" in params:
                left, right = 0, 32
                while var_attempts > 0 and left <= right:
                    mid = (left + right) // 2
                    params["bit_precision"] = mid
                    self.encode(data, method, params)
                    cr = self.efficiency_metrics["compression_ratio"]
                    if cr > compression_rate:
                        left = mid + 1
                    else:
                        right = mid - 1
                    var_attempts -= 1
                    if verbose:
                        print(f"Variable: {variable}, Compression Rate: {cr}, Bit Precision: {mid}")
                optimal_configs[variable] = {"bit_precision": mid, "compression_rate": cr}

        return optimal_configs

    def batch_encode(self, data, save_dir=None):

        from tqdm import tqdm
        for workspace_name in tqdm(self.method.keys(),desc="Batch Encoding"):
            work_dir=os.path.join(save_dir,workspace_name)
            os.makedirs(work_dir,exist_ok=True)     

            if "compressor" in self.method[workspace_name]:
                initial_method=self.method[workspace_name]["compressor"]
                initial_para_dict=self.method[workspace_name].copy()
                del initial_para_dict["compressor"]
            else:
                initial_method=self.method[workspace_name]
                initial_para_dict=None

            compressor=Compressor(initial_method,**initial_para_dict)
            compressor.verbose=0
            compressor.encode(data,save_dir=work_dir)
        return save_dir
    

    def batch_decode(self, data, output_dir=None):
        from tqdm import tqdm
        for workspace_name in tqdm(self.method.keys(),desc="Batch Decoding"):
            work_dir=os.path.join(data,workspace_name)
            initial_method=self.method[workspace_name]["compressor"]
            initial_para_dict=self.method[workspace_name].copy()
            del initial_para_dict["compressor"]
            compressor=Compressor(initial_method,**initial_para_dict)
            compressor.verbose=self.verbose
            ds=compressor.decode(work_dir)
            ds.to_netcdf(os.path.join(output_dir,workspace_name+".nc"))
        return output_dir



    def encode(self, data,method=None, para_dict=None,save_dir=None):
        """Encode data using the configured compression method.
        
        Returns:
            compressed_data: The compressed data
            stats: Dictionary containing compression statistics including:
                - Compression ratio
                - Encoding speed (MB/s)
        """
        if method is None:
            method=self.method
        if para_dict is None:
            para_dict=self.params

        if isinstance(data, str):
            ds_path=data
            mode = "xarray"
            ds = xr.open_dataset(data)
            
        elif isinstance(data, xr.Dataset):
            mode = "xarray"
            ds = data
            if save_dir is None:
                raise ValueError("save_dir is required for encoding")
            else:
                ds_path=save_dir

        else:
            mode = "numpy"

            
        if mode == "numpy":
            package = available_compressors[method]["package"]
            
            start_time = time.time()
            data_size = data.nbytes / (1024 * 1024) # Size in MB
            
            if package == "libpressio":
                compressed = self.libpressio_encode_array(data, method, para_dict)
                
            elif package == "numcodecs":
                compressed = self.numcodecs_encode_array(data, method, para_dict)

            elif package == "compressai":
                compressed = self.compressai_encode_array(data, method, para_dict)
            else:
                raise ValueError(f"Unsupported compression method: {method}")
                
            end_time = time.time()
            self.efficiency_metrics["encoding_speed"]=data_size / (end_time - start_time)


            if isinstance(compressed, list):
                self.efficiency_metrics["compression_ratio"]=data.nbytes/count_bitlist_bytes(compressed)
            else:
                self.efficiency_metrics["compression_ratio"]=data.nbytes/len(compressed)
            self.efficiency_metrics["original_bits"]=data.nbytes

            return compressed
        if mode == "xarray":

            if not os.path.exists(ds_path):
                os.makedirs(ds_path, exist_ok=True)

            if not isinstance(self.method, dict):
                self.method={variable:{"compressor":self.method} | self.params for variable in ds.data_vars}
            
            import yaml
            with open(os.path.join(ds_path, "encoding_list.yaml"), "w") as f:
                yaml.dump(self.method, f)
            meta_dict={}
            for variable in tqdm(ds.data_vars, desc="Processing variables",disable=self.verbose==0):

                data=nc_to_numpy(ds, variable)
              
                method=self.method[variable]["compressor"]
                para_dict = {k:v for k,v in self.method[variable].items() if k != 'compressor'}
           
                compressed=self.encode(data,method, para_dict)
                # Save compressed data
                var_dir = os.path.join(ds_path, variable)
                os.makedirs(var_dir, exist_ok=True)

                if isinstance(compressed, list):
                    save_bitlist(compressed, (data.shape[0], data.shape[1]), os.path.join(var_dir, "compressed.bin"))
                else:
                    with open(os.path.join(var_dir, "compressed.bin"), "wb") as f:
                        f.write(compressed)

                meta_dict[variable]={}
                meta_dict[variable]["min"]=np.nanmin(data)
                meta_dict[variable]["max"]=np.nanmax(data)
                meta_dict[variable]["h"]=data.shape[1]
                meta_dict[variable]["w"]=data.shape[2]
                meta_dict[variable]["l"]=data.shape[0]
                meta_dict[variable]["dtype"]=str(data.dtype)
                meta_dict[variable]["encoding_speed"]=self.efficiency_metrics["encoding_speed"]
                meta_dict[variable]["compression_ratio"]=self.efficiency_metrics["compression_ratio"]
                meta_dict[variable]["original_bits"]=self.efficiency_metrics["original_bits"]
                meta_dict[variable]["method"]=method
                meta_dict[variable]["para_dict"]=str(para_dict)
                #encode_dict = ast.literal_eval(encode_dict_str)


            self.save_meta_cdf(ds, path=os.path.join(ds_path, "meta_ds.nc"),meta_dict=meta_dict)

            # Convert meta_dict to DataFrame for efficiency metrics
            efficiency_df = pd.DataFrame()
            for var, metrics in meta_dict.items():
                df_row = {
                    'variable': var,
                    'method': metrics['method'],
                    'encoding_speed': metrics['encoding_speed'],
                   # 'decoding_speed': metrics.get('decoding_speed', None)
                    'compression_ratio': metrics['compression_ratio'],
                    'original_bits': metrics['original_bits'],
                    'para_dict': str(metrics['para_dict'])
                }
                efficiency_df = pd.concat([efficiency_df, pd.DataFrame([df_row])], ignore_index=True)
            
            # Save efficiency metrics DataFrame
            efficiency_path = os.path.join(ds_path, "efficiency_metrics.csv")
            efficiency_df.to_csv(efficiency_path, index=False)
            self.efficiency_metrics=efficiency_df
            
            
            return ds_path

    def decode(self, data,method=None, para_dict=None):
        
        if method is None:
            method=self.method
        if para_dict is None:
            para_dict=self.params

        if isinstance(data, str):
            mode="xarray"
        else:
            mode="numpy"
        
        if mode=="numpy":
            package=available_compressors[method]["package"]

            start_time = time.time()
            if package == "libpressio":
                decoded= self.libpressio_decode_array(data, method, para_dict)
            
            elif package == "numcodecs":
                decoded= self.numcodecs_decode_array(data, method, para_dict)

            elif package == "compressai":
                decoded= self.compressai_decode_array(data, method, para_dict)
            
            else:
                raise ValueError(f"Unsupported compression method: {method}")

            end_time = time.time()
         
            self.efficiency_metrics["decoding_speed"]=len(data) / (1024 * 1024)  / (end_time - start_time)
            return decoded
        
        if mode=="xarray":
            data_path=data
            meta_ds=xr.open_dataset(os.path.join(data_path, "meta_ds.nc"))

            self.efficiency_metrics={}
            efficiency_df = pd.DataFrame()

            for variable in meta_ds.data_vars:
                self.original_min=meta_ds[variable].attrs["min"]
                self.original_max=meta_ds[variable].attrs["max"]
                self.shape=(meta_ds[variable].attrs["l"], meta_ds[variable].attrs["h"],meta_ds[variable].attrs["w"])
                self.dtype=np.dtype(meta_ds[variable].attrs["dtype"])
                #self.method=meta_ds[variable].attrs["method"]
                #self.para_dict=meta_ds[variable].attrs["para_dict"]

                if available_compressors[meta_ds[variable].attrs["method"]]["package"] == "compressai":
                    compressed_data,_,_=load_bitlist(os.path.join(data_path, variable, "compressed.bin"))
                else:
                    with open(os.path.join(data_path, variable, "compressed.bin"), "rb") as f:
                        compressed_data = f.read()
                        compressed_data = np.frombuffer(compressed_data, dtype=np.uint8)
                
                import ast
                meta_ds[variable].data=self.decode(compressed_data, meta_ds[variable].attrs["method"],ast.literal_eval(meta_ds[variable].attrs["para_dict"]))
                
                meta_ds[variable].attrs['decoding_speed']=self.efficiency_metrics["decoding_speed"]
                # Convert meta_dict to DataFrame for efficiency metrics

                df_row = {
                    'variable': variable,
                    'method': meta_ds[variable].attrs["method"],
                    'encoding_speed': meta_ds[variable].attrs["encoding_speed"],
                    'decoding_speed': meta_ds[variable].attrs["decoding_speed"],  # May not exist yet
                    'compression_ratio': meta_ds[variable].attrs["compression_ratio"],
                    'original_bits': meta_ds[variable].attrs["original_bits"],
                    }
                efficiency_df = pd.concat([efficiency_df, pd.DataFrame([df_row])], ignore_index=True)
                

            self.efficiency_metrics=efficiency_df
            decoded_ds=meta_ds.copy(deep=True)
            meta_ds.close()
            return decoded_ds
        
    def aicompressor_config_translator(self, method, para_dict):
        """Translate numcodecs parameters to compressor parameters."""
    
        from compressai.zoo import bmshj2018_factorized as bmshj2018_factorized_base
        from compressai.zoo import mbt2018 as mbt2018_base
        from compressai.zoo import cheng2020_anchor as cheng2020_anchor_base
     
        method_dict = {
            "cheng2020_anchor": cheng2020_anchor_base,
            "mbt2018": mbt2018_base,
            "bmshj2018_factorized": bmshj2018_factorized_base,
           # "zfp": ZFPY
        }
        
        return method_dict[method](**{"quality":para_dict.get("level",1),"pretrained":True, "device":para_dict.get("device","cpu")})
    
    def compressai_encode_array(self, data, method, para_dict):
        """Encode data using numcodecs."""

        import torch
        mode = "stack"
        net = self.aicompressor_config_translator(method,para_dict)
        device = 'cuda' if next(net.parameters()).device.type == 'cuda' else 'cpu'

        data=normalize_numpy(data)

        self.original_min=np.nanmin(data)
        self.original_max=np.nanmax(data)
        self.shape = data.shape

        numpy_gray_rgb = gray_numpy_to_rgb(data, mode=mode)

        if self.method!="bmshj2018_factorized":
            h, w = numpy_gray_rgb.shape[1:3]
            pad_h = (64 - h % 64) if h % 64 != 0 else 0
            pad_w = (64 - w % 64) if w % 64 != 0 else 0
            if pad_h > 0 or pad_w > 0:
                padded = np.pad(numpy_gray_rgb, ((0,0), (0,pad_h), (0,pad_w), (0,0)), mode='edge')
                numpy_gray_rgb = padded

        all_bitstrings = []
        
        # Compression phase
        for i in range(numpy_gray_rgb.shape[0]):
            # Convert single image numpy array to tensor
            x = torch.from_numpy(numpy_gray_rgb[i]).permute(2, 0, 1).float().unsqueeze(0).to(device)
            bitstrings = net.compress(x)
            all_bitstrings.append(bitstrings)
        

        return all_bitstrings

    def compressai_decode_array(self, data, method, para_dict):
        import torch
        compressed_results = []
        
        (length,image_h, image_w)=self.shape
        mode = "stack"
        all_bitstrings=data
        # Decompression phase

        net = self.aicompressor_config_translator(method,para_dict)
        
        for i in tqdm(range(len(all_bitstrings)),desc="Decompressing",disable=True):
            with torch.no_grad():
                out = net.decompress(all_bitstrings[i]["strings"], all_bitstrings[i]["shape"])
            out_net = out["x_hat"].clamp_(0, 1)
            out_net_cropped = out_net[:, :, :image_h, :image_w]
            compressed = out_net_cropped.squeeze().cpu().permute(1, 2, 0).numpy()
            compressed_results.append(compressed)

        compressed_gray_rgb=np.array(compressed_results)
        compressed_gray_rgb = compressed_gray_rgb[:, :image_h, :image_w, :]
        compressed_gray = gray_rgb_to_numpy(compressed_gray_rgb, mode=mode)
        compressed_gray = rgb_numpy_to_gray(compressed_gray, mode, length=length)
        
        unnormalized_compressed_gray = compressed_gray * (self.original_max - self.original_min) + self.original_min

        return unnormalized_compressed_gray

    def numcodec_config_translator(self, method, para_dict):
        """Translate numcodecs parameters to compressor parameters."""
        from numcodecs import Zlib,LZ4,Blosc,Zstd,GZip# ,ZFPY
        from .Fpzip import FPZip
        translator_dict = {
            "blosc": {"cname": "zlib", "clevel": para_dict.get("level", 1)},
            "lz4": {"acceleration": para_dict.get("level", 1)},
            "fpzip": {"precision": relative_to_precision(para_dict.get("rel_precision", para_dict.get("bit_precision", 10)))}
        }
        parameters = {}
        for param, value in para_dict.items():
            if method in translator_dict:
                parameters = translator_dict[method]
            else:
                parameters[param] = value

        method_dict = {
            "zlib": Zlib,
            "lz4": LZ4,
            "blosc": Blosc,
            "gzip": GZip,
            "zstd": Zstd,
            "fpzip": FPZip,
           # "zfp": ZFPY
        }
        return method_dict[method], parameters

    def numcodecs_encode_array(self, data, method, para_dict):
        """Encode data using numcodecs."""
        compressor_class, parameters = self.numcodec_config_translator(method, para_dict)

        compressor = compressor_class(**parameters)
        self.shape = data.shape
        self.dtype = data.dtype
        return compressor.encode(data)
    
    def numcodecs_decode_array(self, data, method, para_dict):
        """Decode data using numcodecs."""
        compressor_class, parameters = self.numcodec_config_translator(method, para_dict)
        
        compressor = compressor_class(**parameters)

        res=compressor.decode(data,out=np.empty(self.shape,dtype=self.dtype))
        if res.shape != self.shape:
            res = res.reshape(self.shape)

        return res
        
    def libpressio_config_translator(self, method,para_dict):
        """Translate libpressio parameters to compressor parameters."""
        translator_dict = {
            "abs_precision": ["sz3:abs_error_bound", "zfp:abs_error_bound", "mgard:tolerance"],
            "mode": ["zfp:mode", "mgard:error_bound_type"],
            "fixed_ratio": ["zfp:rate", "mgard:fixed_ratio"],
            "bit_precision": ["zfp:precision"],

        }

        config = {
            "compressor_id": method,
            "compressor_config": {}
        }

        for param, value in para_dict.items():
            if param in translator_dict:
                # Find matching translated parameter name for this method
                translated_names = translator_dict[param]
                for name in translated_names:
                    if method.lower() in name.lower():
                        config["compressor_config"][name] = value
                        break
        return config
    

    def libpressio_encode_array(self,data, method,para_dict):
        """Encode data using libpressio."""
        import libpressio
        config=self.libpressio_config_translator(method,para_dict)
        compressor = libpressio.PressioCompressor.from_config(config)
        self.shape = data.shape
        self.dtype = data.dtype
        return compressor.encode(data)

    def libpressio_decode_array(self, data,method,para_dict):
        """Decode data using libpressio."""
        import libpressio
        config=self.libpressio_config_translator(method,para_dict)
        compressor = libpressio.PressioCompressor.from_config(config)
        return compressor.decode(data,out=np.empty(self.shape,dtype=self.dtype))

    def check_input_available(self, method, params):
        """Check if the input is valid."""
        if method is None:
            pass
        elif isinstance(method, dict):
            # Check compression settings for each variable
            for var, settings in method.items():
                if 'compressor' not in settings:
                    raise ValueError(f"Missing compressor specification for variable {var}")
                compressor = settings['compressor'].lower()
                if compressor not in supported_compressors:
                    raise ValueError(f"Unsupported compressor for {var}: {compressor}. Supported compressors are: {supported_compressors}")
                
                # Check error bound parameters
                error_bounds = [key for key in settings.keys() if key != 'compressor']
                for bound in error_bounds:
                    if bound not in supported_error_bounds[compressor]:
                        raise ValueError(f"Invalid error bound '{bound}' for {var}. Supported error bounds for {compressor} are: {supported_error_bounds[compressor]}")
        else:
            # Single compression method
            method = method.lower()
            if method not in supported_compressors:
                raise ValueError(f"Unsupported compressor: {method}. Supported compressors are: {supported_compressors}")
            
            # Check if any of the provided parameters are valid error bounds
            for param in params.keys():
                if param not in supported_error_bounds[method]:
                    raise ValueError(f"Invalid error bound '{param}'. Supported error bounds for {method} are: {supported_error_bounds[method]}")


    def save_meta_cdf(self,ds, path="meta_ds.nc",meta_dict=None):
        """
        Create a new xarray Dataset with the same structure as the input,
        but without the actual data for each variable. Add min and max attributes to each variable.
        
        Args:
        ds (xarray.Dataset): The input dataset
        path (str): Path to save the metadata netCDF file
        
        Returns:
        xarray.Dataset: A new dataset with the same structure but no data, and min/max attributes
        """
        meta_ds = ds.copy(deep=True)
 
        # Remove data from all variables and add min/max attributes
        for var in meta_ds.data_vars:
            shape = meta_ds[var].shape
            meta_ds[var].data = np.full(shape, np.nan)
            
            (h,w,l)=shape

            if meta_dict is not None:
            # Add min and max attributes
                meta_ds[var].attrs['min']=meta_dict[var]["min"]
                meta_ds[var].attrs['max']=meta_dict[var]["max"]
                meta_ds[var].attrs['h']=meta_dict[var]["h"]
                meta_ds[var].attrs['w']=meta_dict[var]["w"]
                meta_ds[var].attrs['l']=meta_dict[var]["l"]
                meta_ds[var].attrs['dtype']=meta_dict[var]["dtype"]
                meta_ds[var].attrs['method']=meta_dict[var]["method"]
                meta_ds[var].attrs['para_dict']=meta_dict[var]["para_dict"]
                meta_ds[var].attrs['encoding_speed']=meta_dict[var]["encoding_speed"]
                meta_ds[var].attrs['compression_ratio']=meta_dict[var]["compression_ratio"]
                meta_ds[var].attrs['original_bits']=meta_dict[var]["original_bits"]
        meta_ds.to_netcdf(path)
        meta_ds.close()
        return meta_ds
