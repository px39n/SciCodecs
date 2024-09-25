# 1 Packarray

pds1=Packarray(original_path=original_path,)
pds1.compress(Zlib())
pds1.decompress()


2. fixed the name

3. realize the nn

## Function


### update_parameter(update_list)
update_list: the list of flags (parameters) to be updated.

if update_list contains flag:

1. "original", 
    - update self.global["original_size", "var_list", "min", "max"] 
    - Create self.var(dict)

2. compress_pre
    - update self.workspace_name, zip_dir, unzip_path, compression_log

3. compress
    - update self.glob["compressed_size", "compression_ratio", "encoding_speed", "decoding_speed"]
    - update self.var["method", "compressed_size", "compressed_precision", "compression_ratio", "encoding_speed", "decoding_speed"]


## Class Parameters 
- `workspace_name` (str, optional): Name of the workspace. Default is None.
- `original_path` (str, optional): Path to the original NetCDF (.nc) file. Default is None.
- `log_path` (str, optional): Path to the compression log YAML file. Default is None.
- `verbose` (int, optional): Verbosity level. 0 for silent, 1 for basic info. Default is 1.

The `Packarray` class initializes with these parameters and sets up various attributes:

- `glob`: A dictionary to store global metadata and metrics.
- `encoding`: Stores the compression encoding method.
- `zip_dir`: Directory for compressed data.
- `unzip_path`: Path for decompressed data.
- `original_list`, `compress_list`, `decompress_list`, `accuracy_list`: Lists of metrics for different stages.
- `report`: A dictionary to track the status of different operations.

