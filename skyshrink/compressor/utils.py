from PIL import Image
import os
import torch
import os
import struct
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
def check_memory(numpy_array):
    """
    Check the memory usage of a given numpy array.
    
    Args:
    numpy_array (np.ndarray): The numpy array to check.
    
    Returns:
    dict: A dictionary containing shape, size, and memory usage information.
    """
    length = len(numpy_array)
    size = numpy_array.size
    memory_usage = size * np.dtype(numpy_array.dtype).itemsize / (1024 * 1024)  # in MB
    
    return length, size, memory_usage


def find_optimal_batch(memory_info):
    length, size, memory_usage = memory_info
    target_batch_memory = 700  # MB
    
    # Calculate initial batch size
    initial_batch_size = int((target_batch_memory / memory_usage) * length)
    
    # Adjust batch size to be a multiple of 3
    batch_size = initial_batch_size - (initial_batch_size % 3)
    
    # Ensure batch size is at least 3
    batch_size = max(batch_size, 3)
    
    # Calculate actual memory usage for the adjusted batch size
    actual_batch_memory = (batch_size / length) * memory_usage
    
    # Calculate batch_num and ensure it's an integer
    batch_num = math.ceil(length / batch_size)
    
    # Recalculate batch_size to ensure it divides length evenly
    batch_size = math.ceil(length / batch_num)
    
    # Adjust batch_size to be a multiple of 3
    batch_size = batch_size + (3 - (batch_size % 3)) if batch_size % 3 != 0 else batch_size
    
    # Recalculate batch_num with the adjusted batch_size
    batch_num = math.ceil(length / batch_size)
    
    # Calculate remainder
    remainder = (batch_num * batch_size) - length
    
    return batch_size, batch_num, remainder,actual_batch_memory

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


def rgb_numpy_to_gray(gray_rgb_numpy, mode):
    if mode!="RGB":
        gray_rgb_numpy = np.concatenate([gray_rgb_numpy], axis=0)
    if mode=="stack":
        # Check last three frames
        for i in range(1, 4):
            if np.average(gray_rgb_numpy[-i]) < 0.01:
                gray_rgb_numpy = gray_rgb_numpy[:-i]
            else:
                break
    return gray_rgb_numpy
def rgb_to_numpy(image_dirs):
    images = []
    for image_dir in image_dirs:
        img = Image.open(image_dir).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        images.append(img_np)
    return np.array(images)

def gray_to_numpy(image_dirs):
    images = []
    for image_dir in image_dirs:
        img = Image.open(image_dir).convert('L')
        img_np = np.array(img).astype(np.float32) / 255.0
        images.append(img_np)
    return np.array(images)


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



# def save_bitlist(compressed_list, image_size, file_path,mode="stack"):
#     # Length of bitstring_list(4), image size(8), 
#     # iteration ....
#     # length of bitstring data(4), bitstring data(varlength), bitstring shape(8)
    
#     with open(file_path, "wb") as f:



#         # Write number of images
#         f.write(struct.pack('I', len(compressed_list)))
        
#         # Write image size (assuming all images have the same size)
#         h, w = image_size
#         f.write(struct.pack('II', h, w))
        
#         for bitstrings in compressed_list:
#             # Write length of each bitstring
#             data = bitstrings["strings"][0][0]
#             f.write(struct.pack('I', len(data)))
#             # Write bitstring data
#             f.write(data)
#             # Write bitstring shape
#             f.write(struct.pack('II', *bitstrings["shape"]))

def save_bitlist(compressed_list, image_size, file_path, continue_write=False,mode="RGB"):
    # Length of bitstring_list(4), length of mode(4), mode(varlength), image size(8), 
    # iteration ....
    # length of bitstring data(4), bitstring data(varlength), bitstring shape(8)
    mode_write = "rb+" if continue_write else "wb"
    with open(file_path, mode_write) as f:
        if continue_write:
            # Read the current number of images
            f.seek(0)
            current_num_images = struct.unpack('I', f.read(4))[0]
            # Update the number of images
            f.seek(0)
            f.write(struct.pack('I', current_num_images + len(compressed_list)))
            # Move to the end of the file
            f.seek(0, 2)
        else:
            # Write number of images
            f.write(struct.pack('I', len(compressed_list)))
            
            # Save mode
            mode_bytes = mode.encode('utf-8')
            f.write(struct.pack('I', len(mode_bytes)))
            f.write(mode_bytes)

            # Write image size (assuming all images have the same size)
            h, w = image_size
            f.write(struct.pack('II', h, w))
        
        for bitstrings in compressed_list:
            # Write length of each bitstring
            data = bitstrings["strings"][0][0]
            f.write(struct.pack('I', len(data)))
            # Write bitstring data
            f.write(data)
            # Write bitstring shape
            f.write(struct.pack('II', *bitstrings["shape"]))


def save_meta_cdf(ds, path="meta_ds.nc"):
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
        
        # Add min and max attributes
        meta_ds[var].attrs['min'] = np.nanmin(ds[var].values) 
        meta_ds[var].attrs['max'] = np.nanmax(ds[var].values)
    
    meta_ds.to_netcdf(path)
    return meta_ds



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
 

def load_bitlist(file_path):
    loaded_list = []
    with open(file_path, "rb") as f:
        # Read number of images


        num_images = struct.unpack('I', f.read(4))[0]
        
        # Read mode
        mode_length = struct.unpack('I', f.read(4))[0]
        mode = f.read(mode_length).decode('utf-8')

        # Read image size
        image_h, image_w = struct.unpack('II', f.read(8))
        
        for _ in range(num_images):
            # Read bitstring length
            data_length = struct.unpack('I', f.read(4))[0]
            # Read bitstring data
            data = f.read(data_length)
            # Read bitstring shape
            shape = struct.unpack('II', f.read(8))
            
            loaded_list.append({
                "strings": [[data]],
                "shape": torch.Size(shape)
            })
    
    return loaded_list, (image_h, image_w), mode


def compress_numpy(input_numpy, net, zip_path, device="cuda",mode="RGB",continue_write=False):

    # Check if the directory for zip_path exists, if not, create it
    zip_dir = os.path.dirname(zip_path)
    if zip_dir and not os.path.exists(zip_dir):
        os.makedirs(zip_dir)
    all_bitstrings = []
    
    # Compression phase
    for i in range(input_numpy.shape[0]):
        # Convert single image numpy array to tensor
        x = torch.from_numpy(input_numpy[i]).permute(2, 0, 1).float().unsqueeze(0).to(device)
        bitstrings = net.compress(x)
        all_bitstrings.append(bitstrings)

    image_size = (input_numpy.shape[1], input_numpy.shape[2])

    save_bitlist(all_bitstrings, image_size, zip_path,mode=mode,continue_write=continue_write)
    return all_bitstrings

import cv2
from tqdm import tqdm
def decompress_numpy(net, zip_path,verbose=True):
    compressed_results = []
    all_bitstrings, (image_h, image_w), mode = load_bitlist(zip_path)
    
    # Decompression phase
    for i in tqdm(range(len(all_bitstrings)),desc="Decompressing",disable=not verbose):
        with torch.no_grad():
            out = net.decompress(all_bitstrings[i]["strings"], all_bitstrings[i]["shape"])
        out_net = out["x_hat"].clamp_(0, 1)
        out_net_cropped = out_net[:, :, :image_h, :image_w]
        compressed = out_net_cropped.squeeze().cpu().permute(1, 2, 0).numpy()
        
        compressed_results.append(compressed)

    return np.array(compressed_results), mode



def evaluate_metric(image_dirs, zip_path, net,mode="RGB"):

    # Load original images and decompress the compressed images
    if mode == "RGB":
        original_images = rgb_to_numpy(image_dirs)
    else:
        original_images = gray_to_numpy(image_dirs)
        original_images = gray_numpy_to_rgb(original_images, mode=mode)

    # Decompress images
    compressed_images, mode = decompress_numpy(net, zip_path)

    results = []
    for orig, comp in zip(original_images, compressed_images):
        # Calculate difference
        diff = np.mean(np.abs(orig - comp), axis=2)
        
        # Calculate RMSE for each channel
        rmse_r = np.sqrt(np.mean((orig[:,:,0] - comp[:,:,0])**2))
        rmse_g = np.sqrt(np.mean((orig[:,:,1] - comp[:,:,1])**2))
        rmse_b = np.sqrt(np.mean((orig[:,:,2] - comp[:,:,2])**2))
        
        # Calculate original size
        original_size = os.path.getsize(image_dirs[0]) / 1024  # KB
        # Calculate compressed size
        compressed_size = os.path.getsize(zip_path) / 1024  # KB
        
        # Calculate compression ratio
        compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
        
        results.append([diff, rmse_r, rmse_g, rmse_b, original_size, compressed_size, compression_ratio])
    
    return results

def plot_compression_results(image_dirs, zip_path, net,mode="RGB"):


    # Load original images and decompress the compressed images
    if mode == "RGB":
        original_images = rgb_to_numpy(image_dirs)
    else:
        original_images = gray_to_numpy(image_dirs)
        original_images = gray_numpy_to_rgb(original_images, mode=mode)
 
    compressed_images, mode  = decompress_numpy(net, zip_path)
    
    # Evaluate metrics for each image
    results = evaluate_metric(image_dirs, zip_path, net,mode)
    
    n = len(original_images)
    if n == 1:
        fig, axes = plt.subplots(1, 3, figsize=(6, 2))
    else:
        fig, axes = plt.subplots(n, 3, figsize=(6, 2*n))  # Don't change figure size


    for i, (original, compressed, result) in enumerate(zip(original_images, compressed_images, results)):
        difference, rmse_r, rmse_g, rmse_b, orig_size, comp_size, compression_ratio = result
        
        if n == 1:
            ax_row = axes
        else:
            ax_row = axes[i]
        
        ax_row[0].imshow(original)
        ax_row[0].set_title('Original')
        ax_row[0].axis('off')
        ax_row[1].imshow(compressed)
        ax_row[1].set_title('Reconstructed')
        ax_row[1].axis('off')
        
        im = ax_row[2].imshow(difference, cmap='viridis')
        ax_row[2].set_title('Difference')
        ax_row[2].axis('off')
        ax_row[2].text(0.5, 0.1, f'RMSE R: {rmse_r:.4f}\nRMSE G: {rmse_g:.4f}\nRMSE B: {rmse_b:.4f}\nOriginal: {orig_size:.2f} KB\nCompressed: {comp_size:.2f} KB\nCompression Ratio: {compression_ratio:.2f}', 
                        horizontalalignment='center', verticalalignment='bottom', 
                        transform=ax_row[2].transAxes, fontsize=8, color='white')
    plt.tight_layout()
    plt.show()
