
import os
import requests
import xarray as xr
from tqdm import tqdm

def download_benchmark(dataset_name, save_dir="./datasets"):
    """
    Download benchmark dataset from OneDrive.

    Parameters:
        dataset_name (str): Name of the dataset (e.g., "CESM_lens").
        save_dir (str): Directory to save the downloaded dataset.

    Returns:
        str: Path to the downloaded dataset.
    """
    # Define OneDrive direct download links for datasets
    dataset_urls = {
        "era5": "https://1drv.ms/u/s!AvF3CQDS6oij2MU7bDHZd5MVOnfXrA?e=lLX1Us",  
        "wrf_short":"https://figshare.com/ndownloader/files/51907229",
        "wrf":"https://figshare.com/ndownloader/files/51907250",
    }

    # Check if the dataset exists
    if dataset_name not in dataset_urls:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(dataset_urls.keys())}")

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Define the path to save the downloaded dataset
    file_path = os.path.join(save_dir, f"{dataset_name}.nc")

    # Check if file already exists
    if os.path.exists(file_path):
        print(f"Dataset {dataset_name} already exists at {file_path}, reloaded")
        return xr.open_dataset(file_path)

    # Download the dataset if it doesn't exist
    print(f"Downloading {dataset_name} from {dataset_urls[dataset_name]}...")
    
    response = requests.get(dataset_urls[dataset_name], stream=True)
    response.raise_for_status()
    
    # Get total file size
    total_size = int(response.headers.get('content-length', 0))
    
    # Create progress bar
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            progress_bar.update(size)
    
    progress_bar.close()
    print(f"Downloaded {dataset_name} to {file_path}")

    return xr.open_dataset(file_path)
