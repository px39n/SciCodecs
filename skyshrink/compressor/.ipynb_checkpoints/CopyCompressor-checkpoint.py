import shutil
from .base import CompressionMethod
import os
class CopyCompressor(CompressionMethod):
    def __init__(self):
        super().__init__()
        self.name='copy_compressor'
    def compress(self, original_path, zip_dir):
        # Simply copy the file or directory from original_path to zip_dir
        shutil.copy2(original_path, zip_dir)
        print(f"Data copied from {original_path} to {zip_dir} without compression.")

    def decompress(self, zip_dir, unzip_path):
        # Copy back the compressed (in this case, just copied) directory or file
        shutil.copy2(zip_dir, unzip_path)
        print(f"Data copied back from {zip_dir} to {unzip_path} without decompression.")
