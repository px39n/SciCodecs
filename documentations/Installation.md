
sci-codec is a pure Python package, but its dependencies are not. The simplest way to install everything is using PIP:

```bash
pip install scicodec
```

sci-codec originally only supported limited lossy compression methods like fpzip and simple lossless compression like gzip. The library provides interfaces across third-party libraries including:

- pressio
- numcodecs
- compressai

You can install extensions through the following methods:


### Basic Installation
Install with all basic dependencies:
```bash
pip install scicodec[basic]
```


### Individual Extensions

Install specific compression libraries as needed:



**CompressAI Support:**
```bash
pip install scicodec[compressai]
```
or go to [CompressAI installation page](https://interdigitalinc.github.io/CompressAI/installation.html)

**Numcodecs Support:**
```bash
pip install scicodec[numcodecs]
```
or go to [Numcodecs installation page](https://numcodecs.readthedocs.io/en/stable/)

**Pressio Support**

To use Pressio compression in sci-codec, follow these steps:

1. First, clone and install Spack package manager:
```bash
git clone https://github.com/spack/spack.git
source spack/share/spack/setup-env.sh
```

2. Install libpressio with required plugins using Spack:
```bash
spack install libpressio+mgard+sz3+zfp+python
```

3. Configure Python environment to find libpressio:
   
   Create a conda environment activation script:
   ```bash
   mkdir -p ~/anaconda3/envs/YOUR_ENV_NAME/etc/conda/activate.d
   ```

   Create a new file `pythonpath.sh` in this directory:
   ```bash
   echo 'export PYTHONPATH=$(spack location -i libpressio)/lib/python3.13/site-packages:$PYTHONPATH' > ~/anaconda3/envs/YOUR_ENV_NAME/etc/conda/activate.d/pythonpath.sh
   ```

4. If you encounter GLIBCXX version errors, you may need to update the libstdc++ library:
   ```bash
   # Backup existing library
   mv ~/anaconda3/envs/YOUR_ENV_NAME/lib/libstdc++.so.6 ~/anaconda3/envs/YOUR_ENV_NAME/lib/libstdc++.so.6.bak
   
   # Create symlink to system library
   ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ~/anaconda3/envs/YOUR_ENV_NAME/lib/libstdc++.so.6
   ```

Note: Replace `YOUR_ENV_NAME` with your actual conda environment name.

or go to [Pressio installation page](https://github.com/CODARcode/libpressio)

### Development Installation
For contributing or development, install with additional development dependencies:
```bash
git clone https://github.com/sci-codec/sci-codec.git
cd sci-codec
pip install -e .[dev]
```

### System Requirements

- Python 3.7 or higher
- C++ compiler (for some compression libraries)
- CUDA toolkit (optional, for GPU acceleration)

### Troubleshooting

If you encounter issues with binary dependencies, ensure you have the following:
1. Updated pip: `pip install --upgrade pip`
2. Required system libraries: 
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential
   
   # CentOS/RHEL
   sudo yum groupinstall "Development Tools"
   ```

For GPU support, make sure CUDA toolkit is properly installed and visible in your system path.