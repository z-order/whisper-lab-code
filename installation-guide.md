# Installation Guide for Lambda

## Hands-on remove all previous installed python version and reinstall

```sh
# Remove All Python Versions

sudo apt-get remove --purge python2._ python3._ -y

## Remove pip

sudo apt-get remove --purge python-pip python3-pip -y

# Clean up residual configuration:

sudo apt autoremove --purge -y
sudo apt clean

# Remove Python Package Directories

sudo rm -rf /usr/lib/python2._
sudo rm -rf /usr/lib/python3._
sudo rm -rf /usr/local/lib/python2._
sudo rm -rf /usr/local/lib/python3._
sudo rm -rf ~/.local/lib/python2._
sudo rm -rf ~/.local/lib/python3._

# Remove User-Specific Directories

sudo rm -rf ~/.local/lib/python2._
sudo rm -rf ~/.local/lib/python3._
sudo rm -rf ~/.cache/pip
sudo rm -rf ~/.config/pip

# Remove Python Binaries

sudo rm -rf /usr/bin/python*
sudo rm -rf /usr/local/bin/python*

# Remove Cache and Configuration Files

sudo rm -rf ~/.cache/pip
sudo rm -rf ~/.config/pip
sudo rm -rf ~/.pip

# Clear Environment Variables

echo $PYTHONHOME
echo $PYTHONPATH
unset PYTHONHOME
unset PYTHONPATH

# Update Alternatives (if used)

sudo update-alternatives --remove-all python 2> /dev/null
sudo update-alternatives --remove-all python3 2> /dev/null

## Remove dpkg Old Configurations

# Check for any Python-related configurations:

dpkg -l | grep python

# Remove Problematic Configurations

sudo dpkg --purge python3 python3-minimal python3.10 python3.10-minimal 2> /dev/null

# Clean apt Cache

sudo apt clean
sudo rm -rf /var/lib/apt/lists/\*
sudo apt update

# Fix dpkg Errors

sudo dpkg --configure -a

# Verify Removal

which python
which python3
which pip
```

## Reinstall the latest Python version

```sh
# Reboot System

sudo reboot

# Update System Packages

sudo apt update && sudo apt upgrade -y

# Fix broken install before installing

sudo apt --fix-broken install

# Check the Latest Python Version

sudo apt show python3 # or sudo apt show -a python3

# Install Python

sudo apt install python3-minimal python3 python3-pip python3-venv -y

# Verify Installation

python3 --version
```

## Update Python to the Latest Version (Optional)

```sh
# A. Add Deadsnakes PPA (Easier Method), Deadsnakes PPA provides updated Python versions for Ubuntu:

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install the desired version (e.g., Python 3.12):

sudo apt install python3.12 python3.12-venv python3.12-distutils -y

# Set it as the default:

sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
sudo update-alternatives --config python
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
sudo update-alternatives --config python3

# Verify the Installation

python --version
python3 --version
python3.12 --version

# Install pip for Python 3.12, If pip isn't installed, you can install it manually:

sudo apt install python3-pip -y
sudo python3.12 -m ensurepip --upgrade
sudo python3.12 -m pip install --upgrade pip

# Verify:

pip3 --version
python3.12 -m pip --version
```

## Or upgrade to python3.12

```sh
# Check Python versions

which python
which python3
which pip

# Update System Packages

sudo apt update && sudo apt upgrade -y

# A. Add Deadsnakes PPA (Easier Method), Deadsnakes PPA provides updated Python versions for Ubuntu:

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install the desired version (e.g., Python 3.12):

# sudo apt install python3.12 python3.12-venv python3.12-distutils -y

sudo apt install python3.12 python3.12-venv -y

# Set it as the default:

sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
sudo update-alternatives --config python
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
sudo update-alternatives --config python3

# Verify the Installation

python --version
python3 --version
python3.12 --version

# Install pip for Python 3.12, If pip isn't installed, you can install it manually:

sudo apt install python3-pip -y
sudo python3.12 -m ensurepip --upgrade
sudo python3.12 -m pip install --upgrade pip

# Verify:

pip3 --version
python3.12 -m pip --version
```

## Ann then, fix some problems

```sh
# First, let's completely remove the problematic packages

sudo apt remove --purge python3-apt command-not-found command-not-found-data

# Clean up any leftover configuration

sudo apt autoremove
sudo apt autoclean

# Now reinstall them in the correct order

sudo apt install python3-apt
sudo apt install command-not-found

# Then let's try to rebuild the Python apt binding specifically:

cd /usr/lib/python3/dist-packages
sudo ln -s apt_pkg.cpython-\* apt_pkg.so

# This should create the necessary symbolic link for the apt_pkg module. After that, try:

sudo apt update
```

## Install Whisper ,dependences and API Server

```sh
cd /home/ubuntu
python -m venv .venv
source .venv/bin/activate
echo "source .venv/bin/activate" >> ~/.bshrc
pip install --upgrade pip

sudo apt update && sudo apt install ffmpeg

cd /home/ubuntu/{your-project-home}/openai-whisper-ws
pip install -r requirements.txt
sudo apt-get install python3-setuptools
python -m pip install --upgrade setuptools

# python -m pip install --user wheel build

python -m pip install wheel build
python -m build
python -m pip install dist/*.whl

cd /home/ubuntu/{your-project-home}/whisper-lab-code
pip install -r requirements.txt
pip install "fastapi[standard]"

## Run Whisper API Server

nohup fastapi run whisper-api-server.py >> .server.log 2>&1 &
```

## NVIDA CUDA Toolkit

```sh
nvidia-smi
nvcc --version
    Command 'nvcc' not found, but can be installed with:
    sudo apt install nvidia-cuda-toolkit
    sudo systemctl restart nvidia-persistenced

## Conda compatibility

conda search pytorch-cuda -c nvidia
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda activate pt24_py311
```

## Python code 1 for checking the installation

```python
import sys
import pip
import torch

if __name__ == "__main__" :

    # Python version
    print(f"Python Version: {sys.version}")
    print(f"Python Version Info: {sys.version_info}")
    print(f"Python Version Major: {sys.version_info.major}")
    print(f"Python Version Minor: {sys.version_info.minor}")
    print("\n")

    # pip version
    print(f"pip Version: {pip.__version__}")
    print("\n")

    # Torch version
    print("PyTorch Version:", torch.__version__)
    print("CUDA Version:", torch.version.cuda)
    print("Is CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    print("\n")
```

```sh
Output:

Python Version: 3.11.10 | packaged by conda-forge | (main, Oct 16 2024, 01:27:36) [GCC 13.3.0]
Python Version Info: sys.version_info(major=3, minor=11, micro=10, releaselevel='final', serial=0)
Python Version Major: 3
Python Version Minor: 11

pip Version: 24.2

PyTorch Version: 2.4.1
CUDA Version: 12.4
/home/ubuntu/miniforge3/envs/pt24_py311/lib/python3.11/site-packages/torch/cuda/**init**.py:128: UserWarning: CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 802: system not yet initialized (Triggered internally at /opt/conda/conda-bld/pytorch_1724789121465/work/c10/cuda/CUDAFunctions.cpp:108.)
return torch.\_C.\_cuda_getDeviceCount() > 0
Is CUDA available: False
Number of GPUs: 8
```

## Python code 2 for checking the installation

```python
import sys
import pip
import torch

if __name__ == "__main__":
    # Python and pip versions
    print(f"Python Version: {sys.version}")
    print(f"Python Version Info: {sys.version_info}")
    print(f"Python Version Major: {sys.version_info.major}")
    print(f"Python Version Minor: {sys.version_info.minor}")
    print("\n")
    print(f"pip Version: {pip.__version__}")
    print("\n")

    # CUDA debugging information
    try:
        torch.cuda.init()  # Try initializing CUDA explicitly
        print("PyTorch Version:", torch.__version__)
        print("CUDA Version:", torch.version.cuda)
        print("Is CUDA available:", torch.cuda.is_available())
        print("Number of GPUs:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("Current GPU:", torch.cuda.current_device())
            print("GPU Name:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("CUDA initialization error:", str(e))
    print("\n")

    # Additional system information
    try:
        import subprocess
        nvidia_smi = subprocess.check_output(['nvidia-smi']).decode()
        print("NVIDIA-SMI output:")
        print(nvidia_smi)
    except Exception as e:
        print("Error running nvidia-smi:", str(e))
```

```sh
Output:

Python Version: 3.11.11 | packaged by conda-forge | (main, Dec 5 2024, 14:17:24) [GCC 13.3.0]
Python Version Info: sys.version_info(major=3, minor=11, micro=11, releaselevel='final', serial=0)
Python Version Major: 3
Python Version Minor: 11

pip Version: 24.3.1

CUDA initialization error: Torch not compiled with CUDA enabled

NVIDIA-SMI output:
Tue Jan 7 14:57:13 2025
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.06 Driver Version: 535.183.06 CUDA Version: 12.2 |
|-----------------------------------------+----------------------+----------------------+
| GPU Name Persistence-M | Bus-Id Disp.A | Volatile Uncorr. ECC |
| Fan Temp Perf Pwr:Usage/Cap | Memory-Usage | GPU-Util Compute M. |
| | | MIG M. |
|=========================================+======================+======================|
| 0 NVIDIA A100-SXM4-80GB On | 00000000:00:06.0 Off | 0 |
| N/A 29C P0 51W / 400W | 0MiB / 81920MiB | 0% Default |
| | | Disabled |
+-----------------------------------------+----------------------+----------------------+
| 1 NVIDIA A100-SXM4-80GB On | 00000000:00:07.0 Off | 0 |
| N/A 28C P0 52W / 400W | 0MiB / 81920MiB | 0% Default |
| | | Disabled |
+-----------------------------------------+----------------------+----------------------+
| 2 NVIDIA A100-SXM4-80GB On | 00000000:00:08.0 Off | 0 |
| N/A 28C P0 48W / 400W | 0MiB / 81920MiB | 0% Default |
| | | Disabled |
+-----------------------------------------+----------------------+----------------------+
| 3 NVIDIA A100-SXM4-80GB On | 00000000:00:09.0 Off | 0 |
| N/A 28C P0 50W / 400W | 0MiB / 81920MiB | 0% Default |
| | | Disabled |
+-----------------------------------------+----------------------+----------------------+
| 4 NVIDIA A100-SXM4-80GB On | 00000000:00:0A.0 Off | 0 |
| N/A 27C P0 50W / 400W | 0MiB / 81920MiB | 0% Default |
| | | Disabled |
+-----------------------------------------+----------------------+----------------------+
| 5 NVIDIA A100-SXM4-80GB On | 00000000:00:0B.0 Off | 0 |
| N/A 28C P0 51W / 400W | 0MiB / 81920MiB | 0% Default |
| | | Disabled |
+-----------------------------------------+----------------------+----------------------+
| 6 NVIDIA A100-SXM4-80GB On | 00000000:00:12.0 Off | 0 |
| N/A 28C P0 51W / 400W | 0MiB / 81920MiB | 0% Default |
| | | Disabled |
+-----------------------------------------+----------------------+----------------------+
| 7 NVIDIA A100-SXM4-80GB On | 00000000:00:13.0 Off | 0 |
| N/A 28C P0 50W / 400W | 0MiB / 81920MiB | 0% Default |
| | | Disabled |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes: |
| GPU GI CI PID Type Process name GPU Memory |
| ID ID Usage |
|=======================================================================================|
| No running processes found |
+---------------------------------------------------------------------------------------+
```

## Fixing the PyTorch Installation

```sh
conda remove pytorch torchvision torchaudio

# For CUDA 12.0

# conda install pytorch torchvision torchaudio pytorch-cuda=12.0 -c pytorch -c nvidia

# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# conda install pytorch torchvision torchaudio pytorch-cuda=12.2 -c pytorch -c nvidia

# conda install pytorch torchvision torchaudio pytorch-cuda=12.3 -c pytorch -c nvidia

# For CUDA 12.4

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

## Python code 3 for checking the installation

```python
import sys
import pip
import torch

if __name__ == "__main__":
    # Python and pip versions
    print(f"Python Version: {sys.version}")
    print(f"Python Version Info: {sys.version_info}")
    print(f"Python Version Major: {sys.version_info.major}")
    print(f"Python Version Minor: {sys.version_info.minor}")
    print("\n")
    print(f"pip Version: {pip.__version__}")
    print("\n")

    # CUDA debugging information
    try:
        torch.cuda.init()  # Try initializing CUDA explicitly
        print("PyTorch Version:", torch.__version__)
        print("CUDA Version:", torch.version.cuda)
        print("Is CUDA available:", torch.cuda.is_available())
        print("Number of GPUs:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("Current GPU:", torch.cuda.current_device())
            print("GPU Name:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("CUDA initialization error:", str(e))
    print("\n")

    # Additional system information
    try:
        import subprocess
        nvidia_smi = subprocess.check_output(['nvidia-smi']).decode()
        print("NVIDIA-SMI output:")
        print(nvidia_smi)
    except Exception as e:
        print("Error running nvidia-smi:", str(e))

    try:
        import subprocess
        nvcc = subprocess.check_output(['nvcc', '--version']).decode()
        print("nvcc --version output:")
        print(nvcc)
    except Exception as e:
        print("Error running nvcc --version:", str(e))
```

```sh
Output:

Python Version: 3.12.8 | packaged by conda-forge | (main, Dec 5 2024, 14:24:40) [GCC 13.3.0]
Python Version Info: sys.version_info(major=3, minor=12, micro=8, releaselevel='final', serial=0)
Python Version Major: 3
Python Version Minor: 12

pip Version: 24.3.1

CUDA initialization error: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 802: system not yet initialized

NVIDIA-SMI output:
Tue Jan 7 15:23:14 2025
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.06 Driver Version: 535.183.06 CUDA Version: 12.2 |
|-----------------------------------------+----------------------+----------------------+
| GPU Name Persistence-M | Bus-Id Disp.A | Volatile Uncorr. ECC |
| Fan Temp Perf Pwr:Usage/Cap | Memory-Usage | GPU-Util Compute M. |
| | | MIG M. |
|=========================================+======================+======================|
| 0 NVIDIA A100-SXM4-80GB On | 00000000:00:06.0 Off | 0 |
| N/A 29C P0 51W / 400W | 0MiB / 81920MiB | 0% Default |
| | | Disabled |
+-----------------------------------------+----------------------+----------------------+
| 1 NVIDIA A100-SXM4-80GB On | 00000000:00:07.0 Off | 0 |
| N/A 29C P0 52W / 400W | 0MiB / 81920MiB | 0% Default |
| | | Disabled |
+-----------------------------------------+----------------------+----------------------+
| 2 NVIDIA A100-SXM4-80GB On | 00000000:00:08.0 Off | 0 |
| N/A 28C P0 48W / 400W | 0MiB / 81920MiB | 0% Default |
| | | Disabled |
+-----------------------------------------+----------------------+----------------------+
| 3 NVIDIA A100-SXM4-80GB On | 00000000:00:09.0 Off | 0 |
| N/A 28C P0 50W / 400W | 0MiB / 81920MiB | 0% Default |
| | | Disabled |
+-----------------------------------------+----------------------+----------------------+
| 4 NVIDIA A100-SXM4-80GB On | 00000000:00:0A.0 Off | 0 |
| N/A 27C P0 50W / 400W | 0MiB / 81920MiB | 0% Default |
| | | Disabled |
+-----------------------------------------+----------------------+----------------------+
| 5 NVIDIA A100-SXM4-80GB On | 00000000:00:0B.0 Off | 0 |
| N/A 28C P0 51W / 400W | 0MiB / 81920MiB | 0% Default |
| | | Disabled |
+-----------------------------------------+----------------------+----------------------+
| 6 NVIDIA A100-SXM4-80GB On | 00000000:00:12.0 Off | 0 |
| N/A 28C P0 51W / 400W | 0MiB / 81920MiB | 0% Default |
| | | Disabled |
+-----------------------------------------+----------------------+----------------------+
| 7 NVIDIA A100-SXM4-80GB On | 00000000:00:13.0 Off | 0 |
| N/A 28C P0 50W / 400W | 0MiB / 81920MiB | 0% Default |
| | | Disabled |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes: |
| GPU GI CI PID Type Process name GPU Memory |
| ID ID Usage |
|=======================================================================================|
| No running processes found |
+---------------------------------------------------------------------------------------+

nvcc --version output:
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0
```

## Fixing the CUDA Toolkit and PyTorch Installation

The issues are, CUDA Version Mismatch:

Your system has CUDA 12.2 (from nvidia-smi)
But nvcc shows CUDA 11.5
PyTorch needs matching CUDA versions

CUDA Initialization Error with error 802 suggests a system initialization problem. Here are the recommended steps to fix these issues:

1. First, update CUDA toolkit to match your driver's supported version (12.2):

   ```sh
   # Remove existing CUDA toolkit
   conda remove cuda-toolkit

   # Install CUDA 12.2
   conda install cuda-toolkit=12.2 -c nvidia
   ```

2. Reinstall PyTorch with the matching CUDA version:

   ```sh
   # Remove existing PyTorch
   conda remove pytorch torchvision torchaudio

   # Install PyTorch with CUDA 12.2
   conda install pytorch torchvision torchaudio pytorch-cuda=12.2 -c pytorch -c nvidia
   ```

3. Set CUDA environment variables:

   ```sh
   export CUDA_HOME=/usr/local/cuda-12.2
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   export PATH=$CUDA_HOME/bin:$PATH
   ```

4. Create a new test script to verify the installation:

   ```python
   import sys
   import pip
   import torch

   if __name__ == "__main__":
       # Python and pip versions
       print(f"Python Version: {sys.version}")
       print(f"Python Version Info: {sys.version_info}")
       print(f"Python Version Major: {sys.version_info.major}")
       print(f"Python Version Minor: {sys.version_info.minor}")
       print(f"pip Version: {pip.__version__}")

       # CUDA initialization
       try:
           torch.cuda.init()  # Try initializing CUDA explicitly
       except Exception as e:
           print("CUDA initialization error:", str(e))

       # CUDA debugging information
       try:
           print("PyTorch Version:", torch.__version__)
           print("CUDA Version:", torch.version.cuda)
           print("Is CUDA available:", torch.cuda.is_available())
           print("Number of GPUs:", torch.cuda.device_count())
           if torch.cuda.is_available():
               print("Current GPU:", torch.cuda.current_device())
               print("GPU Name:", torch.cuda.get_device_name(0))
       except Exception as e:
           print("CUDA initialization error:", str(e))

       # Test Tensor operations on CUDA
       if torch.cuda.is_available():
           device = torch.device("cuda")
           x = torch.tensor([1.0, 2.0, 3.0], device=device)
           print(f"Tensor on GPU: {x}")
           print(f"GPU Device: {torch.cuda.get_device_name(0)}")

       # Additional system information
       try:
           import subprocess
           nvidia_smi = subprocess.check_output(['nvidia-smi']).decode()
           print("\nNVIDIA-SMI output:")
           print(nvidia_smi)
       except Exception as e:
           print("Error running nvidia-smi:", str(e))

       try:
           import subprocess
           nvcc = subprocess.check_output(['nvcc', '--version']).decode()
           print("nvcc --version output:")
           print(nvcc)
       except Exception as e:
           print("Error running nvcc --version:", str(e))
   ```

## Ref: Installation Guider - Conda, PyTorch, CUDA, and NVIDIA Drivers

### **Step 1: Install NVIDIA Drivers**

1. **Identify Your GPU Model:**

   ```bash
   lspci | grep -i nvidia
   ```

2. **Remove Existing Drivers (Optional):**

   ```bash
   sudo apt remove --purge '^nvidia-.*'
   sudo apt autoremove
   ```

3. **Add NVIDIA PPA and Install Drivers:**

   ```bash
   sudo add-apt-repository ppa:graphics-drivers/ppa
   sudo apt update
   sudo apt install nvidia-driver-<version>
   ```

   Replace `<version>` with the recommended version from the [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx).

4. **Reboot:**

   ```bash
   sudo reboot
   ```

5. **Verify Installation:**

   ```bash
   nvidia-smi
   ```

---

### **Step 2: Install Miniconda**

1. **Download the Installer:**

   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```

2. **Run the Installer:**

   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

3. **Follow On-Screen Instructions.**

4. **Initialize Conda:**

   ```bash
   source ~/.bashrc
   conda init
   ```

---

### **Step 3: Create a Conda Environment**

1. **Create a New Environment for PyTorch:**

   ```bash
   conda create --name pytorch_env python=3.9
   ```

2. **Activate the Environment:**

   ```bash
   conda activate pytorch_env
   ```

---

### **Step 4: Install PyTorch and CUDA**

1. **Install PyTorch with CUDA Support:**
   Use the official [PyTorch Installation Page](https://pytorch.org/get-started/locally/) to select the right command. Example:

   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.2 -c pytorch -c nvidia
   ```

2. **Verify PyTorch Installation:**

   ```python
   python -c "import torch; print(torch.cuda.is_available())"
   ```

   If `True` is returned, PyTorch is installed and can use the GPU.

---

### **Step 5: Verify CUDA Toolkit Installation**

1. **Check CUDA Version:**

   ```bash
   nvcc --version
   ```

   If the `nvcc` command isn’t available, add CUDA to the PATH:

   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

2. **Test with a PyTorch Program:**

   ```python
   python -c "print(torch.cuda.get_device_name(0))"
   ```

---

### **Step 6: Optional - Install Other Dependencies**

For additional libraries (e.g., OpenCV, Scikit-learn):

```bash
conda install opencv scikit-learn matplotlib -c conda-forge
```

---

### **Step 7: Update and Maintain the Setup**

1. **Update Conda Environment:**

   ```bash
   conda update --all
   ```

2. **Install Specific NVIDIA Packages if Needed:**

   ```bash
   conda install -c nvidia cuda-toolkit
   ```

---

### **Quick Summary of Commands**

```bash
# NVIDIA Drivers
sudo apt install nvidia-driver-<version>
nvidia-smi

# Miniconda Installation
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create and Activate Environment
conda create --name pytorch_env python=3.9
conda activate pytorch_env

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.2 -c pytorch -c nvidia

# Verify Installation
python -c "import torch; print(torch.cuda.is_available())"

```

## Cuda compatibility: Pytorch version for cuda 12.2

You can install the nightly build. Note you should have cudnn installed already, I am using cudnn v8.9.3. The 12.1 PyTorch version works fine with CUDA v12.2.2:

```sh
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
```

I had a similar issue of Torch not compiled with CUDA enabled. Here are the steps I took:

1. Created a new conda environment

   ```sh
   conda create -n newenv python=3.10
   conda activate newenv
   ```

2. Installed pytorch-nightly

   ```sh
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
   ```

3. Check if the CUDA is compatible with the installed PyTorch by running

   ```sh
   python -c "import torch; print(torch.cuda.is_available())"
   ```

It would return True if the CUDA is configured properly.

## Installation cuda-toolkit using conda which must be compatible with PyTorch and NVIDIA drivers (nvidia-smi version)

You can install the `cuda-toolkit` package using conda. Here's how you can do it:

1. Check the installed NVIDIA driver version:

   ```sh
   nvidia-smi
   ```

2. Check the installed CUDA version:

   ```sh
   nvcc --version
   which nvcc
   ```

3. Check cuda-toolkit directory:

   ```sh
   ls -al /usr/local | grep cuda
   ```

4. Search for available versions:

   ```sh
   conda search cuda-toolkit
   ```

5. nstall a specific version:

   ```sh
   # CUDA 11.7
   conda install cuda-toolkit=11.7 -c nvidia

   # CUDA 11.8
   conda install cuda-toolkit=11.8 -c nvidia

   # CUDA 12.0
   conda install cuda-toolkit=12.0 -c nvidia

   # CUDA 12.1
   conda install cuda-toolkit=12.1 -c nvidia

   # CUDA 12.2
   conda install cuda-toolkit=12.2 -c nvidia

   # CUDA 12.4
   conda install cuda-toolkit=12.4 -c nvidia
   ```

6. Set the environment variables:

   ```sh
   export CUDA_HOME=/usr/local/cuda-12.2
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   export PATH=$CUDA_HOME/bin:$PATH
   ```

7. Set the installation version as the default:

   ```sh
   sudo rm /usr/bin/nvcc
   sudo ln -s /usr/local/cuda-12.2/bin/nvcc /usr/bin/nvcc
   sudo update-alternatives --remove-all nvcc 2> /dev/null
   sudo update-alternatives --install /usr/bin/nvcc nvcc /usr/local/cuda-12.2/bin/nvcc 1
   ```

8. Verify the installation:

   ```sh
   ls -al /usr/local | grep cuda
   nvcc --version
   /usr/bin/nvcc --version
   ldconfig -p | grep cuda
   ```

## Installation of PyTorch with CUDA 12.x

1. Check the version compatibility PyTorch with CUDA [[https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/), [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive), [https://www.nvidia.com/en-us/drivers/](https://www.nvidia.com/en-us/drivers/), [https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation), [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads), [https://docs.nvidia.com/deploy/cuda-compatibility/](https://docs.nvidia.com/deploy/cuda-compatibility/) ]

   PyTorch 2.5.1

   ```sh
   # CUDA 12.4
   conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
   # CPU Only
   conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 cpuonly -c pytorch

   # Refs using pip:
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu124


   # CUDA 12.4 Toolkit Installer (https://developer.nvidia.com/cuda-downloads)
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-4

   # CUDA 12.4 Driver Installer (NVIDIA Driver Instructions (choose one option))
   # Option 1: To install the open kernel module flavor:
   sudo apt-get install -y nvidia-open
   # Option 2: To install the legacy kernel module flavor:
   sudo apt-get install -y cuda-drivers
   ```

   PyTorch 2.5.0

   ```sh
   # CUDA 11.8
   conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0  pytorch-cuda=11.8 -c pytorch -c nvidia
   # CUDA 12.1
   conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia
   # CUDA 12.4
   conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
   # CPU Only
   conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 cpuonly -c pytorch
   ```

   PyTorch 2.4.0

   ```sh
   # CUDA 11.8
   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia
   # CUDA 12.1
   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
   # CUDA 12.4
   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
   # CPU Only
   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 cpuonly -c pytorch
   ```

   You can use PyTorch 2.4.0 with CUDA 12.1 for CUDA 12.2

   ```sh
   # CUDA 12.1
   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

## Installation of NVIDIA Drivers

### Step 1: Remove Existing NVIDIA Drivers and CUDA Toolkit

Before proceeding, remove any pre-insatlled NVIDIA drivers or CUDA toolkit:

1. Purge all installed NVIDIA drivers:

   ```sh
   sudo apt-get remove --purge '^nvidia-.*'
   ```

2. Remove residual NVIDIA packages:

   ```sh
   sudo apt autoremove --purge
   sudo apt-get clean
   ```

3. Verify that no NVIDIA drivers are active:

   ```sh
   lsmod | grep nvidia
   ```

   If the command shows NVIDIA modules still loaded, reboot the system:

   ```sh
   sudo reboot
   ```

### Step 2: Install the New NVIDIA Driver

1. Ensure your system is prepared for the driver installation:

   Install required dependencies:

   ```sh
    sudo apt update
    sudo apt install build-essential dkms
   ```

2. Navigate to the directory where you downloaded the driver:

   ```sh
   cd /path/to/driver
   ```

   Replace /path/to/driver with the location of the .deb file.

3. nstall the NVIDIA driver .deb package:

   ```sh
   sudo dpkg -i nvidia-driver-local-repo-ubuntu2204-550.127.08_1.0-1_amd64.deb
   ```

4. Add the repository to your system:

   ```sh
   sudo cp /var/nvidia-driver-local-repo-ubuntu2204-550.127.08/nvidia-driver-local-42AEA102-keyring.gpg /usr/share/keyrings/

   or

   sudo apt-key add /var/nvidia-driver-local-repo-*/keyring.gpg
   ```

5. Update and install the driver:

   ```sh
   sudo apt update
   sudo apt install nvidia-driver-550
   ```

6. Reboot and Verify

   Reboot the system:

   ```sh
   sudo reboot
   ```

   After reboot, verify the installation:

   ```sh
   nvidia-smi
   ```

7. (Optional Steps) If you have CUDA installed, verify its functionality:

   ```sh
   nvcc --version
   ```

   If the command is not found, add the CUDA path to the PATH environment variable:

   ```sh
   export PATH=/usr/local/cuda/bin:$PATH
   ```

   Verify the CUDA version:

   ```sh
   nvcc --version
   ```

## Installation Key Points

1. Install the NVIDIA driver using the .deb package. (Check with 'nvidia-smi')
2. Install the same version of CUDA Toolkit as the NVIDIA CUDA Comiler using the .deb package. (Check with 'nvcc --version')
3. Install your PyTorch version with the matching PyTorch-CUDA version.
4. Lastly, verify the installation with a simple PyTorch script after rebooting the system.

## CUDA initialization: Unexpected error from cudaGetDeviceCount() Error 802: system not yet initialized

1. Add the following environment variables to your system:

   ```sh
   CUDA_DEVICE_ORDER="PCI_BUS_ID" PYTORCH_NVML_BASED_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3
   ```

2. Install Fabric Manager:

   Add the NVIDIA package repository (if not already added):

   ```sh
   sudo apt update
   sudo apt install -y software-properties-common
   sudo add-apt-repository ppa:graphics-drivers/ppa
   ```

   Install Fabric Manager:

   ```sh
   sudo apt update
   sudo apt install -y nvidia-fabricmanager-<version>
   ```

   Replace `<version>` with the appropriate version matching your driver version (e.g., nvidia-fabricmanager-550).

   Start and enable the Fabric Manager service:

   ```sh
   sudo systemctl enable nvidia-fabricmanager
   sudo systemctl start nvidia-fabricmanager
   ```

3. Ensure GPU persistence mode is on:

   ```sh
   nvidia-smi -pm 1
   ```

## References for well-made setups

Lambdalabs A10 / Ubuntu 24.04 LTS / Python 3.10 / PyTorch 2.5.1 / CUDA 12.4

```sh
Python Version: 3.10.12 (main, Nov  6 2024, 20:22:13) [GCC 11.4.0]
Python Version Info: sys.version_info(major=3, minor=10, micro=12, releaselevel='final', serial=0)
Python Version Major: 3
Python Version Minor: 10
pip Version: 22.0.2
PyTorch Version: 2.5.1
CUDA Version: 12.4
Is CUDA available: True
Number of GPUs: 1
Current GPU: 0
GPU Name: NVIDIA A10
Tensor on GPU: tensor([1., 2., 3.], device='cuda:0')
GPU Device: NVIDIA A10

NVIDIA-SMI output:
Tue Jan  7 12:53:50 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A10                     On  |   00000000:07:00.0 Off |                    0 |
|  0%   31C    P0             27W /  150W |     275MiB /  23028MiB |      2%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      2624      C   python                                        266MiB |
+-----------------------------------------------------------------------------------------+

nvcc --version output:
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```

NHN Cloud (AICA) / Ubuntu 22.04 LTS / Python 3.11 / PyTorch 2.5.1 / CUDA 12.4 : In case of CUDA initialization error,

```sh
Python Version: 3.11.9 | packaged by conda-forge | (main, Apr 19 2024, 18:36:13) [GCC 12.3.0]
Python Version Info: sys.version_info(major=3, minor=11, micro=9, releaselevel='final', serial=0)
Python Version Major: 3
Python Version Minor: 11
pip Version: 24.3.1
CUDA initialization error: No CUDA GPUs are available
PyTorch Version: 2.5.1+cu124
CUDA Version: 12.4
Is CUDA available: False
Number of GPUs: 8

NVIDIA-SMI output:
Wed Jan  8 00:03:51 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.08             Driver Version: 550.127.08     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-80GB          Off |   00000000:00:06.0 Off |                   On |
| N/A   29C    P0             51W /  400W |       1MiB /  81920MiB |     N/A      Default |
|                                         |                        |              Enabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA A100-SXM4-80GB          Off |   00000000:00:07.0 Off |                   On |
| N/A   28C    P0             52W /  400W |       1MiB /  81920MiB |     N/A      Default |
|                                         |                        |              Enabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA A100-SXM4-80GB          Off |   00000000:00:08.0 Off |                   On |
| N/A   28C    P0             48W /  400W |       1MiB /  81920MiB |     N/A      Default |
|                                         |                        |              Enabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA A100-SXM4-80GB          Off |   00000000:00:09.0 Off |                   On |
| N/A   28C    P0             49W /  400W |       1MiB /  81920MiB |     N/A      Default |
|                                         |                        |              Enabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA A100-SXM4-80GB          Off |   00000000:00:0A.0 Off |                   On |
| N/A   27C    P0             50W /  400W |       1MiB /  81920MiB |     N/A      Default |
|                                         |                        |              Enabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA A100-SXM4-80GB          Off |   00000000:00:0B.0 Off |                   On |
| N/A   28C    P0             51W /  400W |       1MiB /  81920MiB |     N/A      Default |
|                                         |                        |              Enabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA A100-SXM4-80GB          Off |   00000000:00:12.0 Off |                   On |
| N/A   28C    P0             51W /  400W |       1MiB /  81920MiB |     N/A      Default |
|                                         |                        |              Enabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA A100-SXM4-80GB          Off |   00000000:00:13.0 Off |                   On |
| N/A   28C    P0             51W /  400W |       1MiB /  81920MiB |     N/A      Default |
|                                         |                        |              Enabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| MIG devices:                                                                            |
+------------------+----------------------------------+-----------+-----------------------+
| GPU  GI  CI  MIG |                     Memory-Usage |        Vol|      Shared           |
|      ID  ID  Dev |                       BAR1-Usage | SM     Unc| CE ENC DEC OFA JPG    |
|                  |                                  |        ECC|                       |
|==================+==================================+===========+=======================|
|  No MIG devices found                                                                   |
+-----------------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

nvcc --version output:
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```

NHN Cloud (AICA) / Ubuntu 22.04 LTS / Python 3.11 / PyTorch 2.5.1 / CUDA 12.4 : In case of normal initialization,

```sh
Python Version: 3.11.9 | packaged by conda-forge | (main, Apr 19 2024, 18:36:13) [GCC 12.3.0]
Python Version Info: sys.version_info(major=3, minor=11, micro=9, releaselevel='final', serial=0)
Python Version Major: 3
Python Version Minor: 11
pip Version: 24.3.1
PyTorch Version: 2.5.1+cu124
CUDA Version: 12.4
Is CUDA available: True
Number of GPUs: 8
Current GPU: 0
GPU Name: NVIDIA A100-SXM4-80GB
Tensor on GPU: tensor([1., 2., 3.], device='cuda:0')
GPU Device: NVIDIA A100-SXM4-80GB

NVIDIA-SMI output:
Wed Jan  8 00:50:16 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.08             Driver Version: 550.127.08     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-80GB          On  |   00000000:00:06.0 Off |                    0 |
| N/A   30C    P0             67W /  400W |     501MiB /  81920MiB |      7%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA A100-SXM4-80GB          On  |   00000000:00:07.0 Off |                    0 |
| N/A   30C    P0             62W /  400W |       4MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA A100-SXM4-80GB          On  |   00000000:00:08.0 Off |                    0 |
| N/A   29C    P0             58W /  400W |       4MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA A100-SXM4-80GB          On  |   00000000:00:09.0 Off |                    0 |
| N/A   29C    P0             59W /  400W |       4MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA A100-SXM4-80GB          On  |   00000000:00:0A.0 Off |                    0 |
| N/A   28C    P0             62W /  400W |       4MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA A100-SXM4-80GB          On  |   00000000:00:0B.0 Off |                    0 |
| N/A   30C    P0             61W /  400W |       4MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA A100-SXM4-80GB          On  |   00000000:00:12.0 Off |                    0 |
| N/A   29C    P0             61W /  400W |       4MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA A100-SXM4-80GB          On  |   00000000:00:13.0 Off |                    0 |
| N/A   29C    P0             60W /  400W |       4MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      4588      C   python                                        492MiB |
+-----------------------------------------------------------------------------------------+

nvcc --version output:
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```
