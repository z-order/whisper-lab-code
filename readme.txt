# Remove All Python Versions
sudo apt-get remove --purge python2.* python3.* -y

# Remove pip
sudo apt-get remove --purge python-pip python3-pip -y

# Clean up residual configuration:
sudo apt autoremove --purge -y
sudo apt clean

# Remove Python Package Directories
sudo rm -rf /usr/lib/python2.*
sudo rm -rf /usr/lib/python3.*
sudo rm -rf /usr/local/lib/python2.*
sudo rm -rf /usr/local/lib/python3.*
sudo rm -rf ~/.local/lib/python2.*
sudo rm -rf ~/.local/lib/python3.*

# Remove User-Specific Directories
sudo rm -rf ~/.local/lib/python2.*
sudo rm -rf ~/.local/lib/python3.*
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

## Remove dpkg Old Configurations ##

# Check for any Python-related configurations:
dpkg -l | grep python

# Remove Problematic Configurations
sudo dpkg --purge python3 python3-minimal python3.10 python3.10-minimal 2> /dev/null

# Clean apt Cache
sudo apt clean
sudo rm -rf /var/lib/apt/lists/*
sudo apt update

# Fix dpkg Errors
sudo dpkg --configure -a

# Verify Removal
which python
which python3
which pip

## Reinstall the latest Python version ##

# Update System Packages
sudo apt update && sudo apt upgrade -y

# Fix broken install before installing
sudo apt --fix-broken install

# Check the Latest Python Version
sudo apt show python3     # or sudo apt show -a python3 

# Install Python
sudo apt install python3-minimal python3 python3-pip python3-venv -y

# Verify Installation
python3 --version

## Update Python to the Latest Version (Optional) ##

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




## Or upgrade to python3.12 ##




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

## Ann then, fix some problems ##

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
sudo ln -s apt_pkg.cpython-* apt_pkg.so

# This should create the necessary symbolic link for the apt_pkg module. After that, try:
sudo apt update




## Install Whisper and Dependences ##




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
python -m pip install --user wheel build
python -m build
python -m pip install dist/*.whl

cd /home/ubuntu/{your-project-home}/whisper-lab-code
pip install -r requirements.txt
pip install "fastapi[standard]"





## Run Whisper API Server ##




nohup fastapi run whisper-api-server.py >> .server.log 2>&1 & 



