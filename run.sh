#!/bin/bash

# Environment name variable
ENV_NAME="tf"

# Download Miniconda installer
wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

# Install Miniconda
bash Miniconda3-latest-Linux-x86_64.sh -b -u

# Add Miniconda to PATH
export PATH="/home/max/miniconda3/bin:$PATH"

# Initialize Conda
conda init

# Source the bashrc file
source ~/.bashrc

# Create a Conda environment with Python 3.9
conda create --name $ENV_NAME python=3.9 -y

# reinitialize
source activate base

# Activate the 'tf' environment
conda activate $ENV_NAME

# Install CUDA Toolkit and cuDNN
conda install -y -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0

# Install CUDA NVCC
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-nvcc

# Install Python packages from requirements.txt
pip install -r requirements.txt

# Install NVIDIA PyIndex
pip install nvidia-pyindex

# Upgrade NVIDIA TensorRT
pip install --upgrade nvidia-tensorrt

# Update LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# Find libnvinfer.so.* file and set up LD_LIBRARY_PATH
LIBNVINFER_PATH=$(dirname $(find ~/ -type f -name 'libnvinfer.so.*'))

# Navigate to TensorRT libs directory
cd $LIBNVINFER_PATH

# Update LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBNVINFER_PATH

# Create symbolic links for libnvinfer_plugin and libnvinfer
if [ -f "libnvinfer_plugin.so.8" ] && [ -f "libnvinfer.so.8" ]; then
    ln -s libnvinfer_plugin.so.8 libnvinfer_plugin.so.7 &&
    ln -s libnvinfer.so.8 libnvinfer.so.7
fi

cd /home/max/MA
source ~/.bashrc
conda activate tf

