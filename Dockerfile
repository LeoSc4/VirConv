# Base image from nvidia with cuda 11.7.1 in development mode to include nvcc as part of NVIDIA Toolkit
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

WORKDIR /workspace

# Request the NVIDIA public key from NVIDIA's package repository to verify the authenticity of the packages
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub


# Install basic packages and Python
RUN apt-get update && apt-get install -y \
python3 python3-pip python3-dev git \
&& rm -rf /var/lib/apt/lists/*

# Set Python3 as default 
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install PyTorch 1.13.1 mit CUDA 11.7
RUN pip3 install --no-cache-dir torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117


# Set non-interactive mode to avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

ARG CACHE_BUST=1
# Install dependencies
COPY requirements.txt /workspace/
# without --use-feature=2020-resolver as newer pip versions use it per default
RUN pip install --user -r requirements.txt 

# Resolve ownership conflicts when using mounted volumes and user permissions
RUN git config --global --add safe.directory /workspace


CMD ["bash"]