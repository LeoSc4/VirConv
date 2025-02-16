# Use the official PyTorch image with CUDA 11.1 and cuDNN 8 in DEVELOPMENT mode to include nvcc 
# Python version 3.8.10
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

WORKDIR /workspace

# Request the NVIDIA public key from NVIDIA's package repository to verify the authenticity of the packages
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub



RUN apt-get update && apt-get install -y git

# Install the cv2 dependencies for the train.py script in tools
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
ARG CACHE_BUST=1


# Install Python dependencies 
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt --use-feature=2020-resolver

RUN git config --global --add safe.directory /workspace


CMD ["bash"]