# Base image with CUDA 11.7, cuDNN, Ubuntu 20.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

# Install basic tools, Python 3.8
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3-pip python3-distutils git curl \
    ffmpeg libsm6 libxext6 \
    libxcb-xinerama0 libxcb-xinerama0-dev libxcb1 libx11-xcb1 libglu1-mesa \
    libxrender1 libxi6 libxcomposite1 libxcursor1 libxrandr2 libxinerama1 \
    libxss1 libglib2.0-0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-render-util0 libxcb-xkb1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Pin typing-extensions to Python 3.8 compatible version
RUN pip install --no-cache-dir typing-extensions==3.10.0.2

# Install PyTorch 1.12.1 + CUDA 11.7
RUN pip3 install --no-cache-dir torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117

# Install Waymo Open Dataset (TF 2.5)
RUN pip install --no-cache-dir waymo-open-dataset-tf-2-5-0

# Install other dependencies from requirements.txt
COPY requirements.txt /workspace/
# Remove waymo-open-dataset-tf-2-5-0 from requirements.txt if present
RUN sed -i '/waymo-open-dataset-tf-2-5-0/d' requirements.txt \
 && pip install --no-cache-dir -r requirements.txt

# Ensure git works safely in mounted volumes
RUN git config --global --add safe.directory /workspace

CMD ["bash"]
