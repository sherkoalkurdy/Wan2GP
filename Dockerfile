FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Build arg for GPU architectures - specify which CUDA compute capabilities to compile for
# Common values:
#   7.0  - Tesla V100
#   7.5  - RTX 2060, 2070, 2080, Titan RTX
#   8.0  - A100, A800 (Ampere data center)
#   8.6  - RTX 3060, 3070, 3080, 3090 (Ampere consumer)
#   8.9  - RTX 4070, 4080, 4090 (Ada Lovelace)
#   9.0  - H100, H800 (Hopper data center)
#   12.0 - RTX 5070, 5080, 5090 (Blackwell) - Note: sm_120 architecture
#
# Examples:
#   RTX 3060: --build-arg CUDA_ARCHITECTURES="8.6"
#   RTX 4090: --build-arg CUDA_ARCHITECTURES="8.9"
#   Multiple: --build-arg CUDA_ARCHITECTURES="8.0;8.6;8.9"
#
# Note: Including 8.9 or 9.0 may cause compilation issues on some setups
# Default includes 8.0 and 8.6 for broad Ampere compatibility
ARG CUDA_ARCHITECTURES="8.0;8.6"

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && \
    apt install -y \
    python3 python3-pip git wget curl cmake ninja-build \
    libgl1 libglib2.0-0 ffmpeg && \
    apt clean

WORKDIR /workspace

COPY requirements.txt .

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch dependencies individually to avoid timeouts and create cache layers
RUN pip install --no-cache-dir --retries 10 --extra-index-url https://download.pytorch.org/whl/cu124 \
    nvidia-cudnn-cu12==9.1.0.70
RUN pip install --no-cache-dir --retries 10 --extra-index-url https://download.pytorch.org/whl/cu124 \
    nvidia-cublas-cu12==12.4.5.8
RUN pip install --no-cache-dir --retries 10 --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 xformers

# Install requirements (moving largest ones to their own layer if needed, starting with general)
RUN pip install --no-cache-dir --retries 10 -r requirements.txt

RUN useradd -u 1000 -ms /bin/bash user

RUN chown -R user:user /workspace

RUN mkdir /home/user/.cache && \
    chown -R user:user /home/user/.cache

COPY entrypoint.sh /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
