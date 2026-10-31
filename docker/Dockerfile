# DataMate Zalo AI Challenge 2026 - Dockerfile
# Optimized for 2x NVIDIA RTX 5090 (Blackwell architecture, sm_120, CUDA 12.8+)

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies & Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    curl \
    git \
    sqlite3 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set workspace directory
WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Attempt to install vLLM for high-throughput multi-GPU serving
# (Fall back to transformers automatically if platform-specific wheel unavailable)
RUN pip install --no-cache-dir vllm || echo "vLLM optional installation skipped, transformers will be used"

# Copy source code and resources
COPY src /app/src
COPY prompts /app/prompts
COPY docker/predict.sh /app/predict.sh

# Make entrypoint script executable
RUN chmod +x /app/predict.sh

# Default mounts:
# /data - input directory (read-only)
# /result - output directory (writable)
# /app/models - pre-downloaded model weights
VOLUME ["/data", "/result"]

# Entrypoint per competition specification
ENTRYPOINT ["/bin/bash", "/app/predict.sh"]
