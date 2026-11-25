FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG MAX_JOBS=4
ARG TRELLIS2_REPO=https://github.com/AmrNabil12/TRELLIS.2.git
ARG TRELLIS2_BRANCH=main

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    MAX_JOBS=${MAX_JOBS}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    ca-certificates \
    build-essential \
    ninja-build \
    cmake \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    libgl1 \
    libglib2.0-0 \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libgomp1 \
    ffmpeg \
    openssh-server \
    && mkdir -p /run/sshd \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel packaging ninja

# Install PyTorch 2.6.0 + CUDA 12.4
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install core Python & FastAPI dependencies
# transformers is pinned to >=4.56.0,<5.0.0: DINOv3ViTModel requires >=4.56.0 while briaai/RMBG-2.0 breaks on >=5.0.0
RUN pip install \
    imageio \
    imageio-ffmpeg \
    tqdm \
    easydict \
    opencv-python-headless \
    trimesh \
    "transformers>=4.56.0,<5.0.0" \
    pandas \
    lpips \
    zstandard \
    kornia \
    timm \
    einops \
    scipy \
    safetensors \
    huggingface-hub \
    hf_transfer \
    pillow \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pydantic && \
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Install prebuilt flash-attn 2.7.3 wheel matching PyTorch's C++11 ABI
RUN ABI=$(python -c "import torch; print('TRUE' if torch.compiled_with_cxx11_abi() else 'FALSE')") && \
    pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3%2Bcu12torch2.6cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl"

# Compile & install CUDA extensions in cached layers
RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && \
    pip install /tmp/nvdiffrast --no-build-isolation && \
    rm -rf /tmp/nvdiffrast

# nvdiffrec links against -lcuda; expose CUDA stubs only during this build layer
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/nvdiffrec && \
    LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}" pip install /tmp/nvdiffrec --no-build-isolation && \
    rm -rf /tmp/nvdiffrec

RUN git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh && \
    pip install /tmp/CuMesh --no-build-isolation && \
    rm -rf /tmp/CuMesh

RUN git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM && \
    pip install /tmp/FlexGEMM --no-build-isolation && \
    rm -rf /tmp/FlexGEMM

# Clone TRELLIS.2 fork outside /workspace (RunPod mounts persistent volume on /workspace) and build o-voxel
RUN git clone --recursive -b "${TRELLIS2_BRANCH}" "${TRELLIS2_REPO}" /opt/TRELLIS.2 && \
    cp -r /opt/TRELLIS.2/o-voxel /tmp/o-voxel && \
    pip install /tmp/o-voxel --no-build-isolation && \
    rm -rf /tmp/o-voxel

# CPU-safe sanity check (avoid importing CUDA-initializing extensions at build time)
RUN python -c "import fastapi, uvicorn, trimesh, cv2, PIL; from transformers import DINOv3ViTModel"

WORKDIR /app
COPY server.py download_weights.py start.sh /app/
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

ENV PYTHONPATH=/opt/TRELLIS.2:/app \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8 \
    FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRELLIS2_MODEL_ID=microsoft/TRELLIS.2-4B \
    TRELLIS2_LOW_VRAM=auto \
    TRELLIS2_FAST_VRAM=1 \
    PORT=8000 \
    PORT_HEALTH=8000

EXPOSE 8000

CMD ["/app/start.sh"]
