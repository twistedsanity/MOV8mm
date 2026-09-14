# ============================================================================
# MOV8mm - 8mm film restoration on Intel Arc GPUs (XPU)
#
# Build:
#   docker build -f Dockerfile.intel -t mov8mm-intel .
#
# Run (Linux hosts with an Arc GPU + i915 kernel driver):
#   docker run --rm -it \
#     --device=/dev/dri --group-add video \
#     -v "$(pwd)/input_videos:/app/input_videos" \
#     -v "$(pwd)/output_videos:/app/output_videos" \
#     mov8mm-intel scripts/enhance_video.py \
#     input_videos/MyFilm.mp4 --steps stabilization denoising upscaling --upscale 2 \
#     -o output_videos/MyFilm_restored.mp4
#
# Windows / Docker Desktop (WSL2 backend): add
#   -v /usr/lib/wsl/lib:/usr/lib/wsl/lib
# and install the Intel WSL2 GPU driver in the host Windows install.
#
# Verify the GPU is visible from inside the container:
#   docker run --rm --device=/dev/dri --group-add video mov8mm-intel -c \
#     "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
#
# Base image ships the oneAPI runtime + Level Zero loader + IPEX.
# ============================================================================
FROM intel/intel-extension-for-pytorch:2.5.10-xpu
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    SYCL_CACHE_PERSISTENT=1 \
    ZE_AFFINITY_MASK=0 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
# System libraries: FFmpeg, OpenCV runtime deps, pymediainfo native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libmediainfo0v5 \
        libzen0v5 \
        wget \
    && rm -rf /var/lib/apt/lists/*
# PyTorch from the official XPU channel - matches the base image's oneAPI
# runtime. Pinned to the same versions as the upstream CUDA build.
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/xpu \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
# Running setup.py here would re-install requirements.txt (CUDA build), so we
# install the Intel requirements directly instead.
COPY requirements-intel.txt /app/requirements-intel.txt
RUN python -m pip install --no-cache-dir -r /app/requirements-intel.txt
# Application code
COPY . /app
# Download models. BasicVSR++ (246MB, super-slow) is skipped; Real-ESRGAN x4/x2
# cover upscaling and GFPGAN covers face restoration.
RUN mkdir -p models input_videos output_videos temp scripts \
    && wget -q -O models/RealESRGAN_x4plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    && wget -q -O models/RealESRGAN_x2plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth \
    && wget -q -O models/GFPGANv1.4.pth \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
# Sanity check the image has XPU-capable torch (no GPU needed at build time;
# actual device visibility is checked at runtime on the host).
RUN python -c "import torch; assert hasattr(torch, 'xpu'), 'torch built without XPU support'; print('torch XPU module OK')"
ENTRYPOINT ["python"]
CMD ["scripts/enhance_video.py", "--help"]
