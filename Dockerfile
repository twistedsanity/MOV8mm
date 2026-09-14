# Intel XPU image: Ubuntu 22.04 + oneAPI runtime + Level Zero loader baked in
FROM intel/intel-extension-for-pytorch:2.5.10-xpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    SYCL_CACHE_PERSISTENT=1 \
    ZE_AFFINITY_MASK=0

RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/xpu \
 && pip install --no-cache-dir -r requirements-intel.txt

WORKDIR /app
COPY . .
RUN echo "n" | python setup.py \
 && wget -q -O models/GFPGANv1.4.pth \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth

CMD ["bash"]
