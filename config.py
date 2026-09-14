# 8mm Film Enhancement Configuration
# Adjust these settings based on your hardware and quality preferences

import torch

# =============================================================================
# HARDWARE SETTINGS
# =============================================================================
# GPU settings
USE_GPU = True
GPU_ID = 0  # Set to -1 for CPU processing (much slower)
BATCH_SIZE = 1  # Increase if you have more VRAM (RTX 4060 should handle 1-2)

# Memory management
MAX_MEMORY_GB = 8  # Adjust based on your available RAM

# =============================================================================
# DEVICE DETECTION (Intel Arc / NVIDIA / CPU)
# =============================================================================
# Automatically selects the best available compute device.
# Priority: Intel XPU (Arc) > NVIDIA CUDA > CPU.

def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{GPU_ID}")
    return torch.device("cpu")

DEVICE = get_device()

# On Intel Arc: transparently redirect torch.cuda.* helpers used by third-party
# libraries (realesrgan, basicsr, gfpgan) to their torch.xpu.* equivalents so
# they do not fail or silently fall back to CPU.
if DEVICE.type == "xpu":
    torch.cuda.is_available = torch.xpu.is_available
    torch.cuda.device_count = torch.xpu.device_count
    torch.cuda.current_device = torch.xpu.current_device
    torch.cuda.get_device_name = torch.xpu.get_device_name
    torch.cuda.get_device_properties = torch.xpu.get_device_properties
    torch.cuda.empty_cache = torch.xpu.empty_cache
    torch.cuda.memory_reserved = torch.xpu.memory_reserved
    torch.cuda.synchronize = torch.xpu.synchronize

# =============================================================================


# =============================================================================
# VIDEO PROCESSING SETTINGS  
# =============================================================================
# Input settings
INPUT_FPS = 16  # Typical 8mm frame rate (adjust if different)
INPUT_DIR = "input_videos"
OUTPUT_DIR = "output_videos"
TEMP_DIR = "temp"

# Processing pipeline order (recommended)
PROCESSING_STEPS = [
    "stabilization",    # First: stabilize shaky footage
    "upscaling",       # Second: AI upscaling
    "denoising",       # Third: remove noise
    "frame_interpolation",  # Fourth: increase frame rate
    "color_grading"    # Last: manual color correction
]

# =============================================================================
# AI UPSCALING SETTINGS
# =============================================================================
# Model selection
UPSCALE_MODEL = "realesrgan"  # Options: "realesrgan" (fast, per-frame) or "basicvsr" (slow, temporal)
UPSCALE_FACTOR = 2  # 2x or 4x (4x takes much longer)

# Quality vs Speed trade-off
UPSCALE_TILE_SIZE = 2048  # Smaller = less VRAM, larger = faster processing (RTX 4060: 2048 for max GPU usage)
UPSCALE_TILE_PAD = 32
UPSCALE_PRE_PAD = 0

# =============================================================================
# FRAME INTERPOLATION SETTINGS (RIFE)
# =============================================================================
# Target frame rates
TARGET_FPS = 24  # Options: 24, 30, 60 (higher = more processing time)
INTERPOLATION_MODEL = "rife"  # RIFE model for frame interpolation
INTERPOLATION_SCALE = 1.0  # Keep at 1.0 for standard interpolation

# =============================================================================
# DENOISING SETTINGS
# =============================================================================
DENOISE_STRENGTH = 0.7  # 0.0 = no denoising, 1.0 = maximum denoising (0.7 = aggressive grain removal)
DENOISE_MODEL = "basicvsr++"  # Built into BasicVSR++ or separate pass

# =============================================================================
# STABILIZATION SETTINGS
# =============================================================================
STABILIZE_SMOOTHING = 30  # Higher = smoother but may crop more
STABILIZE_MAX_SHIFT = 50  # Maximum pixels to shift for stabilization
STABILIZE_MAX_ANGLE = 5   # Maximum degrees to rotate for stabilization

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================
# Video codec and quality
OUTPUT_CODEC = "libx264"  # H.264 codec (software) or "h264_nvenc" (hardware - much faster)
USE_NVENC = False  # Set to True to use NVIDIA hardware encoding (RTX 4060 supported)
OUTPUT_CRF = 18  # Lower = higher quality, higher file size (18-23 recommended)
OUTPUT_PRESET = "slow"  # Encoding speed vs compression (ultrafast, fast, medium, slow, veryslow)

# File naming
OUTPUT_SUFFIX = "_enhanced"  # Added to original filename
KEEP_INTERMEDIATE_FILES = False  # Set to True for debugging

# =============================================================================
# ADVANCED SETTINGS
# =============================================================================
# Chunk processing for long videos
MAX_CHUNK_DURATION = 300  # Process videos in 5-minute chunks to manage memory
OVERLAP_SECONDS = 2  # Overlap between chunks for seamless stitching

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
SAVE_PROGRESS_IMAGES = True  # Save before/after comparison images

# Performance monitoring
SHOW_PROGRESS_BAR = True
ESTIMATE_PROCESSING_TIME = True

# =============================================================================
# MODEL PATHS (automatically set by setup.py)
# =============================================================================
MODELS_DIR = "models"
REALESRGAN_MODEL_PATH = "models/RealESRGAN_x4plus.pth"
BASICVSR_MODEL_PATH = "models/BasicVSR_REDS4.pth"
RIFE_MODEL_PATH = "models/flownet.pkl"
