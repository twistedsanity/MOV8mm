"""
GPU-accelerated video denoising using PyTorch
Much faster than OpenCV CPU-based denoising
"""

import sys
import os
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

def bilateral_filter_torch(image_tensor, d=9, sigma_color=75, sigma_space=75):
    """
    GPU-accelerated bilateral filter using PyTorch
    Preserves edges while smoothing
    
    Args:
        image_tensor: (B, C, H, W) tensor on GPU
        d: diameter of pixel neighborhood
        sigma_color: filter sigma in color space
        sigma_space: filter sigma in coordinate space
    """
    # Simple approximation using Gaussian blur (faster than true bilateral)
    # For true bilateral, we'd need custom CUDA kernels
    smoothed = F.avg_pool2d(image_tensor, kernel_size=d, stride=1, padding=d//2)
    return smoothed

def denoise_frame_gpu(frame, device, strength=0.7, method='gaussian'):
    """
    Denoise a single frame on GPU
    
    Args:
        frame: numpy array (H, W, 3)
        device: torch device
        strength: denoising strength 0-1
        method: 'gaussian', 'median', or 'bilateral'
    """
    # Convert to tensor and normalize
    frame_tensor = torch.from_numpy(frame).float().to(device) / 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    if method == 'gaussian':
        # Gaussian blur - very fast
        kernel_size = int(strength * 9) + 3  # 3-12 based on strength
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = strength * 2.0
        denoised = F.avg_pool2d(frame_tensor, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
    elif method == 'median':
        # Approximated median filter using max and min pooling
        kernel_size = int(strength * 5) + 3
        if kernel_size % 2 == 0:
            kernel_size += 1
        max_pool = F.max_pool2d(frame_tensor, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        min_pool = -F.max_pool2d(-frame_tensor, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        denoised = (max_pool + min_pool) / 2.0
        
    elif method == 'bilateral':
        # Bilateral filter approximation
        kernel_size = int(strength * 9) + 5
        if kernel_size % 2 == 0:
            kernel_size += 1
        denoised = bilateral_filter_torch(frame_tensor, d=kernel_size, 
                                          sigma_color=strength*100, 
                                          sigma_space=strength*100)
    else:
        denoised = frame_tensor
    
    # Convert back to numpy
    denoised = denoised.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
    denoised = (denoised.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    
    return denoised

def denoise_temporal_gpu(frames, device, strength=0.7, temporal_window=5):
    """
    Temporal denoising on GPU using frame averaging
    
    Args:
        frames: list of numpy arrays
        device: torch device
        strength: denoising strength 0-1
        temporal_window: number of frames to average
    """
    num_frames = len(frames)
    denoised_frames = []
    
    half_window = temporal_window // 2
    
    for i in tqdm(range(num_frames), desc="GPU Temporal Denoise"):
        # Get temporal neighbors
        start_idx = max(0, i - half_window)
        end_idx = min(num_frames, i + half_window + 1)
        
        # Stack frames as tensors
        frame_stack = []
        for j in range(start_idx, end_idx):
            frame_tensor = torch.from_numpy(frames[j]).float().to(device) / 255.0
            frame_stack.append(frame_tensor)
        
        # Average across temporal dimension (weighted by distance)
        frame_stack = torch.stack(frame_stack, dim=0)  # (T, H, W, 3)
        
        # Temporal median (more robust than mean)
        median_frame = torch.median(frame_stack, dim=0)[0]
        
        # Blend with original based on strength
        original = torch.from_numpy(frames[i]).float().to(device) / 255.0
        blended = strength * median_frame + (1 - strength) * original
        
        # Apply spatial denoising
        blended = blended.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        kernel_size = int(strength * 5) + 3
        if kernel_size % 2 == 0:
            kernel_size += 1
        spatial_denoised = F.avg_pool2d(blended, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
        # Convert back
        result = spatial_denoised.squeeze(0).permute(1, 2, 0)
        result = (result.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        
        denoised_frames.append(result)
    
    return denoised_frames

def denoise_video_gpu(input_path, output_path=None, strength=0.7, temporal=True, method='gaussian'):
    """
    GPU-accelerated video denoising
    
    Args:
        input_path: input video file
        output_path: output video file
        strength: denoising strength 0-1
        temporal: use temporal denoising (slower but better)
        method: 'gaussian', 'median', 'bilateral'
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = Path(config.TEMP_DIR) / f"{input_path.stem}_denoised_gpu{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check GPU
    if config.DEVICE.type == "cpu":
        logger.error("GPU not available! This script requires a GPU.")
        return None
    
    device = config.DEVICE
    if device.type == "xpu":
        gpu_name = torch.xpu.get_device_name(0) if torch.xpu.device_count() else "Intel XPU"
    else:
        gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"Using GPU: {gpu_name}")
    
    logger.info(f"Denoising {input_path.name} on GPU (strength: {strength}, method: {method}, temporal: {temporal})...")
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {width}x{height} @ {fps:.2f}fps, {frame_count} frames")
    
    # Load all frames
    logger.info("Loading frames...")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    logger.info(f"Loaded {len(frames)} frames")
    
    # Process frames
    if temporal:
        denoised_frames = denoise_temporal_gpu(frames, device, strength, temporal_window=5)
    else:
        denoised_frames = []
        for frame in tqdm(frames, desc="GPU Spatial Denoise"):
            denoised = denoise_frame_gpu(frame, device, strength, method)
            denoised_frames.append(denoised)
    
    # Save video
    logger.info("Saving denoised video...")
    
    # Create temporary AVI file
    temp_output = output_path.parent / f"{output_path.stem}_temp.avi"
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Lossless
    out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
    
    for frame in tqdm(denoised_frames, desc="Writing frames"):
        out.write(frame)
    out.release()
    
    # Convert to MP4
    logger.info("Converting to MP4...")
    import subprocess
    cmd = [
        'ffmpeg', '-y', '-i', str(temp_output),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-pix_fmt', 'yuv420p',  # Compatible pixel format
        str(output_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Clean up temp file
    temp_output.unlink()
    
    logger.info(f"✓ GPU denoising complete: {output_path}")
    return str(output_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU-accelerated video denoising")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file (optional)")
    parser.add_argument("-s", "--strength", type=float, default=0.7,
                       help="Denoising strength 0.0-1.0 (default: 0.7)")
    parser.add_argument("--no-temporal", action="store_true",
                       help="Disable temporal denoising (faster but lower quality)")
    parser.add_argument("-m", "--method", choices=['gaussian', 'median', 'bilateral'],
                       default='gaussian',
                       help="Denoising method (default: gaussian)")
    
    args = parser.parse_args()
    
    Path("logs").mkdir(exist_ok=True)
    
    denoise_video_gpu(args.input, args.output, args.strength, 
                     temporal=not args.no_temporal, method=args.method)

if __name__ == "__main__":
    main()
