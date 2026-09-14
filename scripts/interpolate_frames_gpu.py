"""
Professional frame interpolation using PyTorch
Fast GPU-accelerated optical flow-based interpolation
Alternative to FILM that works with existing PyTorch/CUDA setup
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

def compute_optical_flow_gpu(frame1, frame2):
    """
    Compute optical flow between frames using OpenCV's DISFlow (GPU-friendly)
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Use DIS optical flow (faster than Farneback)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(gray1, gray2, None)
    
    return flow

def warp_frame_torch(frame, flow, t, device):
    """
    Warp frame using optical flow on GPU with PyTorch
    
    Args:
        frame: numpy array (H, W, 3)
        flow: numpy array (H, W, 2) - optical flow
        t: float - interpolation factor (0 to 1)
        device: torch device
    """
    h, w = frame.shape[:2]
    
    # Convert to torch tensors
    frame_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    flow_t = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float().to(device) * t
    
    # Create normalized grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing='ij'
    )
    
    # Apply flow (need to normalize flow to [-1, 1] range)
    flow_x_norm = 2.0 * flow_t[0, 0] / (w - 1)
    flow_y_norm = 2.0 * flow_t[0, 1] / (h - 1)
    
    grid = torch.stack([
        grid_x + flow_x_norm,
        grid_y + flow_y_norm
    ], dim=2).unsqueeze(0)
    
    # Warp using grid_sample
    warped = F.grid_sample(frame_t, grid, mode='bilinear', 
                          padding_mode='border', align_corners=True)
    
    # Convert back to numpy
    warped = warped.squeeze(0).permute(1, 2, 0).cpu().numpy()
    warped = (warped * 255).clip(0, 255).astype(np.uint8)
    
    return warped

def interpolate_pair_gpu(frame1, frame2, num_frames, device):
    """
    Interpolate multiple frames between two frames using GPU
    
    Args:
        frame1, frame2: Input frames
        num_frames: Number of intermediate frames
        device: torch device
        
    Returns:
        List of interpolated frames (excluding frame1 and frame2)
    """
    # Compute bidirectional optical flow
    flow_forward = compute_optical_flow_gpu(frame1, frame2)
    flow_backward = compute_optical_flow_gpu(frame2, frame1)
    
    interpolated = []
    for i in range(1, num_frames + 1):
        t = i / (num_frames + 1)
        
        # Bidirectional warping
        warp1 = warp_frame_torch(frame1, flow_forward, t, device)
        warp2 = warp_frame_torch(frame2, flow_backward, 1 - t, device)
        
        # Adaptive blending based on interpolation position
        alpha = t
        blended = cv2.addWeighted(warp1, 1 - alpha, warp2, alpha, 0)
        
        interpolated.append(blended)
    
    return interpolated

def interpolate_video_gpu(input_path, output_path=None, target_fps=60):
    """
    Interpolate video to higher framerate using GPU
    
    Args:
        input_path: Input video file
        output_path: Output video file
        target_fps: Target framerate (default: 60)
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = Path(config.TEMP_DIR) / f"{input_path.stem}_interpolated_{target_fps}fps{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check GPU
    if config.DEVICE.type == "cpu":
        logger.error("GPU not available! This script works best with a GPU.")
        logger.info("Will use CPU but it will be slower...")
        device = torch.device("cpu")
    else:
        device = config.DEVICE
        if device.type == "xpu":
            gpu_name = torch.xpu.get_device_name(0) if torch.xpu.device_count() else "Intel XPU"
    else:
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Input: {width}x{height} @ {source_fps:.2f}fps, {frame_count} frames")
    logger.info(f"Target: {target_fps}fps")
    
    # Calculate how many frames to insert
    fps_ratio = target_fps / source_fps
    num_intermediate = int(round(fps_ratio)) - 1
    
    logger.info(f"Inserting {num_intermediate} frame(s) between each original frame")
    estimated_frames = frame_count * (num_intermediate + 1)
    logger.info(f"Output will have ~{estimated_frames} frames")
    
    # Load frames
    logger.info("Loading frames...")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    logger.info(f"Loaded {len(frames)} frames")
    
    # Interpolate
    output_frames = []
    
    for i in tqdm(range(len(frames) - 1), desc="GPU Frame Interpolation"):
        # Add original frame
        output_frames.append(frames[i])
        
        # Interpolate between frames[i] and frames[i+1]
        if num_intermediate > 0:
            interpolated = interpolate_pair_gpu(
                frames[i], frames[i + 1], 
                num_intermediate, device
            )
            output_frames.extend(interpolated)
    
    # Add last frame
    output_frames.append(frames[-1])
    
    logger.info(f"Created {len(output_frames)} frames")
    
    # Save video
    logger.info("Saving interpolated video...")
    
    # Write to temp AVI first (lossless)
    temp_output = output_path.parent / f"{output_path.stem}_temp.avi"
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out = cv2.VideoWriter(str(temp_output), fourcc, target_fps, (width, height))
    
    for frame in tqdm(output_frames, desc="Writing frames"):
        out.write(frame)
    out.release()
    
    # Convert to MP4 with proper framerate
    logger.info("Converting to MP4...")
    import subprocess
    cmd = [
        'ffmpeg', '-y', '-i', str(temp_output),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-r', str(target_fps),
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True)
    
    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr.decode()}")
        return None
    
    # Clean up
    temp_output.unlink()
    
    duration = len(output_frames) / target_fps
    logger.info(f"✓ Frame interpolation complete: {output_path}")
    logger.info(f"  Output: {len(output_frames)} frames @ {target_fps}fps ({duration:.1f}s)")
    
    return str(output_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU-accelerated frame interpolation")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file")
    parser.add_argument("--fps", type=int, default=60,
                       help="Target framerate (default: 60)")
    
    args = parser.parse_args()
    
    Path("logs").mkdir(exist_ok=True)
    
    interpolate_video_gpu(args.input, args.output, args.fps)

if __name__ == "__main__":
    main()
