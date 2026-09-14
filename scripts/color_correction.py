"""
Color correction and lighting enhancement for 8mm film footage
Includes CLAHE, white balance, color grading, and gamma correction
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

def apply_clahe(frame, clip_limit=2.0, tile_size=8):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Enhances local contrast while preventing over-amplification
    
    Args:
        frame: Input BGR frame
        clip_limit: Threshold for contrast limiting (1.0-4.0, default 2.0)
        tile_size: Grid size for histogram equalization (default 8)
    
    Returns:
        Enhanced frame
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l)
    
    # Merge channels and convert back
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr

def auto_white_balance(frame, method='gray_world'):
    """
    Automatic white balance correction
    
    Args:
        frame: Input BGR frame
        method: 'gray_world' or 'white_patch'
    
    Returns:
        Color-corrected frame
    """
    if method == 'gray_world':
        # Gray World assumption: average color should be gray
        result = frame.copy().astype(np.float32)
        
        # Calculate average for each channel
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        
        # Calculate overall average
        avg = (avg_b + avg_g + avg_r) / 3.0
        
        # Scale each channel
        result[:, :, 0] = np.clip(result[:, :, 0] * (avg / avg_b), 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * (avg / avg_g), 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * (avg / avg_r), 0, 255)
        
        return result.astype(np.uint8)
    
    elif method == 'white_patch':
        # White Patch assumption: brightest area should be white
        result = frame.copy().astype(np.float32)
        
        # Find maximum for each channel
        max_b = np.max(result[:, :, 0])
        max_g = np.max(result[:, :, 1])
        max_r = np.max(result[:, :, 2])
        
        # Scale to make maximum = 255
        if max_b > 0:
            result[:, :, 0] = np.clip(result[:, :, 0] * (255.0 / max_b), 0, 255)
        if max_g > 0:
            result[:, :, 1] = np.clip(result[:, :, 1] * (255.0 / max_g), 0, 255)
        if max_r > 0:
            result[:, :, 2] = np.clip(result[:, :, 2] * (255.0 / max_r), 0, 255)
        
        return result.astype(np.uint8)
    
    return frame

def adjust_gamma(frame, gamma=1.0):
    """
    Apply gamma correction
    
    Args:
        frame: Input BGR frame
        gamma: Gamma value (< 1.0 = brighter, > 1.0 = darker, default 1.0)
    
    Returns:
        Gamma-corrected frame
    """
    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                     for i in range(256)]).astype(np.uint8)
    
    # Apply gamma correction
    return cv2.LUT(frame, table)

def enhance_saturation(frame, saturation_factor=1.2):
    """
    Enhance color saturation
    
    Args:
        frame: Input BGR frame
        saturation_factor: Multiplier for saturation (1.0 = no change, > 1.0 = more saturated)
    
    Returns:
        Saturated frame
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Enhance saturation channel
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    
    # Convert back
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_shadows_highlights(frame, shadows=0.0, highlights=0.0, midtones=0.0):
    """
    Adjust shadows, midtones, and highlights separately
    
    Args:
        frame: Input BGR frame
        shadows: Adjustment for dark areas (-1.0 to 1.0)
        highlights: Adjustment for bright areas (-1.0 to 1.0)
        midtones: Adjustment for mid-tones (-1.0 to 1.0)
    
    Returns:
        Adjusted frame
    """
    frame_float = frame.astype(np.float32) / 255.0
    
    # Create luminosity mask
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # Masks for different tonal ranges
    shadow_mask = np.maximum(0, 1.0 - gray * 2.0)  # 0-0.5 range
    highlight_mask = np.maximum(0, gray * 2.0 - 1.0)  # 0.5-1.0 range
    midtone_mask = 1.0 - shadow_mask - highlight_mask
    
    # Expand masks to 3 channels
    shadow_mask_3ch = np.stack([shadow_mask] * 3, axis=2)
    highlight_mask_3ch = np.stack([highlight_mask] * 3, axis=2)
    midtone_mask_3ch = np.stack([midtone_mask] * 3, axis=2)
    
    # Apply adjustments
    result = frame_float.copy()
    result += shadow_mask_3ch * shadows * 0.3
    result += highlight_mask_3ch * highlights * 0.3
    result += midtone_mask_3ch * midtones * 0.3
    
    # Clip and convert back
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result

def color_correction_gpu(frame, device, clahe=True, white_balance=True, gamma=1.0, 
                        saturation=1.0, shadows=0.0, highlights=0.0):
    """
    Apply comprehensive color correction on GPU using PyTorch
    
    Args:
        frame: Input BGR frame (numpy array)
        device: torch device
        clahe: Apply CLAHE contrast enhancement
        white_balance: Apply auto white balance
        gamma: Gamma correction value
        saturation: Saturation multiplier
        shadows: Shadow adjustment (-1 to 1)
        highlights: Highlight adjustment (-1 to 1)
    
    Returns:
        Color-corrected frame
    """
    result = frame.copy()
    
    # CPU-based corrections (OpenCV is optimized for these)
    if clahe:
        result = apply_clahe(result, clip_limit=2.0, tile_size=8)
    
    if white_balance:
        result = auto_white_balance(result, method='gray_world')
    
    if gamma != 1.0:
        result = adjust_gamma(result, gamma)
    
    if saturation != 1.0:
        result = enhance_saturation(result, saturation)
    
    if shadows != 0.0 or highlights != 0.0:
        result = adjust_shadows_highlights(result, shadows, highlights, midtones=0.0)
    
    return result

def correct_video_colors(input_path, output_path=None, 
                        clahe=True, white_balance=True, gamma=1.0,
                        saturation=1.2, shadows=0.0, highlights=0.0):
    """
    Apply color correction to entire video
    
    Args:
        input_path: Input video file
        output_path: Output video file (optional)
        clahe: Enable CLAHE contrast enhancement
        white_balance: Enable auto white balance
        gamma: Gamma correction (0.5-2.0, default 1.0)
        saturation: Saturation boost (0.5-2.0, default 1.2)
        shadows: Lift shadows (-1 to 1, default 0)
        highlights: Adjust highlights (-1 to 1, default 0)
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = Path(config.TEMP_DIR) / f"{input_path.stem}_color_corrected{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check GPU
    if config.DEVICE.type == "xpu":
        device = config.DEVICE
        gpu_name = torch.xpu.get_device_name(0) if torch.xpu.device_count() else "Intel XPU"
        logger.info(f"Using GPU: {gpu_name}")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{config.GPU_ID}")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        logger.info("GPU not available, using CPU")
    
    logger.info(f"Color correction settings:")
    logger.info(f"  CLAHE: {clahe}")
    logger.info(f"  White Balance: {white_balance}")
    logger.info(f"  Gamma: {gamma}")
    logger.info(f"  Saturation: {saturation}")
    logger.info(f"  Shadows: {shadows:+.2f}, Highlights: {highlights:+.2f}")
    
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
    corrected_frames = []
    for frame in tqdm(frames, desc="Color Correction"):
        corrected = color_correction_gpu(frame, device, clahe, white_balance, 
                                        gamma, saturation, shadows, highlights)
        corrected_frames.append(corrected)
    
    # Save video
    logger.info("Saving color-corrected video...")
    
    # Create temporary AVI file
    temp_output = output_path.parent / f"{output_path.stem}_temp.avi"
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Lossless
    out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
    
    for frame in tqdm(corrected_frames, desc="Writing frames"):
        out.write(frame)
    out.release()
    
    # Convert to MP4
    logger.info("Converting to MP4...")
    import subprocess
    cmd = [
        'ffmpeg', '-y', '-i', str(temp_output),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Clean up temp file
    temp_output.unlink()
    
    logger.info(f"✓ Color correction complete: {output_path}")
    return str(output_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Color correction for 8mm film")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file (optional)")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE contrast enhancement")
    parser.add_argument("--no-wb", action="store_true", help="Disable white balance")
    parser.add_argument("-g", "--gamma", type=float, default=1.0,
                       help="Gamma correction (0.5-2.0, default 1.0)")
    parser.add_argument("-s", "--saturation", type=float, default=1.2,
                       help="Saturation multiplier (0.5-2.0, default 1.2)")
    parser.add_argument("--shadows", type=float, default=0.0,
                       help="Lift shadows (-1.0 to 1.0, default 0)")
    parser.add_argument("--highlights", type=float, default=0.0,
                       help="Adjust highlights (-1.0 to 1.0, default 0)")
    
    args = parser.parse_args()
    
    Path("logs").mkdir(exist_ok=True)
    
    correct_video_colors(
        args.input, 
        args.output,
        clahe=not args.no_clahe,
        white_balance=not args.no_wb,
        gamma=args.gamma,
        saturation=args.saturation,
        shadows=args.shadows,
        highlights=args.highlights
    )

if __name__ == "__main__":
    main()
