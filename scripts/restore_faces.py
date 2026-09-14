"""
Face restoration module using GFPGAN
Enhances faces in old 8mm home movies
"""

import sys
from pathlib import Path
import cv2
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
import config

def restore_faces(input_path, output_path=None):
    """
    Restore and enhance faces using GFPGAN
    Perfect for 8mm home movies with people
    
    Args:
        input_path: Path to input video
        output_path: Path to save enhanced video (optional)
    
    Returns:
        Path to face-restored video
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = Path(config.TEMP_DIR) / f"{input_path.stem}_faces{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check GPU availability
    use_gpu = config.USE_GPU and (config.DEVICE.type != "cpu")
    device = config.DEVICE if use_gpu else torch.device("cpu")
    
    logger.info(f"Face restoration on {input_path.name} using device: {device}...")
    
    # Import GFPGAN
    try:
        from gfpgan import GFPGANer
    except ImportError:
        logger.error("GFPGAN not installed! Run: pip install gfpgan")
        raise
    
    # Download model if needed
    model_path = Path(config.MODELS_DIR) / "GFPGANv1.4.pth"
    if not model_path.exists():
        logger.info("Downloading GFPGAN model (first time only)...")
        import urllib.request
        model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(model_url, model_path)
        logger.info("Model downloaded successfully!")
    
    # Initialize GFPGAN
    logger.info(f"Loading GFPGAN model: {model_path.name}")
    restorer = GFPGANer(
        model_path=str(model_path),
        upscale=1,  # Don't upscale, just restore faces
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,  # Don't upscale background
        device=device
    )
    
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create temporary output using cv2 (more reliable)
    temp_output = output_path.parent / f"{output_path.stem}_temp.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
    
    logger.info(f"Processing {frame_count} frames with GFPGAN face restoration...")
    pbar = tqdm(total=frame_count, desc="Face Restoration", disable=not config.SHOW_PROGRESS_BAR)
    
    # Process frames
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Restore faces in frame
        # GFPGAN expects BGR input (OpenCV format)
        try:
            _, _, restored_frame = restorer.enhance(
                frame,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.5  # Blend restored faces with original (0.5 = 50% blend)
            )
            out.write(restored_frame)
        except Exception as e:
            # If face restoration fails (no faces detected), use original frame
            logger.debug(f"Frame {i}: No faces detected or error: {e}")
            out.write(frame)
        
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    # Convert to H.264 MP4 using FFmpeg command-line (more reliable)
    logger.info("Converting to H.264...")
    import subprocess
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', str(temp_output),
        '-c:v', 'libx264', '-crf', str(config.OUTPUT_CRF),
        '-preset', config.OUTPUT_PRESET, '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(output_path)
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        temp_output.unlink()  # Delete temp file
        logger.info(f"✓ Face restoration complete: {output_path}")
    else:
        logger.error(f"FFmpeg conversion failed: {result.stderr}")
        raise RuntimeError(f"Failed to convert video: {result.stderr}")
    
    return output_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python restore_faces.py <input_video>")
        sys.exit(1)
    
    input_video = sys.argv[1]
    restore_faces(input_video)
