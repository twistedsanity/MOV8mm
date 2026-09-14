"""
AI upscaling module using Real-ESRGAN
Upscales video resolution by 2x or 4x
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import torch
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
import config

def upscale(input_path, output_path=None, scale=None, model=None):
    """
    Upscale video using AI models (Real-ESRGAN or BasicVSR++)
    
    Args:
        input_path: Path to input video
        output_path: Path to save upscaled video (optional)
        scale: Upscale factor (2 or 4, optional - uses config if not specified)
        model: Model to use ("realesrgan" or "basicvsr", optional - uses config if not specified)
    
    Returns:
        Path to upscaled video
    """
    input_path = Path(input_path)
    scale = scale or config.UPSCALE_FACTOR
    model = model or config.UPSCALE_MODEL
    
    if output_path is None:
        output_path = Path(config.TEMP_DIR) / f"{input_path.stem}_upscaled_{scale}x{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Upscaling {input_path.name} by {scale}x...")
    logger.info(f"Model: {model}")
    
    # Choose upscaling method based on model
    if model == "basicvsr":
        return upscale_basicvsr(input_path, output_path, scale)
    else:  # default to realesrgan
        return upscale_realesrgan(input_path, output_path, scale)

def upscale_realesrgan(input_path, output_path, scale):
    """
    Upscale video using Real-ESRGAN (per-frame processing)
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        scale: Upscale factor (2 or 4)
    
    Returns:
        Path to upscaled video
"""

# Check GPU availability
if torch.cuda.is_available():
if config.DEVICE.type == "xpu":
    logger.info(f"✓ Intel XPU available: {torch.xpu.get_device_name(0)}")
elif torch.cuda.is_available():
    logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    logger.info(f"  CUDA version: {torch.version.cuda}")
    logger.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_path = Path(config.MODELS_DIR) / 'RealESRGAN_x4plus.pth'
        else:  # 2x
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            model_path = Path(config.MODELS_DIR) / 'RealESRGAN_x2plus.pth'
        
        # Move model to GPU if available
        if torch.cuda.is_available() and config.USE_GPU:
            model = model.cuda()
            logger.info("✓ Model moved to GPU")
        if config.DEVICE.type != "cpu":
        model = model.to(config.DEVICE)
        logger.info(f"✓ Model moved to {config.DEVICE}")
            
        # Check if model exists
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")

            logger.info("https://github.com/xinntao/Real-ESRGAN/releases")
            return input_path
        
        # Initialize upsampler
        device = 'cuda' if torch.cuda.is_available() and config.USE_GPU else 'cpu'
        device = config.DEVICE
        half_enabled = device.type == "cuda"
        logger.info(f"Using device: {device}")
        
        upsampler = RealESRGANer(
            scale=scale,
            model_path=str(model_path),
            model=model,
            tile=config.UPSCALE_TILE_SIZE,
            tile_pad=config.UPSCALE_TILE_PAD,
            pre_pad=config.UPSCALE_PRE_PAD,
            half=True if device == 'cuda' else False,
            gpu_id=config.GPU_ID if device == 'cuda' else None,
            half=half_enabled,
            gpu_id=config.GPU_ID if device.type == 'cuda' else None,
            device=device
        )
        
        logger.info(f"Model loaded: {model_path.name}")
        logger.info(f"GPU ID: {config.GPU_ID}, Half precision: {device == 'cuda'}")
        
        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output dimensions
        out_width = width * scale
        out_height = height * scale
        
        # Create temporary output using cv2 (more reliable)
        temp_output = output_path.parent / f"{output_path.stem}_temp.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(temp_output), fourcc, fps, (out_width, out_height))
        
        logger.info(f"Processing {frame_count} frames: {width}x{height} -> {out_width}x{out_height}")
        logger.info(f"Writing to temporary AVI, will convert to H.264...")
        
        # Process frames
        pbar = tqdm(total=frame_count, desc="Upscaling", disable=not config.SHOW_PROGRESS_BAR)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Upscale frame
            try:
                output_frame, _ = upsampler.enhance(frame, outscale=scale)
                out.write(output_frame)
            except Exception as e:
                logger.error(f"Error processing frame {frame_idx}: {e}")
                # Write original frame resized as fallback
                fallback = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
                out.write(fallback)
            
            frame_idx += 1
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
            logger.info(f"✓ Upscaling complete: {output_path}")
        else:
            logger.error(f"FFmpeg conversion failed: {result.stderr}")
            raise RuntimeError(f"Failed to convert video: {result.stderr}")
        
        return output_path
        
    except ImportError as e:
        logger.error(f"Required package not installed: {e}")
        logger.info("Install with: pip install realesrgan basicsr")
        return input_path
    except Exception as e:
        logger.error(f"Upscaling failed: {e}")
        return input_path

def upscale_basicvsr(input_path, output_path, scale):
    """
    Upscale video using BasicVSR++ (temporal processing with multiple frames)
    
    NOTE: BasicVSR++ requires additional dependencies not included in basicsr.
    For best results with 8mm film restoration, use Real-ESRGAN instead.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video  
        scale: Upscale factor (2 or 4)
    
    Returns:
        Path to upscaled video
    """
    logger.error("BasicVSR++ is not available in the installed version of basicsr")
    logger.info("")
    logger.info("BasicVSR++ requires MMEditing framework which has complex dependencies.")
    logger.info("For 8mm film restoration, Real-ESRGAN provides excellent results:")
    logger.info("  • Fast processing (~4 sec/frame with GPU)")
    logger.info("  • High quality upscaling (optimized for old film)")
    logger.info("  • Stable and well-tested")
    logger.info("")
    logger.info("Use Real-ESRGAN instead:")
    logger.info("  python scripts/enhance_video.py Movie0030.MP4 --cut 170 175 --steps upscaling --upscale 2 --nvenc")
    logger.info("  (Note: --model realesrgan is the default, no need to specify)")
    logger.info("")
    
    # Alternative: If you really need BasicVSR++, you would need to:
    # 1. Install MMEditing framework (pip install mmedit)
    # 2. Use their inference scripts
    # But for 8mm film, Real-ESRGAN is the recommended and proven choice.
    
    return input_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Upscale video using Real-ESRGAN")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file (optional)")
    parser.add_argument("-s", "--scale", type=int, choices=[2, 4], default=config.UPSCALE_FACTOR,
                       help=f"Upscale factor (default: {config.UPSCALE_FACTOR})")
    
    args = parser.parse_args()
    
    Path("logs").mkdir(exist_ok=True)
    upscale(args.input, args.output, args.scale)

if __name__ == "__main__":
    main()
