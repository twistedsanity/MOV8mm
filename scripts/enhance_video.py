"""
Main video enhancement script for 8mm film footage
Combines stabilization, upscaling, denoising, and frame interpolation
"""

import sys
import os
from pathlib import Path
import argparse
import time
import subprocess
from loguru import logger
import torch

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
import config

# Configure logger
logger.remove()
logger.add(sys.stderr, level=config.LOG_LEVEL)
logger.add("logs/enhancement_{time}.log", rotation="100 MB")

def check_gpu():
    """Check if GPU is available and log info"""
    if torch.cuda.is_available():
    if config.DEVICE.type == "xpu":
        logger.info(f"✓ Intel XPU detected: {torch.xpu.get_device_name(0)}")
        return True
    elif torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✓ GPU detected: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
        return True
    else:
        logger.warning("⚠ No GPU detected. Processing will be MUCH slower on CPU.")
        return False

def estimate_processing_time(video_duration, upscale_factor, target_fps, input_fps):
    """Estimate total processing time"""
    # Base time for upscaling (minutes per minute of video)
    upscale_time = {2: 20, 4: 60}  # RTX 4060 estimates
    base_time = upscale_time.get(upscale_factor, 30)
    
    # Frame interpolation multiplier
    fps_multiplier = target_fps / input_fps
    if fps_multiplier > 1:
        base_time *= 1.5  # Add 50% for frame interpolation
    
    estimated_minutes = (video_duration / 60) * base_time
    return estimated_minutes

def resolve_input_path(input_arg):
    """
    Resolve input path - accepts filename, relative path, or absolute path
    If just filename, looks in input_videos directory
    """
    input_path = Path(input_arg)
    
    # If it's an absolute path or relative path that exists, use it directly
    if input_path.is_absolute() or input_path.exists():
        return input_path
    
    # Otherwise, look in input_videos directory
    default_input_path = Path(config.INPUT_DIR) / input_arg
    if default_input_path.exists():
        return default_input_path
    
    # If neither exists, return the original path (will error later with clear message)
    return input_path

def cut_video(input_path, start_sec, end_sec, output_path=None):
    """
    Cut video from start_sec to end_sec using FFmpeg
    
    Args:
        input_path: Input video file
        start_sec: Start time in seconds
        end_sec: End time in seconds
        output_path: Output path (optional)
    
    Returns:
        Path to cut video file
    """
    import cv2
    import subprocess
    
    # Validate time range
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    if start_sec < 0 or end_sec > duration or start_sec >= end_sec:
        logger.error(f"Invalid time range: {start_sec}-{end_sec}s (video duration: {duration:.1f}s)")
        raise ValueError(f"Time range must be within 0-{duration:.1f}s and start < end")
    
    # Set output path
    if output_path is None:
        input_path = Path(input_path)
        output_path = Path(config.TEMP_DIR) / f"{input_path.stem}_cut_{int(start_sec)}-{int(end_sec)}.mp4"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Cutting video from {start_sec}s to {end_sec}s ({end_sec - start_sec}s duration)")
    logger.info(f"Original duration: {duration:.1f}s")
    
    # Use FFmpeg to cut (fast, no re-encoding)
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_sec),
        '-i', str(input_path),
        '-to', str(end_sec - start_sec),
        '-c', 'copy',
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"✓ Video cut saved to: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr.decode()}")
        raise

def clean_temp_directory():
    """Clean temp directory before processing"""
    import shutil
    temp_dir = Path("temp")
    if temp_dir.exists():
        logger.info("Cleaning temp directory...")
        # Remove all files but keep the directory
        for item in temp_dir.glob("*"):
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    # Skip git repos and other important directories
                    if item.name not in ["Practical-RIFE", "ECCV2022-RIFE", ".git"]:
                        shutil.rmtree(item)
            except Exception as e:
                logger.warning(f"Could not remove {item}: {e}")
        logger.info("✓ Temp directory cleaned")

def enhance_video(input_path, output_path=None, steps=None):
    """
    Main enhancement pipeline
    
    Args:
        input_path: Path to input video file
        output_path: Path to save enhanced video (optional)
        steps: List of processing steps to perform (optional)
    """
    logger.info("="*60)
    logger.info("8mm Film Enhancement Pipeline")
    logger.info("="*60)
    
    # Check GPU
    has_gpu = check_gpu()
    if not has_gpu and config.USE_GPU:
        logger.warning("GPU not available but USE_GPU=True in config. Proceeding with CPU...")
    
    # Validate input
    input_path = Path(input_path)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return False
    
    # Set output path
    if output_path is None:
        output_path = Path(config.OUTPUT_DIR) / f"{input_path.stem}{config.OUTPUT_SUFFIX}{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Processing steps
    if steps is None:
        steps = config.PROCESSING_STEPS
    
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Processing steps: {', '.join(steps)}")
    
    # Get video info
    import cv2
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    logger.info(f"Video info: {width}x{height} @ {fps:.2f}fps, {frame_count} frames ({duration:.1f}s)")
    
    # Estimate processing time
    if config.ESTIMATE_PROCESSING_TIME:
        est_time = estimate_processing_time(duration, config.UPSCALE_FACTOR, config.TARGET_FPS, fps)
        logger.info(f"Estimated processing time: {est_time:.1f} minutes")
    
    # Process each step
    current_file = input_path
    start_time = time.time()
    step_counter = 1
    
    for step in steps:
        logger.info(f"\n{'='*60}")
        logger.info(f"Step {step_counter}/{len(steps)}: {step.upper()}")
        logger.info(f"{'='*60}")
        
        step_start = time.time()
        
        if step == "stabilization":
            from scripts.stabilize_video import stabilize
            output_name = Path("temp") / f"{input_path.stem}_step{step_counter:02d}_stabilized.mp4"
            current_file = stabilize(current_file, output_path=str(output_name))
        
        elif step == "scratch_removal":
            from scripts.film_restoration import remove_defects_ai
            output_name = Path("temp") / f"{input_path.stem}_step{step_counter:02d}_scratch_removed.mp4"
            # AI-based scratch/dust removal: Real-ESRGAN + temporal defect detection + inpainting
            current_file = remove_defects_ai(current_file, output_path=str(output_name), denoise_strength=0.7)
        
        elif step == "upscaling":
            from scripts.upscale_video import upscale
            output_name = Path("temp") / f"{input_path.stem}_step{step_counter:02d}_upscaled_{config.UPSCALE_FACTOR}x.mp4"
            current_file = upscale(current_file, output_path=str(output_name))
        
        elif step == "denoising":
            from scripts.denoise_video import denoise_gpu
            output_name = Path("temp") / f"{input_path.stem}_step{step_counter:02d}_denoised.mp4"
            current_file = denoise_gpu(current_file, output_path=str(output_name))
        
        elif step == "face_restoration":
            from scripts.restore_faces import restore_faces
            output_name = Path("temp") / f"{input_path.stem}_step{step_counter:02d}_faces_restored.mp4"
            current_file = restore_faces(current_file, output_path=str(output_name))
        
        elif step == "detail_enhancement":
            from scripts.enhance_details import enhance_details
            output_name = Path("temp") / f"{input_path.stem}_step{step_counter:02d}_detailed.mp4"
            current_file = enhance_details(current_file, output_video=str(output_name))
        
        elif step == "deflicker":
            from scripts.deflicker_video import deflicker
            output_name = Path("temp") / f"{input_path.stem}_step{step_counter:02d}_deflickered.mp4"
            current_file = deflicker(current_file, output_path=str(output_name))
        
        elif step == "frame_interpolation":
            from scripts.interpolate_frames import interpolate
            output_name = Path("temp") / f"{input_path.stem}_step{step_counter:02d}_interpolated_{config.TARGET_FPS}fps.mp4"
            current_file = interpolate(current_file, output_path=str(output_name))
        
        elif step == "color_grading":
            logger.info("Color grading is a manual step - use DaVinci Resolve or similar")
        
        else:
            logger.warning(f"Unknown step: {step}")
        
        step_time = time.time() - step_start
        logger.info(f"✓ Step {step_counter}/{len(steps)} ({step}) completed in {step_time/60:.1f} minutes ({step_time:.1f} seconds)")
        step_counter += 1
    
    # Move final result to output path
    if current_file != output_path:
        import shutil
        shutil.move(str(current_file), str(output_path))
    
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Enhancement complete!")
    logger.info(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    logger.info(f"Output saved to: {output_path}")
    logger.info(f"All intermediate steps saved in temp/ directory")
    logger.info(f"{'='*60}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="8mm Film Enhancement Pipeline - AI-powered restoration for home movies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
✅ WORKING FEATURES:
  - Stabilization: Remove camera shake (OpenCV motion estimation)
  - Deflicker: Remove brightness flickering from projector scans (temporal median)
  - Denoising: 7-frame temporal Non-Local Means (aggressive grain removal)
  - Upscaling: Real-ESRGAN 2x/4x (98%% GPU utilization, ~4 sec/frame @ 2x)
  - Face Restoration: GFPGAN v1.4 (enhance faces after upscaling)
  - Detail Enhancement: Smart unsharp masking with edge detection (adaptive sharpening)
  - Scratch Removal: AI-based temporal defect detection + inpainting

📋 EXAMPLES:

  # Test on 10-second clip (recommended first step)
  python enhance_video.py Movie0030.MP4 --cut 172 182 --steps stabilization denoising upscaling --upscale 2
  
  # CLASSIC PIPELINE - All working steps in recommended order (8mm projector scans)
  python enhance_video.py Movie0030.MP4 --steps stabilization deflicker denoising upscaling face_restoration detail_enhancement --upscale 2
  
  # CLASSIC + Scratch removal for heavily damaged film (WARNING: ~32 sec/frame, very slow)
  python enhance_video.py Movie0030.MP4 --steps stabilization deflicker scratch_removal denoising upscaling face_restoration detail_enhancement --upscale 2
  
  # Quick enhancement without deflicker (for non-projector sources)
  python enhance_video.py Movie0030.MP4 --steps stabilization denoising upscaling face_restoration detail_enhancement --upscale 2
  
  # 4x upscaling for very low resolution footage
  python enhance_video.py Movie0030.MP4 --steps denoising upscaling --upscale 4

⏱️  PROCESSING TIME (RTX 4060):
  - Stabilization: ~1 sec/frame
  - Deflicker: ~0.3 sec/frame
  - Denoising: ~2 sec/frame  
  - Upscaling 2x: ~4 sec/frame
  - Face Restoration: ~5 sec/frame
  - Detail Enhancement: ~1 sec/frame (upscaled resolution)
  - Total for 10 sec @ 20fps: ~4-6 minutes

💾 OUTPUT:
  - All intermediate steps saved in temp/ with meaningful names
  - Final output saved to output_videos/
        """
    )
    
    parser.add_argument("input", 
                       help="Input video filename (looks in input_videos/), relative path, or absolute path")
    parser.add_argument("-o", "--output", 
                       help="Output video file path (optional)")
    parser.add_argument("--cut", nargs=2, type=float, metavar=("START", "END"),
                       help="Cut video from START second to END second (optional first step)")
    parser.add_argument("-s", "--steps", nargs="+", 
                       choices=["stabilization", "deflicker", "scratch_removal", "denoising", "upscaling", "face_restoration", "detail_enhancement", "frame_interpolation", "color_grading"],
                       help="Processing steps to perform (default: all from config)")
    parser.add_argument("--skip-steps", nargs="+", 
                       choices=["stabilization", "deflicker", "scratch_removal", "denoising", "upscaling", "face_restoration", "detail_enhancement", "frame_interpolation", "color_grading"],
                       help="Steps to skip (useful for resuming)")
    parser.add_argument("--resume-from", 
                       help="Path to intermediate file to resume from (e.g., upscaled file)")
    parser.add_argument("--upscale", type=int, choices=[2, 4], 
                       help="Upscale factor (overrides config)")
    parser.add_argument("--fps", type=int, 
                       help="Target frame rate (overrides config)")
    parser.add_argument("--nvenc", action="store_true",
                       help="Use NVIDIA hardware encoding (h264_nvenc) - much faster encoding")
    parser.add_argument("--model", type=str, choices=["realesrgan", "basicvsr"],
                       help="AI upscaling model: 'realesrgan' (fast, per-frame) or 'basicvsr' (slow, temporal)")
    
    args = parser.parse_args()
    
    # If no arguments provided (just input), show help and exit
    if not any([args.output, args.cut, args.steps, args.skip_steps, args.resume_from, args.upscale, args.fps, args.nvenc, args.model]):
        parser.print_help()
        sys.exit(0)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Note: NOT cleaning temp directory - keeping intermediate files for debugging/resume
    # Temp files are saved with meaningful step names for progress tracking
    
    # Resolve input path (handles filename, relative, or absolute path)
    input_file = resolve_input_path(args.input)
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.error(f"Tried: {args.input}")
        if not Path(args.input).is_absolute():
            logger.error(f"Also tried: {Path(config.INPUT_DIR) / args.input}")
        sys.exit(1)
    
    logger.info(f"Input file: {input_file}")
    
    # Handle --cut option (optional first step)
    if args.cut:
        start_sec, end_sec = args.cut
        logger.info(f"Cutting video from {start_sec}s to {end_sec}s")
        
        # Check if user wants ONLY cutting (no other processing steps)
        only_cutting = not (args.steps or args.upscale or args.fps or args.skip_steps or args.resume_from)
        
        # Determine output path for cut video
        if only_cutting:
            # Only cutting - save to output directory
            if args.output:
                cut_output = args.output
            else:
                # Default output path for cut-only
                stem = Path(input_file).stem
                cut_output = Path(config.OUTPUT_DIR) / f"{stem}_cut_{int(start_sec)}-{int(end_sec)}.mp4"
        else:
            # Cutting as part of pipeline - save to temp
            cut_output = None
        
        try:
            input_file = cut_video(input_file, start_sec, end_sec, cut_output)
            
            # If we only wanted to cut (no other steps), we're done
            if only_cutting:
                logger.info(f"✓ Video cut complete: {cut_output}")
                sys.exit(0)
                
        except (ValueError, subprocess.CalledProcessError) as e:
            logger.error(f"Failed to cut video: {e}")
            sys.exit(1)
    
    # Override config if specified
    if args.upscale:
        config.UPSCALE_FACTOR = args.upscale
    if args.fps:
        config.TARGET_FPS = args.fps
    if args.nvenc:
        config.USE_NVENC = True
        logger.info("✓ NVENC hardware encoding enabled")
    if args.model:
        config.UPSCALE_MODEL = args.model
        logger.info(f"✓ Using {args.model} upscaling model")
    
    # Handle skip steps
    steps = args.steps
    if args.skip_steps and steps is None:
        # Start with default steps and remove the ones to skip
        steps = [s for s in config.PROCESSING_STEPS if s not in args.skip_steps]
    elif args.skip_steps and steps:
        logger.warning("Both --steps and --skip-steps specified. Using --steps only.")
    
    # Adjust output path to include cut info if video was cut
    output_path = args.output
    if args.cut and output_path is None:
        # Add cut info to default output filename
        stem = Path(input_file).stem
        if "_cut_" in stem:
            # Use the cut filename as base
            output_path = Path(config.OUTPUT_DIR) / f"{stem}{config.OUTPUT_SUFFIX}.mp4"
    
    # Run enhancement
    if args.resume_from:
        input_file = args.resume_from
    
    success = enhance_video(input_file, output_path, steps)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
