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
import json
from loguru import logger
import torch

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
import config

# Configure logger
logger.remove()
logger.add(sys.stderr, level=config.LOG_LEVEL)
logger.add("logs/enhancement_{time}.log", rotation="100 MB")

def load_setup_config(setup_file=None):
    """
    Load setup configuration from JSON file
    
    Args:
        setup_file: Path to setup JSON file. If None, uses 'setup.json'
    
    Returns:
        dict: Configuration dictionary with steps and parameters
    """
    if setup_file is None:
        setup_file = "setup.json"
    
    setup_path = Path(setup_file)
    
    # If not absolute path, look in project root
    if not setup_path.is_absolute():
        project_root = Path(__file__).parent.parent
        setup_path = project_root / setup_file
    
    if not setup_path.exists():
        logger.error(f"Setup file not found: {setup_path}")
        logger.info("Available setup files in project root:")
        project_root = Path(__file__).parent.parent
        for json_file in project_root.glob("setup*.json"):
            logger.info(f"  - {json_file.name}")
        sys.exit(1)
    
    try:
        with open(setup_path, 'r', encoding='utf-8') as f:
            setup_config = json.load(f)
        
        logger.info(f"✓ Loaded setup: {setup_path}")
        logger.info(f"  Description: {setup_config.get('description', 'N/A')}")
        
        return setup_config
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in setup file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading setup file: {e}")
        sys.exit(1)

def apply_setup_config(setup_config, args):
    """
    Apply setup configuration to override default parameters
    
    Args:
        setup_config: Configuration dictionary from JSON
        args: Command line arguments namespace
    
    Returns:
        tuple: (steps_to_run, parameters_dict)
    """
    pipeline = setup_config.get('pipeline', {})
    steps_config = pipeline.get('steps', [])
    output_config = pipeline.get('output', {})
    processing_config = pipeline.get('processing', {})
    
    # Build list of enabled steps
    steps_to_run = []
    parameters_dict = {}
    
    for step_config in steps_config:
        step_name = step_config.get('name')
        if step_config.get('enabled', False):
            steps_to_run.append(step_name)
            parameters_dict[step_name] = step_config.get('parameters', {})
    
    logger.info(f"Enabled steps from setup: {', '.join(steps_to_run)}")
    
    # Apply output configuration
    if output_config.get('fps') is not None:
        config.TARGET_FPS = output_config['fps']
    if output_config.get('codec'):
        config.OUTPUT_CODEC = output_config['codec']
    if output_config.get('use_nvenc') is not None:
        config.USE_NVENC = output_config['use_nvenc']
    
    # Apply processing configuration
    if processing_config.get('temp_directory'):
        config.TEMP_DIR = processing_config['temp_directory']
    
    # Command line arguments override setup file
    if args.upscale:
        config.UPSCALE_FACTOR = args.upscale
        logger.info(f"Command line override: upscale = {args.upscale}")
    elif 'upscaling' in parameters_dict and 'scale' in parameters_dict['upscaling']:
        config.UPSCALE_FACTOR = parameters_dict['upscaling']['scale']
    
    if args.fps:
        config.TARGET_FPS = args.fps
        logger.info(f"Command line override: fps = {args.fps}")
    
    if args.nvenc:
        config.USE_NVENC = True
        logger.info("Command line override: nvenc = True")
    
    return steps_to_run, parameters_dict

def check_gpu():
    """Check if GPU is available and log info"""
    if torch.xpu.is_available():
        gpu_name = torch.xpu.get_device_name(0)
        gpu_memory = torch.xpu.get_device_properties(0).total_memory / 1024**3
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
    
    # If end_sec is None, cut to end of video
    if end_sec is None:
        end_sec = duration
    
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

def enhance_video(input_path, output_path=None, steps=None, step_parameters=None):
    """
    Main enhancement pipeline
    
    Args:
        input_path: Path to input video file
        output_path: Path to save enhanced video (optional)
        steps: List of processing steps to perform (optional)
        step_parameters: Dictionary of parameters for each step from setup JSON (optional)
    """
    if step_parameters is None:
        step_parameters = {}
    
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
            
            # Get crop parameter from config (default 10%)
            params = step_parameters.get('stabilization', {}) if step_parameters else {}
            crop_percent = params.get('crop_percent', 10.0)
            current_file = stabilize(current_file, output_path=str(output_name), crop_percent=crop_percent)
        
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
            from scripts.denoise_video_gpu import denoise_video_gpu
            output_name = Path("temp") / f"{input_path.stem}_step{step_counter:02d}_denoised.mp4"
            # Get denoising parameters from step_parameters or use defaults
            params = step_parameters.get('denoising', {}) if step_parameters else {}
            strength = params.get('strength', 0.7)
            temporal = params.get('temporal', True)
            method = params.get('method', 'gaussian')
            logger.info(f"GPU denoising with strength={strength}, temporal={temporal}, method={method}")
            current_file = denoise_video_gpu(current_file, output_path=str(output_name), 
                                            strength=strength, temporal=temporal, method=method)
        
        elif step == "face_restoration":
            from scripts.restore_faces import restore_faces
            output_name = Path("temp") / f"{input_path.stem}_step{step_counter:02d}_faces_restored.mp4"
            # Get face restoration parameters from config
            params = step_parameters.get('face_restoration', {}) if step_parameters else {}
            weight = params.get('weight', 0.7)
            logger.info(f"Face restoration with weight={weight}")
            current_file = restore_faces(current_file, output_path=str(output_name), weight=weight)
        
        elif step == "detail_enhancement":
            from scripts.enhance_details import enhance_details
            output_name = Path("temp") / f"{input_path.stem}_step{step_counter:02d}_detailed.mp4"
            # Get detail enhancement parameters from config
            params = step_parameters.get('detail_enhancement', {}) if step_parameters else {}
            method = params.get('method', 'adaptive')
            radius = params.get('radius', 2.0)
            amount = params.get('amount', 2.0)
            strength = params.get('strength', 1.0)
            logger.info(f"Detail enhancement: method={method}, radius={radius}, amount={amount}")
            current_file = enhance_details(current_file, output_video=str(output_name),
                                          method=method, radius=radius, amount=amount, strength=strength)
        
        elif step == "deflicker":
            from scripts.deflicker_video import deflicker
            output_name = Path("temp") / f"{input_path.stem}_step{step_counter:02d}_deflickered.mp4"
            current_file = deflicker(current_file, output_path=str(output_name))
        
        elif step == "frame_interpolation":
            from scripts.interpolate_frames_gpu import interpolate_video_gpu
            output_name = Path("temp") / f"{input_path.stem}_step{step_counter:02d}_interpolated_{config.TARGET_FPS}fps.mp4"
            # Get interpolation parameters from config
            params = step_parameters.get('frame_interpolation', {}) if step_parameters else {}
            target_fps = params.get('target_fps', 60)
            logger.info(f"GPU optical flow interpolation to {target_fps}fps")
            current_file = interpolate_video_gpu(current_file, output_path=str(output_name), target_fps=target_fps)
        
        elif step == "color_grading" or step == "color_correction":
            from scripts.color_correction import correct_video_colors
            output_name = Path("temp") / f"{input_path.stem}_step{step_counter:02d}_color_corrected.mp4"
            # Get color correction parameters from step_parameters or use defaults
            params = step_parameters.get('color_correction', {}) if step_parameters else {}
            params = params if params else step_parameters.get('color_grading', {}) if step_parameters else {}
            clahe = params.get('clahe', True)
            white_balance = params.get('white_balance', True)
            gamma = params.get('gamma', 1.0)
            saturation = params.get('saturation', 1.2)
            shadows = params.get('shadows', 0.0)
            highlights = params.get('highlights', 0.0)
            logger.info(f"Color correction: CLAHE={clahe}, WB={white_balance}, gamma={gamma}, sat={saturation}")
            current_file = correct_video_colors(current_file, output_path=str(output_name),
                                               clahe=clahe, white_balance=white_balance,
                                               gamma=gamma, saturation=saturation,
                                               shadows=shadows, highlights=highlights)
        
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
OPTIMIZED PIPELINE (GPU-Accelerated):
  1. Stabilization: Remove camera shake + 10%% edge crop (removes black borders)
  2. Deflicker: Fix brightness flickering from old projectors
  3. Denoising: GPU-accelerated temporal denoising (93x faster than CPU)
  4. Face Restoration: GFPGAN v1.4 enhancement (weight 0.7 for natural look)
  5. Detail Enhancement: Adaptive unsharp masking (crisp preset)
  6. Color Correction: Natural color enhancement (soft preset)

CRITICAL PIPELINE ORDER:
  - Face restoration BEFORE interpolation (prevents flickering on morphed frames)
  - Detail enhancement BEFORE interpolation (sharpen original, not blurred frames)
  - Frame interpolation DISABLED (20fps preserves authentic film feel)

EXAMPLES:

  # Simplest usage - process with default setup.json (auto-detects input_videos/ folder)
  python enhance_video.py friends.MP4
  
  # Run optimized pipeline on 10-second cut (RECOMMENDED)
  python enhance_video.py input_videos/friends.MP4 --cut 155 165 -o output.mp4 --setup
  
  # Cut from 155 seconds to end of video
  python enhance_video.py input_videos/friends.MP4 --cut 155 --setup
  
  # Use specific setup configuration
  python enhance_video.py input_videos/Movie.MP4 --setup setup_optimized.json
  
  # Manual step selection (overrides setup.json)
  python enhance_video.py Movie.MP4 --steps stabilization deflicker denoising face_restoration detail_enhancement color_correction
  
  # Full quality with upscaling (slow - adds 15+ minutes)
  python enhance_video.py Movie.MP4 --setup setup_optimized.json --steps stabilization deflicker denoising upscaling face_restoration

PROCESSING TIME (NVIDIA RTX 4060 Laptop):
  10-second clip (~200 frames @ 20fps):
  - Stabilization (10%% crop): ~13 sec
  - Deflicker: ~6 sec  
  - GPU Denoising (0.7): ~25 sec (93x faster than CPU!)
  - Face Restoration (0.7): ~220 sec (~3.7 min)
  - Detail Enhancement (crisp): ~40 sec
  - Color Correction (soft): ~12 sec
  TOTAL: ~5-6 minutes (without upscaling)
  
  WITH UPSCALING: +40 minutes (200 frames × 12 sec/frame)

TESTED PARAMETERS (all optimized):
  - Denoising strength: 0.7 (tested 0.7/0.8/0.9/1.0, selected 0.7)
  - Face weight: 0.7 (tested 0.5/0.7, selected 0.7 for better enhancement)
  - Detail: crisp (radius 2.0, amount 2.0 - tested subtle/balanced/crisp/strong)
  - Color: soft preset (no CLAHE/WB, gamma 0.97, sat 1.05)
  - Stabilization crop: 10%% (removes black borders + damaged film edges)

SETUP FILES:
  - setup.json: Optimized pipeline (6 steps, ~5 min for 10sec clip)
  - setup_optimized.json: Same as setup.json (60fps natural look)
  - setup_quick.json: Fast preview mode
  - setup_max_quality.json: Full quality with upscaling (slow)
        """
    )
    
    parser.add_argument("input", 
                       help="Input video filename (looks in input_videos/), relative path, or absolute path")
    parser.add_argument("-o", "--output", 
                       help="Output video file path (optional)")
    parser.add_argument("--cut", nargs="+", type=float, metavar=("START", "[END]"),
                       help="Cut video from START second to END second (or to end of video if END not specified)")
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
    parser.add_argument("--setup", nargs='?', const='setup.json', metavar='JSON_FILE',
                       help="Use setup configuration from JSON file (default: setup.json). Defines pipeline steps and parameters.")
    
    args = parser.parse_args()
    
    # Handle setup configuration
    step_parameters = None
    
    # If no processing arguments provided, automatically use setup.json as default
    if not any([args.cut, args.steps, args.skip_steps, args.resume_from, args.upscale, args.fps, args.nvenc, args.model, args.setup]):
        logger.info("No processing arguments specified - using default setup.json")
        args.setup = 'setup.json'
    
    if args.setup:
        setup_config = load_setup_config(args.setup)
        steps_from_setup, step_parameters = apply_setup_config(setup_config, args)
        
        # Use steps from setup if not overridden by command line
        if not args.steps:
            args.steps = steps_from_setup
    
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
        # Validate cut arguments (1 or 2 values)
        if len(args.cut) == 1:
            start_sec = args.cut[0]
            end_sec = None  # Cut to end of video
            logger.info(f"Cutting video from {start_sec}s to end of video")
        elif len(args.cut) == 2:
            start_sec, end_sec = args.cut
            logger.info(f"Cutting video from {start_sec}s to {end_sec}s")
        else:
            logger.error("--cut requires 1 or 2 arguments: START [END]")
            sys.exit(1)
        
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
                if end_sec is None:
                    cut_output = Path(config.OUTPUT_DIR) / f"{stem}_cut_{int(start_sec)}-end.mp4"
                else:
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
    
    success = enhance_video(input_file, output_path, steps, step_parameters)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
