#!/usr/bin/env python3
"""
Setup script for 8mm film enhancement project
This script helps set up the environment and download necessary models
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"Running: {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ Success: {description or command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {description or command}")
        print(f"Error output: {e.stderr}")
        return False

def download_file(url, filename):
    """Download a file from URL"""
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    dirs = [
        "models",
        "input_videos",
        "output_videos", 
        "temp",
        "scripts"
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created directory: {dir_name}")

def download_models():
    """Download pre-trained models"""
    models_dir = Path("models")
    
    # Real-ESRGAN models
    realesrgan_models = {
        "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    }
    
    # BasicVSR++ models (from OpenMMLab)
    basicvsr_models = {
        "BasicVSR_REDS4.pth": "https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"
    }
    
    print("\nDownloading Real-ESRGAN models...")
    for model_name, url in realesrgan_models.items():
        model_path = models_dir / model_name
        if not model_path.exists():
            download_file(url, str(model_path))
        else:
            print(f"✓ {model_name} already exists")
    
    print("\nDownloading BasicVSR++ models (optional - for best quality)...")
    for model_name, url in basicvsr_models.items():
        model_path = models_dir / model_name
        if not model_path.exists():
            print(f"⚠ BasicVSR++ model is large (~246MB) and optional")
            print("  BasicVSR++ provides better quality but is 3-5x slower than Real-ESRGAN")
            response = input(f"Download {model_name}? (y/n): ").lower()
            if response == 'y':
                download_file(url, str(model_path))
            else:
                print(f"⏩ Skipped {model_name} (you can download later with: python setup.py)")
        else:
            print(f"✓ {model_name} already exists")

def main():
    print("=== 8mm Film Enhancement Setup ===\n")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Setup directories
    print("\nSetting up directories...")
    setup_directories()
    
    # Install requirements
    print("\nInstalling Python packages...")
    requirements_file = "requirements.txt"
    if Path("requirements-intel.txt").exists():
        requirements_file = "requirements-intel.txt"
        print(f"Intel build detected - using {requirements_file}")
    if not run_command(f"pip install -r {requirements_file}", "Installing requirements"):
        print("Warning: Some packages failed to install. You may need to install them manually.")
    
    # Download models
    print("\nDownloading AI models...")
    download_models()
    
    print(f"\n{'='*50}")
    print("Setup complete! Next steps:")
    print("1. Place your 8mm MP4 files in the 'input_videos' folder")
    print("2. Run the enhancement scripts from the 'scripts' folder")
    print("3. Enhanced videos will be saved in 'output_videos' folder")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
