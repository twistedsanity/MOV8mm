# Use an official Python base image as a starting point
FROM python:3.9-slim

# Set working directory to /app
WORKDIR /app

# Copy required files from MOV8MM repository
COPY scripts/ /app/scripts/
COPY models/ /app/models/
COPY input_videos/ /app/input_videos/
COPY output_videos/ /app/output_videos/

# Install dependencies
RUN pip install -r requirements.txt
RUN conda install -c conda-forge ffmpeg

# Define environment variables
ENV GPU_ID 0
ENV CUDA_HOME /usr/local/cuda
ENV DENOISE_STRENGTH 0.7
ENV UPSCALE_TILE_SIZE 2048

# Run the pipeline
CMD ["python", "scripts/enhance_video.py", "input_videos/MyFilm.mp4", "--cut", "0", "60", "--steps", "stabilization,deflicker,denoising,upscale,face_restoration", "--upscale", "2", "-o", "output_videos/MyFilm_restored.mp4"]
