FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-app.txt

# Create necessary directories
RUN mkdir -p retinaface/
RUN mkdir -p modnet_photographic_portrait_matting/
RUN mkdir -p hivision/creator/weights/

# Copy model files
COPY retinaface/RetinaFace-R50.pth retinaface/
COPY modnet_photographic_portrait_matting/modnet_photographic_portrait_matting.ckpt modnet_photographic_portrait_matting/
COPY hivision/creator/weights/birefnet-v1-lite.onnx hivision/creator/weights/

# Copy the rest of the application
COPY . .

# Create temp directory
RUN mkdir -p temp

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]