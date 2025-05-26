FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-app.txt

# Create necessary directories
RUN mkdir -p retinaface/
RUN mkdir -p modnet_photographic_portrait_matting/
RUN mkdir -p hivision/creator/weights/

# Download model files during build with authentication here
RUN curl -L -H "Authorization: token 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" \
    "https://github.com/KingOfPeru/myHiVisionIDPhotos/releases/download/v1.0.0-models/RetinaFace-R50.pth" -o retinaface/RetinaFace-R50.pth && \
    curl -L -H "Authorization: token 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" \
    "https://github.com/KingOfPeru/myHiVisionIDPhotos/releases/download/v1.0.0-models/modnet_photographic_portrait_matting.ckpt" -o modnet_photographic_portrait_matting/modnet_photographic_portrait_matting.ckpt && \
    curl -L -H "Authorization: token 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" \
    "https://github.com/KingOfPeru/myHiVisionIDPhotos/releases/download/v1.0.0-models/birefnet-v1-lite.onnx" -o hivision/creator/weights/birefnet-v1-lite.onnx

# Copy the rest of the application
COPY . .

# Create temp directory
RUN mkdir -p temp

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]