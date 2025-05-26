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

# Download model files during build with proper headers and verification
RUN set -x && \
    curl -v -L \
         -H "User-Agent: Mozilla/5.0" \
         -H "Accept: */*" \
         "https://huggingface.co/akhaliq/RetinaFace-R50/resolve/main/RetinaFace-R50.pth" \
         -o retinaface/RetinaFace-R50.pth && \
    curl -v -L \
         -H "User-Agent: Mozilla/5.0" \
         -H "Accept: */*" \
         "https://huggingface.co/yao123/test/resolve/main/modnet_photographic_portrait_matting.ckpt" \
         -o modnet_photographic_portrait_matting/modnet_photographic_portrait_matting.ckpt && \
    curl -v -L \
         -H "User-Agent: Mozilla/5.0" \
         -H "Accept: */*" \
         --retry 3 \
         --retry-delay 2 \
         --retry-max-time 30 \
         "https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/resolve/45df3ded47167a551b8b17b61af4fb6e324051da/birefnet-v1-lite.onnx" \
         -o hivision/creator/weights/birefnet-v1-lite.onnx && \
    # Verify the ONNX model file with memory optimization
    python3 -c "import onnx; onnx.load('hivision/creator/weights/birefnet-v1-lite.onnx', load_external_data=False)"

# Copy the rest of the application
COPY . .

# Create temp directory
RUN mkdir -p temp

# Set environment variables for memory optimization
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Expose the port
EXPOSE 8000

# Command to run the application with memory optimization
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]