FROM python:3.10-slim

# Set environment variables for better memory management
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PYTHONDONTWRITEBYTECODE=1

# Railway-specific optimizations
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make scripts executable
RUN chmod +x scripts/railway_start.sh

# Create necessary directories
RUN mkdir -p temp logs

# Set Railway environment variable
ENV RAILWAY_ENVIRONMENT=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

EXPOSE 8000

# Use Railway startup script
CMD ["./scripts/railway_start.sh"]
