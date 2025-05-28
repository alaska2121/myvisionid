#!/bin/bash
set -e

echo "=== Railway Startup Script ==="
echo "Environment: RAILWAY"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"

# Check if we're on Railway (free command may not be available)
if command -v free >/dev/null 2>&1; then
    echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $2}') total"
else
    echo "Memory info: Railway container (free command not available)"
fi

# Set Railway environment variables
export RAILWAY_ENVIRONMENT=true
export MAX_CONCURRENT_WORKERS=1
export MEMORY_THRESHOLD_MB=400
export MAX_FILE_SIZE_MB=2

echo "=== Configuration ==="
echo "MAX_CONCURRENT_WORKERS: $MAX_CONCURRENT_WORKERS"
echo "MEMORY_THRESHOLD_MB: $MEMORY_THRESHOLD_MB"
echo "MAX_FILE_SIZE_MB: $MAX_FILE_SIZE_MB"

# Create necessary directories
mkdir -p temp logs retinaface hivision/creator/weights

# Download models if they don't exist
echo "=== Checking Model Files ==="
python scripts/download_models_railway.py

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download model files"
    exit 1
fi

echo "=== Starting Application ==="

# Check memory before start (Railway-compatible)
if command -v free >/dev/null 2>&1; then
    echo "Memory before start: $(free -h | grep '^Mem:' | awk '{print $3}') used"
else
    echo "Memory before start: Railway container ready"
fi

# Start the FastAPI application
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1 \
    --access-log \
    --log-level info 