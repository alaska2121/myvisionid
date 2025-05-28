#!/usr/bin/env python3
"""
Script to download required model files for Railway deployment.
This script is designed to handle limited memory environments.
"""

import os
import sys
import urllib.request
import hashlib
import tempfile
import shutil
from pathlib import Path

def download_with_progress(url, destination, expected_size=None):
    """Download a file with progress tracking and memory efficiency."""
    print(f"Downloading {os.path.basename(destination)}...")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download to temporary file first
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        # Download with memory-efficient streaming
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            if expected_size and total_size > 0 and abs(total_size - expected_size) > expected_size * 0.1:
                print(f"Warning: Expected size {expected_size}, got {total_size}")
            
            with open(tmp_path, 'wb') as f:
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks
                
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}% ({downloaded}/{total_size})", end='')
                    else:
                        print(f"\rDownloaded: {downloaded} bytes", end='')
        
        print()  # New line after progress
        
        # Move to final destination
        shutil.move(tmp_path, destination)
        
        # Verify file exists and has reasonable size
        if os.path.exists(destination):
            size = os.path.getsize(destination)
            print(f"Successfully downloaded {os.path.basename(destination)} ({size} bytes)")
            return True
        else:
            print(f"Error: Failed to create {destination}")
            return False
            
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        # Clean up temporary file if it exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False

def main():
    """Download all required model files."""
    print("Starting model download for Railway deployment...")
    
    # Model definitions with URLs and expected sizes (approximate)
    models = [
        {
            "url": "https://huggingface.co/akhaliq/RetinaFace-R50/resolve/main/RetinaFace-R50.pth",
            "path": "retinaface/RetinaFace-R50.pth",
            "expected_size": 109 * 1024 * 1024  # ~109MB
        },
        {
            "url": "https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/resolve/45df3ded47167a551b8b17b61af4fb6e324051da/birefnet-v1-lite.onnx",
            "path": "hivision/creator/weights/birefnet-v1-lite.onnx",
            "expected_size": 224 * 1024 * 1024  # ~224MB
        }
    ]
    
    success_count = 0
    total_count = len(models)
    
    for model in models:
        print(f"\n--- Downloading {model['path']} ---")
        
        # Skip if file already exists and has reasonable size
        if os.path.exists(model['path']):
            existing_size = os.path.getsize(model['path'])
            if existing_size > model['expected_size'] * 0.8:  # At least 80% of expected size
                print(f"File already exists with good size ({existing_size} bytes), skipping...")
                success_count += 1
                continue
            else:
                print(f"File exists but too small ({existing_size} bytes), re-downloading...")
                os.remove(model['path'])
        
        if download_with_progress(model['url'], model['path'], model['expected_size']):
            success_count += 1
        else:
            print(f"Failed to download {model['path']}")
    
    print(f"\n--- Download Summary ---")
    print(f"Successfully downloaded: {success_count}/{total_count} models")
    
    if success_count == total_count:
        print("All models downloaded successfully!")
        return 0
    else:
        print("Some models failed to download. Check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 