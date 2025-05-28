from fastapi import APIRouter, File, UploadFile
from app.core.image_processor import ImageProcessor
from app.core.config import Config

# Initialize router
router = APIRouter()

# Initialize configuration and image processor
config = Config()
image_processor = ImageProcessor(config, max_concurrent_workers=config.max_concurrent_workers)

@router.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    """FastAPI endpoint for processing images."""
    return await image_processor.process_image(file)

@router.get("/status")
async def get_status():
    """Get the current status of the image processor."""
    return {
        "status": "healthy",
        "queue_size": image_processor.processing_queue.qsize(),
        "active_requests": len(image_processor.active_requests),
        "queue_maxsize": image_processor.processing_queue.maxsize,
        "max_concurrent_workers": image_processor.max_concurrent_workers,
        "memory_threshold_mb": image_processor.memory_threshold_mb
    }

@router.get("/config")
async def get_config():
    """Get the current configuration."""
    return {
        "max_concurrent_workers": config.max_concurrent_workers,
        "memory_threshold_mb": config.memory_threshold_mb,
        "max_file_size_mb": config.max_file_size_mb,
        "matting_model": config.matting_model,
        "face_detect_model": config.face_detect_model
    } 