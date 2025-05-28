from fastapi import APIRouter, File, UploadFile
from app.core.image_processor import ImageProcessor
from app.core.config import Config

# Initialize router
router = APIRouter()

# Initialize configuration and image processor
config = Config()
image_processor = ImageProcessor(config)

@router.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    """FastAPI endpoint for processing images."""
    return await image_processor.process_image(file) 