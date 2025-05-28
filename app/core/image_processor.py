import os
import logging
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import UploadFile
from fastapi.responses import Response, JSONResponse
import traceback

from app.core.config import Config
from app.utils.memory import log_memory_usage, force_garbage_collection
from app.utils.timeout import timeout
from inference import run

class ImageProcessor:
    """Main class for processing images."""
    
    def __init__(self, config: Config):
        self.config = config
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
    
    async def validate_and_save_file(self, file: UploadFile, temp_input: str) -> tuple[bool, JSONResponse | None]:
        """Validate and save the uploaded file."""
        content = await file.read()
        logging.info(f"File size: {len(content)} bytes")
        
        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            logging.warning(f"File too large: {len(content)} bytes")
            return False, JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "File too large. Maximum size is 10MB."
                }
            )
        
        logging.info(f"Saving to {temp_input}")
        with open(temp_input, "wb") as f:
            f.write(content)
        return True, None

    async def process_image_with_timeout(self, temp_input: str, temp_output: str) -> tuple[bool, JSONResponse | None]:
        """Process the image with a timeout."""
        try:
            loop = asyncio.get_event_loop()
            with timeout(30):  # 30 second timeout
                result = await loop.run_in_executor(
                    self.thread_pool,
                    lambda: run(
                        input_image_path=temp_input,
                        output_image_path=temp_output,
                        type="add_background",
                        height=288,
                        width=240,
                        color="FFFFFF",  # Pure white
                        hd=True,
                        kb=None,
                        render=0,
                        dpi=300,
                        face_align=False,
                        matting_model=self.config.matting_model,
                        face_detect_model="retinaface-resnet50",
                    )
                )
            return True, None
        except TimeoutError:
            logging.error("Image processing timed out after 30 seconds")
            return False, JSONResponse(
                status_code=504,
                content={
                    "status": "error",
                    "message": "Image processing timed out. Please try again with a smaller image or contact support if the issue persists.",
                    "type": "TimeoutError"
                }
            )
        except Exception as processing_error:
            logging.error(f"Error during image processing: {str(processing_error)}")
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            return False, JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Error during image processing: {str(processing_error)}",
                    "type": type(processing_error).__name__
                }
            )

    def read_processed_image(self, temp_output: str) -> tuple[bytes | None, JSONResponse | None]:
        """Read the processed image from the temporary file."""
        try:
            with open(temp_output, "rb") as f:
                processed_image = f.read()
            logging.info(f"Processed image size: {len(processed_image)} bytes")
            return processed_image, None
        except Exception as read_error:
            logging.error(f"Error reading processed image: {str(read_error)}")
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            return None, JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Error reading processed image: {str(read_error)}",
                    "type": type(read_error).__name__
                }
            )

    def cleanup_temp_files(self, temp_input: str, temp_output: str):
        """Clean up temporary files."""
        try:
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(temp_output):
                os.remove(temp_output)
        except Exception as cleanup_error:
            logging.warning(f"Warning: Error during cleanup: {str(cleanup_error)}")

    async def process_image(self, file: UploadFile) -> Response | JSONResponse:
        """
        Process an uploaded image by adding a pure white background
        """
        logging.info(f"Received file: {file.filename}")
        log_memory_usage()
        
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        # Generate two UUIDs for unique filenames
        uuid1 = str(uuid.uuid4())
        uuid2 = str(uuid.uuid4())
        
        # Save uploaded file temporarily with double UUID
        temp_input = f"temp/{uuid1}_{uuid2}_input.jpg"
        temp_output = f"temp/{uuid1}_{uuid2}_output.jpg"
        
        try:
            # Validate and save the uploaded file
            success, error_response = await self.validate_and_save_file(file, temp_input)
            if not success:
                return error_response
            
            logging.info("Processing image...")
            log_memory_usage()
            
            # Process the image with timeout
            success, error_response = await self.process_image_with_timeout(temp_input, temp_output)
            if not success:
                self.cleanup_temp_files(temp_input, temp_output)
                return error_response
            
            force_garbage_collection()
            
            # Read the processed image
            processed_image, error_response = self.read_processed_image(temp_output)
            if processed_image is None:
                self.cleanup_temp_files(temp_input, temp_output)
                return error_response
            
            # Clean up temp files
            self.cleanup_temp_files(temp_input, temp_output)
            
            force_garbage_collection()
            
            logging.info("Returning response...")
            # Return the processed image
            return Response(
                content=processed_image,
                media_type="image/jpeg",
                headers={
                    "Content-Type": "image/jpeg",
                    "Content-Length": str(len(processed_image))
                }
            )
            
        except Exception as e:
            logging.error(f"Error occurred: {str(e)}")
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            # Clean up temp files in case of error
            self.cleanup_temp_files(temp_input, temp_output)
            # Return a more detailed error response
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": str(e),
                    "type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            ) 