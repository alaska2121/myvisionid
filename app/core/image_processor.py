import os
import logging
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import UploadFile
from fastapi.responses import Response, JSONResponse
import traceback
import psutil
import aiofiles
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from app.core.config import Config
from app.utils.memory import log_memory_usage, force_garbage_collection
from app.utils.timeout import timeout
from inference import run

@dataclass
class ProcessingRequest:
    file: UploadFile
    temp_input: str
    temp_output: str
    start_time: datetime
    request_id: str

class ImageProcessor:
    """Main class for processing images."""
    
    def __init__(self, config: Config):
        self.config = config
        # Use number of CPU cores for max_workers, but leave one core free for system
        cpu_count = max(1, psutil.cpu_count() - 1)
        self.thread_pool = ThreadPoolExecutor(max_workers=cpu_count)
        self.processing_queue = asyncio.Queue(maxsize=100)  # Limit queue size
        self.active_requests: Dict[str, ProcessingRequest] = {}
        self.processing = False
        
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        logging.info(f"Initialized ImageProcessor with {cpu_count} workers")
        
        # Start the queue processor
        asyncio.create_task(self._process_queue())
    
    async def _process_queue(self):
        """Process queued requests."""
        while True:
            if not self.processing and not self.processing_queue.empty():
                self.processing = True
                try:
                    request = await self.processing_queue.get()
                    logging.info(f"Processing queued request {request.request_id}")
                    await self._process_single_request(request)
                finally:
                    self.processing = False
                    self.processing_queue.task_done()
            await asyncio.sleep(0.1)  # Prevent CPU spinning
    
    async def _process_single_request(self, request: ProcessingRequest):
        """Process a single request from the queue."""
        try:
            # Validate and save the uploaded file
            success, error_response = await self.validate_and_save_file(request.file, request.temp_input)
            if not success:
                return error_response
            
            logging.info(f"Processing image for request {request.request_id}...")
            log_memory_usage()
            
            # Process the image with timeout
            success, error_response = await self.process_image_with_timeout(request.temp_input, request.temp_output)
            if not success:
                await self.cleanup_temp_files(request.temp_input, request.temp_output)
                return error_response
            
            force_garbage_collection()
            
            # Read the processed image
            processed_image, error_response = await self.read_processed_image(request.temp_output)
            if processed_image is None:
                await self.cleanup_temp_files(request.temp_input, request.temp_output)
                return error_response
            
            # Clean up temp files
            await self.cleanup_temp_files(request.temp_input, request.temp_output)
            
            force_garbage_collection()
            
            logging.info(f"Completed processing request {request.request_id}")
            return Response(
                content=processed_image,
                media_type="image/jpeg",
                headers={
                    "Content-Type": "image/jpeg",
                    "Content-Length": str(len(processed_image))
                }
            )
            
        except Exception as e:
            logging.error(f"Error processing request {request.request_id}: {str(e)}")
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            await self.cleanup_temp_files(request.temp_input, request.temp_output)
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": str(e),
                    "type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            )
        finally:
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]

    async def validate_and_save_file(self, file: UploadFile, temp_input: str) -> tuple[bool, JSONResponse | None]:
        """Validate and save the uploaded file."""
        content = await file.read()
        logging.info(f"File size: {len(content)} bytes")
        
        if len(content) > 2 * 1024 * 1024:  # 2MB limit
            logging.warning(f"File too large: {len(content)} bytes")
            return False, JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "File too large. Maximum size is 2MB."
                }
            )
        
        logging.info(f"Saving to {temp_input}")
        async with aiofiles.open(temp_input, 'wb') as f:
            await f.write(content)
        return True, None

    async def process_image_with_timeout(self, temp_input: str, temp_output: str) -> tuple[bool, JSONResponse | None]:
        """Process the image with a timeout."""
        try:
            loop = asyncio.get_event_loop()
            with timeout(50):  # 30 second timeout
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

    async def read_processed_image(self, temp_output: str) -> tuple[bytes | None, JSONResponse | None]:
        """Read the processed image from the temporary file."""
        try:
            async with aiofiles.open(temp_output, 'rb') as f:
                processed_image = await f.read()
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

    async def cleanup_temp_files(self, temp_input: str, temp_output: str):
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
        
        # Generate request ID and temp file paths
        request_id = str(uuid.uuid4())
        temp_input = f"temp/{request_id}_input.jpg"
        temp_output = f"temp/{request_id}_output.jpg"
        
        # Create processing request
        request = ProcessingRequest(
            file=file,
            temp_input=temp_input,
            temp_output=temp_output,
            start_time=datetime.now(),
            request_id=request_id
        )
        
        # Check if queue is full
        if self.processing_queue.full():
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "Server is busy. Please try again in a few moments.",
                    "type": "ServiceUnavailable"
                }
            )
        
        # Add to active requests
        self.active_requests[request_id] = request
        
        try:
            # Add to processing queue
            await self.processing_queue.put(request)
            
            # Wait for processing to complete
            while request_id in self.active_requests:
                await asyncio.sleep(0.1)
            
            # Get the result
            return await self._process_single_request(request)
            
        except Exception as e:
            logging.error(f"Error occurred: {str(e)}")
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            # Clean up temp files in case of error
            await self.cleanup_temp_files(temp_input, temp_output)
            # Remove from active requests
            if request_id in self.active_requests:
                del self.active_requests[request_id]
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