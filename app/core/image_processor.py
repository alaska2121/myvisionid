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
    result_future: asyncio.Future

class ImageProcessor:
    """Main class for processing images with configurable concurrent processing."""
    
    def __init__(self, config: Config, max_concurrent_workers: int = 2):
        self.config = config
        self.max_concurrent_workers = max_concurrent_workers
        
        # Use configurable workers, but cap at 3 to prevent memory issues
        workers = min(max_concurrent_workers, 3)
        self.thread_pool = ThreadPoolExecutor(max_workers=workers)
        
        # Semaphore to control concurrent processing
        self.processing_semaphore = asyncio.Semaphore(workers)
        
        # Queue for requests
        self.processing_queue = asyncio.Queue(maxsize=50)
        self.active_requests: Dict[str, ProcessingRequest] = {}
        
        # Memory monitoring from config
        self.memory_threshold_mb = config.memory_threshold_mb
        
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        logging.info(f"Initialized ImageProcessor with {workers} concurrent workers")
        logging.info(f"Memory threshold: {self.memory_threshold_mb}MB")
        logging.info(f"Max file size: {config.max_file_size_mb}MB")
        
        # Start multiple queue processors for concurrent processing
        for i in range(workers):
            asyncio.create_task(self._process_queue(worker_id=i))
    
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # More aggressive memory checking for Railway
            is_railway = os.getenv("RAILWAY_ENVIRONMENT") is not None
            threshold = self.memory_threshold_mb
            
            if is_railway:
                # Even more strict on Railway
                threshold = min(threshold, 500)
            
            if memory_mb > threshold:
                logging.warning(f"Memory usage too high: {memory_mb:.1f}MB > {threshold}MB (Railway: {is_railway})")
                
                # Force garbage collection immediately
                force_garbage_collection()
                
                # Check again after cleanup
                memory_mb_after = process.memory_info().rss / 1024 / 1024
                if memory_mb_after > threshold:
                    logging.warning(f"Memory still high after cleanup: {memory_mb_after:.1f}MB > {threshold}MB")
                    return False
                else:
                    logging.info(f"Memory cleaned up: {memory_mb:.1f}MB -> {memory_mb_after:.1f}MB")
            
            return True
        except Exception as e:
            logging.error(f"Error checking memory usage: {str(e)}")
            return True  # Assume OK if we can't check
    
    async def _process_queue(self, worker_id: int = 0):
        """Process queued requests with concurrent processing."""
        logging.info(f"Starting queue processor worker {worker_id}...")
        while True:
            try:
                # Get request from queue (this blocks until a request is available)
                request = await self.processing_queue.get()
                
                # Check memory before processing
                if not self._check_memory_usage():
                    # If memory is too high, requeue the request and wait
                    await self.processing_queue.put(request)
                    await asyncio.sleep(2)  # Wait before retrying
                    continue
                
                logging.info(f"Worker {worker_id} processing request {request.request_id}")
                
                try:
                    # Acquire semaphore for concurrent processing control
                    async with self.processing_semaphore:
                        # Process the request
                        result = await self._process_single_request(request)
                        # Set the result in the future
                        if not request.result_future.done():
                            request.result_future.set_result(result)
                            
                except Exception as e:
                    # Set the exception in the future
                    if not request.result_future.done():
                        request.result_future.set_exception(e)
                finally:
                    # Mark task as done
                    self.processing_queue.task_done()
                    # Remove from active requests
                    if request.request_id in self.active_requests:
                        del self.active_requests[request.request_id]
                    
                    # Force garbage collection after each request
                    force_garbage_collection()
                    
            except Exception as e:
                logging.error(f"Error in queue processor worker {worker_id}: {str(e)}")
                logging.error(traceback.format_exc())
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _process_single_request(self, request: ProcessingRequest):
        """Process a single request from the queue."""
        try:
            logging.info(f"Starting processing for request {request.request_id}")
            
            # Validate and save the uploaded file
            success, error_response = await self.validate_and_save_file(request.file, request.temp_input)
            if not success:
                await self.cleanup_temp_files(request.temp_input, request.temp_output)
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
            log_memory_usage()
            
            processing_time = (datetime.now() - request.start_time).total_seconds()
            logging.info(f"Request completed: POST /process-image - Status: 200 - Time: {processing_time:.2f}s")
            
            # Return the processed image directly
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
                    "request_id": request.request_id
                }
            )

    async def validate_and_save_file(self, file: UploadFile, temp_input: str) -> tuple[bool, JSONResponse | None]:
        """Validate and save the uploaded file."""
        try:
            content = await file.read()
            logging.info(f"File size: {len(content)} bytes")
            
            max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
            if len(content) > max_size_bytes:
                logging.warning(f"File too large: {len(content)} bytes > {max_size_bytes} bytes")
                return False, JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": f"File too large. Maximum size is {self.config.max_file_size_mb}MB."
                    }
                )
            
            if len(content) == 0:
                logging.warning("Empty file received")
                return False, JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": "Empty file received."
                    }
                )
            
            # Ensure temp directory exists
            os.makedirs(os.path.dirname(temp_input), exist_ok=True)
            
            logging.info(f"Saving to {temp_input}")
            async with aiofiles.open(temp_input, 'wb') as f:
                await f.write(content)
            
            # Verify file was saved correctly
            if not os.path.exists(temp_input):
                logging.error(f"Failed to save file to {temp_input}")
                return False, JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "message": "Failed to save uploaded file."
                    }
                )
            
            # Verify file is readable and is a valid image
            try:
                # First check file size
                with open(temp_input, 'rb') as f:
                    test_content = f.read()
                if len(test_content) != len(content):
                    logging.error(f"File size mismatch: saved={len(test_content)}, original={len(content)}")
                    return False, JSONResponse(
                        status_code=500,
                        content={
                            "status": "error",
                            "message": "File verification failed."
                        }
                    )
                
                # Then verify it's a valid image
                import cv2
                test_image = cv2.imread(temp_input)
                if test_image is None:
                    logging.error(f"Failed to read image with OpenCV: {temp_input}")
                    return False, JSONResponse(
                        status_code=400,
                        content={
                            "status": "error",
                            "message": "Invalid image file. Please ensure the file is a valid image format (JPEG, PNG)."
                        }
                    )
                
                logging.info(f"Image verification successful. Image shape: {test_image.shape}")
                
            except Exception as verify_error:
                logging.error(f"Error verifying saved file: {str(verify_error)}")
                return False, JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "message": "Failed to verify saved file."
                    }
                )
            
            logging.info("File saved and verified successfully")
            return True, None
            
        except Exception as e:
            logging.error(f"Error in validate_and_save_file: {str(e)}")
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            return False, JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Error saving file: {str(e)}",
                    "type": type(e).__name__
                }
            )

    async def process_image_with_timeout(self, temp_input: str, temp_output: str) -> tuple[bool, JSONResponse | None]:
        """Process the image with a timeout."""
        try:
            # Shorter timeout for Railway
            is_railway = os.getenv("RAILWAY_ENVIRONMENT") is not None
            timeout_seconds = 35 if is_railway else 50
            
            loop = asyncio.get_event_loop()
            with timeout(timeout_seconds):
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
            logging.error(f"Image processing timed out after {timeout_seconds} seconds (Railway: {is_railway})")
            return False, JSONResponse(
                status_code=504,
                content={
                    "status": "error",
                    "message": f"Image processing timed out after {timeout_seconds} seconds. Please try again with a smaller image.",
                    "type": "TimeoutError",
                    "environment": "railway" if is_railway else "other"
                }
            )
        except Exception as processing_error:
            is_railway = os.getenv("RAILWAY_ENVIRONMENT") is not None
            logging.error(f"Error during image processing: {str(processing_error)}")
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            return False, JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Error during image processing: {str(processing_error)}",
                    "type": type(processing_error).__name__,
                    "environment": "railway" if is_railway else "other"
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
        request_id = str(uuid.uuid4())
        logging.info(f"Request started: POST /process-image")
        logging.info(f"Received file: {file.filename}")
        log_memory_usage()
        
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        # Generate temp file paths
        temp_input = f"temp/{request_id}_input.jpg"
        temp_output = f"temp/{request_id}_output.jpg"
        
        try:
            # Create processing request with future for result
            request = ProcessingRequest(
                file=file,
                temp_input=temp_input,
                temp_output=temp_output,
                start_time=datetime.now(),
                request_id=request_id,
                result_future=asyncio.Future()
            )
            
            # Check if queue is full
            if self.processing_queue.full():
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "error",
                        "message": "Server is busy. Please try again in a few moments.",
                        "type": "ServiceUnavailable",
                        "request_id": request_id
                    }
                )
            
            # Add to active requests
            self.active_requests[request_id] = request
            
            # Add request to queue for processing
            await self.processing_queue.put(request)
            logging.info(f"Request {request_id} added to queue. Queue size: {self.processing_queue.qsize()}")
            
            # Wait for the result (this will block until processing is complete)
            try:
                result = await request.result_future
                return result
            except Exception as e:
                logging.error(f"Error waiting for result: {str(e)}")
                raise e
            
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
                    "request_id": request_id
                }
            ) 