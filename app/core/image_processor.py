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
import time

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
    result: Response | JSONResponse | None = None

class ImageProcessor:
    """Main class for processing images."""
    
    def __init__(self, config: Config):
        self.config = config
        # Use number of CPU cores for max_workers, but leave one core free for system
        cpu_count = max(1, psutil.cpu_count() - 1)
        self.thread_pool = ThreadPoolExecutor(max_workers=cpu_count)
        self.processing_queue = asyncio.Queue(maxsize=50)  # Reduced queue size
        self.active_requests: Dict[str, ProcessingRequest] = {}
        self.active_workers = 0  # Track number of active workers
        self.max_workers = min(cpu_count, 2)  # Limit to 2 workers
        self.memory_threshold = 0.7  # 70% memory threshold
        self.critical_memory_threshold = 0.85  # 85% critical threshold
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 5  # Cleanup every 5 seconds
        
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        logging.info(f"Initialized ImageProcessor with {self.max_workers} workers")
        
        # Start the queue processor
        asyncio.create_task(self._process_queue())
        # Start the cleanup task
        asyncio.create_task(self._periodic_cleanup())
    
    def _get_memory_info(self) -> dict:
        """Get detailed memory information."""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024 * 1024),  # Convert to MB
            'available': memory.available / (1024 * 1024),
            'used': memory.used / (1024 * 1024),
            'percent': memory.percent,
            'available_percent': (memory.available / memory.total) * 100
        }
    
    def _log_memory_state(self, context: str):
        """Log detailed memory state."""
        mem_info = self._get_memory_info()
        logging.info(f"Memory state ({context}):")
        logging.info(f"- Total: {mem_info['total']:.2f} MB")
        logging.info(f"- Available: {mem_info['available']:.2f} MB ({mem_info['available_percent']:.2f}%)")
        logging.info(f"- Used: {mem_info['used']:.2f} MB ({mem_info['percent']}%)")
        logging.info(f"- Active workers: {self.active_workers}/{self.max_workers}")
        logging.info(f"- Queue size: {self.processing_queue.qsize()}/{self.processing_queue.maxsize}")
    
    async def _periodic_cleanup(self):
        """Periodically clean up memory and temp files."""
        while True:
            try:
                current_time = time.time()
                if current_time - self.last_cleanup_time >= self.cleanup_interval:
                    self._force_cleanup()
                    self.last_cleanup_time = current_time
                await asyncio.sleep(1)
            except Exception as e:
                logging.error(f"Error in periodic cleanup: {str(e)}")
                await asyncio.sleep(5)
    
    def _force_cleanup(self):
        """Force cleanup of memory and temp files."""
        try:
            # Clean up temp directory
            temp_dir = "temp"
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    try:
                        file_path = os.path.join(temp_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logging.warning(f"Failed to remove temp file {file}: {str(e)}")
            
            # Force garbage collection
            force_garbage_collection()
            
            # Log memory state after cleanup
            self._log_memory_state("After cleanup")
        except Exception as e:
            logging.error(f"Error in force cleanup: {str(e)}")
    
    def _get_available_memory_percent(self) -> float:
        """Get available memory as a percentage."""
        memory = psutil.virtual_memory()
        return memory.available / memory.total
    
    def _should_start_new_worker(self) -> bool:
        """Check if we should start a new worker based on memory usage."""
        available_memory = self._get_available_memory_percent()
        should_start = available_memory > (1 - self.memory_threshold)
        if not should_start:
            self._log_memory_state("Worker start check failed")
        return should_start
    
    def _is_memory_critical(self) -> bool:
        """Check if memory usage is at critical levels."""
        available_memory = self._get_available_memory_percent()
        is_critical = available_memory < (1 - self.critical_memory_threshold)
        if is_critical:
            self._log_memory_state("Critical memory detected")
        return is_critical
    
    async def _process_queue(self):
        """Process queued requests using multiple workers."""
        while True:
            try:
                # Check if memory is critical
                if self._is_memory_critical():
                    logging.warning("Critical memory usage detected, pausing queue processing")
                    self._force_cleanup()  # Try to free up memory
                    await asyncio.sleep(5)  # Wait longer when memory is critical
                    continue
                
                # Check if we can start more workers
                while (self.active_workers < self.max_workers and 
                       not self.processing_queue.empty() and 
                       self._should_start_new_worker()):
                    request = await self.processing_queue.get()
                    self.active_workers += 1
                    logging.info(f"Starting worker {self.active_workers} for request {request.request_id}")
                    self._log_memory_state(f"Starting worker {self.active_workers}")
                    
                    # Process request in background
                    asyncio.create_task(self._process_with_worker(request))
                
                await asyncio.sleep(0.1)  # Prevent CPU spinning
            except Exception as e:
                logging.error(f"Error in queue processor: {str(e)}")
                logging.error(traceback.format_exc())
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _process_with_worker(self, request: ProcessingRequest):
        """Process a single request using a worker."""
        try:
            # Check memory before processing
            if self._is_memory_critical():
                logging.warning(f"Critical memory detected, rejecting request {request.request_id}")
                self._force_cleanup()  # Try to free up memory
                if self._is_memory_critical():  # Check again after cleanup
                    request.result = JSONResponse(
                        status_code=503,
                        content={
                            "status": "error",
                            "message": "Server is currently under heavy load. Please try again in a few moments.",
                            "type": "ServiceUnavailable"
                        }
                    )
                    return
            
            if not self._should_start_new_worker():
                logging.warning(f"Low memory detected, delaying request {request.request_id}")
                self._force_cleanup()  # Try to free up memory
                await asyncio.sleep(2)  # Wait longer for memory to free up
                if not self._should_start_new_worker():
                    request.result = JSONResponse(
                        status_code=503,
                        content={
                            "status": "error",
                            "message": "Server is currently under heavy load. Please try again in a few moments.",
                            "type": "ServiceUnavailable"
                        }
                    )
                    return
            
            result = await self._process_single_request(request)
            request.result = result
        except Exception as e:
            logging.error(f"Error in worker processing request {request.request_id}: {str(e)}")
            request.result = JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": str(e),
                    "type": type(e).__name__
                }
            )
        finally:
            self.active_workers -= 1
            self.processing_queue.task_done()
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
            # Force cleanup after each request
            self._force_cleanup()
    
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
                    "traceback": traceback.format_exc()
                }
            )
        finally:
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]

    async def validate_and_save_file(self, file: UploadFile, temp_input: str) -> tuple[bool, JSONResponse | None]:
        """Validate and save the uploaded file."""
        try:
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
        self._log_memory_state("New request received")
        
        # Check memory before accepting new request
        if self._is_memory_critical():
            self._force_cleanup()  # Try to free up memory
            if self._is_memory_critical():  # Check again after cleanup
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "error",
                        "message": "Server is currently under heavy load. Please try again in a few moments.",
                        "type": "ServiceUnavailable"
                    }
                )
        
        if not self._should_start_new_worker():
            self._force_cleanup()  # Try to free up memory
            if not self._should_start_new_worker():  # Check again after cleanup
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "error",
                        "message": "Server is currently under heavy load. Please try again in a few moments.",
                        "type": "ServiceUnavailable"
                    }
                )
        
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
            self._log_memory_state("Queue full")
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
            self._log_memory_state(f"Request {request_id} queued")
            
            # Wait for processing to complete with timeout
            timeout_seconds = 30  # Reduced timeout to 30 seconds
            start_time = time.time()
            while request_id in self.active_requests:
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError("Request processing timed out")
                await asyncio.sleep(0.1)
            
            # Return the result
            return request.result
            
        except TimeoutError:
            logging.error(f"Request {request_id} timed out after {timeout_seconds} seconds")
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            return JSONResponse(
                status_code=504,
                content={
                    "status": "error",
                    "message": "Request processing timed out. Please try again.",
                    "type": "TimeoutError"
                }
            )
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