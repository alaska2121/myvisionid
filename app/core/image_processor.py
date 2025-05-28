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
from typing import Dict, Optional, Any
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
        self.processing_queue = asyncio.Queue(maxsize=25)  # Reduced queue size
        self.active_requests: Dict[str, ProcessingRequest] = {}
        self.active_workers = 0  # Track number of active workers
        self.max_workers = 1  # Limit to 1 worker for Railway 8GB plan
        self.memory_threshold = 0.3  # 30% memory threshold
        self.critical_memory_threshold = 0.5  # 50% critical threshold
        self.max_memory_mb = 7000  # 7GB max (leaving 1GB buffer)
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 3  # Cleanup every 3 seconds
        
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        logging.info(f"Initialized ImageProcessor with {self.max_workers} workers")
        logging.info(f"Memory limits: {self.max_memory_mb}MB max, {self.memory_threshold*100}% threshold, {self.critical_memory_threshold*100}% critical")
        
        # Start the queue processor
        asyncio.create_task(self._process_queue())
        # Start the cleanup task
        asyncio.create_task(self._periodic_cleanup())
    
    def _get_memory_info(self) -> dict:
        """Get detailed memory information."""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info().rss / (1024 * 1024)  # Process memory in MB
        return {
            'total': memory.total / (1024 * 1024),  # Convert to MB
            'available': memory.available / (1024 * 1024),
            'used': memory.used / (1024 * 1024),
            'percent': memory.percent,
            'available_percent': (memory.available / memory.total) * 100,
            'process_memory': process_memory
        }
    
    def _log_memory_state(self, context: str):
        """Log detailed memory state."""
        mem_info = self._get_memory_info()
        logging.info(f"Memory state ({context}):")
        logging.info(f"- Total: {mem_info['total']:.2f} MB")
        logging.info(f"- Available: {mem_info['available']:.2f} MB ({mem_info['available_percent']:.2f}%)")
        logging.info(f"- Used: {mem_info['used']:.2f} MB ({mem_info['percent']}%)")
        logging.info(f"- Process memory: {mem_info['process_memory']:.2f} MB")
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
            
            # Force garbage collection multiple times
            for _ in range(3):
                force_garbage_collection()
                time.sleep(0.1)  # Small delay between GC calls
            
            # Log memory state after cleanup
            self._log_memory_state("After cleanup")
        except Exception as e:
            logging.error(f"Error in force cleanup: {str(e)}")
    
    def _should_start_new_worker(self) -> bool:
        """Check if we should start a new worker based on memory usage."""
        mem_info = self._get_memory_info()
        available_memory = mem_info['available_percent'] / 100
        process_memory = mem_info['process_memory']
        
        # Check both percentage and absolute memory
        # We should start if we have enough available memory (above threshold)
        # and our process memory is below the maximum
        should_start = (
            available_memory >= self.memory_threshold and  # Changed from > (1 - threshold)
            process_memory < self.max_memory_mb
        )
        
        if not should_start:
            self._log_memory_state("Worker start check failed")
            logging.warning(f"Memory check failed: available={available_memory:.2%}, process={process_memory:.0f}MB")
        else:
            logging.info(f"Memory check passed: available={available_memory:.2%}, process={process_memory:.0f}MB")
        
        return should_start
    
    def _is_memory_critical(self) -> bool:
        """Check if memory usage is at critical levels."""
        mem_info = self._get_memory_info()
        available_memory = mem_info['available_percent'] / 100
        process_memory = mem_info['process_memory']
        
        # Check both percentage and absolute memory
        # Memory is critical if available memory is below critical threshold
        # or process memory exceeds maximum
        is_critical = (
            available_memory < self.critical_memory_threshold or  # Changed from < (1 - threshold)
            process_memory > self.max_memory_mb
        )
        
        if is_critical:
            self._log_memory_state("Critical memory detected")
            logging.warning(f"Critical memory: available={available_memory:.2%}, process={process_memory:.0f}MB")
        else:
            logging.info(f"Memory state healthy: available={available_memory:.2%}, process={process_memory:.0f}MB")
        
        return is_critical
    
    async def _process_queue(self):
        """Process the queue of requests."""
        while True:
            try:
                # Check memory before processing next request
                if self._is_memory_critical():
                    logging.warning("Memory critical, waiting before processing next request")
                    await asyncio.sleep(5)  # Wait longer when memory is critical
                    self._force_cleanup()
                    continue
                
                request = await self.processing_queue.get()
                
                if self.active_workers >= self.max_workers:
                    logging.warning("Max workers reached, waiting")
                    await asyncio.sleep(1)
                    continue
                
                if not self._should_start_new_worker():
                    logging.warning("Memory too high for new worker, waiting")
                    await asyncio.sleep(2)
                    continue
                
                asyncio.create_task(self._process_with_worker(request))
                
            except Exception as e:
                logging.error(f"Error in queue processor: {str(e)}")
                await asyncio.sleep(1)
    
    async def _process_with_worker(self, request: ProcessingRequest):
        """Process a request with a worker."""
        try:
            self.active_workers += 1
            self._log_memory_state("Before processing")
            
            # Check memory before starting
            if self._is_memory_critical():
                raise Exception("Memory usage too high to process request")
            
            # Process the request using run_in_executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                lambda: run(
                    input_image_path=request.temp_input,
                    output_image_path=request.temp_output,
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
            
            # Force cleanup after processing
            self._force_cleanup()
            
            # Check memory after processing and cleanup
            if self._is_memory_critical():
                # Try one more aggressive cleanup
                self._force_cleanup()
                if self._is_memory_critical():
                    raise Exception("Memory usage too high after processing")
            
            request.result = result
            
        except Exception as e:
            logging.error(f"Error processing request: {str(e)}")
            request.result = JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": str(e),
                    "type": type(e).__name__
                }
            )
            # Force cleanup on error
            self._force_cleanup()
            
        finally:
            self.active_workers -= 1
            self.processing_queue.task_done()
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
            # Always cleanup after processing
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
            
            # Ensure temp directory exists with proper permissions
            temp_dir = os.path.dirname(temp_input)
            try:
                os.makedirs(temp_dir, mode=0o755, exist_ok=True)
                logging.info(f"Ensured temp directory exists: {temp_dir}")
            except Exception as dir_error:
                logging.error(f"Failed to create temp directory: {str(dir_error)}")
                return False, JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "message": "Failed to create temporary directory."
                    }
                )
            
            # Save file with retries and verification
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logging.info(f"Saving to {temp_input} (attempt {attempt + 1}/{max_retries})")
                    
                    # Save file using aiofiles
                    async with aiofiles.open(temp_input, 'wb') as f:
                        await f.write(content)
                        await f.flush()  # Ensure content is written to disk
                    
                    # Force sync to disk
                    os.sync()
                    
                    # Verify file exists
                    if not os.path.exists(temp_input):
                        raise FileNotFoundError(f"File not found after save: {temp_input}")
                    
                    # Verify file permissions
                    if not os.access(temp_input, os.R_OK):
                        raise PermissionError(f"No read permission for file: {temp_input}")
                    
                    # Verify file size
                    file_size = os.path.getsize(temp_input)
                    if file_size != len(content):
                        raise ValueError(f"File size mismatch: saved={file_size}, original={len(content)}")
                    
                    # Verify file is readable
                    try:
                        with open(temp_input, 'rb') as f:
                            test_content = f.read()
                        if len(test_content) != len(content):
                            raise ValueError("File content verification failed")
                    except Exception as read_error:
                        raise IOError(f"Failed to read saved file: {str(read_error)}")
                    
                    # Verify it's a valid image
                    import cv2
                    test_image = cv2.imread(temp_input)
                    if test_image is None:
                        raise ValueError(f"Failed to read image with OpenCV: {temp_input}")
                    
                    logging.info(f"Image verification successful. Image shape: {test_image.shape}")
                    logging.info(f"File successfully saved and verified at: {temp_input}")
                    
                    # Set file permissions to ensure readability
                    os.chmod(temp_input, 0o644)
                    
                    # Final verification before returning
                    if not os.path.exists(temp_input):
                        raise FileNotFoundError("File disappeared after verification")
                    
                    return True, None
                    
                except Exception as save_error:
                    logging.warning(f"Attempt {attempt + 1} failed: {str(save_error)}")
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(0.1)  # Small delay before retry
            
            return False, JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Failed to save and verify file after multiple attempts."
                }
            )
            
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
            # Verify input file exists and is readable before processing
            if not os.path.exists(temp_input):
                raise FileNotFoundError(f"Input file not found: {temp_input}")
            
            if not os.access(temp_input, os.R_OK):
                raise PermissionError(f"No read permission for input file: {temp_input}")
            
            # Verify file is a valid image
            import cv2
            test_image = cv2.imread(temp_input)
            if test_image is None:
                raise ValueError(f"Failed to read input image with OpenCV: {temp_input}")
            
            logging.info(f"Input image verified. Shape: {test_image.shape}")
            
            # Ensure file is flushed to disk
            os.sync()
            
            loop = asyncio.get_event_loop()
            with timeout(50):  # 50 second timeout
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
            logging.error("Image processing timed out after 50 seconds")
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
        temp_dir = "temp"
        try:
            os.makedirs(temp_dir, mode=0o755, exist_ok=True)
            logging.info(f"Ensured temp directory exists: {temp_dir}")
        except Exception as dir_error:
            logging.error(f"Failed to create temp directory: {str(dir_error)}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Failed to create temporary directory."
                }
            )
        
        # Generate request ID and temp file paths
        request_id = str(uuid.uuid4())
        temp_input = os.path.join(temp_dir, f"{request_id}_input.jpg")
        temp_output = os.path.join(temp_dir, f"{request_id}_output.jpg")
        
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
            # Validate and save the file
            success, error_response = await self.validate_and_save_file(file, temp_input)
            if not success:
                if request_id in self.active_requests:
                    del self.active_requests[request_id]
                return error_response
            
            # Verify file exists and is readable before adding to queue
            if not os.path.exists(temp_input):
                raise FileNotFoundError(f"Input file not found after validation: {temp_input}")
            
            if not os.access(temp_input, os.R_OK):
                raise PermissionError(f"No read permission for input file: {temp_input}")
            
            # Verify file is a valid image
            import cv2
            test_image = cv2.imread(temp_input)
            if test_image is None:
                raise ValueError(f"Failed to read input image with OpenCV: {temp_input}")
            
            logging.info(f"Input image verified before queue. Shape: {test_image.shape}")
            
            # Ensure file is flushed to disk
            os.sync()
            
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