import asyncio
import time
from fastapi import Request
from fastapi.responses import JSONResponse
import logging

async def timeout_middleware(request: Request, call_next):
    """Middleware to timeout requests after a certain duration."""
    
    # Set different timeouts for different endpoints
    if "/process-image" in str(request.url):
        timeout_seconds = 60  # 60 seconds for image processing
    else:
        timeout_seconds = 30  # 30 seconds for other endpoints
    
    start_time = time.time()
    
    try:
        # Wait for the request to complete or timeout
        response = await asyncio.wait_for(
            call_next(request),
            timeout=timeout_seconds
        )
        
        elapsed_time = time.time() - start_time
        logging.info(f"Request completed in {elapsed_time:.2f} seconds")
        
        return response
        
    except asyncio.TimeoutError:
        elapsed_time = time.time() - start_time
        logging.error(f"Request timed out after {elapsed_time:.2f} seconds")
        
        return JSONResponse(
            status_code=504,
            content={
                "status": "error",
                "code": 504,
                "message": f"Request timed out after {timeout_seconds} seconds",
                "request_id": str(time.time())
            }
        )
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"Request failed after {elapsed_time:.2f} seconds: {str(e)}")
        raise 