import asyncio
import time
import os
from fastapi import Request
from fastapi.responses import JSONResponse
import logging

async def timeout_middleware(request: Request, call_next):
    """Middleware to timeout requests after a certain duration."""
    
    # Detect Railway environment
    is_railway = os.getenv("RAILWAY_ENVIRONMENT") is not None
    
    # Set different timeouts for different endpoints and environments
    if "/process-image" in str(request.url):
        if is_railway:
            timeout_seconds = 45  # Shorter timeout for Railway
        else:
            timeout_seconds = 60  # 60 seconds for local/other environments
    else:
        timeout_seconds = 15 if is_railway else 30  # Shorter for Railway
    
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
        logging.error(f"Request timed out after {elapsed_time:.2f} seconds (Railway: {is_railway})")
        
        return JSONResponse(
            status_code=504,
            content={
                "status": "error",
                "code": 504,
                "message": f"Request timed out after {timeout_seconds} seconds. Please try with a smaller image or try again later.",
                "request_id": str(time.time()),
                "environment": "railway" if is_railway else "other"
            }
        )
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"Request failed after {elapsed_time:.2f} seconds: {str(e)}")
        raise 