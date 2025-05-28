from fastapi import Request
import time
import logging

async def request_logger_middleware(request: Request, call_next):
    # Log request start
    start_time = time.time()
    logging.info(f"Request started: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log request end
    process_time = time.time() - start_time
    logging.info(
        f"Request completed: {request.method} {request.url.path} "
        f"- Status: {response.status_code} - Time: {process_time:.2f}s"
    )
    
    return response 