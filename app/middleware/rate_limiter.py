from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import time
import os
from collections import defaultdict
import asyncio
from typing import Dict
import logging

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60, process_image_per_minute: int = 20):
        # Detect Railway environment and adjust limits
        is_railway = os.getenv("RAILWAY_ENVIRONMENT") is not None
        
        if is_railway:
            # Much more restrictive limits for Railway
            self.requests_per_minute = min(requests_per_minute, 30)
            self.process_image_per_minute = min(process_image_per_minute, 3)  # Very restrictive
            logging.info("Railway environment detected - using restrictive rate limits")
        else:
            self.requests_per_minute = requests_per_minute
            self.process_image_per_minute = process_image_per_minute
        
        self.requests: Dict[str, list] = defaultdict(list)
        self.process_image_requests: Dict[str, list] = defaultdict(list)
        self.lock = asyncio.Lock()
        
        logging.info(f"Rate limiter initialized - General: {self.requests_per_minute}/min, Process Image: {self.process_image_per_minute}/min")
        
    async def check_rate_limit(self, request: Request) -> bool:
        client_ip = request.client.host
        current_time = time.time()
        
        async with self.lock:
            # Check if this is a process-image request
            is_process_image = "/process-image" in str(request.url)
            
            if is_process_image:
                # Clean old process-image requests
                self.process_image_requests[client_ip] = [
                    req_time for req_time in self.process_image_requests[client_ip]
                    if current_time - req_time < 60
                ]
                
                # Check if process-image rate limit is exceeded
                if len(self.process_image_requests[client_ip]) >= self.process_image_per_minute:
                    logging.warning(f"Rate limit exceeded for /process-image from {client_ip}: {len(self.process_image_requests[client_ip])}/{self.process_image_per_minute}")
                    return False
                    
                # Add new process-image request
                self.process_image_requests[client_ip].append(current_time)
            
            # Clean old general requests
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < 60
            ]
            
            # Check if general rate limit is exceeded
            if len(self.requests[client_ip]) >= self.requests_per_minute:
                logging.warning(f"General rate limit exceeded from {client_ip}: {len(self.requests[client_ip])}/{self.requests_per_minute}")
                return False
                
            # Add new request
            self.requests[client_ip].append(current_time)
            return True

rate_limiter = RateLimiter()

async def rate_limit_middleware(request: Request, call_next):
    if not await rate_limiter.check_rate_limit(request):
        is_railway = os.getenv("RAILWAY_ENVIRONMENT") is not None
        return JSONResponse(
            status_code=429,
            content={
                "status": "error",
                "code": 429,
                "message": "Too many requests. Please try again in a minute." + 
                         (" Railway limits are stricter for stability." if is_railway else ""),
                "request_id": str(time.time()),
                "environment": "railway" if is_railway else "other"
            }
        )
    
    return await call_next(request) 