from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import time
from collections import defaultdict
import asyncio
from typing import Dict
import logging

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = defaultdict(list)
        self.lock = asyncio.Lock()
        
    async def check_rate_limit(self, request: Request) -> bool:
        client_ip = request.client.host
        current_time = time.time()
        
        async with self.lock:
            # Clean old requests
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < 60
            ]
            
            # Check if rate limit is exceeded
            if len(self.requests[client_ip]) >= self.requests_per_minute:
                return False
                
            # Add new request
            self.requests[client_ip].append(current_time)
            return True

rate_limiter = RateLimiter()

async def rate_limit_middleware(request: Request, call_next):
    if not await rate_limiter.check_rate_limit(request):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please try again in a minute."}
        )
    
    return await call_next(request) 