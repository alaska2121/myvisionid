from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.middleware.rate_limiter import rate_limit_middleware
from app.middleware.error_handler import error_handler_middleware
from app.middleware.request_logger import request_logger_middleware
from app.middleware.timeout import timeout_middleware
import psutil
import os

app = FastAPI(
    title="Background Removal API",
    description="API for removing backgrounds from student photos",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware (order matters - timeout should be first)
app.middleware("http")(timeout_middleware)
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(error_handler_middleware)
app.middleware("http")(request_logger_middleware)

# Include routes
app.include_router(router)

# Enhanced health check endpoint with memory monitoring
@app.get("/health")
async def health_check():
    try:
        # Check memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Get memory threshold based on environment
        is_railway = os.getenv("RAILWAY_ENVIRONMENT") is not None
        threshold_mb = int(os.getenv("MEMORY_THRESHOLD_MB", "800"))
        
        if is_railway:
            threshold_mb = min(threshold_mb, 400)
        
        # Check if memory is within limits
        memory_status = "healthy" if memory_mb < threshold_mb else "warning"
        
        return {
            "status": "healthy",
            "memory": {
                "current_mb": round(memory_mb, 1),
                "threshold_mb": threshold_mb,
                "status": memory_status
            },
            "environment": "railway" if is_railway else "local"
        }
    except Exception as e:
        return {
            "status": "healthy", 
            "memory": {
                "error": str(e)
            }
        } 