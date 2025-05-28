from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.middleware.rate_limiter import rate_limit_middleware
from app.middleware.error_handler import error_handler_middleware
from app.middleware.request_logger import request_logger_middleware

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

# Add custom middleware
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(error_handler_middleware)
app.middleware("http")(request_logger_middleware)

# Include routes
app.include_router(router)

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"} 