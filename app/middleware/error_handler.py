from fastapi import Request
from fastapi.responses import JSONResponse
import logging
import traceback
import asyncio
import time

async def error_handler_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except asyncio.TimeoutError:
        logging.error("Request timed out")
        return JSONResponse(
            status_code=504,
            content={
                "status": "error",
                "code": 504,
                "message": "Request timed out. The server took too long to process your request.",
                "request_id": str(time.time())
            }
        )
    except MemoryError:
        logging.error("Out of memory error")
        return JSONResponse(
            status_code=507,
            content={
                "status": "error",
                "code": 507,
                "message": "Server is out of memory. Please try again with a smaller image or try again later.",
                "request_id": str(time.time())
            }
        )
    except FileNotFoundError as e:
        logging.error(f"File not found: {str(e)}")
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "code": 404,
                "message": "Required file not found. Please check your request.",
                "request_id": str(time.time())
            }
        )
    except Exception as e:
        # Log the error
        logging.error(f"Error processing request: {str(e)}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())
        
        # Check if it's a connection error
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            return JSONResponse(
                status_code=502,
                content={
                    "status": "error",
                    "code": 502,
                    "message": "Application failed to respond",
                    "request_id": str(time.time())
                }
            )
        
        # Return a user-friendly error response
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "code": 500,
                "message": "An error occurred while processing your request. Please try again later.",
                "type": type(e).__name__,
                "request_id": str(time.time())
            }
        ) 