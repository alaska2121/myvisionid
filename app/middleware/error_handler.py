from fastapi import Request
from fastapi.responses import JSONResponse
import logging
import traceback

async def error_handler_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        # Log the error
        logging.error(f"Error processing request: {str(e)}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())
        
        # Return a user-friendly error response
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "An error occurred while processing your request. Please try again later.",
                "type": type(e).__name__
            }
        ) 