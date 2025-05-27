from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse
import os
from inference import run
import io
import traceback
import sys
import logging
from datetime import datetime

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

app = FastAPI()

# Set paths relative to the app directory
retinaface_model_path = "retinaface/RetinaFace-R50.pth"
modnet_model_path = "modnet_photographic_portrait_matting/modnet_photographic_portrait_matting.ckpt"
onnx_model_path = "hivision/creator/weights/birefnet-v1-lite.onnx"

# Add detailed logging for model files
logging.info("Checking model files...")
logging.info(f"Current working directory: {os.getcwd()}")
logging.info(f"Directory contents: {os.listdir('.')}")

# Check model files
if not os.path.isfile(retinaface_model_path):
    logging.error(f"RetinaFace model not found at: {retinaface_model_path}")
    logging.error(f"Directory contents of retinaface/: {os.listdir('retinaface') if os.path.exists('retinaface') else 'Directory not found'}")
    raise FileNotFoundError(f"RetinaFace model not found at: {retinaface_model_path}")
else:
    logging.info(f"RetinaFace model found at: {retinaface_model_path}")

# Determine matting model
if os.path.isfile(onnx_model_path):
    matting_model = "birefnet-v1-lite"
    logging.info(f"Using birefnet-v1-lite model at: {onnx_model_path}")
elif os.path.isfile(modnet_model_path):
    matting_model = "birefnet-v1-lite"
    logging.info(f"Using birefnet-v1-lite model from modnet path: {modnet_model_path}")
else:
    logging.warning("Warning: MODNet model not found. Falling back to hivision_modnet.")
    matting_model = "hivision_modnet"
    onnx_model_path = "hivision/creator/weights/hivision_modnet.onnx"
    if not os.path.isfile(onnx_model_path):
        logging.error(f"Fallback model not found at: {onnx_model_path}")
        logging.error(f"Directory contents of hivision/creator/weights/: {os.listdir('hivision/creator/weights') if os.path.exists('hivision/creator/weights') else 'Directory not found'}")
        raise FileNotFoundError(f"Fallback model not found at: {onnx_model_path}")
    else:
        logging.info(f"Fallback model found at: {onnx_model_path}")

@app.get("/process-image")
async def process_image_get():
    """
    GET endpoint for process-image that returns usage instructions
    """
    return JSONResponse(
        content={
            "status": "info",
            "message": "This endpoint only accepts POST requests. Please send your image as a POST request with the image file in the request body.",
            "example": "curl -X POST -F 'file=@your_image.jpg' https://myvisionid-production.up.railway.app/process-image"
        }
    )

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    """
    Process an uploaded image by adding a pure white background
    """
    logging.info(f"Received file: {file.filename}")
    
    # Create temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    
    # Save uploaded file temporarily
    temp_input = "temp/temp_input.jpg"
    temp_output = "temp/temp_output.jpg"
    
    try:
        # Save uploaded file
        logging.info("Reading uploaded file...")
        content = await file.read()
        logging.info(f"File size: {len(content)} bytes")
        
        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            logging.warning(f"File too large: {len(content)} bytes")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "File too large. Maximum size is 10MB."
                }
            )
        
        logging.info(f"Saving to {temp_input}")
        with open(temp_input, "wb") as f:
            f.write(content)
        
        logging.info("Processing image...")
        try:
            # Process using the same function as addbackground_multiprocess
            run(
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
                matting_model=matting_model,
                face_detect_model="retinaface-resnet50",
            )
        except Exception as processing_error:
            logging.error(f"Error during image processing: {str(processing_error)}")
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Error during image processing: {str(processing_error)}",
                    "type": type(processing_error).__name__
                }
            )
        
        logging.info("Reading processed image...")
        try:
            # Read the processed image
            with open(temp_output, "rb") as f:
                processed_image = f.read()
            
            logging.info(f"Processed image size: {len(processed_image)} bytes")
        except Exception as read_error:
            logging.error(f"Error reading processed image: {str(read_error)}")
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Error reading processed image: {str(read_error)}",
                    "type": type(read_error).__name__
                }
            )
            
        # Clean up temp files
        try:
            os.remove(temp_input)
            os.remove(temp_output)
        except Exception as cleanup_error:
            logging.warning(f"Warning: Error during cleanup: {str(cleanup_error)}")
        
        logging.info("Returning response...")
        # Return the processed image
        return Response(
            content=processed_image,
            media_type="image/jpeg",
            headers={
                "Content-Type": "image/jpeg",
                "Content-Length": str(len(processed_image))
            }
        )
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())
        # Clean up temp files in case of error
        try:
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(temp_output):
                os.remove(temp_output)
        except Exception as cleanup_error:
            logging.warning(f"Warning: Error during cleanup: {str(cleanup_error)}")
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