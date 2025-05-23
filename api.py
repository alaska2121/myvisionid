from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import os
from inference import run
import io

app = FastAPI()

# Set paths
retinaface_model_path = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/retinaface/RetinaFace-R50.pth"
modnet_model_path = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/modnet_photographic_portrait_matting/modnet_photographic_portrait_matting.ckpt"
onnx_model_path = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/hivision/creator/weights/birefnet-v1-lite.onnx"

# Check model files
if not os.path.isfile(retinaface_model_path):
    raise FileNotFoundError(f"RetinaFace model not found at: {retinaface_model_path}")

# Determine matting model
if os.path.isfile(onnx_model_path):
    matting_model = "birefnet-v1-lite"
    print("Using birefnet-v1-lite model")
elif os.path.isfile(modnet_model_path):
    matting_model = "birefnet-v1-lite"
    print("Using birefnet-v1-lite model from modnet path")
else:
    print("Warning: MODNet model not found. Falling back to hivision_modnet.")
    matting_model = "hivision_modnet"
    onnx_model_path = "hivision/creator/weights/hivision_modnet.onnx"
    if not os.path.isfile(onnx_model_path):
        raise FileNotFoundError(f"Fallback model not found at: {onnx_model_path}")

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    """
    Process an uploaded image by adding a pure white background
    """
    print(f"Received file: {file.filename}")
    
    # Create temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    
    # Save uploaded file temporarily
    temp_input = "temp/temp_input.jpg"
    temp_output = "temp/temp_output.jpg"
    
    try:
        # Save uploaded file
        print("Reading uploaded file...")
        content = await file.read()
        print(f"File size: {len(content)} bytes")
        
        print(f"Saving to {temp_input}")
        with open(temp_input, "wb") as f:
            f.write(content)
        
        print("Processing image...")
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
        
        print("Reading processed image...")
        # Read the processed image
        with open(temp_output, "rb") as f:
            processed_image = f.read()
        
        print(f"Processed image size: {len(processed_image)} bytes")
            
        # Clean up temp files
        os.remove(temp_input)
        os.remove(temp_output)
        
        print("Returning response...")
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
        print(f"Error occurred: {str(e)}")
        # Clean up temp files in case of error
        if os.path.exists(temp_input):
            os.remove(temp_input)
        if os.path.exists(temp_output):
            os.remove(temp_output)
        raise e 