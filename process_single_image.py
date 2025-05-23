import os
import cv2
import numpy as np
from hivision import IDCreator
from hivision.utils import add_background, save_image_dpi_to_bytes
from hivision.creator.choose_handler import choose_handler
import io

# Set model paths
retinaface_model_path = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/retinaface/RetinaFace-R50.pth"
modnet_model_path = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/modnet_photographic_portrait_matting/modnet_photographic_portrait_matting.ckpt"
onnx_model_path = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/hivision/creator/weights/birefnet-v1-lite.onnx"

# Check model files
if not os.path.isfile(retinaface_model_path):
    raise FileNotFoundError(f"RetinaFace model not found at: {retinaface_model_path}")

# Determine matting model
if os.path.isfile(onnx_model_path):
    default_matting_model = "birefnet-v1-lite"
elif os.path.isfile(modnet_model_path):
    default_matting_model = "birefnet-v1-lite"
else:
    print("Warning: MODNet model not found. Falling back to hivision_modnet.")
    default_matting_model = "hivision_modnet"
    onnx_model_path = "hivision/creator/weights/hivision_modnet.onnx"
    if not os.path.isfile(onnx_model_path):
        raise FileNotFoundError(f"Fallback model not found at: {onnx_model_path}")

def process_single_image(image_bytes, matting_model=None, face_detect_model="retinaface-resnet50"):
    """
    Process a single image from bytes, adding a pure white background.
    
    Args:
        image_bytes (bytes): The input image in bytes format
        matting_model (str): The matting model to use (default: determined automatically)
        face_detect_model (str): The face detection model to use (default: retinaface-resnet50)
    
    Returns:
        bytes: The processed image in JPEG format
    """
    if matting_model is None:
        matting_model = default_matting_model
        
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    input_image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    # Initialize IDCreator for matting
    creator = IDCreator()
    choose_handler(creator, matting_model, face_detect_model)
    
    # Perform human matting
    result = creator(input_image, change_bg_only=True)
    matted_image = result.hd  # Use HD matted output (RGBA)
    
    # Pure white background (BGR format)
    white_bgr = (255, 255, 255)
    
    # Apply white background
    result_image = add_background(
        matted_image, 
        bgr=white_bgr, 
        mode="pure_color"
    )
    result_image = result_image.astype(np.uint8)
    
    # Ensure no alpha channel (convert to BGR for solid background)
    if result_image.shape[2] == 4:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGBA2BGR)
    else:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    
    # Create a buffer to store the output image
    buffer = io.BytesIO()
    
    # Save the image to the buffer with DPI setting
    save_image_dpi_to_bytes(result_image, buffer, dpi=300)
    
    # Get the buffer content
    processed_image_bytes = buffer.getvalue()
    buffer.close()
    
    return processed_image_bytes 