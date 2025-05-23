import os
from inference import run
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Set paths
input_dir = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/input_images"
output_dir = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/output_images"
retinaface_model_path = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/retinaface/RetinaFace-R50.pth"
modnet_model_path = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/modnet_photographic_portrait_matting/modnet_photographic_portrait_matting.ckpt"
onnx_model_path = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/hivision/creator/weights/birefnet-v1-lite.onnx"
# onnx_model_path = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/hivision/creator/weights/modnet_photographic_portrait_matting.onnx"

# Check model files
if not os.path.isfile(retinaface_model_path):
    raise FileNotFoundError(f"RetinaFace model not found at: {retinaface_model_path}")

# Determine matting model
if os.path.isfile(onnx_model_path):
    matting_model = "birefnet-v1-lite"
    # matting_model = "modnet_photographic_portrait_matting"
    print('using first!!!')
elif os.path.isfile(modnet_model_path):
    matting_model = "birefnet-v1-lite"
    # matting_model = "modnet_photographic_portrait_matting"
    print(f"Warning: Using .ckpt model at {modnet_model_path}. Ensure compatibility.")
else:
    print("Warning: MODNet model not found. Falling back to hivision_modnet.")
    matting_model = "hivision_modnet"
    onnx_model_path = "hivision/creator/weights/hivision_modnet.onnx"
    if not os.path.isfile(onnx_model_path):
        raise FileNotFoundError(f"Fallback model not found at: {onnx_model_path}")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to process a single image
def process_image(args):
    filename, matting_model = args
    if filename.lower().endswith((".jpg", ".jpeg")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpg")

        try:
            print(f"Processing {filename}...")
            run(
                input_image_path=input_path,
                output_image_path=output_path,
                type="idphoto",
                height=288,
                width=240,
                color="fafafa",
                hd=True,
                kb=None,
                render=0,
                dpi=300,
                face_align=False,
                matting_model=matting_model,
                face_detect_model="retinaface-resnet50",
            )
            print(f"✅ Saved: {output_path}")
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

# Main block to use multiprocessing
if __name__ == '__main__':
    filenames = [
        f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg"))
    ]

    args_list = [(filename, matting_model) for filename in filenames]

    # Use 2 processes (as you requested)
    with ProcessPoolExecutor(max_workers=2) as executor:
        executor.map(process_image, args_list)

    # List all output files
    print("\nListing output files:")
    os.system(f"ls -l {output_dir}")