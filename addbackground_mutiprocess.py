import os
from inference import run
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing
import random

# Set paths
input_dir = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/input_images"
output_dir = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/output_images"
retinaface_model_path = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/retinaface/RetinaFace-R50.pth"
modnet_model_path = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/modnet_photographic_portrait_matting/modnet_photographic_portrait_matting.ckpt"
onnx_model_path = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/hivision/creator/weights/birefnet-v1-lite.onnx"
# onnx_model_path = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/hivision/creator/weights/modnet_photographic_portrait_matting.onnx"

# Fixed list of colors (20 shades)
fixed_colors = [

]

# Check if model files exist
if not os.path.isfile(retinaface_model_path):
    raise FileNotFoundError(f"RetinaFace model not found at: {retinaface_model_path}")

# Check for matting model
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

# Load the model once in the main process
def load_models():
    print("Loading models...")
    return matting_model

# Function to generate a random color within the specified range
def generate_random_color(min_val=220, max_val=255):
    while True:
        red = random.randint(min_val, max_val)
        green = random.randint(min_val, max_val)
        blue = random.randint(min_val, max_val)

        if abs(red - green) <= 4 and abs(green - blue) <= 4 and abs(red - blue) <= 4:
            return f"#{red:02X}{green:02X}{blue:02X}"

# Function to process a single image
def process_image(args):
    filename, matting_model, index = args

    if filename.lower().endswith((".jpg", ".jpeg")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpg")

        # Select fixed color if available, else random color
        if index < len(fixed_colors):
            selected_color = fixed_colors[index]
        else:
            selected_color = generate_random_color()

        # Alternate render value 1 and 2
        render_value = (index % 2) + 1

        try:
            print(f"Processing {filename} | Color: {selected_color} | Render: {render_value}...")
            run(
                input_image_path=input_path,
                output_image_path=output_path,
                type="add_background",
                height=288,
                width=240,
                color=selected_color,
                hd=True,
                kb=None,
                render=render_value,
                dpi=300,
                face_align=False,
                matting_model=matting_model,
                face_detect_model="retinaface-resnet50",
            )
            print(f"✅ Saved: {output_path}")
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

# Main entry point
if __name__ == '__main__':
    matting_model = load_models()

    # Read all images
    filenames = [
        f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg"))
    ]

    # Prepare arguments for each image
    args_list = [(filename, matting_model, idx) for idx, filename in enumerate(filenames)]

    # Process all images in parallel using ProcessPoolExecutor
    max_workers = 2

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for future in executor.map(process_image, args_list):
            try:
                pass  # Each process_image handles its own errors
            except TimeoutError:
                print("Timeout occurred while processing an image.")
            except Exception as e:
                print(f"Error processing image: {e}")

    # List all output files
    print("\nListing output files:")
    os.system(f"ls -l {output_dir}")

#this scrip paint images background with whites allowed by sunedu
