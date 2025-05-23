import os
from inference import run
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing
import random

from addbackground_mutiprocess import load_models

# Set paths
input_dir = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/input_images"
output_dir = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/output_images"
retinaface_model_path = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/retinaface/RetinaFace-R50.pth"
modnet_model_path = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/modnet_photographic_portrait_matting/modnet_photographic_portrait_matting.ckpt"
onnx_model_path = "/Users/albertoperedarojas/ai/mytools/HivisionIDPhotos/hivision/creator/weights/birefnet-v1-lite.onnx"
colors_log_path = os.path.join(output_dir, "colors_used.txt")  # Log file to store colors used

# Check if model files exist
if not os.path.isfile(retinaface_model_path):
    raise FileNotFoundError(f"RetinaFace model not found at: {retinaface_model_path}")

# Check for matting model
if os.path.isfile(onnx_model_path):
    matting_model = "birefnet-v1-lite"
elif os.path.isfile(modnet_model_path):
    matting_model = "birefnet-v1-lite"
    print(f"Warning: Using .ckpt model at {modnet_model_path}. Ensure compatibility.")
else:
    print("Warning: MODNet model not found. Falling back to hivision_modnet.")
    matting_model = "hivision_modnet"
    onnx_model_path = "hivision/creator/weights/hivision_modnet.onnx"
    if not os.path.isfile(onnx_model_path):
        raise FileNotFoundError(f"Fallback model not found at: {onnx_model_path}")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to generate a random color within the specified range (more colorful and pastel-like)
def generate_random_color(min_val=215, max_val=255, min_diff=15, max_val_diff=240):
    while True:
        red = random.randint(min_val, max_val)
        green = random.randint(min_val, max_val)
        blue = random.randint(min_val, max_val)

        # Ensure there is a minimum difference between color components
        if abs(red - green) <= min_diff or abs(green - blue) <= min_diff or abs(red - blue) <= min_diff:
            continue  # Skip if the colors are too similar

        # Prevent colors that are too close to pure white by ensuring a max threshold
        if red > max_val_diff and green > max_val_diff and blue > max_val_diff:
            continue  # Skip if all components are too close to 255

        # NEW CHECK: prevent colors that are too bright
        avg_brightness = (red + green + blue) / 3
        if avg_brightness > 240:
            continue  # Skip overly bright colors

        return f"#{red:02X}{green:02X}{blue:02X}"

# Function to process a single image
def process_image(args):
    filename, matting_model, index = args

    if filename.lower().endswith((".jpg", ".jpeg")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpg")

        # Select color from the predefined list or generate a random color
        selected_color = generate_random_color()

        # Alternate render value 1 and 2
        render_value = (index % 2) + 1

        try:
            print(f"Processing {filename} | Color: {selected_color} | Render: {render_value}...")
            run(
                input_image_path=input_path,
                output_image_path=output_path,
                type="add_background",
                height=413,
                width=295,
                color=selected_color,
                hd=True,
                kb=None,
                render=0,
                dpi=300,
                face_align=False,
                matting_model=matting_model,
                face_detect_model="retinaface-resnet50",
            )
            print(f"✅ Saved: {output_path}")

            # Log the color and image filename to the text file
            with open(colors_log_path, "a") as log_file:
                log_file.write(f"{filename} : {selected_color}\n")
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
    max_workers = 2  # Limit to 2 workers to reduce CPU and memory load

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