import os
import random
from PIL import Image

# Fixed color list
fixed_colors = [

]

# Folder paths
input_folder = "/Users/albertoperedarojas/test_images/lol"
output_folder = "/Users/albertoperedarojas/test_images/wii"

def generate_random_color(min_val=220, max_val=255):
    while True:
        r = random.randint(min_val, max_val)
        g = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        if abs(r - g) <= 4 and abs(g - b) <= 4 and abs(r - b) <= 4:
            return f"#{r:02X}{g:02X}{b:02X}"

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def add_dynamic_background(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]

    for index, filename in enumerate(sorted(image_files)):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpg")

        # Choose background color
        if index < len(fixed_colors):
            selected_color = fixed_colors[index]
        else:
            selected_color = generate_random_color()

        rgb_color = hex_to_rgb(selected_color)

        try:
            with Image.open(input_path) as im:
                if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                    im = im.convert("RGBA")
                    bg = Image.new("RGBA", im.size, rgb_color + (255,))
                    combined = Image.alpha_composite(bg, im)
                    combined.convert("RGB").save(output_path, "JPEG")
                else:
                    im.convert("RGB").save(output_path, "JPEG")
            print(f"Processed {filename} with bg color {selected_color}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    add_dynamic_background(input_folder, output_folder)