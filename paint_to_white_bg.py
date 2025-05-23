from PIL import Image
import os

input_folder = "/Users/albertoperedarojas/ai/data/visiva_id_photo_validator/valid_data/transparent_bg"
output_folder = "/Users/albertoperedarojas/ai/data/visiva_id_photo_validator/valid_data/passport_style_white_bg"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".jpg")

        with Image.open(input_path) as im:
            if im.mode in ("RGBA", "LA"):
                # Convert image with transparency to RGB with white background
                background = Image.new("RGB", im.size, (255, 255, 255))
                background.paste(im, mask=im.split()[-1])  # Paste using alpha channel as mask
                background.save(output_path, "JPEG")
            else:
                # Already RGB, just save as JPG
                rgb_im = im.convert("RGB")
                rgb_im.save(output_path, "JPEG")

        print(f"✅ Processed: {filename}")