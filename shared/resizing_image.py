import os
from tqdm import tqdm
from PIL import Image

def resize_images(generated_image_dir, resized_generated_image_dir, image_dimensions):
    # Use tqdm to create a progress bar for the image resizing process
    for filename in tqdm(os.listdir(generated_image_dir), desc="Resizing images"):
        generated_path = os.path.join(generated_image_dir, filename)
        resized_path = os.path.join(resized_generated_image_dir, filename)

        # Ensure it's a file and not a directory
        if os.path.isfile(generated_path):
            # Get original dimensions from dictionary
            if filename in image_dimensions:
                original_width, original_height = image_dimensions[filename]

                try:
                    # Open generated image
                    img = Image.open(generated_path)

                    # Resize image
                    # Use Image.LANCZOS for higher quality when resizing
                    resized_img = img.resize((original_width, original_height), Image.LANCZOS)

                    # Save the resized image to the new directory
                    resized_img.save(resized_path)
                    print(f"Resized and saved {filename} to {resized_generated_image_dir}")

                except Exception as e:
                    print(f"Error when resizing or saving {filename}: {e}")
            else:
                print(f"No found size for {filename} in dictionary image_dimensions")