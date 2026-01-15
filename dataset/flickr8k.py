import os
from PIL import Image

def process_flickr8k(flickr_images_dir, flickr_captions_path):
    """
    Process Flickr8k dataset and return captions dictionary.
    
    Returns:
        dict: Dictionary mapping image filenames to their captions.
    """
    # Dictionary to store image dimensions
    image_dimensions = {}

    # Load captions from captions.txt
    # The format is likely "image_filename.jpg, caption text"
    captions = {}
    if os.path.exists(flickr_captions_path):
        with open(flickr_captions_path, 'r') as f:
            # Skip header if it exists
            lines = f.readlines()
            if lines and ',' in lines[0]: # Simple check for header
                lines = lines[1:]
            for line in lines:
                line = line.strip()
                if line and ',' in line:
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        filename = parts[0].strip()
                        caption = parts[1].strip()
                        if filename not in captions: # Store only the first caption for each image
                            captions[filename] = caption
                            
                            # Try to get image dimensions if the image file exists
                            img_path = os.path.join(flickr_images_dir, filename)
                            if os.path.exists(img_path):
                                try:
                                    with Image.open(img_path) as img:
                                        image_dimensions[filename] = img.size
                                except Exception as e:
                                    print(f"Error reading image dimensions for {filename}: {e}")
                    else:
                        print(f"Skipping malformed line in captions.txt: {line}")
        print(f"Loaded {len(captions)} captions from {flickr_captions_path}")
    else:
        print(f"Error: Captions file not found at {flickr_captions_path}")
        captions = {} # Ensure captions dictionary is empty if file not found
    
    return captions, image_dimensions