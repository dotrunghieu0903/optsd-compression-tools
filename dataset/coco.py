import json
import itertools
import os
import random

def enhance_caption(caption):
    """
    Enhance a caption to make it more suitable for image generation.
    
    Args:
        caption: The original caption
        
    Returns:
        Enhanced caption
    """
    # Remove any dots at the end
    caption = caption.rstrip('.')
    
    # Add prefixes that help with generation quality
    enhanced_prefixes = [
        "a high quality photo of",
        "a detailed image showing",
        "a professional photograph of",
        "a clear picture depicting"
    ]
    
    # Select a random prefix
    prefix = random.choice(enhanced_prefixes)
    
    # Combine prefix with caption and ensure first letter of caption is lowercase
    if caption and caption[0].isupper():
        caption = caption[0].lower() + caption[1:]
    
    enhanced_caption = f"{prefix} {caption}"
    
    # Add suffix to improve quality
    enhanced_suffixes = [
        ", high detail, professional lighting",
        ", sharp focus, high resolution",
        ", vibrant colors, professional photograph",
        ", detailed, realistic, high quality"
    ]
    
    suffix = random.choice(enhanced_suffixes)
    enhanced_caption = enhanced_caption + suffix
    
    return enhanced_caption

def filter_good_captions(caption):
    """
    Filter out captions that might lead to poor generation.
    
    Args:
        caption: The caption to check
        
    Returns:
        True if the caption is good, False otherwise
    """
    # Skip very short captions
    if len(caption.split()) < 4:
        return False
    
    # Skip captions with unwanted patterns
    unwanted_patterns = [
        "this is", "there is", "these are", "this picture",
        "photo of", "picture of", "image of", "snapshot",
        "it is", "it's a", "its a"
    ]
    
    lower_caption = caption.lower()
    for pattern in unwanted_patterns:
        if pattern in lower_caption:
            return False
    
    return True

def load_coco_captions(annotations_file, limit=500):
    """
    Load captions from COCO dataset with filtering and enhancement.
    
    Args:
        annotations_file: Path to COCO annotations file
        limit: Maximum number of captions to load
        
    Returns:
        Dictionary mapping image filenames to captions
    """
    print(f"Loading COCO captions from {annotations_file}")
    
    with open(annotations_file, 'r') as f:
        captions_data = json.load(f)
    
    # Build mapping from image_id to dimensions and filename
    image_id_to_dimensions = {
        img['id']: (img['width'], img['height'], img['file_name'])
        for img in captions_data['images']
    }
    
    print(f"Loaded {len(captions_data['annotations'])} annotations from COCO")
    
    # Group captions by image_id
    captions_by_image = {}
    for annotation in captions_data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        if image_id not in captions_by_image:
            captions_by_image[image_id] = []
        captions_by_image[image_id].append(caption)
    
    print(f"Found captions for {len(captions_by_image)} unique images")
    
    # Select the best caption for each image
    image_filename_to_caption = {}
    image_dimensions = {}  # Store {filename: (width, height)}
    processed_count = 0
    
    for image_id, captions in captions_by_image.items():
        if image_id not in image_id_to_dimensions:
            continue
            
        # Find the best caption (longest caption after filtering)
        good_captions = [c for c in captions if filter_good_captions(c)]
        if not good_captions:
            # If no good captions, use the longest one
            good_captions = captions
            
        # Sort by length and choose the longest
        best_caption = sorted(good_captions, key=len, reverse=True)[0]
        
        # Enhance the caption
        enhanced_caption = enhance_caption(best_caption)
        
        # Store the enhanced caption and dimensions
        width, height, original_filename = image_id_to_dimensions[image_id]
        image_filename_to_caption[original_filename] = enhanced_caption
        image_dimensions[original_filename] = (width, height)
        
        processed_count += 1
        if processed_count >= limit:
            break
    
    print(f"Selected and enhanced {len(image_filename_to_caption)} captions (limit: {limit})")
    
    # Print some examples of enhanced captions
    print("\nExample enhanced captions:")

    for filename, caption in itertools.islice(image_filename_to_caption.items(), 3):
        print(f"- {filename}: {caption}")
    
    return image_filename_to_caption, image_dimensions

def process_coco(annotations_dir):
    """
    Load COCO captions and image dimensions from annotations
    
    Args:
        annotations_dir: Directory containing COCO annotations
        
    Returns:
        image_filename_to_caption: Dictionary mapping image filenames to captions
        image_dimensions: Dictionary mapping image filenames to dimensions (width, height)
        image_id_to_dimensions: Dictionary mapping image IDs to (width, height, filename)
    """
    # Path to the captions annotation file
    captions_file = os.path.join(annotations_dir, 'captions_val2017.json')

    # Load the captions data
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)

    # Build dictionary mapping image_id to size
    image_id_to_dimensions = {img['id']: (img['width'], img['height'], img['file_name'])
                            for img in captions_data['images']}

    print(f"Read {len(captions_data['annotations'])} captions from COCO annotation file...")
    # To ensure each original image is processed only once for the main prompt purpose
    processed_image_ids = set()

    image_filename_to_caption = {}
    # Store {filename: (width, height)}
    image_dimensions = {}
    # Create a dictionary to store captions by image ID
    for annotation in captions_data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']

        if image_id in image_id_to_dimensions and image_id not in processed_image_ids:
            width, height, original_filename = image_id_to_dimensions[image_id]
            image_filename_to_caption[original_filename] = caption
            image_dimensions[original_filename] = (width, height)
            processed_image_ids.add(image_id)

    print(f"Extracted captions for {len(processed_image_ids)} images.")
    print(f"Created mapping for {len(image_filename_to_caption)} images.")
    print(f"Extracted dimensions for {len(image_dimensions)} images from annotations.")
    return image_filename_to_caption, image_dimensions, image_id_to_dimensions