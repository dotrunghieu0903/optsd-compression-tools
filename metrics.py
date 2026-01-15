from cleanfid import fid
import os
import ImageReward as RM
import torch
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
import lpips
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel

def calculate_fid(generated_image_dir, resized_generated_image_dir, validation_dataset_dir):
    """
    Calculate the FID score between the generated images and the COCO validation set.
    """
    if not os.path.exists(generated_image_dir):
        print(f"Error: Generated image directory does not exist: {generated_image_dir}")
        return None

    if not os.path.exists(resized_generated_image_dir):
        print(f"Error: Resized generated image directory does not exist: {resized_generated_image_dir}")
        return None

    if not os.path.exists(validation_dataset_dir):
        print(f"Error: COCO validation directory does not exist: {validation_dataset_dir}")
        return None
    print(f"Calculating FID between COCO dataset: {validation_dataset_dir}, Generated Image: {generated_image_dir}")

    # Make sure the directories exist and contain images before calculating FID
    try:
        score_fid = fid.compute_fid(
            validation_dataset_dir,
            resized_generated_image_dir,
            mode="clean",  # Use the 'clean' mode for the official cleanfid scores
            num_workers=8,  # Adjust based on your CPU cores
            batch_size=256  # Adjust based on your GPU memory
        )

        print(f"FID Score: {score_fid:.2f}")
        return score_fid
    except Exception as e:
        print(f"Error when calculating FID: {e}")
        return None

# Calculate Image Reward
def compute_image_reward(generated_dirpath, captions_dict):
    scores = []
    try:
        model = RM.load("ImageReward-v1.0")
        print("ImageReward model loaded.")
        # Filter captions_dict to only include filenames for which images were successfully generated
        generated_files = [f for f in os.listdir(generated_dirpath) if os.path.isfile(os.path.join(generated_dirpath, f))]
        filtered_captions = {filename: prompt for filename, prompt in captions_dict.items() if filename in generated_files}

        if not filtered_captions:
            print("No generated images found to compute Image Reward.")
            return None

        print(f"Calculating Image Reward for {len(filtered_captions)} images...")

        for filename, prompt in filtered_captions.items():
            path = os.path.join(generated_dirpath, filename)
            try:
                with torch.inference_mode():
                    score = model.score(prompt, path)
                scores.append(score)
                #print(f"Image Reward for {filename} (Prompt: {prompt[:50]}...): {score:.4f}")
            except Exception as e:
                print(f"Error when computing Image Reward for {filename}: {e}")
                scores.append(None) # Append None to indicate failure

        # Calculate average score, excluding None values from errors
        valid_scores = [s for s in scores if s is not None]
        if valid_scores:
            average_score = sum(valid_scores) / len(valid_scores)
            print(f"Average Image Reward Score: {average_score:.4f}")
            return average_score
        else:
            print("No valid Image Reward scores could be computed.")
            return None

    except Exception as e:
        print(f"Error when loading ImageReward model or during computation: {e}")
        return None
    
# Calculate PSNR using the resized generated images
def calculate_psnr_resized(original_dir, generated_dir, filenames):
    """Calculates the average PSNR between original and resized generated images."""
    psnr_scores = []
    print(f"Calculating PSNR for {len(filenames)} resized images...")

    for filename in filenames:
        original_path = os.path.join(original_dir, filename)
        generated_path = os.path.join(generated_dir, filename)

        if not os.path.exists(original_path):
            print(f"Original image not found: {original_path}")
            continue
        if not os.path.exists(generated_path):
             print(f"Resized generated image not found: {generated_path}")
             continue

        try:
            # Open images
            original_img = Image.open(original_path).convert("RGB")
            generated_img = Image.open(generated_path).convert("RGB")

            # Convert images to numpy arrays
            original_np = np.array(original_img)
            generated_np = np.array(generated_img)

            # Ensure images have the same dimensions (should be the case after resizing)
            if original_np.shape != generated_np.shape:
                print(f"Skipping PSNR for {filename}: dimensions mismatch (should not happen after resizing)")
                continue

            # Calculate PSNR
            psnr = peak_signal_noise_ratio(original_np, generated_np)
            psnr_scores.append(psnr)
            #print(f"PSNR for {filename}: {psnr:.2f}")

        except Exception as e:
            print(f"Error calculating PSNR for {filename}: {e}")
            # psnr_scores.append(None) # Optionally append None for errors

    if psnr_scores:
        average_psnr = np.mean(psnr_scores)
        print(f"Average PSNR (Resized Images): {average_psnr:.2f}")
        return average_psnr
    else:
        print("No PSNR scores could be calculated for resized images.")
        return None
    
# Function to calculate LPIPS
def calculate_lpips(original_dir, generated_dir, filenames):
    """Calculates the average LPIPS score between original and generated images."""
    lpips_scores = []
    print(f"Calculating LPIPS for {len(filenames)} images...")

    try:
        loss_fn_alex = lpips.LPIPS(net='alex') # Using AlexNet
        if torch.cuda.is_available():
            loss_fn_alex.cuda()

        # Define image transformations
        transform = transforms.ToTensor()

        for filename in tqdm(filenames, desc="Calculating LPIPS"):
            original_path = os.path.join(original_dir, filename)
            generated_path = os.path.join(generated_dir, filename)

            if not os.path.exists(original_path):
                print(f"Original image not found: {original_path}")
                continue
            if not os.path.exists(generated_path):
                 print(f"Generated image not found: {generated_path}")
                 continue

            try:
                # Open and transform images
                original_img = Image.open(original_path).convert("RGB")
                generated_img = Image.open(generated_path).convert("RGB")

                # Ensure images have the same dimensions for LPIPS calculation
                if original_img.size != generated_img.size:
                    print(f"Skipping LPIPS for {filename}: dimensions mismatch")
                    continue

                img1 = transform(original_img).unsqueeze(0)
                img2 = transform(generated_img).unsqueeze(0)

                if torch.cuda.is_available():
                    img1 = img1.cuda()
                    img2 = img2.cuda()

                # Calculate LPIPS
                with torch.no_grad():
                    score = loss_fn_alex(img1, img2)
                lpips_scores.append(score.item())

            except Exception as e:
                print(f"Error when calculating LPIPS for {filename}: {e}")
                lpips_scores.append(None)

        valid_scores = [s for s in lpips_scores if s is not None]
        if valid_scores:
            average_score = sum(valid_scores) / len(valid_scores)
            print(f"\nAverage LPIPS Score: {average_score:.4f}")
            return average_score
        else:
            print("No valid LPIPS scores could be calculated.")
            return None

    except Exception as e:
        print(f"Error loading LPIPS model or during computation: {e}")
        return None
    
def calculate_clip_score(generated_image_dir, captions):
    # --- Starting CLIP Score Calculation ---
    print("\n--- Starting CLIP Score Calculation ---")

    # Path to the directory containing generated images
    image_files = [f for f in os.listdir(generated_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    if not image_files:
        print(f"Error: Generated image directory not found or empty: {generated_image_dir}")
        print("Cannot calculate CLIP Score without generated images and corresponding prompts.")
    else:
        images_to_process = []
        texts_to_process = []

        # Use the actual captions from the dictionary for the generated images
        for filename in image_files:
            img_path = os.path.join(generated_image_dir, filename)
            # Get the corresponding caption using the filename as the key
            if filename in captions: # Corrected variable name from image_filename_to_caption to captions
                prompt = captions[filename]
                try:
                    img = Image.open(img_path).convert("RGB")
                    images_to_process.append(img)
                    texts_to_process.append(prompt)
                except Exception as e:
                    print(f"Could not load image {img_path}: {e}")
                    continue
            else:
                print(f"Warning: No caption found for image {filename}. Skipping.")

        if not images_to_process:
            print("No valid images found to calculate CLIP Score.")
        else:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Using device: {device}")

                # Load pre-trained CLIP model and processor
                model_name = "openai/clip-vit-base-patch32"
                processor = CLIPProcessor.from_pretrained(model_name)
                model = CLIPModel.from_pretrained(model_name).to(device)

                # Verify we loaded a CLIP model, not a video model
                is_sam3 = 'Sam3VideoModel' in str(type(model)) or 'Sam3' in str(type(model))
                if is_sam3:
                    print(f"Warning: Expected CLIP model but got {type(model).__name__}. Cannot calculate CLIP Score with video model.")
                    return None

                clip_scores = []
                batch_size = 32 # Can adjust batch_size according to GPU RAM

                for i in tqdm(range(0, len(images_to_process), batch_size), desc="Calculating CLIP Scores"):
                    batch_images = images_to_process[i:i + batch_size]
                    batch_texts = texts_to_process[i:i + batch_size]

                    if not batch_images: # Handle empty batch case
                        continue

                    # Preprocess images and tokenize texts
                    inputs = processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True)

                    # Move inputs to device
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model(**inputs)

                    # A simple CLIP Score definition for text-to-image is often
                    # the cosine similarity between image and text features, scaled by 100.

                    image_features = outputs.image_embeds
                    text_features = outputs.text_embeds

                    # Normalize features
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

                    # Calculate cosine similarity for each pair
                    # Shape: (batch_size, 1) if multiplying features directly for each pair
                    # Or, using torch.nn.functional.cosine_similarity

                    # Using the original scaling from CLIP
                    logit_scale = model.logit_scale.exp()
                    batch_clip_scores = logit_scale * torch.sum(image_features * text_features, dim=1)

                    # Take max(0, score) as per some CLIP Score definitions
                    batch_clip_scores = torch.max(torch.zeros_like(batch_clip_scores), batch_clip_scores)

                    # Detach the tensor before converting to numpy
                    clip_scores.extend(batch_clip_scores.detach().cpu().numpy())

                if clip_scores:
                    avg_clip_score = np.mean(clip_scores)
                    print(f"\nAverage CLIP Score: {avg_clip_score:.2f}")
                    return avg_clip_score
                else:
                    print("No CLIP scores calculated.")
                    return None

            except Exception as e:
                print(f"Error when calculating CLIP Score: {e}")
                return None