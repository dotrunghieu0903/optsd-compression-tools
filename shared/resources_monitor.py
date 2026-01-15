import json
import threading
import time
from pynvml import *

print("Importing GPU monitoring...")
try:
    from nvidia_ml_py import nvidia_smi
    USE_NVIDIA_SMI = True
except ImportError:
    try:
        from pynvml import *
        USE_NVIDIA_SMI = False
    except ImportError:
        print("Warning: Neither nvidia_smi nor pynvml available for GPU monitoring")
        USE_NVIDIA_SMI = None

# A list to store VRAM usage samples
vram_samples = []
stop_monitoring = threading.Event()
generation_metadata = [] # List to store metadata for each image

def monitor_vram(device_index=0):
    """
    Monitors VRAM usage of a specified GPU at a regular interval.
    """
    if USE_NVIDIA_SMI is None:
        # If no GPU monitoring available, just return
        return
    
    if USE_NVIDIA_SMI:
        # Use nvidia_smi
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_index)
        
        while not stop_monitoring.is_set():
            try:
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                # Convert bytes to GB
                used_vram_gb = info.used / 1024**3
                vram_samples.append(used_vram_gb)
                time.sleep(0.1)  # Sample every 100 milliseconds
            except Exception as error:
                print(f"Error during VRAM monitoring: {error}")
                break
        nvidia_smi.nvmlShutdown()
    else:
        # Use pynvml
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(device_index)
            
            while not stop_monitoring.is_set():
                try:
                    info = nvmlDeviceGetMemoryInfo(handle)
                    # Convert bytes to GB
                    used_vram_gb = info.used / 1024**3
                    vram_samples.append(used_vram_gb)
                    time.sleep(0.1)  # Sample every 100 milliseconds
                except Exception as error:
                    print(f"Error during VRAM monitoring: {error}")
                    break
            nvmlShutdown()
        except Exception as e:
            print(f"Error initializing VRAM monitoring: {e}")
            return

def generate_image_and_monitor(pipeline, prompt, output_path, filename, num_inference_steps=50, guidance_scale=3.5):
    """
    Generates an image using the provided pipeline while monitoring VRAM.
    Returns generation time and metadata.
    """
    # Reset VRAM samples and stop event for this run
    global vram_samples
    vram_samples = []
    stop_monitoring.clear()

    # Start the monitoring thread
    monitor_thread = threading.Thread(target=monitor_vram)
    monitor_thread.start()

    # Run the image generation function. Initialize with a value indicating failure
    generation_time = -1
    metadata = {}
    try:
        start_time = time.time()
        # Assuming the pipeline expects prompt and returns an image object (e.g., PIL Image)
        generated_image = pipeline(
            prompt, 
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0] # Assuming the pipeline returns a list of images
        end_time = time.time()
        generation_time = end_time - start_time

        # Save the generated image
        generated_image.save(output_path)
        print(f"\n✅ Image saved to: {output_path}")
        
        # Create metadata
        metadata = {
            "generated_image_path": output_path,
            "original_flickr_filename": filename,
            "caption_used": prompt,
            "generation_time": generation_time,
            "guidance_scale": guidance_scale,
            "num_steps": num_inference_steps
        }
        generation_metadata.append(metadata) # Append metadata for this image
        print(f"Image generation completed in {generation_time:.2f} seconds.")

    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        generation_time = -1 # Indicate failure
    finally:
        # Stop the monitoring thread
        stop_monitoring.set()
        monitor_thread.join()

    # Analyze the collected VRAM data (optional, can be done later with collected metadata)
    if vram_samples:
        avg_vram = sum(vram_samples) / len(vram_samples)
        peak_vram = max(vram_samples)

        print(f"\n--- VRAM Usage Statistics ---")
        print(f"Average VRAM used: {avg_vram:.2f} GB")
        print(f"Peak VRAM used: {peak_vram:.2f} GB")
        print(f"Number of samples: {len(vram_samples)}")
        metadata["average_vram_gb"] = avg_vram
        metadata["peak_vram_gb"] = peak_vram

    else:
        print("No VRAM data was collected.")
        metadata["average_vram_gb"] = None
        metadata["peak_vram_gb"] = None

    return generation_time, metadata # Return both generation time and metadata

def write_generation_metadata_to_file(file_path, metadata = None):
    """
    Writes the collected generation metadata to a JSON file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(generation_metadata if metadata is None else metadata, f, ensure_ascii=False, indent=4)
        print(f"Stored info to metadata: {file_path}")
    except Exception as e:
        print(f"Error when saving file metadata JSON: {e}")