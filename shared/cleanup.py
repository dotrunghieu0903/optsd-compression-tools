import torch
import gc

def setup_memory_optimizations():
    """Apply memory optimizations to avoid CUDA OOM errors"""
    # PyTorch memory optimizations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def setup_memory_optimizations_pruning():
    """
    Configure memory optimizations for PyTorch to avoid CUDA OOM errors.
    """
    # Clear any cached memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Print available GPU memory for debugging
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device)
        total_memory = gpu_properties.total_memory / (1024 ** 3)
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)
        print(f"GPU: {gpu_properties.name}")
        print(f"Total GPU memory: {total_memory:.2f} GiB")
        print(f"Allocated GPU memory: {allocated_memory:.2f} GiB")
        print(f"Reserved GPU memory: {reserved_memory:.2f} GiB")
        print(f"Free GPU memory: {total_memory - allocated_memory:.2f} GiB")
    
    # Set memory fraction to avoid OOM
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.9)