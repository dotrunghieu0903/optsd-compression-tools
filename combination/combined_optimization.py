#!/usr/bin/env python3
"""
Combined optimization for diffusion models with COCO dataset.
This script applies three optimization techniques to a diffusion model:
1. Quantization - Reduces precision to decrease model size
2. Pruning - Removes unnecessary weights to further reduce model size
3. KV Caching - Caches key-value pairs in attention to speed up inference

It evaluates performance on captions from the COCO dataset and calculates 
image quality metrics: FID, CLIP Score, ImageReward, LPIPS, and PSNR.
"""

import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gc
import json
import time
import argparse
import random
import itertools
import threading
import numpy as np
from shared.cleanup import setup_memory_optimizations_pruning
import torch
from tqdm import tqdm
from PIL import Image
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
    HAVE_PYNVML = True
except ImportError:
    print("Warning: pynvml not found. Falling back to torch.cuda for VRAM monitoring.")
    HAVE_PYNVML = False

import torch
from diffusers import DiffusionPipeline, FluxPipeline, SanaPipeline
from huggingface_hub import login

# Add parent directory to path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.resources_monitor import generate_image_and_monitor, monitor_vram, write_generation_metadata_to_file
from metrics import calculate_fid, compute_image_reward, calculate_clip_score, calculate_lpips, calculate_psnr_resized
from shared.resizing_image import resize_images
from dataset.flickr8k import process_flickr8k

# Try to import model-specific modules
try:
    # First try to import quantization modules
    from nunchaku import NunchakuFluxTransformer2dModel, NunchakuSanaTransformer2DModel
    from nunchaku.utils import get_precision
    HAS_NUNCHAKU = True
except ImportError:
    print("Warning: Nunchaku not found. Quantization features will be limited.")
    HAS_NUNCHAKU = False

# Try to import pruning module
try:
    from pruning import get_model_size, get_sparsity, apply_magnitude_pruning
except ImportError:
    # Create fallback functions for pruning
    def get_model_size(model):
        """Calculate the size of the model in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb

    def get_sparsity(model):
        """Calculate the sparsity of the model."""
        total_params = 0
        zero_params = 0
        for param in model.parameters():
            if param.requires_grad:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
        
        if total_params == 0:
            return 0.0
            
        return float(zero_params) / float(total_params)

    def apply_magnitude_pruning(model, amount=0.3):
        """Apply magnitude pruning to the model."""
        print(f"Applying magnitude pruning with amount {amount}...")
        
        # Copy the model to avoid modifying the original
        pruned_model = type(model)(**model.__dict__)
        
        for name, param in pruned_model.named_parameters():
            if param.dim() > 1:  # Only prune weights, not biases
                tensor = param.data.clone()
                threshold = torch.quantile(tensor.abs().flatten(), amount)
                mask = tensor.abs() > threshold
                tensor = tensor * mask
                param.data = tensor
                
        print(f"Pruning complete. New sparsity: {get_sparsity(pruned_model):.4f}")
        return pruned_model

# Configure PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Global variables for VRAM monitoring
vram_samples = []
stop_monitoring = threading.Event()

# ======================================================================
# SAM3 VIDEO MODEL HELPER FUNCTIONS
# ======================================================================

def ensure_inference_session(func, model=None, *args, **kwargs):
    """Wrapper to ensure inference_session parameter for Sam3VideoModel calls
    
    Args:
        func: The function/method to call
        model: Optional model instance to get inference_session from (for bound methods)
        *args, **kwargs: Arguments to pass to the function
    """
    try:
        return func(*args, **kwargs)
    except TypeError as e:
        if 'inference_session' in str(e):
            # Get the actual model from bound method's __self__ or from passed model
            actual_model = None
            if hasattr(func, '__self__'):
                actual_model = func.__self__
            elif model is not None:
                actual_model = model
            
            # Check if it's a Sam3VideoModel
            is_sam3 = False
            if actual_model is not None:
                is_sam3 = 'Sam3VideoModel' in str(type(actual_model))
            
            if is_sam3 or 'Sam3VideoModel' in str(e):
                print(f"Auto-fixing missing inference_session parameter")
                # Try to get inference_session from various sources
                inference_session = None
                
                # Method 1: Check if model has create_inference_session
                if actual_model is not None and hasattr(actual_model, 'create_inference_session'):
                    try:
                        inference_session = actual_model.create_inference_session()
                        print("Created inference_session using model.create_inference_session()")
                    except Exception as ce:
                        print(f"Failed to create inference_session: {ce}")
                
                # Method 2: Check if model has inference_session attribute
                if inference_session is None and actual_model is not None and hasattr(actual_model, 'inference_session'):
                    inference_session = actual_model.inference_session
                    print("Using model.inference_session attribute")
                
                # Method 3: Check func if it has these attributes (unlikely for bound method)
                if inference_session is None and hasattr(func, 'create_inference_session'):
                    try:
                        inference_session = func.create_inference_session()
                    except:
                        pass
                
                if inference_session is None and hasattr(func, 'inference_session'):
                    inference_session = func.inference_session
                
                # Method 4: Create dummy inference session
                if inference_session is None:
                    device = 'cuda'
                    dtype = 'float32'
                    if actual_model is not None:
                        device = getattr(actual_model, 'device', 'cuda')
                        dtype = str(getattr(actual_model, 'dtype', 'float32'))
                    
                    inference_session = type('InferenceSession', (), {
                        'device': device,
                        'dtype': dtype,
                        'batch_size': 1,
                        'max_sequence_length': 1024
                    })()
                    print("Created dummy inference_session")
                
                # Add to kwargs and retry
                kwargs['inference_session'] = inference_session
                return func(*args, **kwargs)
            else:
                raise e
        else:
            raise e

def safe_forward_call(model, *args, **kwargs):
    """Safe wrapper for model forward calls that handles Sam3VideoModel"""
    # Check if this is Sam3VideoModel
    is_sam3 = (hasattr(model, '__class__') and 'Sam3VideoModel' in str(model.__class__)) or \
              (hasattr(model, '__name__') and 'Sam3VideoModel' in model.__name__) or \
              ('Sam3VideoModel' in str(type(model)))
    
    if is_sam3 and 'inference_session' not in kwargs:
        print(f"Detected Sam3VideoModel, ensuring inference_session parameter")
        
        # Try multiple methods to get inference_session
        inference_session = None
        
        # Method 1: From model's create_inference_session
        if hasattr(model, 'create_inference_session'):
            try:
                inference_session = model.create_inference_session()
                print("Created inference_session using model.create_inference_session()")
            except Exception as e:
                print(f"Failed to create inference_session: {e}")
        
        # Method 2: From model's existing inference_session
        if inference_session is None and hasattr(model, 'inference_session'):
            inference_session = model.inference_session
            print("Using existing model.inference_session")
        
        # Method 3: Create basic inference session
        if inference_session is None:
            inference_session = type('InferenceSession', (), {
                'device': getattr(model, 'device', 'cuda'),
                'dtype': str(getattr(model, 'dtype', 'float32')),
                'batch_size': 1,
                'max_sequence_length': getattr(model, 'max_sequence_length', 1024)
            })()
            print("Created basic inference_session")
        
        kwargs['inference_session'] = inference_session
    
    # Call the model's forward method, passing model reference for fallback
    if hasattr(model, 'forward'):
        return ensure_inference_session(model.forward, model, *args, **kwargs)
    else:
        return ensure_inference_session(model, model, *args, **kwargs)

# ======================================================================
# KV CACHE IMPLEMENTATION
# ======================================================================

class KVCacheTransformer(torch.nn.Module):
    @staticmethod
    def cache_context(model):
        """
        Context manager to temporarily enable KV caching within a with-block.
        Usage:
            with KVCacheTransformer.cache_context(model):
                ...
        """
        # Add stack trace to help debug where this is called from
        import traceback
        stack = traceback.extract_stack()
        caller = stack[-2]
        print(f"cache_context called from: {caller.filename}:{caller.lineno}")
        
        # Check if the model is a valid object with use_kv_cache attribute
        if model is None:
            print(f"Error: Model is None! Creating dummy context.")
            class DummyContext:
                def __enter__(self):
                    return None
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
            return DummyContext()
        elif isinstance(model, str):
            print(f"Error: Model is a string: '{model}'! Creating dummy context.")
            class DummyContext:
                def __enter__(self):
                    return model
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
            return DummyContext()
        elif not hasattr(model, 'use_kv_cache'):
            print(f"Warning: Model of type {type(model)} does not have use_kv_cache attribute. Creating dummy context.")
            # Create a dummy context that does nothing
            class DummyContext:
                def __enter__(self):
                    return model
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
            return DummyContext()
        
        # Static method doesn't have a self reference
        prev_value = model.use_kv_cache
        model.use_kv_cache = True
        
        # Create a context manager as a class
        class SimpleContext:
            def __init__(self):
                pass
                
            def __enter__(self):
                return model
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                model.use_kv_cache = prev_value
                
        return SimpleContext()
    """Transformer model with Key-Value caching for transformer blocks
    
    This class wraps a transformer model and adds KV caching to speed up
    the image generation process by caching key-value pairs in attention layers.
    """
    
    def __init__(self, transformer):
        super().__init__()
        # Store the original transformer
        self.transformer = transformer
        
        # Initialize KV cache
        self.kv_cache = {}
        
        # Flag to indicate whether to use KV cache
        self.use_kv_cache = False
        
        # Copy device and dtype from original transformer if available
        self.device = getattr(transformer, 'device', torch.device('cuda'))
        self.dtype = getattr(transformer, 'dtype', None)
        
        # Fallback to transformer parameters
        if self.dtype is None:
            # Try to infer dtype from transformer parameters
            for param in transformer.parameters():
                if param.dtype:
                    self.dtype = param.dtype
                    break
        
        # Copy config from original transformer or create a new one if not available
        if hasattr(transformer, 'config'):
            self.config = transformer.config
        else:
            # Create a temporary config to avoid errors
            from types import SimpleNamespace
            self.config = SimpleNamespace()
            
            # Add necessary attributes to config
            if hasattr(transformer, 'dim'):
                self.config.hidden_size = transformer.dim
            elif hasattr(transformer, 'hidden_size'):
                self.config.hidden_size = transformer.hidden_size
            else:
                self.config.hidden_size = 768  # Common default value
                
            # Add other attributes if needed
            self.config.model_type = "transformer"
            
            # Copy additional attributes from original transformer
            for attr in dir(transformer):
                if not attr.startswith('_') and not hasattr(self.config, attr):
                    try:
                        value = getattr(transformer, attr)
                        if not callable(value) and not isinstance(value, torch.nn.Module):
                            setattr(self.config, attr, value)
                    except:
                        pass
            
            print("Created fallback config for transformer")
        
        # Save information about transformer structure for debugging
        self.transformer_structure = self._inspect_structure(transformer)
        
    def _inspect_structure(self, module, prefix="", max_depth=3, current_depth=0):
        """Inspect the structure of a module to better understand the transformer"""
        if current_depth >= max_depth:
            return {"type": str(type(module).__name__), "truncated": True}
            
        result = {"type": str(type(module).__name__)}
        
        if hasattr(module, "named_children"):
            children = {}
            has_children = False
            
            for name, child in module.named_children():
                has_children = True
                children[name] = self._inspect_structure(
                    child, 
                    prefix=f"{prefix}.{name}", 
                    max_depth=max_depth, 
                    current_depth=current_depth+1
                )
                
            if has_children:
                result["children"] = children
                
        return result
        
    def _apply_kv_caching_to_attention_blocks(self):
        """Apply KV caching to attention blocks in the transformer"""
        # Find all attention layers
        attention_layers_count = 0
        
        for name, module in self.transformer.named_modules():
            # Find attention modules based on name or structure
            if self._is_attention_module(name, module):
                if hasattr(module, "forward"):
                    # Save original forward function
                    original_forward = module.forward
                    module.name = name
                    module.original_forward = original_forward
                    
                    # Replace with cached version
                    module.forward = self._make_kv_cached_attn_forward(module)
                    attention_layers_count += 1
                    print(f"Applied KV caching to {name}")
        
        if attention_layers_count == 0:
            print("Warning: No attention layers found for KV caching")
            # Try alternative methods to apply KV caching
            self._try_alternative_attention_detection()
        else:
            print(f"Successfully applied KV caching to {attention_layers_count} attention layers")
    
    def _is_attention_module(self, name, module):
        """Determine if a module is an attention layer"""
        # Method 1: Based on name
        if any(pattern in name.lower() for pattern in ["attn", "attention"]):
            return True
            
        # Method 2: Based on characteristic attributes of attention
        attn_attributes = ["q_proj", "k_proj", "v_proj", "out_proj"]
        if all(hasattr(module, attr) for attr in attn_attributes):
            return True
            
        # Method 3: Based on data structure
        if hasattr(module, "forward") and any(param.name in ["query", "key", "value"] for param in module.parameters()):
            return True
            
        return False
    
    def _try_alternative_attention_detection(self):
        """Try alternative methods to find and apply KV caching"""
        print("Trying alternative methods to detect attention layers...")
        
        # Method 1: Search by method structure
        found_layers = []
        for name, module in self.transformer.named_modules():
            if hasattr(module, "forward") and "attention" in str(module.forward).lower():
                found_layers.append((name, module))
                
        # Method 2: Look at layers with names suggesting attention
        for name, module in self.transformer.named_modules():
            if any(hint in name.lower() for hint in ["self", "cross", "mha", "mhsa"]):
                if (name, module) not in found_layers:
                    found_layers.append((name, module))
        
        # Apply KV caching to found layers
        for name, module in found_layers:
            original_forward = module.forward
            module.name = name
            module.original_forward = original_forward
            module.forward = self._make_kv_cached_attn_forward(module)
            print(f"Applied KV caching to {name} using alternative detection")
    
    def _make_kv_cached_attn_forward(self, module):
        """Create a forward function with KV caching for attention blocks
        
        Main logic:
        1. First time generating token: Calculate and store KV values
        2. Subsequent times: Reuse stored KV values, only calculate new Q
        3. Efficient attention calculation: Use cached K, V with new Q
        """
        def cached_forward(hidden_states, *args, **kwargs):
            # Get timestep information from kwargs if available
            timestep = kwargs.get('timestep', 0)
            cache_key = f"{module.name}_{timestep}"
            use_cache = kwargs.get('use_cache', False)
            
            # Check if this is Sam3VideoModel and add inference_session if needed
            if hasattr(module, '__class__') and 'Sam3VideoModel' in str(module.__class__):
                if 'inference_session' not in kwargs:
                    # Create a dummy inference session or get from model if available
                    if hasattr(module, 'create_inference_session'):
                        kwargs['inference_session'] = module.create_inference_session()
                    elif hasattr(module, 'inference_session'):
                        kwargs['inference_session'] = module.inference_session
                    else:
                        # Create a basic inference session object
                        kwargs['inference_session'] = type('InferenceSession', (), {})()
                        print(f"Warning: Created dummy inference_session for {module.__class__}")
            
            # Check if this is the first run or caching is disabled
            if not use_cache or cache_key not in self.kv_cache:
                # Call original forward function with proper error handling
                try:
                    output = safe_forward_call(module.original_forward, hidden_states, *args, **kwargs)
                except Exception as e:
                    print(f"Error in safe_forward_call, trying direct call: {e}")
                    output = module.original_forward(hidden_states, *args, **kwargs)
                
                # Store KV values in cache if module has necessary components
                if use_cache:
                    # Method 1: Modules with separate q_proj, k_proj, v_proj
                    if all(hasattr(module, attr) for attr in ["q_proj", "k_proj", "v_proj"]):
                        key = module.k_proj(hidden_states)
                        value = module.v_proj(hidden_states)
                        self.kv_cache[cache_key] = (key, value)
                    
                    # Method 2: Module with get_key_value function
                    elif hasattr(module, "get_key_value"):
                        key, value = module.get_key_value(hidden_states)
                        self.kv_cache[cache_key] = (key, value)
                        
                    # Method 3: Module with key, value attributes
                    elif hasattr(module, "key") and hasattr(module, "value"):
                        try:
                            key = module.key(hidden_states)
                            value = module.value(hidden_states)
                            self.kv_cache[cache_key] = (key, value)
                        except Exception as e:
                            print(f"Failed to cache key-value: {e}")
            else:
                # Get KV values from cache
                key, value = self.kv_cache[cache_key]
                
                # Calculate query for current token
                if hasattr(module, "q_proj"):
                    query = module.q_proj(hidden_states)
                    
                    # Efficient attention calculation
                    # Adjust according to SANA attention architecture
                    try:
                        # Prepare input data
                        batch_size, seq_len, _ = query.size()
                        head_dim = query.size(-1) // getattr(module, "num_heads", 8)
                        
                        # Reshape for attention calculation
                        query = query.view(batch_size, seq_len, -1, head_dim)
                        key = key.view(batch_size, -1, query.size(2), head_dim)
                        value = value.view(batch_size, -1, query.size(2), head_dim)
                        
                        # Calculate attention scores
                        attention_scores = torch.matmul(query, key.transpose(-1, -2))
                        attention_scores = attention_scores / (head_dim ** 0.5)
                        
                        # Apply softmax
                        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
                        
                        # Calculate context layer
                        context_layer = torch.matmul(attention_probs, value)
                        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
                        
                        # Reshape output
                        output = context_layer.view(batch_size, seq_len, -1)
                        
                        # Apply output projection if available
                        if hasattr(module, "out_proj"):
                            output = module.out_proj(output)
                    except Exception as e:
                        # If error occurs, use original forward
                        print(f"Error in custom attention: {e}. Using original forward.")
                        output = module.original_forward(hidden_states, *args, **kwargs)
                else:
                    # Use original forward if attention structure is not suitable
                    output = module.original_forward(hidden_states, *args, **kwargs)
            
            return output
        
        return cached_forward
    
    def clear_cache(self):
        """Clear the entire KV cache"""
        self.kv_cache = {}
        print("KV cache cleared")
    
    def enable_kv_caching(self):
        """Enable KV caching in the transformer"""
        self._apply_kv_caching_to_attention_blocks()
        print("KV caching enabled for Transformer model")
    
    def to(self, *args, **kwargs):
        """Custom to() method to handle device and dtype"""
        # Update internal device and dtype attributes
        if 'device' in kwargs:
            self.device = kwargs['device']
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']
            
        # Call parent to() method to move module parameters
        result = super().to(*args, **kwargs)
        
        # Also move the wrapped transformer if possible
        try:
            self.transformer.to(*args, **kwargs)
        except Exception as e:
            print(f"Warning: Could not move wrapped transformer: {e}")
            
        return result
    
    def forward(self, *args, **kwargs):
        """Forward method with KV caching support"""
        # Check if use_cache is provided and handle it
        if 'use_cache' in kwargs:
            self.use_kv_cache = kwargs.pop('use_cache')
        
        # Handle Sam3VideoModel inference_session requirement
        if hasattr(self.transformer, '__class__') and 'Sam3VideoModel' in str(self.transformer.__class__):
            if 'inference_session' not in kwargs:
                if hasattr(self.transformer, 'create_inference_session'):
                    kwargs['inference_session'] = self.transformer.create_inference_session()
                elif hasattr(self.transformer, 'inference_session'):
                    kwargs['inference_session'] = self.transformer.inference_session
                else:
                    # Create a basic inference session
                    kwargs['inference_session'] = type('InferenceSession', (), {})()
                    print(f"Warning: Created dummy inference_session for {self.transformer.__class__}")
        
        # Save all input parameters for debugging
        if not hasattr(self, '_last_inputs'):
            self._last_inputs = {'args': args, 'kwargs': kwargs}
        
        # Use use_kv_cache attribute if available
        use_cache = getattr(self, "use_kv_cache", False)
        
        # Check if transformer supports use_cache
        supports_use_cache = False
        if use_cache:
            # Check signature of forward method
            import inspect
            if hasattr(self.transformer, 'forward') and inspect.isfunction(self.transformer.forward):
                sig = inspect.signature(self.transformer.forward)
                supports_use_cache = 'use_cache' in sig.parameters
            
            # Or check from other attributes if available
            model_config = getattr(self.transformer, 'config', None)
            if model_config and hasattr(model_config, 'use_cache'):
                supports_use_cache = True
        
        try:
            # Try with use_cache if appropriate and supported
            if use_cache and supports_use_cache:
                try:
                    # Clone kwargs to avoid affecting original parameters
                    new_kwargs = dict(kwargs)
                    new_kwargs["use_cache"] = True
                    return safe_forward_call(self.transformer, *args, **new_kwargs)
                except Exception as e:
                    print(f"Warning: KV cache parameter supported but failed: {e}")
                    # Continue to try without use_cache
            
            # Perform forward through replaced transformer
            return safe_forward_call(self.transformer, *args, **kwargs)
        except Exception as e:
            print(f"Error in safe_forward_call, trying fallback methods...")
            print(f"Error in KVCacheTransformer.forward: {e}")
            print(f"Args types: {[type(arg) for arg in args]}")
            print(f"Kwargs keys: {list(kwargs.keys())}")
            
            # Try with various adjustments
            try:
                # Try removing parameters that might cause errors, but keep inference_session
                kwargs_filtered = {k: v for k, v in kwargs.items() 
                                if k not in ['return_dict', 'output_attentions', 'use_cache']}
                # Ensure inference_session is present for Sam3VideoModel
                if 'Sam3VideoModel' in str(type(self.transformer)) and 'inference_session' not in kwargs_filtered:
                    if hasattr(self.transformer, 'inference_session'):
                        kwargs_filtered['inference_session'] = self.transformer.inference_session
                    elif hasattr(self.transformer, 'create_inference_session'):
                        kwargs_filtered['inference_session'] = self.transformer.create_inference_session()
                return safe_forward_call(self.transformer, *args, **kwargs_filtered)
            except Exception as e2:
                print(f"Error with filtered kwargs: {e2}")
                
                # Try with only args, no kwargs - but still include inference_session if needed
                try:
                    if 'Sam3VideoModel' in str(type(self.transformer)):
                        inf_session = None
                        if hasattr(self.transformer, 'inference_session'):
                            inf_session = self.transformer.inference_session
                        elif hasattr(self.transformer, 'create_inference_session'):
                            inf_session = self.transformer.create_inference_session()
                        if inf_session:
                            return safe_forward_call(self.transformer, *args, inference_session=inf_session)
                    return safe_forward_call(self.transformer, *args)
                except Exception as e3:
                    print(f"Error with only args: {e3}")
                    raise e  # Throw original exception

# ======================================================================
# OPTIMIZED PIPELINE
# ======================================================================

class OptimizedDiffusionPipeline:
    """Pipeline with combined optimizations: quantization, pruning, and KV caching
    
    This class initializes and manages a diffusion pipeline with all three optimizations.
    It supports both SANA models (transformer-based) and standard diffusion models.
    """
    
    def __init__(self, model_path=None, precision=None, pruning_amount=0.0, use_kv_cache=True):
        """Initialize the pipeline with combined optimizations
        
        Args:
            model_path: Path to the model (HuggingFace repo ID or local path)
            precision: Precision to use for quantization (int4, int8, etc.)
            pruning_amount: Amount of weights to prune (0.0 to 0.9)
            use_kv_cache: Whether to use KV caching
        """
        self.model_path = model_path
        self.precision = precision
        self.pruning_amount = pruning_amount
        self.use_kv_cache = use_kv_cache
        self.pipeline = None
        self.kv_cached_transformer = None
        self.pipeline_type = None
        
        # Load model if model_path is provided
        if model_path:
            self.load_model()
        
    def load_model(self):
        """Load model with optimizations (quantization, pruning, KV caching)"""
        print(f"Loading model from {self.model_path}...")
        
        # Apply memory optimizations
        setup_memory_optimizations_pruning()
        
        # Detect if this is a FLUX model or other type of model
        is_flux_model = "flux" in self.model_path.lower()
        
        # Step 1: Load model with quantization if applicable
        try:
            # if is_flux_model and HAS_NUNCHAKU:
            print(f"Nunchaku available: {HAS_NUNCHAKU}")
            if HAS_NUNCHAKU:
                # Load quantized FLUX model
                transformer, self.precision = self._load_quantized_model()
                quantized_size = get_model_size(transformer)
                print(f"Quantized model size: {quantized_size:.2f} MB")
                
                # Apply pruning if requested
                if self.pruning_amount > 0:
                    print(f"Applying pruning with amount {self.pruning_amount}...")
                    transformer = apply_magnitude_pruning(transformer, amount=self.pruning_amount)
                    pruned_size = get_model_size(transformer)
                    print(f"Pruned model size: {pruned_size:.2f} MB")
                    print(f"Size reduction: {(1 - pruned_size/quantized_size) * 100:.2f}%")
                
                pipe_path = "black-forest-labs/FLUX.1-dev"
                # pipe_path = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
                # Create FLUX pipeline
                # self.pipeline = SanaPipeline.from_pretrained(
                self.pipeline = FluxPipeline.from_pretrained(
                    pipe_path,
                    transformer=transformer,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True
                ).to("cuda")
                
                # Apply memory optimizations
                if hasattr(self.pipeline, "enable_attention_slicing"):
                    self.pipeline.enable_attention_slicing(1)
                
                if hasattr(self.pipeline, "enable_model_cpu_offload"):
                    self.pipeline.enable_model_cpu_offload()
                    
                # Save transformer for KV caching
                original_transformer = transformer
                
            else:
                # Load standard diffusion pipeline (SANA or other)
                self.pipeline = DiffusionPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                ).to("cuda")
                
                # Find transformer component
                original_transformer = self._locate_transformer()
            
            # Save pipeline type
            self.pipeline_type = type(self.pipeline).__name__
            print(f"Loaded pipeline: {self.pipeline_type}")
            
            # Check if it's a Sam3VideoModel
            is_sam3_model = (
                hasattr(original_transformer, '__class__') and 'Sam3VideoModel' in str(original_transformer.__class__)
            ) or (
                hasattr(original_transformer, '__name__') and 'Sam3VideoModel' in str(original_transformer.__name__)
            ) or (
                'Sam3VideoModel' in str(type(original_transformer))
            )
            
            if is_sam3_model:
                print("\n============================================")
                print("WARNING: Detected Sam3VideoModel")
                print("This model requires inference_session parameter")
                print("Auto-handling will be applied")
                print("============================================\n")
                
                # Initialize inference session if needed
                if not hasattr(original_transformer, 'inference_session'):
                    if hasattr(original_transformer, 'create_inference_session'):
                        try:
                            original_transformer.inference_session = original_transformer.create_inference_session()
                            print("Created inference session for Sam3VideoModel using create_inference_session()")
                        except Exception as e:
                            print(f"Warning: Could not create inference session: {e}")
                            # Create a basic inference session as fallback
                            original_transformer.inference_session = type('InferenceSession', (), {
                                'device': getattr(original_transformer, 'device', 'cuda'),
                                'dtype': str(getattr(original_transformer, 'dtype', 'float32')),
                                'batch_size': 1
                            })()
                            print("Created fallback inference session")
                    else:
                        # Create a basic inference session
                        original_transformer.inference_session = type('InferenceSession', (), {
                            'device': getattr(original_transformer, 'device', 'cuda'),
                            'dtype': str(getattr(original_transformer, 'dtype', 'float32')),
                            'batch_size': 1
                        })()
                        print("Created basic inference session (no create_inference_session method found)")
                else:
                    print("Sam3VideoModel already has inference_session")
            
            # Check if it's an SCM model
            if ("scm" in self.model_path.lower() or 
                "consistency" in self.pipeline_type.lower() or
                hasattr(self.pipeline, "is_scm")):
                print("\n============================================")
                print("WARNING: Detected SCM (Stable Consistency Model)")
                print("SCM models require exactly 2 inference steps")
                print("Any other value will be automatically adjusted to 2")
                print("============================================\n")
            
            # Apply KV caching if requested
            if self.use_kv_cache:
                # Create KV cached transformer
                self.kv_cached_transformer = KVCacheTransformer(original_transformer)
                
                # Get device and dtype from transformer if available
                device = original_transformer.device if hasattr(original_transformer, 'device') else torch.device('cuda')
                dtype = original_transformer.dtype if hasattr(original_transformer, 'dtype') else None
                
                # Move transformer to appropriate device and dtype
                if dtype is not None:
                    self.kv_cached_transformer.to(device=device, dtype=dtype)
                else:
                    self.kv_cached_transformer.to(device=device)
                
                # Enable KV caching
                self.kv_cached_transformer.enable_kv_caching()
                
                # Replace transformer in pipeline
                self._replace_transformer(original_transformer)
                
                print(f"Model loaded and KV caching enabled")
            else:
                print(f"Model loaded without KV caching")
            
            # Test pipeline with a simple prompt
            print("Testing pipeline with a simple prompt...")
            test_image = self.pipeline(
                "a high quality photo of a mountain landscape",
                num_inference_steps=1,
                guidance_scale=7.5
            ).images[0]
            print("Pipeline test successful!")
            
            # Clear memory
            del test_image
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _load_quantized_model(self):
        """Load FLUX model with quantization
        
        Returns:
            tuple: (transformer, precision)
        """
        precision = self.precision
        
        if precision is None:
            # Auto-detect precision
            precision = get_precision()
            print(f"Auto-detected precision: {precision}")
        
        # Default to int4 if precision is still None
        if precision is None:
            precision = "int4"
        
        # Make sure precision is a string without extra whitespace
        if isinstance(precision, str):
            precision = precision.strip()
        
        print(f"Loading model with {precision} quantization...")
        
        # Construct model path based on precision
        # model_path = f"mit-han-lab/svdq-{precision}-flux.1-dev"
        model_path = f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
        # model_path = f"nunchaku-tech/nunchaku-sana/svdq-{precision}_r32-sana1.6b.safetensors"
        
        try:
            transformer = NunchakuFluxTransformer2dModel.from_pretrained(model_path, offload=True)
            # transformer = NunchakuSanaTransformer2DModel.from_pretrained(model_path, offload=True)
        except Exception as e:
            print(f"Error loading model with precision {precision}: {e}")
        
        return transformer, precision
    
    def _locate_transformer(self):
        """Find the transformer component in the pipeline"""
        # Method 1: Check direct attribute
        if hasattr(self.pipeline, 'transformer'):
            transformer = self.pipeline.transformer
            print("Found transformer at pipeline.transformer")
            return transformer
        
        # Method 2: Check in components
        elif hasattr(self.pipeline, 'components') and 'transformer' in self.pipeline.components:
            transformer = self.pipeline.components['transformer']
            print("Found transformer at pipeline.components['transformer']")
            return transformer
        
        # Debug information when transformer not found
        else:
            print("\n=== Pipeline structure debug information ===")
            print(f"Pipeline type: {self.pipeline_type}")
            print("\nPipeline attributes:")
            
            # List pipeline attributes
            for attr in dir(self.pipeline):
                if not attr.startswith("_"):
                    try:
                        attr_value = getattr(self.pipeline, attr)
                        print(f"- {attr} ({type(attr_value).__name__})")
                        
                        # Show detailed information about components and model
                        if attr == "components" and isinstance(attr_value, dict):
                            print("  Components keys:", list(attr_value.keys()))
                        elif attr == "model" and attr_value is not None:
                            print("  Model attributes:")
                            for model_attr in dir(attr_value):
                                if not model_attr.startswith("_"):
                                    try:
                                        model_attr_value = getattr(attr_value, model_attr)
                                        print(f"    - {model_attr} ({type(model_attr_value).__name__})")
                                    except:
                                        print(f"    - {model_attr} (error accessing)")
                    except:
                        print(f"- {attr} (error accessing)")
            
            # Search for attributes that might contain transformer
            potential_transformer_containers = []
            for attr in dir(self.pipeline):
                if not attr.startswith("_"):
                    try:
                        attr_value = getattr(self.pipeline, attr)
                        if "transform" in str(attr_value).lower():
                            potential_transformer_containers.append(attr)
                    except:
                        pass
            
            if potential_transformer_containers:
                print("\nAttributes that might contain transformer:", potential_transformer_containers)
                
                # Try to get transformer from first attribute
                if potential_transformer_containers:
                    try:
                        transformer = getattr(self.pipeline, potential_transformer_containers[0])
                        print(f"Using {potential_transformer_containers[0]} as transformer")
                        return transformer
                    except:
                        pass
            
            # Try UNet as last resort for standard diffusion models
            if hasattr(self.pipeline, 'unet'):
                print("No transformer found, using UNet instead")
                return self.pipeline.unet
            
            raise AttributeError("Could not locate transformer in the pipeline structure")
    
    def _replace_transformer(self, original_transformer):
        """Replace original transformer with KV cached version"""
        # Save original transformer for potential restoration
        if not hasattr(self, 'original_transformer') or self.original_transformer is None:
            self.original_transformer = original_transformer
            print("Saved original transformer reference for fallback")
        
        # Save replacement locations for restoration
        self.transformer_locations = []
        replaced = False
        
        # Method 1: Replace via direct attribute
        if hasattr(self.pipeline, 'transformer'):
            self.pipeline.transformer = self.kv_cached_transformer
            self.transformer_locations.append(('direct', 'transformer'))
            print("Replaced transformer at pipeline.transformer")
            replaced = True
            
        # Method 2: Replace in components
        if hasattr(self.pipeline, 'components') and 'transformer' in self.pipeline.components:
            self.pipeline.components['transformer'] = self.kv_cached_transformer
            self.transformer_locations.append(('components', 'transformer'))
            print("Replaced transformer at pipeline.components['transformer']")
            replaced = True
            
        # Find and replace in all possible locations
        for attr in dir(self.pipeline):
            if attr.startswith("_") or attr in ['transformer', 'components']:
                continue
                
            try:
                attr_value = getattr(self.pipeline, attr)
                if attr_value is original_transformer:
                    setattr(self.pipeline, attr, self.kv_cached_transformer)
                    self.transformer_locations.append(('attr', attr))
                    print(f"Replaced transformer at pipeline.{attr}")
                    replaced = True
            except Exception as e:
                pass
        
        # Search in nested attributes
        for attr in dir(self.pipeline):
            if attr.startswith("_"):
                continue
                
            try:
                container = getattr(self.pipeline, attr)
                if hasattr(container, "__dict__"):
                    for sub_attr, sub_value in container.__dict__.items():
                        if sub_value is original_transformer:
                            setattr(container, sub_attr, self.kv_cached_transformer)
                            self.transformer_locations.append(('nested', attr, sub_attr))
                            print(f"Replaced transformer at pipeline.{attr}.{sub_attr}")
                            replaced = True
            except Exception as e:
                pass
                
        # Check result
        if replaced:
            print(f"Successfully replaced transformer at {len(self.transformer_locations)} locations")
        else:
            print("Warning: Could not find exact location to replace transformer")
            print("Will still use KV caching through the original reference")
    
    def restore_original_transformer(self):
        """Restore original transformer"""
        if hasattr(self, 'original_transformer') and self.original_transformer is not None:
            for loc_type, *loc_path in self.transformer_locations:
                try:
                    if loc_type == 'direct':
                        self.pipeline.transformer = self.original_transformer
                    elif loc_type == 'components':
                        self.pipeline.components['transformer'] = self.original_transformer
                    elif loc_type == 'attr':
                        setattr(self.pipeline, loc_path[0], self.original_transformer)
                    elif loc_type == 'nested':
                        container = getattr(self.pipeline, loc_path[0])
                        setattr(container, loc_path[1], self.original_transformer)
                except Exception as e:
                    print(f"Error restoring transformer at {loc_type} {loc_path}: {e}")
            print("Restored original transformer")
        else:
            print("No original transformer to restore")
    
    def generate_image(self, prompt, negative_prompt="", num_inference_steps=30, 
                      guidance_scale=7.5, seed=None, use_cache=True, height=1024, width=1024):
        """Generate an image using the optimized pipeline"""
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Clear KV cache before starting new image generation
        if use_cache and hasattr(self.kv_cached_transformer, 'clear_cache'):
            self.kv_cached_transformer.clear_cache()
            print("KV cache cleared for new image generation")
        
        # Start timing
        start_time = time.time()
        
        # Process API compatibility
        print(f"Using pipeline type: {self.pipeline_type}")
        
        # Check pipeline type
        pipeline_type = self.pipeline_type.lower() if self.pipeline_type else ""
        is_sana_sprint = "sanasprint" in pipeline_type
        
        # Prepare basic parameters
        kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
        }
        
        # Add negative_prompt only when supported
        if negative_prompt and not is_sana_sprint:
            kwargs["negative_prompt"] = negative_prompt
        
        # Set use_kv_cache state for transformer
        if hasattr(self.kv_cached_transformer, "use_kv_cache"):
            prev_use_cache = self.kv_cached_transformer.use_kv_cache
            self.kv_cached_transformer.use_kv_cache = use_cache
            print(f"Set KV caching state: {use_cache}")
        
        # Check if using SCM model
        is_scm_model = (
            "scm" in self.model_path.lower() if self.model_path else False or 
            hasattr(self.pipeline, "is_scm") or
            (hasattr(self.pipeline, "components") and 
             any("scm" in str(v).lower() for v in self.pipeline.components.values())) or
            "scm" in pipeline_type or
            "consistency" in pipeline_type
        )
        
        # If SCM model, adjust inference steps to 2
        if is_scm_model and num_inference_steps != 2:
            print(f"WARNING: SCM model detected. Forcing num_inference_steps=2 (was {num_inference_steps})")
            num_inference_steps = 2
            kwargs["num_inference_steps"] = 2
        
        # Try different methods to generate image
        for attempt, method_name in enumerate([
            "Standard parameters", 
            "Reduced parameters", 
            "Minimal parameters",
            "Original transformer"
        ]):
            try:
                if attempt == 0:
                    # Method 1: Full parameters with Sam3VideoModel handling
                    try:
                        image = self.pipeline(**kwargs).images[0]
                    except TypeError as e:
                        if 'inference_session' in str(e):
                            print("Detected Sam3VideoModel error, adding inference_session to pipeline call")
                            # Try to add inference_session at pipeline level
                            if hasattr(self.pipeline, 'transformer') and hasattr(self.pipeline.transformer, 'inference_session'):
                                kwargs['inference_session'] = self.pipeline.transformer.inference_session
                            image = self.pipeline(**kwargs).images[0]
                        else:
                            raise e
                
                elif attempt == 1:
                    # Method 2: Reduced parameters
                    print(f"Attempt {attempt+1}: Trying with fewer parameters...")
                    simple_kwargs = {
                        "prompt": prompt,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale
                    }
                    try:
                        image = self.pipeline(**simple_kwargs).images[0]
                    except TypeError as e:
                        if 'inference_session' in str(e):
                            if hasattr(self.pipeline, 'transformer') and hasattr(self.pipeline.transformer, 'inference_session'):
                                simple_kwargs['inference_session'] = self.pipeline.transformer.inference_session
                            image = self.pipeline(**simple_kwargs).images[0]
                        else:
                            raise e
                
                elif attempt == 2:
                    # Method 3: Minimal parameters
                    print(f"Attempt {attempt+1}: Trying with minimal parameters...")
                    try:
                        image = self.pipeline(prompt=prompt).images[0]
                    except TypeError as e:
                        if 'inference_session' in str(e):
                            minimal_kwargs = {'prompt': prompt}
                            if hasattr(self.pipeline, 'transformer') and hasattr(self.pipeline.transformer, 'inference_session'):
                                minimal_kwargs['inference_session'] = self.pipeline.transformer.inference_session
                            image = self.pipeline(**minimal_kwargs).images[0]
                        else:
                            raise e
                
                else:
                    # Method 4: Use original transformer
                    print(f"Attempt {attempt+1}: Using original transformer...")
                    if hasattr(self, 'original_transformer') and hasattr(self.pipeline, 'transformer'):
                        # Save current transformer and restore original
                        temp_transformer = self.pipeline.transformer
                        self.pipeline.transformer = self.original_transformer
                        
                        # Try generation
                        try:
                            image = self.pipeline(**kwargs).images[0]
                        finally:
                            # Restore KV cached transformer
                            self.pipeline.transformer = temp_transformer
                    else:
                        # Last resort
                        image = self.pipeline(prompt).images[0]
                
                # If successful, break loop
                print(f"Successfully generated image using method: {method_name}")
                break
                
            except Exception as e:
                print(f"Error with {method_name}: {str(e)}")
                
                # If tried all methods
                if attempt == 3:
                    # Print debug information
                    print("\n=== Debug Information ===")
                    print(f"Pipeline type: {self.pipeline_type}")
                    if hasattr(self.pipeline, 'components'):
                        print(f"Available components: {list(self.pipeline.components.keys())}")
                    if hasattr(self.kv_cached_transformer, 'transformer_structure'):
                        print(f"KV cached transformer structure: {str(self.kv_cached_transformer.transformer_structure)[:500]}...")
                    
                    raise RuntimeError(f"Failed to generate image after all attempts: {e}")
                    
        # Restore original use_kv_cache state if changed
        if hasattr(self.kv_cached_transformer, "use_kv_cache") and 'prev_use_cache' in locals():
            self.kv_cached_transformer.use_kv_cache = prev_use_cache
            
        # Calculate time
        generation_time = time.time() - start_time
        
        print(f"Image generated in {generation_time:.2f} seconds with KV caching {'enabled' if use_cache else 'disabled'}")
        
        return image, generation_time
    
    def generate_images_with_dataset(self, image_filename_to_caption, output_dir, num_images=10, 
                                 num_inference_steps=30, guidance_scale=7.5, use_cache=True):
        """
        Generate images using captions from COCO dataset
        
        Args:
            image_filename_to_caption: Dictionary mapping filenames to captions
            output_dir: Directory to save generated images
            num_images: Number of images to generate
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            use_cache: Whether to use KV cache
            
        Returns:
            generation_times: Dictionary mapping filenames to generation times
            generation_metadata: List of metadata for each generated image
        """
        # Limit the number of images to generate
        filenames_captions = list(image_filename_to_caption.items())[:num_images]
        
        generation_times = {}
        generation_metadata = []
        
        # Generate images for each caption
        for i, (filename, prompt) in enumerate(tqdm(filenames_captions, desc="Generating images with captions")):
            output_path = os.path.join(output_dir, filename)
            
            print(f"\n\n{'='*80}")
            print(f"Processing image {i+1}/{len(filenames_captions)}: {filename}")
            print(f"Prompt: {prompt}")
            print(f"Output will be saved to: {output_path}")
            print(f"{'='*80}")
            
            # Skip if the image already exists
            if os.path.exists(output_path):
                print(f"Skipping generation for {filename} (already exists)")
                continue
                
            try:
                # Generate image with optimizations
                image, generation_time = self.generate_image(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    use_cache=use_cache
                )
                
                # Save the image
                image.save(output_path)
                
                print(f"Generated image {i+1}/{len(filenames_captions)}: {filename}")
                print(f"Generation time: {generation_time:.2f} seconds")
                
                # Create metadata
                metadata = {
                    "generated_image_path": output_path,
                    "original_filename": filename,
                    "caption_used": prompt,
                    "generation_time": generation_time,
                    "guidance_scale": guidance_scale,
                    "num_steps": num_inference_steps,
                    "use_cache": use_cache,
                    "precision": self.precision,
                    "pruning_amount": self.pruning_amount
                }
                
                generation_metadata.append(metadata)
                generation_times[filename] = generation_time
                
            except Exception as e:
                print(f"Error when generating image for prompt '{prompt[:50]}...': {e}")
                generation_times[filename] = -1  # Indicate an error
                
            # Force GC to free memory
            if i % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        return generation_times, generation_metadata

# ======================================================================
# CAPTION UTILITIES
# ======================================================================

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
        print(f"  - {filename}: {caption}")
    
    return image_filename_to_caption, image_dimensions

# ======================================================================
# VRAM MONITORING
# ======================================================================

def monitor_generation_vram(pipeline, prompt, use_cache=True, num_inference_steps=30, guidance_scale=7.5, 
                    height=1024, width=1024, negative_prompt="", seed=None):
    """
    Monitor VRAM usage during image generation
    
    Args:
        pipeline: The OptimizedDiffusionPipeline instance
        prompt: Text prompt for image generation
        use_cache: Whether to use KV cache
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        height: Image height
        width: Image width
        negative_prompt: Negative prompt
        seed: Random seed
        
    Returns:
        Dictionary with VRAM usage statistics and generation time
    """
    
    # Set up monitoring
    global vram_samples, stop_monitoring
    vram_samples = []
    stop_monitoring.clear()
    
    # Define a monitoring function
    def _monitor_vram(device_index=0):
        try:
            # Try with pynvml first
            if HAVE_PYNVML:
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
            else:
                # Fall back to torch.cuda
                while not stop_monitoring.is_set():
                    try:
                        if torch.cuda.is_available():
                            # Get current allocated memory in bytes and convert to GB
                            allocated_gb = torch.cuda.memory_allocated(device_index) / 1024**3
                            vram_samples.append(allocated_gb)
                            time.sleep(0.1)  # Sample every 100 milliseconds
                    except Exception as error:
                        print(f"Error during VRAM monitoring with torch.cuda: {error}")
                        break
        except Exception as e:
            print(f"Error in VRAM monitoring: {e}")
    
    # Clean up memory before starting
    torch.cuda.empty_cache()
    gc.collect()
    
    # Start the monitoring thread
    monitor_thread = threading.Thread(target=_monitor_vram)
    monitor_thread.start()
    
    # Run the image generation
    start_time = time.time()
    try:
        if hasattr(pipeline, 'kv_cached_transformer'):
            # Enable/disable KV caching based on parameter
            if hasattr(pipeline.kv_cached_transformer, "use_kv_cache"):
                prev_use_cache = pipeline.kv_cached_transformer.use_kv_cache
                pipeline.kv_cached_transformer.use_kv_cache = use_cache
        
        # Generate the image
        pipeline.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            use_cache=use_cache,
            height=height,
            width=width
        )
        
        # Restore previous KV cache setting
        if hasattr(pipeline, 'kv_cached_transformer') and hasattr(pipeline.kv_cached_transformer, "use_kv_cache"):
            pipeline.kv_cached_transformer.use_kv_cache = prev_use_cache
            
    except Exception as e:
        print(f"Error during image generation: {e}")
    finally:
        generation_time = time.time() - start_time
        
        # Stop the monitoring thread
        stop_monitoring.set()
        monitor_thread.join()
    
    # Calculate statistics
    results = {
        "generation_time_seconds": generation_time,
        "used_kv_cache": use_cache,
    }
    
    if vram_samples:
        avg_vram = sum(vram_samples) / len(vram_samples)
        peak_vram = max(vram_samples)
        min_vram = min(vram_samples)
        initial_vram = vram_samples[0] if vram_samples else 0
        
        # Calculate standard deviation
        variance = sum((x - avg_vram) ** 2 for x in vram_samples) / len(vram_samples)
        std_dev = variance ** 0.5
        
        results.update({
            "average_vram_gb": avg_vram,
            "peak_vram_gb": peak_vram,
            "min_vram_gb": min_vram,
            "initial_vram_gb": initial_vram,
            "vram_increase_gb": peak_vram - initial_vram,
            "std_dev_gb": std_dev,
            "samples_count": len(vram_samples),
        })
        
        # Add a subset of samples for visualization
        if len(vram_samples) > 100:
            # Take every Nth sample to get around 100 samples
            n = max(1, len(vram_samples) // 100)
            results["vram_samples"] = vram_samples[::n]
        else:
            results["vram_samples"] = vram_samples
    else:
        results["error"] = "No VRAM samples collected"
    
    return results

# ======================================================================
# MAIN FUNCTION
# ======================================================================

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Combined Optimizations (Quantization, Pruning, KV Caching) with COCO dataset")
    
    # Model and optimization parameters
    parser.add_argument("--model_path", type=str, default="Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
                        help="Path to the model (HuggingFace repo ID or local path)")
    parser.add_argument("--pruning_amount", type=float, default=0.3, 
                        help="Amount of weights to prune (0.0 to 0.9)")
    parser.add_argument("--precision", type=str, default="int4", 
                        choices=["int8", "int4", "int2"],
                        help="Precision to use for quantization (FLUX models only)")
    parser.add_argument("--use_kv_cache", action="store_true", default=True,
                        help="Whether to use KV caching")
    
    # Generation parameters
    parser.add_argument("--num_images", type=int, default=10,
                        help="Number of COCO images to process")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                        help="Guidance scale for image generation")
    
    # Metrics parameters
    parser.add_argument("--skip_metrics", action="store_true",
                        help="Skip calculation of image quality metrics")
    parser.add_argument("--metrics_subset", type=int, default=500,
                        help="Number of images to use for metrics calculation")
    
    # Monitoring parameters
    parser.add_argument("--monitor_vram", action="store_true",
                        help="Monitor VRAM usage during image generation")
    parser.add_argument("--use_coco", action="store_true",
                        help="Use COCO dataset captions for image generation")
    parser.add_argument("--use_flickr8k", action="store_true",
                        help="Use Flickr8k dataset captions for image generation")
    args = parser.parse_args()
    
    # Parse arguments
    args = parser.parse_args()
    
    # Authenticate with Hugging Face (if token available)
    login(token="")
    
    # Setup memory optimizations
    setup_memory_optimizations_pruning()

    # Create output directories based on optimizations used
    optim_methods = []
    if "flux" in args.model_path.lower() or args.precision != "none":
        optim_methods.append(f"quant_{args.precision}")
    if args.pruning_amount > 0:
        optim_methods.append(f"prune_{int(args.pruning_amount*100)}")
    if args.use_kv_cache:
        optim_methods.append("kvcache")
    
    optimization_name = "_".join(optim_methods) if optim_methods else "no_optim"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    
    image_filename_to_caption = {}
    image_dimensions = {}
    original_dir = ""
    output_dir = ""

    if args.use_coco:
        # Define COCO paths
        coco_dir = "coco"
        annotations_file = os.path.join(coco_dir, "annotations", "captions_val2017.json")
        
        # 1. Load COCO captions
        print("\n=== Loading COCO Captions ===")
        image_filename_to_caption, image_dimensions = load_coco_captions(annotations_file, limit=args.num_images)
    
        # 3. Generate images with optimizations
        print(f"\n=== Generating Images for {len(image_filename_to_caption)} COCO Captions ===")
        output_dir = f"combination/combined_outputs_{coco_dir}_{optimization_name}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        original_dir = os.path.join(coco_dir, "val2017")

    elif args.use_flickr8k:
        # Define paths
        flickr8k_dir = "flickr8k"
        images_dir = os.path.join(flickr8k_dir, "Images")
        captions_file = os.path.join(flickr8k_dir, "captions.txt")
        print("\n=== Loading Flickr8k Captions ===")
        image_filename_to_caption, image_dimensions = process_flickr8k(images_dir, captions_file)

        print(f"\n=== Generating Images for {len(image_filename_to_caption)} Flickr8k Captions ===")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"combination/combined_outputs_{flickr8k_dir}_{optimization_name}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        original_dir = images_dir
    
    # 2. Load the optimized model
    print("\n=== Loading Model with Optimizations ===")
    try:
        # Create optimized pipeline
        pipeline = OptimizedDiffusionPipeline(
            model_path=args.model_path,
            precision=args.precision,
            pruning_amount=args.pruning_amount,
            use_kv_cache=args.use_kv_cache
        )
    except Exception as e:
        print(f"Fatal error loading model: {e}")
        return
    
    # If VRAM monitoring is enabled, run a single test with monitoring first
    if args.monitor_vram:
        print("\n=== Running VRAM Monitoring Test ===")
        # Take the first caption for the test
        test_filename, test_caption = next(iter(image_filename_to_caption.items()))
        print(f"Running VRAM monitoring with test caption: {test_filename}")
        
        vram_stats = monitor_generation_vram(
            pipeline=pipeline,
            prompt=test_caption,
            use_cache=args.use_kv_cache,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale
        )
        
        # Display VRAM usage statistics
        print("\n=== VRAM Usage Statistics ===")
        print(f"Generation time: {vram_stats['generation_time_seconds']:.2f} seconds")
        print(f"Average VRAM usage: {vram_stats['average_vram_gb']:.2f} GB")
        print(f"Peak VRAM usage: {vram_stats['peak_vram_gb']:.2f} GB")
        print(f"VRAM increase during generation: {vram_stats['vram_increase_gb']:.2f} GB")
        print(f"Standard deviation: {vram_stats['std_dev_gb']:.4f} GB")
        
        # Save VRAM statistics to a JSON file
        vram_stats_path = os.path.join(output_dir, "vram_stats.json")
        with open(vram_stats_path, 'w', encoding='utf-8') as f:
            json.dump(vram_stats, f, indent=4)
        print(f"VRAM statistics saved to {vram_stats_path}")
    
    # Continue with normal generation for all captions
    generation_times, generation_metadata = pipeline.generate_images_with_dataset(
        image_filename_to_caption, 
        output_dir, 
        num_images=args.num_images,
        num_inference_steps=args.steps, 
        guidance_scale=args.guidance_scale,
        use_cache=args.use_kv_cache
    )
    write_generation_metadata_to_file(os.path.join(output_dir, "generation_metadata.json"), generation_metadata)
    
    # 4. Calculate Image Quality Metrics (if not skipped)
    metrics_results = {}
    
    if args.skip_metrics:
        print("\n=== Skipping Image Quality Metrics (--skip_metrics flag set) ===")
    else:
        print("\n=== Calculating Image Quality Metrics ===")
        
        # Create directory for resized images (needed for FID and PSNR)
        resized_output_dir = os.path.join(output_dir, "resized")
        os.makedirs(resized_output_dir, exist_ok=True)
        
        # Resize images for metrics calculation
        try:
            print("\n--- Resizing Images for Metrics Calculation ---")
            resize_images(output_dir, resized_output_dir, image_dimensions)
        except Exception as e:
            print(f"Error resizing images: {e}")
        
        # Calculate FID score
        try:
            print("\n--- Calculating FID Score ---")
            fid_score = calculate_fid(output_dir, resized_output_dir, original_dir)
            metrics_results["fid_score"] = fid_score
            print(f"FID Score: {fid_score:.4f}")
        except Exception as e:
            print(f"Error calculating FID: {e}")
        
        # Calculate CLIP Score
        try:
            print("\n--- Calculating CLIP Score ---")
            clip_score = calculate_clip_score(output_dir, image_filename_to_caption)
            metrics_results["clip_score"] = clip_score
            print(f"CLIP Score: {clip_score:.4f}")
        except Exception as e:
            print(f"Error calculating CLIP Score: {e}")
        
        # Calculate ImageReward
        try:
            print("\n--- Calculating ImageReward ---")
            image_reward = compute_image_reward(output_dir, image_filename_to_caption)
            metrics_results["image_reward"] = image_reward
            print(f"ImageReward: {image_reward:.4f}")
        except Exception as e:
            print(f"Error calculating ImageReward: {e}")
        
        # Calculate LPIPS - we need original images to compare with generated images
        try:
            print("\n--- Calculating LPIPS ---")
            # For dataset, we need to select a subset of filenames for LPIPS calculation
            # We should compare resized images to ensure dimensions match
            
            # Create a directory for resized original images
            resized_original_dir = os.path.join(output_dir, "resized_original")
            os.makedirs(resized_original_dir, exist_ok=True)
            
            # Get list of generated filenames that have been resized
            generated_filenames = [f for f in os.listdir(resized_output_dir) if os.path.isfile(os.path.join(resized_output_dir, f)) 
                                and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            
            # Use the metrics_subset parameter to limit the number of images for metrics
            subset_size = min(args.metrics_subset, len(generated_filenames))
            selected_filenames = generated_filenames[:subset_size]
            
            # Manually resize original images dataset to match generated images for LPIPS calculation
            for filename in tqdm(selected_filenames, desc="Resizing original images for LPIPS"):
                original_path = os.path.join(original_dir, filename)
                resized_original_path = os.path.join(resized_original_dir, filename)
                
                if os.path.exists(original_path):
                    try:
                        # Get the size of the resized generated image for consistency
                        generated_img_path = os.path.join(resized_output_dir, filename)
                        generated_img = Image.open(generated_img_path)
                        target_size = generated_img.size
                        
                        # Resize the original image to match the generated image
                        original_img = Image.open(original_path).convert("RGB")
                        resized_original_img = original_img.resize(target_size, Image.LANCZOS)
                        resized_original_img.save(resized_original_path)
                    except Exception as e:
                        print(f"Error resizing original image {filename}: {e}")
            
            # Now calculate LPIPS using the resized original and generated images
            lpips_score = calculate_lpips(resized_original_dir, resized_output_dir, selected_filenames)
            metrics_results["lpips"] = lpips_score
        except Exception as e:
            print(f"Error calculating LPIPS: {e}")
        
        # Calculate PSNR - we need original images to compare with generated images
        try:
            print("\n--- Calculating PSNR ---")
            # For dataset, we use resized generated images
            generated_filenames = [f for f in os.listdir(resized_output_dir) if os.path.isfile(os.path.join(resized_output_dir, f))
                                and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            
            # Use the metrics_subset parameter to limit the number of images for metrics
            subset_size = min(args.metrics_subset, len(generated_filenames))
            psnr_score = calculate_psnr_resized(original_dir, resized_output_dir, generated_filenames[:subset_size])
            metrics_results["psnr"] = psnr_score
        except Exception as e:
            print(f"Error calculating PSNR: {e}")

    # Save metadata to JSON file
    # metadata_file = os.path.join(output_dir, "optimization_metadata.json")
    # try:
    #     # Add metrics and model info to metadata
    #     combined_metadata = {
    #         "model_info": {
    #             "model_path": args.model_path,
    #             "precision": args.precision if "flux" in args.model_path.lower() else "N/A",
    #             "pruning_amount": args.pruning_amount,
    #             "use_kv_cache": args.use_kv_cache
    #         },
    #         "generation_settings": {
    #             "num_inference_steps": args.steps,
    #             "guidance_scale": args.guidance_scale,
    #             "num_images": args.num_images
    #         },
    #         "metrics": metrics_results,
    #         "generation_metadata": generation_metadata,
    #     }
        
        # with open(metadata_file, 'w', encoding='utf-8') as f:
        #     json.dump(combined_metadata, f, indent=2)
        # print(f"Saved metadata to: {metadata_file}")
    # except Exception as e:
    #     print(f"Error when saving metadata JSON: {e}")
    
    # Calculate and print statistics
    successful_generations = [t for t in generation_times.values() if t > 0]
    if successful_generations:
        avg_time = sum(successful_generations) / len(successful_generations)
        print(f"\n=== Generation Statistics ===")
        print(f"Successfully generated {len(successful_generations)}/{len(image_filename_to_caption)} images")
        print(f"Average generation time: {avg_time:.2f} seconds per image")
        print(f"Total generation time: {sum(successful_generations):.2f} seconds")
    
    # Save summary report
    with open(os.path.join(output_dir, "optimization_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Combined Optimizations Summary ===\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Optimizations applied:\n")
        
        if "flux" in args.model_path.lower():
            f.write(f"- Quantization: {args.precision}\n")
            
        if args.pruning_amount > 0:
            f.write(f"- Pruning: {args.pruning_amount*100:.1f}%\n")
            
        if args.use_kv_cache:
            f.write(f"- KV Caching: Enabled\n")

        f.write(f"\nProcessed {len(image_filename_to_caption)} Dataset captions\n")
        f.write(f"Inference steps: {args.steps}\n")
        f.write(f"Guidance scale: {args.guidance_scale}\n\n")
        
        if successful_generations:
            f.write(f"Successfully generated {len(successful_generations)}/{len(image_filename_to_caption)} images\n")
            f.write(f"Average generation time: {avg_time:.2f} seconds per image\n")
            f.write(f"Total generation time: {sum(successful_generations):.2f} seconds\n\n")
        
        # Add metrics results to summary
        if metrics_results:
            f.write("=== Image Quality Metrics ===\n")
            for metric_name, value in metrics_results.items():
                f.write(f"{metric_name}: {value:.4f}\n")
    
    print(f"\nCompleted. Results saved to {output_dir}")
    print(f"Summary available at: {os.path.join(output_dir, 'optimization_summary.txt')}")

if __name__ == "__main__":
    main()
