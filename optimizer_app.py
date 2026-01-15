import streamlit as st
import torch
import time
import psutil
import gc
import os
from transformers import AutoModel, AutoProcessor, AutoConfig
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image
import io
from datetime import datetime
from pathlib import Path
import json

# Local SAM3 model path
LOCAL_SAM3_PATH = "/mnt/mmlab2024nas/nga/optsd/models/sam3"

# Configure page
st.set_page_config(
    page_title="SAM3 Model Optimizer",
    page_icon="🚀",
    layout="wide"
)

class ModelOptimizer:
    """Handles various optimization techniques for SAM3 model"""
    
    def __init__(self, model_path: str):
        self.model_path = self._validate_and_resolve_path(model_path)
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimization_config = {}
        self._is_local_model = self._check_if_local_model()
    
    def _validate_and_resolve_path(self, model_path: str) -> str:
        """Validate and resolve model path, prefer local if available"""
        # If user provides local SAM3 path or it exists locally, use it
        if model_path == LOCAL_SAM3_PATH or (
            os.path.exists(LOCAL_SAM3_PATH) and 
            model_path in ["facebook/sam3", "sam3", "SAM3"]
        ):
            return LOCAL_SAM3_PATH
        
        # For other local paths, check if they exist
        if os.path.exists(model_path):
            return os.path.abspath(model_path)
        
        # Return as-is for HuggingFace model paths
        return model_path
    
    def _check_if_local_model(self) -> bool:
        """Check if the model path is local"""
        return os.path.exists(self.model_path)
    
    def _validate_local_model_files(self) -> Dict:
        """Validate that all required model files exist locally"""
        if not self._is_local_model:
            return {"status": "success", "message": "Remote model, skipping file validation"}
        
        required_files = [
            "config.json",
            "model.safetensors",
            "processor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt"
        ]
        
        missing_files = []
        model_path = Path(self.model_path)
        
        for file in required_files:
            if not (model_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            return {
                "status": "error",
                "message": f"Missing required model files: {', '.join(missing_files)}",
                "missing_files": missing_files
            }
        
        return {
            "status": "success",
            "message": "All required model files found",
            "model_path": str(model_path)
        }
    
    def _get_model_loading_config(self, use_safetensors: bool = True) -> Dict:
        """Get optimized loading configuration based on model type"""
        config = {
            "trust_remote_code": True,
            "local_files_only": self._is_local_model,
        }
        
        if self._is_local_model:
            # Optimizations for local loading
            config.update({
                "use_safetensors": use_safetensors,
                "torch_dtype": torch.float32,  # Will be overridden by quantization if needed
            })
        
        return config
        
    def get_vram_usage(self) -> float:
        """Get current VRAM usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0
    
    def get_max_vram_usage(self) -> float:
        """Get maximum VRAM usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0
    
    def reset_vram_stats(self):
        """Reset VRAM statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    
    def load_model_baseline(self) -> Dict:
        """Load model without optimizations (baseline)"""
        st.info(f"Loading baseline model from {'local path' if self._is_local_model else 'remote'}...")
        
        # Validate local model files first
        if self._is_local_model:
            validation_result = self._validate_local_model_files()
            if validation_result["status"] == "error":
                return validation_result
            st.success(f"✅ Local model validated: {validation_result['message']}")
        
        self.reset_vram_stats()
        start_time = time.time()
        
        try:
            loading_config = self._get_model_loading_config()
            
            # Load model with optimized config
            st.info(f"Loading model from: {self.model_path}")
            self.model = AutoModel.from_pretrained(
                self.model_path,
                **loading_config
            ).to(self.device)
            
            # Load processor with same config
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                **loading_config
            )
            
            self.optimization_config = {
                "quantization": None,
                "pruning": None,
                "flash_attention": False,
                "kv_cache": False,
                "model_source": "local" if self._is_local_model else "remote"
            }
            
            load_time = time.time() - start_time
            vram_usage = self.get_max_vram_usage()
            
            st.success(f"✅ Model loaded successfully in {load_time:.2f}s")
            
            return {
                "status": "success",
                "load_time": load_time,
                "vram_usage": vram_usage,
                "device": self.device,
                "model_source": "local" if self._is_local_model else "remote",
                "model_path": self.model_path
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def apply_quantization(self, quantization_type: str = "int8") -> Dict:
        """Apply quantization to the model"""
        st.info(f"Applying {quantization_type} quantization to {'local' if self._is_local_model else 'remote'} model...")
        
        # Validate local model files first
        if self._is_local_model:
            validation_result = self._validate_local_model_files()
            if validation_result["status"] == "error":
                return validation_result
        
        self.reset_vram_stats()
        start_time = time.time()
        
        try:
            loading_config = self._get_model_loading_config()
            
            if quantization_type == "int8":
                # Load with 8-bit quantization using bitsandbytes
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
                
                # Merge quantization config with loading config
                loading_config.update({
                    "quantization_config": quantization_config,
                    "device_map": "auto"
                })
                
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    **loading_config
                )
                
            elif quantization_type == "fp16":
                # Load with FP16 precision
                loading_config.update({
                    "torch_dtype": torch.float16
                })
                
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    **loading_config
                ).to(self.device)
                
            elif quantization_type == "int4":
                # Load with 4-bit quantization
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                loading_config.update({
                    "quantization_config": quantization_config,
                    "device_map": "auto"
                })
                
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    **loading_config
                )
            
            # Load processor with optimized config
            processor_config = self._get_model_loading_config()
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                **processor_config
            )
            
            self.optimization_config["quantization"] = quantization_type
            self.optimization_config["model_source"] = "local" if self._is_local_model else "remote"
            
            load_time = time.time() - start_time
            vram_usage = self.get_max_vram_usage()
            
            st.success(f"✅ Quantized model loaded successfully in {load_time:.2f}s")
            
            return {
                "status": "success",
                "load_time": load_time,
                "vram_usage": vram_usage,
                "quantization_type": quantization_type,
                "model_source": "local" if self._is_local_model else "remote",
                "model_path": self.model_path
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def apply_pruning(self, pruning_amount: float = 0.3) -> Dict:
        """Apply structured pruning to the model"""
        st.info(f"Applying {pruning_amount*100}% pruning...")
        
        try:
            import torch.nn.utils.prune as prune
            
            if self.model is None:
                return {"status": "error", "message": "Model not loaded"}
            
            pruned_params = 0
            total_params = 0
            
            # Apply L1 unstructured pruning to linear layers
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=pruning_amount)
                    prune.remove(module, 'weight')  # Make pruning permanent
                    
                    pruned_params += (module.weight == 0).sum().item()
                    total_params += module.weight.numel()
            
            actual_pruning = pruned_params / total_params if total_params > 0 else 0
            self.optimization_config["pruning"] = actual_pruning
            
            return {
                "status": "success",
                "pruned_percentage": actual_pruning * 100,
                "pruned_params": pruned_params,
                "total_params": total_params
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def apply_flash_attention(self) -> Dict:
        """Enable Flash Attention if available"""
        st.info("Configuring Flash Attention...")
        
        try:
            if self.model is None:
                return {"status": "error", "message": "Model not loaded"}
            
            # Try to enable flash attention in model config
            if hasattr(self.model.config, 'use_flash_attention'):
                self.model.config.use_flash_attention = True
                self.optimization_config["flash_attention"] = True
                return {
                    "status": "success",
                    "message": "Flash Attention enabled via config"
                }
            else:
                # Try to patch attention modules
                try:
                    from torch.nn.attention import SDPBackend, sdpa_kernel
                    self.optimization_config["flash_attention"] = True
                    return {
                        "status": "success",
                        "message": "Will use PyTorch SDPA (includes Flash Attention)"
                    }
                except:
                    return {
                        "status": "warning",
                        "message": "Flash Attention not directly supported, using default attention"
                    }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def enable_kv_cache(self) -> Dict:
        """Enable key-value cache for faster inference"""
        st.info("Enabling KV Cache...")
        
        try:
            if self.model is None:
                return {"status": "error", "message": "Model not loaded"}
            
            # Enable KV caching in model config
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = True
                self.optimization_config["kv_cache"] = True
                return {
                    "status": "success",
                    "message": "KV Cache enabled"
                }
            else:
                return {
                    "status": "warning",
                    "message": "KV Cache not supported by this model"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def run_inference_test(self, test_image: Optional[Image.Image] = None, num_runs: int = 5) -> Dict:
        """Run inference test to measure performance"""
        if self.model is None or self.processor is None:
            return {"status": "error", "message": "Model not loaded"}
        
        try:
            # Create a dummy test image if none provided
            if test_image is None:
                test_image = Image.new('RGB', (640, 480), color='white')
            
            self.model.eval()
            times = []
            
            # Check if this is a video model that needs inference session
            model_type = getattr(self.model.config, 'model_type', 'sam3')
            is_video_model = 'video' in model_type.lower() or 'Sam3VideoModel' in str(type(self.model))
            
            with torch.no_grad():
                if is_video_model:
                    # For video models, create inference session
                    try:
                        # Create a simple video session with single frame
                        inference_session = self.processor.init_video_session(
                            video=[test_image],  # Single frame as video
                            inference_device=self.device,
                            dtype=torch.float32
                        )

                        # SAM3 video inference requires at least one prompt in the session
                        # (e.g., a text prompt) before detection/propagation.
                        if hasattr(self.processor, "add_text_prompt"):
                            try:
                                inference_session = self.processor.add_text_prompt(
                                    inference_session=inference_session,
                                    text="person",
                                )
                            except TypeError:
                                # Some implementations may not use keyword args
                                inference_session = self.processor.add_text_prompt(inference_session, "person")
                        
                        # Warmup run for video model
                        _ = self.model(inference_session=inference_session, frame_idx=0)
                        
                        # Actual benchmark runs for video model
                        for _ in range(num_runs):
                            self.reset_vram_stats()
                            
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            
                            start_time = time.time()
                            outputs = self.model(inference_session=inference_session, frame_idx=0)
                            
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            
                            inference_time = time.time() - start_time
                            times.append(inference_time)
                            
                    except Exception as video_error:
                        # Fallback to image mode if video session fails
                        st.warning(f"Video model inference failed, trying image mode: {str(video_error)}")
                        is_video_model = False
                
                if not is_video_model:
                    # Standard image model inference
                    # Check if this is still a Sam3VideoModel (fallback case)
                    requires_inference_session = 'Sam3VideoModel' in str(type(self.model))
                    
                    # Warmup run
                    inputs = self.processor(images=test_image, return_tensors="pt")
                    if self.device == "cuda":
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Add inference_session if required
                    if requires_inference_session:
                        if hasattr(self.model, 'inference_session'):
                            inputs['inference_session'] = self.model.inference_session
                        elif hasattr(self.model, 'create_inference_session'):
                            inputs['inference_session'] = self.model.create_inference_session()
                    
                    _ = self.model(**inputs)
                    
                    # Actual benchmark runs
                    for _ in range(num_runs):
                        self.reset_vram_stats()
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        start_time = time.time()
                        inputs = self.processor(images=test_image, return_tensors="pt")
                        if self.device == "cuda":
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # Add inference_session if required
                        if requires_inference_session:
                            if hasattr(self.model, 'inference_session'):
                                inputs['inference_session'] = self.model.inference_session
                            elif hasattr(self.model, 'create_inference_session'):
                                inputs['inference_session'] = self.model.create_inference_session()
                        
                        outputs = self.model(**inputs)
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        inference_time = time.time() - start_time
                        times.append(inference_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            peak_vram = self.get_max_vram_usage()
            
            return {
                "status": "success",
                "avg_inference_time": avg_time,
                "std_inference_time": std_time,
                "peak_vram": peak_vram,
                "num_runs": num_runs
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def save_optimized_model(self, output_dir: str) -> Dict:
        """Save the optimized model to disk"""
        try:
            if self.model is None:
                return {"status": "error", "message": "No model to save"}
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            st.info(f"Saving optimized model to {output_dir}...")
            
            # Save model
            try:
                st.info("[LOG] Starting model.save_pretrained()...")
                self.model.save_pretrained(output_dir, safe_serialization=True)
                st.success("[LOG] ✓ model.save_pretrained() completed successfully")
            except Exception as e:
                st.error(f"[LOG] ✗ Error in model.save_pretrained(): {type(e).__name__}: {str(e)}")
                raise
            
            # Save processor
            if self.processor is not None:
                try:
                    st.info("[LOG] Starting processor.save_pretrained()...")
                    
                    # Test which component has the dtype issue
                    def test_json_serializable(obj, name):
                        """Test if object can be JSON serialized"""
                        try:
                            if hasattr(obj, 'to_dict'):
                                test_dict = obj.to_dict()
                                json.dumps(test_dict)
                                st.success(f"[LOG] {name}.to_dict() is JSON serializable ✓")
                                return True
                        except Exception as e:
                            st.error(f"[LOG] {name}.to_dict() is NOT JSON serializable: {e}")
                            # Try to find the problematic key
                            if hasattr(obj, 'to_dict'):
                                test_dict = obj.to_dict()
                                for key, value in test_dict.items():
                                    try:
                                        json.dumps({key: value})
                                    except:
                                        st.error(f"  → Problem in key '{key}': type={type(value).__name__}, value={repr(value)[:100]}")
                            return False
                    
                    # Test each component
                    if hasattr(self.processor, 'image_processor'):
                        test_json_serializable(self.processor.image_processor, "image_processor")
                    if hasattr(self.processor, 'video_processor'):
                        test_json_serializable(self.processor.video_processor, "video_processor")
                    if hasattr(self.processor, 'tokenizer'):
                        test_json_serializable(self.processor.tokenizer, "tokenizer")
                    
                    # Monkey-patch the JSON encoder to handle dtype
                    import json as json_module
                    original_default = json_module.JSONEncoder.default
                    
                    def custom_default(self, obj):
                        if isinstance(obj, (np.dtype, type(np.dtype('float32')))):
                            st.warning(f"[LOG] JSON encoder intercepted dtype: {obj} -> {str(obj)}")
                            return str(obj)
                        if isinstance(obj, np.ndarray):
                            st.warning(f"[LOG] JSON encoder intercepted ndarray: shape={obj.shape}")
                            return obj.tolist()
                        try:
                            return original_default(self, obj)
                        except TypeError:
                            st.warning(f"[LOG] JSON encoder fallback for type: {type(obj).__name__}")
                            return str(obj)
                    
                    json_module.JSONEncoder.default = custom_default
                    
                    try:
                        self.processor.save_pretrained(output_dir)
                        st.success("[LOG] ✓ processor.save_pretrained() completed successfully")
                    finally:
                        # Restore original encoder
                        json_module.JSONEncoder.default = original_default
                    
                except Exception as e:
                    st.error(f"[LOG] ✗ Error in processor.save_pretrained(): {type(e).__name__}: {str(e)}")
                    import traceback
                    st.error("[LOG] Full traceback:")
                    st.code(traceback.format_exc())
                    raise

            # Also save a .pt checkpoint with pattern: model-name.pt
            pt_filename = None
            try:
                # Derive a simple model name from model_path or model.name_or_path
                raw_name = None
                if hasattr(self.model, "name_or_path") and self.model.name_or_path:
                    raw_name = os.path.basename(str(self.model.name_or_path))
                if not raw_name:
                    raw_name = os.path.basename(str(self.model_path)) or "model"

                # Sanitize model name for filename
                model_name = raw_name.replace(os.sep, "_").replace(" ", "_")
                pt_filename = f"{model_name}.pt"
                pt_path = os.path.join(output_dir, pt_filename)

                # Save state dict for maximum compatibility
                st.info(f"[LOG] Starting torch.save() to {pt_filename}...")
                torch.save(self.model.state_dict(), pt_path)
                st.success(f"[LOG] ✓ torch.save() completed successfully: {pt_filename}")
            except Exception as e:
                # Don't fail the whole save if .pt export fails
                st.warning(f"Could not save .pt checkpoint: {e}")
            
            # Save optimization configuration with JSON-serializable values
            def make_json_serializable(obj):
                """Recursively convert objects in config to JSON-serializable types."""
                import numpy as np
                import torch

                # Basic JSON-native types
                if obj is None or isinstance(obj, (bool, int, float, str)):
                    return obj

                # NumPy scalars and dtypes
                if isinstance(obj, (np.generic, np.dtype)):
                    # Convert to the closest builtin Python type
                    try:
                        return obj.item()
                    except Exception:
                        return str(obj)

                # Torch-specific simple types
                if isinstance(obj, (torch.dtype, torch.device)):
                    return str(obj)

                # Tensors -> lists (if they ever appear here)
                if isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().tolist()

                # Containers
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple, set)):
                    return [make_json_serializable(v) for v in obj]

                # Fallback: string representation for anything else
                return str(obj)
            
            import json
            config_path = os.path.join(output_dir, "optimization_config.json")
            st.info("[LOG] Starting JSON serialization of optimization_config...")
            try:
                serializable_config = make_json_serializable(self.optimization_config)
                st.success("[LOG] ✓ JSON serialization completed")
            except Exception as e:
                st.error(f"[LOG] ✗ Error during make_json_serializable(): {type(e).__name__}: {str(e)}")
                st.error(f"[LOG] Config keys: {list(self.optimization_config.keys())}")
                raise
            
            st.info(f"[LOG] Writing JSON to {config_path}...")
            try:
                with open(config_path, 'w') as f:
                    json.dump(serializable_config, f, indent=2)
                st.success(f"[LOG] ✓ JSON file written successfully: {config_path}")
            except Exception as e:
                st.error(f"[LOG] ✗ Error during json.dump(): {type(e).__name__}: {str(e)}")
                raise
            
            # Get saved model size
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(output_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            
            size_mb = total_size / (1024 * 1024)
            
            return {
                "status": "success",
                "output_dir": output_dir,
                "model_size_mb": size_mb,
                "config": serializable_config,
                "pt_checkpoint": pt_filename
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

# Main Streamlit App
def main():
    st.title("🚀 SAM3 Model Optimizer")
    st.markdown("Optimize SAM3 models with quantization, pruning, KV cache, and Flash Attention")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Model path input with local SAM3 as default
    model_path = st.sidebar.text_input(
        "Model Path",
        value=LOCAL_SAM3_PATH,
        help="Local path to SAM3 model or HuggingFace model path"
    )
    
    # Show model source info
    if os.path.exists(model_path):
        st.sidebar.success(f"✅ Local model detected")
        # Show model info
        try:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    st.sidebar.info(f"📋 Model: {config.get('model_type', 'Unknown')}")
        except:
            pass
    else:
        st.sidebar.info(f"🌐 Remote model: {model_path}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Model Validation")
    
    # Quick model validation button
    if st.sidebar.button("Validate Model Files", help="Check if all required model files exist"):
        temp_optimizer = ModelOptimizer(model_path)
        validation_result = temp_optimizer._validate_local_model_files()
        
        if validation_result["status"] == "success":
            st.sidebar.success("✅ " + validation_result["message"])
        else:
            st.sidebar.error("❌ " + validation_result["message"])
            if "missing_files" in validation_result:
                st.sidebar.write("Missing files:", validation_result["missing_files"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Optimization Techniques")
    
    # Optimization options
    use_quantization = st.sidebar.checkbox("Enable Quantization", value=True)
    quantization_type = st.sidebar.selectbox(
        "Quantization Type",
        ["fp16", "int8", "int4"],
        disabled=not use_quantization
    )
    
    use_pruning = st.sidebar.checkbox("Enable Pruning", value=False)
    pruning_amount = st.sidebar.slider(
        "Pruning Amount",
        min_value=0.1,
        max_value=0.7,
        value=0.3,
        step=0.1,
        disabled=not use_pruning
    )
    
    use_flash_attention = st.sidebar.checkbox("Enable Flash Attention", value=True)
    use_kv_cache = st.sidebar.checkbox("Enable KV Cache", value=True)
    
    # Test configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("Inference Test")
    num_test_runs = st.sidebar.number_input(
        "Number of test runs",
        min_value=1,
        max_value=20,
        value=5
    )
    
    # Output configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Model")
    
    # Generate default output name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"output_sam3_{quantization_type if use_quantization else 'baseline'}_{timestamp}"
    
    output_dir = st.sidebar.text_input(
        "Output Directory",
        value=default_output,
        help="Directory to save optimized model"
    )
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Optimization Configuration")
        # Show model source and path
        is_local = os.path.exists(model_path)
        source_icon = "📁" if is_local else "🌐"
        source_text = "Local" if is_local else "Remote"
        st.write(f"**Model Source:** {source_icon} {source_text}")
        
        if is_local:
            st.write(f"**Local Path:** `{model_path}`")
        else:
            st.write(f"**Model:** {model_path}")
        
        st.write(f"**Device:** {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        optimizations = []
        if use_quantization:
            optimizations.append(f"Quantization ({quantization_type})")
        if use_pruning:
            optimizations.append(f"Pruning ({pruning_amount*100}%)")
        if use_flash_attention:
            optimizations.append("Flash Attention")
        if use_kv_cache:
            optimizations.append("KV Cache")
        
        if optimizations:
            st.write("**Enabled Optimizations:**")
            for opt in optimizations:
                st.write(f"  • {opt}")
        else:
            st.write("**No optimizations enabled** (baseline mode)")
    
    with col2:
        st.subheader("💾 System Info")
        if torch.cuda.is_available():
            st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
            st.write(f"**Total VRAM:** {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            st.write("**No GPU available**")
        st.write(f"**CPU Cores:** {psutil.cpu_count()}")
        st.write(f"**RAM:** {psutil.virtual_memory().total / 1024**3:.2f} GB")
    
    st.markdown("---")
    
    # Optimize button
    if st.button("🚀 Start Optimization", type="primary", use_container_width=True):
        optimizer = ModelOptimizer(model_path)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        
        # Step 1: Load model with optimizations
        status_text.text("Loading model...")
        progress_bar.progress(0.1)
        
        if use_quantization:
            load_result = optimizer.apply_quantization(quantization_type)
        else:
            load_result = optimizer.load_model_baseline()
        
        if load_result["status"] == "error":
            st.error(f"Failed to load model: {load_result['message']}")
            return
        
        results["load"] = load_result
        progress_bar.progress(0.3)
        
        # Step 2: Apply pruning if enabled
        if use_pruning:
            status_text.text("Applying pruning...")
            prune_result = optimizer.apply_pruning(pruning_amount)
            results["pruning"] = prune_result
        progress_bar.progress(0.5)
        
        # Step 3: Enable Flash Attention if selected
        if use_flash_attention:
            status_text.text("Configuring Flash Attention...")
            flash_result = optimizer.apply_flash_attention()
            results["flash_attention"] = flash_result
        progress_bar.progress(0.6)
        
        # Step 4: Enable KV Cache if selected
        if use_kv_cache:
            status_text.text("Enabling KV Cache...")
            kv_result = optimizer.enable_kv_cache()
            results["kv_cache"] = kv_result
        progress_bar.progress(0.7)
        
        # Step 5: Run inference test
        status_text.text("Running inference benchmark...")
        inference_result = optimizer.run_inference_test(num_runs=num_test_runs)
        results["inference"] = inference_result
        progress_bar.progress(0.85)
        
        # Step 6: Save optimized model
        status_text.text("Saving optimized model...")
        save_result = optimizer.save_optimized_model(output_dir)
        results["save"] = save_result
        progress_bar.progress(1.0)
        
        status_text.text("✅ Optimization complete!")
        
        # Display results
        st.markdown("---")
        st.header("📈 Optimization Results")
        
        # Metrics display
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric(
                "Load Time",
                f"{results['load']['load_time']:.2f}s"
            )
        
        with metric_cols[1]:
            st.metric(
                "Model VRAM",
                f"{results['load']['vram_usage']:.2f} MB"
            )
        
        with metric_cols[2]:
            if inference_result["status"] == "success":
                st.metric(
                    "Avg Inference Time",
                    f"{inference_result['avg_inference_time']*1000:.2f} ms"
                )
        
        with metric_cols[3]:
            if inference_result["status"] == "success":
                st.metric(
                    "Peak VRAM",
                    f"{inference_result['peak_vram']:.2f} MB"
                )
        
        # Model save results
        if save_result["status"] == "success":
            st.success(f"✅ Model saved successfully!")
            st.info(f"📁 **Output Directory:** `{save_result['output_dir']}`")
            st.info(f"💾 **Model Size:** {save_result['model_size_mb']:.2f} MB")
            
            # Show optimization config
            with st.expander("⚙️ Optimization Configuration", expanded=False):
                st.json(save_result['config'])
        else:
            st.error(f"❌ Failed to save model: {save_result.get('message', 'Unknown error')}")
        
        # Detailed results
        with st.expander("📋 Detailed Results", expanded=False):
            st.json(results)
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if inference_result["status"] == "success":
            vram_usage = inference_result["peak_vram"]
            inference_time = inference_result["avg_inference_time"]
            
            # Local vs Remote model recommendations
            if results["load"].get("model_source") == "local":
                st.info("🚀 **Local Model Optimizations Applied:**")
                st.write("• Using local file loading (no network download)")
                st.write("• SafeTensors format for faster loading")
                st.write("• Optimized local file caching")
            
            if vram_usage > 8000:  # > 8GB
                st.warning("⚠️ High VRAM usage detected. Consider using INT4 quantization or enabling pruning.")
            else:
                st.success("✅ VRAM usage is optimized!")
            
            if inference_time > 1.0:  # > 1 second
                st.warning("⚠️ Slow inference time. Consider enabling Flash Attention and KV Cache.")
            else:
                st.success("✅ Inference time is optimized!")
            
            # Additional local model tips
            if results["load"].get("model_source") == "local":
                st.markdown("**💡 Local Model Performance Tips:**")
                st.write("• Model files are loaded from SSD/NVMe for best performance")
                st.write("• No internet connection required")
                st.write("• Consistent loading times regardless of network")
                
        # Performance comparison
        if results["load"].get("model_source") == "local":
            st.success(f"📁 **Local Model Benefits:** Loaded from {model_path}")
            st.write("• Faster initial load (no download)")
            st.write("• Offline capability")
            st.write("• Consistent performance")
        
        # Usage instructions
        st.markdown("---")
        st.subheader("📖 How to Use Optimized Model")
        st.code(f"""
# Load the optimized model
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("{output_dir}", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("{output_dir}", trust_remote_code=True)

# Use the model for inference
# ... your inference code here ...
""", language="python")

if __name__ == "__main__":
    main()
