#!/usr/bin/env python3
"""
GPU-OPTIMIZED STREAMLIT VLM POWER EQUIPMENT ANALYZER - SYNTAX FIXED
==================================================================
High-performance local VLM analysis with proper GPU utilization

Usage: streamlit run fixed_streamlit_vlm.py
"""

import streamlit as st
import sys
import subprocess
import os
import json
import io
import gc
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Streamlit page config
st.set_page_config(
    page_title="GPU VLM Power Equipment Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def install_gpu_packages():
    """Install GPU-optimized packages with proper order"""

    core_packages = [
        ('torch_gpu', 'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'),
        ('transformers', 'transformers>=4.35.0'),
        ('accelerate', 'accelerate'),
        ('pillow', 'pillow'),
        ('numpy', 'numpy'),
        ('requests', 'requests')
    ]

    optional_packages = [
        ('bitsandbytes', 'bitsandbytes'),
        ('matplotlib', 'matplotlib'),
        ('psutil', 'psutil'),
        ('GPUtil', 'GPUtil')
    ]

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_packages = len(core_packages) + len(optional_packages)
    current_step = 0

    # Install core packages first
    for package_name, install_cmd in core_packages:
        status_text.text(f"Installing {package_name}...")
        try:
            if package_name == 'torch_gpu':
                try:
                    import torch
                    if torch.cuda.is_available():
                        status_text.text(f"✅ {package_name} (GPU) already available")
                    else:
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install"
                        ] + install_cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        status_text.text(f"✅ {package_name} installed")
                except ImportError:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install"
                    ] + install_cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    status_text.text(f"✅ {package_name} installed")
            else:
                try:
                    if package_name == 'transformers':
                        __import__('transformers')
                    elif package_name == 'accelerate':
                        __import__('accelerate')
                    elif package_name == 'pillow':
                        __import__('PIL')
                    else:
                        __import__(package_name)
                    status_text.text(f"✅ {package_name} already available")
                except ImportError:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", install_cmd
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    status_text.text(f"✅ {package_name} installed")
        except Exception as e:
            st.error(f"❌ Failed to install {package_name}: {e}")
            st.warning("⚠️ Please install manually with:")
            st.code(f"pip install {install_cmd}")

        current_step += 1
        progress_bar.progress(current_step / total_packages)

    # Install optional packages
    for package_name, install_cmd in optional_packages:
        status_text.text(f"Installing optional {package_name}...")
        try:
            try:
                if package_name == 'GPUtil':
                    __import__('GPUtil')
                else:
                    __import__(package_name)
                status_text.text(f"✅ {package_name} already available")
            except ImportError:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", install_cmd
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                status_text.text(f"✅ {package_name} installed")
        except Exception:
            status_text.text(f"⚠️ {package_name} (optional) - skipped")

        current_step += 1
        progress_bar.progress(current_step / total_packages)

    status_text.text("✅ Package installation complete!")
    return True

def check_initial_setup():
    """Check if core packages are available"""
    try:
        import torch
        import transformers
        from PIL import Image
        return True
    except ImportError:
        return False

# Show installation UI if needed
if not check_initial_setup():
    st.warning("🔧 Setting up GPU-optimized environment...")
    st.info("This is a one-time setup process that may take a few minutes.")

    if st.button("🚀 Install Required Packages", type="primary"):
        with st.spinner("Installing packages..."):
            success = install_gpu_packages()

        if success:
            st.success("✅ Installation complete! Please refresh the page.")
            st.balloons()
        else:
            st.error("❌ Installation failed. Please try manual installation below.")

    # Manual installation instructions
    with st.expander("📖 Manual Installation Instructions"):
        st.markdown("""
        If automatic installation fails, run these commands in your terminal:

        **For GPU (CUDA 11.8):**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install transformers>=4.35.0 accelerate pillow numpy requests
        pip install bitsandbytes  # Optional for memory optimization
        ```

        **For CPU only:**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        pip install transformers>=4.35.0 accelerate pillow numpy requests
        ```

        **Optional monitoring packages:**
        ```bash
        pip install GPUtil psutil matplotlib
        ```
        """)

    st.stop()  # Stop execution until packages are installed

# Import with error handling
try:
    import torch
    import torchvision.transforms as transforms
    HAS_TORCH = True
except ImportError as e:
    HAS_TORCH = False
    st.error(f"PyTorch not available: {e}")

try:
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration,
        InstructBlipProcessor, InstructBlipForConditionalGeneration,
        LlavaProcessor, LlavaForConditionalGeneration,
        BitsAndBytesConfig
    )
    HAS_TRANSFORMERS = True
except ImportError as e:
    HAS_TRANSFORMERS = False
    st.error(f"Transformers not available: {e}")

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    st.error("PIL not available")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import GPUtil
    import psutil
    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False

class GPUOptimizedVLMAnalyzer:
    """GPU-Optimized VLM Analyzer that FORCES GPU utilization"""

    def __init__(self, model_name="blip2", use_quantization=True, max_memory_gb=None, force_cpu=False):
        self.model_name = model_name
        self.use_quantization = use_quantization and not force_cpu
        self.max_memory_gb = max_memory_gb
        self.force_cpu = force_cpu
        self.model = None
        self.processor = None

        # FORCE GPU detection and setup
        if force_cpu:
            self.device = torch.device("cpu")
            st.info("💻 CPU mode forced by user")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # Explicitly use GPU 0
            # CRITICAL: Set default tensor device
            torch.cuda.set_device(0)
            # Enable GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            st.success(f"🚀 GPU mode ACTIVATED: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            st.warning("⚠️ No GPU detected - falling back to CPU")

        # Verify GPU setup
        self._verify_gpu_setup()

        if HAS_TRANSFORMERS:
            self._load_optimized_model()

    def _verify_gpu_setup(self):
        """Verify GPU is properly set up and accessible"""
        if self.device.type == "cuda":
            try:
                # Test GPU accessibility
                test_tensor = torch.randn(10, 10).to(self.device)
                result = torch.matmul(test_tensor, test_tensor)

                # Display GPU verification
                st.success(f"✅ GPU verification passed: {self.device}")
                st.info(f"🎯 Active GPU: {torch.cuda.get_device_name()}")
                st.info(f"🧠 CUDA Version: {torch.version.cuda}")
                st.info(f"⚡ cuDNN Available: {torch.backends.cudnn.is_available()}")

                # Clean up test tensor
                del test_tensor, result
                torch.cuda.empty_cache()

            except Exception as e:
                st.error(f"❌ GPU verification failed: {e}")
                st.warning("🔄 Falling back to CPU mode")
                self.device = torch.device("cpu")
                self.force_cpu = True

    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    @st.cache_resource
    def _load_optimized_model(_self):
        """Load model with ENFORCED GPU utilization and better memory management"""

        # Initialize attributes to None first
        _self.model = None
        _self.processor = None

        try:
            # Clear any existing GPU memory
            if _self.device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            # Use most reliable model configuration
            if _self.model_name.lower() == "blip2":
                model_id = "Salesforce/blip2-opt-2.7b"
                processor_class = BlipProcessor
                model_class = BlipForConditionalGeneration
            elif _self.model_name.lower() == "llava":
                model_id = "llava-hf/llava-1.5-7b-hf"
                processor_class = LlavaProcessor
                model_class = LlavaForConditionalGeneration
                # Force quantization for 7B model
                _self.use_quantization = True
            else:
                # Default to BLIP-2
                model_id = "Salesforce/blip2-opt-2.7b"
                processor_class = BlipProcessor
                model_class = BlipForConditionalGeneration

            st.info(f"🔄 Loading {model_id}...")

            # Load processor first (smaller memory footprint)
            _self.processor = processor_class.from_pretrained(
                model_id,
                cache_dir="./model_cache"
            )
            st.success("✅ Processor loaded")

            # Configure model loading based on device with better memory management
            load_kwargs = {
                "cache_dir": "./model_cache",
                "low_cpu_mem_usage": True,
                "resume_download": True,  # Resume interrupted downloads
            }

            if _self.device.type == "cuda" and not _self.force_cpu:
                # Check available GPU memory first
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                available_memory = gpu_memory - (torch.cuda.memory_allocated() / 1e9)
                st.info(f"📊 Available GPU Memory: {available_memory:.1f}GB")

                # GPU configuration with automatic device mapping
                load_kwargs.update({
                    "device_map": "auto",  # Let transformers handle device placement
                    "torch_dtype": torch.float16,
                    "max_memory": {0: f"{max(1, int(available_memory * 0.8))}GB"}  # Reserve some memory
                })

                # Always use quantization for large models to prevent OOM
                if _self.use_quantization or "7b" in model_id.lower():
                    try:
                        from transformers import BitsAndBytesConfig
                        load_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        st.info("🔧 4-bit quantization enabled (auto-enabled for large models)")
                    except ImportError:
                        st.warning("⚠️ BitsAndBytesConfig not available, using CPU offloading")
                        load_kwargs.update({
                            "device_map": "auto",
                            "offload_folder": "./offload_cache"
                        })

            else:
                # CPU configuration
                load_kwargs.update({
                    "torch_dtype": torch.float32,
                    "device_map": "cpu",
                })

            # Add progress callback for loading
            st.info(f"🧠 Loading model on {_self.device}...")

            # Try loading with gradual memory allocation
            try:
                # Force garbage collection before loading
                gc.collect()
                if _self.device.type == "cuda":
                    torch.cuda.empty_cache()

                # Load with minimal memory footprint first
                minimal_kwargs = load_kwargs.copy()
                minimal_kwargs.update({
                    "device_map": "cpu",  # Start on CPU
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True,
                })

                st.info("🔄 Step 1: Loading model to CPU first...")
                _self.model = model_class.from_pretrained(model_id, **minimal_kwargs)

                # Then move to GPU if needed
                if _self.device.type == "cuda" and not _self.force_cpu:
                    st.info("🔄 Step 2: Moving model to GPU...")
                    _self.model = _self.model.to(_self.device)

            except (torch.cuda.OutOfMemoryError, RuntimeError, OSError) as e:
                st.warning(f"⚠️ Standard loading failed: {str(e)[:100]}...")
                st.info("🔄 Trying with aggressive memory optimization...")

                # Clean up failed attempt
                if hasattr(_self, 'model'):
                    del _self.model
                gc.collect()

                # Ultra-conservative loading
                load_kwargs.update({
                    "device_map": "auto",
                    "offload_folder": "./offload_cache",
                    "offload_state_dict": True,
                    "max_memory": {0: "4GB", "cpu": "16GB"}  # Strict limits
                })
                _self.model = model_class.from_pretrained(model_id, **load_kwargs)

            # Verify model loading and show memory status
            if _self.device.type == "cuda":
                # Check where model components are actually loaded
                devices_used = set()
                for name, param in _self.model.named_parameters():
                    devices_used.add(str(param.device))

                st.success(f"✅ Model loaded across devices: {', '.join(devices_used)}")

                # Show GPU memory usage
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                st.success(f"🎯 GPU Memory Used: {memory_used:.1f}GB / {memory_total:.1f}GB")
            else:
                st.success("✅ Model loaded on CPU")

            _self.model.eval()

            # Clear cache after loading
            if _self.device.type == "cuda":
                torch.cuda.empty_cache()

            st.success(f"✅ Model ready for inference!")
            return True

        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")
            # Clean up any partial loading
            if hasattr(_self, 'model'):
                del _self.model
            if _self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            return _self._load_fallback_model()

    def _load_fallback_model(self):
        """Improved CPU fallback when GPU loading fails"""
        st.warning("🔄 Loading CPU fallback model...")

        # Initialize attributes if they don't exist
        if not hasattr(self, 'model'):
            self.model = None
        if not hasattr(self, 'processor'):
            self.processor = None

        try:
            # Clean up any existing model
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            gc.collect()

            # Force CPU device
            self.device = torch.device("cpu")
            self.force_cpu = True

            # Use smaller model for CPU to avoid memory issues
            model_id = "Salesforce/blip2-opt-2.7b"  # Smaller than 7B models

            # Load processor
            self.processor = BlipProcessor.from_pretrained(
                model_id,
                cache_dir="./model_cache"
            )

            # Load model for CPU with conservative memory settings
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_id,
                cache_dir="./model_cache",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu"
            )

            self.model.eval()

            st.success("✅ CPU fallback model loaded successfully")
            st.info("💻 Running in CPU mode - slower but functional")
            return True

        except Exception as e:
            st.error(f"❌ CPU fallback failed: {e}")
            st.error("💥 All loading methods failed. Try restarting or using a smaller model.")
            return False

    def analyze_uploaded_image(self, uploaded_file, equipment_type: str) -> Dict:
        """GPU-enforced image analysis with monitoring"""

        if self.model is None:
            return self._mock_analysis_streamlit(uploaded_file.name, equipment_type)

        # Pre-analysis setup
        if self.device.type == "cuda":
            initial_memory = torch.cuda.memory_allocated() / 1e9
            st.info(f"🔄 Starting analysis on {self.device} (Memory: {initial_memory:.1f}GB)")

        try:
            # Clear cache
            self.clear_gpu_cache()

            # Load and preprocess image
            image = Image.open(uploaded_file).convert('RGB')

            # Resize for optimal processing
            max_size = 1024 if self.device.type == "cuda" else 512
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Create prompt
            prompt = self._create_analysis_prompt(equipment_type)

            # Run inference
            start_time = datetime.now()

            with torch.inference_mode():
                if self.model_name.lower() == "llava":
                    response = self._run_llava_inference(image, prompt)
                else:  # BLIP-2 or others
                    response = self._run_blip2_inference(image, prompt)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Post-analysis monitoring
            if self.device.type == "cuda":
                final_memory = torch.cuda.memory_allocated() / 1e9
                st.success(f"✅ Analysis completed on GPU in {duration:.2f}s")
                st.info(f"📊 GPU Memory: {initial_memory:.1f}GB → {final_memory:.1f}GB")

            # Clear cache
            self.clear_gpu_cache()

            # Parse results
            result = self._parse_vlm_response(response, uploaded_file.name, equipment_type)

            # Add GPU info to result
            if self.device.type == "cuda":
                result['gpu_utilization'] = {
                    'device_used': str(self.device),
                    'analysis_duration_seconds': duration,
                    'gpu_verified': True
                }

            return result

        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            self.clear_gpu_cache()
            return self._mock_analysis_streamlit(uploaded_file.name, equipment_type)

    def _run_llava_inference(self, image, prompt):
        """LLaVA inference with GPU verification"""

        # Verify model device
        model_device = next(self.model.parameters()).device
        st.info(f"🧠 LLaVA model running on: {model_device}")

        # Prepare conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]

        # Process inputs
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text, return_tensors="pt")

        # Move inputs to device
        if self.device.type == "cuda":
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        # Decode
        generated_ids = outputs[:, inputs['input_ids'].shape[-1]:]
        response = self.processor.decode(generated_ids[0], skip_special_tokens=True)

        return response

    def _run_blip2_inference(self, image, prompt):
        """BLIP-2 inference with GPU verification"""

        # Verify model device
        model_device = next(self.model.parameters()).device
        st.info(f"🧠 BLIP-2 model running on: {model_device}")

        # Process inputs
        inputs = self.processor(image, prompt, return_tensors="pt")

        # Move to device
        if self.device.type == "cuda":
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            do_sample=True,
        )

        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return response

    def _create_analysis_prompt(self, equipment_type: str) -> str:
        """Create analysis prompt"""

        prompts = {
            "insulator": "Analyze this electrical insulator for defects, contamination, cracks, and overall condition.",
            "transformer": "Analyze this transformer for oil leaks, corrosion, damage, and maintenance issues.",
            "steam_leak": "Analyze this equipment for steam leaks, vapor emissions, and moisture problems.",
            "oil_leak": "Analyze this equipment for oil leaks, stains, spills, and contamination.",
            "coal_ash_deposit": "Analyze this equipment for coal ash deposits, buildup, and cleaning needs."
        }

        return prompts.get(equipment_type, f"Analyze this {equipment_type} for any issues or defects.")

    def _parse_vlm_response(self, response: str, filename: str, equipment_type: str) -> Dict:
        """Parse VLM response"""

        # Simple condition detection
        response_lower = response.lower()

        # Check for issues
        issue_keywords = ['crack', 'damage', 'leak', 'contamination', 'problem', 'defect', 'fault']
        issues_found = [word for word in issue_keywords if word in response_lower]

        if issues_found:
            condition = "DEFECTIVE"
            anomaly_detected = True
            priority = "HIGH" if len(issues_found) > 2 else "MEDIUM"
        else:
            condition = "NORMAL"
            anomaly_detected = False
            priority = "LOW"

        return {
            'image_name': filename,
            'equipment_type': equipment_type.replace('_', ' ').title(),
            'description': response,
            'condition': condition,
            'confidence': 0.85,
            'anomaly_detected': anomaly_detected,
            'specific_issues': issues_found,
            'maintenance_priority': priority,
            'recommendations': [f"Inspect {equipment_type.replace('_', ' ')} based on analysis"],
            'safety_concerns': ["Professional inspection recommended"] if anomaly_detected else [],
            'analysis_method': f'GPU {self.model_name.upper()}' if self.device.type == "cuda" else f'CPU {self.model_name.upper()}',
            'model_device': str(self.device),
            'timestamp': datetime.now().isoformat()
        }

    def _mock_analysis_streamlit(self, filename: str, equipment_type: str) -> Dict:
        """Mock analysis when model fails"""

        return {
            'image_name': filename,
            'equipment_type': equipment_type.replace('_', ' ').title(),
            'description': f"Analysis unavailable - model loading failed. Equipment appears to need inspection.",
            'condition': 'UNKNOWN',
            'confidence': 0.50,
            'anomaly_detected': True,
            'specific_issues': ['Model loading failed'],
            'maintenance_priority': 'MEDIUM',
            'recommendations': ['Load VLM model for accurate analysis'],
            'safety_concerns': ['Professional inspection recommended'],
            'analysis_method': 'Mock Analysis (Model Failed)',
            'model_device': str(self.device),
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Main Streamlit application"""

    # Header
    st.title("🚀 GPU-Powered VLM Equipment Analyzer")
    st.markdown("**High-Performance Local Vision Language Models**")

    # GPU Status
    if torch.cuda.is_available():
        st.success(f"🚀 GPU Available: {torch.cuda.get_device_name()}")

        # GPU memory info
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        memory_used = torch.cuda.memory_allocated() / 1e9
        st.info(f"💾 GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB used")
    else:
        st.warning("⚠️ No GPU detected - using CPU mode")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model selection
        model_choice = st.selectbox(
            "Select Model",
            ["llava", "blip2"],
            help="BLIP-2: Fast and reliable, LLaVA: Advanced reasoning"
        )

        # Equipment type
        equipment_type = st.selectbox(
            "Equipment Type",
            ["steam_leak", "oil_leak", "coal_ash_deposit", "insulator", "transformer"]
        )

        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            use_quantization = st.checkbox("4-bit Quantization", value=True)
            force_cpu = st.checkbox("Force CPU Mode", value=False)

        # System info
        st.header("📊 System Info")
        if torch.cuda.is_available():
            st.success(f"✅ CUDA {torch.version.cuda}")
        st.info(f"🐍 Python {sys.version[:5]}")
        st.info(f"🔥 PyTorch {torch.__version__}")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📸 Upload Image")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['png', 'jpg', 'jpeg']
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_container_width=True)

            if st.button("🔍 Analyze Equipment", type="primary"):
                with st.spinner("Analyzing..."):
                    # Initialize analyzer
                    if 'analyzer' not in st.session_state:
                        st.session_state.analyzer = GPUOptimizedVLMAnalyzer(
                            model_choice, use_quantization, None, force_cpu
                        )

                    # Run analysis
                    result = st.session_state.analyzer.analyze_uploaded_image(uploaded_file, equipment_type)
                    st.session_state.result = result

    with col2:
        st.header("🧠 Analysis Results")

        if 'result' in st.session_state:
            result = st.session_state.result

            # Status
            if result['anomaly_detected']:
                st.error(f"🚨 {result['condition']}")
            else:
                st.success(f"✅ {result['condition']}")

            # Metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Confidence", f"{result['confidence']*100:.0f}%")
            with col_b:
                st.metric("Priority", result['maintenance_priority'])
            with col_c:
                st.metric("Device", result['model_device'])

            # Description
            st.write("**Analysis:**")
            st.write(result['description'])

            # Issues
            if result['specific_issues']:
                st.write("**Issues:**")
                for issue in result['specific_issues']:
                    st.write(f"• {issue}")

            # GPU utilization info
            if 'gpu_utilization' in result:
                gpu_util = result['gpu_utilization']
                st.success(f"✅ Analyzed on GPU in {gpu_util['analysis_duration_seconds']:.2f}s")

            # Download
            result_json = json.dumps(result, indent=2)
            st.download_button(
                "📥 Download Report",
                data=result_json,
                file_name=f"analysis_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        else:
            st.info("Upload an image and click 'Analyze Equipment'")

if __name__ == "__main__":
    if HAS_TORCH and HAS_TRANSFORMERS and HAS_PIL:
        main()
    else:
        st.error("❌ Required packages missing")
        st.code("pip install torch torchvision transformers pillow --index-url https://download.pytorch.org/whl/cu118")