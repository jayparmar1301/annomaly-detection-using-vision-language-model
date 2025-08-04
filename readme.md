# 🚀 GPU-Optimized VLM Equipment Analyzer

A high-performance Streamlit application that uses Vision Language Models (VLMs) to analyze power equipment for defects, maintenance issues, and safety concerns. Built for GPU acceleration with automatic fallbacks and memory optimization.
<img width="1857" height="805" alt="image (6)" src="https://github.com/user-attachments/assets/25d47049-b6f8-484c-a056-4313cc78a697" />

## ✨ Features

- **GPU-Accelerated Analysis**: Leverages NVIDIA GPUs for fast inference
- **Multiple VLM Models**: Supports BLIP-2 and LLaVA models
- **Equipment-Specific Analysis**: Specialized prompts for different equipment types
- **Automatic Memory Management**: 4-bit quantization and smart memory allocation
- **Real-time GPU Monitoring**: Live memory usage and performance metrics
- **Professional Reports**: Downloadable JSON analysis reports
- **Robust Fallbacks**: Automatic CPU fallback when GPU unavailable

## 🔧 Supported Equipment Types

- **Electrical Insulators**: Crack detection, contamination analysis
- **Transformers**: Oil leak detection, corrosion assessment
- **Steam Systems**: Steam leak identification, vapor analysis
- **Oil Systems**: Oil leak detection, spill assessment  
- **Coal Ash Equipment**: Deposit analysis, cleaning recommendations

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Ubuntu 18.04+ / Windows 10+ / macOS 10.15+
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 10GB+ free space
- **Python**: 3.8+

### GPU Requirements (Optional but Recommended)
- **NVIDIA GPU**: GTX 1060+ or better
- **VRAM**: 6GB+ (24GB recommended for LLaVA 7B)
- **CUDA**: 11.8+ 
- **Drivers**: Latest NVIDIA drivers

### Tested Configurations
- ✅ AWS EC2 g4dn instances (NVIDIA T4)
- ✅ AWS EC2 g5 instances (NVIDIA A10G) 
- ✅ Local RTX 3080/4080/4090
- ✅ CPU-only fallback mode

## 📦 Installation

### Quick Start (Automatic)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd annomaly-detection-using-vision-language-model
   ```

2. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```
   
   The app will automatically detect missing packages and guide you through installation.

### Manual Installation

#### For GPU (CUDA 11.8)
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install transformers>=4.35.0 accelerate pillow numpy requests streamlit

# Install optional packages for optimization
pip install bitsandbytes  # For 4-bit quantization
pip install GPUtil psutil matplotlib  # For monitoring
```

#### For CPU Only
```bash
# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install transformers>=4.35.0 accelerate pillow numpy requests streamlit
```

## 🚀 Usage

### Starting the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

### Basic Workflow

1. **Select Model**: Choose between BLIP-2 (fast) or LLaVA (advanced)
2. **Choose Equipment Type**: Select the type of equipment you're analyzing
3. **Upload Image**: Upload a JPG/PNG image of the equipment
4. **Analyze**: Click "Analyze Equipment" to run the analysis
5. **Review Results**: View the analysis results and download the report

### Advanced Settings

- **4-bit Quantization**: Reduces memory usage by ~75%
- **Force CPU Mode**: Override GPU detection for testing
- **Memory Limits**: Set custom GPU memory limits

## 🔍 Model Information

### BLIP-2 (Default)
- **Model**: `Salesforce/blip2-opt-2.7b`
- **Size**: ~5-6GB
- **Speed**: Fast inference
- **Best for**: Quick analysis, production use

### LLaVA 
- **Model**: `llava-hf/llava-1.5-7b-hf`
- **Size**: ~14GB (7GB with quantization)
- **Speed**: Slower but more detailed
- **Best for**: Detailed analysis, research

## 🛠️ Configuration

### Environment Variables
```bash
# Optional: Optimize memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TRANSFORMERS_CACHE=./model_cache
export HF_HOME=./model_cache
```

### GPU Setup (AWS EC2)

For AWS EC2 instances with NVIDIA GPUs:

```bash
# Install NVIDIA drivers
sudo apt update
sudo apt install -y nvidia-driver-470-server
sudo reboot

# Verify installation
nvidia-smi
```

## 📊 Performance Benchmarks

### GPU Performance (NVIDIA A10G)
- **BLIP-2**: ~2-3 seconds per image
- **LLaVA**: ~5-8 seconds per image
- **Memory Usage**: 4-8GB VRAM (with quantization)

### CPU Performance 
- **BLIP-2**: ~15-30 seconds per image
- **Memory Usage**: 8-12GB RAM

## 🔧 Troubleshooting

### Common Issues

#### "Killed" Error During Model Loading
```bash
# Add swap space (temporary fix)
sudo fallocate -l 8G /tmp/swapfile
sudo chmod 600 /tmp/swapfile
sudo mkswap /tmp/swapfile
sudo swapon /tmp/swapfile

# Remove after use
sudo swapoff /tmp/swapfile
sudo rm /tmp/swapfile
```

#### CUDA Out of Memory
- Enable 4-bit quantization in settings
- Use smaller model (BLIP-2 instead of LLaVA)
- Reduce image size (app auto-resizes to 1024px)

#### GPU Not Detected
```bash
# Check GPU availability
nvidia-smi

# Test PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

#### Model Download Issues
```bash
# Clear cache and retry
rm -rf ./model_cache
# Restart the application
```

### Performance Optimization

#### For Low Memory Systems
1. Use BLIP-2 model instead of LLaVA
2. Enable 4-bit quantization
3. Set force_cpu=True if necessary

#### For High Performance
1. Use LLaVA model for detailed analysis
2. Disable quantization on high-VRAM GPUs
3. Increase batch processing (future feature)

## 📁 Project Structure

```
vlm-equipment-analyzer/
├── streamlit_app.py              # Main Streamlit application
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── model_cache/           # Downloaded models (auto-created)
├── offload_cache/         # CPU offloading cache
└── examples/              # Sample images for testing
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Salesforce**: BLIP-2 model
- **LLaVA Team**: LLaVA vision-language model
- **Hugging Face**: Transformers library and model hosting
- **Streamlit**: Web application framework

## 📞 Support

### Getting Help
- 📖 Check this README for common solutions
- 🐛 Report bugs via GitHub Issues
- 💬 Ask questions in GitHub Discussions

### Hardware Recommendations

#### Budget Setup ($500-1000)
- NVIDIA GTX 1660 Ti (6GB VRAM)
- 16GB RAM
- Use BLIP-2 with quantization

#### Professional Setup ($2000-5000)
- NVIDIA RTX 3080/4080 (12-16GB VRAM)
- 32GB RAM
- Run LLaVA without quantization

#### Enterprise Setup ($5000+)
- NVIDIA A100/H100 (40-80GB VRAM)
- 64GB+ RAM
- Multiple models simultaneously

---

**Made with ❤️ for industrial equipment maintenance and safety**
