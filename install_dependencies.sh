#!/bin/bash
# Installation script for Soccer Analysis with GPU support

set -e

echo "=========================================="
echo "Soccer Analysis - GPU-Optimized Setup"
echo "=========================================="
echo ""

# Check for CUDA availability
echo "Checking for CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected!"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
    
    # Detect CUDA version
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    if [ -z "$CUDA_VERSION" ]; then
        CUDA_VERSION="12.1"  # Default to CUDA 12.1
    fi
    echo "Detected CUDA version: $CUDA_VERSION"
    echo ""
else
    echo "⚠️  NVIDIA GPU not detected. Will install CPU-only PyTorch."
    CUDA_VERSION="cpu"
fi

# Install PyTorch with CUDA support
echo "Installing PyTorch with GPU support..."
if [ "$CUDA_VERSION" != "cpu" ]; then
    # Install PyTorch with CUDA 12.1 (works with most modern GPUs)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    # Install CPU-only PyTorch
    pip install torch torchvision torchaudio
fi

echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

# Verify PyTorch installation
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('⚠️  CUDA not available - will use CPU')
"

echo ""
echo "=========================================="
echo "✅ Installation complete!"
echo "=========================================="
echo ""
echo "To verify GPU usage, run:"
echo "  python3 -c \"import torch; print('GPU available:', torch.cuda.is_available())\""
echo ""

