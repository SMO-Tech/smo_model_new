#!/bin/bash
# Install PyTorch with RTX 5090 (sm_120) support

set -e

echo "=========================================="
echo "Installing PyTorch for RTX 5090 Support"
echo "=========================================="
echo ""

echo "Installing PyTorch 2.9.1+ with CUDA 12.8 (supports sm_120)..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

echo ""
echo "Verifying installation..."
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test conv2d (what models use)
    try:
        conv = torch.nn.Conv2d(3, 64, 3).cuda()
        test_input = torch.randn(1, 3, 224, 224).cuda()
        output = conv(test_input)
        del conv, test_input, output
        torch.cuda.empty_cache()
        print("✅ GPU fully compatible! All operations work.")
    except RuntimeError as e:
        print(f"❌ GPU test failed: {e}")
        exit(1)
else:
    print("❌ CUDA not available")
    exit(1)
EOF

echo ""
echo "=========================================="
echo "✅ PyTorch installed with RTX 5090 support!"
echo "=========================================="
echo ""
echo "GPU will be automatically used. Run: python3 main.py"

