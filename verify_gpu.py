#!/usr/bin/env python3
"""
GPU Verification Script for Soccer Analysis
Checks GPU availability and configuration for optimal performance.
"""

import sys
import torch

def check_gpu():
    """Check GPU availability and print configuration."""
    print("=" * 60)
    print("GPU Configuration Check for Soccer Analysis")
    print("=" * 60)
    print()
    
    # PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print()
        
        # GPU device info
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        print()
        
        for i in range(num_gpus):
            print(f"GPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multiprocessors: {props.multi_processor_count}")
            print()
        
        # Test GPU computation
        print("Testing GPU computation...")
        try:
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.matmul(x, y)
            print("✅ GPU computation test: PASSED")
            print()
        except Exception as e:
            print(f"❌ GPU computation test: FAILED - {e}")
            print()
    else:
        print("⚠️  CUDA not available - will use CPU")
        print("   This will significantly slow down processing.")
        print()
    
    # Check other dependencies
    print("Checking other dependencies...")
    try:
        from ultralytics import YOLO
        print("✅ ultralytics: OK")
    except ImportError:
        print("❌ ultralytics: NOT INSTALLED")
    
    try:
        import supervision as sv
        print("✅ supervision: OK")
    except ImportError:
        print("❌ supervision: NOT INSTALLED")
    
    try:
        import transformers
        print("✅ transformers: OK")
    except ImportError:
        print("❌ transformers: NOT INSTALLED")
    
    try:
        import umap  # type: ignore
        print("✅ umap-learn: OK")
    except ImportError:
        print("❌ umap-learn: NOT INSTALLED")
    
    print()
    print("=" * 60)
    
    if torch.cuda.is_available():
        print("✅ GPU is ready for use!")
        print("   Models will automatically use GPU for inference.")
    else:
        print("⚠️  GPU not available - processing will be slower.")
    
    print("=" * 60)

if __name__ == "__main__":
    check_gpu()

