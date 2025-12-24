# GPU Setup and Optimization Guide

This document describes the GPU optimization setup for the Soccer Analysis project.

## ✅ Installation Complete

All dependencies have been installed with GPU support. The system is configured to maximize GPU utilization.

## 🚀 GPU Configuration

### Current Setup
- **GPU**: NVIDIA GeForce RTX 5090
- **GPU Memory**: 33.68 GB
- **CUDA Version**: 12.8
- **PyTorch**: 2.9.1+cu128 (with full RTX 5090 sm_120 support)
- **Status**: ✅ GPU is fully compatible and ready

### GPU Usage

The project has been optimized to use GPU in the following components:

1. **YOLO Models (Detection & Keypoint)**
   - Models automatically use GPU device from `constants.py`
   - Device is set via `USE_GPU` and `GPU_DEVICE` settings
   - Inference runs on GPU by default

2. **SigLIP Embeddings (Team Assignment)**
   - Already configured to use GPU via `torch.cuda`
   - Batch processing optimized for GPU memory

3. **Transformers Models**
   - All transformer models (SigLIP) use GPU automatically

## 📋 Configuration

### GPU Settings in `constants.py`

```python
USE_GPU = True                    # Set to False to force CPU usage
GPU_DEVICE = 0                    # GPU device index (if multiple GPUs)
```

### Model Loading

Models are automatically loaded with GPU support:
- Detection models: `player_detection/detect_players.py`
- Keypoint models: `keypoint_detection/detect_keypoints.py`
- Embedding models: `player_clustering/embeddings.py`

## 🔍 Verification

Run the GPU verification script to check your setup:

```bash
python3 verify_gpu.py
```

This will show:
- GPU availability and details
- CUDA version
- Memory information
- Test GPU computation
- Dependency status

## ⚡ Performance Optimization

### Batch Processing
- Embedding extraction uses batch size of 24 (configurable in `constants.py`)
- YOLO models process frames efficiently on GPU
- All tensor operations run on GPU

### Memory Management
- Models are loaded once and reused
- Batch processing minimizes GPU memory usage
- Automatic device selection (GPU if available, CPU otherwise)

## 🛠️ Troubleshooting

### GPU Not Detected
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify CUDA installation: `nvcc --version`
3. Reinstall PyTorch with CUDA: See `install_dependencies.sh`

### CUDA Capability Warning
If you see a warning about CUDA capability (like sm_120), the GPU will still work but may not use all optimizations. This is normal for very new GPUs.

### NumPy Version Conflict
There's a known compatibility issue between NumPy 2.x and some packages. The project uses NumPy 1.x for compatibility. If you see warnings, they can be safely ignored.

## 📦 Dependencies

All dependencies are listed in `requirements.txt`:
- PyTorch 2.9.1+ with CUDA 12.8 (supports RTX 5090 sm_120)
- Ultralytics (YOLO)
- Supervision
- Transformers
- UMAP-learn
- scikit-learn
- And more...

**Note**: For RTX 5090, install PyTorch with CUDA 12.8:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
```

Or use the provided script:
```bash
./install_gpu_pytorch.sh
```

## 🎯 Usage

The system automatically uses GPU when available. No additional configuration needed!

Models will:
- Load on GPU automatically
- Process inference on GPU
- Use batch processing for efficiency

## 📝 Notes

- GPU is used automatically - no code changes needed
- CPU fallback is available if GPU is not detected
- All models support both GPU and CPU execution
- Performance is significantly better on GPU (10-100x faster)

