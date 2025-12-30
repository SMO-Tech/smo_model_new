"""
Global Configuration Constants for Soccer Analysis Project

This file contains the main configuration parameters for the soccer analysis system.
Update these paths according to your setup before running the system.

Model Download Instructions:
1. Visit: https://huggingface.co/Adit-jain/soccana
2. Download the trained model file
3. Place it in the path specified by model_path below
4. Update video paths to point to your test videos

For automatic model download, run:
    pip install huggingface_hub
    python -c "
    from huggingface_hub import hf_hub_download
    import os, shutil
    model_file = hf_hub_download(repo_id='Adit-jain/soccana', filename='best.pt')
    os.makedirs('Models/Trained/yolov11_sahi_1280/First/weights', exist_ok=True)
    shutil.copy(model_file, 'Models/Trained/yolov11_sahi_1280/First/weights/best.pt')
    print('Model downloaded successfully!')
    "
"""

import sys
from pathlib import Path

# Project root directory
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Path to the trained YOLO model
# Download from: https://huggingface.co/Adit-jain/soccana
model_path = r"Models/Trained/yolov11_sahi_1280/Model/weights/best.pt"
model_path = PROJECT_DIR / model_path

# Alternative model paths (uncomment if using different models)
# model_path = PROJECT_DIR / "Models/Pretrained/yolo11n.pt"  # Base YOLO model
# model_path = PROJECT_DIR / "path/to/your/custom/model.pt"   # Custom model

# =============================================================================
# VIDEO CONFIGURATION
# =============================================================================

# Input test video path
# UPDATE THIS: Point to your actual test video file
test_video = str(PROJECT_DIR / "videos/youtube_video.mp4")

# Output video path
# UPDATE THIS: Where you want the tracked video to be saved
test_video_output = r"F:\Datasets\SoccerNet\Data\Samples\output2.mp4"

# Alternative video paths (examples)
# test_video = PROJECT_DIR / "test_videos/sample.mp4"
# test_video_output = PROJECT_DIR / "output/tracked_sample.mp4"

# For webcam input (use with real-time detection)
# webcam_index = 0  # Usually 0 for default webcam

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# Team assignment training parameters
TRAINING_FRAME_STRIDE = 12        # Skip frames during training data collection
TRAINING_FRAME_LIMIT = 120 * 24   # Maximum frames for training (120*24 = ~2 mins at 24fps)

# Clustering parameters  

EMBEDDING_BATCH_SIZE = 256        # Batch size for SigLIP embedding extraction (optimized for Tesla T4 15GB)
UMAP_COMPONENTS = 3               # UMAP dimensionality reduction components
N_TEAMS = 2                       # Number of teams to cluster (usually 2)

# Tracking parameters
TRACKER_MATCH_THRESH = 0.5        # ByteTrack matching threshold
TRACKER_BUFFER_SIZE = 120         # Number of frames to keep in tracking buffer

# Pass Detection Configuration
PASS_DETECTION_ENABLED = True     # Enable player-only pass detection
PASS_MIN_CONFIDENCE = 0.5         # Minimum confidence to accept a pass
PASS_FADE_FRAMES = 90              # Frames to keep pass lines visible (3 seconds at 30fps)

# =============================================================================
# DETECTION CLASSES
# =============================================================================

CLASS_NAMES = {
    0: "Player",
    1: "Ball", 
    2: "Referee"
}

# Class colors for visualization (BGR format)
CLASS_COLORS = {
    0: (0, 255, 0),    # Green for players
    1: (0, 0, 255),    # Red for ball
    2: (255, 0, 0)     # Blue for referees
}

# =============================================================================
# CUDA LIBRARY PATH SETUP (MUST BE BEFORE TORCH IMPORT)
# =============================================================================

# Setup CUDA library paths BEFORE importing torch
# This ensures PyTorch can find CUDA libraries
import os

# Disable problematic libraries that require CUDA 12.x but we have CUDA 11.x
# Set environment variables before importing torch
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'
nvidia_base = Path.home() / ".local" / "lib" / "python3.10" / "site-packages" / "nvidia"
if not nvidia_base.exists():
    # Try alternative Python version path
    import sys as sys_module
    py_version = f"{sys_module.version_info.major}.{sys_module.version_info.minor}"
    nvidia_base = Path.home() / ".local" / "lib" / f"python{py_version}" / "site-packages" / "nvidia"

if nvidia_base.exists():
    cuda_lib_paths = []
    # Prioritize CUDA 12.x libraries, exclude CUDA 11.x cusolver
    cu12_paths = []
    cu11_cusolver_paths = []
    other_paths = []
    
    for lib_dir in nvidia_base.rglob("lib"):
        if lib_dir.is_dir():
            lib_path_str = str(lib_dir)
            # Check if this is CUDA 12.x (look for cu12 in parent path)
            parent_path = str(lib_dir.parent)
            if 'cu12' in parent_path.lower() or 'cusolver_cu12' in parent_path.lower():
                cu12_paths.append(lib_path_str)
            elif 'cusolver' in lib_path_str and 'cu11' in parent_path.lower():
                # Exclude CUDA 11.x cusolver
                cu11_cusolver_paths.append(lib_path_str)
            else:
                other_paths.append(lib_path_str)
    
    # Prioritize CUDA 12.x, then others (excluding cu11 cusolver)
    cuda_lib_paths = cu12_paths + other_paths
    
    if cuda_lib_paths:
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld_path = ":".join(cuda_lib_paths)
        os.environ["LD_LIBRARY_PATH"] = f"{new_ld_path}:{current_ld_path}" if current_ld_path else new_ld_path
        print("✅ CUDA library path configured (CUDA 12.x prioritized):")
        for p in cuda_lib_paths[:5]:  # Show first 5
            print(f"   {p}")
        if len(cuda_lib_paths) > 5:
            print(f"   ... and {len(cuda_lib_paths) - 5} more")
    else:
        print("⚠️  Could not find NVIDIA CUDA libraries in expected locations.")
else:
    print("⚠️  NVIDIA site-packages directory not found.")

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

# GPU settings
USE_GPU = True                    # Set to False to force CPU usage
GPU_DEVICE = 0                    # GPU device index (if multiple GPUs)

# Auto-detect GPU compatibility and fallback to CPU if needed
# Note: RTX 5090 (sm_120) requires PyTorch 2.7.0+ for full support
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)  # Ignore sm_120 warning
    try:
        import torch
        # Ensure CUDA is available, raise error if not
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GPU is required.")
        
        # Test GPU immediately
        device = f'cuda:{GPU_DEVICE}'
        test_tensor = torch.zeros(1).to(device)
        _ = test_tensor * 2
        del test_tensor
        torch.cuda.empty_cache()
        
        # Disable cuDNN to avoid version mismatch issues
        torch.backends.cudnn.enabled = False
        
        # Test GPU with conv2d operation (used by models) to verify real compatibility
        try:
            conv = torch.nn.Conv2d(3, 64, 3).cuda()
            test_input = torch.randn(1, 3, 224, 224).cuda()
            output = conv(test_input)  # This will fail if sm_120 not supported
            del conv, test_input, output
            torch.cuda.empty_cache()
            USE_GPU = True
            print("✅ GPU is fully compatible and ready to use!")
            print(f"✅ GPU Device: {torch.cuda.get_device_name(GPU_DEVICE)}")
        except RuntimeError as conv_error:
            error_msg = str(conv_error).lower()
            if "no kernel image" in error_msg or "cuda capability" in error_msg:
                print(f"⚠️  GPU compatibility warning: {conv_error}")
                print("   Continuing with GPU anyway - GPU is required.")
            USE_GPU = True  # Force GPU - no CPU fallback
            print(f"✅ GPU Device: {torch.cuda.get_device_name(GPU_DEVICE)}")
    except (RuntimeError, Exception) as e:
        error_str = str(e)
        print(f"❌ Fatal GPU error: {error_str}")
        raise RuntimeError(f"GPU is required but failed to initialize: {error_str}")

# Processing settings
MAX_VIDEO_FRAMES = -1             # Max frames to process (-1 for all frames)
OUTPUT_FPS = 30                   # Output video FPS

# Memory optimization
ENABLE_SAHI = False               # Enable SAHI for large image inference
SAHI_SLICE_HEIGHT = 640           # SAHI slice height
SAHI_SLICE_WIDTH = 640            # SAHI slice width
SAHI_OVERLAP_HEIGHT = 0.2         # SAHI overlap ratio
SAHI_OVERLAP_WIDTH = 0.2          # SAHI overlap ratio

# =============================================================================
# VALIDATION & DEBUGGING
# =============================================================================

# Validate paths on import
def validate_config():
    """Validate configuration and provide helpful messages."""
    issues = []
    
    # Check model path
    if not model_path.exists():
        issues.append(f"Model not found at: {model_path}")
        issues.append("Download from: https://huggingface.co/Adit-jain/soccana")
    
    return issues

# Print configuration status
if __name__ == "__main__":
    print("=== Soccer Analysis Configuration ===")
    print(f"Project Directory: {PROJECT_DIR}")
    print(f"Model Path: {model_path}")
    print(f"Test Video: {test_video}")
    print(f"Output Path: {test_video_output}")
    print()
    
    issues = validate_config()
    if issues:
        print("⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease fix these issues before running the system.")
    else:
        print("✅ Configuration looks good!")
        print("\nRun 'python main.py' to start the complete pipeline.")