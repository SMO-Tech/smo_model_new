"""Core Player Detection Functions for Soccer Analysis.

This module provides core functionality for detecting players and referees
using YOLO models. Ball detection has been removed.
Pipeline functions have been moved to detection_pipeline.py.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional, List

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from ultralytics import YOLO
import numpy as np
import supervision as sv
import torch

# Import GPU settings from constants
try:
    from constants import USE_GPU, GPU_DEVICE
except ImportError:
    USE_GPU = True
    GPU_DEVICE = 0

# ================================================
# Core Detection Functions
# ================================================

def get_device():
    """Get the appropriate device for inference."""
    if USE_GPU and torch.cuda.is_available():
        try:
            # Check if GPU is compatible by testing a simple operation
            device = GPU_DEVICE if isinstance(GPU_DEVICE, int) else 0
            test_tensor = torch.zeros(1).to(f'cuda:{device}')
            _ = test_tensor * 2  # Simple operation to test compatibility
            return device
        except RuntimeError as e:
            if "no kernel image" in str(e) or "CUDA capability" in str(e):
                print(f"⚠️  GPU not compatible with current PyTorch version. Falling back to CPU.")
                print(f"   Error: {e}")
                return 'cpu'
            raise
    return 'cpu'

def load_detection_model(model_path: str) -> YOLO:
    """Load and return a YOLO detection model with GPU optimization.
    
    Args:
        model_path: Path to the YOLO model file
        
    Returns:
        YOLO model instance configured for GPU or CPU
    """
    device = get_device()
    # YOLO will automatically use the device, but we can specify it
    if device == 'cpu':
        # Force CPU to avoid CUDA compatibility issues
        model = YOLO(model_path)
        model.to('cpu')
    else:
        try:
            model = YOLO(model_path)
            model.to(device)
        except RuntimeError as e:
            if "no kernel image" in str(e) or "CUDA capability" in str(e):
                print(f"⚠️  GPU incompatible, using CPU instead")
                model = YOLO(model_path)
                model.to('cpu')
            else:
                raise
    return model


def detect_objects_in_frames(model: YOLO, frames, device: str = None) -> List:
    """Detect objects in video frames using YOLO model with GPU acceleration.
    
    Args:
        model: Loaded YOLO model
        frames: Video frames or single frame
        device: Device to use for inference (auto-detected if None)
        
    Returns:
        Detection results from YOLO model
    """
    if device is None:
        device = get_device()
    
    # Use device parameter for inference with error handling
    try:
        if device != 'cpu':
            return model(frames, device=device)
        else:
            return model(frames, device='cpu')
    except RuntimeError as e:
        if "no kernel image" in str(e) or "CUDA capability" in str(e):
            print(f"⚠️  GPU operation failed, retrying with CPU")
            return model(frames, device='cpu')
        raise

def get_detections(detection_model: YOLO, frame: np.ndarray, use_slicer: bool = False) -> Tuple[sv.Detections, sv.Detections]:
    """Get separated detections for players and referees with GPU acceleration.
    
    Note: Ball detection has been removed. This function only returns players and referees.
    
    Args:
        detection_model: Loaded YOLO model
        frame: Input frame as numpy array
        use_slicer: Whether to use inference slicer for large images
        
    Returns:
        Tuple of (player_detections, referee_detections)
    """
    device = get_device()
    
    def inference_callback(frame: np.ndarray) -> sv.Detections:
        """Convert YOLO results to supervision format."""
        result = detect_objects_in_frames(detection_model, frame, device=device)[0]
        return sv.Detections.from_ultralytics(result)

    # Get detections using slicer or direct inference
    if use_slicer:
        slicer = sv.InferenceSlicer(callback=inference_callback)
        detections = slicer(frame)
    else:
        detections = inference_callback(frame)

    # Separate detections by class (skip ball class_id == 1)
    player_detections = detections[detections.class_id == 0]
    referee_detections = detections[detections.class_id == 2]

    return player_detections, referee_detections