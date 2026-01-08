"""Core Player Detection Functions for Soccer Analysis.

This module provides core functionality for detecting players, ball, and referees
using YOLO models. Pipeline functions have been moved to detection_pipeline.py.
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

def get_detections(detection_model: YOLO, frame: np.ndarray, use_slicer: bool = False,
                   tracknet_detector=None) -> Tuple[sv.Detections, sv.Detections, sv.Detections]:
    """Get separated detections for players, ball, and referees with GPU acceleration.
    
    Args:
        detection_model: Loaded YOLO model
        frame: Input frame as numpy array
        use_slicer: Whether to use inference slicer for large images
        tracknet_detector: Optional TrackNet detector for ball detection (if None, uses YOLO)
        
    Returns:
        Tuple of (player_detections, ball_detections, referee_detections)
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

    # Separate detections by class
    player_detections = detections[detections.class_id == 0]
    referee_detections = detections[detections.class_id == 2]
    
    # Ball detection: Use TrackNet if provided, otherwise use YOLO
    if tracknet_detector is not None:
        # Use TrackNet for ball detection
        ball_position = tracknet_detector.detect_ball(frame)
        if ball_position is not None and len(ball_position) >= 2:
            # Convert TrackNet position to supervision Detections format
            # Create a visible bounding box around the detected position
            x, y = float(ball_position[0]), float(ball_position[1])
            frame_h, frame_w = frame.shape[0], frame.shape[1]
            
            # Ensure coordinates are within frame bounds
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            
            # Create a larger bounding box for better visibility (like YOLO would)
            box_size = 40  # Larger box for better visibility
            x1 = max(0, x - box_size)
            y1 = max(0, y - box_size)
            x2 = min(frame_w, x + box_size)
            y2 = min(frame_h, y + box_size)
            
            # Only create detection if box is valid
            if x2 > x1 and y2 > y1 and (x2 - x1) >= 10 and (y2 - y1) >= 10:
                xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                ball_detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=np.array([0.95]),  # High confidence for TrackNet
                    class_id=np.array([1])  # Ball class ID
                )
            else:
                ball_detections = sv.Detections.empty()
        else:
            # No ball detected
            ball_detections = sv.Detections.empty()
    else:
        # Use YOLO for ball detection (original behavior)
        ball_detections = detections[detections.class_id == 1]
        
        # Conservative filtering for lower quality videos:
        # 1. Filter by confidence (only accept reasonably confident detections)
        # 2. Filter by size (balls should be small, not too large)
        if len(ball_detections.xyxy) > 0:
            frame_h, frame_w = frame.shape[0], frame.shape[1]
            frame_area = frame_w * frame_h
            
            # Filter detections
            valid_indices = []
            for i in range(len(ball_detections.xyxy)):
                bbox = ball_detections.xyxy[i]
                conf = ball_detections.confidence[i] if ball_detections.confidence is not None and len(ball_detections.confidence) > i else 0.5
                
                # Calculate bounding box size
                bbox_w = bbox[2] - bbox[0]
                bbox_h = bbox[3] - bbox[1]
                bbox_area = bbox_w * bbox_h
                bbox_area_ratio = bbox_area / frame_area
                
                # Conservative filters:
                # 1. Minimum confidence: 0.3 (reject very low confidence detections)
                # 2. Maximum size: 2% of frame area (reject very large detections - likely false positives)
                # 3. Minimum size: 0.01% of frame area (reject extremely tiny detections)
                if conf >= 0.3 and 0.0001 <= bbox_area_ratio <= 0.02:
                    valid_indices.append(i)
            
            # Keep only valid detections
            if len(valid_indices) > 0:
                ball_detections = ball_detections[valid_indices]
            else:
                ball_detections = sv.Detections.empty()

    return player_detections, ball_detections, referee_detections