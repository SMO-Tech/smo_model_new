"""
TrackNet Ball Detector

Uses TrackNetV2/V3 model to detect ball positions using 3 consecutive frames.
TrackNet is specifically designed for tracking small, fast-moving objects like balls
and outputs a heatmap indicating ball location probability.

Key features:
- Processes 3 consecutive frames (t-1, t, t+1) to understand motion
- Outputs heatmap instead of bounding boxes
- Better for small, fast objects than YOLO
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from collections import deque

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

try:
    from constants import USE_GPU, GPU_DEVICE
except ImportError:
    USE_GPU = True
    GPU_DEVICE = 0


class TrackNetModel(nn.Module):
    """
    TrackNet model architecture for ball tracking.
    Based on TrackNetV2/V3 architecture.
    """
    
    def __init__(self, input_channels=9, input_size=(640, 360)):  # 3 frames * 3 RGB channels = 9
        super(TrackNetModel, self).__init__()
        self.input_size = input_size  # Store for final upsampling
        
        # Encoder (downsampling)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 640x360 -> 320x180
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 320x180 -> 160x90
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 160x90 -> 80x45
        
        # Decoder (upsampling) - Fixed to match sizes properly
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # No skip connection at final level
        self.bn9 = nn.BatchNorm2d(32)
        
        # Output layer (heatmap)
        self.conv_out = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Encoder
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x = self.pool1(x1)
        
        x2 = F.relu(self.bn3(self.conv3(x)))
        x2 = F.relu(self.bn4(self.conv4(x2)))
        x = self.pool2(x2)
        
        x3 = F.relu(self.bn5(self.conv5(x)))
        x3 = F.relu(self.bn6(self.conv6(x3)))
        x = self.pool3(x3)
        
        # Decoder with skip connections - Fixed to handle size mismatches
        x = self.up1(x)
        # Ensure sizes match for concatenation (handle odd dimensions from pooling)
        if x.size(2) != x2.size(2) or x.size(3) != x2.size(3):
            x = F.interpolate(x, size=(x2.size(2), x2.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.bn7(self.conv7(x)))
        
        x = self.up2(x)
        # Ensure sizes match for concatenation
        if x.size(2) != x1.size(2) or x.size(3) != x1.size(3):
            x = F.interpolate(x, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = F.relu(self.bn8(self.conv8(x)))
        
        x = self.up3(x)
        # Final upsampling to match input size (640x360)
        target_h, target_w = self.input_size[1], self.input_size[0]  # (H, W) = (360, 640)
        if x.size(2) != target_h or x.size(3) != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        x = F.relu(self.bn9(self.conv9(x)))
        
        # Output heatmap
        x = self.conv_out(x)
        x = self.sigmoid(x)
        
        return x


class TrackNetDetector:
    """
    TrackNet-based ball detector.
    
    Processes 3 consecutive frames to generate a heatmap of ball positions.
    """
    
    def __init__(self, model_path: Optional[str] = None, input_size: Tuple[int, int] = (640, 360)):
        """
        Initialize TrackNet detector.
        
        Args:
            model_path: Path to trained TrackNet model weights (.pth file)
            input_size: Input image size (width, height) for TrackNet
        """
        self.input_size = input_size  # (width, height)
        self.device = self._get_device()
        
        # Initialize model
        self.model = TrackNetModel(input_channels=9).to(self.device)
        self.model.eval()
        
        # Load weights if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            print("⚠️  No TrackNet model weights provided. Using untrained model.")
            print("   Please train or download TrackNet weights for best results.")
        
        # Frame buffer for 3 consecutive frames
        self.frame_buffer = deque(maxlen=3)
        
        # Store original frame size for coordinate scaling
        self.original_frame_size = None
        
    def _get_device(self):
        """Get the appropriate device for inference."""
        if USE_GPU and torch.cuda.is_available():
            try:
                device = GPU_DEVICE if isinstance(GPU_DEVICE, int) else 0
                test_tensor = torch.zeros(1).to(f'cuda:{device}')
                _ = test_tensor * 2
                return f'cuda:{device}'
            except RuntimeError:
                return 'cpu'
        return 'cpu'
    
    def load_model(self, model_path: str):
        """Load trained TrackNet model weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✓ Loaded TrackNet model from {model_path}")
        except Exception as e:
            print(f"⚠️  Failed to load TrackNet model: {e}")
            print("   Using untrained model.")
    
    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess 3 consecutive frames for TrackNet input.
        
        Args:
            frames: List of 3 frames (t-1, t, t+1) as numpy arrays (H, W, 3)
            
        Returns:
            Preprocessed tensor of shape (1, 9, H, W)
        """
        if len(frames) != 3:
            raise ValueError(f"TrackNet requires exactly 3 frames, got {len(frames)}")
        
        processed_frames = []
        for frame in frames:
            # Resize to input size
            frame_resized = cv2.resize(frame, self.input_size)
            # Normalize to [0, 1]
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            # Convert to (C, H, W)
            frame_chw = np.transpose(frame_normalized, (2, 0, 1))
            processed_frames.append(frame_chw)
        
        # Concatenate 3 frames along channel dimension: (3, H, W) * 3 -> (9, H, W)
        combined = np.concatenate(processed_frames, axis=0)
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(combined).float().unsqueeze(0).to(self.device)
        
        return tensor
    
    def heatmap_to_position(self, heatmap: np.ndarray, threshold: float = 0.5) -> Optional[np.ndarray]:
        """
        Convert heatmap to ball position coordinates.
        
        Args:
            heatmap: Heatmap array of shape (H, W) with values in [0, 1]
            threshold: Minimum probability threshold for ball detection
            
        Returns:
            Ball position [x, y] in original image coordinates, or None if not detected
        """
        # Find maximum value in heatmap
        max_val = np.max(heatmap)
        
        if max_val < threshold:
            return None
        
        # Find position of maximum value
        max_pos = np.argmax(heatmap)
        y, x = np.unravel_index(max_pos, heatmap.shape)
        
        # Scale back to original frame coordinates
        if self.original_frame_size is not None:
            orig_h, orig_w = self.original_frame_size
            scale_x = orig_w / self.input_size[0]  # width scale
            scale_y = orig_h / self.input_size[1]  # height scale
            x = x * scale_x
            y = y * scale_y
        
        position = np.array([x, y], dtype=np.float32)
        
        return position
    
    def detect_ball(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect ball position in current frame using TrackNet.
        
        Args:
            frame: Current frame as numpy array (H, W, 3)
            
        Returns:
            Ball position [x, y] in pixels, or None if not detected
        """
        # Store original frame size for coordinate scaling
        self.original_frame_size = frame.shape[:2]  # (H, W)
        
        # Add frame to buffer
        self.frame_buffer.append(frame.copy())
        
        # Need at least 3 frames for TrackNet
        if len(self.frame_buffer) < 3:
            return None
        
        # Get 3 consecutive frames
        frames = list(self.frame_buffer)
        
        # Preprocess
        input_tensor = self.preprocess_frames(frames)
        
        # Inference
        with torch.no_grad():
            heatmap_tensor = self.model(input_tensor)
        
        # Convert heatmap to numpy
        heatmap = heatmap_tensor.squeeze().cpu().numpy()
        
        # Convert heatmap to position (scaled to original frame size)
        position = self.heatmap_to_position(heatmap)
        
        # #region agent log
        if position is not None:
            with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                orig_h, orig_w = self.original_frame_size
                f.write(f'{{"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"tracknet_detector.py:281","message":"TrackNet ball detected","data":{{"x":{position[0]},"y":{position[1]},"frame_size":[{orig_w},{orig_h}]}}}}\n')
        # #endregion
        
        return position
    
    def detect_ball_batch(self, frames: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Detect ball positions for a batch of frames.
        
        Args:
            frames: List of frames to process
            
        Returns:
            List of ball positions (or None if not detected) for each frame
        """
        results = []
        
        for frame in frames:
            position = self.detect_ball(frame)
            results.append(position)
        
        return results


def create_tracknet_detector(model_path: Optional[str] = None, 
                            input_size: Tuple[int, int] = (640, 360)) -> TrackNetDetector:
    """
    Factory function to create a TrackNet detector.
    
    Args:
        model_path: Path to trained TrackNet model weights
        input_size: Input image size (width, height)
        
    Returns:
        TrackNetDetector instance
    """
    return TrackNetDetector(model_path=model_path, input_size=input_size)
