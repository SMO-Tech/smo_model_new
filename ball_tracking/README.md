# TrackNet Ball Detection

This module provides TrackNet-based ball detection as an alternative to YOLO for tracking small, fast-moving objects like soccer balls.

## Overview

TrackNet is specifically designed for tracking high-speed, tiny objects in sports applications. It uses 3 consecutive frames (t-1, t, t+1) to understand motion and outputs a heatmap indicating ball location probability.

## Key Features

- **3-Frame Input**: Processes consecutive frames to understand ball motion
- **Heatmap Output**: Generates probability heatmaps instead of bounding boxes
- **Better for Small Objects**: More accurate than YOLO for small, fast-moving balls
- **Motion-Aware**: Uses temporal information to track ball trajectory

## Usage

### Basic Usage

```python
from ball_tracking import create_tracknet_detector

# Create detector (model_path is optional - will use untrained model if not provided)
detector = create_tracknet_detector(
    model_path="path/to/tracknet_weights.pth",
    input_size=(640, 360)  # Standard TrackNet input size
)

# Detect ball in frame
ball_position = detector.detect_ball(frame)
if ball_position is not None:
    x, y = ball_position
    print(f"Ball detected at ({x}, {y})")
```

### Integration with Detection Pipeline

```python
from pipelines import DetectionPipeline

# Initialize pipeline with TrackNet
pipeline = DetectionPipeline(
    model_path="path/to/yolo_model.pt",
    tracknet_model_path="path/to/tracknet_weights.pth",
    use_tracknet=True  # Enable TrackNet for ball detection
)

# Use pipeline normally
player_detections, ball_detections, referee_detections = pipeline.detect_frame_objects(frame)
```

### Integration with Main Pipeline

```python
from main import CompleteSoccerAnalysisPipeline

pipeline = CompleteSoccerAnalysisPipeline(
    detection_model_path="path/to/yolo_model.pt",
    keypoint_model_path="path/to/keypoint_model.pt",
    tracknet_model_path="path/to/tracknet_weights.pth",
    use_tracknet=True  # Enable TrackNet
)
```

## Model Training

To train your own TrackNet model:

1. **Prepare Dataset**: Collect video clips with labeled ball positions
2. **Format Data**: Organize frames and ground truth heatmaps
3. **Train Model**: Use TrackNet training scripts (see TrackNet repository)
4. **Save Weights**: Save trained model weights as `.pth` file

## Model Architecture

The TrackNet model uses:
- **Encoder**: Downsampling layers to extract features
- **Decoder**: Upsampling layers with skip connections
- **Output**: Single-channel heatmap (probability of ball at each pixel)

## Input Requirements

- **Frame Size**: 640x360 pixels (standard TrackNet input)
- **Frame Buffer**: Requires 3 consecutive frames
- **Color Format**: RGB (3 channels per frame)

## Output Format

- **Heatmap**: Probability map of shape (H, W) with values in [0, 1]
- **Position**: Ball coordinates [x, y] in pixels (extracted from heatmap maximum)

## Notes

- TrackNet requires at least 3 frames before it can detect the ball
- The first 2 frames will return `None` for ball position
- For best results, use a trained TrackNet model (not the default untrained model)
- TrackNet is more accurate than YOLO for small, fast-moving balls but requires more computation

## References

- TrackNet Paper: "TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications"
- TrackNet Repository: https://github.com/yastrebksv/TrackNet
