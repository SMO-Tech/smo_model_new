# Downloading TrackNet Pre-trained Models

## Automatic Download

Run the download script:

```bash
python download_tracknet_model.py
```

This script will attempt to download TrackNet weights from:
- HuggingFace (if available)
- GitHub releases
- Known download links

## Manual Download

If automatic download fails, you can manually download TrackNet models:

### Option 1: From TrackNet Repository

1. Visit the official TrackNet repository:
   - https://github.com/yastrebksv/TrackNet

2. Check the **Releases** section for pre-trained weights

3. Download the model file (usually `.pth` or `.h5` format)

4. Save it to:
   ```
   Models/Pretrained/TrackNet/tracknet_weights.pth
   ```

### Option 2: From Research Paper

1. Check the original TrackNet paper:
   - "TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications"
   - https://arxiv.org/abs/1907.03698

2. Look for supplementary materials or model weights links

3. Download and save to the model directory

### Option 3: Train Your Own

If no pre-trained model is available, you can train TrackNet:

1. **Prepare Dataset**:
   - Collect video clips with labeled ball positions
   - Create ground truth heatmaps for each frame

2. **Use TrackNet Training Scripts**:
   ```bash
   git clone https://github.com/yastrebksv/TrackNet.git
   cd TrackNet
   # Follow training instructions in the repository
   ```

3. **Save Trained Weights**:
   - Save the trained model as `.pth` file
   - Place in `Models/Pretrained/TrackNet/`

## Model File Location

The code expects TrackNet weights at:
```
Models/Pretrained/TrackNet/tracknet_weights.pth
```

You can also specify a custom path when creating the detector:
```python
from ball_tracking import create_tracknet_detector

detector = create_tracknet_detector(
    model_path="path/to/your/tracknet_weights.pth"
)
```

## Model Format

TrackNet models are typically saved as:
- **PyTorch**: `.pth` or `.pth.tar` files
- **Keras/TensorFlow**: `.h5` files

The current implementation supports PyTorch `.pth` files.

## Verification

To verify your model is loaded correctly:

```python
from ball_tracking import create_tracknet_detector

detector = create_tracknet_detector(
    model_path="Models/Pretrained/TrackNet/tracknet_weights.pth"
)

# Test with a frame
import cv2
frame = cv2.imread("test_frame.jpg")
ball_position = detector.detect_ball(frame)

if ball_position:
    print(f"✓ Model loaded successfully! Ball detected at {ball_position}")
else:
    print("Model loaded, but no ball detected in this frame")
```

## Notes

- **Untrained Model**: The code will work with an untrained model, but results will be poor
- **Model Size**: TrackNet models are typically 10-50 MB
- **Compatibility**: Ensure the model architecture matches the TrackNetModel class in `tracknet_detector.py`

## Troubleshooting

**Issue**: Model file not found
- **Solution**: Check the file path and ensure the file exists
- **Solution**: Use absolute path instead of relative path

**Issue**: Model architecture mismatch
- **Solution**: Ensure the model was trained with the same architecture
- **Solution**: Check model input/output dimensions match

**Issue**: CUDA out of memory
- **Solution**: Use CPU instead: Set `USE_GPU=False` in constants.py
- **Solution**: Reduce batch size or input image size
