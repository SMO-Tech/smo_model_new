"""
Model Loading and Validation Script

This script validates that all required models are present and loads them
to verify they work correctly before processing videos.
"""

import sys
from pathlib import Path
import time

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

from constants import model_path
from keypoint_detection.keypoint_constants import keypoint_model_path
from pipelines import DetectionPipeline, KeypointPipeline, TrackingPipeline, TacticalPipeline
from pass_detection import PlayerOnlyPassDetector


def check_model_path(model_path: Path, model_name: str) -> bool:
    """Check if a model file exists."""
    if model_path.exists():
        print(f"✅ {model_name}: Found at {model_path}")
        return True
    else:
        print(f"❌ {model_name}: NOT FOUND at {model_path}")
        return False


def load_all_models():
    """Load and validate all required models."""
    print("=" * 60)
    print("MODEL LOADING AND VALIDATION")
    print("=" * 60)
    print()
    
    # Check model paths exist
    print("Step 1: Checking model file paths...")
    print("-" * 60)
    detection_ok = check_model_path(model_path, "Detection Model")
    keypoint_ok = check_model_path(keypoint_model_path, "Keypoint Model")
    print()
    
    if not detection_ok or not keypoint_ok:
        print("⚠️  WARNING: Some model files are missing!")
        print("   Please download the models before proceeding.")
        print()
        print("Detection Model:")
        print("   https://huggingface.co/Adit-jain/soccana")
        print()
        print("Keypoint Model:")
        print("   https://huggingface.co/Adit-jain/Soccana_Keypoint")
        print()
        return False
    
    # Load models
    print("Step 2: Loading models into memory...")
    print("-" * 60)
    start_time = time.time()
    
    try:
        # Initialize pipelines
        print("Initializing Detection Pipeline...")
        detection_pipeline = DetectionPipeline(str(model_path))
        detection_pipeline.initialize_model()
        print("✅ Detection model loaded successfully")
        
        print("Initializing Keypoint Pipeline...")
        keypoint_pipeline = KeypointPipeline(str(keypoint_model_path))
        keypoint_pipeline.initialize_model()
        print("✅ Keypoint model loaded successfully")
        
        print("Initializing Tracking Pipeline...")
        tracking_pipeline = TrackingPipeline(str(model_path))
        tracking_pipeline.initialize_models()
        print("✅ Tracking pipeline initialized successfully")
        
        print("Initializing Tactical Pipeline...")
        tactical_pipeline = TacticalPipeline(str(keypoint_model_path), str(model_path))
        tactical_pipeline.initialize_models()
        print("✅ Tactical pipeline initialized successfully")
        
        print("Initializing Pass Detector...")
        pass_detector = PlayerOnlyPassDetector()
        print("✅ Pass detector initialized successfully")
        
        load_time = time.time() - start_time
        print()
        print(f"✅ All models loaded successfully in {load_time:.2f}s")
        print()
        
        # Test with dummy data
        print("Step 3: Testing models with dummy data...")
        print("-" * 60)
        import numpy as np
        
        # Create dummy frame
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        print("Created dummy frame (1280x720)")
        
        # Test detection
        try:
            player_detections, _, referee_detections = detection_pipeline.detect_frame_objects(dummy_frame)
            print(f"✅ Detection test: Found {len(player_detections.xyxy)} players, {len(referee_detections.xyxy)} referees")
        except Exception as e:
            print(f"❌ Detection test failed: {e}")
            return False
        
        # Test keypoint detection
        try:
            keypoints, _ = keypoint_pipeline.detect_keypoints_in_frame(dummy_frame)
            if keypoints is not None:
                print(f"✅ Keypoint test: Detected {len(keypoints)} keypoint sets")
            else:
                print("⚠️  Keypoint test: No keypoints detected (expected for dummy frame)")
        except Exception as e:
            print(f"❌ Keypoint test failed: {e}")
            return False
        
        print()
        print("=" * 60)
        print("✅ ALL MODELS LOADED AND TESTED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Ready to process videos!")
        print()
        print("Next steps:")
        print("  1. Update test_video path in constants.py")
        print("  2. Run: python main.py")
        print()
        
        return True
        
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ ERROR LOADING MODELS")
        print("=" * 60)
        print(f"Error: {type(e).__name__}: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = load_all_models()
    sys.exit(0 if success else 1)

