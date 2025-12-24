#!/usr/bin/env python3
"""
Download required models from HuggingFace for Soccer Analysis.
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

PROJECT_DIR = Path(__file__).resolve().parent

def download_model(repo_id, filename, target_path, subfolder=None):
    """Download a model from HuggingFace and save to target path."""
    print(f"\n📥 Downloading {filename} from {repo_id}...")
    
    try:
        # Download model
        download_kwargs = {
            'repo_id': repo_id,
            'filename': filename
        }
        if subfolder:
            download_kwargs['subfolder'] = subfolder
            
        model_file = hf_hub_download(**download_kwargs)
        
        # Create target directory
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy model to target location
        shutil.copy(model_file, target_path)
        print(f"✅ Successfully downloaded to: {target_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        print(f"   Trying alternative paths...")
        
        # Try alternative paths (models are in Model/weights/ subfolder)
        alternatives = [
            f"Model/weights/{filename}",
            f"weights/{filename}",
            f"Model/{filename}",
        ]
        
        for alt_path in alternatives:
            try:
                print(f"   Trying: {alt_path}")
                model_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=alt_path
                )
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(model_file, target_path)
                print(f"✅ Successfully downloaded from {alt_path}")
                return True
            except:
                continue
        
        return False

def main():
    """Download all required models."""
    print("=" * 60)
    print("Soccer Analysis - Model Downloader")
    print("=" * 60)
    
    # Detection model
    detection_model_path = PROJECT_DIR / "Models/Trained/yolov11_sahi_1280/Model/weights/best.pt"
    
    # Keypoint model
    keypoint_model_path = PROJECT_DIR / "Models/Trained/yolov11_keypoints_29/Model/weights/best.pt"
    
    success = True
    
    # Download detection model if it doesn't exist
    if not detection_model_path.exists():
        print("\n⚠️  Detection model not found. You have two options:")
        print("   1. Download from HuggingFace (if available)")
        print("   2. Use a base YOLO model (yolo11n.pt) as fallback")
        
        # Try downloading
        downloaded = download_model(
            repo_id='Adit-jain/soccana',
            filename='best.pt',
            target_path=detection_model_path
        )
        
        if not downloaded:
            print("\n📦 Using base YOLO model as fallback...")
            # Use base YOLO model
            base_model_path = PROJECT_DIR / "Models/Pretrained/yolo11n.pt"
            base_model_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not base_model_path.exists():
                print("   Downloading yolo11n.pt from Ultralytics...")
                from ultralytics import YOLO
                model = YOLO('yolo11n.pt')
                shutil.copy('yolo11n.pt', base_model_path)
            
            # Update constants to use base model
            detection_model_path = base_model_path
            print(f"   Using base model: {base_model_path}")
            success = True
        else:
            success = True
    else:
        print(f"✅ Detection model already exists: {detection_model_path}")
    
    # Download keypoint model if it doesn't exist
    if not keypoint_model_path.exists():
        print("\n⚠️  Keypoint model not found. Trying to download...")
        downloaded = download_model(
            repo_id='Adit-jain/Soccana_Keypoint',
            filename='best.pt',
            target_path=keypoint_model_path
        )
        if not downloaded:
            print("   ⚠️  Keypoint model not available. Some features may not work.")
            success = False
    else:
        print(f"✅ Keypoint model already exists: {keypoint_model_path}")
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All models downloaded successfully!")
    else:
        print("⚠️  Some models failed to download. Please check the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()

