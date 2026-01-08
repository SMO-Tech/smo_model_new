#!/usr/bin/env python3
"""
Download TrackNet pre-trained model weights.

TrackNet models are typically available from:
1. Official TrackNet repository releases
2. Community-trained models
3. HuggingFace (if available)

This script attempts to download from multiple sources.
"""

import os
import sys
import shutil
import requests
from pathlib import Path
from urllib.parse import urlparse

PROJECT_DIR = Path(__file__).resolve().parent
TRACKNET_MODEL_DIR = PROJECT_DIR / "Models" / "Pretrained" / "TrackNet"
TRACKNET_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, target_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL to target path.
    
    Args:
        url: URL to download from
        target_path: Path to save the file
        chunk_size: Chunk size for streaming download
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"📥 Downloading from: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   Progress: {percent:.1f}%", end='', flush=True)
        
        print(f"\n✅ Successfully downloaded to: {target_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading: {e}")
        if target_path.exists():
            target_path.unlink()  # Remove partial download
        return False


def download_from_huggingface(target_path: Path) -> bool:
    """
    Download TrackNet model from HuggingFace (if available).
    
    Args:
        target_path: Path to save the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import hf_hub_download
        
        print("🔍 Searching HuggingFace for TrackNet models...")
        
        # Common TrackNet model names to try
        possible_files = [
            'tracknet_weights.pth',
            'model.pth',
            'best.pth',
            'tracknet_model.h5',
            'weights.pth'
        ]
        
        # Common repository IDs to try
        repos = [
            'yastrebksv/TrackNet',
            'tracknet/tracknet',
            'sports-analytics/tracknet'
        ]
        
        for repo_id in repos:
            for filename in possible_files:
                try:
                    print(f"   Trying: {repo_id}/{filename}")
                    model_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename
                    )
                    shutil.copy(model_file, target_path)
                    print(f"✅ Successfully downloaded from HuggingFace")
                    return True
                except:
                    continue
        
        return False
        
    except ImportError:
        print("   ⚠️  huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"   ⚠️  Error accessing HuggingFace: {e}")
        return False


def download_from_google_drive(file_id: str, target_path: Path) -> bool:
    """
    Download a file from Google Drive using file ID.
    
    Args:
        file_id: Google Drive file ID
        target_path: Path to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"📥 Downloading from Google Drive (file ID: {file_id})...")
        
        # Google Drive direct download URL
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # First request to get the download link (Google Drive may require confirmation)
        session = requests.Session()
        response = session.get(url, stream=True, timeout=30)
        
        # Check if we need to confirm download (large files)
        if 'confirm' in response.url:
            # Extract confirm token
            confirm_token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    confirm_token = value
                    break
            
            if confirm_token:
                url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
                response = session.get(url, stream=True, timeout=30)
        
        response.raise_for_status()
        
        # Download the file
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   Progress: {percent:.1f}%", end='', flush=True)
        
        print(f"\n✅ Successfully downloaded from Google Drive")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading from Google Drive: {e}")
        if target_path.exists():
            target_path.unlink()
        return False


def main():
    """Download TrackNet pre-trained model."""
    print("=" * 60)
    print("TrackNet Model Downloader")
    print("=" * 60)
    
    model_filename = "tracknet_weights.pth"
    target_path = TRACKNET_MODEL_DIR / model_filename
    
    # Check if model already exists
    if target_path.exists():
        print(f"\n✅ TrackNet model already exists: {target_path}")
        print(f"   To re-download, delete the file first.")
        return
    
    print(f"\n📦 Target location: {target_path}")
    print("\n🔍 Searching for pre-trained TrackNet model...")
    print("   (This may take a few moments)")
    
    success = False
    
    # Try Google Drive first (known pre-trained model location)
    print("\n1️⃣  Trying Google Drive (official pre-trained model)...")
    google_drive_file_id = "1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"
    if download_from_google_drive(google_drive_file_id, target_path):
        success = True
    
    # Try HuggingFace
    if not success:
        print("\n2️⃣  Trying HuggingFace...")
        if download_from_huggingface(target_path):
            success = True
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TrackNet model downloaded successfully!")
        print(f"   Model saved to: {target_path}")
        print("\n💡 Usage:")
        print(f"   from ball_tracking import create_tracknet_detector")
        print(f"   detector = create_tracknet_detector(model_path='{target_path}')")
    else:
        print("⚠️  Could not automatically download TrackNet model.")
        print("\n📝 Manual download options:")
        print("   1. Download from Google Drive:")
        print("      https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl/view")
        print("      Save to:", target_path)
        print("   2. Check the official TrackNet repository:")
        print("      https://github.com/yastrebksv/TrackNet")
        print("   3. Train your own model using the TrackNet training scripts")
        print("\n💡 Note: The code will work with an untrained model, but")
        print("   results will be better with a trained model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
