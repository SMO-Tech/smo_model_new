#!/usr/bin/env python3
"""
Video Downloader Utility

Downloads videos from various sources including Veo matches, YouTube, etc.
"""

import sys
from pathlib import Path
import subprocess
import os

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))


def download_with_ytdlp(url: str, output_path: str = None) -> str:
    """
    Download video using yt-dlp (recommended for most video sources).
    
    Args:
        url: Video URL to download
        output_path: Optional output path (if None, uses video title)
        
    Returns:
        Path to downloaded video file
    """
    try:
        # Check if yt-dlp is installed
        result = subprocess.run(['yt-dlp', '--version'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError("yt-dlp not found")
    except FileNotFoundError:
        print("❌ yt-dlp is not installed.")
        print("   Installing yt-dlp...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'yt-dlp'], check=True)
        print("✅ yt-dlp installed successfully!")
    
    # Prepare output path
    if output_path is None:
        output_path = str(PROJECT_DIR / "input_videos" / "%(title)s.%(ext)s")
    else:
        output_path = str(Path(output_path))
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading video from: {url}")
    print(f"💾 Saving to: {output_path}")
    
    # Build yt-dlp command
    cmd = [
        'yt-dlp',
        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Best quality MP4
        '--merge-output-format', 'mp4',
        '-o', output_path,
        url
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Video downloaded successfully!")
        
        # Try to find the actual output file
        if '%(title)s' in output_path or '%(ext)s' in output_path:
            # yt-dlp will have replaced placeholders, need to find the actual file
            output_dir = Path(output_path).parent
            # List files and find the most recent one
            files = list(output_dir.glob("*.mp4"))
            if files:
                # Get most recently modified
                latest_file = max(files, key=lambda p: p.stat().st_mtime)
                return str(latest_file)
        else:
            return output_path
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading video: {e}")
        print(f"   stderr: {e.stderr}")
        raise


def download_veo_match(url: str, output_path: str = None) -> str:
    """
    Download video from a Veo match URL.
    
    Args:
        url: Veo match URL
        output_path: Optional output path
        
    Returns:
        Path to downloaded video file
    """
    print(f"🎥 Downloading Veo match video...")
    
    # Veo videos can often be downloaded with yt-dlp
    # If that doesn't work, we may need to use browser automation
    try:
        return download_with_ytdlp(url, output_path)
    except Exception as e:
        print(f"⚠️  yt-dlp failed: {e}")
        print("   Veo videos may require authentication or browser automation.")
        print("   Please try:")
        print("   1. Log in to Veo in your browser")
        print("   2. Use browser extension to download")
        print("   3. Or check if Veo provides a direct download link")
        raise


def main():
    """Main function to download video from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download video from URL')
    parser.add_argument('url', help='Video URL to download')
    parser.add_argument('-o', '--output', help='Output file path', default=None)
    parser.add_argument('--veo', action='store_true', help='Force Veo download method')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = PROJECT_DIR / "input_videos"
    output_dir.mkdir(exist_ok=True)
    
    if args.output:
        output_path = args.output
    else:
        # Generate output path from URL
        if 'veo.co' in args.url:
            # Extract match ID from URL
            match_id = args.url.split('/')[-1].split('?')[0]
            output_path = str(output_dir / f"veo_match_{match_id}.mp4")
        else:
            output_path = str(output_dir / "downloaded_video.mp4")
    
    try:
        if args.veo or 'veo.co' in args.url:
            video_path = download_veo_match(args.url, output_path)
        else:
            video_path = download_with_ytdlp(args.url, output_path)
        
        print(f"\n✅ Video downloaded to: {video_path}")
        return video_path
    except Exception as e:
        print(f"\n❌ Failed to download video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
