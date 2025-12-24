import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import pandas as pd
import numpy as np
from utils import read_video, write_video


class ProcessingPipeline:
    """
    Pipeline for video processing utilities like reading, writing, and ball interpolation.
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def read_video_frames(video_path, frame_count=-1):
        """
        Read video frames from a file.
        
        Args:
            video_path: Path to the video file
            frame_count: Number of frames to read (-1 for all frames)
            
        Returns:
            List of video frames
        """
        # #region agent log
        import json
        with open('/root/Soccer_Analysis/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"location":"processing_pipeline.py:read_video_frames:entry","message":"Function entry","data":{"video_path":str(video_path),"frame_count":frame_count},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"L"})+"\n")
        # #endregion
        print(f"Reading video from {video_path}...")
        # #region agent log
        with open('/root/Soccer_Analysis/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"location":"processing_pipeline.py:read_video_frames:before_read","message":"Before read_video call","data":{},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"M"})+"\n")
        # #endregion
        frames = read_video(video_path, frame_count=frame_count)
        # #region agent log
        with open('/root/Soccer_Analysis/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"location":"processing_pipeline.py:read_video_frames:after_read","message":"After read_video call","data":{"frames_len":len(frames) if frames else 0},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"N"})+"\n")
        # #endregion
        return frames
    
    @staticmethod
    def write_video_output(frames, output_path, fps=30):
        """
        Write video frames to an output file.
        
        Args:
            frames: List of video frames to write
            output_path: Output video file path
            fps: Frames per second for output video
        """
        print(f"Writing video to {output_path}...")
        write_video(frames, output_path, fps=fps)
    
    @staticmethod
    def interpolate_ball_tracks(tracks):
        """
        Interpolate ball tracks to fill in missing detections with improved smoothing.
        
        Args:
            tracks: Dictionary containing tracking data
            
        Returns:
            Updated tracks with interpolated ball positions
        """
        print("Interpolating ball tracks...")
        
        # Get ball tracks
        ball_tracks = tracks['ball']
        
        # Convert to DataFrame for interpolation
        df = pd.DataFrame.from_dict(ball_tracks, orient='index')
        df.columns = ['x1', 'y1', 'x2', 'y2']
        
        # Replace None values with NaN for proper interpolation
        df = df.replace([None], np.nan)
        
        # Perform linear interpolation with smaller gap limit for smoother tracking
        df = df.interpolate(method='linear', limit_direction='both', limit=20)
        
        # Apply simple moving average smoothing to reduce jitter (window=3)
        for col in df.columns:
            df[col] = df[col].rolling(window=3, center=True, min_periods=1).mean()
        
        # Fill any remaining NaN values
        df = df.bfill().ffill()
        
        # Convert back to dictionary format
        new_tracks = {}
        for i, box in enumerate(df.to_numpy()):
            new_tracks[i] = box.tolist()
        
        # Update original tracks
        tracks['ball'] = new_tracks
        return tracks
    
    @staticmethod
    def generate_output_path(input_path, suffix="_tracked"):
        """
        Generate output path based on input path with a suffix.
        
        Args:
            input_path: Original video path
            suffix: Suffix to add before file extension
            
        Returns:
            Generated output path
        """
        return input_path.replace(".mp4", f"{suffix}.mp4")