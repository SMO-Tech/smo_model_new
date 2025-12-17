#!/usr/bin/env python3
"""
Extract ball data from video - ONLY from actual detections (green), not predictions
Processes video frame by frame and extracts ball position and height data
"""

import sys
import os
import cv2
import numpy as np
import csv
from ultralytics import YOLO
import supervision as sv
from utils import get_center_of_bbox, read_video

def detect_ground_line(image):
    """
    Detect the ground/pitch line in the image
    Returns y-coordinate of ground line
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect horizontal lines (pitch lines)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                           minLineLength=width//4, maxLineGap=50)
    
    if lines is not None:
        ground_y = height
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:  # Horizontal line
                y_avg = (y1 + y2) // 2
                if y_avg > ground_y * 0.7:
                    ground_y = min(ground_y, y_avg)
        return ground_y
    
    # Fallback: use bottom 10% of image as ground estimate
    return int(height * 0.95)

def extract_ball_data_from_video(video_path, model_path="models/best.pt"):
    """
    Extract ball data from video - only actual detections, not predictions
    
    Returns:
        List of dictionaries with ball data for each frame
    """
    print(f"Reading video: {video_path}")
    video_frames, fps = read_video(video_path)
    print(f"Read {len(video_frames)} frames at {fps} FPS")
    
    # Detect ground line from first frame
    ground_y = detect_ground_line(video_frames[0])
    height, width = video_frames[0].shape[:2]
    print(f"Image size: {width}x{height}")
    print(f"Estimated ground line at y={ground_y}")
    
    # Load YOLO model
    print("Loading model...")
    model = YOLO(model_path)
    
    # Initialize tracker
    tracker = sv.ByteTrack()
    
    all_ball_data = []
    
    print("Processing frames...")
    batch_size = 20
    
    for batch_start in range(0, len(video_frames), batch_size):
        batch_end = min(batch_start + batch_size, len(video_frames))
        batch_frames = video_frames[batch_start:batch_end]
        
        print(f"Processing frames {batch_start} to {batch_end-1}...")
        
        # Run detection on batch
        detections = model.predict(batch_frames, conf=0.1, verbose=False)
        
        for i, detection in enumerate(detections):
            frame_num = batch_start + i
            frame = batch_frames[i]
            
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            
            # Convert to Supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # Convert goalkeeper to player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
            
            # Track objects
            detection_with_tracks = tracker.update_with_detections(detection_supervision)
            
            # Extract ball data - ONLY from actual detections (not tracked, ball doesn't use tracking)
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = float(frame_detection[2]) if len(frame_detection) > 2 else 0.0
                
                if cls_id == cls_names_inv['ball']:
                    # Get ball center
                    x_center, y_center = get_center_of_bbox(bbox)
                    
                    # Calculate height from ground
                    height_from_ground = ground_y - y_center
                    
                    # Ball bounding box dimensions
                    x1, y1, x2, y2 = bbox
                    ball_width = x2 - x1
                    ball_height = y2 - y1
                    
                    # Distance from bottom of image
                    distance_from_bottom = height - y_center
                    
                    # Time in video
                    time_seconds = frame_num / fps
                    
                    ball_info = {
                        'frame_number': frame_num,
                        'time_seconds': round(time_seconds, 3),
                        'fps': fps,
                        'image_width': width,
                        'image_height': height,
                        'ball_x': int(x_center),
                        'ball_y': int(y_center),
                        'ball_bbox_x1': int(x1),
                        'ball_bbox_y1': int(y1),
                        'ball_bbox_x2': int(x2),
                        'ball_bbox_y2': int(y2),
                        'ball_width_pixels': int(ball_width),
                        'ball_height_pixels': int(ball_height),
                        'ground_line_y': int(ground_y),
                        'height_from_ground_pixels': int(height_from_ground),
                        'distance_from_bottom_pixels': int(distance_from_bottom),
                        'confidence': round(confidence, 4),
                        'is_predicted': False  # Always False - only actual detections
                    }
                    
                    all_ball_data.append(ball_info)
        
        # Progress update
        if len(all_ball_data) > 0:
            print(f"  Found {len(all_ball_data)} ball detections so far...")
    
    print(f"\nTotal ball detections: {len(all_ball_data)} out of {len(video_frames)} frames")
    print(f"Detection rate: {len(all_ball_data)/len(video_frames)*100:.1f}%")
    
    return all_ball_data

def save_to_csv(ball_data, output_path):
    """Save ball data to CSV file"""
    if len(ball_data) == 0:
        print("No ball data to save")
        return
    
    # Get all keys from first ball data entry
    fieldnames = list(ball_data[0].keys())
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ball_data)
    
    print(f"✓ Saved {len(ball_data)} ball detections to: {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_video_ball_data.py <video_path> [output_csv]")
        print("Example: python extract_video_ball_data.py input_videos/video.mp4")
        sys.exit(1)
    
    input_video = sys.argv[1]
    
    if not os.path.exists(input_video):
        print(f"Error: Video file not found: {input_video}")
        sys.exit(1)
    
    # Generate output CSV path if not provided
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]
    else:
        base_name = os.path.splitext(os.path.basename(input_video))[0]
        output_dir = "output_videos"
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, f"{base_name}_ball_data.csv")
    
    try:
        # Extract ball data (only actual detections)
        ball_data = extract_ball_data_from_video(input_video)
        
        if len(ball_data) == 0:
            print("\n⚠ No ball detections found in video")
            sys.exit(1)
        
        # Save to CSV
        save_to_csv(ball_data, output_csv)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

