#Import All the Required Libraries
import sys
import os
import csv
import cv2
import numpy as np
from utils import read_video, save_video, get_center_of_bbox
from trackers import Tracker

def detect_ground_line(image):
    """Detect the ground/pitch line in the image"""
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

def extract_ball_data_to_csv(tracks, video_frames, fps, output_csv_path):
    """
    Extract ball data from tracks - ONLY actual detections (not predictions)
    Save to CSV file
    """
    if len(video_frames) == 0:
        return
    
    # Detect ground line from first frame
    ground_y = detect_ground_line(video_frames[0])
    height, width = video_frames[0].shape[:2]
    
    ball_data = []
    
    for frame_num in range(len(tracks["ball"])):
        ball_dict = tracks["ball"][frame_num]
        
        # Only extract data from actual detections (not predictions)
        for track_id, ball in ball_dict.items():
            if not ball.get("predicted", False):  # Only actual detections
                bbox = ball["bbox"]
                
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
                    'track_id': track_id,
                    'is_predicted': False
                }
                
                ball_data.append(ball_info)
    
    # Save to CSV
    if len(ball_data) > 0:
        fieldnames = list(ball_data[0].keys())
        
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ball_data)
        
        print(f"✓ Saved {len(ball_data)} ball detections to: {output_csv_path}")
    else:
        print("⚠ No ball detections found for CSV export")

def main():
    # Get input video path from command line argument or use default
    if len(sys.argv) > 1:
        input_video_path = sys.argv[1]
    else:
        input_video_path = "input_videos/video.mp4"
    
    # Check if video file exists
    if not os.path.exists(input_video_path):
        print(f"Error: Video file not found at {input_video_path}")
        print("Usage: python main.py [path_to_video]")
        sys.exit(1)
    
    # Generate output filename based on input filename
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_video_path = f'output_videos/{video_name}_output.mp4'
    output_csv_path = f'output_videos/{video_name}_ball_data.csv'
    
    # Ensure output directory exists
    os.makedirs('output_videos', exist_ok=True)
    
    print(f"Reading video: {input_video_path}")
    #Read Video
    video_frames, original_fps = read_video(input_video_path)
    print(f"Read {len(video_frames)} frames at {original_fps} FPS")

    print("Initializing tracker with physics-based ball prediction...")
    #Initialize Tracker with FPS for physics calculations
    tracker = Tracker("models/best.pt", fps=original_fps)
    
    print("Detecting and tracking objects...")
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path='tracker_stubs/player_detection.pkl')

    print("Drawing annotations...")
    #Draw Output
    #Draw Object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    print(f"Processed {len(output_video_frames)} frames")

    print(f"Saving output video: {output_video_path}")
    #Save Video with original FPS
    save_video(output_video_frames, output_video_path, fps=original_fps)
    
    print("\nExtracting ball data to CSV...")
    # Extract ball data (only actual detections, not predictions)
    extract_ball_data_to_csv(tracks, video_frames, original_fps, output_csv_path)
    
    print("\n✓ Done!")
    print(f"  Video: {output_video_path}")
    print(f"  CSV: {output_csv_path}")


if __name__ == "__main__":
    main()