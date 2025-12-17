#Import All the Required Libraries
import sys
import os
import csv
import cv2
import numpy as np
from utils import get_center_of_bbox
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

def process_video_chunked(input_video_path, chunk_size=500):
    """
    Process video in chunks to save memory while maintaining full resolution
    """
    # Generate output filename based on input filename
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_video_path = f'output_videos/{video_name}_output.mp4'
    output_csv_path = f'output_videos/{video_name}_ball_data.csv'
    
    # Ensure output directory exists
    os.makedirs('output_videos', exist_ok=True)
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {total_frames} frames, {width}x{height}, {fps:.2f} FPS")
    
    # Initialize output video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Initialize tracker
    print("Initializing tracker...")
    tracker = Tracker("models/best.pt", fps=fps)
    
    # Process video in chunks
    all_tracks = {
        "players": [],
        "referees": [],
        "ball": []
    }
    
    ball_data = []
    ground_y = None
    
    frame_num = 0
    chunk_frames = []
    chunk_start_frame = 0
    
    print(f"Processing video in chunks of {chunk_size} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Process remaining frames
            if len(chunk_frames) > 0:
                print(f"Processing final chunk: frames {chunk_start_frame} to {frame_num-1} ({len(chunk_frames)} frames)")
                chunk_tracks = tracker.get_object_tracks(chunk_frames, read_from_stub=False, stub_path=None)
                
                # Merge tracks
                for track_type in ["players", "referees", "ball"]:
                    all_tracks[track_type].extend(chunk_tracks[track_type])
                
                # Draw and write frames
                for i, frame in enumerate(chunk_frames):
                    frame_num_global = chunk_start_frame + i
                    frame = draw_annotations_on_frame(frame, chunk_tracks, i, tracker)
                    out.write(frame)
                    
                    # Extract ball data
                    if ground_y is None and len(chunk_frames) > 0:
                        ground_y = detect_ground_line(chunk_frames[0])
                    
                    extract_ball_data_from_frame(chunk_tracks, i, frame_num_global, fps, width, height, ground_y, ball_data)
            break
        
        chunk_frames.append(frame)
        
        # Process chunk when it reaches chunk_size
        if len(chunk_frames) >= chunk_size:
            print(f"Processing chunk: frames {chunk_start_frame} to {frame_num} ({len(chunk_frames)} frames)")
            chunk_tracks = tracker.get_object_tracks(chunk_frames, read_from_stub=False, stub_path=None)
            
            # Merge tracks
            for track_type in ["players", "referees", "ball"]:
                all_tracks[track_type].extend(chunk_tracks[track_type])
            
            # Draw and write frames
            for i, frame in enumerate(chunk_frames):
                frame_num_global = chunk_start_frame + i
                frame = draw_annotations_on_frame(frame, chunk_tracks, i, tracker)
                out.write(frame)
                
                # Extract ball data
                if ground_y is None and len(chunk_frames) > 0:
                    ground_y = detect_ground_line(chunk_frames[0])
                
                extract_ball_data_from_frame(chunk_tracks, i, frame_num_global, fps, width, height, ground_y, ball_data)
            
            # Reset for next chunk
            chunk_frames = []
            chunk_start_frame = frame_num + 1
        
        frame_num += 1
        
        if frame_num % 100 == 0:
            print(f"  Processed {frame_num}/{total_frames} frames ({frame_num/total_frames*100:.1f}%)")
    
    cap.release()
    out.release()
    
    # Save CSV
    if len(ball_data) > 0:
        fieldnames = list(ball_data[0].keys())
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ball_data)
        print(f"✓ Saved {len(ball_data)} ball detections to: {output_csv_path}")
    else:
        print("⚠ No ball detections found for CSV export")
    
    print(f"\n✓ Done! Output saved to: {output_video_path}")
    return output_video_path, output_csv_path

def draw_annotations_on_frame(frame, tracks, frame_idx, tracker):
    """Draw annotations on a single frame"""
    frame = frame.copy()
    
    player_dict = tracks["players"][frame_idx]
    ball_dict = tracks["ball"][frame_idx]
    referee_dict = tracks["referees"][frame_idx]
    
    # Draw Players
    for track_id, player in player_dict.items():
        frame = tracker.draw_ellipse(frame, player["bbox"], (0,0,255), track_id)
    
    # Draw Referee
    for _, referee in referee_dict.items():
        frame = tracker.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
    
    # Draw ball
    for track_id, ball in ball_dict.items():
        if ball.get("predicted", False):
            frame = tracker.draw_traingle(frame, ball["bbox"], (0, 165, 255))
        else:
            frame = tracker.draw_traingle(frame, ball["bbox"], (0, 255, 0))
    
    return frame

def extract_ball_data_from_frame(tracks, frame_idx, global_frame_num, fps, width, height, ground_y, ball_data):
    """Extract ball data from a single frame"""
    if ground_y is None:
        return
    
    ball_dict = tracks["ball"][frame_idx]
    
    for track_id, ball in ball_dict.items():
        if not ball.get("predicted", False):  # Only actual detections
            bbox = ball["bbox"]
            x_center, y_center = get_center_of_bbox(bbox)
            height_from_ground = ground_y - y_center
            
            x1, y1, x2, y2 = bbox
            ball_width = x2 - x1
            ball_height = y2 - y1
            distance_from_bottom = height - y_center
            time_seconds = global_frame_num / fps
            
            ball_info = {
                'frame_number': global_frame_num,
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

def main():
    # Get input video path from command line argument or use default
    if len(sys.argv) > 1:
        input_video_path = sys.argv[1]
    else:
        input_video_path = "input_videos/video.mp4"
    
    # Check if video file exists
    if not os.path.exists(input_video_path):
        print(f"Error: Video file not found at {input_video_path}")
        print("Usage: python main_chunked.py [path_to_video]")
        sys.exit(1)
    
    print(f"Processing video: {input_video_path}")
    print("Using chunked processing to maintain full resolution...")
    
    process_video_chunked(input_video_path, chunk_size=500)

if __name__ == "__main__":
    main()

