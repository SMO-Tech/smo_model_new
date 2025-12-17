#!/usr/bin/env python3
"""
Extract ball data from image and save to CSV
Includes ball position and height from ground
"""

import sys
import os
import cv2
import numpy as np
import csv
from ultralytics import YOLO
import supervision as sv
from utils import get_center_of_bbox

def detect_ground_line(image):
    """
    Detect the ground/pitch line in the image
    Uses edge detection and horizontal line detection
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
        # Find the lowest horizontal line (likely the ground/pitch line)
        ground_y = height  # Default to bottom of image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if it's roughly horizontal
            if abs(y2 - y1) < 10:  # Horizontal line
                y_avg = (y1 + y2) // 2
                if y_avg > ground_y * 0.7:  # Lower part of image
                    ground_y = min(ground_y, y_avg)
        return ground_y
    
    # Fallback: use bottom 10% of image as ground estimate
    return int(height * 0.95)

def extract_ball_data(image_path, model_path="models/best.pt"):
    """
    Extract ball detection data from image
    
    Returns:
        List of dictionaries with ball data
    """
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    height, width = image.shape[:2]
    print(f"Processing image: {image_path}")
    print(f"Image size: {width}x{height}")
    
    # Detect ground line
    ground_y = detect_ground_line(image)
    print(f"Estimated ground line at y={ground_y} (image height={height})")
    
    # Load YOLO model
    model = YOLO(model_path)
    
    # Run detection
    results = model.predict(image, conf=0.1, verbose=False)
    detection = results[0]
    cls_names = detection.names
    cls_names_inv = {v: k for k, v in cls_names.items()}
    
    # Convert to Supervision format
    detection_supervision = sv.Detections.from_ultralytics(detection)
    
    # Extract ball data
    ball_data = []
    
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
            
            # Distance from bottom of image (alternative measure)
            distance_from_bottom = height - y_center
            
            ball_info = {
                'image_path': os.path.basename(image_path),
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
                'confidence': round(confidence, 4)
            }
            
            ball_data.append(ball_info)
            
            print(f"\nBall detected:")
            print(f"  Position: ({x_center}, {y_center})")
            print(f"  Height from ground: {height_from_ground} pixels")
            print(f"  Distance from bottom: {distance_from_bottom} pixels")
            print(f"  Confidence: {confidence:.2%}")
    
    return ball_data

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
    
    print(f"\n✓ Saved ball data to: {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_ball_data.py <image_path> [output_csv]")
        print("Example: python extract_ball_data.py input_videos/screenshot.png")
        sys.exit(1)
    
    input_image = sys.argv[1]
    
    # Generate output CSV path if not provided
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]
    else:
        base_name = os.path.splitext(os.path.basename(input_image))[0]
        output_dir = os.path.dirname(input_image) or "output_videos"
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, f"{base_name}_ball_data.csv")
    
    try:
        # Extract ball data
        ball_data = extract_ball_data(input_image)
        
        if len(ball_data) == 0:
            print("\n⚠ No ball detected in image")
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

