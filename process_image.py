#!/usr/bin/env python3
"""
Process a single image with football detection and tracking
"""

import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from utils import get_center_of_bbox, get_bbox_width

def draw_ellipse(frame, bbox, color, track_id=None):
    """Draw ellipse annotation for players/referees"""
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width), int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4
    )

    if track_id is not None:
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        cv2.rectangle(frame,
                      (int(x1_rect), int(y1_rect)),
                      (int(x2_rect), int(y2_rect)),
                      color,
                      cv2.FILLED)

        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10

        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text), int(y1_rect + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return frame

def draw_triangle(frame, bbox, color):
    """Draw triangle annotation for ball"""
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x, y],
        [x - 10, y - 20],
        [x + 10, y - 20],
    ])
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    return frame

def process_image(image_path, model_path="models/best.pt"):
    """
    Process a single image and return annotated image
    
    Args:
        image_path: Path to input image
        model_path: Path to YOLO model
        
    Returns:
        Annotated image (numpy array)
    """
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    print(f"Processing image: {image_path}")
    print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Load YOLO model
    print("Loading model...")
    model = YOLO(model_path)
    
    # Run detection
    print("Running detection...")
    results = model.predict(frame, conf=0.1, verbose=False)
    
    # Get detections
    detection = results[0]
    cls_names = detection.names
    cls_names_inv = {v: k for k, v in cls_names.items()}
    
    # Convert to Supervision format
    detection_supervision = sv.Detections.from_ultralytics(detection)
    
    # Convert goalkeeper to player
    for object_ind, class_id in enumerate(detection_supervision.class_id):
        if cls_names[class_id] == "goalkeeper":
            detection_supervision.class_id[object_ind] = cls_names_inv["player"]
    
    # Track objects (for ID assignment)
    tracker = sv.ByteTrack()
    detection_with_tracks = tracker.update_with_detections(detection_supervision)
    
    # Create annotated frame
    annotated_frame = frame.copy()
    
    # Draw players and referees (with tracking IDs)
    for frame_detection in detection_with_tracks:
        bbox = frame_detection[0].tolist()
        cls_id = frame_detection[3]
        track_id = frame_detection[4]
        
        if cls_id == cls_names_inv['player']:
            annotated_frame = draw_ellipse(annotated_frame, bbox, (0, 0, 255), track_id)
        elif cls_id == cls_names_inv['referee']:
            annotated_frame = draw_ellipse(annotated_frame, bbox, (0, 255, 255))
    
    # Draw ball (no tracking ID needed for single image)
    for frame_detection in detection_supervision:
        bbox = frame_detection[0].tolist()
        cls_id = frame_detection[3]
        
        if cls_id == cls_names_inv['ball']:
            annotated_frame = draw_triangle(annotated_frame, bbox, (0, 255, 0))
    
    # Count detections
    players = sum(1 for d in detection_with_tracks if d[3] == cls_names_inv['player'])
    referees = sum(1 for d in detection_with_tracks if d[3] == cls_names_inv['referee'])
    balls = sum(1 for d in detection_supervision if d[3] == cls_names_inv['ball'])
    
    print(f"Detected: {players} players, {referees} referees, {balls} ball(s)")
    
    return annotated_frame

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_image.py <image_path> [output_path]")
        print("Example: python process_image.py input_videos/screenshot.png")
        sys.exit(1)
    
    input_image = sys.argv[1]
    
    # Generate output path if not provided
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        base_name = os.path.splitext(os.path.basename(input_image))[0]
        output_dir = os.path.dirname(input_image) or "output_videos"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
    
    try:
        # Process image
        annotated_image = process_image(input_image)
        
        # Save result
        cv2.imwrite(output_path, annotated_image)
        print(f"\n✓ Saved annotated image: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

