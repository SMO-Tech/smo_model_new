#Import All the Required Libraries
import cv2
import os
import numpy as np
import pickle
from ultralytics import YOLO
import supervision as sv
from utils import get_center_of_bbox, get_bbox_width
from .ball_physics import BallPhysicsTracker

class Tracker:
    def __init__(self, model_path, fps=30.0, enable_path_tracking=True, device=None):
        # Determine device - try GPU first, fallback to CPU
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = 0  # Use GPU 0
                print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                print("⚠ GPU not available, using CPU")
        
        # Load model - YOLO will automatically use GPU if device=0
        self.model = YOLO(model_path)
        if device != 'cpu':
            # Force GPU usage
            self.model.to(device)
        self.device = device
        self.tracker = sv.ByteTrack()
        self.fps = fps
        self.ball_physics = BallPhysicsTracker(fps=fps)
        self.enable_path_tracking = enable_path_tracking
        
        # Initialize path tracker if enabled
        if enable_path_tracking:
            from utils.path_tracker import BallPathTracker
            self.ball_path = BallPathTracker(max_history=60, fade_alpha=True)
        else:
            self.ball_path = None
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            # Explicitly use device for prediction
            detections_batch = self.model.predict(
                frames[i: i + batch_size], 
                conf=0.1,
                device=self.device  # Force device usage
            )
            detections += detections_batch
        return detections
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k, v in cls_names.items()}

            #Convert to Supervision Detection Format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Convert GoalKeeper to Player Object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Check if ball is detected in this frame
            ball_detected = False
            ball_bbox = None
            for i, cls_id in enumerate(detection_supervision.class_id):
                if cls_id == cls_names_inv['ball']:
                    ball_detected = True
                    # Update physics tracker with detected ball
                    ball_bbox = detection_supervision.xyxy[i].tolist()
                    self.ball_physics.update(ball_bbox, frame_num)
                    break

            # If ball NOT detected, use physics prediction to help ByteTrack re-acquire
            if not ball_detected and self.ball_physics.is_tracking():
                predicted_pos = self.ball_physics.predict(frame_num)
                
                if predicted_pos is not None:
                    # Create synthetic detection box at predicted position
                    x, y = predicted_pos
                    bbox_size = 20  # Approximate ball size
                    predicted_bbox = np.array([
                        [x - bbox_size, y - bbox_size, x + bbox_size, y + bbox_size]
                    ], dtype=np.float32)
                    
                    # Get ball class ID
                    ball_class_id = cls_names_inv['ball']
                    
                    # Create new Detections object with predicted ball added
                    # This ensures all arrays stay in sync
                    if len(detection_supervision.xyxy) == 0:
                        # If no detections, create new detection arrays
                        new_xyxy = predicted_bbox
                        new_confidence = np.array([0.5], dtype=np.float32)
                        new_class_id = np.array([ball_class_id], dtype=np.int64)
                    else:
                        # Append to existing detections
                        new_xyxy = np.vstack([detection_supervision.xyxy, predicted_bbox])
                        new_confidence = np.append(detection_supervision.confidence, np.array([0.5], dtype=np.float32))
                        new_class_id = np.append(detection_supervision.class_id, np.array([ball_class_id], dtype=np.int64))
                    
                    # Create new Detections object to ensure all arrays are in sync
                    detection_supervision = sv.Detections(
                        xyxy=new_xyxy,
                        confidence=new_confidence,
                        class_id=new_class_id
                    )

            #Track Objects (now includes predicted ball if ball was missing)
            try:
                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            except (IndexError, ValueError) as e:
                # If tracking fails due to array mismatch, use original detections without predicted ball
                print(f"Warning: Tracking error at frame {frame_num}, using original detections: {e}")
                # Recreate detections without predicted ball
                detection_supervision = sv.Detections.from_ultralytics(detection)
                # Convert GoalKeeper to Player Object again
                for object_ind, class_id in enumerate(detection_supervision.class_id):
                    if cls_names[class_id] == "goalkeeper":
                        detection_supervision.class_id[object_ind] = cls_names_inv["player"]
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

            # Extract ball from tracked detections (includes both real and predicted)
            ball_tracked = False
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['ball']:
                    # Check if this was a predicted position (not originally detected)
                    is_predicted = not ball_detected
                    tracks["ball"][frame_num][track_id] = {
                        "bbox": bbox,
                        "predicted": is_predicted
                    }
                    ball_tracked = True
                    
                    # Update path tracker
                    if self.ball_path is not None:
                        color = (0, 165, 255) if is_predicted else (0, 255, 0)
                        self.ball_path.add_point(bbox, frame_num, color, is_predicted)
                    
                    # If it was predicted and ByteTrack accepted it, update physics with tracked position
                    if is_predicted:
                        self.ball_physics.update(bbox, frame_num)
            
            # If ball was detected but not tracked (shouldn't happen, but handle it)
            if ball_detected and not ball_tracked and ball_bbox is not None:
                tracks["ball"][frame_num][1] = {
                    "bbox": ball_bbox,
                    "predicted": False
                }

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def _apply_physics_prediction(self, tracks, total_frames):
        """
        Fill gaps in ball tracking using physics-based trajectory prediction
        """
        # Reset physics tracker
        self.ball_physics.reset()
        
        # First pass: build physics model from detected balls
        for frame_num in range(total_frames):
            ball_dict = tracks["ball"][frame_num]
            if len(ball_dict) > 0:
                # Ball detected - update physics tracker
                for track_id, ball in ball_dict.items():
                    bbox = ball["bbox"]
                    self.ball_physics.update(bbox, frame_num)
        
        # Second pass: fill gaps with physics predictions
        for frame_num in range(total_frames):
            ball_dict = tracks["ball"][frame_num]
            
            if len(ball_dict) == 0:
                # Ball not detected - try to predict
                predicted_pos = self.ball_physics.predict(frame_num)
                
                if predicted_pos is not None:
                    # Create bbox around predicted position
                    x, y = predicted_pos
                    bbox_size = 20  # Approximate ball size in pixels
                    bbox = [
                        x - bbox_size,  # x1
                        y - bbox_size,  # y1
                        x + bbox_size,  # x2
                        y + bbox_size   # y2
                    ]
                    # Add predicted ball position
                    tracks["ball"][frame_num][1] = {"bbox": bbox, "predicted": True}
            else:
                # Ball detected - update physics tracker for next prediction
                for track_id, ball in ball_dict.items():
                    bbox = ball["bbox"]
                    self.ball_physics.update(bbox, frame_num)
        
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
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

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
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

    def draw_traingle(self, frame, bbox, color):
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

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0,0,255), track_id)

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball path first (so it appears behind the ball)
            if self.ball_path is not None:
                frame = self.ball_path.draw_path(frame, thickness=3, draw_arrows=True, arrow_frequency=15)

            # Draw ball
            for track_id, ball in ball_dict.items():
                # Use different color for predicted positions
                if ball.get("predicted", False):
                    # Orange/yellow for predicted positions
                    frame = self.draw_traingle(frame, ball["bbox"], (0, 165, 255))
                else:
                    # Green for detected positions
                    frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            output_video_frames.append(frame)

        return output_video_frames