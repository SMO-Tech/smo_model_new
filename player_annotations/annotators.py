import sys
from pathlib import Path
from typing import List, Dict
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
import supervision as sv
import cv2


class AnnotatorManager:
    """
    Manager class for all annotation functionality.
    Provides a unified interface for annotating players, ball, referees, and keypoints.
    """

    def __init__(self, edges=None):
        """Initialize all annotation tools."""
        # Basic annotators
        self.ellipse_annotator = sv.EllipseAnnotator()
        self.triangle_annotator = sv.TriangleAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.box_annotator = sv.BoxAnnotator()

        # Keypoint annotators (optional - only used if keypoint annotation is needed)
        # Note: VertexAnnotator and EdgeAnnotator may not be available in all supervision versions
        self.vertex_annotator = None
        self.edge_annotator = None
        try:
            if hasattr(sv, 'VertexAnnotator'):
                self.vertex_annotator = sv.VertexAnnotator(color=sv.Color.GREEN, radius=8)
            if hasattr(sv, 'EdgeAnnotator'):
                self.edge_annotator = sv.EdgeAnnotator(color=sv.Color.BLUE, thickness=3, edges=edges)
        except AttributeError:
            # Keypoint annotators not available - will use manual drawing in annotate_keypoints
            pass

    def annotate_players(self, frame: np.ndarray, player_detections: sv.Detections) -> np.ndarray:
        """
        Annotate only players on the frame.

        Args:
            frame: Input video frame
            player_detections: Player detection results

        Returns:
            Annotated frame with player detections visualized
        """
        if player_detections is None or len(player_detections.xyxy) == 0:
            return frame

        if player_detections.tracker_id is None:
            player_detections.tracker_id = np.arange(len(player_detections.xyxy))
        player_labels = [f'#{tracker_id}' for tracker_id in player_detections.tracker_id]
        frame = self.ellipse_annotator.annotate(frame, player_detections)
        frame = self.label_annotator.annotate(frame, detections=player_detections, labels=player_labels)

        return frame

    def annotate_passes(self, frame: np.ndarray, passes: List, player_positions: Dict[int, np.ndarray], 
                       fade_frames: int = 90, current_frame: int = None) -> np.ndarray:
        """
        Annotate pass lines between players on frame.
        
        Only draws accepted passes (PassEvent objects that passed validation).
        Passes fade out after fade_frames.
        
        Args:
            frame: Input video frame
            passes: List of PassEvent objects to draw (only accepted passes)
            player_positions: Dictionary mapping player_id to [x, y] frame coordinates
            fade_frames: Number of frames to keep pass lines visible (default 90 = 3 seconds at 30fps)
            current_frame: Current frame index (if None, uses max end_frame from passes)
            
        Returns:
            Annotated frame with pass lines
        """
        if not passes:
            return frame
        
        annotated_frame = frame.copy()
        # Use provided current_frame or infer from passes
        if current_frame is None:
            current_frame = max([p.end_frame for p in passes]) if passes else 0
        
        for pass_event in passes:
            # Only draw passes that are recent (within fade_frames after end)
            age = current_frame - pass_event.end_frame
            if age < 0 or age > fade_frames:
                continue
            
            from_id = pass_event.from_player_id
            to_id = pass_event.to_player_id
            
            if from_id not in player_positions or to_id not in player_positions:
                continue
            
            # Get player positions in frame coordinates
            from_pos = player_positions[from_id]
            to_pos = player_positions[to_id]
            
            # Calculate fade alpha based on age
            age = current_frame - pass_event.end_frame
            alpha = max(0.3, 1.0 - (age / fade_frames))
            
            # Color based on team (purple for team 0, red for team 1)
            if pass_event.team_id == 0:
                color = (128, 0, 128)  # Purple
            else:
                color = (0, 0, 255)  # Red
            
            # Adjust color intensity based on confidence and fade
            color_intensity = int(alpha * 255)
            if pass_event.team_id == 0:
                line_color = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
            else:
                line_color = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
            
            # Draw pass line
            thickness = max(1, int(2 * alpha))
            cv2.line(annotated_frame, 
                    (int(from_pos[0]), int(from_pos[1])),
                    (int(to_pos[0]), int(to_pos[1])),
                    line_color, thickness)
            
            # Draw arrow head
            angle = np.arctan2(to_pos[1] - from_pos[1], to_pos[0] - from_pos[0])
            arrow_length = 10
            arrow_x = int(to_pos[0] - arrow_length * np.cos(angle))
            arrow_y = int(to_pos[1] - arrow_length * np.sin(angle))
            cv2.line(annotated_frame,
                    (int(to_pos[0]), int(to_pos[1])),
                    (arrow_x, arrow_y),
                    line_color, thickness)
        
        return annotated_frame

    def annotate_referees(self, frame: np.ndarray, referee_detections: sv.Detections) -> np.ndarray:
        """
        Annotate referee detections on frame.

        Args:
            frame: Input video frame
            referee_detections: Referee detection results

        Returns:
            Annotated frame with referee detections
        """
        if referee_detections is not None and len(referee_detections.xyxy) > 0:
            if referee_detections.tracker_id is None:
                referee_detections.tracker_id = np.arange(len(referee_detections.xyxy))
            return self.ellipse_annotator.annotate(frame, referee_detections)
        return frame

    def annotate_ball(self, frame: np.ndarray, ball_detections: sv.Detections) -> np.ndarray:
        """
        Annotate ball detections on frame using triangle (pointing upward) with tracker ID.

        Args:
            frame: Input video frame
            ball_detections: Ball detection results

        Returns:
            Annotated frame with ball triangle annotations and tracker ID
        """
        if ball_detections is None or len(ball_detections.xyxy) == 0:
            return frame
        
        # Ensure tracker_id exists - ALWAYS use 0 for ball (since there's only one ball)
        if ball_detections.tracker_id is None:
            ball_detections.tracker_id = np.array([0] * len(ball_detections.xyxy))
        else:
            # Force all tracker_ids to 0 (in case old tracks have different IDs)
            ball_detections.tracker_id = np.array([0] * len(ball_detections.xyxy))
        
        # Draw bounding box for ball (like players)
        annotated_frame = self.box_annotator.annotate(frame, ball_detections)
        
        # Add tracker ID label (like players)
        ball_labels = [f'Ball #{tracker_id}' for tracker_id in ball_detections.tracker_id]
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections=ball_detections, labels=ball_labels)
        
        return annotated_frame

    def annotate_all(self, frame: np.ndarray, player_detections, ball_detections, referee_detections) -> np.ndarray:
        """
        Annotate players, ball, and referees on the frame.

        Args:
            frame: Input video frame
            player_detections: Player detection results
            ball_detections: Ball detection results (drawn as triangle)
            referee_detections: Referee detection results

        Returns:
            Annotated frame with all detections visualized
        """
        target_frame = frame.copy()

        # Annotate each type separately
        target_frame = self.annotate_players(target_frame, player_detections)
        target_frame = self.annotate_ball(target_frame, ball_detections)
        target_frame = self.annotate_referees(target_frame, referee_detections)

        return target_frame

    def annotate_bboxes(self, frame: np.ndarray, detections: sv.Detections, class_names: dict = None) -> np.ndarray:
        """
        Annotate frame with object detections bboxes.

        Args:
            frame: Input frame
            detections: Supervision Detections object
            class_names: Dictionary mapping class IDs to names

        Returns:
            Annotated frame with detection boxes and labels
        """
        if detections is None or len(detections.xyxy) == 0:
            return frame

        annotated_frame = frame.copy()

        # Annotate with boxes
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)

        # Add labels if class_names provided
        if class_names is not None:
            labels = []
            for class_id, conf in zip(detections.class_id, detections.confidence):
                class_name = class_names.get(class_id, f'Class {class_id}')
                labels.append(f"{class_name} {conf:.2f}")
            annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)

        return annotated_frame

    def annotate_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, confidence_threshold: float = 0.5,
                          draw_vertices: bool = True, draw_edges = None, draw_labels: bool = True,
                          KEYPOINT_CONNECTIONS=[], KEYPOINT_NAMES={}, 
                          KEYPOINT_COLOR=(0, 255, 0), CONNECTION_COLOR=(255, 0, 0), TEXT_COLOR=(255, 255, 255)) -> np.ndarray:
        """
        Annotate frame with detected keypoints using Vertex and Edge annotators.

        Args:
            frame: Input frame to annotate
            keypoints: Detected keypoints array with shape (N, 27, 3)
            confidence_threshold: Minimum confidence to draw keypoint
            draw_vertices: Whether to draw keypoint vertices
            draw_edges: Whether to draw connections between keypoints
            draw_labels: Whether to draw keypoint labels

        Returns:
            Annotated frame
        """

        # Draw keypoints and connections for each detection
        for kpts in keypoints:

            # Draw keypoint connections
            if draw_edges:
                for connection in KEYPOINT_CONNECTIONS:
                    pt1_idx, pt2_idx = connection
                    pt1 = kpts[pt1_idx]
                    pt2 = kpts[pt2_idx]

                    # Only draw if both points are visible
                    if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                        cv2.line(frame,
                            (int(pt1[0]), int(pt1[1])),
                            (int(pt2[0]), int(pt2[1])),
                            CONNECTION_COLOR, 2)

            # Draw keypoints
            for kpt_idx, kpt in enumerate(kpts):
                if kpt[2] > confidence_threshold:  # Check visibility
                    x, y = int(kpt[0]), int(kpt[1])

                    # Draw keypoint circle
                    cv2.circle(frame, (x, y), 5, KEYPOINT_COLOR, -1)

                    # Draw keypoint label
                    if draw_labels:
                        label = f"{kpt_idx}: {KEYPOINT_NAMES.get(kpt_idx, 'Unknown')}"
                        cv2.putText(frame, label, (x + 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)

        return frame

    def convert_tracks_to_detections(self, player_tracks, ball_tracks, referee_tracks, player_classids=None, ball_tracker_id=None):
        """
        Convert tracking data back to supervision detections format.

        Args:
            player_tracks: Player tracking data for a frame
            ball_tracks: Ball tracking data for a frame [x1, y1, x2, y2] or None
            referee_tracks: Referee tracking data for a frame
            player_classids: Player class ID data for a frame (optional)
            ball_tracker_id: Ball tracker ID for this frame (optional)

        Returns:
            Tuple of converted detection objects (player_detections, ball_detections, referee_detections)
        """
        # Get the player detections
        if player_tracks is not None:
            if player_classids is not None:
                # Use stored class IDs
                class_ids = [player_classids[tracker_id] for tracker_id in player_tracks.keys()]
            else:
                # Fall back to default class ID (0)
                class_ids = [0] * len(player_tracks)
            
            player_detections = sv.Detections(
                xyxy=np.array(list(player_tracks.values())),
                class_id=np.array(class_ids),
                tracker_id=np.array(list(player_tracks.keys()))
            )
        else:
            player_detections = None

        # Convert ball tracks to detections
        if ball_tracks is not None and len(ball_tracks) == 4 and ball_tracks[0] is not None:
            # ball_tracks is [x1, y1, x2, y2]
            # ALWAYS assign tracker_id 0 for ball (since there's only one ball)
            # Ignore any stored tracker_id from old tracks - force it to 0
            tracker_id_array = np.array([0])
            ball_detections = sv.Detections(
                xyxy=np.array([[ball_tracks[0], ball_tracks[1], ball_tracks[2], ball_tracks[3]]]),
                class_id=np.array([1]),  # Ball class ID
                confidence=np.array([1.0]),  # Default confidence
                tracker_id=tracker_id_array
            )
        else:
            ball_detections = None

        # Get the referee detections
        if referee_tracks is not None:
            referee_detections = sv.Detections(
                xyxy=np.array(list(referee_tracks.values())),
                class_id=np.array([3] * len(referee_tracks)),
                tracker_id=np.array(list(referee_tracks.keys()))
            )
        else:
            referee_detections = None

        return player_detections, ball_detections, referee_detections