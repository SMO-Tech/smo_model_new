"""
Pass Visualizer Module

Handles all pass visualization:
- Pass line drawing
- Team color consistency
- Fading effects
- No speculative or low-confidence passes drawn
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from .pass_event import PassEvent, PassLifecycleStage
from .ball_tracker import StrictBallTracker, BallState


class PassVisualizer:
    """
    Visualizes confirmed passes on video frames.
    
    Rules:
    - Only draw CONFIRMED passes (never speculative)
    - Use locked team colors (from team_map, not live clustering)
    - Fade lines after configurable duration
    - No visual clutter
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            # Fading
            'fade_duration_frames': 90,  # 3 seconds at 30fps
            'min_opacity': 0.2,
            
            # Line style
            'line_thickness': 2,
            'arrow_length': 15,
            
            # Colors (BGR format)
            'team_colors': {
                0: (128, 0, 128),  # Purple for team 0
                1: (0, 0, 255),    # Red for team 1
            },
            'default_color': (255, 255, 255),  # White fallback
            
            # Frame rate
            'fps': 30.0,
        }
        
        if config:
            self.config.update(config)
    
    def draw_passes(self, frame: np.ndarray, 
                   passes: List[PassEvent],
                   player_frame_positions: Dict[int, Tuple[int, int]],
                   current_frame: int,
                   team_map: Optional[Dict[int, int]] = None) -> np.ndarray:
        """
        Draw confirmed passes on frame.
        
        Args:
            frame: Input video frame
            passes: List of PassEvent objects
            player_frame_positions: Dict of player_id -> (x, y) in frame coordinates
            current_frame: Current frame number
            team_map: Optional locked team assignments
            
        Returns:
            Annotated frame
        """
        if not passes:
            return frame
        
        annotated = frame.copy()
        
        for pass_event in passes:
            # Only draw CONFIRMED passes that have been emitted
            if pass_event.lifecycle_stage != PassLifecycleStage.CONFIRMED:
                continue
            
            if not pass_event.is_emitted:
                continue
            
            # Check if within fade window
            if pass_event.end_frame is None:
                continue
            
            age = current_frame - pass_event.end_frame
            if age > self.config['fade_duration_frames']:
                continue
            if age < 0:
                continue  # Pass hasn't happened yet
            
            # Get player positions
            from_id = pass_event.from_player_id
            to_id = pass_event.to_player_id
            
            if from_id not in player_frame_positions or to_id not in player_frame_positions:
                continue
            
            from_pos = player_frame_positions[from_id]
            to_pos = player_frame_positions[to_id]
            
            # Calculate opacity based on age
            fade_ratio = age / self.config['fade_duration_frames']
            opacity = max(self.config['min_opacity'], 1.0 - fade_ratio)
            
            # Get team color (from locked team_map if available)
            team_id = pass_event.team_id
            if team_map is not None and from_id in team_map:
                team_id = team_map[from_id]
            
            base_color = self.config['team_colors'].get(team_id, self.config['default_color'])
            
            # Apply opacity to color
            color = tuple(int(c * opacity) for c in base_color)
            
            # Draw line
            thickness = max(1, int(self.config['line_thickness'] * opacity))
            cv2.line(annotated,
                    (int(from_pos[0]), int(from_pos[1])),
                    (int(to_pos[0]), int(to_pos[1])),
                    color, thickness, cv2.LINE_AA)
            
            # Draw arrow head
            self._draw_arrow_head(annotated, from_pos, to_pos, color, thickness)
            
            # Optional: Draw confidence indicator (small dot at midpoint)
            if pass_event.confidence > 0.7:
                mid_x = int((from_pos[0] + to_pos[0]) / 2)
                mid_y = int((from_pos[1] + to_pos[1]) / 2)
                cv2.circle(annotated, (mid_x, mid_y), 3, color, -1)
        
        return annotated
    
    def draw_ball(self, frame: np.ndarray,
                  ball_tracker: StrictBallTracker,
                  current_frame: int,
                  ball_frame_position: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Draw ball ONLY when DETECTED (not predicted or lost).
        
        Args:
            frame: Input video frame
            ball_tracker: BallTracker instance
            current_frame: Current frame number
            ball_frame_position: Ball position in frame coordinates (x, y), or None
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        ball_obs = ball_tracker.get_current_ball()
        if ball_obs is None:
            return annotated
        
        # Only draw if DETECTED
        if ball_obs.state != BallState.DETECTED:
            return annotated
        
        if ball_frame_position is None:
            return annotated
        
        # Draw ball as small circle
        x, y = int(ball_frame_position[0]), int(ball_frame_position[1])
        cv2.circle(annotated, (x, y), 8, (0, 255, 255), -1)  # Yellow ball
        cv2.circle(annotated, (x, y), 8, (0, 0, 0), 2)  # Black outline
        
        return annotated
    
    def _draw_arrow_head(self, frame: np.ndarray, 
                        from_pos: Tuple[int, int],
                        to_pos: Tuple[int, int],
                        color: Tuple[int, int, int],
                        thickness: int):
        """Draw an arrow head at the receiving end."""
        # Calculate direction
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        length = np.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return
        
        # Normalize
        dx /= length
        dy /= length
        
        # Arrow head length
        arrow_len = self.config['arrow_length']
        
        # Arrow head points (30 degree angle)
        angle = 0.5  # ~30 degrees in radians
        
        # Left arrow point
        left_x = int(to_pos[0] - arrow_len * (dx * np.cos(angle) + dy * np.sin(angle)))
        left_y = int(to_pos[1] - arrow_len * (dy * np.cos(angle) - dx * np.sin(angle)))
        
        # Right arrow point
        right_x = int(to_pos[0] - arrow_len * (dx * np.cos(angle) - dy * np.sin(angle)))
        right_y = int(to_pos[1] - arrow_len * (dy * np.cos(angle) + dx * np.sin(angle)))
        
        # Draw arrow head lines
        cv2.line(frame, (int(to_pos[0]), int(to_pos[1])), (left_x, left_y), 
                color, thickness, cv2.LINE_AA)
        cv2.line(frame, (int(to_pos[0]), int(to_pos[1])), (right_x, right_y), 
                color, thickness, cv2.LINE_AA)
    
    def draw_player_boxes(self, frame: np.ndarray,
                         player_boxes: Dict[int, Tuple[int, int, int, int]],
                         team_map: Dict[int, int],
                         player_ids: Dict[int, int]) -> np.ndarray:
        """
        Draw player bounding boxes with consistent team colors.
        
        Args:
            frame: Input video frame
            player_boxes: Dict of player_id -> (x1, y1, x2, y2)
            team_map: Locked team assignments
            player_ids: Dict of player_id -> tracking ID for display
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for player_id, box in player_boxes.items():
            x1, y1, x2, y2 = box
            
            # Get team color from locked map
            team_id = team_map.get(player_id, 0)
            color = self.config['team_colors'].get(team_id, self.config['default_color'])
            
            # Draw box
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), 
                         color, 2)
            
            # Draw ID label
            display_id = player_ids.get(player_id, player_id)
            label = str(display_id)
            
            # Label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(annotated,
                         (int(x1), int(y1) - text_height - 5),
                         (int(x1) + text_width + 4, int(y1)),
                         color, -1)
            
            # Label text
            cv2.putText(annotated, label,
                       (int(x1) + 2, int(y1) - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def get_pass_stats_overlay(self, passes: List[PassEvent], 
                               current_frame: int,
                               fps: float) -> str:
        """
        Generate a stats string for overlay.
        
        Args:
            passes: List of confirmed passes
            current_frame: Current frame
            fps: Video FPS
            
        Returns:
            Stats string
        """
        team_0_passes = len([p for p in passes if p.team_id == 0])
        team_1_passes = len([p for p in passes if p.team_id == 1])
        
        time_seconds = current_frame / fps
        minutes = int(time_seconds // 60)
        seconds = int(time_seconds % 60)
        
        return f"Purple: {team_0_passes} | Red: {team_1_passes} | Time: {minutes:02d}:{seconds:02d}"

