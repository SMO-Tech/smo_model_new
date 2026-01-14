"""
Corner Detection Module

Detects corners by analyzing:
1. Ball out-of-bounds detection near corner areas
2. Corner position detection (ball/player near corner flag area)
3. Subsequent kick detection from corner position
4. Ball trajectory analysis (ball enters field from corner)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .pass_event import PassEvent


class CornerSide(Enum):
    """Which corner of the field."""
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


@dataclass
class CornerEvent:
    """Represents a corner kick event."""
    event_id: str
    kicker_id: int
    team_id: int
    start_frame: int
    end_frame: int
    start_position: List[float]
    end_position: List[float]
    corner_side: CornerSide
    distance_pixels: float = 0.0
    duration_seconds: float = 0.0
    speed_pixels_per_second: float = 0.0
    confidence: float = 0.0
    
    @property
    def start_time(self) -> float:
        """Get start time in seconds."""
        return self.start_frame / 30.0  # Assuming 30fps
    
    @property
    def end_time(self) -> float:
        """Get end time in seconds."""
        return self.end_frame / 30.0


class CornerDetector:
    """
    Detects corner kicks by analyzing ball out-of-bounds and corner positions.
    
    Key indicators:
    1. Ball goes out of bounds near corner (within 200px of corner keypoint)
    2. Ball velocity toward corner before going out
    3. Ball stationary or near corner position
    4. Player near corner position (within 200px)
    5. Ball re-enters field from corner (trajectory from corner into field)
    6. Ball speed > 100 px/s
    7. Ball enters within 5 seconds of going out
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize corner detector."""
        self.config = {
            # Corner proximity
            'corner_proximity_threshold': 200.0,  # Pixels - how close to corner to consider
            'out_of_bounds_threshold': 50.0,  # Pixels - how far outside field boundary
            
            # Timing
            'max_time_between_out_and_kick': 150,  # Frames (5 seconds at 30fps)
            'min_kick_speed': 100.0,  # px/s - minimum speed for corner kick
            
            # Player proximity
            'player_proximity_radius': 200.0,  # Pixels - player must be within this of corner
            
            # Ball re-entry detection
            're_entry_window': 150,  # Frames to look for ball re-entering field
            'min_re_entry_distance': 100.0,  # Pixels - ball must move at least this far into field
            
            # Frame rate
            'fps': 30.0,
            
            # Video resolution (will be set automatically)
            'frame_width': 1920,
            'frame_height': 1080,
        }
        
        if config:
            self.config.update(config)
        
        # Field corners (from keypoint detection)
        self.field_corners: Optional[Dict[str, Tuple[float, float]]] = None
        self.field_keypoints: Optional[np.ndarray] = None
        
        # Ball position history
        self.ball_position_history: List[Tuple[int, np.ndarray]] = []
        self.max_history_frames = 200
        
        # Out-of-bounds events (ball went out near corner)
        self.out_of_bounds_events: List[Dict] = []
        
        # Corner candidates (ball near corner, waiting for kick)
        self.corner_candidates: List[Dict] = []
        
        # Detected corners
        self.detected_corners: List[CornerEvent] = []
        
        # Statistics
        self.stats = {
            'corners_detected': 0,
            'out_of_bounds_detected': 0,
            'candidates_created': 0,
            'candidates_rejected': 0,
        }
    
    def set_field_keypoints(self, keypoints: np.ndarray, field_corners: Optional[Dict[str, Tuple[float, float]]] = None):
        """Set field keypoints and corners for position analysis."""
        self.field_keypoints = keypoints
        self.field_corners = field_corners
    
    def initialize_field_dimensions(self, frame_width: int, frame_height: int):
        """Initialize field dimensions for adaptive thresholds."""
        self.config['frame_width'] = frame_width
        self.config['frame_height'] = frame_height
        
        # Adaptive thresholds
        self.config['corner_proximity_threshold'] = min(frame_width, frame_height) * 0.1  # 10% of smaller dimension
        self.config['out_of_bounds_threshold'] = min(frame_width, frame_height) * 0.02  # 2% of smaller dimension
    
    def _is_ball_out_of_bounds(self, ball_position: np.ndarray) -> Tuple[bool, Optional[CornerSide]]:
        """
        Check if ball is out of bounds and near a corner.
        
        Returns:
            (is_out_of_bounds, corner_side)
        """
        if self.field_corners is None:
            # Fallback: use frame-based detection
            frame_width = self.config['frame_width']
            frame_height = self.config['frame_height']
            
            # Check if ball is near frame corners
            corner_threshold = self.config['corner_proximity_threshold']
            out_of_bounds_threshold = self.config['out_of_bounds_threshold']
            
            # Top-left corner
            if (ball_position[0] < -out_of_bounds_threshold or ball_position[1] < -out_of_bounds_threshold):
                if (abs(ball_position[0]) < corner_threshold and abs(ball_position[1]) < corner_threshold):
                    return True, CornerSide.TOP_LEFT
            
            # Top-right corner
            if (ball_position[0] > frame_width + out_of_bounds_threshold or ball_position[1] < -out_of_bounds_threshold):
                if (abs(ball_position[0] - frame_width) < corner_threshold and abs(ball_position[1]) < corner_threshold):
                    return True, CornerSide.TOP_RIGHT
            
            # Bottom-left corner
            if (ball_position[0] < -out_of_bounds_threshold or ball_position[1] > frame_height + out_of_bounds_threshold):
                if (abs(ball_position[0]) < corner_threshold and abs(ball_position[1] - frame_height) < corner_threshold):
                    return True, CornerSide.BOTTOM_LEFT
            
            # Bottom-right corner
            if (ball_position[0] > frame_width + out_of_bounds_threshold or ball_position[1] > frame_height + out_of_bounds_threshold):
                if (abs(ball_position[0] - frame_width) < corner_threshold and abs(ball_position[1] - frame_height) < corner_threshold):
                    return True, CornerSide.BOTTOM_RIGHT
            
            return False, None
        
        # Use keypoint-based detection
        corner_threshold = self.config['corner_proximity_threshold']
        out_of_bounds_threshold = self.config['out_of_bounds_threshold']
        
        # Check each corner
        for corner_name, corner_pos in self.field_corners.items():
            corner_side = self._corner_name_to_enum(corner_name)
            if corner_side is None:
                continue
            
            distance = np.linalg.norm(ball_position - np.array(corner_pos))
            
            # Check if ball is out of bounds (beyond field boundary)
            # For now, use simple heuristic: if ball is far from corner but in wrong direction
            # This is simplified - ideally would check against field boundary
            
            # If ball is very close to corner, it might be out of bounds
            if distance < corner_threshold:
                # Check if ball is moving toward corner (from history)
                if len(self.ball_position_history) >= 2:
                    prev_pos = self.ball_position_history[-2][1]
                    direction = ball_position - prev_pos
                    to_corner = np.array(corner_pos) - prev_pos
                    
                    # Normalize
                    if np.linalg.norm(direction) > 1e-6 and np.linalg.norm(to_corner) > 1e-6:
                        direction_unit = direction / np.linalg.norm(direction)
                        to_corner_unit = to_corner / np.linalg.norm(to_corner)
                        similarity = np.dot(direction_unit, to_corner_unit)
                        
                        # If ball is moving toward corner and very close, it's likely out of bounds
                        if similarity > 0.5 and distance < corner_threshold * 0.5:
                            return True, corner_side
        
        return False, None
    
    def _corner_name_to_enum(self, corner_name: str) -> Optional[CornerSide]:
        """Convert corner name string to CornerSide enum."""
        mapping = {
            'top_left': CornerSide.TOP_LEFT,
            'top_right': CornerSide.TOP_RIGHT,
            'bottom_left': CornerSide.BOTTOM_LEFT,
            'bottom_right': CornerSide.BOTTOM_RIGHT,
        }
        return mapping.get(corner_name)
    
    def _get_corner_position(self, corner_side: CornerSide) -> Optional[np.ndarray]:
        """Get corner position from keypoints or estimate."""
        if self.field_corners:
            corner_name = corner_side.value
            if corner_name in self.field_corners:
                return np.array(self.field_corners[corner_name])
        
        # Fallback: estimate from frame
        frame_width = self.config['frame_width']
        frame_height = self.config['frame_height']
        
        if corner_side == CornerSide.TOP_LEFT:
            return np.array([0.0, 0.0])
        elif corner_side == CornerSide.TOP_RIGHT:
            return np.array([float(frame_width), 0.0])
        elif corner_side == CornerSide.BOTTOM_LEFT:
            return np.array([0.0, float(frame_height)])
        elif corner_side == CornerSide.BOTTOM_RIGHT:
            return np.array([float(frame_width), float(frame_height)])
        
        return None
    
    def _is_ball_near_corner(self, ball_position: np.ndarray, corner_side: CornerSide) -> bool:
        """Check if ball is near a specific corner."""
        corner_pos = self._get_corner_position(corner_side)
        if corner_pos is None:
            return False
        
        distance = np.linalg.norm(ball_position - corner_pos)
        return distance < self.config['corner_proximity_threshold']
    
    def _is_ball_re_entering_field(self, ball_position: np.ndarray, corner_side: CornerSide,
                                   out_of_bounds_frame: int, current_frame: int) -> Tuple[bool, float]:
        """
        Check if ball is re-entering field from corner.
        
        Returns:
            (is_re_entering, distance_into_field)
        """
        if current_frame - out_of_bounds_frame > self.config['re_entry_window']:
            return False, 0.0
        
        corner_pos = self._get_corner_position(corner_side)
        if corner_pos is None:
            return False, 0.0
        
        # Check if ball moved away from corner (into field)
        distance_from_corner = np.linalg.norm(ball_position - corner_pos)
        
        # Check if ball is moving into field (away from corner)
        if len(self.ball_position_history) >= 2:
            prev_pos = self.ball_position_history[-2][1]
            prev_distance = np.linalg.norm(prev_pos - corner_pos)
            
            # Ball moved away from corner
            if distance_from_corner > prev_distance:
                distance_into_field = distance_from_corner - prev_distance
                if distance_into_field >= self.config['min_re_entry_distance']:
                    return True, distance_into_field
        
        return False, 0.0
    
    def process_frame(self, frame: int, ball_position: Optional[np.ndarray],
                     player_positions: Dict[int, np.ndarray],
                     player_teams: Dict[int, int]) -> List[CornerEvent]:
        """
        Process a single frame for corner detection.
        
        Args:
            frame: Current frame number
            ball_position: Ball position [x, y] or None
            player_positions: Dict of player_id -> position [x, y]
            player_teams: Dict of player_id -> team_id
            
        Returns:
            List of newly detected corners
        """
        new_corners = []
        
        if ball_position is not None:
            # Add to history
            self.ball_position_history.append((frame, ball_position.copy()))
            
            # Keep history at reasonable size
            if len(self.ball_position_history) > self.max_history_frames:
                self.ball_position_history.pop(0)
            
            # Check if ball is out of bounds near corner
            is_out_of_bounds, corner_side = self._is_ball_out_of_bounds(ball_position)
            
            if is_out_of_bounds and corner_side is not None:
                # Ball went out near corner - create out-of-bounds event
                out_of_bounds_event = {
                    'frame': frame,
                    'ball_position': ball_position.copy(),
                    'corner_side': corner_side,
                }
                self.out_of_bounds_events.append(out_of_bounds_event)
                self.stats['out_of_bounds_detected'] += 1
                
                # Create corner candidate
                corner_pos = self._get_corner_position(corner_side)
                if corner_pos is not None:
                    # Find closest player to corner
                    closest_player = None
                    closest_distance = float('inf')
                    for player_id, player_pos in player_positions.items():
                        if player_pos is None:
                            continue
                        distance = np.linalg.norm(corner_pos - player_pos)
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_player = player_id
                    
                    if closest_player is not None and closest_distance < self.config['player_proximity_radius']:
                        team_id = player_teams.get(closest_player, 0)
                        
                        candidate = {
                            'kicker_id': closest_player,
                            'team_id': team_id,
                            'out_of_bounds_frame': frame,
                            'corner_side': corner_side,
                            'corner_position': corner_pos.copy(),
                            'start_frame': frame,  # Will be updated when kick detected
                        }
                        self.corner_candidates.append(candidate)
                        self.stats['candidates_created'] += 1
            else:
                # Ball is in bounds - check if it's re-entering from corner
                for candidate in list(self.corner_candidates):
                    # Check if ball is re-entering field
                    is_re_entering, distance_into_field = self._is_ball_re_entering_field(
                        ball_position, candidate['corner_side'],
                        candidate['out_of_bounds_frame'], frame
                    )
                    
                    if is_re_entering:
                        # Check ball speed
                        if len(self.ball_position_history) >= 2:
                            last_pos = self.ball_position_history[-1][1]
                            prev_pos = self.ball_position_history[-2][1]
                            velocity = np.linalg.norm(last_pos - prev_pos)
                            speed_px_per_sec = velocity * self.config['fps']
                            
                            if speed_px_per_sec >= self.config['min_kick_speed']:
                                # Corner kick detected
                                corner = self._create_corner_event(candidate, frame, ball_position, speed_px_per_sec)
                                if corner is not None:
                                    new_corners.append(corner)
                                    self.detected_corners.append(corner)
                                    self.stats['corners_detected'] += 1
                                    
                                    # Remove candidate
                                    self.corner_candidates.remove(candidate)
                    elif frame - candidate['out_of_bounds_frame'] > self.config['max_time_between_out_and_kick']:
                        # Too long - reject candidate
                        self.corner_candidates.remove(candidate)
                        self.stats['candidates_rejected'] += 1
        else:
            # No ball - clear old candidates if too long
            for candidate in list(self.corner_candidates):
                if frame - candidate['out_of_bounds_frame'] > self.config['max_time_between_out_and_kick']:
                    self.corner_candidates.remove(candidate)
                    self.stats['candidates_rejected'] += 1
        
        return new_corners
    
    def _create_corner_event(self, candidate: Dict, end_frame: int, end_position: np.ndarray, speed: float) -> Optional[CornerEvent]:
        """Create a CornerEvent from a validated candidate."""
        start_pos = candidate['corner_position']
        distance = np.linalg.norm(end_position - start_pos)
        duration_frames = end_frame - candidate['start_frame']
        duration_seconds = duration_frames / self.config['fps']
        
        # Calculate confidence
        confidence = self._calculate_confidence(candidate, distance, speed)
        
        # Only create if confidence is high enough
        if confidence < 0.5:
            return None
        
        corner_event = CornerEvent(
            event_id=f"corner_{candidate['start_frame']}",
            kicker_id=candidate['kicker_id'],
            team_id=candidate['team_id'],
            start_frame=candidate['start_frame'],
            end_frame=end_frame,
            start_position=start_pos.tolist(),
            end_position=end_position.tolist(),
            corner_side=candidate['corner_side'],
            distance_pixels=distance,
            duration_seconds=duration_seconds,
            speed_pixels_per_second=speed,
            confidence=confidence
        )
        
        return corner_event
    
    def _calculate_confidence(self, candidate: Dict, distance: float, speed: float) -> float:
        """Calculate confidence score for corner event."""
        confidence = 0.0
        
        # Base confidence for corner detection
        confidence += 0.3
        
        # Speed confidence
        if speed >= 150.0:  # Fast corner
            confidence += 0.3
        elif speed >= 100.0:  # Medium speed
            confidence += 0.2
        
        # Distance confidence (corners usually travel some distance)
        if distance >= 200.0:
            confidence += 0.2
        elif distance >= 100.0:
            confidence += 0.1
        
        # Timing confidence (ball re-entered quickly)
        duration = candidate.get('duration_seconds', 0.0)
        if duration < 3.0:  # Re-entered within 3 seconds
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def get_detected_corners(self) -> List[CornerEvent]:
        """Get all detected corners."""
        return self.detected_corners.copy()
    
    def get_stats(self) -> Dict:
        """Get corner detection statistics."""
        return self.stats.copy()
