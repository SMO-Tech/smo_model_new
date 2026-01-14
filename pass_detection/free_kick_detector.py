"""
Free Kick Detection Module

Detects free kicks by analyzing:
1. Ball stationary period (2-5 seconds)
2. Field position (near penalty area, defensive zones)
3. Kick pattern (ball moves away from stationary position)
4. Player proximity to ball before kick
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .pass_event import PassEvent


class FreeKickType(Enum):
    """Types of free kicks."""
    DIRECT = "direct"  # Direct free kick (can score directly)
    INDIRECT = "indirect"  # Indirect free kick (must touch another player first)


@dataclass
class FreeKickEvent:
    """Represents a free kick event."""
    event_id: str
    kicker_id: int
    team_id: int
    start_frame: int
    end_frame: int
    start_position: List[float]
    end_position: List[float]
    free_kick_type: FreeKickType
    field_zone: str  # e.g., "penalty_area", "defensive_half", "near_goal"
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


class FreeKickDetector:
    """
    Detects free kicks by analyzing ball stationary periods and field positions.
    
    Key indicators:
    1. Ball stationary for 2-5 seconds (velocity < 10 px/frame)
    2. Ball position in specific zones (penalty area, defensive half)
    3. Player near ball (within 150px) before kick
    4. Ball moves away with speed > 150 px/s after stationary period
    5. No player receiver initially (ball goes to open space or goal)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize free kick detector."""
        self.config = {
            # Stationary detection
            'min_stationary_frames': 60,  # 2 seconds at 30fps
            'max_stationary_frames': 150,  # 5 seconds at 30fps
            'stationary_velocity_threshold': 10.0,  # px/frame - ball is stationary if velocity < this
            
            # Field position zones (in pixels, will be adaptive)
            'penalty_area_proximity': 200.0,  # Distance to penalty area to consider "near"
            'defensive_half_threshold': 0.5,  # Fraction of field width (0.5 = center line)
            'near_goal_distance': 800.0,  # Pixels - within this distance of goal
            
            # Player proximity
            'player_proximity_radius': 150.0,  # Pixels - player must be within this of ball
            
            # Kick detection
            'min_kick_speed': 150.0,  # px/s - ball must move at least this fast after stationary
            'kick_detection_window': 30,  # Frames to look for kick after stationary period
            
            # Frame rate
            'fps': 30.0,
            
            # Video resolution (will be set automatically)
            'frame_width': 1920,
            'frame_height': 1080,
        }
        
        if config:
            self.config.update(config)
        
        # Ball position history for stationary detection
        self.ball_position_history: List[Tuple[int, np.ndarray]] = []
        self.max_history_frames = 200  # Keep ~6.7 seconds at 30fps
        
        # Free kick candidates (ball stationary periods)
        self.free_kick_candidates: List[Dict] = []
        
        # Detected free kicks
        self.detected_free_kicks: List[FreeKickEvent] = []
        
        # Field keypoints (will be set from keypoint detection)
        self.field_keypoints: Optional[np.ndarray] = None
        self.field_corners: Optional[Dict[str, Tuple[float, float]]] = None
        
        # Statistics
        self.stats = {
            'free_kicks_detected': 0,
            'candidates_created': 0,
            'candidates_rejected': 0,
        }
    
    def set_field_keypoints(self, keypoints: np.ndarray, field_corners: Optional[Dict[str, Tuple[float, float]]] = None):
        """Set field keypoints for position analysis."""
        self.field_keypoints = keypoints
        self.field_corners = field_corners
    
    def initialize_field_dimensions(self, frame_width: int, frame_height: int):
        """Initialize field dimensions for adaptive thresholds."""
        self.config['frame_width'] = frame_width
        self.config['frame_height'] = frame_height
        
        # Adaptive thresholds based on frame size
        self.config['penalty_area_proximity'] = frame_width * 0.1  # 10% of width
        self.config['near_goal_distance'] = frame_width * 0.4  # 40% of width
    
    def _is_ball_stationary(self, current_pos: np.ndarray, history: List[Tuple[int, np.ndarray]]) -> Tuple[bool, int]:
        """
        Check if ball is stationary based on recent history.
        
        Returns:
            (is_stationary, stationary_frames)
        """
        if len(history) < 2:
            return False, 0
        
        # Check velocity over recent frames
        stationary_frames = 0
        for i in range(len(history) - 1, -1, -1):
            frame_idx, pos = history[i]
            if i == len(history) - 1:
                # Current frame - check distance from previous
                if i > 0:
                    prev_frame_idx, prev_pos = history[i - 1]
                    velocity = np.linalg.norm(pos - prev_pos)
                    if velocity < self.config['stationary_velocity_threshold']:
                        stationary_frames = 1
                    else:
                        break
            else:
                # Check distance from next frame
                next_frame_idx, next_pos = history[i + 1]
                velocity = np.linalg.norm(next_pos - pos)
                if velocity < self.config['stationary_velocity_threshold']:
                    stationary_frames += 1
                else:
                    break
        
        min_frames = self.config['min_stationary_frames']
        max_frames = self.config['max_stationary_frames']
        
        is_stationary = min_frames <= stationary_frames <= max_frames
        return is_stationary, stationary_frames
    
    def _get_field_zone(self, ball_position: np.ndarray) -> str:
        """
        Determine which field zone the ball is in.
        
        Returns:
            Zone name: 'penalty_area', 'defensive_half', 'near_goal', 'other'
        """
        if self.field_keypoints is None or self.field_corners is None:
            # Fallback: use frame-based estimation
            frame_width = self.config['frame_width']
            frame_height = self.config['frame_height']
            
            # Estimate zones based on frame position
            x_ratio = ball_position[0] / frame_width
            y_ratio = ball_position[1] / frame_height
            
            # Near left goal (penalty area estimate)
            if x_ratio < 0.2:
                return 'penalty_area'
            # Near right goal
            elif x_ratio > 0.8:
                return 'penalty_area'
            # Defensive half (left or right side)
            elif x_ratio < 0.35 or x_ratio > 0.65:
                return 'defensive_half'
            else:
                return 'other'
        
        # Use keypoint-based detection
        # Check if near penalty area keypoints
        penalty_area_keypoints = [1, 2, 3, 4, 17, 18, 19, 20]  # Penalty area keypoints
        
        if self.field_keypoints.shape[0] > 0:
            kpts = self.field_keypoints[0]  # First detection
            
            # Check proximity to penalty area
            for kpt_idx in penalty_area_keypoints:
                if kpt_idx < kpts.shape[0] and kpts[kpt_idx, 2] > 0.5:  # Confidence > 0.5
                    kpt_pos = np.array([kpts[kpt_idx, 0], kpts[kpt_idx, 1]])
                    distance = np.linalg.norm(ball_position - kpt_pos)
                    if distance < self.config['penalty_area_proximity']:
                        return 'penalty_area'
            
            # Check if in defensive half
            if self.field_corners:
                # Get center line (keypoint 11 or 12)
                center_line_y = None
                if kpts.shape[0] > 12 and kpts[11, 2] > 0.5:
                    center_line_y = kpts[11, 1]
                elif kpts.shape[0] > 12 and kpts[12, 2] > 0.5:
                    center_line_y = kpts[12, 1]
                
                if center_line_y is not None:
                    # Check which side of center line
                    if ball_position[1] < center_line_y - 50 or ball_position[1] > center_line_y + 50:
                        # Near goal areas
                        if self.field_corners.get('top_left') or self.field_corners.get('bottom_left'):
                            # Check distance to goals
                            if 'top_left' in self.field_corners:
                                goal_pos = np.array(self.field_corners['top_left'])
                                if np.linalg.norm(ball_position - goal_pos) < self.config['near_goal_distance']:
                                    return 'near_goal'
                            if 'bottom_left' in self.field_corners:
                                goal_pos = np.array(self.field_corners['bottom_left'])
                                if np.linalg.norm(ball_position - goal_pos) < self.config['near_goal_distance']:
                                    return 'near_goal'
                        return 'defensive_half'
        
        return 'other'
    
    def process_frame(self, frame: int, ball_position: Optional[np.ndarray],
                     player_positions: Dict[int, np.ndarray],
                     player_teams: Dict[int, int]) -> List[FreeKickEvent]:
        """
        Process a single frame for free kick detection.
        
        Args:
            frame: Current frame number
            ball_position: Ball position [x, y] or None
            player_positions: Dict of player_id -> position [x, y]
            player_teams: Dict of player_id -> team_id
            
        Returns:
            List of newly detected free kicks
        """
        new_free_kicks = []
        
        if ball_position is None:
            # No ball - clear history if too long
            if len(self.ball_position_history) > 0:
                last_frame = self.ball_position_history[-1][0]
                if frame - last_frame > 30:  # 1 second gap
                    self.ball_position_history = []
            return new_free_kicks
        
        # Add to history
        self.ball_position_history.append((frame, ball_position.copy()))
        
        # Keep history at reasonable size
        if len(self.ball_position_history) > self.max_history_frames:
            self.ball_position_history.pop(0)
        
        # Check if ball is stationary
        is_stationary, stationary_frames = self._is_ball_stationary(
            ball_position, self.ball_position_history
        )
        
        if is_stationary:
            # Check if we already have a candidate for this stationary period
            existing_candidate = None
            for candidate in self.free_kick_candidates:
                if candidate['end_frame'] == frame - 1:  # Previous frame
                    existing_candidate = candidate
                    break
            
            if existing_candidate is None:
                # Create new candidate
                field_zone = self._get_field_zone(ball_position)
                
                # Check if in relevant zone
                if field_zone in ['penalty_area', 'defensive_half', 'near_goal']:
                    # Find closest player
                    closest_player = None
                    closest_distance = float('inf')
                    for player_id, player_pos in player_positions.items():
                        if player_pos is None:
                            continue
                        distance = np.linalg.norm(ball_position - player_pos)
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_player = player_id
                    
                    # Check if player is near ball
                    if closest_player is not None and closest_distance < self.config['player_proximity_radius']:
                        team_id = player_teams.get(closest_player, 0)
                        
                        candidate = {
                            'kicker_id': closest_player,
                            'team_id': team_id,
                            'start_frame': frame - stationary_frames,
                            'stationary_start_frame': frame - stationary_frames,
                            'stationary_end_frame': frame,
                            'stationary_position': ball_position.copy(),
                            'field_zone': field_zone,
                            'stationary_frames': stationary_frames,
                        }
                        self.free_kick_candidates.append(candidate)
                        self.stats['candidates_created'] += 1
        else:
            # Ball is moving - check if we should validate candidates
            for candidate in list(self.free_kick_candidates):
                # Check if kick happened (ball moved away)
                if frame <= candidate['stationary_end_frame'] + self.config['kick_detection_window']:
                    # Check ball speed
                    if len(self.ball_position_history) >= 2:
                        last_pos = self.ball_position_history[-1][1]
                        prev_pos = self.ball_position_history[-2][1]
                        velocity = np.linalg.norm(last_pos - prev_pos)
                        speed_px_per_sec = velocity * self.config['fps']
                        
                        if speed_px_per_sec >= self.config['min_kick_speed']:
                            # Kick detected - create free kick event
                            free_kick = self._create_free_kick_event(candidate, frame, last_pos)
                            if free_kick is not None:
                                new_free_kicks.append(free_kick)
                                self.detected_free_kicks.append(free_kick)
                                self.stats['free_kicks_detected'] += 1
                                
                                # Remove candidate
                                self.free_kick_candidates.remove(candidate)
                else:
                    # Too long - reject candidate
                    self.free_kick_candidates.remove(candidate)
                    self.stats['candidates_rejected'] += 1
        
        return new_free_kicks
    
    def _create_free_kick_event(self, candidate: Dict, end_frame: int, end_position: np.ndarray) -> Optional[FreeKickEvent]:
        """Create a FreeKickEvent from a validated candidate."""
        start_pos = candidate['stationary_position']
        distance = np.linalg.norm(end_position - start_pos)
        duration_frames = end_frame - candidate['start_frame']
        duration_seconds = duration_frames / self.config['fps']
        speed = distance / duration_seconds if duration_seconds > 0 else 0.0
        
        # Determine free kick type (simplified - direct if near goal, indirect otherwise)
        if candidate['field_zone'] == 'penalty_area' or candidate['field_zone'] == 'near_goal':
            free_kick_type = FreeKickType.DIRECT
        else:
            free_kick_type = FreeKickType.INDIRECT
        
        # Calculate confidence
        confidence = self._calculate_confidence(candidate, distance, speed)
        
        # Only create if confidence is high enough
        if confidence < 0.5:
            return None
        
        free_kick_event = FreeKickEvent(
            event_id=f"free_kick_{candidate['start_frame']}",
            kicker_id=candidate['kicker_id'],
            team_id=candidate['team_id'],
            start_frame=candidate['start_frame'],
            end_frame=end_frame,
            start_position=start_pos.tolist(),
            end_position=end_position.tolist(),
            free_kick_type=free_kick_type,
            field_zone=candidate['field_zone'],
            distance_pixels=distance,
            duration_seconds=duration_seconds,
            speed_pixels_per_second=speed,
            confidence=confidence
        )
        
        return free_kick_event
    
    def _calculate_confidence(self, candidate: Dict, distance: float, speed: float) -> float:
        """Calculate confidence score for free kick event."""
        confidence = 0.0
        
        # Zone-based confidence
        zone_weights = {
            'penalty_area': 0.4,
            'near_goal': 0.3,
            'defensive_half': 0.2,
            'other': 0.1
        }
        confidence += zone_weights.get(candidate['field_zone'], 0.1)
        
        # Stationary duration confidence
        stationary_frames = candidate['stationary_frames']
        if 60 <= stationary_frames <= 90:  # 2-3 seconds - ideal
            confidence += 0.3
        elif 90 < stationary_frames <= 120:  # 3-4 seconds
            confidence += 0.2
        else:  # 4-5 seconds
            confidence += 0.1
        
        # Speed confidence
        if speed >= 200.0:  # Fast kick
            confidence += 0.2
        elif speed >= 150.0:  # Medium speed
            confidence += 0.1
        
        # Distance confidence (free kicks are usually longer)
        if distance >= 200.0:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def get_detected_free_kicks(self) -> List[FreeKickEvent]:
        """Get all detected free kicks."""
        return self.detected_free_kicks.copy()
    
    def get_stats(self) -> Dict:
        """Get free kick detection statistics."""
        return self.stats.copy()
