"""
Shot on Target Detection Module

Detects shots on target by analyzing:
1. Ball trajectory direction (toward goal)
2. Ball speed (shots are typically faster)
3. End point (ball reaches goal area, not a player)
4. No player receiver (ball goes to goal, not player)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .pass_event import PassEvent


class ShotType(Enum):
    """Types of shots."""
    SHOT_ON_TARGET = "shot_on_target"  # Ball reaches goal area
    SHOT_OFF_TARGET = "shot_off_target"  # Shot attempted but missed
    SHOT_BLOCKED = "shot_blocked"  # Shot blocked by defender


@dataclass
class ShotEvent:
    """Represents a shot on target event."""
    event_id: str
    shooter_id: int
    team_id: int
    start_frame: int
    end_frame: int
    start_position: List[float]
    end_position: List[float]
    shot_type: ShotType
    goal_position: Optional[Tuple[float, float]] = None  # Goal center in pixels
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


class ShotDetector:
    """
    Detects shots on target by analyzing ball trajectory and goal proximity.
    
    Key indicators of a shot:
    1. Ball moves toward goal (not toward a player)
    2. High speed (faster than typical passes)
    3. Ball ends near goal area (not near a player)
    4. No player receiver (ball goes to goal)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize shot detector."""
        self.config = {
            # Goal detection
            'goal_detection_enabled': True,
            'goal_width_pixels': 200.0,  # Approximate goal width in pixels (adjust based on video)
            'goal_height_pixels': 80.0,  # Approximate goal height in pixels
            'goal_area_tolerance': 150.0,  # Pixels - how close ball must be to goal
            
            # Shot detection thresholds
            'min_shot_speed': 300.0,  # pixels/second - shots are faster than passes
            'min_shot_distance': 100.0,  # pixels - minimum distance for a shot
            'max_shot_distance': 2000.0,  # pixels - maximum realistic shot distance
            
            # Trajectory analysis
            'goal_direction_threshold': 0.7,  # Cosine similarity - ball must move toward goal
            'min_trajectory_frames': 5,  # Minimum frames for trajectory analysis
            
            # Frame rate
            'fps': 30.0,
        }
        
        if config:
            self.config.update(config)
        
        # Goal positions (will be detected/estimated from video)
        self.left_goal_center: Optional[np.ndarray] = None
        self.right_goal_center: Optional[np.ndarray] = None
        
        # Detected shots
        self.detected_shots: List[ShotEvent] = []
        
        # Statistics
        self.stats = {
            'shots_detected': 0,
            'shots_on_target': 0,
            'shots_off_target': 0,
            'shots_blocked': 0,
        }
    
    def estimate_goal_positions(self, frame_width: int, frame_height: int):
        """
        Estimate goal positions based on frame dimensions.
        
        Goals are typically at the left and right edges of the field.
        This is a simple estimation - can be improved with keypoint detection.
        """
        # Estimate goals at left and right edges (middle vertically)
        self.left_goal_center = np.array([
            0.05 * frame_width,  # 5% from left edge
            frame_height / 2.0   # Middle vertically
        ], dtype=np.float32)
        
        self.right_goal_center = np.array([
            0.95 * frame_width,  # 5% from right edge
            frame_height / 2.0   # Middle vertically
        ], dtype=np.float32)
        
        print(f"[Shot Detector] Estimated goal positions:")
        print(f"  Left goal: ({self.left_goal_center[0]:.1f}, {self.left_goal_center[1]:.1f})")
        print(f"  Right goal: ({self.right_goal_center[0]:.1f}, {self.right_goal_center[1]:.1f})")
    
    def set_goal_positions(self, left_goal: np.ndarray, right_goal: np.ndarray):
        """Set goal positions from keypoint detection or manual calibration."""
        self.left_goal_center = left_goal.copy()
        self.right_goal_center = right_goal.copy()
    
    def _get_closest_goal(self, position: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Get the closest goal to a position.
        
        Returns:
            (goal_center, distance)
        """
        if self.left_goal_center is None or self.right_goal_center is None:
            return None, float('inf')
        
        dist_left = np.linalg.norm(position - self.left_goal_center)
        dist_right = np.linalg.norm(position - self.right_goal_center)
        
        if dist_left < dist_right:
            return self.left_goal_center, dist_left
        else:
            return self.right_goal_center, dist_right
    
    def _is_ball_near_goal(self, ball_position: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if ball is near a goal.
        
        Returns:
            (is_near_goal, goal_center)
        """
        goal_center, distance = self._get_closest_goal(ball_position)
        
        if goal_center is None:
            return False, None
        
        # Check if within goal area tolerance
        if distance <= self.config['goal_area_tolerance']:
            return True, goal_center
        
        return False, goal_center
    
    def _calculate_trajectory_direction(self, trajectory: List[np.ndarray], 
                                       target_goal: np.ndarray) -> float:
        """
        Calculate how well the trajectory points toward the goal.
        
        Returns:
            Cosine similarity (0-1, higher = more toward goal)
        """
        if len(trajectory) < 2:
            return 0.0
        
        # Get overall trajectory direction
        start_pos = trajectory[0]
        end_pos = trajectory[-1]
        trajectory_direction = end_pos - start_pos
        
        # Get direction to goal
        goal_direction = target_goal - start_pos
        
        # Normalize vectors
        traj_norm = np.linalg.norm(trajectory_direction)
        goal_norm = np.linalg.norm(goal_direction)
        
        if traj_norm < 1e-6 or goal_norm < 1e-6:
            return 0.0
        
        traj_unit = trajectory_direction / traj_norm
        goal_unit = goal_direction / goal_norm
        
        # Cosine similarity
        similarity = np.dot(traj_unit, goal_unit)
        
        return similarity
    
    def is_shot(self, candidate: Dict, ball_trajectory: List[np.ndarray],
                player_positions: Dict[int, np.ndarray],
                ball_end_position: np.ndarray) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Determine if a candidate event is a shot (not a pass).
        
        Args:
            candidate: Pass candidate dictionary
            ball_trajectory: Ball trajectory during the event
            player_positions: Current player positions
            ball_end_position: Where the ball ended up
            
        Returns:
            (is_shot, confidence, goal_center)
        """
        if self.left_goal_center is None or self.right_goal_center is None:
            # No goal positions - can't detect shots
            return False, 0.0, None
        
        # Get closest goal
        goal_center, goal_distance = self._get_closest_goal(ball_end_position)
        
        if goal_center is None:
            return False, 0.0, None
        
        # Check 1: Ball ends near goal (not near a player)
        is_near_goal, _ = self._is_ball_near_goal(ball_end_position)
        
        # Check distance to nearest player
        min_player_distance = float('inf')
        for player_id, player_pos in player_positions.items():
            if player_pos is None:
                continue
            dist = np.linalg.norm(ball_end_position - player_pos)
            min_player_distance = min(min_player_distance, dist)
        
        # If ball is closer to goal than to any player, likely a shot
        player_vs_goal = min_player_distance > goal_distance
        
        # Check 2: Ball trajectory points toward goal
        trajectory_similarity = 0.0
        if len(ball_trajectory) >= 2:
            trajectory_similarity = self._calculate_trajectory_direction(
                ball_trajectory, goal_center
            )
        
        # Check 3: Ball speed (shots are typically faster)
        start_pos = np.array(candidate['from_pos'])
        end_pos = np.array(candidate['to_pos'])
        distance = np.linalg.norm(end_pos - start_pos)
        duration_frames = candidate['end_frame'] - candidate['start_frame']
        duration_seconds = duration_frames / self.config['fps']
        
        if duration_seconds > 0:
            speed = distance / duration_seconds
        else:
            speed = 0.0
        
        is_fast_enough = speed >= self.config['min_shot_speed']
        
        # Check 4: Distance (shots are usually longer than short passes)
        is_long_enough = distance >= self.config['min_shot_distance']
        
        # Calculate confidence score
        confidence = 0.0
        
        # High confidence if ball ends at goal
        if is_near_goal:
            confidence += 0.4
        
        # High confidence if trajectory points toward goal
        if trajectory_similarity >= self.config['goal_direction_threshold']:
            confidence += 0.3
        
        # Medium confidence if ball is closer to goal than players
        if player_vs_goal:
            confidence += 0.2
        
        # Medium confidence if fast enough
        if is_fast_enough:
            confidence += 0.1
        
        # Shot detected if confidence is high enough
        is_shot = confidence >= 0.5
        
        return is_shot, confidence, goal_center
    
    def classify_shot_type(self, ball_end_position: np.ndarray,
                          goal_center: np.ndarray) -> ShotType:
        """
        Classify the type of shot based on where ball ends.
        
        Args:
            ball_end_position: Where the ball ended
            goal_center: Goal center position
            
        Returns:
            ShotType
        """
        distance_to_goal = np.linalg.norm(ball_end_position - goal_center)
        
        # Check if ball is within goal area
        if distance_to_goal <= self.config['goal_area_tolerance']:
            return ShotType.SHOT_ON_TARGET
        elif distance_to_goal <= self.config['goal_area_tolerance'] * 2:
            # Close but not quite - might be blocked or off target
            return ShotType.SHOT_OFF_TARGET
        else:
            return ShotType.SHOT_OFF_TARGET
    
    def create_shot_event(self, candidate: Dict, ball_trajectory: List[np.ndarray],
                         goal_center: np.ndarray, confidence: float) -> ShotEvent:
        """Create a ShotEvent from a candidate."""
        start_pos = np.array(candidate['from_pos'])
        end_pos = np.array(candidate['to_pos'])
        distance = np.linalg.norm(end_pos - start_pos)
        duration_frames = candidate['end_frame'] - candidate['start_frame']
        duration_seconds = duration_frames / self.config['fps']
        speed = distance / duration_seconds if duration_seconds > 0 else 0.0
        
        shot_type = self.classify_shot_type(end_pos, goal_center)
        
        shot_event = ShotEvent(
            event_id=f"shot_{candidate['start_frame']}",
            shooter_id=candidate['from_player'],
            team_id=candidate['from_team'],
            start_frame=candidate['start_frame'],
            end_frame=candidate['end_frame'],
            start_position=start_pos.tolist(),
            end_position=end_pos.tolist(),
            shot_type=shot_type,
            goal_position=(float(goal_center[0]), float(goal_center[1])),
            distance_pixels=distance,
            duration_seconds=duration_seconds,
            speed_pixels_per_second=speed,
            confidence=confidence
        )
        
        return shot_event
    
    def get_stats(self) -> Dict:
        """Get shot detection statistics."""
        return {
            **self.stats,
            'total_shots': len(self.detected_shots),
        }

