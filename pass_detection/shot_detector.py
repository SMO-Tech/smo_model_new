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
            # Goal detection - ADAPTIVE based on video resolution
            'goal_detection_enabled': True,
            'goal_width_pixels': 200.0,  # Approximate goal width in pixels (adjust based on video)
            'goal_height_pixels': 80.0,  # Approximate goal height in pixels
            'goal_area_tolerance': 400.0,  # Pixels - how close ball must be to goal (adaptive, scales with resolution)
            'goal_area_tolerance_ratio': 0.25,  # Ratio of frame width - adaptive tolerance based on video size (balanced to catch shots but not passes)
            
            # Shot detection thresholds - ADAPTIVE
            'min_shot_speed': 200.0,  # pixels/second - shots are faster than passes (lowered for better detection)
            'min_shot_distance': 50.0,  # pixels - minimum distance for a shot (lowered)
            'max_shot_distance': 3000.0,  # pixels - maximum realistic shot distance (increased)
            
            # Trajectory analysis - MORE LENIENT
            'goal_direction_threshold': 0.3,  # Cosine similarity - lowered from 0.7 for better detection
            'min_trajectory_frames': 2,  # Minimum frames for trajectory analysis (reduced)
            
            # Frame rate
            'fps': 30.0,
            
            # Video resolution (will be set automatically)
            'frame_width': 1920,
            'frame_height': 1080,
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
        # Store frame dimensions for adaptive thresholds
        self.config['frame_width'] = frame_width
        self.config['frame_height'] = frame_height
        
        # Adaptive goal area tolerance based on frame width (15% of width, min 300px)
        adaptive_tolerance = frame_width * self.config.get('goal_area_tolerance_ratio', 0.15)
        self.config['goal_area_tolerance'] = max(adaptive_tolerance, 300.0)
        
        # Estimate goals at left and right edges (middle vertically)
        self.left_goal_center = np.array([
            0.05 * frame_width,  # 5% from left edge
            frame_height / 2.0   # Middle vertically
        ], dtype=np.float32)
        
        self.right_goal_center = np.array([
            0.95 * frame_width,  # 5% from right edge
            frame_height / 2.0   # Middle vertically
        ], dtype=np.float32)
        
        print(f"[Shot Detector] Estimated goal positions (adaptive tolerance: {self.config['goal_area_tolerance']:.1f}px):")
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
        import json
        import time
        
        # #region agent log - SHOT CHECK START
        try:
            with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'hypothesisId': 'E',
                    'location': 'shot_detector.py:215',
                    'message': 'shot_check_start',
                    'data': {
                        'frame': int(candidate.get('start_frame', 0)),
                        'ball_end': ball_end_position.tolist() if hasattr(ball_end_position, 'tolist') else list(ball_end_position),
                        'has_goals': self.left_goal_center is not None and self.right_goal_center is not None
                    },
                    'timestamp': int(time.time() * 1000),
                    'sessionId': 'debug-session',
                    'runId': 'run1'
                }) + '\n')
        except: pass
        # #endregion
        
        if self.left_goal_center is None or self.right_goal_center is None:
            # No goal positions - can't detect shots
            return False, 0.0, None
        
        # Get closest goal
        goal_center, goal_distance = self._get_closest_goal(ball_end_position)
        
        if goal_center is None:
            return False, 0.0, None
        
        # Check 1: Ball ends near goal (not near a player)
        is_near_goal, _ = self._is_ball_near_goal(ball_end_position)
        
        # Check distance to nearest player - IMPORTANT: if ball ends near a player, it's a pass, not a shot
        min_player_distance = float('inf')
        closest_player_id = None
        for player_id, player_pos in player_positions.items():
            if player_pos is None:
                continue
            dist = np.linalg.norm(ball_end_position - player_pos)
            if dist < min_player_distance:
                min_player_distance = dist
                closest_player_id = player_id
        
        # CRITICAL: Check if ball ends near the RECEIVER player (the "to_player" in the candidate)
        # If the ball ends within possession radius of the receiver, it's definitely a pass, not a shot
        # UNLESS: the receiver is on a different team (interception) AND ball is near goal (could be saved shot)
        # If to_player is None, this is a shot candidate (no receiver - ball goes to goal)
        receiver_player_id = candidate.get('to_player')
        receiver_team_id = candidate.get('to_team')
        from_team_id = candidate.get('from_team')
        receiver_is_near = False
        receiver_is_same_team = False
        
        # If no receiver (to_player is None), this is likely a shot
        has_no_receiver = receiver_player_id is None
        if has_no_receiver:
            # No receiver - ball goes to goal, not to a player
            # This is a STRONG indicator of a shot - give significant confidence boost
            pass  # Continue to shot detection logic (will boost confidence later)
        elif receiver_player_id in player_positions:
            receiver_pos = player_positions[receiver_player_id]
            if receiver_pos is not None:
                receiver_distance = np.linalg.norm(ball_end_position - receiver_pos)
                possession_radius = 400.0  # Match pass detector's effective radius
                receiver_is_near = receiver_distance < possession_radius
                # Check if receiver is on same team
                if receiver_team_id is not None and from_team_id is not None:
                    receiver_is_same_team = receiver_team_id == from_team_id
        
        # If ball ends very close to a player (within possession radius), it's definitely a pass, not a shot
        # Use the same effective radius as pass detector (300 base + 100 tolerance = 400)
        possession_radius = 400.0  # Match pass detector's effective radius
        is_near_player = min_player_distance < possession_radius
        
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
        
        # Check for strong shot indicators (trajectory + speed)
        strong_trajectory = trajectory_similarity >= 0.7
        high_speed = speed >= 400.0
        
        # #region agent log - SHOT CHECK INTERMEDIATE VALUES
        try:
            with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'hypothesisId': 'E',
                    'location': 'shot_detector.py:306',
                    'message': 'shot_check_intermediate',
                    'data': {
                        'frame': int(candidate.get('start_frame', 0)),
                        'is_near_goal': bool(is_near_goal),
                        'goal_distance': float(goal_distance),
                        'min_player_distance': float(min_player_distance),
                        'receiver_is_near': bool(receiver_is_near),
                        'receiver_is_same_team': bool(receiver_is_same_team),
                        'is_near_player': bool(is_near_player),
                        'player_vs_goal': bool(player_vs_goal),
                        'trajectory_similarity': float(trajectory_similarity),
                        'speed': float(speed),
                        'is_fast_enough': bool(is_fast_enough),
                        'distance': float(distance),
                        'is_long_enough': bool(is_long_enough)
                    },
                    'timestamp': int(time.time() * 1000),
                    'sessionId': 'debug-session',
                    'runId': 'run1'
                }) + '\n')
        except: pass
        # #endregion
        
        # CRITICAL: If ball ends near the RECEIVER player, it's a pass, not a shot
        # The ONLY exception: receiver is on different team (interception) AND ball is near goal (saved shot)
        if receiver_is_near:
            if receiver_is_same_team:
                # Ball ended near receiver on same team - this is ALWAYS a pass, not a shot
                # #region agent log - REJECTED: RECEIVER SAME TEAM
                try:
                    with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            'hypothesisId': 'E',
                            'location': 'shot_detector.py:374',
                            'message': 'rejected_receiver_same_team',
                            'data': {'frame': int(candidate.get('start_frame', 0)), 'trajectory_similarity': float(trajectory_similarity), 'speed': float(speed), 'is_near_goal': bool(is_near_goal)},
                            'timestamp': int(time.time() * 1000),
                            'sessionId': 'debug-session',
                            'runId': 'run1'
                        }) + '\n')
                except: pass
                # #endregion
                return False, 0.0, goal_center
            else:
                # Receiver is on different team (interception)
                # Consider it a shot if ball is near goal OR if trajectory/speed are strong
                if not is_near_goal and not (strong_trajectory and high_speed):
                    # Ball is not near goal and weak trajectory/speed - this is just an interception, not a shot
                    # #region agent log - REJECTED: INTERCEPTION NOT NEAR GOAL
                    try:
                        with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                'hypothesisId': 'E',
                                'location': 'shot_detector.py:393',
                                'message': 'rejected_interception_not_near_goal',
                                'data': {'frame': int(candidate.get('start_frame', 0)), 'goal_distance': float(goal_distance), 'trajectory_similarity': float(trajectory_similarity), 'speed': float(speed)},
                                'timestamp': int(time.time() * 1000),
                                'sessionId': 'debug-session',
                                'runId': 'run1'
                            }) + '\n')
                    except: pass
                    # #endregion
                    return False, 0.0, goal_center
        
        # If receiver is on different team (interception) AND ball is near goal, it's likely a shot
        # (even if receiver is nearby - they might have saved/blocked it)
        if receiver_is_near and not receiver_is_same_team and is_near_goal:
            # This is an interception near the goal - likely a saved/blocked shot
            # Give it a boost to be detected as a shot
            pass  # Continue to confidence calculation with boost
        
        # CRITICAL: If ball ends near a player (within possession radius), check if it's still a shot
        # Allow shots even if near player IF:
        # 1. Strong trajectory toward goal (similarity > 0.7) AND high speed (>400 px/s), OR
        # 2. Ball is near goal (could be saved/blocked shot)
        if is_near_player and not (receiver_is_near and not receiver_is_same_team and is_near_goal):
            # Ball is near a player, but not an interception near goal
            # Check if it might still be a shot based on trajectory/speed or goal proximity
            if is_near_goal:
                # Ball is near goal - might be a shot even if near player (saved/blocked)
                pass  # Continue to confidence calculation
            elif strong_trajectory and high_speed:
                # Strong trajectory and high speed - might be a shot even if not near goal
                pass  # Continue to confidence calculation
            else:
                # Ball is near a player but not near goal and weak trajectory/speed
                # This is likely a pass, not a shot
                # #region agent log - REJECTED: NEAR PLAYER NOT NEAR GOAL
                try:
                    with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            'hypothesisId': 'E',
                            'location': 'shot_detector.py:420',
                            'message': 'rejected_near_player_not_near_goal',
                            'data': {'frame': int(candidate.get('start_frame', 0)), 'min_player_distance': float(min_player_distance), 'goal_distance': float(goal_distance), 'trajectory_similarity': float(trajectory_similarity), 'speed': float(speed)},
                            'timestamp': int(time.time() * 1000),
                            'sessionId': 'debug-session',
                            'runId': 'run1'
                        }) + '\n')
                except: pass
                # #endregion
                return False, 0.0, goal_center
        
        # Calculate confidence score - ADAPTIVE and MORE LENIENT
        confidence = 0.0
        
        # High confidence if ball ends at goal
        if is_near_goal:
            confidence += 0.4  # Base confidence for near goal
            # Extra boost if very close to goal (adaptive threshold based on frame width)
            close_threshold = self.config.get('frame_width', 1920) * 0.1  # 10% of frame width
            if goal_distance < close_threshold:
                confidence += 0.2
        
        # High confidence if trajectory points toward goal (more lenient)
        if trajectory_similarity >= self.config['goal_direction_threshold']:
            confidence += 0.3
        elif trajectory_similarity >= 0.1:  # Very lenient threshold
            confidence += 0.1
        
        # Medium confidence if ball is closer to goal than players
        if player_vs_goal:
            confidence += 0.2
        
        # Medium confidence if fast enough (lowered threshold)
        if is_fast_enough:
            confidence += 0.1
        elif speed >= self.config['min_shot_speed'] * 0.5:  # Half speed still counts
            confidence += 0.05
        
        # If ball is near goal and near a player, it might be a saved/blocked shot
        if is_near_goal and is_near_player:
            confidence += 0.1
        
        # Extra boost for interceptions near goal (saved/blocked shots)
        if receiver_is_near and not receiver_is_same_team and is_near_goal:
            confidence += 0.3  # Strong boost for saved shots
        
        # MAJOR boost for shots with no receiver (ball goes to goal, not to a player)
        if has_no_receiver:
            confidence += 0.3  # Strong boost - no receiver is a key indicator of a shot
            # If ball is also closer to goal than to any player, even stronger
            if player_vs_goal:
                confidence += 0.2  # Extra boost
        
        # ADAPTIVE threshold: Lower for shots with no receiver, lower for interceptions near goal
        if has_no_receiver:
            # No receiver - ball goes to goal - this is a strong shot indicator
            # Use lower threshold since no receiver is already a strong signal
            is_shot = confidence >= 0.5  # Lower threshold for shots with no receiver
            threshold_used = 0.5
        elif receiver_is_near and not receiver_is_same_team and is_near_goal:
            # Interception near goal - likely a saved/blocked shot
            is_shot = confidence >= 0.5  # Moderate threshold for saved shots
            threshold_used = 0.5
        elif is_near_goal:
            # Ball near goal - stricter threshold to avoid false positives
            is_shot = confidence >= 0.65  # Higher threshold
            threshold_used = 0.65
        else:
            # Regular shot detection
            is_shot = confidence >= 0.7  # High threshold for shots far from goal
            threshold_used = 0.7
        
        # #region agent log - SHOT CHECK RESULT
        try:
            with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'hypothesisId': 'E',
                    'location': 'shot_detector.py:383',
                    'message': 'shot_check_final_result',
                    'data': {
                        'frame': int(candidate.get('start_frame', 0)),
                        'is_shot': bool(is_shot),
                        'confidence': float(confidence),
                        'threshold_used': float(threshold_used),
                        'goal_center': goal_center.tolist() if goal_center is not None else None
                    },
                    'timestamp': int(time.time() * 1000),
                    'sessionId': 'debug-session',
                    'runId': 'run1'
                }) + '\n')
        except: pass
        # #endregion
        
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

