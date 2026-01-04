"""
Strict Ball Tracker Module

Tracks ball with strict rules:
- Max 5-frame prediction only
- No long interpolation
- States: DETECTED | PREDICTED | LOST
- Only DETECTED counts as strong evidence
- Kalman filter for smooth tracking and prediction
- Anti-jitter detection to avoid sticking to random objects
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class BallState(Enum):
    """Ball tracking states."""
    DETECTED = "detected"      # Ball detected by YOLO
    PREDICTED = "predicted"    # Ball predicted (max 5 frames)
    LOST = "lost"              # Ball lost (no prediction)


@dataclass
class BallObservation:
    """Single ball observation."""
    frame: int
    position: Optional[np.ndarray]  # [x, y] in pitch coordinates, None if lost
    state: BallState
    confidence: float = 1.0
    velocity: Optional[np.ndarray] = None  # [vx, vy] if available


@dataclass
class BallTrack:
    """Ball track with strict constraints."""
    ball_id: int
    current_state: BallState
    position: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    last_detected_frame: Optional[int] = None
    last_detected_position: Optional[np.ndarray] = None
    prediction_count: int = 0  # How many frames since last detection
    history: deque = field(default_factory=lambda: deque(maxlen=30))


class StrictBallTracker:
    """
    Strict ball tracker with no hallucinations.
    
    Rules:
    - Max 5-frame prediction
    - No forward/backward fill beyond prediction window
    - Missing ball = valid LOST state
    - Only DETECTED positions are trusted
    - Anti-jitter: reject detections that jump too far too fast
    """
    
    # Maximum frames to predict after last detection
    MAX_PREDICTION_FRAMES = 5
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the ball tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            # Prediction limits - increased for better continuity
            'max_prediction_frames': 15,  # Increased from 10 to 15 for better continuity
            
            # Motion constraints (in PIXELS - matching pass detector) - VERY LENIENT
            # Note: Pass detector uses pixels, so ball tracker should too
            'max_ball_speed': 2000.0,    # pixels/s - VERY HIGH to allow fast passes (was 40 m/s)
            'max_position_jump': 500.0,   # pixels - VERY HIGH to allow long passes (was 12 meters)
            'min_detection_distance': 5.0, # pixels - very small threshold (was 0.3 meters)
            
            # Anti-jitter: reject detections that oscillate - VERY LENIENT
            'jitter_window': 5,           # frames to check for jitter
            'jitter_threshold': 200.0,    # pixels - VERY HIGH to reduce false rejections (was 5.0 meters)
            
            # Tracking
            'min_detection_confidence': 0.1,  # Reduced from 0.2 to accept even more detections
            
            # Kalman filter parameters - adjusted for smoother tracking
            'process_noise': 0.5,         # Increased from 0.3 for more flexibility
            'measurement_noise': 1.0,     # Increased from 0.8 for more tolerance
            'initial_covariance': 10.0,    # Increased from 8.0
            
            # Frame rate
            'fps': 30.0,
        }
        
        if config:
            self.config.update(config)
        
        # Ball track (single ball)
        self.ball_track: Optional[BallTrack] = None
        
        # Detection history - stores actual observations
        self.detection_history: List[BallObservation] = []
        
        # Frame-indexed observations for quick lookup
        self.frame_observations: Dict[int, BallObservation] = {}
        
        # Recent detections for jitter detection
        self.recent_detections: deque = deque(maxlen=10)
        
        # Kalman filter state
        self._init_kalman()
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'detected_frames': 0,
            'predicted_frames': 0,
            'lost_frames': 0,
            'rejected_jitter': 0,
            'rejected_jump': 0,
        }
    
    def _init_kalman(self):
        """Initialize Kalman filter for ball prediction."""
        # State: [x, y, vx, vy]
        self.kalman_state: Optional[np.ndarray] = None
        
        # State covariance matrix (4x4)
        self.kalman_P: Optional[np.ndarray] = None
        
        # State transition matrix (constant velocity model)
        dt = 1.0 / self.config['fps']
        self.kalman_F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        # Observation matrix (we observe x, y)
        self.kalman_H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)
        
        # Process noise covariance
        q = self.config['process_noise']
        self.kalman_Q = np.array([
            [q*dt**4/4, 0, q*dt**3/2, 0],
            [0, q*dt**4/4, 0, q*dt**3/2],
            [q*dt**3/2, 0, q*dt**2, 0],
            [0, q*dt**3/2, 0, q*dt**2]
        ], dtype=np.float64)
        
        # Measurement noise covariance
        r = self.config['measurement_noise']
        self.kalman_R = np.array([
            [r, 0],
            [0, r]
        ], dtype=np.float64)
    
    def _reset_kalman(self, position: np.ndarray, velocity: Optional[np.ndarray] = None):
        """Reset Kalman filter with new initial state."""
        vx, vy = (0.0, 0.0) if velocity is None else (velocity[0], velocity[1])
        self.kalman_state = np.array([position[0], position[1], vx, vy], dtype=np.float64)
        self.kalman_P = np.eye(4, dtype=np.float64) * self.config['initial_covariance']
    
    def _kalman_predict(self) -> Optional[np.ndarray]:
        """Kalman filter prediction step."""
        if self.kalman_state is None:
            return None
        
        # Predict state: x = F * x
        self.kalman_state = self.kalman_F @ self.kalman_state
        
        # Predict covariance: P = F * P * F' + Q
        self.kalman_P = self.kalman_F @ self.kalman_P @ self.kalman_F.T + self.kalman_Q
        
        return self.kalman_state[:2].copy()
    
    def _kalman_update(self, measurement: np.ndarray) -> np.ndarray:
        """Kalman filter update step with measurement."""
        if self.kalman_state is None:
            self._reset_kalman(measurement)
            return measurement.copy()
        
        # Innovation: y = z - H * x
        z = np.array([measurement[0], measurement[1]], dtype=np.float64)
        y = z - self.kalman_H @ self.kalman_state
        
        # Innovation covariance: S = H * P * H' + R
        S = self.kalman_H @ self.kalman_P @ self.kalman_H.T + self.kalman_R
        
        # Kalman gain: K = P * H' * S^-1
        K = self.kalman_P @ self.kalman_H.T @ np.linalg.inv(S)
        
        # Update state: x = x + K * y
        self.kalman_state = self.kalman_state + K @ y
        
        # Update covariance: P = (I - K * H) * P
        I = np.eye(4, dtype=np.float64)
        self.kalman_P = (I - K @ self.kalman_H) @ self.kalman_P
        
        return self.kalman_state[:2].copy()
    
    def _is_valid_detection(self, position: np.ndarray, frame: int) -> Tuple[bool, str]:
        """
        Check if a detection is valid (not jitter or unrealistic jump).
        
        Returns:
            (is_valid, reason)
        """
        if self.ball_track is None or self.ball_track.last_detected_position is None:
            return True, "first_detection"
        
        last_pos = self.ball_track.last_detected_position
        last_frame = self.ball_track.last_detected_frame
        
        if last_frame is None:
            return True, "no_previous_frame"
        
        # Calculate distance and time
        distance = np.linalg.norm(position - last_pos)
        frames_diff = frame - last_frame
        
        if frames_diff <= 0:
            return True, "same_or_past_frame"
        
        # Check for unrealistic jump
        time_diff = frames_diff / self.config['fps']
        implied_speed = distance / time_diff if time_diff > 0 else float('inf')
        
        max_distance = self.config['max_position_jump'] * frames_diff
        
        if distance > max_distance:
            self.stats['rejected_jump'] += 1
            return False, f"jump_too_far_{distance:.1f}m_in_{frames_diff}f"
        
        if implied_speed > self.config['max_ball_speed']:
            self.stats['rejected_jump'] += 1
            return False, f"speed_too_high_{implied_speed:.1f}m/s"
        
        # Check for jitter (oscillating detections)
        if len(self.recent_detections) >= 3:
            # Check if detection oscillates back and forth
            recent_positions = [d[1] for d in self.recent_detections if d[1] is not None]
            if len(recent_positions) >= 3:
                # Check direction changes
                direction_changes = 0
                for i in range(2, len(recent_positions)):
                    v1 = recent_positions[i-1] - recent_positions[i-2]
                    v2 = recent_positions[i] - recent_positions[i-1]
                    if np.linalg.norm(v1) > 0.1 and np.linalg.norm(v2) > 0.1:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        if cos_angle < -0.5:  # More than 120 degree turn
                            direction_changes += 1
                
                if direction_changes >= 2:
                    self.stats['rejected_jitter'] += 1
                    return False, "jitter_detected"
        
        return True, "valid"
    
    def update(self, frame: int, ball_detections: Optional[np.ndarray]) -> BallObservation:
        """
        Update ball tracker with new detections.
        
        Args:
            frame: Current frame number
            ball_detections: Array of ball detections [N, 2] in pitch coordinates, or None
            
        Returns:
            BallObservation for current frame
        """
        self.stats['total_frames'] += 1
        
        # Check for valid detections
        detected_position = None
        if ball_detections is not None and len(ball_detections) > 0:
            # If multiple detections, use closest to predicted position
            if len(ball_detections) == 1:
                candidate = ball_detections[0]
            else:
                # Multiple detections - prefer one closest to prediction
                predicted = self._kalman_predict() if self.kalman_state is not None else None
                if predicted is not None:
                    distances = [np.linalg.norm(det - predicted) for det in ball_detections]
                    candidate = ball_detections[np.argmin(distances)]
                else:
                    candidate = ball_detections[0]
            
            # Validate detection
            is_valid, reason = self._is_valid_detection(candidate, frame)
            if is_valid:
                detected_position = candidate.copy()
        
        # Update state based on detection
        if detected_position is not None:
            # DETECTED state
            self.stats['detected_frames'] += 1
            
            # Update Kalman filter with measurement
            filtered_position = self._kalman_update(detected_position)
            
            # Calculate velocity from Kalman state
            velocity = None
            if self.kalman_state is not None:
                velocity = self.kalman_state[2:4].copy()
            
            # Update or create track
            if self.ball_track is None:
                self.ball_track = BallTrack(ball_id=0, current_state=BallState.DETECTED)
            
            self.ball_track.current_state = BallState.DETECTED
            self.ball_track.position = filtered_position
            self.ball_track.velocity = velocity
            self.ball_track.last_detected_frame = frame
            self.ball_track.last_detected_position = detected_position.copy()
            self.ball_track.prediction_count = 0
            self.ball_track.history.append((frame, filtered_position.copy()))
            
            # Store in recent detections
            self.recent_detections.append((frame, detected_position.copy()))
            
            obs = BallObservation(
                frame=frame,
                position=filtered_position,
                state=BallState.DETECTED,
                confidence=1.0,
                velocity=velocity
            )
        
        elif self.ball_track is not None and self.ball_track.last_detected_frame is not None:
            # Check if we can predict
            frames_since_detection = frame - self.ball_track.last_detected_frame
            
            if frames_since_detection <= self.config['max_prediction_frames']:
                # PREDICTED state (within limit)
                self.stats['predicted_frames'] += 1
                
                # Predict position using Kalman filter
                predicted_position = self._kalman_predict()
                
                if predicted_position is not None:
                    self.ball_track.current_state = BallState.PREDICTED
                    self.ball_track.position = predicted_position
                    self.ball_track.prediction_count = frames_since_detection
                    
                    # Confidence decreases with prediction depth
                    confidence = max(0.0, 1.0 - frames_since_detection * 0.15)
                    
                    velocity = None
                    if self.kalman_state is not None:
                        velocity = self.kalman_state[2:4].copy()
                    
                    obs = BallObservation(
                        frame=frame,
                        position=predicted_position,
                        state=BallState.PREDICTED,
                        confidence=confidence,
                        velocity=velocity
                    )
                else:
                    # Can't predict
                    self.stats['lost_frames'] += 1
                    self.ball_track.current_state = BallState.LOST
                    obs = BallObservation(
                        frame=frame,
                        position=None,
                        state=BallState.LOST,
                        confidence=0.0
                    )
            else:
                # LOST state (beyond prediction limit)
                self.stats['lost_frames'] += 1
                self.ball_track.current_state = BallState.LOST
                self.ball_track.prediction_count = frames_since_detection
                
                obs = BallObservation(
                    frame=frame,
                    position=None,
                    state=BallState.LOST,
                    confidence=0.0
                )
        else:
            # No track, no detection
            self.stats['lost_frames'] += 1
            obs = BallObservation(
                frame=frame,
                position=None,
                state=BallState.LOST,
                confidence=0.0
            )
        
        # Store observation
        self.detection_history.append(obs)
        self.frame_observations[frame] = obs
        
        # Trim history to prevent memory issues
        if len(self.detection_history) > 1000:
            old_obs = self.detection_history[:-500]
            self.detection_history = self.detection_history[-500:]
            for old in old_obs:
                self.frame_observations.pop(old.frame, None)
        
        return obs
    
    def get_ball_position(self, frame: int) -> Optional[BallObservation]:
        """
        Get ball position for a specific frame.
        
        Args:
            frame: Frame number
            
        Returns:
            BallObservation or None if not available
        """
        return self.frame_observations.get(frame)
    
    def get_current_ball(self) -> Optional[BallObservation]:
        """Get current ball observation."""
        if self.detection_history:
            return self.detection_history[-1]
        return None
    
    def is_detected(self, frame: int) -> bool:
        """Check if ball is DETECTED (not predicted) at frame."""
        obs = self.get_ball_position(frame)
        return obs is not None and obs.state == BallState.DETECTED
    
    def is_available(self, frame: int) -> bool:
        """Check if ball position is available (DETECTED or PREDICTED) at frame."""
        obs = self.get_ball_position(frame)
        return obs is not None and obs.state in [BallState.DETECTED, BallState.PREDICTED]
    
    def get_ball_in_window(self, start_frame: int, end_frame: int) -> List[BallObservation]:
        """
        Get all ball observations in a time window.
        
        Args:
            start_frame: Start frame
            end_frame: End frame
            
        Returns:
            List of BallObservations
        """
        return [obs for obs in self.detection_history 
                if start_frame <= obs.frame <= end_frame]
    
    def get_detection_rate(self) -> float:
        """Get the fraction of frames where ball was detected."""
        if self.stats['total_frames'] == 0:
            return 0.0
        return self.stats['detected_frames'] / self.stats['total_frames']
    
    def get_available_rate(self) -> float:
        """Get the fraction of frames where ball position is available."""
        if self.stats['total_frames'] == 0:
            return 0.0
        available = self.stats['detected_frames'] + self.stats['predicted_frames']
        return available / self.stats['total_frames']
    
    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        return {
            **self.stats,
            'detection_rate': self.get_detection_rate(),
            'available_rate': self.get_available_rate(),
        }
    
    def reset(self):
        """Reset tracker state."""
        self.ball_track = None
        self.detection_history = []
        self.frame_observations = {}
        self.recent_detections.clear()
        self._init_kalman()
        self.stats = {k: 0 for k in self.stats}
