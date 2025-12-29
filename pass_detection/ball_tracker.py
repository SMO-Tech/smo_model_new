"""
Strict Ball Tracker Module

Tracks ball with strict rules:
- Max 5-frame prediction only
- No long interpolation
- States: DETECTED | PREDICTED | LOST
- Only DETECTED counts as strong evidence
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
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
    position: np.ndarray  # [x, y] in pitch coordinates
    state: BallState
    confidence: float = 1.0


@dataclass
class BallTrack:
    """Ball track with strict constraints."""
    ball_id: int
    current_state: BallState
    position: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    last_detected_frame: Optional[int] = None
    prediction_count: int = 0  # How many frames since last detection
    history: deque = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = deque(maxlen=10)  # Keep last 10 observations


class StrictBallTracker:
    """
    Strict ball tracker with no hallucinations.
    
    Rules:
    - Max 5-frame prediction
    - No forward/backward fill beyond prediction window
    - Missing ball = valid LOST state
    - Only DETECTED positions are trusted
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the ball tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            # Prediction limits
            'max_prediction_frames': 5,  # Max frames to predict after detection
            
            # Motion constraints
            'max_ball_speed': 30.0,      # m/s (unrealistic if exceeded)
            'max_position_jump': 5.0,    # meters per frame
            
            # Tracking
            'min_detection_confidence': 0.3,
            'tracker_type': 'kalman',    # 'kalman' or 'simple'
            
            # Frame rate
            'fps': 30.0,
        }
        
        if config:
            self.config.update(config)
        
        # Ball track (single ball)
        self.ball_track: Optional[BallTrack] = None
        
        # Detection history
        self.detection_history: List[BallObservation] = []
        
        # Simple Kalman filter for prediction
        self._init_kalman()
    
    def _init_kalman(self):
        """Initialize simple Kalman filter for ball prediction."""
        # Simple 2D constant velocity model
        self.kalman_state = None  # [x, y, vx, vy]
        self.kalman_covariance = None
    
    def _update_kalman(self, position: np.ndarray, is_detection: bool):
        """
        Update Kalman filter state.
        
        Args:
            position: Ball position [x, y]
            is_detection: True if this is a detection, False if prediction
        """
        if is_detection:
            # Reset with detection
            self.kalman_state = np.array([position[0], position[1], 0.0, 0.0])
            self.kalman_covariance = np.eye(4) * 10.0
        else:
            # Predict next state (constant velocity)
            if self.kalman_state is not None:
                dt = 1.0 / self.config['fps']
                # State: [x, y, vx, vy]
                x, y, vx, vy = self.kalman_state
                
                # Predict position
                new_x = x + vx * dt
                new_y = y + vy * dt
                
                # Update state
                self.kalman_state = np.array([new_x, new_y, vx, vy])
                
                # Increase uncertainty
                self.kalman_covariance += np.eye(4) * 0.1
    
    def _predict_position(self) -> Optional[np.ndarray]:
        """Predict ball position using Kalman filter."""
        if self.kalman_state is None:
            return None
        
        x, y, _, _ = self.kalman_state
        return np.array([x, y])
    
    def update(self, frame: int, ball_detections: Optional[np.ndarray]) -> BallObservation:
        """
        Update ball tracker with new detections.
        
        Args:
            frame: Current frame number
            ball_detections: Array of ball detections [N, 2] or None
            
        Returns:
            BallObservation for current frame
        """
        # Check for detections
        detected_position = None
        if ball_detections is not None and len(ball_detections) > 0:
            # Use first detection (or closest to predicted position)
            if len(ball_detections) == 1:
                detected_position = ball_detections[0]
            else:
                # Multiple detections - use closest to predicted position
                predicted = self._predict_position()
                if predicted is not None:
                    distances = [np.linalg.norm(det - predicted) for det in ball_detections]
                    detected_position = ball_detections[np.argmin(distances)]
                else:
                    detected_position = ball_detections[0]
        
        # Update state
        if detected_position is not None:
            # DETECTED
            self.ball_track = BallTrack(
                ball_id=0,
                current_state=BallState.DETECTED,
                position=detected_position.copy(),
                last_detected_frame=frame,
                prediction_count=0
            )
            
            # Update Kalman with detection
            self._update_kalman(detected_position, is_detection=True)
            
            # Calculate velocity from history
            if len(self.detection_history) > 0:
                last_obs = self.detection_history[-1]
                if last_obs.state == BallState.DETECTED:
                    dt = (frame - last_obs.frame) / self.config['fps']
                    if dt > 0:
                        velocity = (detected_position - last_obs.position) / dt
                        # Check for unrealistic speed
                        speed = np.linalg.norm(velocity)
                        if speed <= self.config['max_ball_speed']:
                            self.ball_track.velocity = velocity
                            # Update Kalman velocity
                            if self.kalman_state is not None:
                                self.kalman_state[2] = velocity[0]
                                self.kalman_state[3] = velocity[1]
            
            obs = BallObservation(
                frame=frame,
                position=detected_position,
                state=BallState.DETECTED,
                confidence=1.0
            )
            
        elif self.ball_track is not None and self.ball_track.last_detected_frame is not None:
            # Check if we can predict
            frames_since_detection = frame - self.ball_track.last_detected_frame
            
            if frames_since_detection <= self.config['max_prediction_frames']:
                # PREDICTED (within limit)
                predicted_position = self._predict_position()
                
                if predicted_position is not None:
                    self.ball_track.current_state = BallState.PREDICTED
                    self.ball_track.position = predicted_position
                    self.ball_track.prediction_count = frames_since_detection
                    
                    # Update Kalman (prediction only)
                    self._update_kalman(predicted_position, is_detection=False)
                    
                    obs = BallObservation(
                        frame=frame,
                        position=predicted_position,
                        state=BallState.PREDICTED,
                        confidence=max(0.0, 1.0 - frames_since_detection / self.config['max_prediction_frames'])
                    )
                else:
                    # Can't predict
                    self.ball_track.current_state = BallState.LOST
                    obs = BallObservation(
                        frame=frame,
                        position=None,
                        state=BallState.LOST,
                        confidence=0.0
                    )
            else:
                # LOST (beyond prediction limit)
                self.ball_track.current_state = BallState.LOST
                obs = BallObservation(
                    frame=frame,
                    position=None,
                    state=BallState.LOST,
                    confidence=0.0
                )
        else:
            # No track, no detection
            obs = BallObservation(
                frame=frame,
                position=None,
                state=BallState.LOST,
                confidence=0.0
            )
        
        # Store observation
        self.detection_history.append(obs)
        if len(self.detection_history) > 100:
            self.detection_history = self.detection_history[-100:]
        
        return obs
    
    def get_ball_position(self, frame: int) -> Optional[BallObservation]:
        """
        Get ball position for a specific frame.
        
        Args:
            frame: Frame number
            
        Returns:
            BallObservation or None if not available
        """
        # Find observation in history
        for obs in reversed(self.detection_history):
            if obs.frame == frame:
                return obs
        
        return None
    
    def get_current_ball(self) -> Optional[BallObservation]:
        """Get current ball observation."""
        if self.detection_history:
            return self.detection_history[-1]
        return None
    
    def is_detected(self, frame: int) -> bool:
        """Check if ball is DETECTED (not predicted) at frame."""
        obs = self.get_ball_position(frame)
        return obs is not None and obs.state == BallState.DETECTED
    
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
    
    def reset(self):
        """Reset tracker state."""
        self.ball_track = None
        self.detection_history = []
        self._init_kalman()

