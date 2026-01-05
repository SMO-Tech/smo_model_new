"""
Physics-Based Predictive Ball Tracker

Replaces FSM-based tracking with continuous physics-based prediction:
- Maintains single-ball identity (no duplication)
- Uses velocity and acceleration for motion prediction
- Applies friction/deceleration for realistic motion
- Adaptive detection gating (spatial + velocity matching)
- Always provides position (interpolated if needed)
- No FSM states - all logic is continuous and prediction-driven
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class BallState(Enum):
    """Ball tracking states (for compatibility, but not used in FSM logic)."""
    DETECTED = "detected"      # Ball detected by YOLO
    PREDICTED = "predicted"    # Ball predicted (physics-based)
    INTERPOLATED = "interpolated"  # Ball interpolated (gap filling)


@dataclass
class BallObservation:
    """Single ball observation."""
    frame: int
    position: np.ndarray  # [x, y] - ALWAYS available (never None)
    state: BallState
    confidence: float = 1.0
    velocity: Optional[np.ndarray] = None  # [vx, vy] in pixels/frame
    acceleration: Optional[np.ndarray] = None  # [ax, ay] in pixels/frame²
    is_predicted: bool = False  # True if position is predicted/interpolated


@dataclass
class BallPhysicsState:
    """Continuous ball state (no FSM)."""
    position: np.ndarray  # [x, y] in pixels
    velocity: np.ndarray  # [vx, vy] in pixels/frame
    acceleration: np.ndarray  # [ax, ay] in pixels/frame²
    last_detected_frame: Optional[int] = None
    last_detected_position: Optional[np.ndarray] = None
    frames_since_detection: int = 0
    confidence: float = 1.0  # Decreases with prediction depth
    history: deque = field(default_factory=lambda: deque(maxlen=60))  # 2 seconds at 30fps


class PhysicsBallTracker:
    """
    Physics-based predictive ball tracker.
    
    Key features:
    1. Single-ball identity (no duplication)
    2. Continuous prediction using velocity/acceleration
    3. Friction/deceleration for realistic motion
    4. Adaptive detection gating (spatial + velocity matching)
    5. Interpolation for missing frames
    6. No FSM - all logic is continuous
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the physics-based ball tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            # Frame rate
            'fps': 30.0,
            
            # Physics parameters
            'friction_coefficient': 0.95,  # Per-frame velocity decay (0.95 = 5% reduction)
            'gravity': 0.0,  # Vertical acceleration (pixels/frame²) - set to 0 for 2D tracking
            'max_velocity': 2000.0,  # pixels/frame (very high for fast passes)
            'max_acceleration': 500.0,  # pixels/frame²
            
            # Detection gating (adaptive)
            'base_gate_radius': 150.0,  # Base spatial gate radius (pixels)
            'gate_velocity_factor': 2.0,  # Gate scales with velocity
            'gate_velocity_threshold': 10.0,  # pixels/frame - above this, gate expands
            'max_gate_radius': 500.0,  # Maximum gate radius
            'velocity_match_threshold': 0.5,  # Cosine similarity for velocity matching (0-1)
            
            # Prediction limits
            'max_prediction_frames': 60,  # 2 seconds at 30fps - much longer than FSM
            'confidence_decay_rate': 0.02,  # Per-frame confidence decay when predicting
            
            # Interpolation
            'interpolate_gaps': True,
            'max_interpolation_gap': 30,  # Max frames to interpolate (1 second)
            
            # Smoothing
            'smoothing_enabled': True,
            'smoothing_alpha': 0.7,  # Exponential smoothing factor (0-1, higher = more smoothing)
            
            # Initialization
            'min_detections_to_init': 1,  # Minimum detections to start tracking
        }
        
        if config:
            self.config.update(config)
        
        # Ball state (single ball, always exists once initialized)
        self.ball_state: Optional[BallPhysicsState] = None
        
        # Detection history
        self.detection_history: List[BallObservation] = []
        self.frame_observations: Dict[int, BallObservation] = {}
        
        # Recent detections for velocity estimation
        self.recent_detections: deque = deque(maxlen=5)
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'detected_frames': 0,
            'predicted_frames': 0,
            'interpolated_frames': 0,
            'rejected_detections': 0,
            'gated_detections': 0,
        }
    
    def _calculate_gate_radius(self, velocity: np.ndarray) -> float:
        """
        Calculate adaptive gate radius based on ball velocity.
        
        Args:
            velocity: Current ball velocity [vx, vy]
            
        Returns:
            Gate radius in pixels
        """
        speed = np.linalg.norm(velocity)
        
        if speed < self.config['gate_velocity_threshold']:
            return self.config['base_gate_radius']
        
        # Scale gate with velocity
        velocity_factor = 1.0 + (speed / self.config['gate_velocity_threshold']) * self.config['gate_velocity_factor']
        gate_radius = self.config['base_gate_radius'] * velocity_factor
        
        return min(gate_radius, self.config['max_gate_radius'])
    
    def _is_detection_valid(self, detection: np.ndarray, frame: int) -> Tuple[bool, str]:
        """
        Check if a YOLO detection is valid using adaptive gating.
        
        Args:
            detection: Detected position [x, y]
            frame: Current frame number
            
        Returns:
            (is_valid, reason)
        """
        if self.ball_state is None:
            # No ball state yet - accept first detection
            return True, "initial_detection"
        
        # Get predicted position
        predicted_pos = self.ball_state.position
        predicted_vel = self.ball_state.velocity
        
        # Calculate spatial distance
        spatial_distance = np.linalg.norm(detection - predicted_pos)
        
        # Calculate adaptive gate radius
        gate_radius = self._calculate_gate_radius(predicted_vel)
        
        # Check spatial gate
        if spatial_distance > gate_radius:
            self.stats['rejected_detections'] += 1
            return False, f"outside_gate_{spatial_distance:.1f}px_vs_{gate_radius:.1f}px"
        
        # Check velocity matching (if we have recent detections)
        if len(self.recent_detections) >= 2:
            # Estimate velocity from detection
            last_pos = self.recent_detections[-1][1]
            frames_diff = frame - self.recent_detections[-1][0]
            
            if frames_diff > 0:
                detected_velocity = (detection - last_pos) / frames_diff
                
                # Normalize velocities for comparison
                pred_vel_norm = np.linalg.norm(predicted_vel)
                det_vel_norm = np.linalg.norm(detected_velocity)
                
                if pred_vel_norm > 0.1 and det_vel_norm > 0.1:
                    # Calculate cosine similarity
                    pred_unit = predicted_vel / pred_vel_norm
                    det_unit = detected_velocity / det_vel_norm
                    similarity = np.dot(pred_unit, det_unit)
                    
                    if similarity < self.config['velocity_match_threshold']:
                        self.stats['rejected_detections'] += 1
                        return False, f"velocity_mismatch_{similarity:.2f}"
        
        self.stats['gated_detections'] += 1
        return True, "valid"
    
    def _predict_next_position(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next ball position using physics (velocity + acceleration + friction).
        
        Returns:
            (predicted_position, predicted_velocity)
        """
        if self.ball_state is None:
            raise ValueError("Cannot predict without ball state")
        
        pos = self.ball_state.position
        vel = self.ball_state.velocity
        acc = self.ball_state.acceleration
        
        # Apply friction (velocity decay)
        vel = vel * self.config['friction_coefficient']
        
        # Apply acceleration
        vel = vel + acc
        
        # Clamp velocity to maximum
        speed = np.linalg.norm(vel)
        if speed > self.config['max_velocity']:
            vel = vel / speed * self.config['max_velocity']
        
        # Update position: x = x + v*dt (dt = 1 frame)
        new_pos = pos + vel
        
        # Update acceleration (decay towards zero)
        acc = acc * 0.9  # Decay acceleration
        
        return new_pos, vel
    
    def _estimate_velocity_from_history(self) -> Optional[np.ndarray]:
        """
        Estimate velocity from recent detection history.
        
        Returns:
            Velocity [vx, vy] or None if insufficient data
        """
        if len(self.recent_detections) < 2:
            return None
        
        # Use most recent detections
        recent = list(self.recent_detections)[-3:]  # Last 3 detections
        
        if len(recent) < 2:
            return None
        
        # Calculate velocity from last two detections
        pos1, frame1 = recent[-2][1], recent[-2][0]
        pos2, frame2 = recent[-1][1], recent[-1][0]
        
        if frame2 == frame1:
            return None
        
        velocity = (pos2 - pos1) / (frame2 - frame1)
        return velocity
    
    def _estimate_acceleration_from_history(self) -> Optional[np.ndarray]:
        """
        Estimate acceleration from recent detection history.
        
        Returns:
            Acceleration [ax, ay] or None if insufficient data
        """
        if len(self.recent_detections) < 3:
            return None
        
        # Use last 3 detections
        recent = list(self.recent_detections)[-3:]
        
        if len(recent) < 3:
            return None
        
        # Calculate velocities
        pos1, frame1 = recent[0][1], recent[0][0]
        pos2, frame2 = recent[1][1], recent[1][0]
        pos3, frame3 = recent[2][1], recent[2][0]
        
        if frame2 == frame1 or frame3 == frame2:
            return None
        
        vel1 = (pos2 - pos1) / (frame2 - frame1)
        vel2 = (pos3 - pos2) / (frame3 - frame2)
        
        # Average time difference
        avg_dt = ((frame2 - frame1) + (frame3 - frame2)) / 2.0
        
        if avg_dt == 0:
            return None
        
        acceleration = (vel2 - vel1) / avg_dt
        
        # Clamp acceleration
        acc_mag = np.linalg.norm(acceleration)
        if acc_mag > self.config['max_acceleration']:
            acceleration = acceleration / acc_mag * self.config['max_acceleration']
        
        return acceleration
    
    def _interpolate_position(self, frame: int) -> Optional[np.ndarray]:
        """
        Interpolate ball position for a missing frame.
        
        Args:
            frame: Frame number to interpolate
            
        Returns:
            Interpolated position or None if interpolation not possible
        """
        if self.ball_state is None:
            return None
        
        # Check if we have recent detections on both sides
        if len(self.recent_detections) < 2:
            return None
        
        # Find detections before and after this frame
        before = None
        after = None
        
        for det_frame, det_pos in self.recent_detections:
            if det_frame < frame:
                if before is None or det_frame > before[0]:
                    before = (det_frame, det_pos)
            elif det_frame > frame:
                if after is None or det_frame < after[0]:
                    after = (det_frame, det_pos)
        
        if before is None or after is None:
            return None
        
        # Linear interpolation
        frame1, pos1 = before
        frame2, pos2 = after
        
        if frame2 == frame1:
            return pos1
        
        alpha = (frame - frame1) / (frame2 - frame1)
        interpolated = pos1 + alpha * (pos2 - pos1)
        
        return interpolated
    
    def _apply_smoothing(self, position: np.ndarray) -> np.ndarray:
        """
        Apply exponential smoothing to reduce jitter.
        
        Args:
            position: Current position
            
        Returns:
            Smoothed position
        """
        if not self.config['smoothing_enabled'] or self.ball_state is None:
            return position
        
        # Exponential smoothing: smoothed = alpha * current + (1-alpha) * previous
        alpha = self.config['smoothing_alpha']
        smoothed = alpha * position + (1 - alpha) * self.ball_state.position
        
        return smoothed
    
    def update(self, frame: int, ball_detections: Optional[np.ndarray]) -> BallObservation:
        """
        Update ball tracker with new detections.
        
        Args:
            frame: Current frame number
            ball_detections: Array of ball detections [N, 2] in pixels, or None
            
        Returns:
            BallObservation for current frame (position ALWAYS available)
        """
        self.stats['total_frames'] += 1
        
        # Process YOLO detections
        accepted_detection = None
        
        if ball_detections is not None and len(ball_detections) > 0:
            # If multiple detections, prefer one closest to prediction
            if len(ball_detections) == 1:
                candidate = ball_detections[0]
            else:
                # Multiple detections - prefer closest to predicted position
                if self.ball_state is not None:
                    predicted = self.ball_state.position
                    distances = [np.linalg.norm(det - predicted) for det in ball_detections]
                    candidate = ball_detections[np.argmin(distances)]
                else:
                    candidate = ball_detections[0]
            
            # Validate detection using adaptive gating
            is_valid, reason = self._is_detection_valid(candidate, frame)
            
            if is_valid:
                accepted_detection = candidate.copy()
                self.recent_detections.append((frame, accepted_detection))
        
        # Update ball state
        if accepted_detection is not None:
            # YOLO detection accepted
            self.stats['detected_frames'] += 1
            
            # Apply smoothing
            smoothed_pos = self._apply_smoothing(accepted_detection)
            
            # Estimate velocity and acceleration from history
            velocity = self._estimate_velocity_from_history()
            acceleration = self._estimate_acceleration_from_history()
            
            if velocity is None:
                # First detection or insufficient history - initialize with zero velocity
                velocity = np.array([0.0, 0.0], dtype=np.float32)
            
            if acceleration is None:
                # Initialize with zero acceleration
                acceleration = np.array([0.0, 0.0], dtype=np.float32)
            
            # Update or initialize ball state
            if self.ball_state is None:
                # Initialize ball state
                self.ball_state = BallPhysicsState(
                    position=smoothed_pos,
                    velocity=velocity,
                    acceleration=acceleration,
                    last_detected_frame=frame,
                    last_detected_position=accepted_detection,
                    frames_since_detection=0,
                    confidence=1.0
                )
            else:
                # Update existing state
                self.ball_state.position = smoothed_pos
                self.ball_state.velocity = velocity
                self.ball_state.acceleration = acceleration
                self.ball_state.last_detected_frame = frame
                self.ball_state.last_detected_position = accepted_detection
                self.ball_state.frames_since_detection = 0
                self.ball_state.confidence = 1.0
            
            # Add to history
            self.ball_state.history.append((frame, smoothed_pos.copy()))
            
            # Create observation
            obs = BallObservation(
                frame=frame,
                position=smoothed_pos,
                state=BallState.DETECTED,
                confidence=1.0,
                velocity=velocity,
                acceleration=acceleration,
                is_predicted=False
            )
        
        elif self.ball_state is not None:
            # No valid detection - use prediction
            frames_since = self.ball_state.frames_since_detection + 1
            
            if frames_since <= self.config['max_prediction_frames']:
                # Predict next position using physics
                predicted_pos, predicted_vel = self._predict_next_position()
                
                # Update ball state
                self.ball_state.position = predicted_pos
                self.ball_state.velocity = predicted_vel
                self.ball_state.frames_since_detection = frames_since
                
                # Decay confidence
                self.ball_state.confidence = max(0.0, 1.0 - frames_since * self.config['confidence_decay_rate'])
                
                # Add to history
                self.ball_state.history.append((frame, predicted_pos.copy()))
                
                self.stats['predicted_frames'] += 1
                
                obs = BallObservation(
                    frame=frame,
                    position=predicted_pos,
                    state=BallState.PREDICTED,
                    confidence=self.ball_state.confidence,
                    velocity=predicted_vel,
                    acceleration=self.ball_state.acceleration,
                    is_predicted=True
                )
            else:
                # Beyond prediction limit - try interpolation
                if self.config['interpolate_gaps']:
                    interpolated_pos = self._interpolate_position(frame)
                    
                    if interpolated_pos is not None:
                        self.ball_state.position = interpolated_pos
                        self.ball_state.frames_since_detection = frames_since
                        self.ball_state.confidence = 0.5  # Lower confidence for interpolation
                        
                        self.stats['interpolated_frames'] += 1
                        
                        obs = BallObservation(
                            frame=frame,
                            position=interpolated_pos,
                            state=BallState.INTERPOLATED,
                            confidence=0.5,
                            velocity=self.ball_state.velocity,
                            acceleration=self.ball_state.acceleration,
                            is_predicted=True
                        )
                    else:
                        # Can't interpolate - use last known position (ball stopped)
                        obs = BallObservation(
                            frame=frame,
                            position=self.ball_state.position.copy(),
                            state=BallState.INTERPOLATED,
                            confidence=0.3,
                            velocity=np.array([0.0, 0.0], dtype=np.float32),
                            acceleration=np.array([0.0, 0.0], dtype=np.float32),
                            is_predicted=True
                        )
                else:
                    # No interpolation - use last position
                    obs = BallObservation(
                        frame=frame,
                        position=self.ball_state.position.copy(),
                        state=BallState.INTERPOLATED,
                        confidence=0.3,
                        velocity=np.array([0.0, 0.0], dtype=np.float32),
                        acceleration=np.array([0.0, 0.0], dtype=np.float32),
                        is_predicted=True
                    )
        else:
            # No ball state and no detection - return default position (center of field)
            # This should rarely happen, but ensures we always return a position
            default_pos = np.array([640.0, 360.0], dtype=np.float32)  # Assuming 1280x720 video
            
            obs = BallObservation(
                frame=frame,
                position=default_pos,
                state=BallState.INTERPOLATED,
                confidence=0.0,
                velocity=np.array([0.0, 0.0], dtype=np.float32),
                acceleration=np.array([0.0, 0.0], dtype=np.float32),
                is_predicted=True
            )
        
        # Store observation
        self.detection_history.append(obs)
        self.frame_observations[frame] = obs
        
        # Trim history to prevent memory issues
        if len(self.detection_history) > 2000:
            old_obs = self.detection_history[:-1000]
            self.detection_history = self.detection_history[-1000:]
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
        """Check if ball position is available (ALWAYS True for physics tracker)."""
        obs = self.get_ball_position(frame)
        return obs is not None and obs.position is not None
    
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
        """Get the fraction of frames where ball position is available (should be 1.0)."""
        if self.stats['total_frames'] == 0:
            return 0.0
        return 1.0  # Physics tracker always provides position
    
    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        return {
            **self.stats,
            'detection_rate': self.get_detection_rate(),
            'available_rate': self.get_available_rate(),
        }
    
    def reset(self):
        """Reset tracker state."""
        self.ball_state = None
        self.detection_history = []
        self.frame_observations = {}
        self.recent_detections.clear()
        self.stats = {k: 0 for k in self.stats}

