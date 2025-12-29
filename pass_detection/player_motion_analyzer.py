"""
Player Motion Analyzer Module

Analyzes player motion for pass detection:
- Velocity calculation
- Acceleration detection
- Direction stability
- Movement patterns
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class PlayerMotionState:
    """Current motion state of a player."""
    player_id: int
    frame: int
    position: np.ndarray
    velocity: Optional[np.ndarray] = None
    speed: float = 0.0
    acceleration: float = 0.0
    direction: Optional[np.ndarray] = None
    direction_stable: bool = False
    direction_stability_frames: int = 0


class PlayerMotionAnalyzer:
    """
    Analyzes player motion patterns for pass detection.
    
    Tracks:
    - Position history
    - Velocity vectors
    - Acceleration patterns
    - Direction stability
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the motion analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            # Motion thresholds
            'velocity_spike_threshold': 1.5,    # m/s increase for pass initiation
            'acceleration_threshold': 2.0,       # m/s² for sudden movement
            'min_velocity_for_direction': 0.5,   # m/s minimum to consider direction
            
            # Direction stability
            'direction_stability_angle': 30.0,   # degrees max deviation for stability
            'direction_stability_frames': 2,     # frames direction must be stable
            
            # Teleportation detection
            'max_position_jump': 5.0,            # meters max jump per frame
            
            # History
            'max_history_frames': 150,           # frames to keep (5 seconds at 30fps)
            
            # Frame rate
            'fps': 30.0,
        }
        
        if config:
            self.config.update(config)
        
        # Position history: player_id -> list of (frame, position)
        self.position_history: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        
        # Computed motion states
        self.motion_states: Dict[int, PlayerMotionState] = {}
        
        # Direction history for stability check
        self.direction_history: Dict[int, List[np.ndarray]] = defaultdict(list)
    
    def update_player(self, player_id: int, frame: int, position: np.ndarray) -> PlayerMotionState:
        """
        Update player position and compute motion state.
        
        Args:
            player_id: Player tracking ID
            frame: Current frame number
            position: Player position [x, y] in pitch coordinates
            
        Returns:
            Updated PlayerMotionState
        """
        history = self.position_history[player_id]
        
        # Check for teleportation
        if history:
            last_frame, last_pos = history[-1]
            distance = np.linalg.norm(position - last_pos)
            frames_diff = frame - last_frame
            if frames_diff > 0 and distance / frames_diff > self.config['max_position_jump']:
                # Teleportation detected - clear history
                history.clear()
                self.direction_history[player_id].clear()
        
        # Add to history
        history.append((frame, position.copy()))
        
        # Trim history
        max_history = self.config['max_history_frames']
        if len(history) > max_history:
            self.position_history[player_id] = history[-max_history:]
            history = self.position_history[player_id]
        
        # Compute motion state
        state = self._compute_motion_state(player_id, frame, position, history)
        self.motion_states[player_id] = state
        
        return state
    
    def _compute_motion_state(self, player_id: int, frame: int, position: np.ndarray,
                             history: List[Tuple[int, np.ndarray]]) -> PlayerMotionState:
        """Compute full motion state from history."""
        state = PlayerMotionState(
            player_id=player_id,
            frame=frame,
            position=position
        )
        
        if len(history) < 2:
            return state
        
        # Compute velocity from recent frames
        velocity = self._compute_velocity(history, frames_back=2)
        if velocity is not None:
            state.velocity = velocity
            state.speed = float(np.linalg.norm(velocity))
            
            # Compute direction if moving fast enough
            if state.speed > self.config['min_velocity_for_direction']:
                state.direction = velocity / state.speed
                
                # Update direction history
                self.direction_history[player_id].append(state.direction.copy())
                if len(self.direction_history[player_id]) > 10:
                    self.direction_history[player_id] = self.direction_history[player_id][-10:]
                
                # Check direction stability
                state.direction_stable, state.direction_stability_frames = \
                    self._check_direction_stability(player_id)
        
        # Compute acceleration
        if len(history) >= 3:
            state.acceleration = self._compute_acceleration(history)
        
        return state
    
    def _compute_velocity(self, history: List[Tuple[int, np.ndarray]], 
                         frames_back: int = 2) -> Optional[np.ndarray]:
        """Compute velocity from position history."""
        if len(history) < frames_back + 1:
            return None
        
        current_frame, current_pos = history[-1]
        past_frame, past_pos = history[-frames_back - 1]
        
        dt = (current_frame - past_frame) / self.config['fps']
        if dt <= 0:
            return None
        
        return (current_pos - past_pos) / dt
    
    def _compute_acceleration(self, history: List[Tuple[int, np.ndarray]]) -> float:
        """Compute acceleration magnitude."""
        if len(history) < 4:
            return 0.0
        
        # Velocity at two time points
        v1 = self._compute_velocity(history[-3:], frames_back=2)
        v2 = self._compute_velocity(history[-4:-1], frames_back=2)
        
        if v1 is None or v2 is None:
            return 0.0
        
        dt = 2.0 / self.config['fps']  # Approximate time between velocity samples
        return (np.linalg.norm(v1) - np.linalg.norm(v2)) / dt
    
    def _check_direction_stability(self, player_id: int) -> Tuple[bool, int]:
        """
        Check if player's direction has been stable.
        
        Returns:
            (is_stable, num_stable_frames)
        """
        directions = self.direction_history[player_id]
        if len(directions) < self.config['direction_stability_frames']:
            return False, 0
        
        # Check recent directions
        recent = directions[-self.config['direction_stability_frames']:]
        
        stable_count = 0
        for i in range(len(recent) - 1):
            angle = self._angle_between(recent[i], recent[i + 1])
            if angle <= self.config['direction_stability_angle']:
                stable_count += 1
            else:
                stable_count = 0
        
        is_stable = stable_count >= self.config['direction_stability_frames'] - 1
        return is_stable, stable_count + 1 if is_stable else 0
    
    def _angle_between(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees."""
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 180.0
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.arccos(cos_angle) * 180.0 / np.pi)
    
    def detect_pass_initiation(self, player_id: int) -> Tuple[bool, float]:
        """
        Detect if a player is initiating a pass.
        
        Returns:
            (is_initiating, confidence)
        """
        state = self.motion_states.get(player_id)
        if state is None:
            return False, 0.0
        
        # Need velocity data
        if state.velocity is None:
            return False, 0.0
        
        # Check velocity spike
        history = self.position_history[player_id]
        if len(history) < 4:
            return False, 0.0
        
        # Compare current speed to recent average
        prev_velocity = self._compute_velocity(history[-4:-1], frames_back=2)
        if prev_velocity is None:
            return False, 0.0
        
        prev_speed = np.linalg.norm(prev_velocity)
        velocity_change = state.speed - prev_speed
        
        # Check thresholds
        has_velocity_spike = velocity_change > self.config['velocity_spike_threshold']
        has_acceleration = state.acceleration > self.config['acceleration_threshold']
        has_stable_direction = state.direction_stable
        
        # Must have at least velocity spike OR acceleration, and stable direction
        if not ((has_velocity_spike or has_acceleration) and has_stable_direction):
            return False, 0.0
        
        # Compute confidence
        confidence = 0.0
        if has_velocity_spike:
            confidence += 0.4 * min(1.0, velocity_change / (self.config['velocity_spike_threshold'] * 2))
        if has_acceleration:
            confidence += 0.3 * min(1.0, state.acceleration / (self.config['acceleration_threshold'] * 2))
        if has_stable_direction:
            confidence += 0.3
        
        return True, confidence
    
    def detect_reception_behavior(self, player_id: int, start_frame: int, 
                                  end_frame: int) -> Tuple[bool, float]:
        """
        Detect if a player shows reception behavior.
        
        Returns:
            (has_reception, confidence)
        """
        state = self.motion_states.get(player_id)
        if state is None:
            return False, 0.0
        
        history = self.position_history[player_id]
        
        # Find frames in window
        window = [(f, p) for f, p in history if start_frame <= f <= end_frame]
        if len(window) < 3:
            return False, 0.0
        
        # Check for velocity change or deceleration
        reception_signals = []
        
        for i in range(2, len(window)):
            f1, p1 = window[i - 2]
            f2, p2 = window[i - 1]
            f3, p3 = window[i]
            
            dt1 = (f2 - f1) / self.config['fps']
            dt2 = (f3 - f2) / self.config['fps']
            
            if dt1 > 0 and dt2 > 0:
                v1 = np.linalg.norm(p2 - p1) / dt1
                v2 = np.linalg.norm(p3 - p2) / dt2
                
                # Deceleration
                if v2 < v1 * 0.7:  # 30% slowdown
                    reception_signals.append(0.8)
                
                # Direction change
                dir1 = (p2 - p1) / np.linalg.norm(p2 - p1) if np.linalg.norm(p2 - p1) > 0.1 else None
                dir2 = (p3 - p2) / np.linalg.norm(p3 - p2) if np.linalg.norm(p3 - p2) > 0.1 else None
                
                if dir1 is not None and dir2 is not None:
                    angle = self._angle_between(dir1, dir2)
                    if angle > 45.0:  # Significant direction change
                        reception_signals.append(0.6)
        
        if not reception_signals:
            return False, 0.0
        
        confidence = min(1.0, sum(reception_signals) / len(window))
        return True, confidence
    
    def get_state(self, player_id: int) -> Optional[PlayerMotionState]:
        """Get current motion state for a player."""
        return self.motion_states.get(player_id)
    
    def get_position(self, player_id: int) -> Optional[np.ndarray]:
        """Get current position for a player."""
        state = self.motion_states.get(player_id)
        return state.position if state else None
    
    def get_direction(self, player_id: int) -> Optional[np.ndarray]:
        """Get current direction for a player."""
        state = self.motion_states.get(player_id)
        return state.direction if state else None

