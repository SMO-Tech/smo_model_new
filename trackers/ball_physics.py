"""
Physics-Based Ball Tracker
Uses projectile motion equations to predict ball trajectory when detection fails
"""

import numpy as np
from collections import deque
from utils import get_center_of_bbox

class BallPhysicsTracker:
    def __init__(self, fps=30.0, max_history=10, max_gap_frames=15):
        """
        Args:
            fps: Video frame rate (for time calculations)
            max_history: Number of previous positions to keep for velocity calculation
            max_gap_frames: Maximum frames to predict ahead (beyond this, mark as lost)
        """
        self.fps = fps
        self.dt = 1.0 / fps  # Time between frames in seconds
        self.max_history = max_history
        self.max_gap_frames = max_gap_frames
        
        # Position history: [(x, y, frame_num), ...]
        self.position_history = deque(maxlen=max_history)
        
        # Velocity history: [(vx, vy), ...]
        self.velocity_history = deque(maxlen=max_history)
        
        # Current state
        self.last_position = None
        self.last_velocity = None
        self.last_acceleration = None
        self.gap_start_frame = None
        self.gap_frame_count = 0
        
    def update(self, bbox, frame_num):
        """
        Update tracker with new ball detection
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            frame_num: Current frame number
        """
        # Get center position
        x, y = get_center_of_bbox(bbox)
        
        # Calculate velocity if we have previous position
        if self.last_position is not None:
            prev_x, prev_y, prev_frame = self.last_position
            
            # Time difference
            frame_diff = frame_num - prev_frame
            if frame_diff > 0:
                dt = frame_diff * self.dt
                
                # Velocity in pixels per second
                vx = (x - prev_x) / dt
                vy = (y - prev_y) / dt
                
                self.velocity_history.append((vx, vy))
                
                # Calculate acceleration if we have velocity history
                if len(self.velocity_history) >= 2:
                    prev_vx, prev_vy = self.velocity_history[-2]
                    ax = (vx - prev_vx) / dt
                    ay = (vy - prev_vy) / dt
                    self.last_acceleration = (ax, ay)
                else:
                    self.last_acceleration = (0, 0)
                
                self.last_velocity = (vx, vy)
        
        # Update position history
        self.position_history.append((x, y, frame_num))
        self.last_position = (x, y, frame_num)
        
        # Reset gap tracking
        self.gap_start_frame = None
        self.gap_frame_count = 0
    
    def predict(self, frame_num):
        """
        Predict ball position using physics equations
        
        Args:
            frame_num: Frame number to predict for
            
        Returns:
            (x, y) predicted position or None if prediction not possible
        """
        if self.last_position is None:
            return None
        
        last_x, last_y, last_frame = self.last_position
        
        # Calculate frames since last detection
        frame_diff = frame_num - last_frame
        
        # Don't predict if gap is too large
        if frame_diff > self.max_gap_frames:
            return None
        
        # Time since last detection
        t = frame_diff * self.dt
        
        # If we have velocity, use constant velocity model
        if self.last_velocity is not None:
            vx, vy = self.last_velocity
            
            # Basic constant velocity prediction
            predicted_x = last_x + vx * t
            predicted_y = last_y + vy * t
            
            # If we have acceleration, use constant acceleration model
            if self.last_acceleration is not None:
                ax, ay = self.last_acceleration
                predicted_x = last_x + vx * t + 0.5 * ax * t * t
                predicted_y = last_y + vy * t + 0.5 * ay * t * t
            
            # If we have enough history, use more sophisticated prediction
            if len(self.velocity_history) >= 3:
                # Calculate average velocity and acceleration from history
                velocities = np.array(list(self.velocity_history))
                avg_vx = np.mean(velocities[:, 0])
                avg_vy = np.mean(velocities[:, 1])
                
                # Use weighted average (more recent = more weight)
                weights = np.linspace(0.5, 1.0, len(velocities))
                weights = weights / np.sum(weights)
                weighted_vx = np.sum(velocities[:, 0] * weights)
                weighted_vy = np.sum(velocities[:, 1] * weights)
                
                # Predict with weighted velocity
                predicted_x = last_x + weighted_vx * t
                predicted_y = last_y + weighted_vy * t
                
                # Apply deceleration (friction/air resistance effect)
                # Assume exponential decay: v(t) = v0 * exp(-k*t)
                # For football, k ≈ 0.1-0.3 (ball slows down)
                k = 0.15  # Deceleration coefficient
                decay_factor = np.exp(-k * t)
                predicted_x = last_x + weighted_vx * (1 - decay_factor) / k
                predicted_y = last_y + weighted_vy * (1 - decay_factor) / k
            
            return (int(predicted_x), int(predicted_y))
        
        return None
    
    def get_smoothed_velocity(self):
        """
        Get smoothed velocity from history using exponential moving average
        """
        if len(self.velocity_history) == 0:
            return None
        
        velocities = np.array(list(self.velocity_history))
        
        # Exponential moving average (more weight to recent)
        alpha = 0.7
        smoothed_vx = velocities[0, 0]
        smoothed_vy = velocities[0, 1]
        
        for vx, vy in velocities[1:]:
            smoothed_vx = alpha * vx + (1 - alpha) * smoothed_vx
            smoothed_vy = alpha * vy + (1 - alpha) * smoothed_vy
        
        return (smoothed_vx, smoothed_vy)
    
    def is_tracking(self):
        """Check if tracker has enough data to make predictions"""
        return self.last_position is not None and len(self.position_history) >= 2
    
    def reset(self):
        """Reset tracker state"""
        self.position_history.clear()
        self.velocity_history.clear()
        self.last_position = None
        self.last_velocity = None
        self.last_acceleration = None
        self.gap_start_frame = None
        self.gap_frame_count = 0

