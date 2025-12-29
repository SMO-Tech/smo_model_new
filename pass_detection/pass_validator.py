"""
Pass Validator Module

Validates passes against physics constraints and realism checks.
All validation logic lives in this single module.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of pass validation."""
    is_valid: bool
    rejection_reason: Optional[str] = None
    distance_valid: bool = True
    speed_valid: bool = True
    direction_valid: bool = True
    teleport_valid: bool = True
    time_valid: bool = True
    
    # Computed values for debugging
    distance: float = 0.0
    implied_speed: float = 0.0
    direction_angle: float = 0.0
    duration_seconds: float = 0.0


class PassValidator:
    """
    Validates passes against physics constraints.
    
    All passes must satisfy:
    - Distance realism (min/max meters)
    - Speed realism (derived from time window)
    - Direction alignment (motion vector vs A→B vector)
    - No teleportation (sudden spatial jumps)
    - Time constraints (min/max duration)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            # Distance constraints (meters)
            'min_pass_distance': 3.0,
            'max_pass_distance': 40.0,
            
            # Speed constraints (m/s)
            'min_pass_speed': 2.0,     # Too slow = not a real pass
            'max_pass_speed': 35.0,    # Unrealistic if exceeded
            
            # Direction constraints (degrees)
            'max_angle_deviation': 60.0,  # Max angle between motion and A→B
            
            # Time constraints (seconds)
            'min_pass_time': 0.2,      # Minimum pass duration
            'max_pass_time': 3.0,      # Maximum pass duration
            
            # Teleportation detection (meters per frame)
            'max_position_jump': 5.0,
            
            # Frame rate
            'fps': 30.0,
        }
        
        if config:
            self.config.update(config)
    
    def validate_pass(self, initiator_pos: np.ndarray, receiver_pos: np.ndarray,
                     initiator_direction: Optional[np.ndarray],
                     start_frame: int, end_frame: int) -> ValidationResult:
        """
        Validate a potential pass.
        
        Args:
            initiator_pos: Position of pass initiator [x, y]
            receiver_pos: Position of receiver [x, y]
            initiator_direction: Direction vector of initiator at time of pass
            start_frame: Frame when pass started
            end_frame: Frame when pass ends
            
        Returns:
            ValidationResult with pass/fail and details
        """
        result = ValidationResult(is_valid=True)
        
        # Calculate basic metrics
        result.distance = float(np.linalg.norm(receiver_pos - initiator_pos))
        result.duration_seconds = (end_frame - start_frame) / self.config['fps']
        
        if result.duration_seconds > 0:
            result.implied_speed = result.distance / result.duration_seconds
        else:
            result.implied_speed = float('inf')
        
        # Calculate direction angle
        if initiator_direction is not None and np.linalg.norm(initiator_direction) > 0:
            pass_direction = receiver_pos - initiator_pos
            if np.linalg.norm(pass_direction) > 0:
                pass_direction_norm = pass_direction / np.linalg.norm(pass_direction)
                initiator_direction_norm = initiator_direction / np.linalg.norm(initiator_direction)
                cos_angle = np.dot(pass_direction_norm, initiator_direction_norm)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                result.direction_angle = float(np.arccos(cos_angle) * 180.0 / np.pi)
        
        # Validate distance
        result.distance_valid = self._validate_distance(result.distance)
        if not result.distance_valid:
            result.is_valid = False
            if result.distance < self.config['min_pass_distance']:
                result.rejection_reason = f"Distance too short: {result.distance:.2f}m"
            else:
                result.rejection_reason = f"Distance too long: {result.distance:.2f}m"
        
        # Validate speed
        if result.is_valid:
            result.speed_valid = self._validate_speed(result.implied_speed)
            if not result.speed_valid:
                result.is_valid = False
                if result.implied_speed < self.config['min_pass_speed']:
                    result.rejection_reason = f"Speed too slow: {result.implied_speed:.2f}m/s"
                else:
                    result.rejection_reason = f"Speed unrealistic: {result.implied_speed:.2f}m/s"
        
        # Validate direction
        if result.is_valid and initiator_direction is not None:
            result.direction_valid = self._validate_direction(result.direction_angle)
            if not result.direction_valid:
                result.is_valid = False
                result.rejection_reason = f"Direction mismatch: {result.direction_angle:.1f}°"
        
        # Validate time
        if result.is_valid:
            result.time_valid = self._validate_time(result.duration_seconds)
            if not result.time_valid:
                result.is_valid = False
                if result.duration_seconds < self.config['min_pass_time']:
                    result.rejection_reason = f"Too fast: {result.duration_seconds:.3f}s"
                else:
                    result.rejection_reason = f"Too slow: {result.duration_seconds:.3f}s"
        
        return result
    
    def _validate_distance(self, distance: float) -> bool:
        """Check if distance is within valid range."""
        return self.config['min_pass_distance'] <= distance <= self.config['max_pass_distance']
    
    def _validate_speed(self, speed: float) -> bool:
        """Check if implied speed is realistic."""
        return self.config['min_pass_speed'] <= speed <= self.config['max_pass_speed']
    
    def _validate_direction(self, angle: float) -> bool:
        """Check if direction alignment is acceptable."""
        return angle <= self.config['max_angle_deviation']
    
    def _validate_time(self, duration: float) -> bool:
        """Check if pass duration is within valid range."""
        return self.config['min_pass_time'] <= duration <= self.config['max_pass_time']
    
    def validate_receiver_candidate(self, initiator_pos: np.ndarray, 
                                   receiver_pos: np.ndarray,
                                   initiator_direction: Optional[np.ndarray],
                                   initiator_team: int,
                                   receiver_team: int) -> Tuple[bool, str]:
        """
        Validate a potential receiver candidate.
        
        Args:
            initiator_pos: Position of initiator
            receiver_pos: Position of receiver candidate
            initiator_direction: Direction vector of initiator
            initiator_team: Team ID of initiator
            receiver_team: Team ID of receiver
            
        Returns:
            (is_valid, rejection_reason)
        """
        # Must be same team
        if initiator_team != receiver_team:
            return False, "Different team"
        
        # Check distance
        distance = np.linalg.norm(receiver_pos - initiator_pos)
        if distance < self.config['min_pass_distance']:
            return False, f"Too close: {distance:.2f}m"
        if distance > self.config['max_pass_distance']:
            return False, f"Too far: {distance:.2f}m"
        
        # Check direction alignment
        if initiator_direction is not None and np.linalg.norm(initiator_direction) > 0:
            pass_direction = receiver_pos - initiator_pos
            if np.linalg.norm(pass_direction) > 0:
                pass_direction_norm = pass_direction / np.linalg.norm(pass_direction)
                initiator_direction_norm = initiator_direction / np.linalg.norm(initiator_direction)
                cos_angle = np.dot(pass_direction_norm, initiator_direction_norm)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = float(np.arccos(cos_angle) * 180.0 / np.pi)
                
                if angle > self.config['max_angle_deviation']:
                    return False, f"Direction mismatch: {angle:.1f}°"
        
        return True, ""
    
    def check_teleportation(self, positions: List[Tuple[int, np.ndarray]]) -> bool:
        """
        Check if position history contains teleportation.
        
        Args:
            positions: List of (frame, position) tuples
            
        Returns:
            True if teleportation detected
        """
        if len(positions) < 2:
            return False
        
        for i in range(1, len(positions)):
            frame_diff = positions[i][0] - positions[i-1][0]
            if frame_diff <= 0:
                continue
            
            distance = np.linalg.norm(positions[i][1] - positions[i-1][1])
            max_distance = self.config['max_position_jump'] * frame_diff
            
            if distance > max_distance:
                return True
        
        return False

