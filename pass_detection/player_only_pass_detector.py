"""
Player-Only Pass Detection Module

Detects passes between players using motion physics and spatiotemporal reasoning,
without any ball tracking or ball-based detection.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))


@dataclass
class PassEvent:
    """Represents a detected pass event."""
    from_player_id: int
    to_player_id: int
    team_id: int
    start_frame: int
    end_frame: int
    start_position: List[float]
    end_position: List[float]
    confidence: float


class PlayerOnlyPassDetector:
    """
    Detects passes between players using motion physics and spatiotemporal reasoning.
    
    No ball tracking or ball-based detection is used. Passes are inferred purely
    from player motion patterns, direction vectors, and temporal relationships.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the pass detector with configuration.
        
        Args:
            config: Configuration dictionary. If None, uses defaults.
        """
        # Default configuration
        self.config = {
            # Pass initiation thresholds
            'velocity_spike_threshold': 1.5,  # m/s increase to trigger pass initiation
            'acceleration_threshold': 2.0,   # m/s² to detect sudden movement
            'direction_stability_frames': 2,  # Frames direction must be stable
            
            # Candidate receiver selection
            'min_pass_distance': 3.0,       # meters
            'max_pass_distance': 25.0,       # meters
            'max_angle_deviation': 45.0,     # degrees
            
            # Temporal validation
            'min_pass_time': 0.3,            # seconds (9 frames at 30fps)
            'max_pass_time': 2.5,            # seconds (75 frames at 30fps)
            'reception_velocity_change': 0.5, # m/s change to detect reception
            
            # Physics constraints
            'max_pass_speed': 30.0,          # m/s (unrealistic if exceeded)
            'max_position_jump': 5.0,         # meters (teleport detection)
            
            # Confidence scoring weights
            'initiation_weight': 0.3,
            'direction_weight': 0.25,
            'distance_weight': 0.2,
            'reception_weight': 0.25,
            
            # Output thresholds
            'min_confidence': 0.5,          # Minimum confidence to accept pass
            'fps': 30.0,                     # Video frame rate
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Internal state
        self.player_history: Dict[int, List[Dict]] = defaultdict(list)  # Track player positions over time
        self.pass_candidates: List[Dict] = []
        self.accepted_passes: List[PassEvent] = []
        self.rejected_passes: List[Dict] = []
        
    def _calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate Euclidean distance between two positions in meters."""
        return np.linalg.norm(pos1 - pos2)
    
    def _calculate_angle(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees."""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 180.0
        
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        angle = np.arccos(cos_angle) * 180.0 / np.pi
        return angle
    
    def _get_player_velocity(self, player_id: int, frames_back: int = 2) -> Optional[np.ndarray]:
        """Calculate player velocity vector from recent history."""
        history = self.player_history[player_id]
        if len(history) < frames_back + 1:
            return None
        
        current = history[-1]
        past = history[-frames_back-1]
        
        if current['position'] is None or past['position'] is None:
            return None
        
        dt = (current['frame'] - past['frame']) / self.config['fps']
        if dt <= 0:
            return None
        
        velocity = (np.array(current['position']) - np.array(past['position'])) / dt
        return velocity
    
    def _get_player_acceleration(self, player_id: int) -> Optional[float]:
        """Calculate player acceleration magnitude."""
        history = self.player_history[player_id]
        if len(history) < 3:
            return None
        
        v1 = self._get_player_velocity(player_id, frames_back=1)
        v2 = self._get_player_velocity(player_id, frames_back=2)
        
        if v1 is None or v2 is None:
            return None
        
        dt = 1.0 / self.config['fps']
        acceleration = (np.linalg.norm(v1) - np.linalg.norm(v2)) / dt
        return acceleration
    
    def _detect_pass_initiation(self, player_id: int, frame: int) -> bool:
        """
        Detect if a player is initiating a pass.
        
        Returns True if:
        - Sudden velocity or acceleration spike
        - Direction stabilizes for required frames
        - Not random sprinting
        """
        history = self.player_history[player_id]
        if len(history) < self.config['direction_stability_frames'] + 1:
            return False
        
        # Check velocity spike
        velocity = self._get_player_velocity(player_id)
        if velocity is None:
            return False
        
        velocity_magnitude = np.linalg.norm(velocity)
        
        # Check acceleration
        acceleration = self._get_player_acceleration(player_id)
        if acceleration is None:
            return False
        
        # Velocity spike check
        if len(history) >= 3:
            prev_velocity = self._get_player_velocity(player_id, frames_back=3)
            if prev_velocity is not None:
                prev_magnitude = np.linalg.norm(prev_velocity)
                velocity_change = velocity_magnitude - prev_magnitude
                if velocity_change < self.config['velocity_spike_threshold']:
                    return False
        
        # Acceleration check
        if acceleration < self.config['acceleration_threshold']:
            return False
        
        # Direction stability check
        if len(history) >= self.config['direction_stability_frames']:
            directions = []
            for i in range(1, self.config['direction_stability_frames'] + 1):
                if len(history) >= i + 1:
                    vel = self._get_player_velocity(player_id, frames_back=i)
                    if vel is not None and np.linalg.norm(vel) > 0.1:
                        directions.append(vel / np.linalg.norm(vel))
            
            if len(directions) >= 2:
                # Check if directions are similar
                angles = []
                for i in range(len(directions) - 1):
                    angle = self._calculate_angle(directions[i], directions[i+1])
                    angles.append(angle)
                
                if angles and max(angles) > 30.0:  # Direction not stable
                    return False
        
        return True
    
    def _find_candidate_receivers(self, initiator_id: int, initiator_pos: np.ndarray, 
                                  initiator_team: int, frame: int) -> List[Dict]:
        """
        Find potential pass receivers from teammates.
        
        Returns list of candidates with:
        - Distance in valid range
        - Angle alignment within threshold
        - Not moving away from initiator
        """
        candidates = []
        initiator_velocity = self._get_player_velocity(initiator_id)
        
        if initiator_velocity is None or np.linalg.norm(initiator_velocity) < 0.1:
            return candidates
        
        initiator_direction = initiator_velocity / np.linalg.norm(initiator_velocity)
        
        # Check all players in current frame
        for player_id, history in self.player_history.items():
            if player_id == initiator_id:
                continue
            
            if not history:
                continue
            
            current_state = history[-1]
            if current_state['frame'] != frame:
                continue
            
            # Must be same team
            if current_state['team_id'] != initiator_team:
                continue
            
            receiver_pos = np.array(current_state['position'])
            if receiver_pos is None:
                continue
            
            # Calculate distance
            distance = self._calculate_distance(initiator_pos, receiver_pos)
            if distance < self.config['min_pass_distance'] or distance > self.config['max_pass_distance']:
                continue
            
            # Calculate angle between initiator direction and vector to receiver
            to_receiver = receiver_pos - initiator_pos
            to_receiver_norm = to_receiver / np.linalg.norm(to_receiver)
            angle = self._calculate_angle(initiator_direction, to_receiver_norm)
            
            if angle > self.config['max_angle_deviation']:
                continue
            
            # Check if receiver is moving away
            receiver_velocity = self._get_player_velocity(player_id)
            if receiver_velocity is not None:
                to_receiver_normalized = to_receiver / np.linalg.norm(to_receiver)
                receiver_direction = receiver_velocity / np.linalg.norm(receiver_velocity) if np.linalg.norm(receiver_velocity) > 0.1 else None
                
                if receiver_direction is not None:
                    angle_to_receiver = self._calculate_angle(receiver_direction, to_receiver_normalized)
                    if angle_to_receiver > 90.0:  # Moving away
                        continue
            
            candidates.append({
                'player_id': player_id,
                'position': receiver_pos,
                'distance': distance,
                'angle': angle,
            })
        
        return candidates
    
    def _validate_reception(self, receiver_id: int, start_frame: int, end_frame: int) -> Tuple[bool, float]:
        """
        Validate if a receiver shows reception behavior.
        
        Returns (is_valid, confidence_score)
        """
        history = self.player_history[receiver_id]
        if len(history) < 2:
            return False, 0.0
        
        # Find frames in the time window
        frame_window = []
        for state in history:
            if start_frame <= state['frame'] <= end_frame:
                frame_window.append(state)
        
        if len(frame_window) < 2:
            return False, 0.0
        
        # Check for velocity change, deceleration, or direction change
        reception_signals = []
        
        for i in range(1, len(frame_window)):
            current = frame_window[i]
            previous = frame_window[i-1]
            
            if current['position'] is None or previous['position'] is None:
                continue
            
            dt = (current['frame'] - previous['frame']) / self.config['fps']
            if dt <= 0:
                continue
            
            velocity = (np.array(current['position']) - np.array(previous['position'])) / dt
            velocity_magnitude = np.linalg.norm(velocity)
            
            if i > 1:
                prev_velocity = (np.array(previous['position']) - np.array(frame_window[i-2]['position'])) / dt
                prev_magnitude = np.linalg.norm(prev_velocity) if prev_velocity is not None else 0
                
                # Velocity change
                if abs(velocity_magnitude - prev_magnitude) > self.config['reception_velocity_change']:
                    reception_signals.append(1.0)
                
                # Deceleration
                if velocity_magnitude < prev_magnitude:
                    reception_signals.append(0.8)
        
        if not reception_signals:
            return False, 0.0
        
        # Confidence based on number and strength of signals
        confidence = min(1.0, sum(reception_signals) / len(frame_window))
        return True, confidence
    
    def _apply_physics_constraints(self, initiator_pos: np.ndarray, receiver_pos: np.ndarray,
                                  start_frame: int, end_frame: int) -> Tuple[bool, str]:
        """
        Apply physics constraints to validate pass realism.
        
        Returns (is_valid, rejection_reason)
        """
        distance = self._calculate_distance(initiator_pos, receiver_pos)
        time_span = (end_frame - start_frame) / self.config['fps']
        
        if time_span <= 0:
            return False, "Invalid time span"
        
        # Calculate implied pass speed
        pass_speed = distance / time_span if time_span > 0 else float('inf')
        
        if pass_speed > self.config['max_pass_speed']:
            return False, f"Unrealistic pass speed: {pass_speed:.2f} m/s"
        
        # Check for position jumps (teleportation)
        if distance > self.config['max_position_jump'] * 2:  # Allow some margin
            return False, f"Position jump too large: {distance:.2f} m"
        
        # Time constraints
        if time_span < self.config['min_pass_time']:
            return False, f"Pass time too short: {time_span:.3f} s"
        
        if time_span > self.config['max_pass_time']:
            return False, f"Pass time too long: {time_span:.3f} s"
        
        return True, ""
    
    def _calculate_confidence(self, initiator_id: int, receiver_id: int,
                              initiator_pos: np.ndarray, receiver_pos: np.ndarray,
                              start_frame: int, end_frame: int) -> float:
        """
        Calculate confidence score for a pass detection.
        
        Combines:
        - Initiation clarity
        - Direction alignment
        - Distance realism
        - Reception strength
        """
        # Initiation clarity (based on velocity/acceleration)
        initiator_velocity = self._get_player_velocity(initiator_id)
        initiator_acceleration = self._get_player_acceleration(initiator_id)
        
        initiation_score = 0.0
        if initiator_velocity is not None:
            vel_magnitude = np.linalg.norm(initiator_velocity)
            initiation_score += min(1.0, vel_magnitude / 5.0) * 0.5
        if initiator_acceleration is not None:
            initiation_score += min(1.0, initiator_acceleration / 5.0) * 0.5
        
        # Direction alignment
        if initiator_velocity is not None and np.linalg.norm(initiator_velocity) > 0.1:
            initiator_direction = initiator_velocity / np.linalg.norm(initiator_velocity)
            to_receiver = receiver_pos - initiator_pos
            to_receiver_norm = to_receiver / np.linalg.norm(to_receiver)
            angle = self._calculate_angle(initiator_direction, to_receiver_norm)
            direction_score = 1.0 - (angle / self.config['max_angle_deviation'])
            direction_score = max(0.0, direction_score)
        else:
            direction_score = 0.0
        
        # Distance realism (closer to optimal distance is better)
        distance = self._calculate_distance(initiator_pos, receiver_pos)
        optimal_distance = (self.config['min_pass_distance'] + self.config['max_pass_distance']) / 2
        distance_score = 1.0 - abs(distance - optimal_distance) / optimal_distance
        distance_score = max(0.0, min(1.0, distance_score))
        
        # Reception strength
        is_valid, reception_confidence = self._validate_reception(receiver_id, start_frame, end_frame)
        reception_score = reception_confidence if is_valid else 0.0
        
        # Weighted combination
        confidence = (
            initiation_score * self.config['initiation_weight'] +
            direction_score * self.config['direction_weight'] +
            distance_score * self.config['distance_weight'] +
            reception_score * self.config['reception_weight']
        )
        
        return confidence
    
    def update_player_state(self, frame: int, player_id: int, position: np.ndarray, team_id: int):
        """
        Update player state for a frame.
        
        Args:
            frame: Current frame number
            player_id: Player tracking ID
            position: Player position in pitch coordinates [x, y]
            team_id: Team assignment (0 or 1)
        """
        self.player_history[player_id].append({
            'frame': frame,
            'position': position,
            'team_id': team_id,
        })
        
        # Keep only recent history (last 150 frames = 5 seconds at 30fps)
        max_history = int(self.config['max_pass_time'] * self.config['fps'] * 2)
        if len(self.player_history[player_id]) > max_history:
            self.player_history[player_id] = self.player_history[player_id][-max_history:]
    
    def process_frame(self, frame: int, player_tracks: Dict[int, Dict], 
                     player_team_assignments: Dict[int, int],
                     player_pitch_positions: Dict[int, np.ndarray]) -> List[PassEvent]:
        """
        Process a frame to detect passes.
        
        Args:
            frame: Current frame number
            player_tracks: Dictionary of {player_id: {bbox info}}
            player_team_assignments: Dictionary of {player_id: team_id}
            player_pitch_positions: Dictionary of {player_id: [x, y] pitch coordinates}
        
        Returns:
            List of detected PassEvent objects
        """
        detected_passes = []
        
        # Update player states
        for player_id, pitch_pos in player_pitch_positions.items():
            if player_id in player_team_assignments:
                team_id = player_team_assignments[player_id]
                self.update_player_state(frame, player_id, pitch_pos, team_id)
        
        # Check each player for pass initiation
        for player_id, pitch_pos in player_pitch_positions.items():
            if player_id not in player_team_assignments:
                continue
            
            team_id = player_team_assignments[player_id]
            
            # Detect pass initiation
            if self._detect_pass_initiation(player_id, frame):
                # Find candidate receivers
                candidates = self._find_candidate_receivers(
                    player_id, pitch_pos, team_id, frame
                )
                
                for candidate in candidates:
                    receiver_id = candidate['player_id']
                    receiver_pos = candidate['position']
                    
                    # Estimate end frame (based on distance and typical pass speed)
                    distance = candidate['distance']
                    estimated_time = distance / 15.0  # Assume ~15 m/s average pass speed
                    estimated_end_frame = frame + int(estimated_time * self.config['fps'])
                    estimated_end_frame = min(estimated_end_frame, frame + int(self.config['max_pass_time'] * self.config['fps']))
                    
                    # Store candidate
                    candidate_data = {
                        'initiator_id': player_id,
                        'receiver_id': receiver_id,
                        'team_id': team_id,
                        'start_frame': frame,
                        'estimated_end_frame': estimated_end_frame,
                        'start_position': pitch_pos.tolist(),
                        'end_position': receiver_pos.tolist(),
                        'distance': distance,
                        'angle': candidate['angle'],
                    }
                    self.pass_candidates.append(candidate_data)
        
        # Validate candidates that may have completed
        validated_passes = []
        for candidate in self.pass_candidates[:]:
            # Check if enough time has passed
            if frame >= candidate['estimated_end_frame']:
                # Validate reception
                is_valid, reception_conf = self._validate_reception(
                    candidate['receiver_id'],
                    candidate['start_frame'],
                    frame
                )
                
                if is_valid:
                    # Apply physics constraints
                    initiator_pos = np.array(candidate['start_position'])
                    receiver_pos = np.array(candidate['end_position'])
                    
                    physics_valid, rejection_reason = self._apply_physics_constraints(
                        initiator_pos, receiver_pos,
                        candidate['start_frame'], frame
                    )
                    
                    if physics_valid:
                        # Calculate confidence
                        confidence = self._calculate_confidence(
                            candidate['initiator_id'],
                            candidate['receiver_id'],
                            initiator_pos, receiver_pos,
                            candidate['start_frame'], frame
                        )
                        
                        if confidence >= self.config['min_confidence']:
                            # Create pass event
                            pass_event = PassEvent(
                                from_player_id=candidate['initiator_id'],
                                to_player_id=candidate['receiver_id'],
                                team_id=candidate['team_id'],
                                start_frame=candidate['start_frame'],
                                end_frame=frame,
                                start_position=candidate['start_position'],
                                end_position=candidate['end_position'],
                                confidence=confidence
                            )
                            validated_passes.append(pass_event)
                            self.accepted_passes.append(pass_event)
                        else:
                            self.rejected_passes.append({
                                **candidate,
                                'rejection_reason': f'Low confidence: {confidence:.3f}',
                                'confidence': confidence
                            })
                    else:
                        self.rejected_passes.append({
                            **candidate,
                            'rejection_reason': rejection_reason,
                            'confidence': 0.0
                        })
                else:
                    self.rejected_passes.append({
                        **candidate,
                        'rejection_reason': 'No reception behavior detected',
                        'confidence': reception_conf
                    })
                
                # Remove from candidates
                self.pass_candidates.remove(candidate)
        
        return validated_passes
    
    def save_debug_outputs(self, output_dir: Path):
        """
        Save debug outputs to CSV and JSON files.
        
        Args:
            output_dir: Directory to save debug files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save pass candidates
        if self.pass_candidates:
            candidates_df = pd.DataFrame(self.pass_candidates)
            candidates_df.to_csv(output_dir / 'pass_candidates.csv', index=False)
        
        # Save rejected passes
        if self.rejected_passes:
            rejected_df = pd.DataFrame(self.rejected_passes)
            rejected_df.to_csv(output_dir / 'rejected_passes.csv', index=False)
        
        # Save accepted passes
        if self.accepted_passes:
            accepted_data = [
                {
                    'from_player_id': p.from_player_id,
                    'to_player_id': p.to_player_id,
                    'team_id': p.team_id,
                    'start_frame': p.start_frame,
                    'end_frame': p.end_frame,
                    'start_position': p.start_position,
                    'end_position': p.end_position,
                    'confidence': p.confidence
                }
                for p in self.accepted_passes
            ]
            accepted_df = pd.DataFrame(accepted_data)
            accepted_df.to_csv(output_dir / 'accepted_passes.csv', index=False)
        
        # Save metrics
        total_candidates = len(self.pass_candidates) + len(self.accepted_passes) + len(self.rejected_passes)
        metrics = {
            'total_candidates': total_candidates,
            'accepted_passes': len(self.accepted_passes),
            'rejected_passes': len(self.rejected_passes),
            'pending_candidates': len(self.pass_candidates),
            'rejection_breakdown': {},
            'average_confidence': 0.0
        }
        
        if self.rejected_passes:
            rejection_reasons = [r.get('rejection_reason', 'Unknown') for r in self.rejected_passes]
            from collections import Counter
            metrics['rejection_breakdown'] = dict(Counter(rejection_reasons))
        
        if self.accepted_passes:
            metrics['average_confidence'] = np.mean([p.confidence for p in self.accepted_passes])
        
        with open(output_dir / 'pass_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Debug outputs saved to {output_dir}")
        print(f"  - Accepted passes: {len(self.accepted_passes)}")
        print(f"  - Rejected passes: {len(self.rejected_passes)}")
        print(f"  - Pending candidates: {len(self.pass_candidates)}")

