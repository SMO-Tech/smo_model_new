"""
Ball Evidence Validator Module

Validates pass candidates using ball tracking evidence.
Ball can SUPPORT but never CREATE passes.

Key principle: "Players propose. Ball disposes."

Strict rules:
- Only DETECTED or PREDICTED (within 5-frame window) ball positions are used
- LOST ball state = automatic rejection of pass candidate
- Ball must be near initiator at start and receiver at end
- Ball must move in the direction of the pass
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .ball_tracker import StrictBallTracker, BallState, BallObservation
from .player_candidate_generator import PassCandidate


@dataclass
class BallValidationResult:
    """Result of ball evidence validation."""
    is_valid: bool
    evidence_score: float  # 0.0 to 1.0
    validation_details: Dict
    
    # Evidence components
    initiation_evidence: float = 0.0
    transit_evidence: float = 0.0
    reception_evidence: float = 0.0
    
    # Ball state info
    ball_available: bool = False  # True if ball is DETECTED or PREDICTED
    detected_count: int = 0
    predicted_count: int = 0
    lost_count: int = 0
    
    # Rejection reason
    rejection_reason: Optional[str] = None


class BallEvidenceValidator:
    """
    Validates pass candidates using ball tracking.
    
    Rules:
    - Only DETECTED ball positions count as strong evidence
    - PREDICTED positions provide weak evidence (within 5-frame limit)
    - LOST ball = automatic rejection
    - Ball can contradict and reject candidates
    - Ball must be near initiator at start and near receiver at end
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            # Proximity thresholds (meters in pitch coordinates)
            'initiation_proximity': 5.0,    # Ball must be within X meters of initiator
            'reception_proximity': 5.0,     # Ball must be within X meters of receiver
            
            # Time windows (frames)
            'initiation_window': 8,         # ±N frames around start
            'reception_window': 10,         # ±N frames around end
            
            # Speed constraints (m/s)
            'min_ball_speed': 1.0,          # Minimum speed for transit
            'max_ball_speed': 40.0,         # Maximum speed
            
            # Direction alignment (degrees)
            'max_direction_deviation': 75.0,  # Max angle between ball motion and pass direction
            
            # Evidence weights - prioritize initiation and reception
            'initiation_weight': 0.40,
            'transit_weight': 0.20,
            'reception_weight': 0.40,
            
            # Minimum evidence score to accept
            'min_evidence_score': 0.30,
            
            # Minimum available ball frames (DETECTED or PREDICTED)
            'min_available_frames': 2,
            
            # Maximum LOST frames allowed in pass window
            'max_lost_ratio': 0.6,  # If more than 60% of frames are LOST, reject
            
            # Frame rate
            'fps': 30.0,
        }
        
        if config:
            self.config.update(config)
    
    def validate_candidate(self, candidate: PassCandidate,
                          ball_tracker: StrictBallTracker) -> BallValidationResult:
        """
        Validate a pass candidate using ball evidence.
        
        Args:
            candidate: PassCandidate to validate
            ball_tracker: StrictBallTracker instance
            
        Returns:
            BallValidationResult
        """
        result = BallValidationResult(
            is_valid=False,
            evidence_score=0.0,
            validation_details={}
        )
        
        # Get ball observations in candidate window
        start_frame = candidate.start_frame_estimate
        end_frame = candidate.end_frame_estimate
        
        # Expand window for evidence gathering
        extended_start = max(0, start_frame - self.config['initiation_window'])
        extended_end = end_frame + self.config['reception_window']
        
        ball_observations = ball_tracker.get_ball_in_window(extended_start, extended_end)
        
        # Count states
        detected_count = sum(1 for obs in ball_observations if obs.state == BallState.DETECTED)
        predicted_count = sum(1 for obs in ball_observations if obs.state == BallState.PREDICTED)
        lost_count = sum(1 for obs in ball_observations if obs.state == BallState.LOST)
        total_count = len(ball_observations)
        
        result.detected_count = detected_count
        result.predicted_count = predicted_count
        result.lost_count = lost_count
        result.ball_available = (detected_count + predicted_count) > 0
        
        # Check if enough ball evidence
        available_count = detected_count + predicted_count
        if available_count < self.config['min_available_frames']:
            result.rejection_reason = "no_ball_evidence"
            result.is_valid = False
            return result
        
        # Check LOST ratio
        if total_count > 0:
            lost_ratio = lost_count / total_count
            if lost_ratio > self.config['max_lost_ratio']:
                result.rejection_reason = f"ball_mostly_lost_{lost_ratio:.1%}"
                result.is_valid = False
                return result
        
        # Get observations only in core window (not extended)
        core_observations = [obs for obs in ball_observations 
                            if start_frame <= obs.frame <= end_frame]
        
        # Check if ball is LOST during critical moments
        # Critical: start and end of pass
        start_obs = ball_tracker.get_ball_position(start_frame)
        end_obs = ball_tracker.get_ball_position(end_frame)
        
        # If ball is completely LOST at both start and end, reject
        if (start_obs is not None and start_obs.state == BallState.LOST and
            end_obs is not None and end_obs.state == BallState.LOST):
            result.rejection_reason = "ball_lost_at_critical_moments"
            result.is_valid = False
            return result
        
        # Validate initiation evidence
        result.initiation_evidence = self._validate_initiation(
            candidate, ball_observations, ball_tracker
        )
        
        # Validate transit evidence
        result.transit_evidence = self._validate_transit(
            candidate, core_observations
        )
        
        # Validate reception evidence
        result.reception_evidence = self._validate_reception(
            candidate, ball_observations, ball_tracker
        )
        
        # Calculate overall evidence score
        result.evidence_score = (
            result.initiation_evidence * self.config['initiation_weight'] +
            result.transit_evidence * self.config['transit_weight'] +
            result.reception_evidence * self.config['reception_weight']
        )
        
        # Check for contradictions
        contradiction = self._has_contradiction(candidate, core_observations)
        if contradiction:
            result.is_valid = False
            result.rejection_reason = f"contradictory_ball_motion: {contradiction}"
            return result
        
        # Accept if evidence is sufficient
        result.is_valid = result.evidence_score >= self.config['min_evidence_score']
        
        if not result.is_valid:
            result.rejection_reason = f"insufficient_evidence_{result.evidence_score:.2f}"
        
        result.validation_details = {
            'initiation_evidence': result.initiation_evidence,
            'transit_evidence': result.transit_evidence,
            'reception_evidence': result.reception_evidence,
            'total_score': result.evidence_score,
            'ball_observations_count': len(ball_observations),
            'detected_count': detected_count,
            'predicted_count': predicted_count,
            'lost_count': lost_count,
            'available_count': available_count,
        }
        
        return result
    
    def _validate_initiation(self, candidate: PassCandidate,
                            ball_observations: List[BallObservation],
                            ball_tracker: StrictBallTracker) -> float:
        """
        Validate ball proximity at initiation.
        
        Returns evidence score (0.0 to 1.0).
        """
        start_frame = candidate.start_frame_estimate
        window_start = start_frame - self.config['initiation_window']
        window_end = start_frame + self.config['initiation_window']
        
        best_evidence = 0.0
        
        for obs in ball_observations:
            if not (window_start <= obs.frame <= window_end):
                continue
            
            # Skip LOST observations
            if obs.state == BallState.LOST or obs.position is None:
                continue
            
            distance = np.linalg.norm(obs.position - candidate.initiator_position)
            proximity = self.config['initiation_proximity']
            
            if distance <= proximity:
                # Calculate base evidence (closer = higher)
                base_evidence = 1.0 - (distance / proximity)
                
                # Weight by observation state
                if obs.state == BallState.DETECTED:
                    evidence = base_evidence * 1.0  # Full weight for DETECTED
                elif obs.state == BallState.PREDICTED:
                    evidence = base_evidence * 0.6  # Reduced weight for PREDICTED
                else:
                    evidence = 0.0
                
                # Weight by temporal proximity to start frame
                frame_distance = abs(obs.frame - start_frame)
                temporal_weight = 1.0 - (frame_distance / self.config['initiation_window'])
                evidence *= max(0.3, temporal_weight)  # Minimum 30% weight
                
                best_evidence = max(best_evidence, evidence)
        
        return best_evidence
    
    def _validate_transit(self, candidate: PassCandidate,
                         ball_observations: List[BallObservation]) -> float:
        """
        Validate ball transit between initiator and receiver.
        
        Checks:
        1. Ball moves in correct direction
        2. Ball speed is realistic
        
        Returns evidence score (0.0 to 1.0).
        """
        # Get only DETECTED observations for transit (most reliable)
        detected_observations = [
            obs for obs in ball_observations 
            if obs.position is not None and obs.state == BallState.DETECTED
        ]
        
        if len(detected_observations) < 2:
            # Not enough DETECTED observations - provide weak evidence
            # Check if we have at least PREDICTED observations
            predicted_obs = [
                obs for obs in ball_observations
                if obs.position is not None and obs.state == BallState.PREDICTED
            ]
            if len(predicted_obs) >= 2:
                return 0.3  # Weak evidence with only predictions
            return 0.1  # Minimal evidence
        
        # Sort by frame
        detected_observations.sort(key=lambda x: x.frame)
        
        # Check direction alignment
        direction_scores = []
        speed_scores = []
        
        for i in range(len(detected_observations) - 1):
            obs1 = detected_observations[i]
            obs2 = detected_observations[i + 1]
            
            # Ball movement vector
            ball_dir = obs2.position - obs1.position
            ball_distance = np.linalg.norm(ball_dir)
            
            if ball_distance < 0.5:  # Ball barely moved
                continue
            
            ball_dir_norm = ball_dir / ball_distance
            
            # Candidate direction (from initiator to receiver)
            candidate_dir = candidate.direction_vector
            
            # Angle between directions
            cos_angle = np.dot(ball_dir_norm, candidate_dir)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180.0 / np.pi
            
            max_deviation = self.config['max_direction_deviation']
            if angle <= max_deviation:
                direction_scores.append(1.0 - (angle / max_deviation))
            else:
                direction_scores.append(0.0)
            
            # Check speed
            dt = (obs2.frame - obs1.frame) / self.config['fps']
            if dt > 0:
                speed = ball_distance / dt
                
                if self.config['min_ball_speed'] <= speed <= self.config['max_ball_speed']:
                    # Speed in valid range
                    speed_scores.append(1.0)
                elif speed < self.config['min_ball_speed']:
                    # Too slow - might be valid for short pass
                    speed_scores.append(0.5)
                else:
                    # Too fast - unrealistic
                    speed_scores.append(0.0)
        
        if not direction_scores:
            return 0.2  # No valid segments
        
        # Combine direction and speed scores
        avg_direction = np.mean(direction_scores) if direction_scores else 0.0
        avg_speed = np.mean(speed_scores) if speed_scores else 0.5
        
        return avg_direction * 0.7 + avg_speed * 0.3
    
    def _validate_reception(self, candidate: PassCandidate,
                           ball_observations: List[BallObservation],
                           ball_tracker: StrictBallTracker) -> float:
        """
        Validate ball proximity at reception.
        
        Returns evidence score (0.0 to 1.0).
        """
        end_frame = candidate.end_frame_estimate
        window_start = end_frame - self.config['reception_window']
        window_end = end_frame + self.config['reception_window']
        
        best_evidence = 0.0
        
        for obs in ball_observations:
            if not (window_start <= obs.frame <= window_end):
                continue
            
            # Skip LOST observations
            if obs.state == BallState.LOST or obs.position is None:
                continue
            
            distance = np.linalg.norm(obs.position - candidate.receiver_position)
            proximity = self.config['reception_proximity']
            
            if distance <= proximity:
                # Calculate base evidence (closer = higher)
                base_evidence = 1.0 - (distance / proximity)
                
                # Weight by observation state
                if obs.state == BallState.DETECTED:
                    evidence = base_evidence * 1.0
                elif obs.state == BallState.PREDICTED:
                    evidence = base_evidence * 0.6
                else:
                    evidence = 0.0
                
                # Weight by temporal proximity to end frame
                frame_distance = abs(obs.frame - end_frame)
                temporal_weight = 1.0 - (frame_distance / self.config['reception_window'])
                evidence *= max(0.3, temporal_weight)
                
                best_evidence = max(best_evidence, evidence)
        
        return best_evidence
    
    def _has_contradiction(self, candidate: PassCandidate,
                          ball_observations: List[BallObservation]) -> Optional[str]:
        """
        Check if ball motion contradicts the candidate pass.
        
        Returns contradiction reason or None if no contradiction.
        """
        # Get DETECTED positions only for contradiction check
        detected_positions = [
            obs.position for obs in ball_observations 
            if obs.position is not None and obs.state == BallState.DETECTED
        ]
        
        if len(detected_positions) < 3:
            return None  # Not enough data to check contradiction
        
        # Check if ball is stuck (not moving when it should for a pass)
        total_movement = 0.0
        for i in range(len(detected_positions) - 1):
            movement = np.linalg.norm(detected_positions[i + 1] - detected_positions[i])
            total_movement += movement
        
        # For a pass, ball should move at least some distance
        pass_distance = candidate.distance_meters
        if pass_distance > 5.0 and total_movement < 2.0:
            return "ball_stuck"
        
        # Check for teleportation (unrealistic jumps)
        for i in range(len(detected_positions) - 1):
            dist = np.linalg.norm(detected_positions[i + 1] - detected_positions[i])
            if dist > 15.0:  # More than 15 meters between detections
                return "ball_teleported"
        
        # Check if ball is moving in opposite direction
        if len(detected_positions) >= 2:
            # Overall ball direction
            ball_movement = detected_positions[-1] - detected_positions[0]
            ball_distance = np.linalg.norm(ball_movement)
            
            if ball_distance > 3.0:  # Significant movement
                ball_dir = ball_movement / ball_distance
                
                # Compare to pass direction
                cos_angle = np.dot(ball_dir, candidate.direction_vector)
                
                if cos_angle < -0.5:  # More than 120 degrees opposite
                    return "ball_moving_opposite"
        
        return None
