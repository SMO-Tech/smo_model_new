"""
Ball Evidence Validator Module

Validates pass candidates using ball tracking evidence.
Ball can SUPPORT but never CREATE passes.

Key principle: "Players propose. Ball disposes."
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
    
    # Rejection reason
    rejection_reason: Optional[str] = None


class BallEvidenceValidator:
    """
    Validates pass candidates using ball tracking.
    
    Rules:
    - Only DETECTED ball positions count as strong evidence
    - PREDICTED positions provide weak evidence
    - LOST ball = downgrade, not auto-reject
    - Ball can contradict and reject candidates
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            # Proximity thresholds (meters)
            'initiation_proximity': 3.0,   # Ball must be within X meters of initiator
            'reception_proximity': 3.0,    # Ball must be within X meters of receiver
            
            # Time windows (frames)
            'initiation_window': 3,        # ±3 frames around start
            'reception_window': 5,         # ±5 frames around end
            
            # Speed constraints
            'min_ball_speed': 2.0,         # m/s minimum for transit
            'max_ball_speed': 35.0,        # m/s maximum
            
            # Direction alignment
            'max_direction_deviation': 60.0,  # degrees
            
            # Evidence weights
            'initiation_weight': 0.3,
            'transit_weight': 0.4,
            'reception_weight': 0.3,
            
            # Minimum evidence score to accept
            'min_evidence_score': 0.4,
            
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
        
        ball_observations = ball_tracker.get_ball_in_window(start_frame, end_frame)
        
        if not ball_observations:
            # No ball evidence - downgrade but don't auto-reject
            result.evidence_score = 0.0
            result.rejection_reason = "no_ball_evidence"
            result.is_valid = False
            return result
        
        # Validate initiation evidence
        result.initiation_evidence = self._validate_initiation(
            candidate, ball_observations, ball_tracker
        )
        
        # Validate transit evidence
        result.transit_evidence = self._validate_transit(
            candidate, ball_observations
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
        if self._has_contradiction(candidate, ball_observations):
            result.is_valid = False
            result.rejection_reason = "contradictory_ball_motion"
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
            'detected_count': sum(1 for obs in ball_observations if obs.state == BallState.DETECTED)
        }
        
        return result
    
    def _validate_initiation(self, candidate: PassCandidate,
                            ball_observations: List[BallObservation],
                            ball_tracker: StrictBallTracker) -> float:
        """Validate ball proximity at initiation."""
        start_frame = candidate.start_frame_estimate
        window_start = start_frame - self.config['initiation_window']
        window_end = start_frame + self.config['initiation_window']
        
        evidence_score = 0.0
        
        for obs in ball_observations:
            if window_start <= obs.frame <= window_end:
                if obs.position is not None:
                    distance = np.linalg.norm(obs.position - candidate.initiator_position)
                    
                    if distance <= self.config['initiation_proximity']:
                        # Strong evidence if DETECTED
                        if obs.state == BallState.DETECTED:
                            evidence_score = max(evidence_score, 1.0 - (distance / self.config['initiation_proximity']))
                        elif obs.state == BallState.PREDICTED:
                            # Weak evidence if PREDICTED
                            evidence_score = max(evidence_score, 0.5 * (1.0 - (distance / self.config['initiation_proximity'])))
        
        return evidence_score
    
    def _validate_transit(self, candidate: PassCandidate,
                         ball_observations: List[BallObservation]) -> float:
        """Validate ball transit between initiator and receiver."""
        if len(ball_observations) < 2:
            return 0.0
        
        # Check if ball moves in correct direction
        detected_observations = [obs for obs in ball_observations 
                                 if obs.position is not None and obs.state == BallState.DETECTED]
        
        if len(detected_observations) < 2:
            # Not enough DETECTED observations
            return 0.2  # Weak evidence
        
        # Check direction alignment
        direction_scores = []
        for i in range(len(detected_observations) - 1):
            obs1 = detected_observations[i]
            obs2 = detected_observations[i + 1]
            
            # Ball movement vector
            ball_dir = obs2.position - obs1.position
            if np.linalg.norm(ball_dir) == 0:
                continue
            
            ball_dir_norm = ball_dir / np.linalg.norm(ball_dir)
            
            # Candidate direction
            candidate_dir = candidate.direction_vector
            
            # Angle between directions
            cos_angle = np.dot(ball_dir_norm, candidate_dir)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180.0 / np.pi
            
            if angle <= self.config['max_direction_deviation']:
                direction_scores.append(1.0 - (angle / self.config['max_direction_deviation']))
            else:
                direction_scores.append(0.0)
        
        if not direction_scores:
            return 0.0
        
        # Check speed
        speed_scores = []
        for i in range(len(detected_observations) - 1):
            obs1 = detected_observations[i]
            obs2 = detected_observations[i + 1]
            
            dt = (obs2.frame - obs1.frame) / self.config['fps']
            if dt <= 0:
                continue
            
            distance = np.linalg.norm(obs2.position - obs1.position)
            speed = distance / dt
            
            if self.config['min_ball_speed'] <= speed <= self.config['max_ball_speed']:
                speed_scores.append(1.0)
            else:
                speed_scores.append(0.0)
        
        if not speed_scores:
            return 0.0
        
        # Combine direction and speed
        avg_direction = np.mean(direction_scores) if direction_scores else 0.0
        avg_speed = np.mean(speed_scores) if speed_scores else 0.0
        
        return (avg_direction * 0.6 + avg_speed * 0.4)
    
    def _validate_reception(self, candidate: PassCandidate,
                           ball_observations: List[BallObservation],
                           ball_tracker: StrictBallTracker) -> float:
        """Validate ball proximity at reception."""
        end_frame = candidate.end_frame_estimate
        window_start = end_frame - self.config['reception_window']
        window_end = end_frame + self.config['reception_window']
        
        evidence_score = 0.0
        
        for obs in ball_observations:
            if window_start <= obs.frame <= window_end:
                if obs.position is not None:
                    distance = np.linalg.norm(obs.position - candidate.receiver_position)
                    
                    if distance <= self.config['reception_proximity']:
                        # Strong evidence if DETECTED
                        if obs.state == BallState.DETECTED:
                            evidence_score = max(evidence_score, 1.0 - (distance / self.config['reception_proximity']))
                        elif obs.state == BallState.PREDICTED:
                            # Weak evidence if PREDICTED
                            evidence_score = max(evidence_score, 0.5 * (1.0 - (distance / self.config['reception_proximity'])))
        
        return evidence_score
    
    def _has_contradiction(self, candidate: PassCandidate,
                          ball_observations: List[BallObservation]) -> bool:
        """Check if ball motion contradicts the candidate pass."""
        # Check for ball sticking (not moving)
        detected_positions = [obs.position for obs in ball_observations 
                             if obs.position is not None and obs.state == BallState.DETECTED]
        
        if len(detected_positions) >= 3:
            # Check if ball is stuck (not moving)
            distances = []
            for i in range(len(detected_positions) - 1):
                dist = np.linalg.norm(detected_positions[i + 1] - detected_positions[i])
                distances.append(dist)
            
            if distances and max(distances) < 1.0:  # Ball stuck (less than 1m movement)
                return True
        
        # Check for teleportation
        for i in range(len(detected_positions) - 1):
            dist = np.linalg.norm(detected_positions[i + 1] - detected_positions[i])
            frames_diff = 1  # Assume consecutive
            max_distance = 10.0 * frames_diff  # Unrealistic jump
            
            if dist > max_distance:
                return True
        
        return False

