"""
Player Candidate Generator Module

Generates pass candidates from player motion ONLY.
Looser thresholds than final pass detection - allows more candidates.
These candidates are then validated by ball evidence.

Key principle: Players propose, never finalize.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .player_motion_analyzer import PlayerMotionAnalyzer
from .pass_state_machine import PassStateMachine


@dataclass
class PassCandidate:
    """
    A pass candidate generated from player motion.
    
    This is NOT a final pass - it needs ball validation.
    """
    candidate_id: str
    initiator_id: int
    receiver_id: int
    team_id: int
    start_frame_estimate: int  # Estimated from player motion
    end_frame_estimate: int   # Estimated from player motion
    initiator_position: np.ndarray
    receiver_position: np.ndarray
    direction_vector: np.ndarray  # A → B vector
    confidence_player: float  # Confidence from player motion only
    initiation_confidence: float
    direction_confidence: float
    distance_confidence: float
    reception_confidence: float
    distance_meters: float
    
    # For ball validation
    validated: bool = False
    ball_evidence_score: float = 0.0
    rejection_reason: Optional[str] = None


class PlayerCandidateGenerator:
    """
    Generates pass candidates from player motion.
    
    Uses looser thresholds than final pass detection to allow
    more candidates. These are then filtered by ball evidence.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the candidate generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            # Looser thresholds for candidate generation
            'velocity_spike_threshold': 1.2,  # Lower than final (was 1.5)
            'acceleration_threshold': 1.5,    # Lower than final (was 2.0)
            'direction_stability_frames': 1,  # Lower than final (was 2)
            
            # Candidate selection (looser)
            'min_pass_distance': 2.0,         # Lower than final (was 3.0)
            'max_pass_distance': 45.0,        # Higher than final (was 25.0)
            'max_angle_deviation': 70.0,      # Higher than final (was 45.0)
            
            # Temporal (looser)
            'min_pass_time': 0.2,            # Lower
            'max_pass_time': 3.5,            # Higher
            
            # Confidence (lower threshold for candidates)
            'min_candidate_confidence': 0.25,  # Much lower than final
            
            # Frame rate
            'fps': 30.0,
        }
        
        if config:
            self.config.update(config)
        
        # Motion analyzer (reuse existing)
        self.motion_analyzer = PlayerMotionAnalyzer(self.config)
        
        # State machine (for cooldown/pair locking)
        self.state_machine = PassStateMachine(self.config)
        
        # Active candidates
        self.active_candidates: Dict[str, PassCandidate] = {}
        self.candidate_counter = 0
        
    def generate_candidates(self, frame: int,
                           player_positions: Dict[int, np.ndarray],
                           player_teams: Dict[int, int]) -> List[PassCandidate]:
        """
        Generate pass candidates from player motion.
        
        Args:
            frame: Current frame number
            player_positions: Dict of player_id -> position [x, y] in pitch coordinates
            player_teams: Dict of player_id -> team_id
            
        Returns:
            List of new PassCandidate objects
        """
        # Update motion states
        for player_id, position in player_positions.items():
            self.motion_analyzer.update_player(player_id, frame, position)
        
        # Update state machine
        self.state_machine.update(frame)
        
        new_candidates = []
        
        # Detect initiations (looser thresholds)
        for player_id, position in player_positions.items():
            # Check if can initiate
            if not self.state_machine.can_initiate_pass(player_id, frame):
                continue
            
            # Detect initiation (looser)
            is_initiating, init_conf = self._detect_initiation_loose(player_id)
            
            if is_initiating and init_conf >= self.config['min_candidate_confidence']:
                # Start initiation
                if self.state_machine.start_initiation(player_id, frame, position.tolist()):
                    # Find receiver candidates (looser)
                    receiver_candidates = self._find_receiver_candidates(
                        player_id, frame, player_positions, player_teams
                    )
                    
                    for receiver_id, receiver_pos, score in receiver_candidates:
                        # Lock to receiver
                        if self.state_machine.lock_to_receiver(
                            player_id, receiver_id, frame, receiver_pos.tolist()
                        ):
                            # Create candidate
                            candidate = self._create_candidate(
                                player_id, receiver_id, frame,
                                position, receiver_pos,
                                player_teams.get(player_id, 0),
                                init_conf, score
                            )
                            
                            if candidate:
                                self.active_candidates[candidate.candidate_id] = candidate
                                new_candidates.append(candidate)
        
        # Validate and confirm candidates (player-only validation)
        confirmed_candidates = []
        for candidate_id, candidate in list(self.active_candidates.items()):
            # Check for reception behavior
            has_reception, reception_conf = self.motion_analyzer.detect_reception_behavior(
                candidate.receiver_id,
                candidate.start_frame_estimate,
                frame
            )
            
            if has_reception:
                candidate.reception_confidence = reception_conf
                candidate.end_frame_estimate = frame
                
                # Update confidence
                candidate.confidence_player = (
                    candidate.initiation_confidence * 0.3 +
                    candidate.direction_confidence * 0.25 +
                    candidate.distance_confidence * 0.2 +
                    reception_conf * 0.25
                )
                
                # Confirm in state machine
                self.state_machine.confirm_pass(candidate.initiator_id, frame)
                confirmed_candidates.append(candidate)
                del self.active_candidates[candidate_id]
        
        return confirmed_candidates
    
    def _detect_initiation_loose(self, player_id: int) -> Tuple[bool, float]:
        """Detect initiation with looser thresholds."""
        state = self.motion_analyzer.motion_states.get(player_id)
        if state is None or state.velocity is None:
            return False, 0.0
        
        history = self.motion_analyzer.position_history[player_id]
        if len(history) < 3:
            return False, 0.0
        
        # Compare speeds
        prev_velocity = self.motion_analyzer._compute_velocity(history[-3:-1], frames_back=2)
        if prev_velocity is None:
            return False, 0.0
        
        prev_speed = np.linalg.norm(prev_velocity)
        velocity_change = state.speed - prev_speed
        
        # Looser thresholds
        has_velocity_spike = velocity_change > self.config['velocity_spike_threshold']
        has_acceleration = state.acceleration > self.config['acceleration_threshold']
        has_stable_direction = state.direction_stable or len(history) < 4  # Looser
        
        if not (has_velocity_spike or has_acceleration):
            return False, 0.0
        
        # Confidence (looser)
        confidence = 0.0
        if has_velocity_spike:
            confidence += 0.3
        if has_acceleration:
            confidence += 0.3
        if has_stable_direction:
            confidence += 0.2
        else:
            confidence += 0.1  # Partial credit
        
        return True, confidence
    
    def _find_receiver_candidates(self, initiator_id: int, frame: int,
                                 player_positions: Dict[int, np.ndarray],
                                 player_teams: Dict[int, int]) -> List[Tuple[int, np.ndarray, float]]:
        """Find receiver candidates with looser criteria."""
        initiator_pos = self.motion_analyzer.get_position(initiator_id)
        initiator_dir = self.motion_analyzer.get_direction(initiator_id)
        initiator_team = player_teams.get(initiator_id)
        
        if initiator_pos is None or initiator_team is None:
            return []
        
        candidates = []
        
        for receiver_id, receiver_pos in player_positions.items():
            if receiver_id == initiator_id:
                continue
            
            receiver_team = player_teams.get(receiver_id)
            if receiver_team != initiator_team:
                continue
            
            # Check pair lock
            if self.state_machine.is_pair_locked(initiator_id, receiver_id, frame):
                continue
            
            # Looser distance check
            distance = np.linalg.norm(receiver_pos - initiator_pos)
            if distance < self.config['min_pass_distance'] or distance > self.config['max_pass_distance']:
                continue
            
            # Looser direction check
            pass_dir = receiver_pos - initiator_pos
            pass_dir_norm = pass_dir / np.linalg.norm(pass_dir)
            
            direction_score = 1.0
            if initiator_dir is not None:
                cos_angle = np.dot(pass_dir_norm, initiator_dir / np.linalg.norm(initiator_dir))
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180.0 / np.pi
                if angle > self.config['max_angle_deviation']:
                    continue
                direction_score = 1.0 - (angle / self.config['max_angle_deviation'])
            
            # Score candidate
            distance_score = 1.0 / (1.0 + distance / 15.0)  # Prefer medium distances
            score = (direction_score * 0.6 + distance_score * 0.4)
            
            candidates.append((receiver_id, receiver_pos, score))
        
        # Sort by score
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:3]  # Top 3 candidates
    
    def _create_candidate(self, initiator_id: int, receiver_id: int, frame: int,
                          initiator_pos: np.ndarray, receiver_pos: np.ndarray,
                          team_id: int, init_conf: float, score: float) -> Optional[PassCandidate]:
        """Create a pass candidate."""
        self.candidate_counter += 1
        candidate_id = f"candidate_{self.candidate_counter:06d}"
        
        distance = np.linalg.norm(receiver_pos - initiator_pos)
        direction = receiver_pos - initiator_pos
        direction_norm = direction / np.linalg.norm(direction)
        
        # Calculate confidences
        distance_conf = 1.0 if (self.config['min_pass_distance'] <= distance <= self.config['max_pass_distance']) else 0.5
        
        initiator_dir = self.motion_analyzer.get_direction(initiator_id)
        direction_conf = 1.0
        if initiator_dir is not None:
            cos_angle = np.dot(direction_norm, initiator_dir / np.linalg.norm(initiator_dir))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180.0 / np.pi
            direction_conf = 1.0 - min(1.0, angle / self.config['max_angle_deviation'])
        
        return PassCandidate(
            candidate_id=candidate_id,
            initiator_id=initiator_id,
            receiver_id=receiver_id,
            team_id=team_id,
            start_frame_estimate=frame,
            end_frame_estimate=frame + int(self.config['max_pass_time'] * self.config['fps']),
            initiator_position=initiator_pos.copy(),
            receiver_position=receiver_pos.copy(),
            direction_vector=direction_norm,
            confidence_player=init_conf * score,
            initiation_confidence=init_conf,
            direction_confidence=direction_conf,
            distance_confidence=distance_conf,
            reception_confidence=0.0,  # Will be updated on reception
            distance_meters=distance
        )

