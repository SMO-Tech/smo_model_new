"""
Hybrid Pass Manager Module

Combines player candidate generation + ball evidence validation.
Key principle: "Players propose. Ball disposes."

Architecture:
1. PlayerCandidateGenerator - generates candidates (looser thresholds)
2. BallEvidenceValidator - validates with ball evidence
3. Final passes = intersection of both
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from .player_candidate_generator import PlayerCandidateGenerator, PassCandidate
from .ball_evidence_validator import BallEvidenceValidator, BallValidationResult
from .ball_tracker import StrictBallTracker, BallState
from .pass_event import PassEvent, PassLifecycleStage


class HybridPassManager:
    """
    Manages hybrid pass detection: players propose, ball validates.
    
    Guarantees:
    - Ball can SUPPORT but never CREATE passes
    - Players define events
    - Ball validates events
    - No ball hallucinations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the hybrid pass manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            # Final pass thresholds - reduced for less strict detection
            'min_final_confidence': 0.35,  # Reduced from 0.5 - accept more passes
            
            # Duplicate prevention - more lenient
            'cooldown_duration': 0.5,    # Reduced from 0.75 - allow faster re-initiation
            'pair_lock_duration': 1.0,   # Reduced from 1.5 - allow same pair passes sooner
            'temporal_merge_window': 20, # Increased from 15 - merge similar passes
            
            # Team binding
            'lock_team_assignments': True,
            
            # Frame rate
            'fps': 30.0,
        }
        
        if config:
            self.config.update(config)
        
        # Initialize subsystems
        self.candidate_generator = PlayerCandidateGenerator(config)
        self.ball_validator = BallEvidenceValidator(config)
        self.ball_tracker = StrictBallTracker(config)
        
        # Team binding
        self.team_map: Dict[int, int] = {}
        
        # Final passes
        self.confirmed_passes: List[PassEvent] = []
        
        # Debug tracking
        self.player_candidates: List[PassCandidate] = []
        self.ball_validations: List[Dict] = []
        self.rejected_candidates: List[Dict] = []
        
        # Duplicate prevention (frame-based, not time-based)
        self.recent_passes: List[Tuple[int, int, int, int]] = []  # (from, to, start, end)
        self.initiator_cooldowns: Dict[int, int] = {}  # player_id -> cooldown_end_frame
        self.pair_cooldowns: Dict[Tuple[int, int], int] = {}  # (from, to) -> cooldown_end_frame
        
        # Metrics
        self.metrics = {
            'player_candidates_generated': 0,
            'ball_validated_passes': 0,
            'rejected_no_ball_evidence': 0,
            'rejected_contradictory_ball': 0,
            'rejected_duplicate': 0,
            'rejected_cooldown': 0,
            'rejected_low_confidence': 0,
        }
    
    def set_team(self, player_id: int, team_id: int):
        """Set team for a player (locked)."""
        if self.config['lock_team_assignments']:
            if player_id not in self.team_map:
                self.team_map[player_id] = team_id
        else:
            self.team_map[player_id] = team_id
    
    def get_team(self, player_id: int) -> Optional[int]:
        """Get team for a player."""
        return self.team_map.get(player_id)
    
    def process_frame(self, frame: int,
                     player_positions: Dict[int, np.ndarray],
                     player_teams: Dict[int, int],
                     ball_detections: Optional[np.ndarray] = None) -> List[PassEvent]:
        """
        Process a single frame for hybrid pass detection.
        
        Args:
            frame: Current frame number
            player_positions: Dict of player_id -> position [x, y] in pitch coordinates
            player_teams: Dict of player_id -> team_id
            ball_detections: Array of ball positions [N, 2] in pitch coordinates, or None
            
        Returns:
            List of newly confirmed passes
        """
        # Update team bindings
        for player_id, team_id in player_teams.items():
            self.set_team(player_id, team_id)
        
        # Update ball tracker FIRST (this must happen every frame)
        ball_obs = self.ball_tracker.update(frame, ball_detections)
        
        # Generate player candidates (always generate - validator will reject if ball is LOST)
        candidates = self.candidate_generator.generate_candidates(
            frame, player_positions, player_teams
        )
        
        self.metrics['player_candidates_generated'] += len(candidates)
        
        # Validate candidates with ball evidence
        confirmed = []
        for candidate in candidates:
            # Store candidate for debug
            self.player_candidates.append(candidate)
            
            # Validate with ball
            validation = self.ball_validator.validate_candidate(
                candidate, self.ball_tracker
            )
            
            # Store validation for debug
            self.ball_validations.append({
                'candidate_id': candidate.candidate_id,
                'evidence_score': validation.evidence_score,
                'is_valid': validation.is_valid,
                'rejection_reason': validation.rejection_reason,
                'details': validation.validation_details
            })
            
            if validation.is_valid:
                # Check for duplicates
                if self._is_duplicate(candidate.initiator_id, candidate.receiver_id,
                                     candidate.start_frame_estimate, candidate.end_frame_estimate):
                    self.metrics['rejected_duplicate'] += 1
                    self._reject_candidate(candidate, "duplicate")
                    continue
                
                # Check cooldown
                if self._is_in_cooldown(candidate.initiator_id, frame):
                    self.metrics['rejected_cooldown'] += 1
                    self._reject_candidate(candidate, "cooldown")
                    continue
                
                # Calculate final confidence (player + ball)
                final_confidence = (
                    candidate.confidence_player * 0.6 +
                    validation.evidence_score * 0.4
                )
                
                if final_confidence >= self.config['min_final_confidence']:
                    # Check pair cooldown (same initiator->receiver)
                    if self._is_pair_in_cooldown(candidate.initiator_id, candidate.receiver_id, frame):
                        self.metrics['rejected_cooldown'] += 1
                        self._reject_candidate(candidate, "pair_cooldown")
                        continue
                    
                    # Create final pass event
                    pass_event = self._create_pass_event(
                        candidate, validation, frame
                    )
                    
                    if pass_event.mark_emitted():
                        self.confirmed_passes.append(pass_event)
                        confirmed.append(pass_event)
                        self.metrics['ball_validated_passes'] += 1
                        
                        # Record for duplicate prevention
                        self.recent_passes.append((
                            pass_event.from_player_id,
                            pass_event.to_player_id,
                            pass_event.start_frame,
                            pass_event.end_frame
                        ))
                        
                        # Set cooldowns (frame-based)
                        cooldown_frames = int(self.config['cooldown_duration'] * self.config['fps'])
                        self.initiator_cooldowns[pass_event.from_player_id] = frame + cooldown_frames
                        
                        # Set pair cooldown (prevents same pair for longer)
                        pair_lock_frames = int(self.config['pair_lock_duration'] * self.config['fps'])
                        self.pair_cooldowns[(pass_event.from_player_id, pass_event.to_player_id)] = frame + pair_lock_frames
                        
                        # Keep recent passes limited
                        if len(self.recent_passes) > 100:
                            self.recent_passes = self.recent_passes[-100:]
                else:
                    self.metrics['rejected_low_confidence'] += 1
                    self._reject_candidate(candidate, f"low_confidence_{final_confidence:.2f}")
            else:
                # Rejected by ball validation
                if validation.rejection_reason == "no_ball_evidence":
                    self.metrics['rejected_no_ball_evidence'] += 1
                elif "contradictory" in validation.rejection_reason:
                    self.metrics['rejected_contradictory_ball'] += 1
                
                self._reject_candidate(candidate, validation.rejection_reason)
        
        return confirmed
    
    def _is_duplicate(self, from_id: int, to_id: int, start: int, end: int) -> bool:
        """Check if this pass is a duplicate."""
        merge_window = self.config['temporal_merge_window']
        
        for prev_from, prev_to, prev_start, prev_end in self.recent_passes:
            if prev_from == from_id and prev_to == to_id:
                if abs(start - prev_start) < merge_window or abs(end - prev_end) < merge_window:
                    return True
        
        return False
    
    def _is_in_cooldown(self, player_id: int, frame: int) -> bool:
        """Check if player is in cooldown (frame-based)."""
        if player_id not in self.initiator_cooldowns:
            return False
        
        cooldown_end_frame = self.initiator_cooldowns[player_id]
        return frame < cooldown_end_frame
    
    def _is_pair_in_cooldown(self, from_id: int, to_id: int, frame: int) -> bool:
        """Check if player pair is in cooldown (prevents same pair passes in quick succession)."""
        pair_key = (from_id, to_id)
        if pair_key not in self.pair_cooldowns:
            return False
        
        cooldown_end_frame = self.pair_cooldowns[pair_key]
        return frame < cooldown_end_frame
    
    def _create_pass_event(self, candidate: PassCandidate,
                          validation: BallValidationResult,
                          current_frame: int) -> PassEvent:
        """Create final PassEvent from validated candidate."""
        # Determine timing (earliest of player initiation or ball leaving initiator)
        start_frame = candidate.start_frame_estimate
        
        # Check if ball left initiator earlier
        ball_obs = self.ball_tracker.get_ball_in_window(
            start_frame - 5, start_frame + 5
        )
        for obs in ball_obs:
            if obs.position is not None and obs.state == BallState.DETECTED:
                distance = np.linalg.norm(obs.position - candidate.initiator_position)
                if distance > self.config.get('initiation_proximity', 3.0):
                    # Ball left initiator - use this frame
                    start_frame = min(start_frame, obs.frame)
                    break
        
        # End frame = ball arrival or candidate end
        end_frame = candidate.end_frame_estimate
        
        # Check if ball arrived at receiver
        for obs in ball_obs:
            if obs.position is not None and obs.state == BallState.DETECTED:
                distance = np.linalg.norm(obs.position - candidate.receiver_position)
                if distance <= self.config.get('reception_proximity', 3.0):
                    end_frame = min(end_frame, obs.frame)
                    break
        
        # Calculate final confidence
        final_confidence = (
            candidate.confidence_player * 0.6 +
            validation.evidence_score * 0.4
        )
        
        return PassEvent(
            from_player_id=candidate.initiator_id,
            to_player_id=candidate.receiver_id,
            team_id=candidate.team_id,
            start_frame=start_frame,  # NEVER modified after this
            end_frame=end_frame,
            start_position=candidate.initiator_position.tolist(),
            end_position=candidate.receiver_position.tolist(),
            confidence=final_confidence,
            initiation_confidence=candidate.initiation_confidence,
            direction_confidence=candidate.direction_confidence,
            distance_confidence=candidate.distance_confidence,
            reception_confidence=candidate.reception_confidence,
            distance_meters=candidate.distance_meters,
            duration_seconds=(end_frame - start_frame) / self.config['fps'],
            lifecycle_stage=PassLifecycleStage.CONFIRMED,
            created_timestamp=time.time()
        )
    
    def _reject_candidate(self, candidate: PassCandidate, reason: str):
        """Record rejected candidate for debug."""
        self.rejected_candidates.append({
            'candidate_id': candidate.candidate_id,
            'initiator_id': candidate.initiator_id,
            'receiver_id': candidate.receiver_id,
            'start_frame': candidate.start_frame_estimate,
            'end_frame': candidate.end_frame_estimate,
            'player_confidence': candidate.confidence_player,
            'rejection_reason': reason
        })
    
    def get_confirmed_passes(self) -> List[PassEvent]:
        """Get all confirmed passes."""
        return self.confirmed_passes.copy()
    
    def get_ball_tracker(self) -> StrictBallTracker:
        """Get ball tracker instance."""
        return self.ball_tracker
    
    def save_debug_outputs(self, output_dir: Path):
        """Save debug outputs."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Player candidates
        if self.player_candidates:
            candidates_data = [{
                'candidate_id': c.candidate_id,
                'initiator_id': c.initiator_id,
                'receiver_id': c.receiver_id,
                'team_id': c.team_id,
                'start_frame': c.start_frame_estimate,
                'end_frame': c.end_frame_estimate,
                'player_confidence': c.confidence_player,
                'distance_meters': c.distance_meters
            } for c in self.player_candidates]
            df = pd.DataFrame(candidates_data)
            df.to_csv(output_dir / 'player_candidates.csv', index=False)
        
        # Ball validations
        if self.ball_validations:
            df = pd.DataFrame(self.ball_validations)
            df.to_csv(output_dir / 'ball_validation.csv', index=False)
        
        # Final passes
        if self.confirmed_passes:
            passes_data = [p.to_dict() for p in self.confirmed_passes]
            df = pd.DataFrame(passes_data)
            df.to_csv(output_dir / 'final_passes.csv', index=False)
        
        # Rejected candidates
        if self.rejected_candidates:
            df = pd.DataFrame(self.rejected_candidates)
            df.to_csv(output_dir / 'rejected_candidates.csv', index=False)
        
        # Metrics
        metrics = {
            **self.metrics,
            'total_confirmed': len(self.confirmed_passes),
            'total_rejected': len(self.rejected_candidates),
            'avg_confidence': np.mean([p.confidence for p in self.confirmed_passes]) if self.confirmed_passes else 0.0
        }
        
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Debug outputs saved to {output_dir}")
        print(f"  - Player candidates: {len(self.player_candidates)}")
        print(f"  - Ball validated passes: {len(self.confirmed_passes)}")
        print(f"  - Rejected candidates: {len(self.rejected_candidates)}")

