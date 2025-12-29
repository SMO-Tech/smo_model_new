"""
Pass Lifecycle Manager Module

Manages the complete lifecycle of pass events:
- Event creation
- Confirmation
- Rejection
- Duplicate prevention
- Export

Ensures: One real-world pass = exactly one detected event.
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from .pass_state_machine import PassStateMachine, PassState
from .pass_event import PassEvent, PassLifecycleStage
# PassCandidate moved to player_candidate_generator.py for hybrid system
from .player_motion_analyzer import PlayerMotionAnalyzer
from .pass_validator import PassValidator, ValidationResult


class PassLifecycleManager:
    """
    Manages the complete lifecycle of pass detection.
    
    This is the main entry point for the pass detection system.
    
    Key guarantees:
    - No duplicate passes
    - Correct timing (start_frame never adjusted)
    - Event-based (not frame-based)
    - Ball-independent forever
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the lifecycle manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            # Pass detection
            'min_confidence': 0.4,
            
            # Temporal merge (prevent near-duplicates)
            'temporal_merge_window': 15,  # frames - merge passes within this window
            
            # Team binding
            'lock_team_assignments': True,  # Lock team after first assignment
            
            # Frame rate
            'fps': 30.0,
        }
        
        if config:
            self.config.update(config)
        
        # Initialize components
        self.state_machine = PassStateMachine(config)
        self.motion_analyzer = PlayerMotionAnalyzer(config)
        self.validator = PassValidator(config)
        
        # Team binding (player_id -> team_id, locked)
        self.team_map: Dict[int, int] = {}
        
        # Active pass candidates (being validated)
        self.active_candidates: Dict[str, PassEvent] = {}  # event_id -> PassEvent
        
        # Completed passes (confirmed and emitted)
        self.confirmed_passes: List[PassEvent] = []
        
        # Rejected passes (for debugging)
        self.rejected_passes: List[PassEvent] = []
        
        # Duplicate prevention tracking
        self.recent_passes: List[Tuple[int, int, int, int]] = []  # (from, to, start, end)
        
        # Metrics
        self.metrics = {
            'total_initiations': 0,
            'accepted_passes': 0,
            'rejected_passes': 0,
            'duplicates_prevented': 0,
            'cooldown_rejections': 0,
            'pair_lock_rejections': 0,
            'temporal_merges': 0,
        }
    
    def set_team(self, player_id: int, team_id: int):
        """
        Set team for a player.
        
        If lock_team_assignments is True, team is set only once.
        """
        if self.config['lock_team_assignments']:
            if player_id not in self.team_map:
                self.team_map[player_id] = team_id
        else:
            self.team_map[player_id] = team_id
    
    def get_team(self, player_id: int) -> Optional[int]:
        """Get team for a player (from locked team map)."""
        return self.team_map.get(player_id)
    
    def process_frame(self, frame: int, 
                     player_positions: Dict[int, np.ndarray],
                     player_teams: Dict[int, int]) -> List[PassEvent]:
        """
        Process a single frame for pass detection.
        
        Args:
            frame: Current frame number
            player_positions: Dict of player_id -> position [x, y]
            player_teams: Dict of player_id -> team_id
            
        Returns:
            List of newly confirmed passes (emitted exactly once)
        """
        # Update team bindings
        for player_id, team_id in player_teams.items():
            self.set_team(player_id, team_id)
        
        # Update motion states
        for player_id, position in player_positions.items():
            self.motion_analyzer.update_player(player_id, frame, position)
        
        # Update state machine (handle timeouts)
        self.state_machine.update(frame)
        
        # Detect new pass initiations
        self._detect_initiations(frame, player_positions)
        
        # Find receivers for initiating players
        self._find_receivers(frame, player_positions)
        
        # Validate and confirm passes
        confirmed = self._validate_and_confirm(frame)
        
        return confirmed
    
    def _detect_initiations(self, frame: int, player_positions: Dict[int, np.ndarray]):
        """Detect new pass initiations."""
        for player_id, position in player_positions.items():
            # Check if player can initiate
            if not self.state_machine.can_initiate_pass(player_id, frame):
                continue
            
            # Detect initiation behavior
            is_initiating, confidence = self.motion_analyzer.detect_pass_initiation(player_id)
            
            if is_initiating:
                self.metrics['total_initiations'] += 1
                
                # Start initiation in state machine
                if self.state_machine.start_initiation(player_id, frame, position.tolist()):
                    # Create pass event
                    team_id = self.get_team(player_id) or 0
                    event = PassEvent(
                        from_player_id=player_id,
                        team_id=team_id,
                        start_frame=frame,  # NEVER modified after this
                        start_position=position.tolist(),
                        initiation_confidence=confidence,
                        created_timestamp=time.time()
                    )
                    self.active_candidates[event.event_id] = event
    
    def _find_receivers(self, frame: int, player_positions: Dict[int, np.ndarray]):
        """Find receivers for players in INITIATING state."""
        initiating = self.state_machine.get_initiating_players()
        
        for player_id, state in initiating.items():
            initiator_pos = self.motion_analyzer.get_position(player_id)
            initiator_dir = self.motion_analyzer.get_direction(player_id)
            initiator_team = self.get_team(player_id)
            
            if initiator_pos is None or initiator_team is None:
                continue
            
            # Find best receiver candidate
            best_receiver = None
            best_score = 0.0
            
            for receiver_id, receiver_pos in player_positions.items():
                if receiver_id == player_id:
                    continue
                
                receiver_team = self.get_team(receiver_id)
                if receiver_team is None:
                    continue
                
                # Check pair lock
                if self.state_machine.is_pair_locked(player_id, receiver_id, frame):
                    self.metrics['pair_lock_rejections'] += 1
                    continue
                
                # Validate candidate
                is_valid, reason = self.validator.validate_receiver_candidate(
                    initiator_pos, receiver_pos, initiator_dir,
                    initiator_team, receiver_team
                )
                
                if not is_valid:
                    continue
                
                # Score candidate (distance + direction)
                distance = np.linalg.norm(receiver_pos - initiator_pos)
                score = 1.0 / (1.0 + distance / 10.0)  # Prefer closer receivers
                
                if initiator_dir is not None:
                    pass_dir = receiver_pos - initiator_pos
                    if np.linalg.norm(pass_dir) > 0:
                        cos_angle = np.dot(pass_dir / np.linalg.norm(pass_dir),
                                          initiator_dir / np.linalg.norm(initiator_dir))
                        score *= (1.0 + cos_angle) / 2.0  # Prefer aligned receivers
                
                if score > best_score:
                    best_score = score
                    best_receiver = (receiver_id, receiver_pos.copy())
            
            # Lock to best receiver
            if best_receiver is not None:
                receiver_id, receiver_pos = best_receiver
                if self.state_machine.lock_to_receiver(player_id, receiver_id, frame, 
                                                       receiver_pos.tolist()):
                    # Update active candidate
                    for event in self.active_candidates.values():
                        if event.from_player_id == player_id and event.lifecycle_stage == PassLifecycleStage.INITIATED:
                            event.to_player_id = receiver_id
                            event.end_position = receiver_pos.tolist()
                            event.lifecycle_stage = PassLifecycleStage.CANDIDATE
                            break
    
    def _validate_and_confirm(self, frame: int) -> List[PassEvent]:
        """Validate locked passes and confirm if reception detected."""
        confirmed = []
        locked = self.state_machine.get_locked_players()
        
        for player_id, state in locked.items():
            receiver_id = state.locked_receiver_id
            if receiver_id is None:
                continue
            
            # Check for reception behavior
            has_reception, reception_conf = self.motion_analyzer.detect_reception_behavior(
                receiver_id, state.initiation_frame or state.state_start_frame, frame
            )
            
            if has_reception:
                # Find the active candidate
                event = None
                for e in self.active_candidates.values():
                    if (e.from_player_id == player_id and 
                        e.to_player_id == receiver_id and
                        e.lifecycle_stage == PassLifecycleStage.CANDIDATE):
                        event = e
                        break
                
                if event is None:
                    continue
                
                # Validate pass physics
                initiator_pos = np.array(event.start_position)
                receiver_pos = self.motion_analyzer.get_position(receiver_id)
                if receiver_pos is None:
                    receiver_pos = np.array(event.end_position)
                
                initiator_dir = self.motion_analyzer.get_direction(player_id)
                
                validation = self.validator.validate_pass(
                    initiator_pos, receiver_pos, initiator_dir,
                    event.start_frame, frame
                )
                
                if validation.is_valid:
                    # Check for duplicate (temporal merge)
                    if self._is_duplicate(event.from_player_id, event.to_player_id,
                                         event.start_frame, frame):
                        self.metrics['duplicates_prevented'] += 1
                        self.state_machine.cancel_pass(player_id, frame, "Duplicate")
                        event.reject("Duplicate of recent pass")
                        self.rejected_passes.append(event)
                        del self.active_candidates[event.event_id]
                        continue
                    
                    # Compute confidence
                    event.reception_confidence = reception_conf
                    event.distance_confidence = 1.0 if validation.distance_valid else 0.0
                    event.direction_confidence = 1.0 if validation.direction_valid else 0.0
                    
                    confidence = (
                        event.initiation_confidence * 0.3 +
                        event.direction_confidence * 0.25 +
                        event.distance_confidence * 0.2 +
                        event.reception_confidence * 0.25
                    )
                    
                    if confidence >= self.config['min_confidence']:
                        # Confirm pass
                        event.end_frame = frame
                        event.end_position = receiver_pos.tolist()
                        event.confidence = confidence
                        event.distance_meters = validation.distance
                        event.duration_seconds = validation.duration_seconds
                        event.implied_speed = validation.implied_speed
                        event.lifecycle_stage = PassLifecycleStage.CONFIRMED
                        event.confirmed_timestamp = time.time()
                        
                        # Emit exactly once
                        if event.mark_emitted():
                            self.confirmed_passes.append(event)
                            confirmed.append(event)
                            self.metrics['accepted_passes'] += 1
                            
                            # Record for duplicate prevention
                            self.recent_passes.append((
                                event.from_player_id, event.to_player_id,
                                event.start_frame, frame
                            ))
                            # Keep only recent passes
                            if len(self.recent_passes) > 100:
                                self.recent_passes = self.recent_passes[-100:]
                        
                        # Confirm in state machine
                        self.state_machine.confirm_pass(player_id, frame)
                        del self.active_candidates[event.event_id]
                    else:
                        event.reject(f"Low confidence: {confidence:.3f}")
                        self.rejected_passes.append(event)
                        self.metrics['rejected_passes'] += 1
                        self.state_machine.cancel_pass(player_id, frame, "Low confidence")
                        del self.active_candidates[event.event_id]
                else:
                    event.reject(validation.rejection_reason or "Physics validation failed")
                    self.rejected_passes.append(event)
                    self.metrics['rejected_passes'] += 1
                    self.state_machine.cancel_pass(player_id, frame, validation.rejection_reason)
                    del self.active_candidates[event.event_id]
        
        return confirmed
    
    def _is_duplicate(self, from_id: int, to_id: int, start: int, end: int) -> bool:
        """Check if this pass is a duplicate of a recent one."""
        merge_window = self.config['temporal_merge_window']
        
        for prev_from, prev_to, prev_start, prev_end in self.recent_passes:
            # Same pair
            if prev_from == from_id and prev_to == to_id:
                # Overlapping or close in time
                if abs(start - prev_start) < merge_window or abs(end - prev_end) < merge_window:
                    self.metrics['temporal_merges'] += 1
                    return True
        
        return False
    
    def get_confirmed_passes(self) -> List[PassEvent]:
        """Get all confirmed passes."""
        return self.confirmed_passes.copy()
    
    def save_debug_outputs(self, output_dir: Path):
        """
        Save debug outputs to files.
        
        Creates:
        - pass_lifecycle.csv
        - pass_state_transitions.csv
        - rejected_passes.csv
        - metrics.json
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pass lifecycle
        if self.confirmed_passes:
            lifecycle_data = [p.to_dict() for p in self.confirmed_passes]
            df = pd.DataFrame(lifecycle_data)
            df.to_csv(output_dir / 'pass_lifecycle.csv', index=False)
        
        # State transitions
        state_data = self.state_machine.get_debug_data()
        if state_data['state_transitions']:
            df = pd.DataFrame(state_data['state_transitions'])
            df.to_csv(output_dir / 'pass_state_transitions.csv', index=False)
        
        # Rejected passes
        if self.rejected_passes:
            rejected_data = [p.to_dict() for p in self.rejected_passes]
            df = pd.DataFrame(rejected_data)
            df.to_csv(output_dir / 'rejected_passes.csv', index=False)
        
        # Metrics
        metrics = {
            **self.metrics,
            'avg_confidence': np.mean([p.confidence for p in self.confirmed_passes]) if self.confirmed_passes else 0.0,
            'total_confirmed': len(self.confirmed_passes),
            'total_rejected': len(self.rejected_passes),
        }
        
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Debug outputs saved to {output_dir}")
        print(f"  - Confirmed passes: {len(self.confirmed_passes)}")
        print(f"  - Rejected passes: {len(self.rejected_passes)}")
        print(f"  - Duplicates prevented: {self.metrics['duplicates_prevented']}")
        print(f"  - Temporal merges: {self.metrics['temporal_merges']}")

