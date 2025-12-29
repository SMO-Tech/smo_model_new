"""
Pass State Machine Module

Implements a state machine for each player's pass state:
IDLE → INITIATING → LOCKED → COOLDOWN → IDLE

This prevents duplicate passes and ensures proper timing.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Set
import time


class PassState(Enum):
    """Player pass states."""
    IDLE = "idle"              # Player can initiate a pass
    INITIATING = "initiating"  # Pass initiation detected, looking for receiver
    LOCKED = "locked"          # Pass locked to a receiver, waiting for confirmation
    COOLDOWN = "cooldown"      # Post-pass cooldown, cannot initiate


@dataclass
class PlayerPassState:
    """Tracks the pass state for a single player."""
    player_id: int
    state: PassState = PassState.IDLE
    state_start_frame: int = 0
    state_start_time: float = 0.0
    
    # Initiation data
    initiation_frame: Optional[int] = None
    initiation_position: Optional[list] = None
    
    # Lock data (when receiver is selected)
    locked_receiver_id: Optional[int] = None
    locked_receiver_position: Optional[list] = None
    
    # Cooldown tracking
    cooldown_start_frame: Optional[int] = None
    
    # Pair lock tracking (prevents rapid A→B repeats)
    recent_receivers: Dict[int, int] = field(default_factory=dict)  # receiver_id → last_frame
    
    def reset(self, frame: int):
        """Reset to IDLE state."""
        self.state = PassState.IDLE
        self.state_start_frame = frame
        self.state_start_time = time.time()
        self.initiation_frame = None
        self.initiation_position = None
        self.locked_receiver_id = None
        self.locked_receiver_position = None
        self.cooldown_start_frame = None


class PassStateMachine:
    """
    Manages pass state machines for all players.
    
    Enforces:
    - One pass per lifecycle
    - Cooldown between passes
    - Pair locking to prevent duplicates
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the state machine manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            # Timing (in seconds)
            'initiation_timeout': 0.5,    # Max time in INITIATING before timeout
            'lock_timeout': 2.5,          # Max time locked waiting for reception
            'cooldown_duration': 0.75,    # Cooldown duration after pass
            
            # Pair locking
            'pair_lock_duration': 1.5,    # How long A→B is locked after a pass
            'pair_lock_distance': 5.0,    # Distance threshold to release pair lock (meters)
            
            # Frame rate
            'fps': 30.0,
        }
        
        if config:
            self.config.update(config)
        
        # Player states
        self.player_states: Dict[int, PlayerPassState] = {}
        
        # State transition log for debugging
        self.state_transitions: list = []
        
    def _get_or_create_state(self, player_id: int, frame: int) -> PlayerPassState:
        """Get or create player state."""
        if player_id not in self.player_states:
            self.player_states[player_id] = PlayerPassState(
                player_id=player_id,
                state=PassState.IDLE,
                state_start_frame=frame,
                state_start_time=time.time()
            )
        return self.player_states[player_id]
    
    def _log_transition(self, player_id: int, from_state: PassState, to_state: PassState, 
                       frame: int, reason: str):
        """Log a state transition for debugging."""
        self.state_transitions.append({
            'player_id': player_id,
            'from_state': from_state.value,
            'to_state': to_state.value,
            'frame': frame,
            'timestamp': frame / self.config['fps'],
            'reason': reason
        })
    
    def can_initiate_pass(self, player_id: int, frame: int) -> bool:
        """
        Check if a player can initiate a pass.
        
        Returns True only if player is in IDLE state.
        """
        state = self._get_or_create_state(player_id, frame)
        
        # Must be in IDLE to initiate
        if state.state != PassState.IDLE:
            return False
        
        return True
    
    def start_initiation(self, player_id: int, frame: int, position: list) -> bool:
        """
        Transition player from IDLE → INITIATING.
        
        Args:
            player_id: Player initiating the pass
            frame: Current frame number
            position: Player position at initiation
            
        Returns:
            True if transition successful, False otherwise
        """
        state = self._get_or_create_state(player_id, frame)
        
        if state.state != PassState.IDLE:
            return False
        
        # Transition to INITIATING
        old_state = state.state
        state.state = PassState.INITIATING
        state.state_start_frame = frame
        state.state_start_time = time.time()
        state.initiation_frame = frame
        state.initiation_position = position
        
        self._log_transition(player_id, old_state, state.state, frame, "Pass initiation detected")
        return True
    
    def lock_to_receiver(self, initiator_id: int, receiver_id: int, frame: int, 
                        receiver_position: list) -> bool:
        """
        Transition from INITIATING → LOCKED with a specific receiver.
        
        Args:
            initiator_id: Player who initiated
            receiver_id: Selected receiver
            frame: Current frame
            receiver_position: Receiver position
            
        Returns:
            True if transition successful
        """
        state = self._get_or_create_state(initiator_id, frame)
        
        if state.state != PassState.INITIATING:
            return False
        
        # Check pair lock (prevent rapid A→B repeats)
        if receiver_id in state.recent_receivers:
            last_frame = state.recent_receivers[receiver_id]
            frames_since = frame - last_frame
            lock_frames = int(self.config['pair_lock_duration'] * self.config['fps'])
            if frames_since < lock_frames:
                return False  # Pair still locked
        
        # Transition to LOCKED
        old_state = state.state
        state.state = PassState.LOCKED
        state.state_start_frame = frame
        state.state_start_time = time.time()
        state.locked_receiver_id = receiver_id
        state.locked_receiver_position = receiver_position
        
        self._log_transition(initiator_id, old_state, state.state, frame, 
                           f"Locked to receiver {receiver_id}")
        return True
    
    def confirm_pass(self, initiator_id: int, frame: int) -> bool:
        """
        Confirm pass reception and transition to COOLDOWN.
        
        Args:
            initiator_id: Player who initiated
            frame: Current frame
            
        Returns:
            True if transition successful
        """
        state = self._get_or_create_state(initiator_id, frame)
        
        if state.state != PassState.LOCKED:
            return False
        
        receiver_id = state.locked_receiver_id
        
        # Record pair lock
        if receiver_id is not None:
            state.recent_receivers[receiver_id] = frame
        
        # Transition to COOLDOWN
        old_state = state.state
        state.state = PassState.COOLDOWN
        state.state_start_frame = frame
        state.state_start_time = time.time()
        state.cooldown_start_frame = frame
        
        self._log_transition(initiator_id, old_state, state.state, frame, 
                           f"Pass confirmed to {receiver_id}")
        return True
    
    def cancel_pass(self, initiator_id: int, frame: int, reason: str = "Cancelled"):
        """
        Cancel a pass in INITIATING or LOCKED state, return to IDLE.
        """
        state = self._get_or_create_state(initiator_id, frame)
        
        if state.state in [PassState.INITIATING, PassState.LOCKED]:
            old_state = state.state
            state.reset(frame)
            self._log_transition(initiator_id, old_state, PassState.IDLE, frame, reason)
    
    def update(self, frame: int):
        """
        Update all player states (handle timeouts).
        
        Call this every frame.
        """
        current_time = time.time()
        
        for player_id, state in self.player_states.items():
            elapsed_time = current_time - state.state_start_time
            
            if state.state == PassState.INITIATING:
                # Timeout if no receiver found
                if elapsed_time > self.config['initiation_timeout']:
                    self.cancel_pass(player_id, frame, "Initiation timeout")
            
            elif state.state == PassState.LOCKED:
                # Timeout if no reception confirmed
                if elapsed_time > self.config['lock_timeout']:
                    self.cancel_pass(player_id, frame, "Lock timeout")
            
            elif state.state == PassState.COOLDOWN:
                # Transition back to IDLE after cooldown
                if elapsed_time > self.config['cooldown_duration']:
                    old_state = state.state
                    state.reset(frame)
                    self._log_transition(player_id, old_state, PassState.IDLE, frame, 
                                       "Cooldown complete")
    
    def get_state(self, player_id: int) -> Optional[PlayerPassState]:
        """Get current state for a player."""
        return self.player_states.get(player_id)
    
    def get_initiating_players(self) -> Dict[int, PlayerPassState]:
        """Get all players currently in INITIATING state."""
        return {pid: state for pid, state in self.player_states.items() 
                if state.state == PassState.INITIATING}
    
    def get_locked_players(self) -> Dict[int, PlayerPassState]:
        """Get all players currently in LOCKED state."""
        return {pid: state for pid, state in self.player_states.items() 
                if state.state == PassState.LOCKED}
    
    def is_pair_locked(self, initiator_id: int, receiver_id: int, frame: int) -> bool:
        """Check if A→B pair is still locked (recently passed)."""
        state = self.player_states.get(initiator_id)
        if state is None:
            return False
        
        if receiver_id not in state.recent_receivers:
            return False
        
        last_frame = state.recent_receivers[receiver_id]
        frames_since = frame - last_frame
        lock_frames = int(self.config['pair_lock_duration'] * self.config['fps'])
        
        return frames_since < lock_frames
    
    def get_debug_data(self) -> Dict:
        """Get debug data for export."""
        return {
            'state_transitions': self.state_transitions,
            'player_states': {
                pid: {
                    'state': state.state.value,
                    'state_start_frame': state.state_start_frame,
                    'initiation_frame': state.initiation_frame,
                    'locked_receiver_id': state.locked_receiver_id,
                    'recent_receivers': dict(state.recent_receivers)
                }
                for pid, state in self.player_states.items()
            }
        }

