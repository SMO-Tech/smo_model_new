"""
Pass Event Module

Defines the PassEvent as a discrete, single-occurrence event with full lifecycle tracking.
Each pass is emitted exactly once.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import uuid


class PassLifecycleStage(Enum):
    """Stages of a pass lifecycle."""
    INITIATED = "initiated"      # Pass detected, looking for receiver
    CANDIDATE = "candidate"      # Receiver candidate identified
    CONFIRMED = "confirmed"      # Reception confirmed
    REJECTED = "rejected"        # Pass rejected (failed validation)
    EXPIRED = "expired"          # Timed out without confirmation


@dataclass
class PassEvent:
    """
    Represents a single pass event.
    
    Key principle: One real-world pass = exactly one PassEvent.
    This event is NEVER duplicated or re-emitted.
    """
    # Unique identifier (prevents any possibility of duplication)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Initiator info (set at creation, NEVER modified)
    from_player_id: int = 0
    team_id: int = 0
    start_frame: int = 0  # The EARLIEST detected initiation frame
    start_position: List[float] = field(default_factory=list)
    
    # Receiver info (set when confirmed)
    to_player_id: Optional[int] = None
    end_frame: Optional[int] = None  # Frame where reception confirmed
    end_position: Optional[List[float]] = None
    
    # Lifecycle tracking
    lifecycle_stage: PassLifecycleStage = PassLifecycleStage.INITIATED
    rejection_reason: Optional[str] = None
    
    # Confidence scoring
    confidence: float = 0.0
    initiation_confidence: float = 0.0
    direction_confidence: float = 0.0
    distance_confidence: float = 0.0
    reception_confidence: float = 0.0
    
    # Physics metrics
    distance_meters: float = 0.0
    duration_seconds: float = 0.0
    implied_speed: float = 0.0
    direction_angle: float = 0.0
    
    # Timestamps (for debugging)
    created_timestamp: float = 0.0
    confirmed_timestamp: Optional[float] = None
    
    # Emitted flag (ensures single emission)
    _emitted: bool = False
    
    def confirm(self, receiver_id: int, end_frame: int, end_position: List[float],
                confidence: float) -> bool:
        """
        Confirm the pass with receiver info.
        
        Returns:
            True if confirmation successful (first time), False if already confirmed
        """
        if self.lifecycle_stage != PassLifecycleStage.CANDIDATE:
            return False
        
        self.to_player_id = receiver_id
        self.end_frame = end_frame
        self.end_position = end_position
        self.confidence = confidence
        self.lifecycle_stage = PassLifecycleStage.CONFIRMED
        
        import time
        self.confirmed_timestamp = time.time()
        
        return True
    
    def reject(self, reason: str):
        """Mark the pass as rejected."""
        self.lifecycle_stage = PassLifecycleStage.REJECTED
        self.rejection_reason = reason
    
    def expire(self):
        """Mark the pass as expired (timed out)."""
        self.lifecycle_stage = PassLifecycleStage.EXPIRED
        self.rejection_reason = "Timeout without confirmation"
    
    def mark_emitted(self) -> bool:
        """
        Mark the pass as emitted.
        
        Returns:
            True if this is the first emission, False if already emitted
        """
        if self._emitted:
            return False
        self._emitted = True
        return True
    
    @property
    def is_confirmed(self) -> bool:
        """Check if pass is confirmed."""
        return self.lifecycle_stage == PassLifecycleStage.CONFIRMED
    
    @property
    def is_emitted(self) -> bool:
        """Check if pass has been emitted."""
        return self._emitted
    
    @property
    def duration_frames(self) -> int:
        """Get duration in frames."""
        if self.end_frame is None:
            return 0
        return self.end_frame - self.start_frame
    
    def to_dict(self) -> dict:
        """Convert to dictionary for export."""
        return {
            'event_id': self.event_id,
            'from_player_id': self.from_player_id,
            'to_player_id': self.to_player_id,
            'team_id': self.team_id,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'start_position': self.start_position,
            'end_position': self.end_position,
            'lifecycle_stage': self.lifecycle_stage.value,
            'confidence': self.confidence,
            'initiation_confidence': self.initiation_confidence,
            'direction_confidence': self.direction_confidence,
            'distance_confidence': self.distance_confidence,
            'reception_confidence': self.reception_confidence,
            'distance_meters': self.distance_meters,
            'duration_seconds': self.duration_seconds,
            'implied_speed': self.implied_speed,
            'rejection_reason': self.rejection_reason,
            'emitted': self._emitted
        }


# PassCandidate moved to player_candidate_generator.py for hybrid system

