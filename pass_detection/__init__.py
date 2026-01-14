"""
Pass Detection Module

Implements HYBRID pass detection: Players propose, Ball validates.

Architecture:
- PlayerCandidateGenerator: Generates candidates from player motion (looser thresholds)
- BallEvidenceValidator: Validates candidates with ball tracking
- HybridPassManager: Main entry point, combines player + ball
- StrictBallTracker: Strict ball tracking (DETECTED/PREDICTED/LOST, max 5-frame prediction)
- PassVisualizer: Consistent pass visualization

Key Guarantees:
- Ball can SUPPORT but never CREATE passes
- Players define events, ball validates
- No ball hallucinations (max 5-frame prediction)
- One real-world pass = exactly one detected event
"""

from .pass_state_machine import PassStateMachine, PassState, PlayerPassState
from .pass_event import PassEvent, PassLifecycleStage
# PassCandidate is in player_candidate_generator.py
from .player_motion_analyzer import PlayerMotionAnalyzer, PlayerMotionState
from .pass_validator import PassValidator, ValidationResult
from .pass_lifecycle_manager import PassLifecycleManager
from .pass_visualizer import PassVisualizer

# Hybrid system
from .ball_tracker import StrictBallTracker, BallState, BallObservation
from .player_candidate_generator import PlayerCandidateGenerator, PassCandidate
from .ball_evidence_validator import BallEvidenceValidator, BallValidationResult
from .hybrid_pass_manager import HybridPassManager

# Simple pass detector
from .simple_pass_detector import SimplePassDetector

# Shot detector
from .shot_detector import ShotDetector, ShotEvent, ShotType

# Free kick detector
from .free_kick_detector import FreeKickDetector, FreeKickEvent, FreeKickType

# Corner detector
from .corner_detector import CornerDetector, CornerEvent, CornerSide

# Unified event detector
from .event_detector import EventDetector, AllEvents

__all__ = [
    # State machine
    'PassStateMachine',
    'PassState',
    'PlayerPassState',
    
    # Events
    'PassEvent',
    'PassLifecycleStage',
    
    # Motion analysis
    'PlayerMotionAnalyzer',
    'PlayerMotionState',
    
    # Validation
    'PassValidator',
    'ValidationResult',
    
    # Legacy manager (player-only)
    'PassLifecycleManager',
    
    # Visualization
    'PassVisualizer',
    
    # Hybrid system
    'StrictBallTracker',
    'BallState',
    'BallObservation',
    'PlayerCandidateGenerator',
    'PassCandidate',
    'BallEvidenceValidator',
    'BallValidationResult',
    'HybridPassManager',
    
    # Simple pass detector
    'SimplePassDetector',
    
    # Shot detector
    'ShotDetector',
    'ShotEvent',
    'ShotType',
    
    # Free kick detector
    'FreeKickDetector',
    'FreeKickEvent',
    'FreeKickType',
    
    # Corner detector
    'CornerDetector',
    'CornerEvent',
    'CornerSide',
    
    # Unified event detector
    'EventDetector',
    'AllEvents',
]
