"""
Unified Event Detector

Coordinates all event types:
- Pass detection (existing)
- Shot detection (existing)
- Free kick detection (new)
- Corner detection (new)

Handles event deduplication and prioritization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .simple_pass_detector import SimplePassDetector
from .shot_detector import ShotDetector, ShotEvent
from .free_kick_detector import FreeKickDetector, FreeKickEvent
from .corner_detector import CornerDetector, CornerEvent
from .pass_event import PassEvent


@dataclass
class AllEvents:
    """Container for all detected events."""
    passes: List[PassEvent]
    shots: List[ShotEvent]
    free_kicks: List[FreeKickEvent]
    corners: List[CornerEvent]
    
    def get_all_events(self) -> List[Tuple[str, object]]:
        """Get all events as a list of (event_type, event) tuples."""
        events = []
        for event in self.passes:
            events.append(('pass', event))
        for event in self.shots:
            events.append(('shot', event))
        for event in self.free_kicks:
            events.append(('free_kick', event))
        for event in self.corners:
            events.append(('corner', event))
        return events


class EventDetector:
    """
    Unified event detector that coordinates all event types.
    
    Handles:
    - Pass detection
    - Shot detection
    - Free kick detection
    - Corner detection
    - Event deduplication (e.g., a shot shouldn't also be a pass)
    - Event prioritization (e.g., corner > free kick > shot > pass)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize unified event detector."""
        self.config = {
            'fps': 30.0,
            'frame_width': 1920,
            'frame_height': 1080,
        }
        
        if config:
            self.config.update(config)
        
        # Initialize individual detectors
        pass_config = {'fps': self.config.get('fps', 30.0)}
        self.pass_detector = SimplePassDetector(pass_config)
        
        shot_config = {'fps': self.config.get('fps', 30.0)}
        self.shot_detector = ShotDetector(shot_config)
        
        free_kick_config = {'fps': self.config.get('fps', 30.0)}
        self.free_kick_detector = FreeKickDetector(free_kick_config)
        
        corner_config = {'fps': self.config.get('fps', 30.0)}
        self.corner_detector = CornerDetector(corner_config)
        
        # Event priority (higher = more important, checked first)
        self.event_priority = {
            'corner': 4,
            'free_kick': 3,
            'shot': 2,
            'pass': 1,
        }
        
        # Deduplication window (frames) - events within this window are considered duplicates
        self.deduplication_window = 30  # 1 second at 30fps
    
    def initialize_goal_positions(self, frame_width: int, frame_height: int):
        """Initialize goal positions for shot detection."""
        self.pass_detector.initialize_goal_positions(frame_width, frame_height)
        self.shot_detector.estimate_goal_positions(frame_width, frame_height)
        
        # Initialize field dimensions for free kicks and corners
        self.free_kick_detector.initialize_field_dimensions(frame_width, frame_height)
        self.corner_detector.initialize_field_dimensions(frame_width, frame_height)
    
    def set_field_keypoints(self, keypoints: np.ndarray, field_corners: Optional[Dict[str, Tuple[float, float]]] = None):
        """Set field keypoints for position-based detection."""
        self.free_kick_detector.set_field_keypoints(keypoints, field_corners)
        self.corner_detector.set_field_keypoints(keypoints, field_corners)
        
        # Also set goal positions from keypoints if available
        if field_corners:
            # Estimate goal positions from corners
            # Left goal: between top_left and bottom_left
            # Right goal: between top_right and bottom_right
            if 'top_left' in field_corners and 'bottom_left' in field_corners:
                top_left = np.array(field_corners['top_left'])
                bottom_left = np.array(field_corners['bottom_left'])
                left_goal = (top_left + bottom_left) / 2.0
                self.shot_detector.left_goal_center = left_goal
            
            if 'top_right' in field_corners and 'bottom_right' in field_corners:
                top_right = np.array(field_corners['top_right'])
                bottom_right = np.array(field_corners['bottom_right'])
                right_goal = (top_right + bottom_right) / 2.0
                self.shot_detector.right_goal_center = right_goal
    
    def process_frame(self, frame: int,
                     player_positions: Dict[int, np.ndarray],
                     player_teams: Dict[int, int],
                     ball_detections: Optional[np.ndarray] = None) -> AllEvents:
        """
        Process a single frame for all event types.
        
        Args:
            frame: Current frame number
            player_positions: Dict of player_id -> position [x, y]
            player_teams: Dict of player_id -> team_id
            ball_detections: Ball positions [N, 2] or None
            
        Returns:
            AllEvents container with all detected events
        """
        # Convert ball_detections to single position if needed
        ball_position = None
        if ball_detections is not None:
            if isinstance(ball_detections, np.ndarray):
                if ball_detections.ndim == 1 and len(ball_detections) >= 2:
                    ball_position = ball_detections[:2]
                elif ball_detections.ndim == 2 and ball_detections.shape[0] > 0:
                    ball_position = ball_detections[0, :2]
        
        # Process pass detection (handles shots internally)
        self.pass_detector.process_frame(frame, player_positions, player_teams, ball_detections)
        
        # Process free kick detection
        if ball_position is not None:
            self.free_kick_detector.process_frame(frame, ball_position, player_positions, player_teams)
        
        # Process corner detection
        if ball_position is not None:
            self.corner_detector.process_frame(frame, ball_position, player_positions, player_teams)
        
        # Return empty events for now (events are collected at end)
        return AllEvents(passes=[], shots=[], free_kicks=[], corners=[])
    
    def finalize_events(self) -> AllEvents:
        """
        Finalize all events after processing all frames.
        
        This validates pass candidates and deduplicates events.
        
        Returns:
            AllEvents with all detected events
        """
        # Validate pass candidates (this also detects shots)
        self.pass_detector.validate_all_candidates()
        
        # Get all events
        passes = self.pass_detector.get_confirmed_passes()
        shots = self.pass_detector.get_detected_shots()
        free_kicks = self.free_kick_detector.get_detected_free_kicks()
        corners = self.corner_detector.get_detected_corners()
        
        # Deduplicate events (remove overlaps)
        passes, shots, free_kicks, corners = self._deduplicate_events(
            passes, shots, free_kicks, corners
        )
        
        return AllEvents(
            passes=passes,
            shots=shots,
            free_kicks=free_kicks,
            corners=corners
        )
    
    def _deduplicate_events(self, passes: List[PassEvent], shots: List[ShotEvent],
                           free_kicks: List[FreeKickEvent], corners: List[CornerEvent]) -> Tuple[
        List[PassEvent], List[ShotEvent], List[FreeKickEvent], List[CornerEvent]
    ]:
        """
        Deduplicate events by removing overlaps.
        
        Priority: corner > free_kick > shot > pass
        
        If events overlap in time and involve the same player, keep only the highest priority event.
        """
        # Convert all events to a common format for comparison
        all_events = []
        
        for event in corners:
            all_events.append(('corner', event, event.start_frame, event.end_frame, event.kicker_id))
        for event in free_kicks:
            all_events.append(('free_kick', event, event.start_frame, event.end_frame, event.kicker_id))
        for event in shots:
            all_events.append(('shot', event, event.start_frame, event.end_frame, event.shooter_id))
        for event in passes:
            all_events.append(('pass', event, event.start_frame, event.end_frame if event.end_frame else event.start_frame, event.from_player_id))
        
        # Sort by priority (descending) and then by start frame
        all_events.sort(key=lambda x: (-self.event_priority.get(x[0], 0), x[2]))
        
        # Track which events to keep
        kept_events = {
            'corner': set(),
            'free_kick': set(),
            'shot': set(),
            'pass': set(),
        }
        
        # Track which frames/players are already covered
        covered_frames = {}  # frame -> set of player_ids
        
        for event_type, event, start_frame, end_frame, player_id in all_events:
            # Check if this event overlaps with a higher priority event
            overlaps = False
            
            # Check frames in the event window
            event_frames = set(range(start_frame, end_frame + 1))
            
            for frame in event_frames:
                if frame in covered_frames:
                    # Check if same player is involved
                    if player_id in covered_frames[frame]:
                        overlaps = True
                        break
                    # Also check if events are very close (within deduplication window)
                    for covered_frame, covered_players in covered_frames.items():
                        if abs(frame - covered_frame) <= self.deduplication_window:
                            if player_id in covered_players:
                                overlaps = True
                                break
                    if overlaps:
                        break
            
            if not overlaps:
                # Keep this event
                kept_events[event_type].add(id(event))
                
                # Mark frames as covered
                for frame in event_frames:
                    if frame not in covered_frames:
                        covered_frames[frame] = set()
                    covered_frames[frame].add(player_id)
        
        # Filter events based on what was kept
        filtered_passes = [p for p in passes if id(p) in kept_events['pass']]
        filtered_shots = [s for s in shots if id(s) in kept_events['shot']]
        filtered_free_kicks = [f for f in free_kicks if id(f) in kept_events['free_kick']]
        filtered_corners = [c for c in corners if id(c) in kept_events['corner']]
        
        return filtered_passes, filtered_shots, filtered_free_kicks, filtered_corners
    
    def get_stats(self) -> Dict:
        """Get statistics from all detectors."""
        return {
            'passes': self.pass_detector.get_metrics(),
            'shots': self.shot_detector.get_stats(),
            'free_kicks': self.free_kick_detector.get_stats(),
            'corners': self.corner_detector.get_stats(),
        }
