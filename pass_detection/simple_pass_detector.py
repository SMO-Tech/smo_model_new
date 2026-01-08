"""
Simple Pass Detector Module

Detects passes based on ball proximity to players:
1. Ball is near player A (initiator)
2. Ball moves away from A
3. Ball arrives near player B (receiver)

No complex validation - just proximity-based detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time
import json
import os

from .ball_tracker import StrictBallTracker, BallState as OldBallState, BallObservation as OldBallObservation
from .physics_ball_tracker import PhysicsBallTracker, BallState, BallObservation
from .shot_detector import ShotDetector, ShotEvent, ShotType
from .pass_event import PassEvent, PassLifecycleStage


@dataclass
class BallPossession:
    """Tracks which player has the ball."""
    player_id: int
    team_id: int
    start_frame: int
    position: np.ndarray
    confidence: float = 1.0


class SimplePassDetector:
    """
    Simple proximity-based pass detector.
    
    Logic:
    - Track who has the ball (closest player within threshold)
    - When possession changes from A to B (same team), it's a pass
    - No complex validation needed
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the simple pass detector."""
        self.config = {
            # Possession thresholds - IN PIXELS (MORE LENIENT FOR BETTER DETECTION)
            'possession_radius': 300.0,    # Increased from 200 to 300 (catch more passes)
            'possession_tolerance': 100.0,   # Tolerance margin: if ball is within this distance of radius, still consider in possession
            'min_pass_distance': 50.0,     # Reduced from 80 to 50 (allow shorter passes)
            'max_pass_distance': 1200.0,  # Increased from 800 to 1200 (allow longer passes)
            
            # Timing - MINIMUM for one-touch pass detection
            'min_possession_frames': 1,    # Allow one-touch passes (was 5, reduced to 1)
            'cooldown_frames': 15,         # Reduced from 60 to 15 (0.5s instead of 2s - allow rapid passes)
            'min_pass_duration': 1,        # Allow one-touch passes (1 frame minimum, was 5)
            'max_pass_duration_frames': 150,  # Increased from 120 to 150 (5 seconds max)
            
            # Frame rate
            'fps': 30.0,
            
            # Use pixel coordinates (since homography often fails)
            'use_pixel_coords': True,
            
            # Trajectory validation (MORE LENIENT for realistic soccer passes)
            'validate_trajectory': False,  # DISABLED - too strict, was rejecting valid passes
            'trajectory_similarity_threshold': 0.2,  # Lowered from 0.3 to 0.2 (allow more deviation)
            'max_path_ratio': 3.0,  # Increased from 2.0 to 3.0 (allow 200% extra path length)
            'short_pass_exemption': 300.0,  # Increased from 200 to 300 (skip validation for more passes)
            'quick_pass_exemption_frames': 15,  # Increased from 10 to 15 (more lenient for quick passes)
            
            # Interception detection
            'detect_interceptions': True,  # Enable detection of passes between different teams
            
            # Speed thresholds (for fast passes)
            'max_speed_pixels_per_frame': 80,  # Increased from 50 to 80 for very fast passes
            
            # Dynamic radius (MORE LENIENT)
            'dynamic_radius_enabled': True,  # Enable dynamic possession radius
            'base_radius': 300.0,  # Increased from 200 to 300
            'speed_factor': 2.5,  # Increased from 2.0 to 2.5
            'min_radius': 250.0,  # Increased from 150 to 250
            'max_radius': 500.0,  # Increased from 350 to 500
        }
        
        if config:
            self.config.update(config)
        
        # Ball tracker - using physics-based predictive tracker
        # Set fps in config for physics tracker
        physics_config = config.copy() if config else {}
        physics_config['fps'] = self.config.get('fps', 30.0)
        self.ball_tracker = PhysicsBallTracker(physics_config)
        
        # Shot detector - to differentiate shots from passes
        shot_config = {'fps': self.config.get('fps', 30.0)}
        self.shot_detector = ShotDetector(shot_config)
        
        # Possession tracking
        self.current_possession: Optional[BallPossession] = None
        self.possession_frames: int = 0
        self.last_possession_frame: int = -1
        self.last_shot_candidate_frame: int = -1  # Track last frame where we created a shot candidate for current possession
        
        # Pass tracking - TWO PHASE SYSTEM
        self.pass_candidates: List[Dict] = []  # Phase 1: Collect all candidates
        self.confirmed_passes: List[PassEvent] = []  # Phase 2: Validated passes
        self.detected_shots: List[ShotEvent] = []  # Shots detected (not passes)
        self.pass_counter: int = 0
        
        # Cooldowns
        self.player_cooldowns: Dict[int, int] = {}  # player_id -> end_frame
        self.shot_cooldowns: Dict[int, int] = {}  # player_id -> end_frame (for shot deduplication)
        
        # Team map
        self.team_map: Dict[int, int] = {}
        
        # Ball trajectory buffer for validation
        self.ball_trajectory_buffer: List[Tuple[int, np.ndarray]] = []
        self.max_trajectory_frames = 30
        
        # Frame snapshot storage for pass validation
        self.frame_snapshots = {}  # {frame_idx: {'player_positions': {}, 'player_teams': {}, 'all_tracker_ids': []}}  # 1 second at 30fps
        
        # Debug logging
        self.debug_log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.cursor', 'debug.log')
        self.debug_mode = True
        
        # Metrics
        self.metrics = {
            'total_possessions': 0,
            'passes_detected': 0,
            'passes_rejected_distance': 0,
            'passes_rejected_cooldown': 0,
            'passes_rejected_same_player': 0,
            'passes_rejected_trajectory': 0,
            'passes_rejected_duration': 0,
            'near_miss_radius': 0,  # Players within 150px but not 100px
            'near_miss_duration': 0,  # Possession changes with < 10 but >= 5 frames
        }
    
    def initialize_goal_positions(self, frame_width: int, frame_height: int):
        """Initialize goal positions for shot detection."""
        self.shot_detector.estimate_goal_positions(frame_width, frame_height)
    
    def process_frame(self, frame: int,
                     player_positions: Dict[int, np.ndarray],
                     player_teams: Dict[int, int],
                     ball_detections: Optional[np.ndarray] = None) -> List[PassEvent]:
        """
        Process a single frame with complete frame snapshot storage.
        
        Stores complete frame data for pass validation and debugging.
        """
        # Store complete frame snapshot
        self.frame_snapshots[frame] = {
            'player_positions': player_positions.copy(),
            'player_teams': player_teams.copy(),
            'all_tracker_ids': list(player_positions.keys()),
            'ball_position': ball_detections.copy() if ball_detections is not None else None
        }
        """
        Process a single frame.
        
        Args:
            frame: Current frame number
            player_positions: Dict of player_id -> position [x, y] in pitch coords
            player_teams: Dict of player_id -> team_id
            ball_detections: Ball positions [N, 2] in pitch coords, or None
            
        Returns:
            List of newly detected passes
        """
        # Update team map
        for player_id, team_id in player_teams.items():
            if player_id not in self.team_map:
                self.team_map[player_id] = team_id
        
        # Update ball tracker
        # Convert ball_detections to correct format if needed
        if ball_detections is not None:
            if isinstance(ball_detections, np.ndarray):
                if ball_detections.ndim == 1:
                    # Single position [x, y] -> reshape to [[x, y]]
                    ball_detections = ball_detections.reshape(1, -1)
                elif ball_detections.ndim == 2 and ball_detections.shape[1] == 2:
                    # Already correct shape [N, 2]
                    pass
                else:
                    # Invalid shape, skip
                    ball_detections = None
        
        ball_obs = self.ball_tracker.update(frame, ball_detections)
        
        # #region agent log - BALL TRACKING STATE
        try:
            with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'hypothesisId': 'C',
                    'location': 'simple_pass_detector.py:192',
                    'message': 'ball_tracking_state',
                    'data': {
                        'frame': int(frame),
                        'ball_detections_provided': ball_detections is not None,
                        'ball_obs_state': str(ball_obs.state) if ball_obs else 'None',
                        'ball_position': ball_obs.position.tolist() if ball_obs and ball_obs.position is not None else None,
                        'num_players': len(player_positions)
                    },
                    'timestamp': int(time.time() * 1000),
                    'sessionId': 'debug-session',
                    'runId': 'run1'
                }) + '\n')
        except: pass
        # #endregion
        
        # Physics tracker ALWAYS provides position (never None)
        # Accept all states: DETECTED, PREDICTED, INTERPOLATED
        if ball_obs.position is None:
            # This should never happen with physics tracker, but handle gracefully
            return []
        
        ball_pos = ball_obs.position
        
        # Debug: Track ball detection rate
        if not hasattr(self, '_ball_detection_stats'):
            self._ball_detection_stats = {'detected': 0, 'predicted': 0, 'lost': 0}
        if ball_obs.state == BallState.DETECTED:
            self._ball_detection_stats['detected'] += 1
        elif ball_obs.state == BallState.PREDICTED:
            self._ball_detection_stats['predicted'] += 1
        else:
            self._ball_detection_stats['lost'] += 1
        
        # Store ball position for trajectory analysis
        self.ball_trajectory_buffer.append((frame, ball_pos.copy()))
        # Keep buffer at reasonable size
        if len(self.ball_trajectory_buffer) > self.max_trajectory_frames:
            self.ball_trajectory_buffer.pop(0)
        
        # Find closest player to ball
        closest_player = None
        closest_distance = float('inf')
        
        if len(player_positions) == 0:
            # No players - can't detect possession
            return []
        
        for player_id, pos in player_positions.items():
            if pos is None:
                continue
            distance = np.linalg.norm(ball_pos - pos)
            if distance < closest_distance:
                closest_distance = distance
                closest_player = player_id
        
        # Check if ball is in possession (within radius)
        new_passes = []
        
        # Calculate dynamic possession radius based on ball speed
        ball_speed = 0.0
        if len(self.ball_trajectory_buffer) >= 2:
            last_pos = self.ball_trajectory_buffer[-1][1]
            prev_pos = self.ball_trajectory_buffer[-2][1]
            ball_speed = np.linalg.norm(last_pos - prev_pos)
        
        current_radius = self._calculate_dynamic_radius(ball_speed)
        
        # Track near-misses for debugging (old threshold vs new)
        if closest_player is not None:
            if 100.0 < closest_distance <= 150.0:
                self.metrics['near_miss_radius'] += 1
                # #region agent log - NEAR MISS RADIUS
                try:
                    with open(self.debug_log_path, 'a') as f:
                        f.write(json.dumps({
                            'hypothesisId': 'A',
                            'location': 'simple_pass_detector:possession_check',
                            'message': 'near_miss_radius',
                            'data': {
                                'frame': int(frame),
                                'player_id': int(closest_player),
                                'distance': float(closest_distance),
                                'old_threshold': 100.0,
                                'new_threshold': 150.0,
                                'would_detect': True
                            },
                            'timestamp': int(time.time() * 1000),
                            'sessionId': 'pass-detection-improvement'
                        }) + '\n')
                except:
                    pass
                # #endregion
        

        # #region agent log - POSSESSION CHECK
        try:
            with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'hypothesisId': 'A,C',
                    'location': 'simple_pass_detector.py:293',
                    'message': 'possession_check',
                    'data': {
                        'frame': int(frame),
                        'closest_player': int(closest_player) if closest_player is not None else None,
                        'closest_distance': float(closest_distance) if closest_player is not None else None,
                        'current_radius': float(current_radius),
                        'within_radius': closest_player is not None and closest_distance <= current_radius,
                        'current_possession_player': int(self.current_possession.player_id) if (self.current_possession is not None and hasattr(self.current_possession, 'player_id')) else None,
                        'current_possession_frames': int(self.possession_frames) if (self.current_possession is not None) else 0
                    },
                    'timestamp': int(time.time() * 1000),
                    'sessionId': 'debug-session',
                    'runId': 'run1'
                }) + '\n')
        except: pass
        # #endregion
        
        # Add tolerance margin: if ball is within tolerance of radius, still consider it in possession
        # This helps catch passes where ball is moving quickly and might be slightly outside strict radius
        possession_tolerance = self.config.get('possession_tolerance', 50.0)
        effective_radius = current_radius + possession_tolerance
        
        # #region agent log - POSSESSION CHECK WITH TOLERANCE
        try:
            with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'hypothesisId': 'B',
                    'location': 'simple_pass_detector.py:318',
                    'message': 'possession_check_with_tolerance',
                    'data': {
                        'frame': int(frame),
                        'closest_player': int(closest_player) if closest_player is not None else None,
                        'closest_distance': float(closest_distance) if closest_player is not None else None,
                        'current_radius': float(current_radius),
                        'possession_tolerance': float(possession_tolerance),
                        'effective_radius': float(effective_radius),
                        'within_effective_radius': closest_player is not None and closest_distance <= effective_radius,
                        'within_strict_radius': closest_player is not None and closest_distance <= current_radius
                    },
                    'timestamp': int(time.time() * 1000),
                    'sessionId': 'debug-session',
                    'runId': 'run1'
                }) + '\n')
        except: pass
        # #endregion
        
        if closest_player is not None and closest_distance <= effective_radius:
            team_id = self.team_map.get(closest_player, 0)
            
            if self.current_possession is None:
                # First possession
                self.current_possession = BallPossession(
                    player_id=closest_player,
                    team_id=team_id,
                    start_frame=frame,
                    position=player_positions[closest_player].copy()
                )
                self.possession_frames = 1
                self.last_shot_candidate_frame = -1  # Reset shot candidate tracking for new possession
                # Don't count first possession as a "total possession" - only count when candidates are created
                # self.metrics['total_possessions'] += 1  # REMOVED
                
            elif self.current_possession.player_id != closest_player:
                # Possession change - potential pass
                # #region agent log - POSSESSION CHANGE DETECTED
                try:
                    with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            'hypothesisId': 'A,C',
                            'location': 'simple_pass_detector.py:287',
                            'message': 'possession_change_detected',
                            'data': {
                                'frame': int(frame),
                                'from_player': int(self.current_possession.player_id),
                                'to_player': int(closest_player),
                                'from_team': int(self.current_possession.team_id),
                                'to_team': int(team_id),
                                'possession_frames': int(self.possession_frames),
                                'min_possession_frames': int(self.config['min_possession_frames']),
                                'meets_min_frames': self.possession_frames >= self.config['min_possession_frames']
                            },
                            'timestamp': int(time.time() * 1000),
                            'sessionId': 'debug-session',
                            'runId': 'run1'
                        }) + '\n')
                except: pass
                # #endregion
                
                # Track near-miss duration (old threshold vs new)
                if 5 <= self.possession_frames < 10:
                    self.metrics['near_miss_duration'] += 1
                
                # PHASE 1: COLLECT CANDIDATES (Very lenient - no early rejection)
                # Only check minimum possession frames, collect everything else
                if self.possession_frames >= self.config['min_possession_frames']:
                    # Check if it's a pass (same team, not same player)
                    same_team = (self.current_possession.team_id == team_id)
                    
                    # Allow both same-team passes AND interceptions
                    detect_interceptions = self.config.get('detect_interceptions', True)
                    is_valid_pass_type = same_team or detect_interceptions
                    
                    # #region agent log - CANDIDATE VALIDATION CHECK
                    try:
                        with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                'hypothesisId': 'B,D',
                                'location': 'simple_pass_detector.py:318',
                                'message': 'candidate_validation_check',
                                'data': {
                                    'frame': int(frame),
                                    'from_player': int(self.current_possession.player_id),
                                    'to_player': int(closest_player),
                                    'same_team': bool(same_team),
                                    'detect_interceptions': bool(detect_interceptions),
                                    'is_valid_pass_type': bool(is_valid_pass_type),
                                    'different_player': self.current_possession.player_id != closest_player,
                                    'will_create_candidate': is_valid_pass_type and self.current_possession.player_id != closest_player
                                },
                                'timestamp': int(time.time() * 1000),
                                'sessionId': 'debug-session',
                                'runId': 'run1'
                            }) + '\n')
                    except: pass
                    # #endregion
                    
                    if is_valid_pass_type and self.current_possession.player_id != closest_player:
                        # PHASE 1: Collect candidate (NO validation yet)
                        candidate = {
                            'from_player': self.current_possession.player_id,
                            'to_player': closest_player,
                            'from_team': self.current_possession.team_id,
                            'to_team': team_id,
                            'start_frame': self.current_possession.start_frame,
                            'end_frame': frame,
                            'from_pos': self.current_possession.position.copy(),
                            'to_pos': player_positions[closest_player].copy(),
                            'possession_frames': self.possession_frames,
                            'ball_trajectory': self._extract_ball_trajectory(
                                self.current_possession.start_frame,
                                frame
                            )
                        }
                        
                        # Add to candidates list (will validate later)
                        self.pass_candidates.append(candidate)
                        self.metrics['total_possessions'] += 1
                        
                        # #region agent log - CANDIDATE CREATED
                        try:
                            with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                                f.write(json.dumps({
                                    'hypothesisId': 'A,B,C,D',
                                    'location': 'simple_pass_detector.py:343',
                                    'message': 'candidate_created',
                                    'data': {
                                        'frame': int(frame),
                                        'candidate_num': len(self.pass_candidates),
                                        'from_player': int(candidate['from_player']),
                                        'to_player': int(candidate['to_player']),
                                        'distance': float(np.linalg.norm(candidate['to_pos'] - candidate['from_pos'])),
                                        'possession_frames': int(self.possession_frames)
                                    },
                                    'timestamp': int(time.time() * 1000),
                                    'sessionId': 'debug-session',
                                    'runId': 'run1'
                                }) + '\n')
                        except: pass
                        # #endregion
                        
                        # Debug: Log candidate collection
                        if len(self.pass_candidates) <= 10:
                            print(f"[DEBUG] Collected candidate #{len(self.pass_candidates)}: Frame {frame}, Player {candidate['from_player']} -> {candidate['to_player']}, Distance: {np.linalg.norm(candidate['to_pos'] - candidate['from_pos']):.1f}px, Possession frames: {self.possession_frames}")
                    else:
                        # #region agent log - CANDIDATE REJECTED
                        try:
                            reason = []
                            if not is_valid_pass_type:
                                reason.append('invalid_pass_type')
                            if self.current_possession.player_id == closest_player:
                                reason.append('same_player')
                            with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                                f.write(json.dumps({
                                    'hypothesisId': 'B,D',
                                    'location': 'simple_pass_detector.py:350',
                                    'message': 'candidate_rejected',
                                    'data': {
                                        'frame': int(frame),
                                        'from_player': int(self.current_possession.player_id),
                                        'to_player': int(closest_player),
                                        'reason': reason,
                                        'same_team': bool(same_team),
                                        'is_valid_pass_type': bool(is_valid_pass_type),
                                        'different_player': self.current_possession.player_id != closest_player
                                    },
                                    'timestamp': int(time.time() * 1000),
                                    'sessionId': 'debug-session',
                                    'runId': 'run1'
                                }) + '\n')
                        except: pass
                        # #endregion
                        
                        # Debug: Log why candidate wasn't created
                        if len(self.pass_candidates) <= 10:
                            reason = []
                            if self.possession_frames < self.config['min_possession_frames']:
                                reason.append(f"possession too short ({self.possession_frames} < {self.config['min_possession_frames']})")
                            if not is_valid_pass_type:
                                reason.append(f"not valid pass type (same_team={same_team}, detect_interceptions={detect_interceptions})")
                            if self.current_possession.player_id == closest_player:
                                reason.append("same player")
                            print(f"[DEBUG] Skipped candidate at frame {frame}: {', '.join(reason) if reason else 'unknown'}")
                else:
                    # #region agent log - CANDIDATE REJECTED TOO SHORT
                    try:
                        with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                'hypothesisId': 'A',
                                'location': 'simple_pass_detector.py:361',
                                'message': 'candidate_rejected_too_short',
                                'data': {
                                    'frame': int(frame),
                                    'from_player': int(self.current_possession.player_id),
                                    'to_player': int(closest_player),
                                    'possession_frames': int(self.possession_frames),
                                    'min_possession_frames': int(self.config['min_possession_frames'])
                                },
                                'timestamp': int(time.time() * 1000),
                                'sessionId': 'debug-session',
                                'runId': 'run1'
                            }) + '\n')
                    except: pass
                    # #endregion
                    
                    # Debug: Log why candidate wasn't created (possession too short)
                    if len(self.pass_candidates) <= 10:
                        print(f"[DEBUG] Skipped candidate at frame {frame}: possession too short ({self.possession_frames} < {self.config['min_possession_frames']})")
                
                # Start new possession
                self.current_possession = BallPossession(
                    player_id=closest_player,
                    team_id=team_id,
                    start_frame=frame,
                    position=player_positions[closest_player].copy()
                )
                self.possession_frames = 1
                self.last_shot_candidate_frame = -1  # Reset shot candidate tracking for new possession
                self.metrics['total_possessions'] += 1
            else:
                # Same player still has ball
                self.possession_frames += 1
        else:
            # Ball not in possession of anyone
            # #region agent log - BALL OUT OF POSSESSION
            try:
                with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'hypothesisId': 'C',
                        'location': 'simple_pass_detector.py:519',
                        'message': 'ball_out_of_possession',
                        'data': {
                            'frame': int(frame),
                            'closest_player': int(closest_player) if closest_player is not None else None,
                            'closest_distance': float(closest_distance) if closest_player is not None else None,
                            'current_radius': float(current_radius),
                            'had_possession': self.current_possession is not None,
                            'previous_player': int(self.current_possession.player_id) if self.current_possession else None,
                            'previous_possession_frames': int(self.possession_frames) if self.current_possession else 0
                        },
                        'timestamp': int(time.time() * 1000),
                        'sessionId': 'debug-session',
                        'runId': 'run1'
                    }) + '\n')
            except: pass
            # #endregion
            
            # Check if this could be a shot: player had possession, ball went out of possession
            # This happens when a player shoots and ball goes toward goal (no receiver)
            # CRITICAL: Only create ONE shot candidate per possession loss (not one per frame)
            if (self.current_possession is not None and 
                self.possession_frames >= self.config['min_possession_frames'] and
                frame != self.last_shot_candidate_frame):  # Only create once per possession loss
                # Ball went out of possession - could be a shot
                # Create a shot candidate with no receiver (to_player = None)
                ball_traj = self._extract_ball_trajectory(self.current_possession.start_frame, frame)
                
                # Check if ball trajectory points toward goal (using shot detector logic)
                if self.shot_detector and len(ball_traj) >= 2:
                    # Get ball end position
                    ball_end_pos = ball_traj[-1] if ball_traj else ball_pos
                    
                    # Check if this might be a shot (ball moving toward goal, no receiver)
                    shot_candidate = {
                        'from_player': self.current_possession.player_id,
                        'to_player': None,  # No receiver - ball goes to goal
                        'from_team': self.current_possession.team_id,
                        'to_team': None,
                        'start_frame': self.current_possession.start_frame,
                        'end_frame': frame,
                        'from_pos': self.current_possession.position.copy(),
                        'to_pos': ball_end_pos.copy(),
                        'possession_frames': self.possession_frames,
                        'ball_trajectory': ball_traj
                    }
                    
                    # Add to candidates - will be checked for shots in validate_all_candidates
                    self.pass_candidates.append(shot_candidate)
                    self.last_shot_candidate_frame = frame  # Mark that we created a candidate for this possession
                    
                    # #region agent log - SHOT CANDIDATE CREATED
                    try:
                        with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                'hypothesisId': 'F',
                                'location': 'simple_pass_detector.py:575',
                                'message': 'shot_candidate_created',
                                'data': {
                                    'frame': int(frame),
                                    'from_player': int(self.current_possession.player_id),
                                    'possession_frames': int(self.possession_frames),
                                    'ball_end_pos': ball_end_pos.tolist()
                                },
                                'timestamp': int(time.time() * 1000),
                                'sessionId': 'debug-session',
                                'runId': 'run1'
                            }) + '\n')
                    except: pass
                    # #endregion
            
            # Keep current possession but don't increment frames
            pass
        
        return new_passes
    
    def _create_pass(self, from_player: int, to_player: int, team_id: int,
                    start_frame: int, end_frame: int,
                    from_pos: np.ndarray, to_pos: np.ndarray,
                    receiver_team_id: Optional[int] = None) -> Optional[PassEvent]:
        """Create a pass event if valid."""
        
        # Check same player
        if from_player == to_player:
            self.metrics['passes_rejected_same_player'] += 1
            # #region agent log - REJECTED SAME PLAYER
            try:
                with open(self.debug_log_path, 'a') as f:
                    f.write(json.dumps({
                        'hypothesisId': 'I',
                        'location': 'simple_pass_detector:_create_pass',
                        'message': 'pass_rejected_same_player',
                        'data': {
                            'frame': int(end_frame),
                            'player_id': int(from_player)
                        },
                        'timestamp': int(time.time() * 1000),
                        'sessionId': 'pass-detection-improvement'
                    }) + '\n')
            except:
                pass
            # #endregion
            return None
        
        # Check distance
        distance = np.linalg.norm(to_pos - from_pos)
        if distance < self.config['min_pass_distance']:
            self.metrics['passes_rejected_distance'] += 1
            # #region agent log - REJECTED DISTANCE TOO SHORT
            try:
                with open(self.debug_log_path, 'a') as f:
                    f.write(json.dumps({
                        'hypothesisId': 'I',
                        'location': 'simple_pass_detector:_create_pass',
                        'message': 'pass_rejected_distance_too_short',
                        'data': {
                            'frame': int(end_frame),
                            'from_player': int(from_player),
                            'to_player': int(to_player),
                            'distance': float(distance),
                            'min_required': float(self.config['min_pass_distance'])
                        },
                        'timestamp': int(time.time() * 1000),
                        'sessionId': 'pass-detection-improvement'
                    }) + '\n')
            except:
                pass
            # #endregion
            return None
        if distance > self.config['max_pass_distance']:
            self.metrics['passes_rejected_distance'] += 1
            # #region agent log - REJECTED DISTANCE TOO LONG
            try:
                with open(self.debug_log_path, 'a') as f:
                    f.write(json.dumps({
                        'hypothesisId': 'I',
                        'location': 'simple_pass_detector:_create_pass',
                        'message': 'pass_rejected_distance_too_long',
                        'data': {
                            'frame': int(end_frame),
                            'from_player': int(from_player),
                            'to_player': int(to_player),
                            'distance': float(distance),
                            'max_allowed': float(self.config['max_pass_distance'])
                        },
                        'timestamp': int(time.time() * 1000),
                        'sessionId': 'pass-detection-improvement'
                    }) + '\n')
            except:
                pass
            # #endregion
            return None
        
        # Check minimum pass duration (allow one-touch passes)
        pass_duration_frames = end_frame - start_frame
        min_duration = self.config.get('min_pass_duration', 1)  # Allow 1-frame passes
        if pass_duration_frames < min_duration:
            self.metrics['passes_rejected_duration'] += 1
            # #region agent log - REJECTED DURATION TOO SHORT
            try:
                with open(self.debug_log_path, 'a') as f:
                    f.write(json.dumps({
                        'hypothesisId': 'I',
                        'location': 'simple_pass_detector:_create_pass',
                        'message': 'pass_rejected_duration_too_short',
                        'data': {
                            'frame': int(end_frame),
                            'from_player': int(from_player),
                            'to_player': int(to_player),
                            'duration_frames': int(pass_duration_frames),
                            'min_required': int(min_duration)
                        },
                        'timestamp': int(time.time() * 1000),
                        'sessionId': 'pass-detection-improvement'
                    }) + '\n')
            except:
                pass
            # #endregion
            return None
        
        # Check maximum pass duration (passes can't take 18+ seconds!)
        max_duration_frames = self.config.get('max_pass_duration_frames', 120)  # 4 seconds at 30fps
        if pass_duration_frames > max_duration_frames:
            self.metrics['passes_rejected_distance'] += 1
            return None
        
        # Check cooldown
        if from_player in self.player_cooldowns:
            if end_frame < self.player_cooldowns[from_player]:
                self.metrics['passes_rejected_cooldown'] += 1
                # #region agent log - REJECTED COOLDOWN
                try:
                    with open(self.debug_log_path, 'a') as f:
                        f.write(json.dumps({
                            'hypothesisId': 'I',
                            'location': 'simple_pass_detector:_create_pass',
                            'message': 'pass_rejected_cooldown',
                            'data': {
                                'frame': int(end_frame),
                                'from_player': int(from_player),
                                'to_player': int(to_player),
                                'cooldown_until': int(self.player_cooldowns[from_player]),
                                'current_frame': int(end_frame)
                            },
                            'timestamp': int(time.time() * 1000),
                            'sessionId': 'pass-detection-improvement'
                        }) + '\n')
                except:
                    pass
                # #endregion
                return None
        
        # Create pass
        self.pass_counter += 1
        
        # Set cooldown
        self.player_cooldowns[from_player] = end_frame + self.config['cooldown_frames']
        
        duration = (end_frame - start_frame) / self.config['fps']
        
        return PassEvent(
            event_id=f"pass_{self.pass_counter}",
            from_player_id=from_player,
            to_player_id=to_player,
            team_id=team_id,
            receiver_team_id=receiver_team_id if receiver_team_id is not None else team_id,
            start_frame=start_frame,
            end_frame=end_frame,
            start_position=from_pos.tolist(),
            end_position=to_pos.tolist(),
            confidence=0.8,  # Simple detector = fixed confidence
            distance_meters=float(distance),
            duration_seconds=duration,
            lifecycle_stage=PassLifecycleStage.CONFIRMED,
            created_timestamp=time.time()
        )
    
    def validate_all_candidates(self) -> None:
        """
        PHASE 2: Validate all collected candidates with multi-layer validation.
        This runs after all frames are processed.
        """
        if len(self.pass_candidates) == 0:
            print(f"\n[Pass Detection] ⚠️  WARNING: No candidates collected! Check ball tracking and possession detection.")
            return
        
        print(f"\n[Pass Detection] Validating {len(self.pass_candidates)} candidates...")
        
        rejected_physical = 0
        rejected_trajectory = 0
        rejected_temporal = 0
        rejected_cooldown = 0
        
        shots_detected = 0
        
        for i, candidate in enumerate(self.pass_candidates):
            # Layer 1: Physical reality check
            if not self._validate_physical(candidate):
                rejected_physical += 1
                if len(self.pass_candidates) <= 10:
                    print(f"   [REJECTED - Physical] Candidate #{i+1}: frame {candidate['start_frame']} -> {candidate['end_frame']}")
                continue
            
            # CHECK FOR SHOT FIRST (before pass validation)
            # Get ball trajectory and end position
            ball_trajectory = candidate.get('ball_trajectory', [])
            ball_end_pos = np.array(candidate['to_pos'])
            
            # Get player positions at end frame
            end_frame = candidate['end_frame']
            player_positions_at_end = self.frame_snapshots.get(end_frame, {}).get('player_positions', {})
            
            # #region agent log - SHOT CHECK
            try:
                with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'hypothesisId': 'D',
                        'location': 'simple_pass_detector.py:776',
                        'message': 'shot_check',
                        'data': {
                            'frame': int(candidate['start_frame']),
                            'from_player': int(candidate['from_player']),
                            'to_player': int(candidate['to_player']),
                            'ball_end_pos': ball_end_pos.tolist(),
                            'goal_positions_set': self.shot_detector.left_goal_center is not None and self.shot_detector.right_goal_center is not None,
                            'left_goal': self.shot_detector.left_goal_center.tolist() if self.shot_detector.left_goal_center is not None else None,
                            'right_goal': self.shot_detector.right_goal_center.tolist() if self.shot_detector.right_goal_center is not None else None
                        },
                        'timestamp': int(time.time() * 1000),
                        'sessionId': 'debug-session',
                        'runId': 'run1'
                    }) + '\n')
            except: pass
            # #endregion
            
            # Check if this is a shot
            is_shot, shot_confidence, goal_center = self.shot_detector.is_shot(
                candidate, ball_trajectory, player_positions_at_end, ball_end_pos
            )
            
            # #region agent log - SHOT CHECK RESULT
            try:
                with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'hypothesisId': 'D',
                        'location': 'simple_pass_detector.py:800',
                        'message': 'shot_check_result',
                        'data': {
                            'frame': int(candidate['start_frame']),
                            'is_shot': bool(is_shot),
                            'shot_confidence': float(shot_confidence),
                            'goal_center': goal_center.tolist() if goal_center is not None else None
                        },
                        'timestamp': int(time.time() * 1000),
                        'sessionId': 'debug-session',
                        'runId': 'run1'
                    }) + '\n')
            except: pass
            # #endregion
            
            if is_shot:
                # Check shot cooldown to prevent duplicate shots from same player
                # Use frame-based cooldown: prevent shots from same start_frame (same possession loss)
                # This prevents duplicates from the same shot attempt, but allows different shots
                shooter_id = candidate['from_player']
                shot_start_frame = candidate['start_frame']
                shot_cooldown_key = f"{shooter_id}_{shot_start_frame}"  # Unique per player+possession
                
                # Check if we already detected a shot from this exact possession (same start_frame)
                if shot_cooldown_key in self.shot_cooldowns:
                    # Already detected a shot from this possession - skip duplicate
                    if len(self.pass_candidates) <= 10:
                        print(f"   [SHOT REJECTED - Duplicate] Candidate #{i+1}: frame {candidate['start_frame']} (already detected from this possession)")
                    continue
                
                # Mark this possession as having a detected shot
                self.shot_cooldowns[shot_cooldown_key] = candidate['end_frame']
                
                # This is a shot, not a pass - create shot event
                shot_event = self.shot_detector.create_shot_event(
                    candidate, ball_trajectory, goal_center, shot_confidence
                )
                self.detected_shots.append(shot_event)
                self.shot_detector.stats['shots_detected'] += 1
                if shot_event.shot_type == ShotType.SHOT_ON_TARGET:
                    self.shot_detector.stats['shots_on_target'] += 1
                shots_detected += 1
                if len(self.pass_candidates) <= 10:
                    print(f"   [SHOT DETECTED] Candidate #{i+1}: {shot_event.shot_type.value} at frame {shot_event.start_frame}")
                continue  # Skip pass validation for shots
            
            # Layer 2: Trajectory validation (MORE LENIENT - allow if no trajectory data)
            if not self._validate_trajectory_direction(candidate):
                rejected_trajectory += 1
                if len(self.pass_candidates) <= 10:
                    print(f"   [REJECTED - Trajectory] Candidate #{i+1}: frame {candidate['start_frame']} -> {candidate['end_frame']}")
                continue
            
            # Layer 3: Temporal consistency
            if not self._validate_temporal(candidate):
                rejected_temporal += 1
                if len(self.pass_candidates) <= 10:
                    print(f"   [REJECTED - Temporal] Candidate #{i+1}: frame {candidate['start_frame']} -> {candidate['end_frame']}")
                continue
            
            # All validations passed - create pass event
            pass_event = self._create_pass_from_candidate(candidate)
            if pass_event is not None:
                self.confirmed_passes.append(pass_event)
                self.metrics['passes_detected'] += 1
                if len(self.pass_candidates) <= 10:
                    print(f"   [PASS CONFIRMED] Candidate #{i+1}: Player {candidate['from_player']} -> Player {candidate['to_player']} at frame {candidate['start_frame']}")
            else:
                rejected_cooldown += 1
                if len(self.pass_candidates) <= 10:
                    print(f"   [REJECTED - Cooldown] Candidate #{i+1}: frame {candidate['start_frame']} -> {candidate['end_frame']}")
        
        print(f"   ✓ Confirmed Passes: {len(self.confirmed_passes)}")
        print(f"   🎯 Shots Detected: {shots_detected}")
        print(f"   ✗ Rejected - Physical: {rejected_physical}")
        print(f"   ✗ Rejected - Trajectory: {rejected_trajectory}")
        print(f"   ✗ Rejected - Temporal: {rejected_temporal}")
        print(f"   ✗ Rejected - Cooldown: {rejected_cooldown}")
    
    def _validate_physical(self, candidate: Dict) -> bool:
        """Layer 1: Physical reality check."""
        duration_frames = candidate['end_frame'] - candidate['start_frame']
        duration_seconds = duration_frames / self.config['fps']
        distance = np.linalg.norm(candidate['to_pos'] - candidate['from_pos'])
        speed = distance / duration_seconds if duration_seconds > 0 else float('inf')
        
        # Check minimum duration
        if duration_seconds < 0.05:  # 1-2 frames at 30fps
            self.metrics['passes_rejected_duration'] += 1
            return False
        
        # Check distance bounds
        if distance < self.config['min_pass_distance']:
            self.metrics['passes_rejected_distance'] += 1
            return False
        
        if distance > self.config['max_pass_distance']:
            self.metrics['passes_rejected_distance'] += 1
            return False
        
        # Check speed (impossible speeds)
        if speed > 2000:  # pixels per second
            self.metrics['passes_rejected_distance'] += 1
            return False
        
        return True
    
    def _validate_trajectory_direction(self, candidate: Dict) -> bool:
        """Layer 2: Trajectory validation - check if ball moved from passer to receiver."""
        ball_trajectory = candidate['ball_trajectory']
        
        # VERY LENIENT: If no trajectory data, accept (ball might be occluded)
        if len(ball_trajectory) < 2:
            return True
        
        # Calculate expected vs actual direction
        expected_direction = candidate['to_pos'] - candidate['from_pos']
        actual_direction = ball_trajectory[-1] - ball_trajectory[0]
        
        # Normalize vectors
        expected_norm = np.linalg.norm(expected_direction)
        actual_norm = np.linalg.norm(actual_direction)
        
        # If not enough movement, accept (ball might be close to players)
        if expected_norm < 1e-6 or actual_norm < 1e-6:
            return True
        
        expected_unit = expected_direction / expected_norm
        actual_unit = actual_direction / actual_norm
        
        # Cosine similarity (dot product of unit vectors)
        similarity = np.dot(expected_unit, actual_unit)
        
        # MORE LENIENT: Only reject if ball went COMPLETELY wrong way (opposite direction)
        # Accept if similarity > 0.0 (ball moved in roughly same direction, even if not perfect)
        # This allows up to 90 degree deviation
        if similarity < 0.0:  # Ball went backwards/opposite direction
            self.metrics['passes_rejected_trajectory'] += 1
            return False
        
        return True
    
    def _validate_temporal(self, candidate: Dict) -> bool:
        """Layer 3: Temporal consistency - check if possession was held long enough."""
        if candidate['possession_frames'] < self.config['min_possession_frames']:
            self.metrics['passes_rejected_duration'] += 1
            return False
        return True
    
    def _extract_ball_trajectory(self, start_frame: int, end_frame: int) -> List[np.ndarray]:
        """Extract ball trajectory between two frames."""
        trajectory = []
        for frame_idx, ball_pos in self.ball_trajectory_buffer:
            if start_frame <= frame_idx <= end_frame:
                trajectory.append(ball_pos.copy())
        return trajectory
    
    def _create_pass_from_candidate(self, candidate: Dict) -> Optional[PassEvent]:
        """Create PassEvent from validated candidate."""
        # #region agent log - COOLDOWN CHECK
        try:
            with open('/home/essashah/SWE/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'hypothesisId': 'E',
                    'location': 'simple_pass_detector.py:697',
                    'message': 'cooldown_check',
                    'data': {
                        'from_player': int(candidate['from_player']),
                        'end_frame': int(candidate['end_frame']),
                        'in_cooldown': candidate['from_player'] in self.player_cooldowns,
                        'cooldown_until': int(self.player_cooldowns.get(candidate['from_player'], -1)),
                        'will_reject': candidate['from_player'] in self.player_cooldowns and candidate['end_frame'] < self.player_cooldowns[candidate['from_player']]
                    },
                    'timestamp': int(time.time() * 1000),
                    'sessionId': 'debug-session',
                    'runId': 'run1'
                }) + '\n')
        except: pass
        # #endregion
        
        # Check cooldown
        if candidate['from_player'] in self.player_cooldowns:
            if candidate['end_frame'] < self.player_cooldowns[candidate['from_player']]:
                self.metrics['passes_rejected_cooldown'] += 1
                return None
        
        # Create pass
        self.pass_counter += 1
        duration = (candidate['end_frame'] - candidate['start_frame']) / self.config['fps']
        distance = np.linalg.norm(candidate['to_pos'] - candidate['from_pos'])
        
        # Set cooldown
        self.player_cooldowns[candidate['from_player']] = candidate['end_frame'] + self.config['cooldown_frames']
        
        # Determine outcome
        if candidate['from_team'] == candidate['to_team']:
            outcome = 'success'
        else:
            outcome = 'interception'
        
        return PassEvent(
            event_id=f"pass_{self.pass_counter}",
            from_player_id=candidate['from_player'],
            to_player_id=candidate['to_player'],
            team_id=candidate['from_team'],
            receiver_team_id=candidate['to_team'],
            start_frame=candidate['start_frame'],
            end_frame=candidate['end_frame'],
            start_position=candidate['from_pos'].tolist(),
            end_position=candidate['to_pos'].tolist(),
            confidence=self._calculate_confidence(candidate),
            distance_meters=float(distance),
            duration_seconds=duration,
            lifecycle_stage=PassLifecycleStage.CONFIRMED,
            created_timestamp=time.time()
        )
    
    def _calculate_confidence(self, candidate: Dict) -> float:
        """Calculate confidence score for pass event."""
        confidence = 1.0
        distance = np.linalg.norm(candidate['to_pos'] - candidate['from_pos'])
        duration = (candidate['end_frame'] - candidate['start_frame']) / self.config['fps']
        speed = distance / duration if duration > 0 else 0
        
        # Reduce confidence for very short passes
        if distance < 100:
            confidence *= 0.8
        
        # Reduce confidence for very fast passes
        if speed > 1000:
            confidence *= 0.7
        
        # Increase confidence for clear trajectory
        if len(candidate['ball_trajectory']) >= 3:
            expected = candidate['to_pos'] - candidate['from_pos']
            actual = candidate['ball_trajectory'][-1] - candidate['ball_trajectory'][0]
            if np.linalg.norm(expected) > 1e-6 and np.linalg.norm(actual) > 1e-6:
                expected_unit = expected / np.linalg.norm(expected)
                actual_unit = actual / np.linalg.norm(actual)
                similarity = np.dot(expected_unit, actual_unit)
                if similarity > 0.9:
                    confidence *= 1.2
        
        return min(confidence, 1.0)
    
    def get_detected_shots(self) -> List[ShotEvent]:
        """Get all detected shots."""
        # Validate candidates if not already validated (this also detects shots)
        if len(self.pass_candidates) > 0 and len(self.confirmed_passes) == 0 and len(self.detected_shots) == 0:
            self.validate_all_candidates()
        return self.detected_shots.copy()
    
    def get_confirmed_passes(self) -> List[PassEvent]:
        """Get all confirmed passes. Validates candidates if not already done."""
        # Validate candidates if not already validated
        # Only validate once - if shots were already detected, don't validate again
        if len(self.pass_candidates) > 0 and len(self.confirmed_passes) == 0 and len(self.detected_shots) == 0:
            self.validate_all_candidates()
        
        # Print ball detection stats if available
        if hasattr(self, '_ball_detection_stats'):
            stats = self._ball_detection_stats
            total = stats['detected'] + stats['predicted'] + stats['lost']
            if total > 0:
                print(f"   Ball detection: {stats['detected']} detected, {stats['predicted']} predicted, {stats['lost']} lost")
        return self.confirmed_passes.copy()
    
    def get_ball_tracker(self) -> StrictBallTracker:
        """Get ball tracker instance."""
        return self.ball_tracker
    
    def _calculate_dynamic_radius(self, ball_speed_px_per_frame: float) -> float:
        """
        Calculate dynamic possession radius based on ball speed.
        
        Ball moving fast = larger radius needed (players can't control perfectly)
        Ball stationary/slow = smaller radius (tighter control)
        """
        if not self.config.get('dynamic_radius_enabled', True):
            return self.config['possession_radius']
        
        base_radius = self.config.get('base_radius', 150.0)
        speed_factor_mult = self.config.get('speed_factor', 2.0)
        min_radius = self.config.get('min_radius', 120.0)
        max_radius = self.config.get('max_radius', 250.0)
        
        # Ball speed in pixels/frame
        speed_factor = 1.0 + (ball_speed_px_per_frame / 100.0) * speed_factor_mult
        
        dynamic_radius = base_radius * speed_factor
        
        # Clamp to reasonable range
        return min(max(dynamic_radius, min_radius), max_radius)
    
    def _validate_pass_trajectory(self, passer_pos: np.ndarray, receiver_pos: np.ndarray,
                                  pass_start_frame: int, pass_end_frame: int) -> bool:
        """
        Check if ball actually traveled from passer to receiver.
        
        Uses lenient validation with exemptions for short/quick passes.
        
        Returns:
            True if trajectory is valid, False otherwise
        """
        if not self.config.get('validate_trajectory', True):
            return True
        
        # Calculate pass distance and duration
        pass_distance = np.linalg.norm(np.array(receiver_pos) - np.array(passer_pos))
        pass_duration_frames = pass_end_frame - pass_start_frame
        
        # EXEMPTION 1: Short pass exemption - skip validation for very short passes
        short_pass_threshold = self.config.get('short_pass_exemption', 200.0)
        if pass_distance < short_pass_threshold:
            # #region agent log - SHORT PASS EXEMPTION
            try:
                with open(self.debug_log_path, 'a') as f:
                    f.write(json.dumps({
                        'hypothesisId': 'F',
                        'location': 'simple_pass_detector:trajectory_validation',
                        'message': 'short_pass_exemption',
                        'data': {
                            'frame': int(pass_end_frame),
                            'pass_distance': float(pass_distance),
                            'threshold': float(short_pass_threshold),
                            'reason': 'short_pass_exempted'
                        },
                        'timestamp': int(time.time() * 1000),
                        'sessionId': 'pass-detection-improvement'
                    }) + '\n')
            except:
                pass
            # #endregion
            return True  # Accept all short passes
        
        # Get ball positions during pass duration
        ball_positions = []
        for frame_idx, ball_pos in self.ball_trajectory_buffer:
            if pass_start_frame <= frame_idx <= pass_end_frame:
                ball_positions.append(ball_pos)
        
        if len(ball_positions) < 2:
            return True  # Not enough data, default to accept
        
        # EXEMPTION 2: Quick pass exemption - use very lenient validation
        quick_pass_threshold = self.config.get('quick_pass_exemption_frames', 10)
        is_quick_pass = pass_duration_frames < quick_pass_threshold
        
        if is_quick_pass:
            # For quick passes, just check ball moved generally toward receiver
            if len(ball_positions) >= 2:
                start_pos = np.array(ball_positions[0])
                end_pos = np.array(ball_positions[-1])
                expected_vector = np.array(receiver_pos) - np.array(passer_pos)
                actual_vector = end_pos - start_pos
                
                # Very lenient: just check it's not completely opposite (cos > 0.1)
                if np.linalg.norm(actual_vector) > 1e-6:
                    expected_norm = expected_vector / (np.linalg.norm(expected_vector) + 1e-6)
                    actual_norm = actual_vector / (np.linalg.norm(actual_vector) + 1e-6)
                    cos_similarity = np.dot(expected_norm, actual_norm)
                    
                    if cos_similarity > 0.1:  # Very lenient for quick passes
                        # #region agent log - QUICK PASS EXEMPTION
                        try:
                            with open(self.debug_log_path, 'a') as f:
                                f.write(json.dumps({
                                    'hypothesisId': 'G',
                                    'location': 'simple_pass_detector:trajectory_validation',
                                    'message': 'quick_pass_exemption',
                                    'data': {
                                        'frame': int(pass_end_frame),
                                        'duration_frames': int(pass_duration_frames),
                                        'cos_similarity': float(cos_similarity),
                                        'threshold': 0.1,
                                        'reason': 'quick_pass_lenient_validation'
                                    },
                                    'timestamp': int(time.time() * 1000),
                                    'sessionId': 'pass-detection-improvement'
                                }) + '\n')
                        except:
                            pass
                        # #endregion
                        return True
            # If we can't validate, accept quick passes by default
            return True
        
        # Standard validation for longer passes
        if len(ball_positions) < 3:
            return True  # Not enough data, default to accept
        
        # Calculate overall direction
        start_pos = np.array(ball_positions[0])
        end_pos = np.array(ball_positions[-1])
        expected_vector = np.array(receiver_pos) - np.array(passer_pos)
        actual_vector = end_pos - start_pos
        
        # Normalize vectors
        expected_norm = expected_vector / (np.linalg.norm(expected_vector) + 1e-6)
        actual_norm = actual_vector / (np.linalg.norm(actual_vector) + 1e-6)
        
        # Cosine similarity (1 = same direction, -1 = opposite)
        cos_similarity = np.dot(expected_norm, actual_norm)
        
        # Use lenient threshold (0.3 instead of 0.7)
        similarity_threshold = self.config.get('trajectory_similarity_threshold', 0.3)
        if cos_similarity < similarity_threshold:
            # #region agent log - TRAJECTORY REJECTED COSINE
            try:
                with open(self.debug_log_path, 'a') as f:
                    f.write(json.dumps({
                        'hypothesisId': 'C',
                        'location': 'simple_pass_detector:trajectory_validation',
                        'message': 'pass_rejected_trajectory_cosine',
                        'data': {
                            'frame': int(pass_end_frame),
                            'cos_similarity': float(cos_similarity),
                            'threshold': float(similarity_threshold),
                            'pass_distance': float(pass_distance),
                            'duration_frames': int(pass_duration_frames),
                            'reason': 'cosine_similarity_too_low'
                        },
                        'timestamp': int(time.time() * 1000),
                        'sessionId': 'pass-detection-improvement'
                    }) + '\n')
            except:
                pass
            # #endregion
            return False  # Ball didn't move toward receiver
        
        # Check for reasonable directness (not zigzagging)
        total_path = 0
        for i in range(1, len(ball_positions)):
            segment = np.array(ball_positions[i]) - np.array(ball_positions[i-1])
            total_path += np.linalg.norm(segment)
        
        direct_distance = np.linalg.norm(end_pos - start_pos)
        if direct_distance < 1e-6:
            return True  # Ball didn't move much, accept
        
        path_ratio = total_path / direct_distance
        max_ratio = self.config.get('max_path_ratio', 2.0)  # More lenient (2.0 instead of 1.5)
        if path_ratio > max_ratio:
            # #region agent log - TRAJECTORY REJECTED DIRECTNESS
            try:
                with open(self.debug_log_path, 'a') as f:
                    f.write(json.dumps({
                        'hypothesisId': 'C',
                        'location': 'simple_pass_detector:trajectory_validation',
                        'message': 'pass_rejected_trajectory_directness',
                        'data': {
                            'frame': int(pass_end_frame),
                            'path_ratio': float(path_ratio),
                            'threshold': float(max_ratio),
                            'pass_distance': float(pass_distance),
                            'duration_frames': int(pass_duration_frames),
                            'reason': 'path_too_indirect'
                        },
                        'timestamp': int(time.time() * 1000),
                        'sessionId': 'pass-detection-improvement'
                    }) + '\n')
            except:
                pass
            # #endregion
            return False  # Too much zigzag
        
        # #region agent log - TRAJECTORY ACCEPTED
        try:
            with open(self.debug_log_path, 'a') as f:
                f.write(json.dumps({
                    'hypothesisId': 'H',
                    'location': 'simple_pass_detector:trajectory_validation',
                    'message': 'pass_accepted_trajectory',
                    'data': {
                        'frame': int(pass_end_frame),
                        'cos_similarity': float(cos_similarity),
                        'path_ratio': float(path_ratio),
                        'pass_distance': float(pass_distance),
                        'duration_frames': int(pass_duration_frames)
                    },
                    'timestamp': int(time.time() * 1000),
                    'sessionId': 'pass-detection-improvement'
                }) + '\n')
        except:
            pass
        # #endregion
        
        return True
    
    def get_metrics(self) -> Dict:
        """Get detection metrics."""
        return {
            **self.metrics,
            'ball_stats': self.ball_tracker.get_stats()
        }

