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

from .ball_tracker import StrictBallTracker, BallState, BallObservation
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
            # Possession thresholds - IN PIXELS (IMPROVED)
            'possession_radius': 150.0,    # Increased from 100 to 150 (more realistic)
            'min_pass_distance': 80.0,     # Minimum pass distance in pixels (meaningful distance)
            'max_pass_distance': 800.0,    # Maximum pass distance in pixels
            
            # Timing - MINIMUM for one-touch pass detection
            'min_possession_frames': 1,    # Allow one-touch passes (was 5, reduced to 1)
            'cooldown_frames': 60,         # 2 seconds between passes from same player
            'min_pass_duration': 1,        # Allow one-touch passes (1 frame minimum, was 5)
            'max_pass_duration_frames': 120,  # Pass can't take more than 4 seconds (120 frames at 30fps)
            
            # Frame rate
            'fps': 30.0,
            
            # Use pixel coordinates (since homography often fails)
            'use_pixel_coords': True,
            
            # Trajectory validation (LENIENT for realistic soccer passes)
            'validate_trajectory': True,   # Enable trajectory validation
            'trajectory_similarity_threshold': 0.3,  # Lowered from 0.7 to 0.3 (allow ~72° deviation)
            'max_path_ratio': 2.0,  # Increased from 1.5 to 2.0 (allow 100% extra path length)
            'short_pass_exemption': 200.0,  # Skip validation for passes < 200px
            'quick_pass_exemption_frames': 10,  # Use lenient validation for passes < 10 frames
            
            # Interception detection
            'detect_interceptions': True,  # Enable detection of passes between different teams
            
            # Speed thresholds (for fast passes)
            'max_speed_pixels_per_frame': 50,  # Increased from 25 to 50 for hard/fast passes
            
            # Dynamic radius
            'dynamic_radius_enabled': True,  # Enable dynamic possession radius
            'base_radius': 150.0,
            'speed_factor': 2.0,
            'min_radius': 120.0,
            'max_radius': 250.0,
        }
        
        if config:
            self.config.update(config)
        
        # Ball tracker - using robust tracker that validates temporal consistency
        self.ball_tracker = StrictBallTracker(config)
        
        # Possession tracking
        self.current_possession: Optional[BallPossession] = None
        self.possession_frames: int = 0
        self.last_possession_frame: int = -1
        
        # Pass tracking
        self.confirmed_passes: List[PassEvent] = []
        self.pass_counter: int = 0
        
        # Cooldowns
        self.player_cooldowns: Dict[int, int] = {}  # player_id -> end_frame
        
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
        
        # No ball position = no detection
        if ball_obs.position is None or ball_obs.state == BallState.LOST:
            return []
        
        ball_pos = ball_obs.position
        
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
        
        if closest_player is not None and closest_distance <= current_radius:
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
                self.metrics['total_possessions'] += 1
                
            elif self.current_possession.player_id != closest_player:
                # Possession change - potential pass
                # Track near-miss duration (old threshold vs new)
                if 5 <= self.possession_frames < 10:
                    self.metrics['near_miss_duration'] += 1
                    # #region agent log - NEAR MISS DURATION
                    try:
                        with open(self.debug_log_path, 'a') as f:
                            f.write(json.dumps({
                                'hypothesisId': 'B',
                                'location': 'simple_pass_detector:possession_change',
                                'message': 'near_miss_duration',
                                'data': {
                                    'frame': int(frame),
                                    'from_player': int(self.current_possession.player_id),
                                    'to_player': int(closest_player),
                                    'possession_frames': self.possession_frames,
                                    'old_threshold': 10,
                                    'new_threshold': 5,
                                    'would_detect': True
                                },
                                'timestamp': int(time.time() * 1000),
                                'sessionId': 'pass-detection-improvement'
                            }) + '\n')
                    except:
                        pass
                    # #endregion
                
                if self.possession_frames >= self.config['min_possession_frames']:
                    # Check if it's a pass (same team, not same player)
                    same_team = (self.current_possession.team_id == team_id)
                    
                    # Allow both same-team passes AND interceptions
                    detect_interceptions = self.config.get('detect_interceptions', True)
                    is_valid_pass_type = same_team or detect_interceptions
                    
                    if is_valid_pass_type:
                        # Validate trajectory if enabled
                        trajectory_valid = True
                        if self.config.get('validate_trajectory', True):
                            trajectory_valid = self._validate_pass_trajectory(
                                self.current_possession.position,
                                player_positions[closest_player].copy(),
                                self.current_possession.start_frame,
                                frame
                            )
                            
                            if not trajectory_valid:
                                self.metrics['passes_rejected_trajectory'] += 1
                                # #region agent log - TRAJECTORY REJECTED
                                try:
                                    with open(self.debug_log_path, 'a') as f:
                                        f.write(json.dumps({
                                            'hypothesisId': 'C',
                                            'location': 'simple_pass_detector:trajectory_validation',
                                            'message': 'pass_rejected_trajectory',
                                            'data': {
                                                'frame': int(frame),
                                                'from_player': int(self.current_possession.player_id),
                                                'to_player': int(closest_player),
                                                'start_frame': int(self.current_possession.start_frame),
                                                'end_frame': int(frame)
                                            },
                                            'timestamp': int(time.time() * 1000),
                                            'sessionId': 'pass-detection-improvement'
                                        }) + '\n')
                                except:
                                    pass
                                # #endregion
                        
                        # Use pass even if trajectory invalid but distance is reasonable (fallback)
                        if trajectory_valid or not self.config.get('validate_trajectory', True):
                            pass_event = self._create_pass(
                                from_player=self.current_possession.player_id,
                                to_player=closest_player,
                                team_id=self.current_possession.team_id,
                                start_frame=self.current_possession.start_frame,
                                end_frame=frame,
                                from_pos=self.current_possession.position,
                                to_pos=player_positions[closest_player].copy(),
                                receiver_team_id=team_id  # Store actual receiver team
                            )
                            
                            if pass_event is not None:
                                self.confirmed_passes.append(pass_event)
                                new_passes.append(pass_event)
                                self.metrics['passes_detected'] += 1
                                # #region agent log - PASS DETECTED
                                try:
                                    # Get frame snapshots for start and end frames
                                    start_snapshot = self.frame_snapshots.get(self.current_possession.start_frame, {})
                                    end_snapshot = self.frame_snapshots.get(frame, {})
                                    
                                    with open(self.debug_log_path, 'a') as f:
                                        f.write(json.dumps({
                                            'hypothesisId': 'D',
                                            'location': 'simple_pass_detector:pass_detected',
                                            'message': 'pass_detected',
                                            'data': {
                                                'frame': int(frame),
                                                'from_player_tracker_id': int(pass_event.from_player_id),
                                                'to_player_tracker_id': int(pass_event.to_player_id),
                                                'distance': float(pass_event.distance_meters),
                                                'duration_frames': int(pass_event.end_frame - pass_event.start_frame),
                                                'duration_seconds': float(pass_event.duration_seconds),
                                                'trajectory_valid': bool(trajectory_valid),
                                                'start_frame_snapshot': {
                                                    'all_tracker_ids': start_snapshot.get('all_tracker_ids', []),
                                                    'passer_team': start_snapshot.get('player_teams', {}).get(pass_event.from_player_id, None),
                                                    'all_teams_in_frame': {str(k): int(v) for k, v in start_snapshot.get('player_teams', {}).items()}
                                                },
                                                'end_frame_snapshot': {
                                                    'all_tracker_ids': end_snapshot.get('all_tracker_ids', []),
                                                    'receiver_team': end_snapshot.get('player_teams', {}).get(pass_event.to_player_id, None),
                                                    'all_teams_in_frame': {str(k): int(v) for k, v in end_snapshot.get('player_teams', {}).items()}
                                                }
                                            },
                                            'timestamp': int(time.time() * 1000),
                                            'sessionId': 'pass-detection-improvement'
                                        }) + '\n')
                                except Exception as e:
                                    # Log error but don't fail
                                    try:
                                        with open(self.debug_log_path, 'a') as f:
                                            f.write(json.dumps({
                                                'location': 'simple_pass_detector:pass_detected',
                                                'message': 'pass_detected_log_error',
                                                'data': {'error': str(e)},
                                                'timestamp': int(time.time() * 1000),
                                                'sessionId': 'pass-detection-improvement'
                                            }) + '\n')
                                    except:
                                        pass
                                # #endregion
                
                # Start new possession
                self.current_possession = BallPossession(
                    player_id=closest_player,
                    team_id=team_id,
                    start_frame=frame,
                    position=player_positions[closest_player].copy()
                )
                self.possession_frames = 1
                self.metrics['total_possessions'] += 1
            else:
                # Same player still has ball
                self.possession_frames += 1
        else:
            # Ball not in possession of anyone
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
    
    def get_confirmed_passes(self) -> List[PassEvent]:
        """Get all confirmed passes."""
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

