"""
Physics-Based Ball Tracker (rule-based, physics-aware)

Deterministic (no new ML). Improves stability via:
- Candidate-based selection (YOLO detections + physics prediction)
- Explicit semantic ball state machine:
  CONTROLLED | IN_FLIGHT | LOOSE | OCCLUDED | OUT_OF_PLAY
- State-aware physics constraints (smoothing + acceleration limits + reacquisition rules)
- Foot proximity logic (approximate feet from bottom 15–20% of player bboxes)
- Pitch-aware gating (reject outside pitch unless transitioning OUT_OF_PLAY)
- Track confidence as a first-class signal for downstream detectors

Public interface remains backward compatible:
- `update(frame, ball_detections)` still works with `ball_detections: np.ndarray [N,2]`.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Iterable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class BallState(Enum):
    """
    Explicit ball *game* state machine.

    Inferred every frame using deterministic rules based on:
    - speed / acceleration
    - proximity to player feet (from bboxes)
    - detection presence/absence
    - pitch boundaries (if pitch polygon provided)
    """
    CONTROLLED = "controlled"
    IN_FLIGHT = "in_flight"
    LOOSE = "loose"
    OCCLUDED = "occluded"
    OUT_OF_PLAY = "out_of_play"


class BallObservationSource(Enum):
    """How the position for a frame was produced."""
    DETECTED = "detected"          # Selected from a YOLO detection candidate
    PREDICTED = "predicted"        # Selected from physics prediction candidate
    INTERPOLATED = "interpolated"  # Gap-filled fallback


@dataclass
class BallObservation:
    """Single ball observation."""
    frame: int
    position: np.ndarray  # [x, y] - ALWAYS available (never None)
    state: BallState  # semantic/game state
    source: BallObservationSource
    confidence: float = 1.0
    velocity: Optional[np.ndarray] = None  # [vx, vy] in pixels/frame
    acceleration: Optional[np.ndarray] = None  # [ax, ay] in pixels/frame²
    is_predicted: bool = False  # True if position is predicted/interpolated
    debug: Dict = field(default_factory=dict)  # scoring + gating info (deterministic)


@dataclass
class BallPhysicsState:
    """Continuous ball state (no FSM)."""
    position: np.ndarray  # [x, y] in pixels
    velocity: np.ndarray  # [vx, vy] in pixels/frame
    acceleration: np.ndarray  # [ax, ay] in pixels/frame²
    last_detected_frame: Optional[int] = None
    last_detected_position: Optional[np.ndarray] = None
    frames_since_detection: int = 0
    confidence: float = 1.0  # Decreases with prediction depth
    history: deque = field(default_factory=lambda: deque(maxlen=60))  # 2 seconds at 30fps
    game_state: BallState = BallState.LOOSE


class PhysicsBallTracker:
    """
    Physics-based predictive ball tracker.
    
    Key features:
    1. Single-ball identity (no duplication)
    2. Continuous prediction using velocity/acceleration
    3. Friction/deceleration for realistic motion
    4. Adaptive detection gating (spatial + velocity matching)
    5. Interpolation for missing frames
    6. No FSM - all logic is continuous
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the physics-based ball tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            # Frame rate
            'fps': 30.0,
            
            # Physics parameters
            'friction_coefficient': 0.95,  # Per-frame velocity decay (0.95 = 5% reduction)
            'gravity': 0.0,  # Vertical acceleration (pixels/frame²) - set to 0 for 2D tracking
            'max_velocity': 2000.0,  # pixels/frame (very high for fast passes)
            'max_acceleration': 500.0,  # pixels/frame²

            # Hard constraints (used to reject impossible candidates)
            'hard_max_acceleration': 800.0,   # absolute acceleration cap (px/frame^2)
            'hard_max_speed': 2500.0,         # absolute speed cap (px/frame)
            'hard_max_turn_deg': 120.0,       # max direction change without foot contact
            'hard_turn_requires_foot_px': 120.0,
            
            # Detection gating (adaptive)
            'base_gate_radius': 150.0,  # Base spatial gate radius (pixels)
            'gate_velocity_factor': 2.0,  # Gate scales with velocity
            'gate_velocity_threshold': 10.0,  # pixels/frame - above this, gate expands
            'max_gate_radius': 500.0,  # Maximum gate radius
            'velocity_match_threshold': 0.5,  # Cosine similarity for velocity matching (0-1)
            
            # Prediction limits
            'max_prediction_frames': 60,  # 2 seconds at 30fps - much longer than FSM
            'confidence_decay_rate': 0.02,  # Per-frame confidence decay when predicting
            
            # Interpolation
            'interpolate_gaps': True,
            'max_interpolation_gap': 30,  # Max frames to interpolate (1 second)
            
            # Smoothing
            'smoothing_enabled': True,
            'smoothing_alpha': 0.7,  # Exponential smoothing factor (0-1, higher = more smoothing)

            # State-aware smoothing (override alpha)
            'smoothing_alpha_controlled': 0.85,
            'smoothing_alpha_loose': 0.80,
            'smoothing_alpha_in_flight': 0.45,
            'smoothing_alpha_occluded': 0.90,

            # Foot proximity
            'foot_box_bottom_frac': 0.20,  # bottom 20% of bbox
            'foot_proximity_px': 140.0,

            # Candidate scoring weights
            'w_detector_conf': 0.35,
            'w_dist_to_pred': 0.25,
            'w_velocity_consistency': 0.15,
            'w_foot_proximity': 0.15,
            'w_pitch_validity': 0.10,

            # Pitch gating / out-of-play
            'out_of_play_hysteresis_frames': 10,
            'reentry_min_confidence': 0.35,
            
            # Initialization
            'min_detections_to_init': 1,  # Minimum detections to start tracking
        }
        
        if config:
            self.config.update(config)
        
        # Ball state (single ball, always exists once initialized)
        self.ball_state: Optional[BallPhysicsState] = None
        
        # Detection history
        self.detection_history: List[BallObservation] = []
        self.frame_observations: Dict[int, BallObservation] = {}
        
        # Recent detections for velocity estimation
        self.recent_detections: deque = deque(maxlen=5)
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'detected_frames': 0,
            'predicted_frames': 0,
            'interpolated_frames': 0,
            'rejected_detections': 0,
            'gated_detections': 0,
            'out_of_play_frames': 0,
            'occluded_frames': 0,
        }

        # Hysteresis counter to prevent OUT_OF_PLAY flapping
        self._oop_counter: int = 0

    # ---------------------------
    # Geometry + foot helpers
    # ---------------------------
    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        """
        Ray casting point-in-polygon test (deterministic).

        Args:
            point: (2,) array [x, y]
            polygon: (M, 2) array of polygon vertices
        """
        x, y = float(point[0]), float(point[1])
        inside = False
        n = int(len(polygon))
        if n < 3:
            return True
        x0, y0 = float(polygon[-1][0]), float(polygon[-1][1])
        for i in range(n):
            x1, y1 = float(polygon[i][0]), float(polygon[i][1])
            intersects = ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-9) + x0)
            if intersects:
                inside = not inside
            x0, y0 = x1, y1
        return inside

    def _pitch_validity(self, pos: np.ndarray, pitch_polygon: Optional[np.ndarray]) -> float:
        """1.0 if inside pitch polygon (or polygon missing), else 0.0."""
        if pitch_polygon is None:
            return 1.0
        try:
            return 1.0 if self._point_in_polygon(pos, pitch_polygon) else 0.0
        except Exception:
            # Fail-open: don't crash pipeline if polygon malformed
            return 1.0

    def _extract_feet_positions(self, player_bboxes_xyxy: Optional[np.ndarray]) -> np.ndarray:
        """
        Approximate player foot positions from bounding boxes.

        Uses bottom `foot_box_bottom_frac` of bbox; returns two foot points per player:
        left ~25% width, right ~75% width at the bottom edge.
        """
        if player_bboxes_xyxy is None or len(player_bboxes_xyxy) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        frac = float(self.config.get('foot_box_bottom_frac', 0.20))
        feet: List[List[float]] = []
        for (x1, y1, x2, y2) in player_bboxes_xyxy:
            w = float(x2 - x1)
            h = float(y2 - y1)
            if w <= 1.0 or h <= 1.0:
                continue
            _ = float(y2 - frac * h)  # explicit: bottom band start (not used for point)
            feet.append([float(x1 + 0.25 * w), float(y2)])
            feet.append([float(x1 + 0.75 * w), float(y2)])
        if not feet:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(feet, dtype=np.float32)

    @staticmethod
    def _min_distance_to_points(pos: np.ndarray, points: np.ndarray) -> float:
        if points is None or len(points) == 0:
            return float("inf")
        diffs = points - pos.reshape(1, 2)
        d2 = np.sum(diffs * diffs, axis=1)
        return float(np.sqrt(np.min(d2)))

    @staticmethod
    def _safe_unit(v: np.ndarray) -> Optional[np.ndarray]:
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return None
        return v / n

    # ---------------------------
    # State inference + physics constraints
    # ---------------------------
    def _infer_game_state(
        self,
        *,
        selected_source: BallObservationSource,
        has_detection: bool,
        min_foot_dist: float,
        pitch_valid: float,
        speed: float,
        acc_mag: float,
    ) -> BallState:
        """
        Infer semantic ball state for this frame.

        Priority order:
        - OUT_OF_PLAY if pitch invalid (handled with hysteresis in `update`)
        - OCCLUDED if no detection and relying on prediction
        - CONTROLLED if near a foot and not moving fast
        - IN_FLIGHT if moving fast and not near a foot
        - else LOOSE
        """
        foot_thr = float(self.config.get('foot_proximity_px', 140.0))

        if pitch_valid < 0.5:
            return BallState.OUT_OF_PLAY

        if (not has_detection) and selected_source != BallObservationSource.DETECTED:
            return BallState.OCCLUDED

        if min_foot_dist <= foot_thr and speed <= 80.0:
            return BallState.CONTROLLED

        if speed >= 120.0 and min_foot_dist > foot_thr:
            return BallState.IN_FLIGHT

        if acc_mag >= 120.0 and min_foot_dist <= foot_thr:
            return BallState.CONTROLLED

        return BallState.LOOSE

    def _state_smoothing_alpha(self, state: BallState) -> float:
        if not self.config.get('smoothing_enabled', True):
            return 1.0
        if state == BallState.CONTROLLED:
            return float(self.config.get('smoothing_alpha_controlled', 0.85))
        if state == BallState.LOOSE:
            return float(self.config.get('smoothing_alpha_loose', 0.80))
        if state == BallState.IN_FLIGHT:
            return float(self.config.get('smoothing_alpha_in_flight', 0.45))
        if state == BallState.OCCLUDED:
            return float(self.config.get('smoothing_alpha_occluded', 0.90))
        return float(self.config.get('smoothing_alpha', 0.7))

    def _apply_state_smoothing(self, position: np.ndarray, state: BallState) -> np.ndarray:
        """Exponential smoothing with state-dependent alpha."""
        if self.ball_state is None:
            return position
        alpha = self._state_smoothing_alpha(state)
        return alpha * position + (1.0 - alpha) * self.ball_state.position

    def _hard_physics_reject(
        self,
        *,
        candidate_pos: np.ndarray,
        predicted_pos: np.ndarray,
        prev_vel: np.ndarray,
        min_foot_dist: float,
        frames_dt: float,
    ) -> Optional[str]:
        """
        Hard rejection constraints (deterministic).

        Returns:
            rejection reason string, or None if candidate is acceptable.
        """
        disp = candidate_pos - predicted_pos
        implied_vel = disp / max(1.0, frames_dt)
        speed = float(np.linalg.norm(implied_vel))
        if speed > float(self.config.get('hard_max_speed', 2500.0)):
            return f"hard_speed_{speed:.1f}"

        implied_acc = (implied_vel - prev_vel) / max(1.0, frames_dt)
        acc_mag = float(np.linalg.norm(implied_acc))
        if acc_mag > float(self.config.get('hard_max_acceleration', 800.0)):
            return f"hard_acc_{acc_mag:.1f}"

        prev_u = self._safe_unit(prev_vel)
        new_u = self._safe_unit(implied_vel)
        if prev_u is not None and new_u is not None:
            cosang = float(np.clip(np.dot(prev_u, new_u), -1.0, 1.0))
            ang = float(np.degrees(np.arccos(cosang)))
            if ang > float(self.config.get('hard_max_turn_deg', 120.0)):
                foot_req = float(self.config.get('hard_turn_requires_foot_px', 120.0))
                if min_foot_dist > foot_req:
                    return f"hard_turn_{ang:.1f}_no_foot"

        return None

    def _score_candidate(
        self,
        *,
        detector_conf: float,
        dist_to_pred: float,
        gate_radius: float,
        implied_vel: np.ndarray,
        prev_vel: np.ndarray,
        min_foot_dist: float,
        pitch_valid: float,
    ) -> Tuple[float, Dict]:
        """
        Weighted candidate scoring (higher is better).
        Produces a normalized score in [0,1] with debug components.
        """
        # Distance-to-predicted -> [0,1] (1 at 0 distance, ~0 at large distances)
        scale = max(1.0, float(gate_radius))
        s_dist = float(np.exp(-dist_to_pred / scale))

        # Velocity consistency -> [0,1] from cosine similarity
        prev_u = self._safe_unit(prev_vel)
        new_u = self._safe_unit(implied_vel)
        if prev_u is None or new_u is None:
            s_vel = 0.5
        else:
            s_vel = float((np.clip(np.dot(prev_u, new_u), -1.0, 1.0) + 1.0) * 0.5)

        # Foot proximity -> [0,1]
        foot_thr = float(self.config.get('foot_proximity_px', 140.0))
        if not np.isfinite(min_foot_dist):
            s_foot = 0.0
        else:
            s_foot = float(np.clip(1.0 - (min_foot_dist / max(1.0, foot_thr)), 0.0, 1.0))

        s_pitch = float(np.clip(pitch_valid, 0.0, 1.0))
        s_det = float(np.clip(detector_conf, 0.0, 1.0))

        w_det = float(self.config.get('w_detector_conf', 0.35))
        w_dist = float(self.config.get('w_dist_to_pred', 0.25))
        w_vel = float(self.config.get('w_velocity_consistency', 0.15))
        w_foot = float(self.config.get('w_foot_proximity', 0.15))
        w_pitch = float(self.config.get('w_pitch_validity', 0.10))

        score = w_det * s_det + w_dist * s_dist + w_vel * s_vel + w_foot * s_foot + w_pitch * s_pitch
        score = float(np.clip(score, 0.0, 1.0))

        debug = {
            "s_det": s_det,
            "s_dist": s_dist,
            "s_vel": s_vel,
            "s_foot": s_foot,
            "s_pitch": s_pitch,
            "dist_to_pred": float(dist_to_pred),
            "gate_radius": float(gate_radius),
        }
        return score, debug
    
    def _calculate_gate_radius(self, velocity: np.ndarray) -> float:
        """
        Calculate adaptive gate radius based on ball velocity.
        
        Args:
            velocity: Current ball velocity [vx, vy]
            
        Returns:
            Gate radius in pixels
        """
        speed = np.linalg.norm(velocity)
        
        if speed < self.config['gate_velocity_threshold']:
            return self.config['base_gate_radius']
        
        # Scale gate with velocity
        velocity_factor = 1.0 + (speed / self.config['gate_velocity_threshold']) * self.config['gate_velocity_factor']
        gate_radius = self.config['base_gate_radius'] * velocity_factor
        
        return min(gate_radius, self.config['max_gate_radius'])
    
    def _is_detection_valid(self, detection: np.ndarray, frame: int) -> Tuple[bool, str]:
        """
        Check if a YOLO detection is valid using adaptive gating.
        
        Args:
            detection: Detected position [x, y]
            frame: Current frame number
            
        Returns:
            (is_valid, reason)
        """
        if self.ball_state is None:
            # No ball state yet - accept first detection
            return True, "initial_detection"
        
        # Get predicted position
        predicted_pos = self.ball_state.position
        predicted_vel = self.ball_state.velocity
        
        # Calculate spatial distance
        spatial_distance = np.linalg.norm(detection - predicted_pos)
        
        # Calculate adaptive gate radius
        gate_radius = self._calculate_gate_radius(predicted_vel)
        
        # Check spatial gate
        if spatial_distance > gate_radius:
            self.stats['rejected_detections'] += 1
            return False, f"outside_gate_{spatial_distance:.1f}px_vs_{gate_radius:.1f}px"
        
        # Check velocity matching (if we have recent detections)
        if len(self.recent_detections) >= 2:
            # Estimate velocity from detection
            last_pos = self.recent_detections[-1][1]
            frames_diff = frame - self.recent_detections[-1][0]
            
            if frames_diff > 0:
                detected_velocity = (detection - last_pos) / frames_diff
                
                # Normalize velocities for comparison
                pred_vel_norm = np.linalg.norm(predicted_vel)
                det_vel_norm = np.linalg.norm(detected_velocity)
                
                if pred_vel_norm > 0.1 and det_vel_norm > 0.1:
                    # Calculate cosine similarity
                    pred_unit = predicted_vel / pred_vel_norm
                    det_unit = detected_velocity / det_vel_norm
                    similarity = np.dot(pred_unit, det_unit)
                    
                    if similarity < self.config['velocity_match_threshold']:
                        self.stats['rejected_detections'] += 1
                        return False, f"velocity_mismatch_{similarity:.2f}"
        
        self.stats['gated_detections'] += 1
        return True, "valid"
    
    def _predict_next_position(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next ball position using physics (velocity + acceleration + friction).
        
        Returns:
            (predicted_position, predicted_velocity)
        """
        if self.ball_state is None:
            raise ValueError("Cannot predict without ball state")
        
        pos = self.ball_state.position
        vel = self.ball_state.velocity
        acc = self.ball_state.acceleration
        
        # Apply friction (velocity decay)
        vel = vel * self.config['friction_coefficient']
        
        # Apply acceleration
        vel = vel + acc
        
        # Clamp velocity to maximum
        speed = np.linalg.norm(vel)
        if speed > self.config['max_velocity']:
            vel = vel / speed * self.config['max_velocity']
        
        # Update position: x = x + v*dt (dt = 1 frame)
        new_pos = pos + vel
        
        # Update acceleration (decay towards zero)
        acc = acc * 0.9  # Decay acceleration
        
        return new_pos, vel
    
    def _estimate_velocity_from_history(self) -> Optional[np.ndarray]:
        """
        Estimate velocity from recent detection history.
        
        Returns:
            Velocity [vx, vy] or None if insufficient data
        """
        if len(self.recent_detections) < 2:
            return None
        
        # Use most recent detections
        recent = list(self.recent_detections)[-3:]  # Last 3 detections
        
        if len(recent) < 2:
            return None
        
        # Calculate velocity from last two detections
        pos1, frame1 = recent[-2][1], recent[-2][0]
        pos2, frame2 = recent[-1][1], recent[-1][0]
        
        if frame2 == frame1:
            return None
        
        velocity = (pos2 - pos1) / (frame2 - frame1)
        return velocity
    
    def _estimate_acceleration_from_history(self) -> Optional[np.ndarray]:
        """
        Estimate acceleration from recent detection history.
        
        Returns:
            Acceleration [ax, ay] or None if insufficient data
        """
        if len(self.recent_detections) < 3:
            return None
        
        # Use last 3 detections
        recent = list(self.recent_detections)[-3:]
        
        if len(recent) < 3:
            return None
        
        # Calculate velocities
        pos1, frame1 = recent[0][1], recent[0][0]
        pos2, frame2 = recent[1][1], recent[1][0]
        pos3, frame3 = recent[2][1], recent[2][0]
        
        if frame2 == frame1 or frame3 == frame2:
            return None
        
        vel1 = (pos2 - pos1) / (frame2 - frame1)
        vel2 = (pos3 - pos2) / (frame3 - frame2)
        
        # Average time difference
        avg_dt = ((frame2 - frame1) + (frame3 - frame2)) / 2.0
        
        if avg_dt == 0:
            return None
        
        acceleration = (vel2 - vel1) / avg_dt
        
        # Clamp acceleration
        acc_mag = np.linalg.norm(acceleration)
        if acc_mag > self.config['max_acceleration']:
            acceleration = acceleration / acc_mag * self.config['max_acceleration']
        
        return acceleration
    
    def _interpolate_position(self, frame: int) -> Optional[np.ndarray]:
        """
        Interpolate ball position for a missing frame.
        
        Args:
            frame: Frame number to interpolate
            
        Returns:
            Interpolated position or None if interpolation not possible
        """
        if self.ball_state is None:
            return None
        
        # Check if we have recent detections on both sides
        if len(self.recent_detections) < 2:
            return None
        
        # Find detections before and after this frame
        before = None
        after = None
        
        for det_frame, det_pos in self.recent_detections:
            if det_frame < frame:
                if before is None or det_frame > before[0]:
                    before = (det_frame, det_pos)
            elif det_frame > frame:
                if after is None or det_frame < after[0]:
                    after = (det_frame, det_pos)
        
        if before is None or after is None:
            return None
        
        # Linear interpolation
        frame1, pos1 = before
        frame2, pos2 = after
        
        if frame2 == frame1:
            return pos1
        
        alpha = (frame - frame1) / (frame2 - frame1)
        interpolated = pos1 + alpha * (pos2 - pos1)
        
        return interpolated
    
    def _apply_smoothing(self, position: np.ndarray) -> np.ndarray:
        """
        Backward-compat shim.

        Prefer `_apply_state_smoothing(position, state)` which uses the semantic
        state machine to set smoothing strength.
        """
        if self.ball_state is None:
            return position
        return self._apply_state_smoothing(position, self.ball_state.game_state)
    
    def update(
        self,
        frame: int,
        ball_detections: Optional[np.ndarray],
        *,
        detection_confidences: Optional[Iterable[float]] = None,
        player_bboxes_xyxy: Optional[np.ndarray] = None,
        pitch_polygon: Optional[np.ndarray] = None,
    ) -> BallObservation:
        """
        Update ball tracker with new detections.
        
        Args:
            frame: Current frame number
            ball_detections: Array of ball detections [N, 2] in pixels, or None.
            detection_confidences: Optional iterable of confidences aligned with `ball_detections`.
                If omitted, detector confidence defaults to 0.5.
            player_bboxes_xyxy: Optional player bboxes [M,4] in pixels for foot proximity logic.
            pitch_polygon: Optional pitch polygon [K,2] for pitch-aware gating.
            
        Returns:
            BallObservation for current frame (position ALWAYS available)
        """
        self.stats['total_frames'] += 1
        
        # Precompute feet positions (if provided)
        feet = self._extract_feet_positions(player_bboxes_xyxy)

        # Build detection candidates (pos + conf)
        det_candidates: List[Tuple[np.ndarray, float]] = []
        if ball_detections is not None and len(ball_detections) > 0:
            if detection_confidences is None:
                confs = [0.5 for _ in range(len(ball_detections))]
            else:
                confs = [float(c) for c in detection_confidences]
                if len(confs) != len(ball_detections):
                    confs = (confs + [0.5] * len(ball_detections))[:len(ball_detections)]
            for pos, c in zip(ball_detections, confs):
                det_candidates.append((np.asarray(pos, dtype=np.float32).copy(), float(np.clip(c, 0.0, 1.0))))

        # Initialization: pick best pitch-valid detection, else fallback center (confidence 0)
        if self.ball_state is None:
            chosen = None
            chosen_c = -1.0
            chosen_debug = {}
            for pos, c in det_candidates:
                pitch_valid = self._pitch_validity(pos, pitch_polygon)
                if pitch_valid < 0.5:
                    continue
                if c > chosen_c:
                    chosen = pos
                    chosen_c = c
                    chosen_debug = {"init_from_detection": True, "detector_conf": float(c)}

            if chosen is None:
                default_pos = np.array([640.0, 360.0], dtype=np.float32)
                vel = np.array([0.0, 0.0], dtype=np.float32)
                acc = np.array([0.0, 0.0], dtype=np.float32)
                self.ball_state = BallPhysicsState(
                    position=default_pos,
                    velocity=vel,
                    acceleration=acc,
                    last_detected_frame=None,
                    last_detected_position=None,
                    frames_since_detection=0,
                    confidence=0.0,
                    game_state=BallState.OCCLUDED,
                )
                obs = BallObservation(
                    frame=frame,
                    position=default_pos,
                    state=self.ball_state.game_state,
                    source=BallObservationSource.INTERPOLATED,
                    confidence=0.0,
                    velocity=vel,
                    acceleration=acc,
                    is_predicted=True,
                    debug={"init": "default_center"},
                )
                self.detection_history.append(obs)
                self.frame_observations[frame] = obs
                return obs

            vel = np.array([0.0, 0.0], dtype=np.float32)
            acc = np.array([0.0, 0.0], dtype=np.float32)
            min_foot_dist = self._min_distance_to_points(chosen, feet)
            pitch_valid = self._pitch_validity(chosen, pitch_polygon)
            game_state = self._infer_game_state(
                selected_source=BallObservationSource.DETECTED,
                has_detection=True,
                min_foot_dist=min_foot_dist,
                pitch_valid=pitch_valid,
                speed=0.0,
                acc_mag=0.0,
            )
            self.ball_state = BallPhysicsState(
                position=chosen.copy(),
                velocity=vel,
                acceleration=acc,
                last_detected_frame=frame,
                last_detected_position=chosen.copy(),
                frames_since_detection=0,
                confidence=float(np.clip(chosen_c, 0.0, 1.0)),
                game_state=game_state,
            )
            self.recent_detections.append((frame, chosen.copy()))
            self.ball_state.history.append((frame, chosen.copy()))
            self.stats["detected_frames"] += 1
            obs = BallObservation(
                frame=frame,
                position=chosen.copy(),
                state=game_state,
                source=BallObservationSource.DETECTED,
                confidence=self.ball_state.confidence,
                velocity=vel,
                acceleration=acc,
                is_predicted=False,
                debug=chosen_debug,
            )
            self.detection_history.append(obs)
            self.frame_observations[frame] = obs
            return obs

        # We have an existing state: create a physics prediction candidate
        predicted_pos, predicted_vel = self._predict_next_position()
        gate_radius = self._calculate_gate_radius(self.ball_state.velocity)

        # Track candidates (pos, source, detector_conf, score, debug)
        candidates: List[Tuple[np.ndarray, BallObservationSource, float, float, Dict]] = []

        # Always add physics prediction as a candidate
        pred_pitch_valid = self._pitch_validity(predicted_pos, pitch_polygon)
        pred_min_foot_dist = self._min_distance_to_points(predicted_pos, feet)
        pred_score, pred_debug = self._score_candidate(
            detector_conf=0.0,
            dist_to_pred=0.0,
            gate_radius=gate_radius,
            implied_vel=predicted_vel,
            prev_vel=self.ball_state.velocity,
            min_foot_dist=pred_min_foot_dist,
            pitch_valid=pred_pitch_valid,
        )
        candidates.append((predicted_pos.copy(), BallObservationSource.PREDICTED, 0.0, pred_score, {"candidate": "predicted", **pred_debug}))

        # Add YOLO detections as candidates (with hard-physics rejection)
        has_detection = len(det_candidates) > 0
        for pos, det_conf in det_candidates:
            pitch_valid = self._pitch_validity(pos, pitch_polygon)
            min_foot_dist = self._min_distance_to_points(pos, feet)
            dist_to_pred = float(np.linalg.norm(pos - predicted_pos))
            implied_vel = (pos - self.ball_state.position)  # 1-frame implied vel (consistent with tracker units)
            reject_reason = self._hard_physics_reject(
                candidate_pos=pos,
                predicted_pos=predicted_pos,
                prev_vel=self.ball_state.velocity,
                min_foot_dist=min_foot_dist,
                frames_dt=1.0,
            )
            if reject_reason is not None:
                self.stats["rejected_detections"] += 1
                continue

            # Soft gate: extremely far detections are effectively ignored
            score, dbg = self._score_candidate(
                detector_conf=det_conf,
                dist_to_pred=dist_to_pred,
                gate_radius=gate_radius,
                implied_vel=implied_vel,
                prev_vel=self.ball_state.velocity,
                min_foot_dist=min_foot_dist,
                pitch_valid=pitch_valid,
            )
            candidates.append((pos.copy(), BallObservationSource.DETECTED, det_conf, score, {"candidate": "detected", **dbg}))

        # Select best candidate (argmax score)
        candidates.sort(key=lambda t: t[3], reverse=True)
        selected_pos, selected_source, selected_det_conf, selected_score, selected_debug = candidates[0]

        # OUT_OF_PLAY hysteresis: if selected is outside pitch, accumulate; otherwise decay
        selected_pitch_valid = self._pitch_validity(selected_pos, pitch_polygon)
        if selected_pitch_valid < 0.5:
            self._oop_counter += 1
        else:
            self._oop_counter = max(0, self._oop_counter - 1)

        oop_frames_req = int(self.config.get("out_of_play_hysteresis_frames", 10))
        forced_out_of_play = self._oop_counter >= oop_frames_req

        # If we were OUT_OF_PLAY, do not re-enter unless inside pitch and reasonably confident
        if self.ball_state.game_state == BallState.OUT_OF_PLAY:
            if selected_pitch_valid >= 0.5 and selected_score >= float(self.config.get("reentry_min_confidence", 0.35)):
                # allow re-entry
                self._oop_counter = 0
            else:
                # Stay OUT_OF_PLAY: keep predicting forward (no snapping)
                selected_pos = predicted_pos.copy()
                selected_source = BallObservationSource.PREDICTED
                selected_det_conf = 0.0
                selected_score = pred_score
                selected_debug = {"locked_out_of_play": True, **pred_debug}
                selected_pitch_valid = pred_pitch_valid
                forced_out_of_play = True

        # Compute new velocity/acc from previous
        new_vel = (selected_pos - self.ball_state.position).astype(np.float32)
        new_acc = (new_vel - self.ball_state.velocity).astype(np.float32)
        speed = float(np.linalg.norm(new_vel))
        acc_mag = float(np.linalg.norm(new_acc))

        # Infer semantic state (before smoothing)
        min_foot_dist_sel = self._min_distance_to_points(selected_pos, feet)
        game_state = BallState.OUT_OF_PLAY if forced_out_of_play else self._infer_game_state(
            selected_source=selected_source,
            has_detection=has_detection,
            min_foot_dist=min_foot_dist_sel,
            pitch_valid=selected_pitch_valid,
            speed=speed,
            acc_mag=acc_mag,
        )

        # State-aware smoothing / snapping rules
        if game_state == BallState.OCCLUDED and selected_source == BallObservationSource.DETECTED:
            # During occlusion, require a strong score to snap to a new detection
            if selected_score < 0.55:
                selected_pos = predicted_pos.copy()
                selected_source = BallObservationSource.PREDICTED
                selected_det_conf = 0.0
                selected_score = pred_score
                selected_debug = {"occluded_no_snap": True, **pred_debug}
                new_vel = (selected_pos - self.ball_state.position).astype(np.float32)
                new_acc = (new_vel - self.ball_state.velocity).astype(np.float32)
                speed = float(np.linalg.norm(new_vel))
                acc_mag = float(np.linalg.norm(new_acc))

        # Apply state-dependent smoothing (strong in CONTROLLED/LOOSE, weak in IN_FLIGHT)
        smoothed_pos = self._apply_state_smoothing(selected_pos, game_state).astype(np.float32)
        smoothed_vel = (smoothed_pos - self.ball_state.position).astype(np.float32)
        smoothed_acc = (smoothed_vel - self.ball_state.velocity).astype(np.float32)

        # Update internal physics state
        self.ball_state.position = smoothed_pos
        self.ball_state.velocity = smoothed_vel
        self.ball_state.acceleration = smoothed_acc
        self.ball_state.game_state = game_state

        # Detection bookkeeping + confidence update
        if selected_source == BallObservationSource.DETECTED:
            self.stats["detected_frames"] += 1
            self.ball_state.last_detected_frame = frame
            self.ball_state.last_detected_position = selected_pos.copy()
            self.ball_state.frames_since_detection = 0
            # increase confidence on consistent detections + foot interaction
            foot_bonus = float(np.clip(1.0 - (min_foot_dist_sel / max(1.0, float(self.config.get("foot_proximity_px", 140.0)))), 0.0, 1.0))
            target = float(np.clip(0.6 + 0.4 * selected_det_conf + 0.2 * foot_bonus, 0.0, 1.0))
            self.ball_state.confidence = float(np.clip(0.75 * self.ball_state.confidence + 0.25 * target, 0.0, 1.0))
            self.recent_detections.append((frame, selected_pos.copy()))
        else:
            self.stats["predicted_frames"] += 1
            self.ball_state.frames_since_detection += 1
            decay = float(self.config.get("confidence_decay_rate", 0.02))
            if game_state == BallState.OCCLUDED:
                decay *= 1.5
                self.stats["occluded_frames"] += 1
            if game_state == BallState.OUT_OF_PLAY:
                self.stats["out_of_play_frames"] += 1
                decay *= 1.2
            self.ball_state.confidence = float(np.clip(self.ball_state.confidence - decay, 0.0, 1.0))

        # Keep history for downstream analysis/debug
        self.ball_state.history.append((frame, smoothed_pos.copy()))

        obs = BallObservation(
            frame=frame,
            position=smoothed_pos,
            state=game_state,
            source=selected_source,
            confidence=float(np.clip(self.ball_state.confidence, 0.0, 1.0)),
            velocity=smoothed_vel,
            acceleration=smoothed_acc,
            is_predicted=(selected_source != BallObservationSource.DETECTED),
            debug={
                **selected_debug,
                "selected_score": float(selected_score),
                "selected_source": selected_source.value,
                "semantic_state": game_state.value,
                "min_foot_dist_px": float(min_foot_dist_sel) if np.isfinite(min_foot_dist_sel) else None,
                "pitch_valid": float(selected_pitch_valid),
                "oop_counter": int(self._oop_counter),
            },
        )
        
        # Store observation
        self.detection_history.append(obs)
        self.frame_observations[frame] = obs
        
        # Trim history to prevent memory issues
        if len(self.detection_history) > 2000:
            old_obs = self.detection_history[:-1000]
            self.detection_history = self.detection_history[-1000:]
            for old in old_obs:
                self.frame_observations.pop(old.frame, None)
        
        return obs
    
    def get_ball_position(self, frame: int) -> Optional[BallObservation]:
        """
        Get ball position for a specific frame.
        
        Args:
            frame: Frame number
            
        Returns:
            BallObservation or None if not available
        """
        return self.frame_observations.get(frame)
    
    def get_current_ball(self) -> Optional[BallObservation]:
        """Get current ball observation."""
        if self.detection_history:
            return self.detection_history[-1]
        return None
    
    def is_detected(self, frame: int) -> bool:
        """Check if ball is DETECTED (not predicted) at frame."""
        obs = self.get_ball_position(frame)
        return obs is not None and obs.source == BallObservationSource.DETECTED
    
    def is_available(self, frame: int) -> bool:
        """Check if ball position is available (ALWAYS True for physics tracker)."""
        obs = self.get_ball_position(frame)
        return obs is not None and obs.position is not None
    
    def get_ball_in_window(self, start_frame: int, end_frame: int) -> List[BallObservation]:
        """
        Get all ball observations in a time window.
        
        Args:
            start_frame: Start frame
            end_frame: End frame
            
        Returns:
            List of BallObservations
        """
        return [obs for obs in self.detection_history 
                if start_frame <= obs.frame <= end_frame]
    
    def get_detection_rate(self) -> float:
        """Get the fraction of frames where ball was detected."""
        if self.stats['total_frames'] == 0:
            return 0.0
        return self.stats['detected_frames'] / self.stats['total_frames']
    
    def get_available_rate(self) -> float:
        """Get the fraction of frames where ball position is available (should be 1.0)."""
        if self.stats['total_frames'] == 0:
            return 0.0
        return 1.0  # Physics tracker always provides position
    
    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        return {
            **self.stats,
            'detection_rate': self.get_detection_rate(),
            'available_rate': self.get_available_rate(),
        }
    
    def reset(self):
        """Reset tracker state."""
        self.ball_state = None
        self.detection_history = []
        self.frame_observations = {}
        self.recent_detections.clear()
        self.stats = {k: 0 for k in self.stats}

