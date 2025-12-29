import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

from pipelines import TrackingPipeline, ProcessingPipeline, DetectionPipeline, KeypointPipeline, TacticalPipeline
from constants import model_path, test_video, EMBEDDING_BATCH_SIZE
from keypoint_detection.keypoint_constants import keypoint_model_path
from pass_detection import HybridPassManager, PassVisualizer
from pass_detection.ball_tracker import BallState, BallObservation
import numpy as np
import cv2
import time
from tqdm import tqdm
from dataclasses import dataclass
import supervision as sv
import pandas as pd


@dataclass
class FrameBallState:
    """Per-frame ball state for tracking and visualization."""
    frame_index: int
    ball_x: Optional[float]  # Center x in frame coordinates
    ball_y: Optional[float]  # Center y in frame coordinates
    bbox: Optional[Tuple[float, float, float, float]]  # x1, y1, x2, y2
    state: str  # "DETECTED", "PREDICTED", "LOST"
    confidence: float = 0.0


class CompleteSoccerAnalysisPipeline:
    """Complete end-to-end soccer analysis pipeline integrating all functionalities.
    
    Key architectural features:
    - HYBRID Pass Detection: Players propose, Ball validates
    - Pass State Machine (IDLE → INITIATING → LOCKED → COOLDOWN)
    - Event-based pass detection (not frame-based)
    - Locked team assignments (never updated mid-match)
    - No duplicate passes (temporal merge + cooldown)
    - Ball can SUPPORT but never CREATE passes
    - Ball tracking with per-frame state (DETECTED/PREDICTED/LOST)
    - Maximum 5-frame prediction window for ball tracker
    """
    
    # Ball visualization colors (BGR format)
    BALL_COLOR_DETECTED = (0, 255, 255)    # Yellow/Cyan for detected ball
    BALL_COLOR_PREDICTED = (0, 165, 255)   # Orange for predicted ball
    BALL_OUTLINE_COLOR = (0, 0, 0)         # Black outline
    
    def __init__(self, detection_model_path: str, keypoint_model_path: str):
        """Initialize all pipeline components.
        
        Args:
            detection_model_path: Path to YOLO detection model
            keypoint_model_path: Path to YOLO keypoint detection model
        """
        self.detection_pipeline = DetectionPipeline(detection_model_path)
        self.keypoint_pipeline = KeypointPipeline(keypoint_model_path)
        self.tracking_pipeline = TrackingPipeline(detection_model_path)
        self.tactical_pipeline = TacticalPipeline(keypoint_model_path, detection_model_path)
        self.processing_pipeline = ProcessingPipeline()
        
        # Per-frame ball state storage (for visualization)
        self.ball_frame_states: Dict[int, FrameBallState] = {}
        
    def initialize_models(self):
        """Initialize all models required for complete analysis."""
        
        print("Initializing all pipeline models...")
        start_time = time.time()
        
        # Initialize all pipeline models
        self.detection_pipeline.initialize_model()
        self.keypoint_pipeline.initialize_model()
        self.tracking_pipeline.initialize_models()
        self.tactical_pipeline.initialize_models()
        
        init_time = time.time() - start_time
        print(f"All models initialized in {init_time:.2f}s")
        
    def analyze_video(self, video_path: str, frame_count: int = -1, output_suffix: str = "_complete_analysis"):
        """Run complete end-to-end soccer analysis with player-only pass detection.
        
        Architecture:
        - Pass State Machine ensures: IDLE → INITIATING → LOCKED → COOLDOWN → IDLE
        - One real-world pass = exactly one detected event
        - Locked team assignments (set once, never updated)
        - Physics validation in dedicated module
        - Debug outputs for full traceability
        - Ball tracker with strict 5-frame prediction limit
        
        Args:
            video_path: Path to input video
            frame_count: Number of frames to process (-1 for all)
            output_suffix: Suffix for output video file
            
        Returns:
            Path to output video
        """
        print("=== Starting Complete Soccer Analysis Pipeline ===")
        print("Architecture: HYBRID Pass Detection (Players Propose, Ball Validates)")
        print("Ball Tracker: Strict 5-frame prediction limit, Kalman filter")
        total_start_time = time.time()
        
        # Validate video path exists
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"Input video: {video_path}")
        print(f"Video file size: {video_path_obj.stat().st_size / (1024*1024):.2f} MB")
        
        # Step 1: Initialize all models
        self.initialize_models()
        
        # Step 2: Train team assignment models
        print("\n[Step 2/8] Training team assignment models...")
        self.tracking_pipeline.train_team_assignment_models(video_path)
        
        # Step 3: Read video frames
        print("\n[Step 3/8] Reading video frames...")
        frames = self.processing_pipeline.read_video_frames(video_path, frame_count)
        print(f"Loaded {len(frames)} frames for processing")
        
        # Get video FPS
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        
        # Step 4: Initialize hybrid pass detection system
        print("\n[Step 4/8] Initializing hybrid pass detection system...")
        print("  - Player Candidate Generator (looser thresholds)")
        print("  - Ball Evidence Validator (strict ball tracking)")
        print("  - Hybrid Pass Manager (players propose, ball validates)")
        print("  - Ball Tracker (DETECTED/PREDICTED/LOST states, max 5-frame prediction)")
        
        pass_config = {
            'fps': video_fps,
            'lock_team_assignments': True,
            'min_final_confidence': 0.5,  # Combined player + ball
            'temporal_merge_window': 15,
            'cooldown_duration': 0.75,
            'pair_lock_duration': 1.5,
        }
        hybrid_pass_manager = HybridPassManager(pass_config)
        pass_visualizer = PassVisualizer({'fps': video_fps})
        
        # Reset ball frame states storage
        self.ball_frame_states = {}
        
        # Step 5: Process all frames with detections, tracking, team assignment, and pass detection
        print("\n[Step 5/8] Processing frames with complete analysis and pass detection...")
        all_tracks = {'player': {}, 'referee': {}, 'player_classids': {}, 'ball': {}}
        player_pitch_positions = {}  # Store player positions in pitch coordinates for pass detection
        
        # Store frame ball detections for visualization (before pitch transform)
        frame_ball_detections: Dict[int, sv.Detections] = {}
        
        # Team binding: Lock team assignments per player (read from hybrid_pass_manager)
        locked_team_map = {}  # player_id -> team_id (locked after first assignment)
        
        # Batch process embeddings for faster UMAP transform
        BATCH_SIZE = 30
        frame_embeddings_buffer = []
        
        for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
            
            # Detect keypoints and objects (including ball)
            keypoints, _ = self.keypoint_pipeline.detect_keypoints_in_frame(frame)
            player_detections, ball_detections, referee_detections = self.detection_pipeline.detect_frame_objects(frame)
            
            # Store ball frame detection for visualization (in frame coordinates)
            frame_ball_detections[i] = ball_detections
            
            # Update with tracking (players only, no ball)
            player_detections = self.tracking_pipeline.tracking_callback(player_detections)

            # Extract crops for team assignment (but batch process UMAP for 23x speedup)
            if len(player_detections.xyxy) > 0:
                crops = self.tracking_pipeline.clustering_manager.embedding_extractor.get_player_crops(frame, player_detections)
                frame_embeddings_buffer.append((i, crops, player_detections, ball_detections, referee_detections, keypoints, frame))
            else:
                # No players, skip team assignment
                frame_embeddings_buffer.append((i, [], player_detections, ball_detections, referee_detections, keypoints, frame))
            
            # Process batch when buffer is full or at end
            if len(frame_embeddings_buffer) >= BATCH_SIZE or i == len(frames) - 1:
                # Extract all embeddings in batch (GPU - fast)
                all_crops = []
                frame_data = []
                for frame_idx, crops, p_det, b_det, r_det, kp, orig_frame in frame_embeddings_buffer:
                    if len(crops) > 0:
                        all_crops.extend(crops)
                        frame_data.append((frame_idx, len(crops), p_det, b_det, r_det, kp, orig_frame))
                
                if len(all_crops) > 0:
                    # Batch extract embeddings (GPU - fast)
                    crop_batches = self.tracking_pipeline.clustering_manager.embedding_extractor.create_batches(all_crops, EMBEDDING_BATCH_SIZE)
                    all_embeddings = self.tracking_pipeline.clustering_manager.embedding_extractor.get_embeddings(crop_batches)
                    
                    # Batch UMAP transform (CPU - 23x faster when batched!)
                    reduced_embeddings, _ = self.tracking_pipeline.clustering_manager.project_embeddings(all_embeddings, train=False)
                    
                    # Batch K-means predict (CPU - fast)
                    cluster_labels, _ = self.tracking_pipeline.clustering_manager.cluster_embeddings(reduced_embeddings, train=False)
                    
                    # Assign labels back to detections
                    label_idx = 0
                    for frame_idx, num_players, p_det, b_det, r_det, kp, orig_frame in frame_data:
                        frame_labels = cluster_labels[label_idx:label_idx+num_players]
                        p_det.class_id = frame_labels
                        label_idx += num_players
                        
                        # Store tracks
                        all_tracks = self.tracking_pipeline.convert_detection_to_tracks(p_det, r_det, all_tracks, frame_idx)
                        
                        # Lock team assignments (set once, never updated)
                        if p_det.tracker_id is not None:
                            for tid, team_id in zip(p_det.tracker_id, p_det.class_id):
                                if tid not in locked_team_map:
                                    locked_team_map[tid] = team_id
                        
                        # Get player pitch positions for pass detection
                        view_transformer = None
                        if kp is not None:
                            view_transformer = self.tactical_pipeline.transform_keypoints_to_pitch(kp)
                        
                        if len(p_det.xyxy) > 0 and view_transformer is not None:
                            pitch_points = self.tactical_pipeline.transform_detections_to_pitch(
                                p_det, view_transformer
                            )
                            
                            # Store pitch positions for each player
                            if p_det.tracker_id is not None:
                                for tracker_id, pitch_pos in zip(p_det.tracker_id, pitch_points):
                                    player_pitch_positions[tracker_id] = np.array(pitch_pos)
                        
                        # Get ball positions in pitch coordinates (ALWAYS update ball tracker)
                        ball_pitch_positions = None
                        if len(b_det.xyxy) > 0 and view_transformer is not None:
                            # Transform ball detections to pitch coordinates
                            ball_pitch_positions = self.tactical_pipeline.transform_detections_to_pitch(
                                b_det, view_transformer
                            )
                            if ball_pitch_positions is not None and len(ball_pitch_positions) > 0:
                                ball_pitch_positions = np.array(ball_pitch_positions)
                        
                        # ALWAYS process hybrid pass detection (ensures ball tracker is updated every frame)
                        # Build current frame positions (pitch coordinates)
                        current_positions = {}
                        if p_det.tracker_id is not None:
                            for tid in p_det.tracker_id:
                                if tid in player_pitch_positions:
                                    current_positions[tid] = player_pitch_positions[tid]
                        
                        # Build team assignments (use LOCKED map)
                        player_teams = {}
                        if p_det.tracker_id is not None:
                            player_teams = {tid: locked_team_map.get(tid, 0) for tid in p_det.tracker_id}
                        
                        # Process frame through hybrid manager
                        # This updates ball tracker every frame and checks for pass candidates
                        hybrid_pass_manager.process_frame(
                            frame_idx,
                            current_positions,
                            player_teams,
                            ball_pitch_positions  # Ball positions in pitch coordinates (None if not available)
                        )
                        
                        # Store ball state from tracker for visualization
                        self._store_ball_state_from_tracker(
                            frame_idx,
                            hybrid_pass_manager.ball_tracker,
                            frame_ball_detections.get(frame_idx)
                        )
                
                # Process frames without players (still need to update ball tracker)
                for frame_idx, crops, p_det, b_det, r_det, kp, orig_frame in frame_embeddings_buffer:
                    if len(crops) == 0:
                        all_tracks = self.tracking_pipeline.convert_detection_to_tracks(p_det, r_det, all_tracks, frame_idx)
                        
                        # Still update ball tracker for frames without player detections
                        ball_pitch_positions = None
                        if len(b_det.xyxy) > 0 and kp is not None:
                            view_transformer = self.tactical_pipeline.transform_keypoints_to_pitch(kp)
                            if view_transformer is not None:
                                ball_pitch_positions = self.tactical_pipeline.transform_detections_to_pitch(
                                    b_det, view_transformer
                                )
                                if ball_pitch_positions is not None and len(ball_pitch_positions) > 0:
                                    ball_pitch_positions = np.array(ball_pitch_positions)
                        
                        # Update ball tracker (no players, but ball tracking continues)
                        hybrid_pass_manager.process_frame(
                            frame_idx,
                            {},  # No player positions
                            {},  # No team assignments
                            ball_pitch_positions
                        )
                        
                        # Store ball state from tracker for visualization
                        self._store_ball_state_from_tracker(
                            frame_idx,
                            hybrid_pass_manager.ball_tracker,
                            frame_ball_detections.get(frame_idx)
                        )
                
                # Clear buffer
                frame_embeddings_buffer = []

        # Step 6: Get confirmed passes (no debug CSVs)
        print("\n[Step 6/8] Collecting confirmed passes...")
        
        # Get all confirmed passes (hybrid-validated)
        all_passes = hybrid_pass_manager.get_confirmed_passes()
        ball_tracker = hybrid_pass_manager.get_ball_tracker()
        
        print(f"  - Total confirmed passes: {len(all_passes)}")
        
        # Save debug outputs for rejected passes
        debug_dir = Path(video_path).parent / "pass_detection_debug"
        self._save_debug_outputs(hybrid_pass_manager, debug_dir)
        
        # Step 7: Annotate frames with passes
        print("\n[Step 7/8] Annotating frames with detections and passes...")
        object_annotated_frames = self.tracking_pipeline.annotate_frames(frames, all_tracks)
        
        # Add pass visualization and ball drawing to frames
        output_frames = []
        for i, frame in enumerate(object_annotated_frames):
            # Get player frame positions for this frame
            player_frame_positions = {}
            if i in all_tracks['player']:
                for player_id, bbox in all_tracks['player'][i].items():
                    if bbox and bbox[0] is not None:
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        player_frame_positions[player_id] = (center_x, center_y)
            
            # Draw passes using new visualizer (with locked team map)
            annotated_frame = pass_visualizer.draw_passes(
                frame, 
                all_passes, 
                player_frame_positions,
                current_frame=i,
                team_map=locked_team_map
            )
            
            # Draw ball when DETECTED or PREDICTED (NOT when LOST)
            annotated_frame = self._draw_ball_on_frame(annotated_frame, i)
            
            output_frames.append(annotated_frame)

        # Step 8: Write final output video and export passes CSV
        print("\n[Step 8/8] Writing complete analysis video and exporting passes...")
        output_path = self.processing_pipeline.generate_output_path(video_path, output_suffix)
        self.processing_pipeline.write_video_output(output_frames, output_path)
        
        # Export passes to single CSV file with required columns
        csv_output_path = Path(video_path).parent / "passes.csv"
        self._export_passes_to_csv(all_passes, str(csv_output_path), video_fps, locked_team_map)
        
        # Summary
        total_time = time.time() - total_start_time
        metrics = hybrid_pass_manager.metrics
        ball_stats = ball_tracker.get_stats()
        
        # Count ball states from stored frame states
        detected_frames = sum(1 for s in self.ball_frame_states.values() if s.state == "DETECTED")
        predicted_frames = sum(1 for s in self.ball_frame_states.values() if s.state == "PREDICTED")
        lost_frames = sum(1 for s in self.ball_frame_states.values() if s.state == "LOST")
        
        print(f"\n=== Complete Soccer Analysis Finished ===")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Frames processed: {len(frames)}")
        print(f"Average time per frame: {total_time/len(frames):.3f}s")
        
        print(f"\nBall Tracking Stats:")
        print(f"  - DETECTED frames: {detected_frames}")
        print(f"  - PREDICTED frames: {predicted_frames}")
        print(f"  - LOST frames: {lost_frames}")
        print(f"  - Detection rate: {ball_stats.get('detection_rate', 0):.1%}")
        print(f"  - Available rate: {ball_stats.get('available_rate', 0):.1%}")
        print(f"  - Rejected (jump): {ball_stats.get('rejected_jump', 0)}")
        print(f"  - Rejected (jitter): {ball_stats.get('rejected_jitter', 0)}")
        
        print(f"\nHybrid Pass Detection Metrics:")
        print(f"  - Player candidates generated: {metrics['player_candidates_generated']}")
        print(f"  - Ball validated passes: {metrics['ball_validated_passes']}")
        print(f"  - Rejected (no ball evidence): {metrics['rejected_no_ball_evidence']}")
        print(f"  - Rejected (contradictory ball): {metrics['rejected_contradictory_ball']}")
        print(f"  - Rejected (duplicate): {metrics['rejected_duplicate']}")
        print(f"  - Rejected (cooldown): {metrics['rejected_cooldown']}")
        print(f"  - Rejected (low confidence): {metrics['rejected_low_confidence']}")
        print(f"\nOutput video: {output_path}")
        print(f"Passes CSV: {csv_output_path}")
        print(f"Debug outputs: {debug_dir}")
        
        return output_path
    
    def _store_ball_state_from_tracker(self, frame_index: int, 
                                        ball_tracker, 
                                        ball_detections: Optional[sv.Detections]):
        """
        Store ball state for visualization from the ball tracker.
        
        Uses the tracker's observation which properly handles the 5-frame prediction limit.
        
        Args:
            frame_index: Current frame index
            ball_tracker: The StrictBallTracker instance
            ball_detections: Original ball detections (for frame coordinates)
        """
        # Get observation from tracker
        obs = ball_tracker.get_ball_position(frame_index)
        
        if obs is None:
            # No observation recorded for this frame
            self.ball_frame_states[frame_index] = FrameBallState(
                frame_index=frame_index,
                ball_x=None,
                ball_y=None,
                bbox=None,
                state="LOST",
                confidence=0.0
            )
            return
        
        # Map tracker state to visualization state
        if obs.state == BallState.DETECTED:
            state_str = "DETECTED"
        elif obs.state == BallState.PREDICTED:
            state_str = "PREDICTED"
        else:
            state_str = "LOST"
        
        # Get frame coordinates for visualization
        if ball_detections is not None and len(ball_detections.xyxy) > 0 and obs.state == BallState.DETECTED:
            # Use detection bounding box for detected ball
            bbox = ball_detections.xyxy[0]
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            confidence = ball_detections.confidence[0] if ball_detections.confidence is not None else 1.0
            
            self.ball_frame_states[frame_index] = FrameBallState(
                frame_index=frame_index,
                ball_x=center_x,
                ball_y=center_y,
                bbox=(x1, y1, x2, y2),
                state=state_str,
                confidence=float(confidence)
            )
        elif obs.state == BallState.PREDICTED:
            # For predicted state, use the last known detection position
            # Find the last detected frame's position
            prev_detected_state = None
            for lookback in range(1, 6):  # Max 5 frames back
                prev_idx = frame_index - lookback
                if prev_idx in self.ball_frame_states:
                    prev_state = self.ball_frame_states[prev_idx]
                    if prev_state.state == "DETECTED" and prev_state.bbox is not None:
                        prev_detected_state = prev_state
                        break
            
            if prev_detected_state is not None:
                # Use last detected position with decreased confidence
                self.ball_frame_states[frame_index] = FrameBallState(
                    frame_index=frame_index,
                    ball_x=prev_detected_state.ball_x,
                    ball_y=prev_detected_state.ball_y,
                    bbox=prev_detected_state.bbox,
                    state=state_str,
                    confidence=obs.confidence
                )
            else:
                # No previous detection available
                self.ball_frame_states[frame_index] = FrameBallState(
                    frame_index=frame_index,
                    ball_x=None,
                    ball_y=None,
                    bbox=None,
                    state="LOST",
                    confidence=0.0
                )
        else:
            # LOST state
            self.ball_frame_states[frame_index] = FrameBallState(
                frame_index=frame_index,
                ball_x=None,
                ball_y=None,
                bbox=None,
                state=state_str,
                confidence=0.0
            )
    
    def _draw_ball_on_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """
        Draw ball bounding box and center on frame.
        
        Only draws when ball is DETECTED or PREDICTED, NOT when LOST.
        
        Args:
            frame: Input video frame
            frame_index: Current frame index
            
        Returns:
            Annotated frame
        """
        if frame_index not in self.ball_frame_states:
            return frame
        
        ball_state = self.ball_frame_states[frame_index]
        
        # Only draw if DETECTED or PREDICTED (not LOST)
        if ball_state.state == "LOST" or ball_state.bbox is None:
            return frame
        
        annotated = frame.copy()
        x1, y1, x2, y2 = ball_state.bbox
        
        # Choose color based on state
        if ball_state.state == "DETECTED":
            color = self.BALL_COLOR_DETECTED  # Yellow/Cyan
            thickness = 3
        else:  # PREDICTED
            color = self.BALL_COLOR_PREDICTED  # Orange
            thickness = 2
        
        # Draw bounding box
        cv2.rectangle(annotated, 
                     (int(x1), int(y1)), 
                     (int(x2), int(y2)), 
                     color, thickness)
        
        # Draw center point
        center_x = int(ball_state.ball_x)
        center_y = int(ball_state.ball_y)
        cv2.circle(annotated, (center_x, center_y), 5, color, -1)
        cv2.circle(annotated, (center_x, center_y), 5, self.BALL_OUTLINE_COLOR, 1)
        
        # Draw state label
        label = f"Ball ({ball_state.state})"
        label_y = int(y1) - 10 if int(y1) > 30 else int(y2) + 20
        cv2.putText(annotated, label, 
                   (int(x1), label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def _save_debug_outputs(self, hybrid_pass_manager: HybridPassManager, output_dir: Path):
        """
        Save debug outputs for pass detection analysis.
        
        Args:
            hybrid_pass_manager: The HybridPassManager instance
            output_dir: Directory to save debug outputs
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save rejected candidates
        rejected = hybrid_pass_manager.rejected_candidates
        if rejected:
            df = pd.DataFrame(rejected)
            df.to_csv(output_dir / "rejected_passes.csv", index=False)
            print(f"  - Saved {len(rejected)} rejected candidates to {output_dir / 'rejected_passes.csv'}")
        
        # Save player candidates
        candidates = hybrid_pass_manager.player_candidates
        if candidates:
            candidates_data = [{
                'candidate_id': c.candidate_id,
                'initiator_id': c.initiator_id,
                'receiver_id': c.receiver_id,
                'team_id': c.team_id,
                'start_frame': c.start_frame_estimate,
                'end_frame': c.end_frame_estimate,
                'player_confidence': c.confidence_player,
                'distance_meters': c.distance_meters,
                'validated': c.validated,
                'ball_evidence_score': c.ball_evidence_score,
                'rejection_reason': c.rejection_reason
            } for c in candidates]
            df = pd.DataFrame(candidates_data)
            df.to_csv(output_dir / "player_candidates.csv", index=False)
        
        # Save ball validations
        validations = hybrid_pass_manager.ball_validations
        if validations:
            df = pd.DataFrame(validations)
            df.to_csv(output_dir / "ball_validations.csv", index=False)
        
        # Save ball tracker stats
        ball_stats = hybrid_pass_manager.ball_tracker.get_stats()
        stats_df = pd.DataFrame([ball_stats])
        stats_df.to_csv(output_dir / "ball_tracker_stats.csv", index=False)
        
        # Save metrics
        metrics = hybrid_pass_manager.metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(output_dir / "pass_detection_metrics.csv", index=False)
    
    def _export_passes_to_csv(self, passes: List, csv_path: str, fps: float, 
                              team_map: dict = None):
        """
        Export passes to CSV with required columns.
        
        Required columns:
        - pass_id
        - initiator_player_id
        - receiver_player_id
        - start_frame
        - end_frame
        - start_time_seconds
        - end_time_seconds
        - confidence_score
        
        Args:
            passes: List of PassEvent objects
            csv_path: Path to output CSV file
            fps: Video frames per second
            team_map: Locked team assignments (not used in output, but kept for compatibility)
        """
        # Prepare data for CSV with required columns
        csv_data = []
        for pass_event in passes:
            start_time = pass_event.start_frame / fps if pass_event.start_frame is not None else 0.0
            end_time = pass_event.end_frame / fps if pass_event.end_frame is not None else 0.0
            
            csv_data.append({
                'pass_id': pass_event.event_id,
                'initiator_player_id': pass_event.from_player_id,
                'receiver_player_id': pass_event.to_player_id,
                'start_frame': pass_event.start_frame,
                'end_frame': pass_event.end_frame,
                'start_time_seconds': round(start_time, 3),
                'end_time_seconds': round(end_time, 3),
                'confidence_score': round(pass_event.confidence, 4)
            })
        
        # Create DataFrame and save to CSV
        required_columns = [
            'pass_id', 'initiator_player_id', 'receiver_player_id',
            'start_frame', 'end_frame', 'start_time_seconds', 
            'end_time_seconds', 'confidence_score'
        ]
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            # Ensure column order
            df = df[required_columns]
            df.to_csv(csv_path, index=False)
            print(f"✅ Exported {len(csv_data)} passes to {csv_path}")
        else:
            # Create empty CSV with headers (never empty if passes exist)
            df = pd.DataFrame(columns=required_columns)
            df.to_csv(csv_path, index=False)
            print(f"⚠️  No passes detected. Created CSV with headers at {csv_path}")



if __name__ == "__main__":
    # Run Complete End-to-End Soccer Analysis Pipeline
    print("Starting Soccer Analysis...")
    print("Architecture: State Machine + Event-Based Pass Detection")
    print("Guarantees: No duplicates | Correct timing | Stable colors | Ball-independent")
    print()
    
    pipeline = CompleteSoccerAnalysisPipeline(model_path, keypoint_model_path)
    
    # Use the specified video
    video_path = "/home/essashah/smo_model_new/youtube_video.mp4"
    
    # Process all frames (-1 = all frames)
    output_video = pipeline.analyze_video(video_path, frame_count=-1)    
    print(f"\nAnalysis finished! Output video: {output_video}")
