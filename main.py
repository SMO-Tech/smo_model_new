import sys
from pathlib import Path
from typing import List
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

from pipelines import TrackingPipeline, ProcessingPipeline, DetectionPipeline, KeypointPipeline, TacticalPipeline
from constants import model_path, test_video, EMBEDDING_BATCH_SIZE
from keypoint_detection.keypoint_constants import keypoint_model_path
from pass_detection import HybridPassManager, PassVisualizer
from pass_detection.ball_tracker import BallState
import numpy as np
import time
from tqdm import tqdm
import supervision as sv
import pandas as pd


class CompleteSoccerAnalysisPipeline:
    """Complete end-to-end soccer analysis pipeline integrating all functionalities.
    
    Key architectural features:
    - HYBRID Pass Detection: Players propose, Ball validates
    - Pass State Machine (IDLE → INITIATING → LOCKED → COOLDOWN)
    - Event-based pass detection (not frame-based)
    - Locked team assignments (never updated mid-match)
    - No duplicate passes (temporal merge + cooldown)
    - Ball can SUPPORT but never CREATE passes
    """
    
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
        
        Args:
            video_path: Path to input video
            frame_count: Number of frames to process (-1 for all)
            output_suffix: Suffix for output video file
            
        Returns:
            Path to output video
        """
        print("=== Starting Complete Soccer Analysis Pipeline ===")
        print("Architecture: HYBRID Pass Detection (Players Propose, Ball Validates)")
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
        import cv2
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        
        # Step 4: Initialize hybrid pass detection system
        print("\n[Step 4/8] Initializing hybrid pass detection system...")
        print("  - Player Candidate Generator (looser thresholds)")
        print("  - Ball Evidence Validator (strict ball tracking)")
        print("  - Hybrid Pass Manager (players propose, ball validates)")
        
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
        
        # Step 5: Process all frames with detections, tracking, team assignment, and pass detection
        print("\n[Step 5/8] Processing frames with complete analysis and pass detection...")
        all_tracks = {'player': {}, 'referee': {}, 'player_classids': {}}
        player_pitch_positions = {}  # Store player positions in pitch coordinates for pass detection
        
        # Team binding: Lock team assignments per player (read from hybrid_pass_manager)
        locked_team_map = {}  # player_id -> team_id (locked after first assignment)
        
        # Batch process embeddings for faster UMAP transform
        BATCH_SIZE = 30
        frame_embeddings_buffer = []
        
        for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
            
            # Detect keypoints and objects (including ball)
            keypoints, _ = self.keypoint_pipeline.detect_keypoints_in_frame(frame)
            player_detections, ball_detections, referee_detections = self.detection_pipeline.detect_frame_objects(frame)
            
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
                        if len(p_det.xyxy) > 0 and kp is not None:
                            view_transformer = self.tactical_pipeline.transform_keypoints_to_pitch(kp)
                            if view_transformer is not None:
                                pitch_points = self.tactical_pipeline.transform_detections_to_pitch(
                                    p_det, view_transformer
                                )
                                
                                # Store pitch positions for each player
                                if p_det.tracker_id is not None:
                                    for tracker_id, pitch_pos in zip(p_det.tracker_id, pitch_points):
                                        player_pitch_positions[tracker_id] = np.array(pitch_pos)
                        
                        # Get ball positions in pitch coordinates
                        ball_pitch_positions = None
                        if len(b_det.xyxy) > 0 and kp is not None:
                            view_transformer = self.tactical_pipeline.transform_keypoints_to_pitch(kp)
                            if view_transformer is not None:
                                # Transform ball detections to pitch coordinates
                                ball_centers = np.array([
                                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                                    for bbox in b_det.xyxy
                                ])
                                ball_pitch_positions = self.tactical_pipeline.transform_detections_to_pitch(
                                    b_det, view_transformer
                                )
                        
                        # Process hybrid pass detection
                        if len(p_det.xyxy) > 0 and p_det.tracker_id is not None:
                            # Build current frame positions (pitch coordinates)
                            current_positions = {}
                            for tid in p_det.tracker_id:
                                if tid in player_pitch_positions:
                                    current_positions[tid] = player_pitch_positions[tid]
                            
                            # Build team assignments (use LOCKED map)
                            player_teams = {tid: locked_team_map.get(tid, 0) for tid in p_det.tracker_id}
                            
                            # Process frame through hybrid manager
                            # Returns newly confirmed passes (players propose, ball validates)
                            hybrid_pass_manager.process_frame(
                                frame_idx,
                                current_positions,
                                player_teams,
                                ball_pitch_positions  # Ball positions in pitch coordinates
                            )
                
                # Process frames without players
                for frame_idx, crops, p_det, b_det, r_det, kp, orig_frame in frame_embeddings_buffer:
                    if len(crops) == 0:
                        all_tracks = self.tracking_pipeline.convert_detection_to_tracks(p_det, r_det, all_tracks, frame_idx)
                
                # Clear buffer
                frame_embeddings_buffer = []

        # Step 6: Save debug outputs
        print("\n[Step 6/8] Saving hybrid pass detection debug outputs...")
        debug_dir = Path(video_path).parent / "debug"
        hybrid_pass_manager.save_debug_outputs(debug_dir)
        
        # Get all confirmed passes (hybrid-validated)
        all_passes = hybrid_pass_manager.get_confirmed_passes()
        ball_tracker = hybrid_pass_manager.get_ball_tracker()
        
        # Step 7: Annotate frames with passes
        print("\n[Step 7/8] Annotating frames with detections and passes...")
        object_annotated_frames = self.tracking_pipeline.annotate_frames(frames, all_tracks)
        
        # Add pass visualization to frames using new visualizer
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
            
            # Draw ball ONLY when DETECTED
            ball_obs = ball_tracker.get_ball_position(i)
            if ball_obs and ball_obs.state == BallState.DETECTED and ball_obs.position is not None:
                # Get ball position in frame coordinates (need to transform back from pitch)
                # For now, skip ball drawing in frame coordinates (would need inverse transform)
                # Ball is tracked in pitch coordinates, visualization can be added later
                pass
            
            output_frames.append(annotated_frame)

        # Step 8: Write final output video and export passes CSV
        print("\n[Step 8/8] Writing complete analysis video and exporting passes...")
        output_path = self.processing_pipeline.generate_output_path(video_path, output_suffix)
        self.processing_pipeline.write_video_output(output_frames, output_path)
        
        # Export passes to CSV with formatted timestamps
        csv_output_path = output_path.replace('.mp4', '_passes.csv')
        self._export_passes_to_csv(all_passes, csv_output_path, video_fps, locked_team_map)
        
        # Summary
        total_time = time.time() - total_start_time
        metrics = hybrid_pass_manager.metrics
        
        print(f"\n=== Complete Soccer Analysis Finished ===")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Frames processed: {len(frames)}")
        print(f"Average time per frame: {total_time/len(frames):.3f}s")
        metrics = hybrid_pass_manager.metrics
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
        print(f"Debug outputs saved to: {debug_dir}")
        
        return output_path
    
    def _export_passes_to_csv(self, passes: List, csv_path: str, fps: float, 
                              team_map: dict = None):
        """
        Export passes to CSV with formatted timestamps and team color names.
        
        Args:
            passes: List of PassEvent objects
            csv_path: Path to output CSV file
            fps: Video frames per second
            team_map: Locked team assignments
        """
        def format_timestamp(frame: int, fps: float) -> str:
            """Format frame number to MM:SS timestamp (YouTube style)."""
            if frame is None:
                return "00:00"
            seconds = frame / fps
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        
        def get_team_color_name(team_id: int) -> str:
            """Get team color name from team ID."""
            return "Purple" if team_id == 0 else "Red"
        
        # Prepare data for CSV
        csv_data = []
        for pass_event in passes:
            # Use locked team map if available
            initiator_team = pass_event.team_id
            if team_map and pass_event.from_player_id in team_map:
                initiator_team = team_map[pass_event.from_player_id]
            
            receiver_team = initiator_team  # Same team
            if team_map and pass_event.to_player_id and pass_event.to_player_id in team_map:
                receiver_team = team_map[pass_event.to_player_id]
            
            csv_data.append({
                'Initiated Time': format_timestamp(pass_event.start_frame, fps),
                'Received Time': format_timestamp(pass_event.end_frame, fps),
                'Initiator Team': get_team_color_name(initiator_team),
                'Receiver Team': get_team_color_name(receiver_team),
                'Initiator ID': pass_event.from_player_id,
                'Receiver ID': pass_event.to_player_id,
                'Confidence': f"{pass_event.confidence:.3f}",
                'Distance (m)': f"{pass_event.distance_meters:.1f}",
                'Duration (s)': f"{pass_event.duration_seconds:.2f}",
                'Event ID': pass_event.event_id
            })
        
        # Create DataFrame and save to CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            print(f"✅ Exported {len(csv_data)} passes to {csv_path}")
        else:
            # Create empty CSV with headers
            df = pd.DataFrame(columns=[
                'Initiated Time', 'Received Time', 'Initiator Team', 
                'Receiver Team', 'Initiator ID', 'Receiver ID', 'Confidence',
                'Distance (m)', 'Duration (s)', 'Event ID'
            ])
            df.to_csv(csv_path, index=False)
            print(f"⚠️  No passes detected. Created empty CSV at {csv_path}")



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
