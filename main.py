import sys
from pathlib import Path
from typing import List
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

from pipelines import TrackingPipeline, ProcessingPipeline, DetectionPipeline, KeypointPipeline, TacticalPipeline
from constants import model_path, test_video, EMBEDDING_BATCH_SIZE
from keypoint_detection.keypoint_constants import keypoint_model_path
from pass_detection.simple_pass_detector import SimplePassDetector
import numpy as np
import time
from tqdm import tqdm
import supervision as sv
import cv2
import csv


class CompleteSoccerAnalysisPipeline:
    """Complete end-to-end soccer analysis pipeline integrating all functionalities."""
    
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
        self.pass_detector = None  # Will be initialized with video FPS
        self.video_fps = 30.0  # Default, will be updated from video
        
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
        """Run complete end-to-end soccer analysis.
        
        Flow:
        1. Read video
        2. Detect keypoints and objects (players, ball, referees)
        3. Update with tracking
        4. Tactical Analysis
        5. Interpolate ball tracks
        6. Assign Teams
        7. Tactical Overlay
        8. Save Video
        
        Args:
            video_path: Path to input video
            frame_count: Number of frames to process (-1 for all)
            output_suffix: Suffix for output video file
            
        Returns:
            Path to output video
        """
        print("=== Starting Complete Soccer Analysis Pipeline ===")
        total_start_time = time.time()
        
        # Step 1: Initialize all models
        self.initialize_models()
        
        # Step 2: Train team assignment models
        print("\n[Step 2/8] Training team assignment models...")
        self.tracking_pipeline.train_team_assignment_models(video_path)
        
        # Step 3: Read video frames and get FPS
        print("\n[Step 3/8] Reading video frames...")
        frames = self.processing_pipeline.read_video_frames(video_path, frame_count)
        print(f"Loaded {len(frames)} frames for processing")
        
        # Get video FPS
        cap = cv2.VideoCapture(video_path)
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        print(f"Video FPS: {self.video_fps:.2f}")
        
        # Initialize pass detector with video FPS
        pass_config = {'fps': self.video_fps}
        self.pass_detector = SimplePassDetector(pass_config)
        
        # Initialize goal positions for shot detection (estimate from video dimensions)
        if len(frames) > 0:
            frame_height, frame_width = frames[0].shape[:2]
            self.pass_detector.initialize_goal_positions(frame_width, frame_height)
        
        # Step 4: Process all frames with detections, tracking, and tactical analysis
        print("\n[Step 4/8] Processing frames with complete analysis...")
        tactical_frames = []
        all_tracks = {'player': {}, 'ball': {}, 'referee': {}, 'player_classids': {}}
        
        # Batch process embeddings for faster UMAP transform (batch size = 30 frames for better GPU utilization)
        BATCH_SIZE = 30
        frame_embeddings_buffer = []  # Store (frame_idx, crops, detections) for batching
        
        for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
            
            # Detect keypoints and objects
            keypoints, _ = self.keypoint_pipeline.detect_keypoints_in_frame(frame)
            player_detections, ball_detections, referee_detections = self.detection_pipeline.detect_frame_objects(frame)
            
            # Update with tracking (both players and ball)
            player_detections = self.tracking_pipeline.tracking_callback(player_detections)
            ball_detections = self.tracking_pipeline.ball_tracking_callback(ball_detections)

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
                frame_data = []  # Store (frame_idx, num_players, detections, keypoints, frame)
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
                        
                        # Process pass detection for this frame
                        self._process_pass_detection(frame_idx, p_det, b_det)
                        
                        # Store tracks for interpolation
                        all_tracks = self.tracking_pipeline.convert_detection_to_tracks(p_det, b_det, r_det, all_tracks, frame_idx)
                        
                        # Get tactical frame from detections
                        tactical_frame, _ = self.tactical_pipeline.process_detections_for_tactical_analysis(p_det, r_det, kp)
                        tactical_frames.append(tactical_frame)
                
                # Process frames without players
                for frame_idx, crops, p_det, b_det, r_det, kp, orig_frame in frame_embeddings_buffer:
                    if len(crops) == 0:  # No players in this frame
                        # Process pass detection (even with no players, ball might be present)
                        self._process_pass_detection(frame_idx, p_det, b_det)
                        
                        all_tracks = self.tracking_pipeline.convert_detection_to_tracks(p_det, b_det, r_det, all_tracks, frame_idx)
                        tactical_frame, _ = self.tactical_pipeline.process_detections_for_tactical_analysis(p_det, r_det, kp)
                        tactical_frames.append(tactical_frame)
                
                # Clear buffer
                frame_embeddings_buffer = []

        # Step 5: Ball track interpolation
        print("\n[Step 5/8] Interpolating ball tracks...")
        all_tracks = self.processing_pipeline.interpolate_ball_tracks(all_tracks)
        
        # Step 6: Player Annotation
        print("\n[Step 6/8] Assigning teams and Annotating frames with detections...")
        object_annotated_frames = self.tracking_pipeline.annotate_frames(frames, all_tracks)

        # Step 7: Skip overlay - use annotated frames directly (no tactical overlay)
        print("\n[Step 7/8] Preparing output frames (no overlay)...")
        output_frames = object_annotated_frames  # Use annotated frames directly without tactical overlay

        # Step 8: Write final output video
        print("\n[Step 8/8] Writing complete analysis video...")
        output_path = self.processing_pipeline.generate_output_path(video_path, output_suffix)
        self.processing_pipeline.write_video_output(output_frames, output_path, fps=self.video_fps)
        
        # Step 9: Analyze pass gaps and detect missing passes
        if self.pass_detector:
            print("\n[Step 9/10] Analyzing pass gaps...")
            passes = self.pass_detector.get_confirmed_passes()
            
            # Print pass detection metrics
            if hasattr(self.pass_detector, 'metrics'):
                metrics = self.pass_detector.metrics
                print(f"\n📊 Pass Detection Metrics:")
                print(f"   Total possessions detected: {metrics.get('total_possessions', 0)}")
                print(f"   Passes detected: {metrics.get('passes_detected', 0)}")
                print(f"   Rejected - distance too short: {metrics.get('passes_rejected_distance', 0)}")
                print(f"   Rejected - cooldown: {metrics.get('passes_rejected_cooldown', 0)}")
                print(f"   Rejected - same player: {metrics.get('passes_rejected_same_player', 0)}")
                print(f"   Rejected - trajectory: {metrics.get('passes_rejected_trajectory', 0)}")
                print(f"   Rejected - duration: {metrics.get('passes_rejected_duration', 0)}")
                print(f"   Near misses (radius): {metrics.get('near_miss_radius', 0)}")
                print(f"   Near misses (duration): {metrics.get('near_miss_duration', 0)}")
            
            gaps = self._analyze_pass_gaps(passes, self.video_fps, max_gap_seconds=8)
            if gaps:
                print(f"⚠️  Found {len(gaps)} suspicious gaps (>8 seconds)")
                for gap in gaps:
                    end_time_str = f"{gap['end_time']:.1f}s" if gap['end_time'] is not None else "end"
                    print(f"   Gap: {gap['gap_seconds']:.1f}s from {gap['start_time']:.1f}s to {end_time_str}")
            else:
                print("✓ No suspicious gaps found")
        
        # Step 10: Export passes and shots to CSV
        if self.pass_detector:
            print("\n[Step 10/10] Exporting passes and shots to CSV...")
            # Get passes first (this will trigger validation if needed)
            passes = self.pass_detector.get_confirmed_passes()
            # Get shots (validation already done, so won't duplicate)
            shots = self.pass_detector.get_detected_shots()
            csv_path = self._export_passes_and_shots_to_csv(passes, shots, video_path, output_suffix)
            print(f"Exported {len(passes)} passes and {len(shots)} shots to: {csv_path}")
            
            # Display stats summary
            print("\n" + "=" * 60)
            print("📊 SHOTS & PASSES STATISTICS")
            print("=" * 60)
            self._display_shots_and_passes_stats(passes, shots)
        
        # Summary
        total_time = time.time() - total_start_time
        print(f"\n=== Complete Soccer Analysis Finished ===")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Frames processed: {len(frames)}")
        print(f"Average time per frame: {total_time/len(frames):.3f}s")
        print(f"Output saved to: {output_path}")
        if self.pass_detector:
            print(f"Passes detected: {len(passes)}")
        
        return output_path
    
    def _process_pass_detection(self, frame_idx: int, player_detections: sv.Detections, ball_detections: sv.Detections):
        """
        Process pass detection for a single frame.
        
        Args:
            frame_idx: Current frame index
            player_detections: Player detections with tracker IDs and team IDs (class_id)
            ball_detections: Ball detections
        """
        if self.pass_detector is None:
            return
        
        # Extract player positions (bbox centers) and team IDs
        player_positions = {}
        player_teams = {}
        
        if player_detections is not None and len(player_detections.xyxy) > 0:
            for i in range(len(player_detections.xyxy)):
                bbox = player_detections.xyxy[i]
                tracker_id = int(player_detections.tracker_id[i]) if player_detections.tracker_id is not None else i
                team_id = int(player_detections.class_id[i]) if player_detections.class_id is not None else 0
                
                # Calculate center of bbox (pixel coordinates)
                center_x = (bbox[0] + bbox[2]) / 2.0
                center_y = (bbox[1] + bbox[3]) / 2.0
                
                player_positions[tracker_id] = np.array([center_x, center_y], dtype=np.float32)
                player_teams[tracker_id] = team_id
        
        # Extract ball position (bbox center) if available
        ball_pos = None
        if ball_detections is not None and len(ball_detections.xyxy) > 0:
            # Use first ball detection (or closest to prediction if multiple)
            bbox = ball_detections.xyxy[0]
            center_x = (bbox[0] + bbox[2]) / 2.0
            center_y = (bbox[1] + bbox[3]) / 2.0
            ball_pos = np.array([[center_x, center_y]], dtype=np.float32)
        
        # Process frame through pass detector
        self.pass_detector.process_frame(
            frame=frame_idx,
            player_positions=player_positions,
            player_teams=player_teams,
            ball_detections=ball_pos
        )
    
    def _export_passes_to_csv(self, passes, video_path: str, suffix: str = "_complete_analysis"):
        """
        Export passes to CSV file with time, player IDs, and team colors.
        
        Args:
            passes: List of PassEvent objects
            video_path: Path to input video (for generating output path)
            suffix: Suffix for output filename
            
        Returns:
            Path to CSV file
        """
        # Generate CSV path
        csv_path = video_path.replace(".mp4", f"{suffix}_passes.csv")
        
        # Team color mapping (0=Purple, 1=Red)
        team_colors = {
            0: "Purple",
            1: "Red"
        }
        
        # Write CSV
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'pass_id',
                'time_initiated',
                'time_received',
                'time_initiated_seconds',
                'time_received_seconds',
                'initiator_player_id',
                'receiver_player_id',
                'initiator_team_color',
                'receiver_team_color',
                'initiator_team_id',
                'receiver_team_id',
                'duration_seconds',
                'distance_pixels'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for pass_event in passes:
                if not pass_event.is_confirmed or pass_event.to_player_id is None:
                    continue
                
                # Convert frame numbers to time in seconds
                time_initiated = pass_event.start_frame / self.video_fps
                time_received = pass_event.end_frame / self.video_fps if pass_event.end_frame else None
                
                # Format as video timestamp (MM:SS.mmm)
                def format_video_timestamp(seconds):
                    """Convert seconds to MM:SS.mmm format."""
                    if seconds is None:
                        return ""
                    minutes = int(seconds // 60)
                    secs = seconds % 60
                    return f"{minutes:02d}:{secs:06.3f}"
                
                time_initiated_formatted = format_video_timestamp(time_initiated)
                time_received_formatted = format_video_timestamp(time_received)
                
                # Get team IDs for initiator and receiver
                initiator_team_id = pass_event.team_id
                # Use actual receiver team (may differ for interceptions)
                receiver_team_id = pass_event.receiver_team_id if pass_event.receiver_team_id is not None else initiator_team_id
                
                writer.writerow({
                    'pass_id': pass_event.event_id,
                    'time_initiated': time_initiated_formatted,
                    'time_received': time_received_formatted,
                    'time_initiated_seconds': f"{time_initiated:.3f}",
                    'time_received_seconds': f"{time_received:.3f}" if time_received else "",
                    'initiator_player_id': pass_event.from_player_id,
                    'receiver_player_id': pass_event.to_player_id,
                    'initiator_team_color': team_colors.get(initiator_team_id, "Unknown"),
                    'receiver_team_color': team_colors.get(receiver_team_id, "Unknown"),
                    'initiator_team_id': initiator_team_id,
                    'receiver_team_id': receiver_team_id,
                    'duration_seconds': f"{pass_event.duration_seconds:.3f}",
                    'distance_pixels': f"{pass_event.distance_meters:.2f}"
                })
        
        return csv_path
    
    def _export_passes_and_shots_to_csv(self, passes, shots, video_path: str, suffix: str = "_complete_analysis"):
        """
        Export passes and shots to CSV file.
        
        Args:
            passes: List of PassEvent objects
            shots: List of ShotEvent objects
            video_path: Path to input video (for generating output path)
            suffix: Suffix for output filename
            
        Returns:
            Path to CSV file
        """
        # Generate CSV path
        csv_path = video_path.replace(".mp4", f"{suffix}_events.csv")
        
        # Team color mapping (0=Purple, 1=Red)
        team_colors = {
            0: "Purple",
            1: "Red"
        }
        
        # Write CSV
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'event_type',  # 'pass' or 'shot'
                'event_id',
                'time_start',
                'time_end',
                'time_start_seconds',
                'time_end_seconds',
                'player_id',  # Passer/Shooter
                'receiver_player_id',  # Only for passes
                'team_color',
                'team_id',
                'receiver_team_id',  # Only for passes
                'pass_outcome',  # Only for passes: 'success' or 'interception'
                'shot_type',  # Only for shots: 'shot_on_target', 'shot_off_target', 'shot_blocked'
                'duration_seconds',
                'distance_pixels',
                'speed_pixels_per_second',
                'confidence'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write passes
            for pass_event in passes:
                if not pass_event.is_confirmed or pass_event.to_player_id is None:
                    continue
                
                # Determine pass outcome
                if pass_event.team_id == pass_event.receiver_team_id:
                    outcome = "success"
                else:
                    outcome = "interception"
                
                writer.writerow({
                    'event_type': 'pass',
                    'event_id': pass_event.event_id,
                    'time_start': f"{pass_event.start_frame / 30.0:.2f}s",
                    'time_end': f"{pass_event.end_frame / 30.0:.2f}s" if pass_event.end_frame else "",
                    'time_start_seconds': pass_event.start_frame / 30.0,
                    'time_end_seconds': pass_event.end_frame / 30.0 if pass_event.end_frame else 0.0,
                    'player_id': pass_event.from_player_id,
                    'receiver_player_id': pass_event.to_player_id,
                    'team_color': team_colors.get(pass_event.team_id, "Unknown"),
                    'team_id': pass_event.team_id,
                    'receiver_team_id': pass_event.receiver_team_id,
                    'pass_outcome': outcome,
                    'shot_type': '',
                    'duration_seconds': pass_event.duration_seconds,
                    'distance_pixels': pass_event.distance_meters,  # Actually pixels
                    'speed_pixels_per_second': pass_event.implied_speed,
                    'confidence': pass_event.confidence
                })
            
            # Write shots (avoid duplicates by tracking written IDs)
            written_shot_ids = set()
            for shot_event in shots:
                # Skip duplicates
                if shot_event.event_id in written_shot_ids:
                    continue
                written_shot_ids.add(shot_event.event_id)
                
                writer.writerow({
                    'event_type': 'shot',
                    'event_id': shot_event.event_id,
                    'time_start': f"{shot_event.start_time:.2f}s",
                    'time_end': f"{shot_event.end_time:.2f}s",
                    'time_start_seconds': shot_event.start_time,
                    'time_end_seconds': shot_event.end_time,
                    'player_id': shot_event.shooter_id,
                    'receiver_player_id': '',  # Shots have no receiver
                    'team_color': team_colors.get(shot_event.team_id, "Unknown"),
                    'team_id': shot_event.team_id,
                    'receiver_team_id': '',
                    'pass_outcome': '',
                    'shot_type': shot_event.shot_type.value,
                    'duration_seconds': shot_event.duration_seconds,
                    'distance_pixels': shot_event.distance_pixels,
                    'speed_pixels_per_second': shot_event.speed_pixels_per_second,
                    'confidence': shot_event.confidence
                })
        
        return csv_path
    
    def _analyze_pass_gaps(self, passes, fps: float, max_gap_seconds: float = 8.0):
        """
        Identify suspicious gaps where passes are missing.
        
        Args:
            passes: List of PassEvent objects
            fps: Video frames per second
            max_gap_seconds: Maximum acceptable gap in seconds
            
        Returns:
            List of gap dictionaries with details
        """
        if len(passes) < 2:
            # Not enough passes to analyze gaps
            if len(passes) == 0:
                return [{'start_frame': 0, 'end_frame': None, 'gap_seconds': float('inf'), 
                        'start_time': 0.0, 'end_time': None, 'suspicious': True}]
            return []
        
        # Sort passes by start frame
        passes_sorted = sorted(passes, key=lambda x: x.start_frame)
        
        gaps = []
        for i in range(len(passes_sorted) - 1):
            current_pass = passes_sorted[i]
            next_pass = passes_sorted[i + 1]
            
            # Gap is from end of current pass to start of next pass
            gap_start_frame = current_pass.end_frame if current_pass.end_frame else current_pass.start_frame
            gap_end_frame = next_pass.start_frame
            gap_frames = gap_end_frame - gap_start_frame
            gap_seconds = gap_frames / fps
            
            if gap_seconds > max_gap_seconds:
                gaps.append({
                    'start_frame': gap_start_frame,
                    'end_frame': gap_end_frame,
                    'gap_frames': gap_frames,
                    'gap_seconds': gap_seconds,
                    'start_time': gap_start_frame / fps,
                    'end_time': gap_end_frame / fps,
                    'suspicious': True,
                    'previous_pass': current_pass.event_id,
                    'next_pass': next_pass.event_id
                })
        
        return gaps
    
    def _display_shots_and_passes_stats(self, passes: List, shots: List):
        """
        Display formatted statistics for shots and passes by team.
        
        Args:
            passes: List of PassEvent objects
            shots: List of ShotEvent objects
        """
        from pass_detection.shot_detector import ShotType
        
        # Calculate stats by team
        team_0_passes = [p for p in passes if p.team_id == 0]
        team_1_passes = [p for p in passes if p.team_id == 1]
        
        team_0_shots = [s for s in shots if s.team_id == 0]
        team_1_shots = [s for s in shots if s.team_id == 1]
        
        # Pass stats
        team_0_successful_passes = len([p for p in team_0_passes if p.receiver_team_id == p.team_id])
        team_0_interceptions = len([p for p in team_0_passes if p.receiver_team_id != p.team_id])
        team_1_successful_passes = len([p for p in team_1_passes if p.receiver_team_id == p.team_id])
        team_1_interceptions = len([p for p in team_1_passes if p.receiver_team_id != p.team_id])
        
        # Shot stats
        team_0_total_shots = len(team_0_shots)
        team_0_shots_on_target = len([s for s in team_0_shots if s.shot_type == ShotType.SHOT_ON_TARGET])
        team_0_shots_off_target = len([s for s in team_0_shots if s.shot_type == ShotType.SHOT_OFF_TARGET])
        team_0_shots_blocked = len([s for s in team_0_shots if s.shot_type == ShotType.SHOT_BLOCKED])
        
        team_1_total_shots = len(team_1_shots)
        team_1_shots_on_target = len([s for s in team_1_shots if s.shot_type == ShotType.SHOT_ON_TARGET])
        team_1_shots_off_target = len([s for s in team_1_shots if s.shot_type == ShotType.SHOT_OFF_TARGET])
        team_1_shots_blocked = len([s for s in team_1_shots if s.shot_type == ShotType.SHOT_BLOCKED])
        
        # Calculate pass accuracy
        team_0_pass_accuracy = (team_0_successful_passes / len(team_0_passes) * 100) if len(team_0_passes) > 0 else 0.0
        team_1_pass_accuracy = (team_1_successful_passes / len(team_1_passes) * 100) if len(team_1_passes) > 0 else 0.0
        
        # Display formatted stats
        print(f"\n{'SHOTS':<20} {'Team 0 (Purple)':<20} {'Team 1 (Red)':<20}")
        print("-" * 60)
        print(f"{'Total Shots':<20} {team_0_total_shots:<20} {team_1_total_shots:<20}")
        print(f"{'Shots On Target':<20} {team_0_shots_on_target:<20} {team_1_shots_on_target:<20}")
        print(f"{'Shots Off Target':<20} {team_0_shots_off_target:<20} {team_1_shots_off_target:<20}")
        print(f"{'Shots Blocked':<20} {team_0_shots_blocked:<20} {team_1_shots_blocked:<20}")
        
        print(f"\n{'PASSES':<20} {'Team 0 (Purple)':<20} {'Team 1 (Red)':<20}")
        print("-" * 60)
        print(f"{'Total Passes':<20} {len(team_0_passes):<20} {len(team_1_passes):<20}")
        print(f"{'Successful':<20} {team_0_successful_passes:<20} {team_1_successful_passes:<20}")
        print(f"{'Intercepted':<20} {team_0_interceptions:<20} {team_1_interceptions:<20}")
        print(f"{'Pass Accuracy':<20} {team_0_pass_accuracy:.1f}%{'':<15} {team_1_pass_accuracy:.1f}%")
        
        print(f"\n{'SUMMARY':<20} {'Team 0 (Purple)':<20} {'Team 1 (Red)':<20}")
        print("-" * 60)
        print(f"{'Total Events':<20} {len(team_0_passes) + team_0_total_shots:<20} {len(team_1_passes) + team_1_total_shots:<20}")
        
        print("=" * 60)


if __name__ == "__main__":
    # Run Complete End-to-End Soccer Analysis Pipeline
    print("Starting Soccer Analysis...")
    pipeline = CompleteSoccerAnalysisPipeline(model_path, keypoint_model_path)
    
    # Use the Drogba goal video
    video_path = "/home/essashah/SWE/soccer_video_2k.mp4"

    
    # Process all frames in the trimmed video
    output_video = pipeline.analyze_video(video_path, frame_count=-1)    
    print(f"\nAnalysis finished! Output video: {output_video}")