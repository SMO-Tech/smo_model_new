import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

from pipelines import TrackingPipeline, ProcessingPipeline, DetectionPipeline, KeypointPipeline, TacticalPipeline
from constants import model_path, test_video, EMBEDDING_BATCH_SIZE
from keypoint_detection.keypoint_constants import keypoint_model_path
from pass_detection import PlayerOnlyPassDetector
import numpy as np
import time
from tqdm import tqdm
import supervision as sv
from pathlib import Path


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
        
        Flow:
        1. Read video
        2. Detect keypoints and objects (players, referees)
        3. Update with tracking
        4. Assign Teams
        5. Pass Detection (player-only, no ball)
        6. Tactical Analysis
        7. Annotate with passes
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
        
        # Step 4: Initialize pass detector
        print("\n[Step 4/8] Initializing pass detection system...")
        pass_detector = PlayerOnlyPassDetector()
        
        # Step 5: Process all frames with detections, tracking, team assignment, and pass detection
        print("\n[Step 5/8] Processing frames with complete analysis and pass detection...")
        all_tracks = {'player': {}, 'referee': {}, 'player_classids': {}}
        all_passes = []  # Store all detected passes
        player_pitch_positions = {}  # Store player positions in pitch coordinates for pass detection
        
        # Batch process embeddings for faster UMAP transform (batch size = 30 frames for better GPU utilization)
        BATCH_SIZE = 30
        frame_embeddings_buffer = []  # Store (frame_idx, crops, detections) for batching
        
        for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
            
            # Detect keypoints and objects
            keypoints, _ = self.keypoint_pipeline.detect_keypoints_in_frame(frame)
            player_detections, referee_detections = self.detection_pipeline.detect_frame_objects(frame)
            
            # Update with tracking (players only, no ball)
            player_detections = self.tracking_pipeline.tracking_callback(player_detections)

            # Extract crops for team assignment (but batch process UMAP for 23x speedup)
            if len(player_detections.xyxy) > 0:
                crops = self.tracking_pipeline.clustering_manager.embedding_extractor.get_player_crops(frame, player_detections)
                frame_embeddings_buffer.append((i, crops, player_detections, referee_detections, keypoints, frame))
            else:
                # No players, skip team assignment
                frame_embeddings_buffer.append((i, [], player_detections, referee_detections, keypoints, frame))
            
            # Process batch when buffer is full or at end
            if len(frame_embeddings_buffer) >= BATCH_SIZE or i == len(frames) - 1:
                # Extract all embeddings in batch (GPU - fast)
                all_crops = []
                frame_data = []  # Store (frame_idx, num_players, detections, keypoints, frame)
                for frame_idx, crops, p_det, r_det, kp, orig_frame in frame_embeddings_buffer:
                    if len(crops) > 0:
                        all_crops.extend(crops)
                        frame_data.append((frame_idx, len(crops), p_det, r_det, kp, orig_frame))
                
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
                    for frame_idx, num_players, p_det, r_det, kp, orig_frame in frame_data:
                        frame_labels = cluster_labels[label_idx:label_idx+num_players]
                        p_det.class_id = frame_labels
                        label_idx += num_players
                        
                        # Store tracks
                        all_tracks = self.tracking_pipeline.convert_detection_to_tracks(p_det, r_det, all_tracks, frame_idx)
                        
                        # Get player pitch positions for pass detection
                        if len(p_det.xyxy) > 0 and kp is not None:
                            # Transform player positions to pitch coordinates
                            view_transformer = self.tactical_pipeline.transform_keypoints_to_pitch(kp)
                            if view_transformer is not None:
                                # Get center points of bounding boxes
                                bboxes = p_det.xyxy
                                center_points = np.array([
                                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] 
                                    for bbox in bboxes
                                ])
                                
                                # Transform to pitch coordinates
                                pitch_points = self.tactical_pipeline.transform_detections_to_pitch(
                                    p_det, view_transformer
                                )
                                
                                # Store pitch positions for each player
                                for tracker_id, pitch_pos in zip(p_det.tracker_id, pitch_points):
                                    player_pitch_positions[tracker_id] = pitch_pos
                        
                        # Process pass detection
                        if len(p_det.xyxy) > 0 and p_det.tracker_id is not None:
                            # Build player team assignments dict
                            player_team_assignments = {}
                            for tid, team_id in zip(p_det.tracker_id, p_det.class_id):
                                player_team_assignments[tid] = team_id
                            
                            # Get current frame pitch positions
                            current_pitch_positions = {
                                tid: player_pitch_positions.get(tid, np.array([0, 0]))
                                for tid in p_det.tracker_id
                            }
                            
                            # Detect passes
                            detected_passes = pass_detector.process_frame(
                                frame_idx, 
                                {tid: {} for tid in p_det.tracker_id},  # player_tracks (not needed for pass detection)
                                player_team_assignments,
                                current_pitch_positions
                            )
                            all_passes.extend(detected_passes)
                
                # Process frames without players
                for frame_idx, crops, p_det, r_det, kp, orig_frame in frame_embeddings_buffer:
                    if len(crops) == 0:  # No players in this frame
                        all_tracks = self.tracking_pipeline.convert_detection_to_tracks(p_det, r_det, all_tracks, frame_idx)
                
                # Clear buffer
                frame_embeddings_buffer = []

        # Step 6: Save debug outputs
        print("\n[Step 6/8] Saving pass detection debug outputs...")
        debug_dir = Path(video_path).parent / "debug"
        pass_detector.save_debug_outputs(debug_dir)
        
        # Step 7: Annotate frames with passes
        print("\n[Step 7/8] Annotating frames with detections and passes...")
        object_annotated_frames = self.tracking_pipeline.annotate_frames(frames, all_tracks)
        
        # Add pass visualization to frames
        output_frames = []
        for i, frame in enumerate(object_annotated_frames):
            # Get passes that should be visible in this frame
            visible_passes = [p for p in all_passes if p.start_frame <= i <= p.end_frame + 90]  # Show for 90 frames after end
            
            if visible_passes:
                # Get player positions in frame coordinates for this frame
                if i in all_tracks['player']:
                    player_frame_positions = {}
                    for player_id, bbox in all_tracks['player'][i].items():
                        if bbox and bbox[0] is not None:
                            # Center of bounding box
                            center_x = (bbox[0] + bbox[2]) / 2
                            center_y = (bbox[1] + bbox[3]) / 2
                            player_frame_positions[player_id] = np.array([center_x, center_y])
                    
                    # Annotate passes
                    frame = self.tracking_pipeline.annotator_manager.annotate_passes(
                        frame, visible_passes, player_frame_positions
                    )
            
            output_frames.append(frame)

        # Step 8: Write final output video
        print("\n[Step 8/8] Writing complete analysis video...")
        output_path = self.processing_pipeline.generate_output_path(video_path, output_suffix)
        self.processing_pipeline.write_video_output(output_frames, output_path)
        
        # Summary
        total_time = time.time() - total_start_time
        print(f"\n=== Complete Soccer Analysis Finished ===")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Frames processed: {len(frames)}")
        print(f"Average time per frame: {total_time/len(frames):.3f}s")
        print(f"Passes detected: {len(all_passes)}")
        print(f"Output saved to: {output_path}")
        print(f"Debug outputs saved to: {debug_dir}")
        
        return output_path



if __name__ == "__main__":
    # Run Complete End-to-End Soccer Analysis Pipeline
    print("Starting Soccer Analysis...")
    pipeline = CompleteSoccerAnalysisPipeline(model_path, keypoint_model_path)
    # Process first 40 seconds (1200 frames at 30fps)
    MAX_FRAMES = 1200
    output_video = pipeline.analyze_video(test_video, frame_count=MAX_FRAMES)    
    print(f"\nAnalysis finished! Output video: {output_video}")