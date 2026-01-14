import sys
from pathlib import Path
from typing import List
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

from pipelines import TrackingPipeline, ProcessingPipeline, DetectionPipeline, KeypointPipeline, TacticalPipeline
from constants import model_path, test_video, EMBEDDING_BATCH_SIZE
from keypoint_detection.keypoint_constants import keypoint_model_path
from pass_detection.event_detector import EventDetector, AllEvents
import numpy as np
import time
from tqdm import tqdm
import supervision as sv
import cv2
import csv
import torch  # For GPU memory management


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
        self.event_detector = None  # Will be initialized with video FPS
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
        
        # Step 3: Get video info and determine processing mode
        print("\n[Step 3/8] Analyzing video...")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        
        if frame_count == -1:
            frames_to_process = total_frames
        else:
            frames_to_process = min(frame_count, total_frames)
        
        print(f"Video info: {total_frames} total frames, {frames_to_process} to process")
        print(f"Resolution: {frame_width}x{frame_height}, FPS: {self.video_fps:.2f}")
        
        # Use chunked processing for videos with >10000 frames to avoid OOM
        USE_CHUNKED = frames_to_process > 10000
        # Calculate optimal chunk size based on video resolution
        # For 2048x2048 video: ~12MB per frame, so 500 frames = ~6GB (safe for 43GB available)
        # For smaller videos, can use larger chunks
        if frame_width * frame_height > 2000000:  # High resolution (>~1400x1400)
            CHUNK_SIZE = 250  # Reduced to 250 frames (~3GB per chunk) to prevent OOM
        elif frames_to_process > 50000:
            CHUNK_SIZE = 500  # Reduced chunk size for medium resolution
        else:
            CHUNK_SIZE = 1000  # For smaller videos
        print(f"Using chunk size: {CHUNK_SIZE} frames (~{CHUNK_SIZE * frame_width * frame_height * 3 / 1024 / 1024 / 1024:.1f} GB per chunk)")
        
        if USE_CHUNKED:
            print(f"⚠️  Large video detected - using chunked processing ({CHUNK_SIZE} frames per chunk)")
        
        # Initialize event detector with video FPS
        event_config = {'fps': self.video_fps}
        self.event_detector = EventDetector(event_config)
        
        # Initialize goal positions (use first frame dimensions)
        self.event_detector.initialize_goal_positions(frame_width, frame_height)
        
        # Step 4: Process all frames with detections, tracking, and tactical analysis
        print("\n[Step 4/8] Processing frames with complete analysis...")
        tactical_frames = []
        all_tracks = {'player': {}, 'ball': {}, 'referee': {}, 'player_classids': {}, 'ball_tracker_ids': {}}
        
        # GPU-optimized batch processing settings - MAXIMIZE GPU UTILIZATION
        # Increased batch sizes to better utilize GPU (Tesla T4 has 15GB VRAM)
        # These are optimized for maximum GPU throughput
        DETECTION_BATCH_SIZE = 64  # Increased to 64 - maximize GPU utilization
        KEYPOINT_BATCH_SIZE = 64   # Increased to 64 - maximize GPU utilization  
        EMBEDDING_FRAME_BATCH_SIZE = 100   # Increased to 100 - batch more frames for embeddings
        
        # Initialize models on GPU early to maximize utilization
        print("Initializing models on GPU...")
        self.detection_pipeline.initialize_model()
        self.keypoint_pipeline.initialize_model()
        
        # Clear GPU cache before starting batch processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"✅ GPU ready: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"   Batch sizes: Detection={DETECTION_BATCH_SIZE}, Keypoint={KEYPOINT_BATCH_SIZE}, Embedding={EMBEDDING_FRAME_BATCH_SIZE}")
            print(f"   Embedding batch size: {EMBEDDING_BATCH_SIZE}")
            print(f"🚀 GPU Optimization:")
            print(f"   - Detection: GPU (batch size {DETECTION_BATCH_SIZE})")
            print(f"   - Keypoints: GPU (batch size {KEYPOINT_BATCH_SIZE})")
            print(f"   - Embeddings: GPU (batch size {EMBEDDING_BATCH_SIZE})")
            print(f"   - UMAP/K-means: CPU (cuML not installed - install for GPU clustering)")
            print(f"   ⚠️  ByteTrack tracking: Sequential CPU (algorithm limitation)")
            print(f"   💡 All GPU-accelerated operations are maximized for throughput")
        
        # Process video in chunks if it's large, otherwise load all frames
        if USE_CHUNKED:
            # Process video in chunks to avoid OOM
            chunk_generator = self.processing_pipeline.read_video_frames(video_path, frame_count, chunk_size=CHUNK_SIZE)
            all_frames_processed = 0
            
            for chunk_idx, (frames, chunk_start, chunk_end) in enumerate(chunk_generator):
                print(f"\n📦 Processing chunk {chunk_idx + 1}: frames {chunk_start}-{chunk_end} ({len(frames)} frames)")
                all_frames_processed += len(frames)
                
                # Process this chunk (same logic as before but for chunk)
                self._process_chunk(frames, chunk_start, tactical_frames, all_tracks, 
                                  DETECTION_BATCH_SIZE, EMBEDDING_FRAME_BATCH_SIZE)
                
                # Clear memory after each chunk - AGGRESSIVE cleanup
                del frames
                import gc
                gc.collect()  # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Periodically clear tactical_frames to prevent memory accumulation
                # (We re-read video for annotation anyway, so we don't need to keep all tactical frames)
                if len(tactical_frames) > 5000:  # Keep only last 5000 tactical frames
                    tactical_frames = tactical_frames[-5000:]
                    print(f"  💾 Cleared old tactical frames (keeping last 5000)")
                
                print(f"✅ Chunk {chunk_idx + 1} completed ({all_frames_processed}/{frames_to_process} frames processed)")
        else:
            # Small video - load all frames at once
            print("Loading all frames into memory...")
            frames = self.processing_pipeline.read_video_frames(video_path, frame_count)
            print(f"Loaded {len(frames)} frames for processing")
            self._process_chunk(frames, 0, tactical_frames, all_tracks, 
                              DETECTION_BATCH_SIZE, EMBEDDING_FRAME_BATCH_SIZE)
        
        # Step 5: Ball track interpolation
        print("\n[Step 5/8] Interpolating ball tracks...")
        all_tracks = self.processing_pipeline.interpolate_ball_tracks(all_tracks)
        
        # Step 6: Player Annotation
        print("\n[Step 6/8] Assigning teams and Annotating frames with detections...")
        if USE_CHUNKED:
            # For chunked processing, re-read video in chunks for annotation
            object_annotated_frames = []
            chunk_generator = self.processing_pipeline.read_video_frames(video_path, frame_count, chunk_size=CHUNK_SIZE)
            for chunk_idx, (frames_chunk, chunk_start, chunk_end) in enumerate(chunk_generator):
                print(f"Annotating chunk {chunk_idx + 1}: frames {chunk_start}-{chunk_end}")
                chunk_annotated = self.tracking_pipeline.annotate_frames(frames_chunk, all_tracks, frame_offset=chunk_start)
                object_annotated_frames.extend(chunk_annotated)
                del frames_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            object_annotated_frames = self.tracking_pipeline.annotate_frames(frames, all_tracks)

        # Step 7: Skip overlay - use annotated frames directly (no tactical overlay)
        print("\n[Step 7/8] Preparing output frames (no overlay)...")
        output_frames = object_annotated_frames  # Use annotated frames directly without tactical overlay

        # Step 8: Write final output video
        print("\n[Step 8/8] Writing complete analysis video...")
        output_path = self.processing_pipeline.generate_output_path(video_path, output_suffix)
        self.processing_pipeline.write_video_output(output_frames, output_path, fps=self.video_fps)
    
    def _process_chunk(self, frames, frame_offset, tactical_frames, all_tracks, 
                      DETECTION_BATCH_SIZE, EMBEDDING_FRAME_BATCH_SIZE):
        """Process a chunk of frames with detections, tracking, and events."""
        
        # Batch process frames for detection and keypoints (GPU-intensive operations)
        detection_results = []  # Store (frame_idx, player_detections, ball_detections, referee_detections)
        keypoint_results = []   # Store (frame_idx, keypoints, metadata)
        
        print("Batch processing detections and keypoints on GPU...")
        for batch_start in tqdm(range(0, len(frames), DETECTION_BATCH_SIZE), desc="Batch inference"):
            batch_end = min(batch_start + DETECTION_BATCH_SIZE, len(frames))
            batch_frames = frames[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            
            # Batch detection on GPU (much faster than frame-by-frame)
            try:
                from player_detection.detect_players import detect_objects_in_frames, get_device
                device = get_device()
                detection_model = self.detection_pipeline.model
                
                # Run batch inference
                batch_results = detect_objects_in_frames(detection_model, batch_frames, device=device, conf=0.15)
                
                # Process each result in the batch
                for idx, (local_idx, result) in enumerate(zip(batch_indices, batch_results)):
                    try:
                        frame = batch_frames[idx]
                        frame_idx = frame_offset + local_idx  # Adjust for chunk offset
                        # Convert to supervision format
                        detections = sv.Detections.from_ultralytics(result)
                        # Separate by class
                        player_detections = detections[detections.class_id == 0]
                        ball_detections = detections[detections.class_id == 1]
                        referee_detections = detections[detections.class_id == 2]
                    except Exception as e:
                        print(f"⚠️  Error processing detection for frame {frame_offset + local_idx}: {e}")
                        frame_idx = frame_offset + local_idx
                        player_detections = sv.Detections.empty()
                        ball_detections = sv.Detections.empty()
                        referee_detections = sv.Detections.empty()
                    
                    # Apply ball filtering (same as get_detections)
                    try:
                        if len(ball_detections.xyxy) > 0:
                            frame_h, frame_w = frame.shape[0], frame.shape[1]
                            frame_area = frame_w * frame_h
                            
                            player_centers = []
                            if len(player_detections.xyxy) > 0:
                                for player_bbox in player_detections.xyxy:
                                    player_center = np.array([
                                        (player_bbox[0] + player_bbox[2]) / 2,
                                        (player_bbox[1] + player_bbox[3]) / 2
                                    ])
                                    player_centers.append(player_center)
                            
                            valid_indices = []
                            for i in range(len(ball_detections.xyxy)):
                                bbox = ball_detections.xyxy[i]
                                conf = ball_detections.confidence[i] if ball_detections.confidence is not None and len(ball_detections.confidence) > i else 0.5
                                
                                bbox_w = bbox[2] - bbox[0]
                                bbox_h = bbox[3] - bbox[1]
                                bbox_area = bbox_w * bbox_h
                                bbox_area_ratio = bbox_area / frame_area
                                
                                ball_center = np.array([
                                    (bbox[0] + bbox[2]) / 2,
                                    (bbox[1] + bbox[3]) / 2
                                ])
                                
                                size_valid = 0.0000005 <= bbox_area_ratio <= 0.03
                                conf_valid = conf >= 0.15
                                
                                player_proximity_valid = False
                                if len(player_centers) > 0:
                                    min_player_distance = min([np.linalg.norm(ball_center - pc) for pc in player_centers])
                                    player_proximity_valid = min_player_distance <= 400.0 or conf >= 0.4
                                else:
                                    player_proximity_valid = conf >= 0.4
                                
                                if size_valid and conf_valid and player_proximity_valid:
                                    valid_indices.append(i)
                            
                            if len(valid_indices) > 0:
                                ball_detections = ball_detections[valid_indices]
                            else:
                                ball_detections = sv.Detections.empty()
                    except Exception as e:
                        print(f"⚠️  Error filtering ball detections for frame {frame_idx}: {e}")
                        ball_detections = sv.Detections.empty()
                    
                    detection_results.append((frame_idx, player_detections, ball_detections, referee_detections))
                
                # Clear GPU cache periodically to prevent OOM and maximize throughput
                if device != 'cpu':
                    torch.cuda.empty_cache()
                    # Sync to ensure operations complete before next batch
                    torch.cuda.synchronize()
                    
            except Exception as e:
                print(f"⚠️  Batch detection failed, falling back to frame-by-frame: {e}")
                # Fallback to frame-by-frame
                for idx, local_idx in enumerate(batch_indices):
                    frame = batch_frames[idx]
                    frame_idx = frame_offset + local_idx
                    player_detections, ball_detections, referee_detections = self.detection_pipeline.detect_frame_objects(frame)
                    detection_results.append((frame_idx, player_detections, ball_detections, referee_detections))
            
            # Batch keypoint detection on GPU
            try:
                from keypoint_detection.detect_keypoints import detect_keypoints_in_frames
                device = get_device()
                keypoint_model = self.keypoint_pipeline.model
                
                # Run batch inference on GPU
                batch_keypoint_results = detect_keypoints_in_frames(keypoint_model, batch_frames, device=device)
                
                # Clear GPU cache after keypoint inference
                if device != 'cpu':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Process each result - extract keypoints from YOLO pose results
                for idx, (local_idx, result) in enumerate(zip(batch_indices, batch_keypoint_results)):
                    try:
                        frame_idx = frame_offset + local_idx  # Adjust for chunk offset
                        # Extract keypoints from YOLO pose result
                        if hasattr(result, 'keypoints') and result.keypoints is not None:
                            # YOLO pose returns keypoints in shape (num_people, num_keypoints, 3) where 3 = (x, y, confidence)
                            kp_data = result.keypoints.data
                            if hasattr(kp_data, 'cpu'):
                                keypoints = kp_data.cpu().numpy()
                            else:
                                keypoints = kp_data
                            # Ensure shape is (N, 29, 3) or convert if needed
                            if len(keypoints.shape) == 2 and keypoints.shape[1] == 87:  # Flattened format
                                keypoints = keypoints.reshape(-1, 29, 3)
                        else:
                            # Fallback: empty keypoints
                            keypoints = np.empty((0, 29, 3))
                        keypoint_results.append((frame_idx, keypoints, {}))
                    except Exception as e:
                        print(f"⚠️  Error processing keypoints for frame {frame_offset + local_idx}: {e}")
                        frame_idx = frame_offset + local_idx
                        keypoint_results.append((frame_idx, np.empty((0, 29, 3)), {}))
                    
            except Exception as e:
                print(f"⚠️  Batch keypoint detection failed, falling back to frame-by-frame: {e}")
                # Fallback to frame-by-frame
                for idx, local_idx in enumerate(batch_indices):
                    frame = batch_frames[idx]
                    frame_idx = frame_offset + local_idx
                    keypoints, metadata = self.keypoint_pipeline.detect_keypoints_in_frame(frame)
                    keypoint_results.append((frame_idx, keypoints, metadata))
        
        # Sort results by frame index
        detection_results.sort(key=lambda x: x[0])
        keypoint_results.sort(key=lambda x: x[0])
        
        # Create lookup dictionaries using LOCAL indices (0 to len(frames)-1) for this chunk
        # Map global frame_idx to local index by subtracting frame_offset
        detection_dict = {(idx - frame_offset): (p, b, r) for idx, p, b, r in detection_results}
        keypoint_dict = {(idx - frame_offset): (k, m) for idx, k, m in keypoint_results}
        
        # Verify we have all frames
        if len(detection_dict) != len(frames):
            missing = set(range(len(frames))) - set(detection_dict.keys())
            if missing:
                print(f"⚠️  Warning: Missing detections for {len(missing)} frames in chunk. Filling with empty detections.")
                for missing_idx in missing:
                    detection_dict[missing_idx] = (sv.Detections.empty(), sv.Detections.empty(), sv.Detections.empty())
                    keypoint_dict[missing_idx] = (np.empty((0, 29, 3)), {})
        
        # Now process frames sequentially for tracking (which requires sequential processing)
        frame_embeddings_buffer = []  # Store (frame_idx, crops, detections) for batching
        
        for i, frame in enumerate(tqdm(frames, desc="Processing tracking & events")):
            try:
                frame_idx = frame_offset + i  # Adjust for chunk offset
                
                # Get pre-computed detections and keypoints (using local index i)
                if i not in detection_dict:
                    # Fallback: create empty detections if missing
                    player_detections, ball_detections, referee_detections = sv.Detections.empty(), sv.Detections.empty(), sv.Detections.empty()
                    keypoints = np.empty((0, 29, 3))
                else:
                    player_detections, ball_detections, referee_detections = detection_dict[i]
                    keypoints, _ = keypoint_dict.get(i, (np.empty((0, 29, 3)), {}))
                
                # Update with tracking (both players and ball)
                player_detections = self.tracking_pipeline.tracking_callback(player_detections)
                # Pass player_detections to ball tracking for proximity validation
                ball_detections = self.tracking_pipeline.ball_tracking_callback(ball_detections, frame_idx=frame_idx, player_detections=player_detections)
            except Exception as e:
                print(f"⚠️  Error processing frame {frame_offset + i}: {e}")
                # Continue with empty detections
                player_detections = sv.Detections.empty()
                ball_detections = sv.Detections.empty()
                referee_detections = sv.Detections.empty()
                keypoints = np.empty((0, 29, 3))
                frame_idx = frame_offset + i

            # Extract crops for team assignment (but batch process UMAP for 23x speedup)
            if len(player_detections.xyxy) > 0:
                crops = self.tracking_pipeline.clustering_manager.embedding_extractor.get_player_crops(frame, player_detections)
                frame_embeddings_buffer.append((i, crops, player_detections, ball_detections, referee_detections, keypoints, frame))
            else:
                # No players, skip team assignment
                frame_embeddings_buffer.append((i, [], player_detections, ball_detections, referee_detections, keypoints, frame))
            
            # Process batch when buffer is full or at end
            if len(frame_embeddings_buffer) >= EMBEDDING_FRAME_BATCH_SIZE or i == len(frames) - 1:
                # Extract all embeddings in batch (GPU - fast)
                all_crops = []
                frame_data = []  # Store (frame_idx, num_players, detections, keypoints, frame)
                for local_frame_idx, crops, p_det, b_det, r_det, kp, orig_frame in frame_embeddings_buffer:
                    if len(crops) > 0:
                        all_crops.extend(crops)
                        frame_data.append((local_frame_idx, len(crops), p_det, b_det, r_det, kp, orig_frame))
                
                if len(all_crops) > 0:
                    # Batch extract embeddings (GPU - fast)
                    crop_batches = self.tracking_pipeline.clustering_manager.embedding_extractor.create_batches(all_crops, EMBEDDING_BATCH_SIZE)
                    all_embeddings = self.tracking_pipeline.clustering_manager.embedding_extractor.get_embeddings(crop_batches)
                    
                    # Clear GPU cache after embedding extraction
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Batch UMAP transform (CPU - would be GPU with cuML)
                    reduced_embeddings, _ = self.tracking_pipeline.clustering_manager.project_embeddings(all_embeddings, train=False)
                    
                    # Batch K-means predict (CPU - would be GPU with cuML)
                    cluster_labels, _ = self.tracking_pipeline.clustering_manager.cluster_embeddings(reduced_embeddings, train=False)
                    
                    # Assign labels back to detections
                    label_idx = 0
                    for local_frame_idx, num_players, p_det, b_det, r_det, kp, orig_frame in frame_data:
                        actual_frame_idx = frame_offset + local_frame_idx  # Adjust for chunk offset
                        frame_labels = cluster_labels[label_idx:label_idx+num_players]
                        p_det.class_id = frame_labels
                        label_idx += num_players
                        
                        # Process event detection for this frame (passes, shots, free kicks, corners)
                        self._process_event_detection(actual_frame_idx, p_det, b_det, kp)
                        
                        # Store tracks for interpolation
                        all_tracks = self.tracking_pipeline.convert_detection_to_tracks(p_det, b_det, r_det, all_tracks, actual_frame_idx)
                        
                        # Get tactical frame from detections (but don't accumulate - we re-read for annotation)
                        # tactical_frame, _ = self.tactical_pipeline.process_detections_for_tactical_analysis(p_det, r_det, kp)
                        # tactical_frames.append(tactical_frame)  # DISABLED: Not needed since we re-read video for annotation
                
                # Process frames without players
                for local_frame_idx, crops, p_det, b_det, r_det, kp, orig_frame in frame_embeddings_buffer:
                    if len(crops) == 0:  # No players in this frame
                        actual_frame_idx = frame_offset + local_frame_idx  # Adjust for chunk offset
                        # Process event detection (even with no players, ball might be present)
                        self._process_event_detection(actual_frame_idx, p_det, b_det, kp)
                        
                        all_tracks = self.tracking_pipeline.convert_detection_to_tracks(p_det, b_det, r_det, all_tracks, actual_frame_idx)
                        # tactical_frame, _ = self.tactical_pipeline.process_detections_for_tactical_analysis(p_det, r_det, kp)
                        # tactical_frames.append(tactical_frame)  # DISABLED: Not needed since we re-read video for annotation
                
                # Clear buffer
                frame_embeddings_buffer = []
    
    def _process_event_detection(self, frame_idx: int, player_detections: sv.Detections, 
                                  ball_detections: sv.Detections, keypoints: np.ndarray):
        """
        Process event detection for a single frame (passes, shots, free kicks, corners).
        
        Args:
            frame_idx: Current frame index
            player_detections: Player detections with tracker IDs and team IDs (class_id)
            ball_detections: Ball detections
            keypoints: Field keypoints for position analysis
        """
        if self.event_detector is None:
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
            ball_pos = np.array([center_x, center_y], dtype=np.float32)
        
        # Set field keypoints if available (for free kicks and corners)
        if keypoints is not None and keypoints.size > 0:
            # Extract field corners from keypoints
            from keypoint_detection.detect_keypoints import extract_field_corners
            field_corners = extract_field_corners(keypoints)
            self.event_detector.set_field_keypoints(keypoints, field_corners)
            
            # Also set goal positions from keypoints for shot detection
            self.event_detector.shot_detector.set_goal_positions_from_keypoints(keypoints, field_corners)
        
        # Process frame through event detector
        self.event_detector.process_frame(
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
    
    def _export_all_events_to_csv(self, passes, shots, free_kicks, corners, video_path: str, suffix: str = "_complete_analysis"):
        """
        Export all events (passes, shots, free kicks, corners) to CSV file.
        
        Args:
            passes: List of PassEvent objects
            shots: List of ShotEvent objects
            free_kicks: List of FreeKickEvent objects
            corners: List of CornerEvent objects
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
                'event_type',  # 'pass', 'shot', 'free_kick', 'corner'
                'event_id',
                'time_start',
                'time_end',
                'time_start_seconds',
                'time_end_seconds',
                'player_id',  # Passer/Shooter/Kicker
                'receiver_player_id',  # Only for passes
                'team_color',
                'team_id',
                'receiver_team_id',  # Only for passes
                'pass_outcome',  # Only for passes: 'success' or 'interception'
                'shot_type',  # Only for shots: 'shot_on_target', 'shot_off_target', 'shot_blocked'
                'free_kick_type',  # Only for free kicks: 'direct' or 'indirect'
                'corner_side',  # Only for corners: 'top_left', 'top_right', 'bottom_left', 'bottom_right'
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
                    'free_kick_type': '',
                    'corner_side': '',
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
                    'free_kick_type': '',
                    'corner_side': '',
                    'duration_seconds': shot_event.duration_seconds,
                    'distance_pixels': shot_event.distance_pixels,
                    'speed_pixels_per_second': shot_event.speed_pixels_per_second,
                    'confidence': shot_event.confidence
                })
            
            # Write free kicks
            for fk_event in free_kicks:
                writer.writerow({
                    'event_type': 'free_kick',
                    'event_id': fk_event.event_id,
                    'time_start': f"{fk_event.start_time:.2f}s",
                    'time_end': f"{fk_event.end_time:.2f}s",
                    'time_start_seconds': fk_event.start_time,
                    'time_end_seconds': fk_event.end_time,
                    'player_id': fk_event.kicker_id,
                    'receiver_player_id': '',  # Free kicks have no receiver
                    'team_color': team_colors.get(fk_event.team_id, "Unknown"),
                    'team_id': fk_event.team_id,
                    'receiver_team_id': '',
                    'pass_outcome': '',
                    'shot_type': '',
                    'free_kick_type': fk_event.free_kick_type.value,
                    'corner_side': '',
                    'duration_seconds': fk_event.duration_seconds,
                    'distance_pixels': fk_event.distance_pixels,
                    'speed_pixels_per_second': fk_event.speed_pixels_per_second,
                    'confidence': fk_event.confidence
                })
            
            # Write corners
            for corner_event in corners:
                writer.writerow({
                    'event_type': 'corner',
                    'event_id': corner_event.event_id,
                    'time_start': f"{corner_event.start_time:.2f}s",
                    'time_end': f"{corner_event.end_time:.2f}s",
                    'time_start_seconds': corner_event.start_time,
                    'time_end_seconds': corner_event.end_time,
                    'player_id': corner_event.kicker_id,
                    'receiver_player_id': '',  # Corners have no receiver
                    'team_color': team_colors.get(corner_event.team_id, "Unknown"),
                    'team_id': corner_event.team_id,
                    'receiver_team_id': '',
                    'pass_outcome': '',
                    'shot_type': '',
                    'free_kick_type': '',
                    'corner_side': corner_event.corner_side.value,
                    'duration_seconds': corner_event.duration_seconds,
                    'distance_pixels': corner_event.distance_pixels,
                    'speed_pixels_per_second': corner_event.speed_pixels_per_second,
                    'confidence': corner_event.confidence
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

    def _display_all_events_stats(self, passes: List, shots: List, free_kicks: List, corners: List):
        """
        Display formatted statistics for all events by team.
        
        Args:
            passes: List of PassEvent objects
            shots: List of ShotEvent objects
            free_kicks: List of FreeKickEvent objects
            corners: List of CornerEvent objects
        """
        from pass_detection.shot_detector import ShotType
        
        # Calculate stats by team
        team_0_passes = [p for p in passes if p.team_id == 0]
        team_1_passes = [p for p in passes if p.team_id == 1]
        
        team_0_shots = [s for s in shots if s.team_id == 0]
        team_1_shots = [s for s in shots if s.team_id == 1]
        
        team_0_free_kicks = [f for f in free_kicks if f.team_id == 0]
        team_1_free_kicks = [f for f in free_kicks if f.team_id == 1]
        
        team_0_corners = [c for c in corners if c.team_id == 0]
        team_1_corners = [c for c in corners if c.team_id == 1]
        
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
        
        print(f"\n{'FREE KICKS':<20} {'Team 0 (Purple)':<20} {'Team 1 (Red)':<20}")
        print("-" * 60)
        print(f"{'Total Free Kicks':<20} {len(team_0_free_kicks):<20} {len(team_1_free_kicks):<20}")
        
        print(f"\n{'CORNERS':<20} {'Team 0 (Purple)':<20} {'Team 1 (Red)':<20}")
        print("-" * 60)
        print(f"{'Total Corners':<20} {len(team_0_corners):<20} {len(team_1_corners):<20}")
        
        print(f"\n{'SUMMARY':<20} {'Team 0 (Purple)':<20} {'Team 1 (Red)':<20}")
        print("-" * 60)
        print(f"{'Total Events':<20} {len(team_0_passes) + team_0_total_shots + len(team_0_free_kicks) + len(team_0_corners):<20} {len(team_1_passes) + team_1_total_shots + len(team_1_free_kicks) + len(team_1_corners):<20}")
        
        print("=" * 60)
    
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
    
    # Initialize pipeline with YOLO for ball detection
    pipeline = CompleteSoccerAnalysisPipeline(
        detection_model_path=model_path,
        keypoint_model_path=keypoint_model_path
    )
    
    # Use the Veo match video
    video_path = "/home/essashah/SWE/veo_match_940d1c9b.mp4"

    
    # Process all frames in the trimmed video
    output_video = pipeline.analyze_video(video_path, frame_count=-1)    
    print(f"\nAnalysis finished! Output video: {output_video}")