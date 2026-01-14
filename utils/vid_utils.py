import cv2
import subprocess
import os
import tempfile

def read_video(vid_path, frame_count=300):
    """This function reads a video file and yields each frame of the video"""

    frames = []
    cap = cv2.VideoCapture(vid_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {vid_path}")
    
    # Get video properties for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Determine how many frames to read
    if frame_count == -1:
        frames_to_read = total_frames
        print(f"Reading entire video: {total_frames} frames (~{total_frames/fps:.1f} seconds)")
    else:
        frames_to_read = min(frame_count, total_frames)
        print(f"Reading {frames_to_read} frames from video (total: {total_frames})")
    
    counter = 0
    last_progress_time = __import__('time').time()
    
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            break

        frames.append(frame)
        counter += 1
        
        # Progress logging every 100 frames or every 2 seconds
        current_time = __import__('time').time()
        if counter % 100 == 0 or (current_time - last_progress_time) >= 2.0:
            progress_pct = (counter / frames_to_read * 100) if frames_to_read > 0 else 0
            print(f"  Reading video: {counter}/{frames_to_read} frames ({progress_pct:.1f}%)", end='\r')
            last_progress_time = current_time

        if counter >= frames_to_read:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Loaded {len(frames)} frames from video")

    return frames


def read_video_chunked(vid_path, chunk_size=5000, frame_count=-1):
    """Read video in chunks to avoid loading all frames into memory.
    
    Args:
        vid_path: Path to video file
        chunk_size: Number of frames per chunk (default 5000)
        frame_count: Total frames to read (-1 for all)
        
    Yields:
        Tuple of (chunk_frames, start_idx, end_idx)
    """
    cap = cv2.VideoCapture(vid_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {vid_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if frame_count == -1:
        frames_to_read = total_frames
    else:
        frames_to_read = min(frame_count, total_frames)
    
    print(f"Reading video in chunks: {total_frames} total frames (~{total_frames/fps:.1f} seconds)")
    print(f"Chunk size: {chunk_size} frames")
    
    chunk_frames = []
    counter = 0
    chunk_start = 0
    last_progress_time = __import__('time').time()
    
    while cap.isOpened() and counter < frames_to_read:
        success, frame = cap.read()
        
        if not success:
            break
        
        chunk_frames.append(frame)
        counter += 1
        
        # Progress logging
        current_time = __import__('time').time()
        if counter % 100 == 0 or (current_time - last_progress_time) >= 2.0:
            progress_pct = (counter / frames_to_read * 100) if frames_to_read > 0 else 0
            print(f"  Reading video: {counter}/{frames_to_read} frames ({progress_pct:.1f}%)", end='\r')
            last_progress_time = current_time
        
        # Yield chunk when full or at end
        if len(chunk_frames) >= chunk_size or counter >= frames_to_read:
            chunk_end = chunk_start + len(chunk_frames)
            yield chunk_frames, chunk_start, chunk_end
            chunk_start = chunk_end
            chunk_frames = []  # Clear chunk to free memory
    
    # Yield remaining frames
    if len(chunk_frames) > 0:
        chunk_end = chunk_start + len(chunk_frames)
        yield chunk_frames, chunk_start, chunk_end
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Finished reading video")


def write_video(frames, out_path, fps=30):
    """This function writes the frames to a video file using ffmpeg for H.264 encoding"""

    height, width, _ = frames[0].shape
    
    # Use ffmpeg for proper H.264 encoding (much better compatibility than OpenCV's VideoWriter)
    # Write frames to temporary file, then encode with ffmpeg
    temp_dir = tempfile.gettempdir()
    temp_input = os.path.join(temp_dir, f"temp_video_frames_{os.getpid()}.mp4")
    
    # First, write with OpenCV to temp file (faster for writing)
    print(f"Writing {len(frames)} frames to temporary file...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_writer = cv2.VideoWriter(temp_input, fourcc, fps, (width, height))
    
    if not temp_writer.isOpened():
        # Fallback to XVID if mp4v fails
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
        temp_writer = cv2.VideoWriter(temp_input, fourcc, fps, (width, height))
        if not temp_writer.isOpened():
            raise RuntimeError(f"Failed to initialize temporary video writer")
    
    for i, frame in enumerate(frames):
        temp_writer.write(frame)
        if (i + 1) % 100 == 0:
            print(f"  Written {i + 1}/{len(frames)} frames ({100*(i+1)/len(frames):.1f}%)", end='\r')
    
    temp_writer.release()
    cv2.destroyAllWindows()
    
    # Now re-encode with ffmpeg to H.264 for maximum compatibility
    print(f"\nEncoding to H.264 with ffmpeg...")
    ffmpeg_cmd = [
        'ffmpeg', '-y',  # Overwrite output
        '-i', temp_input,  # Input file
        '-c:v', 'libx264',  # H.264 codec
        '-preset', 'medium',  # Encoding speed/quality balance
        '-crf', '23',  # Quality (18-28, lower = better quality)
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        '-movflags', '+faststart',  # Enable fast start for web playback
        out_path
    ]
    
    try:
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        print(f"✅ Successfully encoded {len(frames)} frames to {out_path}")
    except subprocess.CalledProcessError as e:
        # If ffmpeg fails, try to use the temp file as output
        print(f"Warning: ffmpeg encoding failed, using temporary file: {e.stderr.decode()}")
        if os.path.exists(temp_input):
            os.rename(temp_input, out_path)
            print(f"✅ Saved video to {out_path} (without H.264 re-encoding)")
        else:
            raise RuntimeError(f"Failed to encode video: {e.stderr.decode()}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_input) and os.path.exists(out_path):
            try:
                os.remove(temp_input)
            except:
                pass


def show_image(image, title="Image"):
    """This function displays an image"""

    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()