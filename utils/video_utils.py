#Import All the Required Libraries
import cv2

#Create a Read Video Function
def read_video(video_path, max_width=1280, max_height=720):
    """
    Read video and optionally resize to reduce memory usage
    Args:
        video_path: Path to video file
        max_width: Maximum width (None to keep original)
        max_height: Maximum height (None to keep original)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate resize dimensions if needed
    if max_width and max_height and (original_width > max_width or original_height > max_height):
        scale = min(max_width / original_width, max_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        print(f"Resizing frames from {original_width}x{original_height} to {new_width}x{new_height} to save memory")
    else:
        new_width = original_width
        new_height = original_height
        scale = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scale:
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return frames, fps

#Create a Save Video Function
def save_video(output_video_frames, output_video_path, fps=25.0):
    if len(output_video_frames) == 0:
        print("Warning: No frames to save!")
        return
    
    height, width = output_video_frames[0].shape[:2]
    
    # Try different codecs for better compatibility
    if output_video_path.lower().endswith('.mp4'):
        # Try avc1 (H.264) first, then mp4v
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # If avc1 doesn't work, try mp4v
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for {output_video_path}")
        return
    
    for i, frame in enumerate(output_video_frames):
        out.write(frame)
    
    out.release()
    print(f"Saved {len(output_video_frames)} frames at {fps} FPS to {output_video_path}")

