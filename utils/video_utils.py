#Import All the Required Libraries
import cv2

#Create a Read Video Function
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
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

