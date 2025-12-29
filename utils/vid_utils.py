import cv2

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


def write_video(frames, out_path, fps=30):
    """This function writes the frames to a video file"""

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


def show_image(image, title="Image"):
    """This function displays an image"""

    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()