import cv2
import os
import numpy as np

# ================= CONFIG =================
VIDEO_PATH = "/workspaces/3Drecon/mickey.mp4"
OUTPUT_FOLDER = "Images"

MIN_IMAGES = 80
MAX_IMAGES = 250

MOTION_THRESHOLD = 20      # lower = more frames
IMAGE_FORMAT = ".png"      # lossless
# ==========================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_total_frames(video_path):
    """Get total frame count from video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def extract_frames(video_path, output_root):
    """Extract frames from video with motion-based filtering."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Can't open {video_path}")
        return False

    total_frames = get_total_frames(video_path)
    if total_frames == 0:
        print(f"[ERROR] Video has no frames: {video_path}")
        return False

    # Compute sampling step to target frame count
    target_frames = min(MAX_IMAGES, max(MIN_IMAGES, total_frames // 10))
    step = max(1, total_frames // target_frames)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(output_root, video_name)
    os.makedirs(out_dir, exist_ok=True)

    prev_gray = None
    saved = 0
    frame_id = 0

    print(f"\n[INFO] Processing: {video_name}")
    print(f"  Total frames: {total_frames}")
    print(f"  Sampling step: {step}")
    print(f"  Target frames: {target_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames based on step
        if frame_id % step != 0:
            frame_id += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # First frame always saved
        if prev_gray is None:
            save = True
        else:
            # Calculate motion score
            diff = cv2.absdiff(prev_gray, gray)
            motion = np.mean(diff)
            # Save if high motion or haven't reached minimum yet
            save = motion > MOTION_THRESHOLD or saved < MIN_IMAGES

        if save and saved < MAX_IMAGES:
            filename = f"Image_{saved+1:02d}{IMAGE_FORMAT}"
            filepath = os.path.join(out_dir, filename)
            cv2.imwrite(filepath, frame)
            saved += 1
            
            if saved % 20 == 0:
                print(f"  Saved: {saved} frames...")

        prev_gray = gray
        frame_id += 1

        if saved >= MAX_IMAGES:
            print(f"  Reached max frames ({MAX_IMAGES})")
            break

    cap.release()

    # SAFETY CHECK
    if saved < MIN_IMAGES:
        print(f"[WARNING] Only {saved} frames extracted (minimum: {MIN_IMAGES})")
        print(f"  Reason: Video lacks sufficient motion")
        return False
    else:
        print(f"[DONE] {video_name}: {saved} images saved to {out_dir}")
        return True


# ============== MAIN EXECUTION ==============
if __name__ == "__main__":
    # Check if video path exists
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video path not found: {VIDEO_PATH}")
        print(f"[INFO] Please update VIDEO_PATH in the script")
        exit(1)

    # Process single video or folder
    if os.path.isfile(VIDEO_PATH):
        print(f"[INFO] Processing single video file")
        extract_frames(VIDEO_PATH, OUTPUT_FOLDER)
    
    elif os.path.isdir(VIDEO_PATH):
        print(f"[INFO] Processing video folder: {VIDEO_PATH}")
        video_count = 0
        
        for vid in os.listdir(VIDEO_PATH):
            if vid.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                extract_frames(os.path.join(VIDEO_PATH, vid), OUTPUT_FOLDER)
                video_count += 1
        
        if video_count == 0:
            print(f"[ERROR] No video files found in {VIDEO_PATH}")
            exit(1)
    
    print("\n[SUCCESS] All videos processed.")