import subprocess
import cv2
import os
import glob
from tqdm import tqdm

######### VIDEO LOADING & SAVING #########

def extract_video_frames(video_path, output_dir=None, frame_step=1, starting_second=None, ending_second=None):
    """
    Extract frames from a video within a specified time range and save them.
    Shows progress with tqdm progress bar.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str, optional): Output directory for frames
        frame_step (int): Extract every nth frame
        starting_second (float, optional): Start time in seconds
        ending_second (float, optional): End time in seconds
    
    Returns:
        str: Output directory path
    """
    
    # Get video directory and name
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get basic video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps if fps > 0 else 0
    
    # Check actual frame dimensions vs metadata
    width_meta = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_meta = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Try to get SAR (Sample Aspect Ratio) from ffprobe
    sar_num, sar_den = 1, 1
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
             '-show_entries', 'stream=sample_aspect_ratio',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip() and result.stdout.strip() != 'N/A':
            sar_str = result.stdout.strip()
            if ':' in sar_str:
                sar_num, sar_den = map(int, sar_str.split(':'))
                print(f"Detected Sample Aspect Ratio (SAR): {sar_num}:{sar_den}")
    except:
        pass  # ffprobe not available or failed
    
    # Read a test frame to check for dimension issues
    test_ret, test_frame = cap.read()
    needs_resize = False
    target_width, target_height = width_meta, height_meta
    
    if test_ret:
        actual_h, actual_w = test_frame.shape[:2]
        
        # Calculate correct display dimensions using SAR
        if sar_num != sar_den:
            # Non-square pixels detected
            # Display width = storage width × (SAR_num / SAR_den)
            corrected_width = int(round(actual_w * sar_num / sar_den))
            print(f"⚠️  Non-square pixel aspect ratio detected!")
            print(f"   Storage dimensions: {actual_w}x{actual_h}")
            print(f"   SAR: {sar_num}:{sar_den}")
            print(f"   Corrected display dimensions: {corrected_width}x{actual_h}")
            needs_resize = True
            target_width = corrected_width
            target_height = actual_h
        elif actual_w != width_meta or actual_h != height_meta:
            print(f"⚠️  Video dimension mismatch detected!")
            print(f"   Metadata: {width_meta}x{height_meta}")
            print(f"   Actual frames: {actual_w}x{actual_h}")
            print(f"   Frames will be resized to match metadata dimensions")
            needs_resize = True
            target_width = width_meta
            target_height = height_meta
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    print(f"Video: {video_name}")
    print(f"FPS: {fps}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {video_duration:.2f} seconds")
    if needs_resize:
        print(f"Output resolution (corrected): {target_width}x{target_height}")
    
    # Calculate start and end frames
    start_frame = int(starting_second * fps) if starting_second is not None else 0
    end_frame = int(ending_second * fps) if ending_second is not None else frame_count
    
    # Validate ranges
    start_frame = max(0, start_frame)
    end_frame = min(frame_count, end_frame)
    
    if start_frame >= end_frame:
        print("Error: Start time is after end time or invalid range")
        cap.release()
        return None
    
    print(f"Extracting frames from {start_frame} to {end_frame}")
    if starting_second is not None:
        print(f"Time range: {starting_second:.2f}s to {ending_second if ending_second else video_duration:.2f}s")
    
    # Set output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(video_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate expected number of frames to be extracted
    total_frames_in_range = end_frame - start_frame
    expected_frame_count = (total_frames_in_range - 1) // frame_step + 1 if total_frames_in_range > 0 else 0

    # Check for existing frames in the output directory
    existing_frames = glob.glob(os.path.join(output_dir, "*.jpg"))
    if existing_frames:
        if len(existing_frames) == expected_frame_count:
            print(f"✔️  Found existing frames ({len(existing_frames)} frames). Skipping extraction.\n")
            cap.release()
            return output_dir
        else:
            print(f"ℹ️  Existing frame count ({len(existing_frames)}) does not match expected ({expected_frame_count}). Clearing directory.")
            for frame_file in existing_frames:
                os.remove(frame_file)
    
    print(f"ℹ️  Saving {expected_frame_count} frames to {output_dir}")
    
    # Set video position to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Initialize progress bar
    pbar = tqdm(total=expected_frame_count, desc="Extracting frames",
                unit="frame", dynamic_ncols=True)
    
    # Extract and save frames
    frame_idx = start_frame
    saved_frame_count = 0
    
    while cap.isOpened() and frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if this frame should be extracted based on frame_step
        if (frame_idx - start_frame) % frame_step == 0:
            # Resize frame if needed to correct aspect ratio (PAR issue)
            if needs_resize:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            
            frame_path = os.path.join(output_dir, f"{frame_idx:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frame_count += 1
            pbar.update(1)
        
        frame_idx += 1

    cap.release()
    pbar.close()
    
    print(f"✔️ Extracted {saved_frame_count} frames from {start_frame} to {frame_idx-1}")
        
    return output_dir


def resolve_device(requested: str | None = "auto") -> str:
    """
    Resolve the best torch device based on user request and availability.

    Order:
    1) Respect explicit 'cuda' or 'mps' if available, otherwise fall back to CPU.
    2) For 'auto', prefer CUDA, then MPS, else CPU.
    """
    try:
        import torch
    except Exception:
        return "cpu"

    req = (requested or "auto").lower()

    def mps_available() -> bool:
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_available():
            return "mps"
        return "cpu"

    if req == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req == "mps":
        return "mps" if mps_available() else "cpu"

    return "cpu"


def cleanup_device(device: str) -> None:
    """
    Best-effort device-specific cleanup to release memory.
    """
    try:
        import torch
    except Exception:
        return

    device = (device or "cpu").lower()

    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        elif device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            # MPS exposes empty_cache/synchronize for memory pressure relief
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
    except Exception:
        # Keep silent; cleanup is best-effort.
        pass
