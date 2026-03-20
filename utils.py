import subprocess
import cv2
import os
import glob
from pathlib import Path
import pickle
import re
import unicodedata
import numpy as np
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
        # Allow ±1 tolerance: OpenCV's CAP_PROP_FRAME_COUNT is often off by 1
        # compared to what cap.read() actually delivers.
        if abs(len(existing_frames) - expected_frame_count) <= 1:
            print(f"✔️  Found existing frames ({len(existing_frames)} frames). Skipping extraction.\n")
            cap.release()
            return output_dir
        else:
            print(f"ℹ️  Existing frame count ({len(existing_frames)}) does not match expected (~{expected_frame_count}). Clearing directory.")
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


def make_circular_mask_from_mask(mask_bool: np.ndarray) -> np.ndarray:
    """
    Create a circular mask that fully covers the input mask.
    The circle is centered at the mask's centroid and radius is the furthest
    mask pixel from the center.
    """
    if mask_bool is None:
        return mask_bool
    mask = np.asarray(mask_bool).astype(bool)
    if mask.size == 0 or not mask.any():
        return mask

    ys, xs = np.nonzero(mask)
    center_x = float(xs.mean())
    center_y = float(ys.mean())

    dx = xs - center_x
    dy = ys - center_y
    radius = float(np.sqrt(dx * dx + dy * dy).max())

    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    circle = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
    return circle


def centered_median_smooth(values, window=5):
    """
    Centered rolling median filter with no directional lag.

    Unlike causal (forward-only) filters, this looks equally at past and future
    frames, eliminating the tracking delay that plagues EMA/threshold-based
    stabilization in offline (non-realtime) pipelines.

    Args:
        values: Sequence of numeric values (one per frame in a track).
        window: Odd window size (default 5 = 2 frames each side).

    Returns:
        List of smoothed values, same length as input.
    """
    n = len(values)
    if n <= 1:
        return list(values)
    half = window // 2
    arr = np.asarray(values, dtype=np.float64)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        result[i] = np.median(arr[start:end])
    return result.tolist()


def export_ocr_text_timeline(
    ocr_data_or_path,
    output_path: str | os.PathLike,
    *,
    fps: float | None = None,
    video_path: str | os.PathLike | None = None,
    include_hidden: bool = True,
    min_confidence: float = 0.0,
    compare_mode: str = "normalized",
    collapse_identical: bool = True,
    include_empty: bool = False,
    text_field: str = "text",
    combine_parent_box: bool = True,
    prefer_parent_text: bool = True,
    source_filter: str | None = "ocr",
    sort_mode: str = "reading",
) -> dict:
    """
    Export OCR text per frame (or time) and collapse consecutive frames with identical text.

    Args:
        ocr_data_or_path: OCR dict or path to ocr.pkl.
        output_path: Path to write the summary (txt or md).
        fps: Frames per second for timestamps. If None, no timestamps are written.
        video_path: Optional video path to auto-detect FPS if fps is None.
        include_hidden: Include boxes with to_show=False.
        min_confidence: Minimum score/confidence to include.
        compare_mode: "normalized" or "raw" comparison for collapsing.
        collapse_identical: Collapse consecutive frames with same text signature.
        include_empty: Include frames with no text as "[NO TEXT]".
        text_field: "text", "alterego", or "best" (alterego if present else text).
        combine_parent_box: Merge word-level boxes that share the same parent box.
        prefer_parent_text: Use parent_box_text if available when combining.
        source_filter: Only include annotations matching this source (None to disable).
        sort_mode: "reading" or "none" ordering of text within a frame.

    Returns:
        dict: Summary with segments and output_path.
    """
    def _normalize_text(text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = text.casefold()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _format_timestamp(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
        return f"{minutes:02d}:{secs:06.3f}"

    def _iter_frame_annotations(data: dict):
        for frame_key, frame_data in data.items():
            if isinstance(frame_data, dict) and "annotations" in frame_data:
                anns = frame_data.get("annotations")
            else:
                anns = frame_data
            if not isinstance(anns, list):
                continue
            try:
                frame_idx = int(frame_key)
            except Exception:
                frame_idx = frame_key
            yield frame_idx, anns

    def _load_ocr_data(obj):
        if isinstance(obj, (str, os.PathLike)):
            ocr_path = Path(obj)
            if not ocr_path.exists():
                raise FileNotFoundError(f"OCR file not found: {ocr_path}")
            with open(ocr_path, "rb") as f:
                return pickle.load(f)
        return obj

    if fps is None and video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        try:
            fps_val = cap.get(cv2.CAP_PROP_FPS)
            fps = float(fps_val) if fps_val and fps_val > 0 else None
        finally:
            cap.release()

    text_field = (text_field or "text").lower().strip()
    if text_field not in {"text", "alterego", "best"}:
        raise ValueError("text_field must be one of: text, alterego, best")

    compare_mode = (compare_mode or "normalized").lower().strip()
    if compare_mode not in {"normalized", "raw"}:
        raise ValueError("compare_mode must be one of: normalized, raw")

    sort_mode = (sort_mode or "reading").lower().strip()
    if sort_mode not in {"reading", "none"}:
        raise ValueError("sort_mode must be one of: reading, none")

    ocr_data = _load_ocr_data(ocr_data_or_path)
    if not isinstance(ocr_data, dict):
        raise ValueError("ocr_data_or_path must be a dict or a valid path to ocr.pkl")

    frame_entries = []
    for frame_idx, annotations in _iter_frame_annotations(ocr_data):
        entries = []
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            if source_filter and ann.get("source") != source_filter:
                continue
            if not include_hidden and not ann.get("to_show", True):
                continue
            score = ann.get("score", ann.get("confidence", 1.0))
            try:
                score = float(score)
            except Exception:
                score = 1.0
            if score < min_confidence:
                continue
            if text_field == "alterego":
                text = ann.get("alterego", "")
            elif text_field == "best":
                text = ann.get("alterego", "") or ann.get("text", "")
            else:
                text = ann.get("text", "")
            text = str(text).strip()
            if not text:
                continue
            bbox = ann.get("bbox") or ann.get("original_bbox")
            entries.append((text, bbox, ann))

        if not entries and not include_empty:
            continue

        if combine_parent_box and entries:
            grouped = {}
            for text, bbox, ann in entries:
                parent_box = ann.get("parent_box")
                parent_text = ann.get("parent_box_text")
                key = tuple(parent_box) if parent_box else None
                grouped.setdefault(key, []).append((text, bbox, parent_text))

            combined = []
            for key, items in grouped.items():
                if key is None:
                    combined.extend(items)
                    continue
                parent_texts = [t for _, _, t in items if t]
                if prefer_parent_text and parent_texts:
                    combined_text = max(parent_texts, key=len).strip()
                else:
                    # Sort by x to preserve reading order within the parent box
                    items_sorted = sorted(
                        items,
                        key=lambda x: (float("inf") if not x[1] else x[1][0])
                    )
                    combined_text = " ".join(t for t, _, _ in items_sorted).strip()
                combined.append((combined_text, key, None))
            entries = combined

        if sort_mode == "reading":
            def _sort_key(item):
                _, bbox, _ = item
                if not bbox:
                    return (float("inf"), float("inf"))
                x1, y1, x2, y2 = bbox
                return ((y1 + y2) / 2, x1)
            entries.sort(key=_sort_key)

        if entries:
            lines = [text for text, _, _ in entries]
            if compare_mode == "normalized":
                signature = "\n".join(_normalize_text(t) for t in lines)
            else:
                signature = "\n".join(lines)
        else:
            lines = ["[NO TEXT]"]
            signature = "[NO TEXT]"

        frame_entries.append((frame_idx, lines, signature))

    frame_entries.sort(key=lambda x: x[0])

    segments = []
    for frame_idx, lines, signature in frame_entries:
        if not segments:
            segments.append({
                "start_frame": frame_idx,
                "end_frame": frame_idx,
                "lines": lines,
                "signature": signature,
            })
            continue

        last = segments[-1]
        is_consecutive = frame_idx == last["end_frame"] + 1
        is_same = signature == last["signature"]
        if collapse_identical and is_consecutive and is_same:
            last["end_frame"] = frame_idx
        else:
            segments.append({
                "start_frame": frame_idx,
                "end_frame": frame_idx,
                "lines": lines,
                "signature": signature,
            })

    output_lines = []
    output_lines.append("OCR Text Timeline")
    output_lines.append(f"Segments: {len(segments)}")
    output_lines.append(f"FPS: {fps if fps is not None else 'N/A'}")
    output_lines.append(f"Compare mode: {compare_mode}")
    output_lines.append(f"Collapse identical: {collapse_identical}")
    output_lines.append("")

    for idx, seg in enumerate(segments, start=1):
        start = seg["start_frame"]
        end = seg["end_frame"]
        if start == end:
            frame_label = f"Frame {start}"
        else:
            frame_label = f"Frames {start}-{end}"

        if fps:
            start_ts = _format_timestamp(start / fps)
            end_ts = _format_timestamp(end / fps)
            time_label = f" | {start_ts}" if start == end else f" | {start_ts}-{end_ts}"
        else:
            time_label = ""

        output_lines.append(f"{idx}. {frame_label}{time_label}")
        for line in seg["lines"]:
            output_lines.append(f"{line}")
        output_lines.append("")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines), encoding="utf-8")

    return {
        "segments": segments,
        "output_path": str(output_path),
        "fps": fps,
    }
