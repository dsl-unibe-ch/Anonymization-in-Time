import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads to avoid conflicts
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='threadpoolctl')

import pickle
import json
import numpy as np
import unicodedata
from typing import Any
from tqdm import tqdm
from difflib import SequenceMatcher
import cv2
from joblib import Parallel, delayed
from pathlib import Path

from utils import extract_video_frames
from utils import resolve_device, centered_median_smooth

def _bbox_to_int_tuple(bbox):
    """Normalize a bbox-like iterable to an integer (x1, y1, x2, y2) tuple."""
    x1, y1, x2, y2 = bbox
    return (
        int(round(float(x1))),
        int(round(float(y1))),
        int(round(float(x2))),
        int(round(float(y2))),
    )

def select_device():
    """Select best available device for OCR."""
    device = resolve_device("auto")
    use_gpu = True if device in ['cuda', 'mps'] else False
    if device == 'cuda':
        print("Using CUDA device for OCR.")
    elif device == 'mps':
        print("Using MPS device for OCR.")
    else:
        print("GPU not available. Using CPU.")
    return device, use_gpu

def get_easyocr_reader(languages):
    """
    Get or initialize the global EasyOCR reader.
    
    Returns:
        easyocr.Reader: Initialized EasyOCR reader instance
    """
    try:
        import easyocr
    except Exception as e:
        raise ImportError(
            "EasyOCR is not installed. Install it or choose ocr_engine='paddleocr'."
        ) from e

    device, use_gpu = select_device()
    print("Initializing EasyOCR reader...")
    EASYOCR_READER = easyocr.Reader(languages, gpu=use_gpu)
    print("EasyOCR reader initialized.")
    return EASYOCR_READER

def get_ocr_reader(languages, ocr_engine="easyocr"):
    """
    Initialize OCR reader for the requested backend.
    """
    engine = (ocr_engine or "easyocr").lower().strip()
    if engine == "easyocr":
        return get_easyocr_reader(languages)
    # if engine == "paddleocr":
    #     return get_paddleocr_reader(languages)
    raise ValueError(f"Unsupported ocr_engine: {ocr_engine}")

def extract_text_with_easyocr(image: np.ndarray, reader: Any, min_confidence: float = 0.5) -> list:
    """
    Extract text from an image using EasyOCR.
    
    Args:
        image (np.ndarray): Input image.
        reader: Initialized EasyOCR reader.
        min_confidence (float): Minimum confidence threshold for detections.
        
    Returns:
        list: List of detected text with bounding boxes and confidence scores.
    """
    # Preprocess image for OCR and get any scale factor used during preprocessing
    processed_image, scale_factor = preprocess_image_for_ocr(image)

    # Use cleaner EasyOCR parameters to reduce oversized boxes
    results = reader.readtext(
        processed_image, 
        paragraph=False,
        text_threshold=0.6,  # Lower threshold to catch fainter text
        low_text=0.35,       # Lower low_text to catch fainter text
        canvas_size=2560,    # Larger canvas for better detection
        adjust_contrast=0.5  # Enhance contrast for better detection
    )
    
    detections = []
    for (bbox, text, confidence) in results:
        if confidence >= min_confidence:
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            x_start = int(min(x_coords))
            y_start = int(min(y_coords))
            x_end = int(max(x_coords))
            y_end = int(max(y_coords))

            # If preprocessing scaled the image, map bbox back to original image coordinates
            if scale_factor != 1:
                try:
                    x_start = int(round(x_start / scale_factor))
                    y_start = int(round(y_start / scale_factor))
                    x_end = int(round(x_end / scale_factor))
                    y_end = int(round(y_end / scale_factor))
                except Exception:
                    # Fallback to original values if anything unexpected occurs
                    pass
            
            detections.append({
                'text': text.strip(),
                'bbox': (x_start, y_start, x_end, y_end),
                'confidence': confidence,
                'original_bbox': (x_start, y_start, x_end, y_end)  # Keep original for reference
            })
    
    return detections

def extract_text_with_backend(image: np.ndarray, reader: Any, ocr_engine="easyocr", min_confidence: float = 0.5) -> list:
    """
    Dispatch OCR extraction to selected backend.
    """
    engine = (ocr_engine or "easyocr").lower().strip()
    if engine == "easyocr":
        return extract_text_with_easyocr(image, reader, min_confidence=min_confidence)
    # if engine == "paddleocr":
    #     return extract_text_with_paddleocr(image, reader, min_confidence=min_confidence)
    raise ValueError(f"Unsupported ocr_engine: {ocr_engine}")

def preprocess_image_for_ocr(
    image: np.ndarray,
    upscale_threshold: int = 720,
    noise_ratio_threshold: float = 0.85,
    variance_threshold: float = 2000.0,
    min_edge_strength: float = 140.0,
) -> tuple:
    """
    Dual-strategy preprocessing for OCR on screen recordings:
    - Pass A preserves edges (CLAHE + denoise + sharpen).
    - Pass B suppresses textured backgrounds via background flattening and binarisation.
    The function evaluates simple heuristics (edge strength vs. residual noise) and
    chooses the variant that is most likely to yield stable OCR without destroying text.
    """
    if image is None:
        return image, 1

    # --- Common grayscale & optional upscale -------------------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    # Invert image if it looks like a Dark Theme (dark background, light text).
    # This converts it to "document style" (dark text, light background) which 
    # is generally more reliable for OCR and matches the user's observation 
    # that Light Theme works better.
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)

    h, w = gray.shape
    scale_factor = 2 if min(h, w) < upscale_threshold else 1
    if scale_factor != 1:
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    def compute_metrics(img: np.ndarray) -> tuple[float, float]:
        """Return (edge_strength, residual_noise) metrics."""
        lap = cv2.Laplacian(img, cv2.CV_64F)
        edge_strength = float(lap.var())
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        residual = img.astype(np.float32) - blur.astype(np.float32)
        residual_noise = float(np.var(residual))
        return edge_strength, residual_noise

    # --- Pass A: CLAHE + denoise + sharpen ---------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.medianBlur(enhanced, 3)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    pass_a = cv2.filter2D(denoised, -1, sharpen_kernel)
    edge_a, noise_a = compute_metrics(pass_a)

    # If the image already looks clean, return early
    if noise_a < variance_threshold or edge_a == 0:
        return pass_a, scale_factor

    # --- Pass B: background flattening + binarisation ----------------------------------------
    blur_bg = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Use absdiff to capture text regardless of polarity (though we inverted above, this is safer)
    # This replaces subtract(gray, blur_bg) which only worked for light-on-dark.
    flattened = cv2.absdiff(gray, blur_bg)
    flattened = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX)

    # Adaptive thresholding (Otsu) and clean-up
    _, binary = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 'binary' now has White for text (high difference) and Black for background.
    # We want Dark Text on Light Background to match Pass A and standard document style.
    # So we invert it.
    binary = cv2.bitwise_not(binary)

    binary = cv2.medianBlur(binary, 3)
    pass_b = cv2.filter2D(binary, -1, sharpen_kernel)
    edge_b, noise_b = compute_metrics(pass_b)

    # --- Selection heuristic ----------------------------------------------------------------
    # Prefer the variant with a lower noise ratio while preserving sufficient edge strength.
    noise_ratio = (noise_a / (edge_a + 1e-6)) if edge_a > 0 else float('inf')
    choose_pass_b = (
        (noise_ratio > noise_ratio_threshold and edge_b >= max(min_edge_strength, 0.55 * edge_a))
        or (noise_b < noise_a * 0.6 and edge_b >= min_edge_strength)
    )

    if choose_pass_b and edge_b > 0:
        return pass_b, scale_factor

    return pass_a, scale_factor

def process_frame(frame_path, reader, min_confidence=0.3, ocr_engine="easyocr"):
    """
    Process a single frame to detect text boxes using EasyOCR.
    
    Args:
        frame_path (str): Path to the frame image.

    Returns:
        tuple: (frame_index, detected_boxes)
    """
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Warning: Could not read {frame_path}")
        return None
    
    # Extract frame index
    frame_name = os.path.splitext(os.path.basename(frame_path))[0]
    frame_idx = int(''.join(filter(str.isdigit, frame_name)))
 
    detections = extract_text_with_backend(frame, reader, ocr_engine=ocr_engine, min_confidence=min_confidence)

    return frame_idx, detections

def compute_frame_similarity(frame_a, frame_b):
    """
    Compute a fast similarity score between two frames.
    Uses downscaled grayscale mean absolute difference.
    
    Args:
        frame_a (np.ndarray): First frame (BGR).
        frame_b (np.ndarray): Second frame (BGR).
        
    Returns:
        float: Similarity score in [0, 1]. 1.0 = identical.
    """
    # Downscale for speed (~4x smaller per dimension)
    small_a = cv2.resize(frame_a, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(frame_b, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    gray_a = cv2.cvtColor(small_a, cv2.COLOR_BGR2GRAY) if small_a.ndim == 3 else small_a
    gray_b = cv2.cvtColor(small_b, cv2.COLOR_BGR2GRAY) if small_b.ndim == 3 else small_b
    diff = cv2.absdiff(gray_a, gray_b)
    mean_diff = float(np.mean(diff))
    # Normalize: 0 diff -> 1.0 similarity, 255 diff -> 0.0
    return 1.0 - (mean_diff / 255.0)


def process_frames_sequential(frames_dir, languages, ocr_engine="easyocr",
                              change_threshold=0.985, max_consecutive_skips=30):
    """
    Process frames sequentially with frame change detection.
    
    Consecutive frames that are visually near-identical (similarity >= change_threshold)
    reuse the previous frame's OCR results, skipping the expensive OCR call.
    For typical screen recordings this skips 50-80% of frames.
    
    Important: this does NOT skip frames during scrolling or any visual change.
    It only reuses results when the screen is truly static (same pixels).
    During scrolling, similarity drops to ~0.85-0.95 which is well below the
    threshold, so OCR runs on every scrolling frame.
    
    Anti-drift safety: after max_consecutive_skips frames are skipped in a row,
    OCR is forced to run even if the frame looks unchanged. This prevents slow
    gradual changes (cursor blink, time ticking) from accumulating undetected.
    
    Args:
        frames_dir (str): Directory containing the video frames.
        languages (list): List of languages for OCR.
        ocr_engine (str): OCR backend to use.
        change_threshold (float): Similarity threshold above which a frame is
            considered unchanged and OCR is skipped. Range [0, 1]. Default 0.985.
        max_consecutive_skips (int): Force OCR after this many consecutive skips
            to prevent drift from slow gradual changes. Default 30 (~1s at 30fps).
        
    Returns:
        dict: Dictionary mapping frame indices to detected bounding boxes.
    """
    frame_paths = sorted(
        [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))))
    )
    
    print(f"Found {len(frame_paths)} frames in {frames_dir}")

    reader = get_ocr_reader(languages, ocr_engine=ocr_engine)
    
    frame_boxes = {}
    prev_frame = None
    prev_detections = []
    skipped = 0
    processed = 0
    consecutive_skips = 0
    
    # Track similarity scores for diagnostics
    sim_below_threshold = 0  # Frames where change was detected
    
    for frame_path in tqdm(frame_paths, desc="Processing Frames (with change detection)"):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read {frame_path}")
            continue
        
        frame_name = os.path.splitext(os.path.basename(frame_path))[0]
        frame_idx = int(''.join(filter(str.isdigit, frame_name)))
        
        # Check if this frame is similar enough to the previous one to skip OCR
        if prev_frame is not None:
            similarity = compute_frame_similarity(prev_frame, frame)
            
            if similarity >= change_threshold and consecutive_skips < max_consecutive_skips:
                # Frame is virtually unchanged — reuse previous OCR results
                # Deep-copy detections so downstream mutations don't bleed across frames
                frame_boxes[frame_idx] = [d.copy() for d in prev_detections]
                skipped += 1
                consecutive_skips += 1
                prev_frame = frame
                continue
            else:
                if similarity < change_threshold:
                    sim_below_threshold += 1
                # Reset consecutive skip counter since we're running OCR
                consecutive_skips = 0
        
        # Frame changed (or is the first frame, or anti-drift triggered) — run OCR
        detections = extract_text_with_backend(
            frame, reader, ocr_engine=ocr_engine, min_confidence=0.3
        )
        frame_boxes[frame_idx] = detections
        prev_frame = frame
        prev_detections = detections
        processed += 1
    
    total = skipped + processed
    if total > 0:
        print(f"\nFrame change detection summary:")
        print(f"  Total frames:    {total}")
        print(f"  OCR processed:   {processed} ({processed/total*100:.1f}%)")
        print(f"  Skipped (static):{skipped} ({skipped/total*100:.1f}%)")
        print(f"  Changed frames:  {sim_below_threshold} (visual change detected)")
        print(f"  Effective speedup: ~{total/max(1,processed):.1f}x")
    
    return frame_boxes


# Keep alias for backward compatibility
def process_frames_parallel(frames_dir, languages, num_workers=-1, force_cpu=False, ocr_engine="easyocr"):
    """Alias for process_frames_sequential (parallel removed due to CUDA issues)."""
    return process_frames_sequential(frames_dir, languages, ocr_engine=ocr_engine)


def process_frames_change_regions(frames_dir, languages, ocr_engine="easyocr",
                                   change_threshold=0.985, region_change_threshold=0.95):
    """
    Advanced frame processing: detects changed rectangular regions per frame
    and only runs OCR on the changed parts.
    
    This is an alternative to process_frames_sequential that provides
    even more granular savings: when only a small part of the screen changed
    (e.g., new chat message), OCR only runs on that region.
    
    Args:
        frames_dir (str): Directory containing frame images.
        languages (list): OCR languages.
        ocr_engine (str): OCR backend.
        change_threshold (float): Full-frame similarity to skip entirely.
        region_change_threshold (float): Per-strip similarity to decide
            which horizontal bands need re-OCR.
            
    Returns:
        dict: frame_idx -> list of detections.
    """
    frame_paths = sorted(
        [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))))
    )
    
    print(f"Found {len(frame_paths)} frames in {frames_dir}")
    reader = get_ocr_reader(languages, ocr_engine=ocr_engine)
    
    frame_boxes = {}
    prev_frame = None
    prev_detections = []
    skipped = 0
    partial = 0
    full = 0
    
    num_strips = 6  # Divide frame into horizontal strips
    
    for frame_path in tqdm(frame_paths, desc="Processing Frames (region change)"):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        
        frame_name = os.path.splitext(os.path.basename(frame_path))[0]
        frame_idx = int(''.join(filter(str.isdigit, frame_name)))
        
        if prev_frame is not None:
            similarity = compute_frame_similarity(prev_frame, frame)
            if similarity >= change_threshold:
                frame_boxes[frame_idx] = [d.copy() for d in prev_detections]
                skipped += 1
                prev_frame = frame
                continue
        
            # Identify which horizontal strips changed
            h = frame.shape[0]
            strip_h = h // num_strips
            changed_y_min = h
            changed_y_max = 0
            
            for s in range(num_strips):
                y_start = s * strip_h
                y_end = (s + 1) * strip_h if s < num_strips - 1 else h
                strip_sim = compute_frame_similarity(
                    prev_frame[y_start:y_end], frame[y_start:y_end]
                )
                if strip_sim < region_change_threshold:
                    changed_y_min = min(changed_y_min, y_start)
                    changed_y_max = max(changed_y_max, y_end)
            
            if changed_y_min < changed_y_max:
                # Only OCR the changed region
                roi = frame[changed_y_min:changed_y_max]
                roi_detections = extract_text_with_backend(
                    roi, reader, ocr_engine=ocr_engine, min_confidence=0.3
                )
                # Shift y-coordinates back to full-frame coords
                for det in roi_detections:
                    x1, y1, x2, y2 = det['bbox']
                    det['bbox'] = (x1, y1 + changed_y_min, x2, y2 + changed_y_min)
                    if 'original_bbox' in det:
                        ox1, oy1, ox2, oy2 = det['original_bbox']
                        det['original_bbox'] = (ox1, oy1 + changed_y_min, ox2, oy2 + changed_y_min)
                
                # Keep unchanged-region detections from previous frame
                carried = []
                for d in prev_detections:
                    _, dy1, _, dy2 = d['bbox']
                    box_center_y = (dy1 + dy2) / 2
                    if box_center_y < changed_y_min or box_center_y > changed_y_max:
                        carried.append(d.copy())
                
                detections = carried + roi_detections
                frame_boxes[frame_idx] = detections
                prev_detections = detections
                prev_frame = frame
                partial += 1
                continue
        
        # Full OCR on this frame
        detections = extract_text_with_backend(
            frame, reader, ocr_engine=ocr_engine, min_confidence=0.3
        )
        frame_boxes[frame_idx] = detections
        prev_frame = frame
        prev_detections = detections
        full += 1
    
    total = skipped + partial + full
    if total > 0:
        print(f"\nRegion change detection: {full} full OCR, {partial} partial OCR, "
              f"{skipped} skipped ({skipped/total*100:.1f}% fully skipped, "
              f"{(skipped+partial)/total*100:.1f}% with savings)")
    
    return frame_boxes

def merge_line_boxes(
    frame_boxes,
    x_threshold=20,
    height_rel_tol=0.20,
    center_y_tol_factor=0.50,
    min_vertical_overlap_ratio=0.50,
    max_x_overlap_px=10,
):
    """
    Merge OCR boxes that belong to the same text line using geometry only.

    This function does not filter by names/text content. It groups boxes by
    line compatibility (height, vertical alignment, vertical overlap, and
    horizontal proximity) and merges each compatible cluster left-to-right.
    """
    merged_frame_boxes = {}

    def _bbox_metrics(box):
        bbox = box.get("bbox")
        if bbox is None:
            return None
        x1, y1, x2, y2 = _bbox_to_int_tuple(bbox)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        h = max(1, y2 - y1)
        cy = (y1 + y2) / 2.0
        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "h": h,
            "cy": cy,
        }

    def _cluster_sort_key(box):
        m = _bbox_metrics(box)
        if m is None:
            return (float("inf"), float("inf"))
        return (m["cy"], m["x1"])

    def _left_to_right_key(box):
        m = _bbox_metrics(box)
        if m is None:
            return (float("inf"), float("inf"))
        return (m["x1"], m["cy"])

    def _compatible(a, b):
        ma = _bbox_metrics(a)
        mb = _bbox_metrics(b)
        if ma is None or mb is None:
            return False

        rel_h_diff = abs(ma["h"] - mb["h"]) / float(max(ma["h"], mb["h"]))
        if rel_h_diff > height_rel_tol:
            return False

        center_y_diff = abs(ma["cy"] - mb["cy"])
        if center_y_diff > center_y_tol_factor * min(ma["h"], mb["h"]):
            return False

        overlap_y = max(0, min(ma["y2"], mb["y2"]) - max(ma["y1"], mb["y1"]))
        if overlap_y / float(min(ma["h"], mb["h"])) < min_vertical_overlap_ratio:
            return False

        # Require proximity on x (same idea as normalize_parent_boxes_by_line).
        x_gap = max(0, max(ma["x1"], mb["x1"]) - min(ma["x2"], mb["x2"]))
        if x_gap > x_threshold:
            return False

        # Allow slight overlap from OCR jitter, but reject heavy overlap (likely duplicate boxes).
        overlap_x = max(0, min(ma["x2"], mb["x2"]) - max(ma["x1"], mb["x1"]))
        if overlap_x > max_x_overlap_px:
            return False

        return True

    def _merge_cluster(cluster_boxes):
        cluster_sorted = sorted(cluster_boxes, key=_left_to_right_key)

        valid_bbox_entries = [b for b in cluster_sorted if b.get("bbox") is not None]
        if not valid_bbox_entries:
            return cluster_sorted[0].copy()

        x1 = min(_bbox_to_int_tuple(b["bbox"])[0] for b in valid_bbox_entries)
        y1 = min(_bbox_to_int_tuple(b["bbox"])[1] for b in valid_bbox_entries)
        x2 = max(_bbox_to_int_tuple(b["bbox"])[2] for b in valid_bbox_entries)
        y2 = max(_bbox_to_int_tuple(b["bbox"])[3] for b in valid_bbox_entries)

        orig_boxes = [b.get("original_bbox", b["bbox"]) for b in valid_bbox_entries]
        ox1 = min(_bbox_to_int_tuple(bb)[0] for bb in orig_boxes)
        oy1 = min(_bbox_to_int_tuple(bb)[1] for bb in orig_boxes)
        ox2 = max(_bbox_to_int_tuple(bb)[2] for bb in orig_boxes)
        oy2 = max(_bbox_to_int_tuple(bb)[3] for bb in orig_boxes)

        text_parts = []
        for b in cluster_sorted:
            t = (b.get("text") or "").strip()
            if t:
                text_parts.append(t)

        confidences = []
        for b in cluster_sorted:
            c = b.get("confidence", None)
            if c is None:
                continue
            try:
                confidences.append(float(c))
            except (TypeError, ValueError):
                continue

        merged = cluster_sorted[0].copy()
        merged["bbox"] = (int(x1), int(y1), int(x2), int(y2))
        merged["original_bbox"] = (int(ox1), int(oy1), int(ox2), int(oy2))
        if text_parts:
            merged["text"] = " ".join(text_parts)
        if confidences:
            merged["confidence"] = sum(confidences) / float(len(confidences))

        # Preserve original OCR-detected box boundaries so that
        # split_boxes_into_words can use them as accurate split points
        # instead of guessing proportionally by character count.
        merged["component_boxes"] = [
            (_bbox_to_int_tuple(b["bbox"]), (b.get("text") or "").strip())
            for b in valid_bbox_entries
            if (b.get("text") or "").strip()
        ]

        return merged

    for frame_idx, boxes in frame_boxes.items():
        if not boxes:
            merged_frame_boxes[frame_idx] = []
            continue

        boxes_copy = [b.copy() for b in boxes]
        valid_indices = [i for i, b in enumerate(boxes_copy) if b.get("bbox") is not None]
        invalid_indices = [i for i, b in enumerate(boxes_copy) if b.get("bbox") is None]

        visited = set()
        merged_boxes = []

        sorted_valid = sorted(valid_indices, key=lambda i: _cluster_sort_key(boxes_copy[i]))

        for start in sorted_valid:
            if start in visited:
                continue

            cluster = []
            stack = [start]
            visited.add(start)

            while stack:
                cur = stack.pop()
                cluster.append(cur)
                for j in valid_indices:
                    if j in visited:
                        continue
                    if _compatible(boxes_copy[cur], boxes_copy[j]):
                        visited.add(j)
                        stack.append(j)

            if len(cluster) == 1:
                merged_boxes.append(boxes_copy[cluster[0]])
            else:
                merged_boxes.append(_merge_cluster([boxes_copy[idx] for idx in cluster]))

        for idx in invalid_indices:
            merged_boxes.append(boxes_copy[idx])

        merged_boxes.sort(key=_cluster_sort_key)
        merged_frame_boxes[frame_idx] = merged_boxes

    return merged_frame_boxes

def split_boxes_into_words(frame_boxes):
    """
    Split multi-word text boxes into individual word boxes.
    Uses proportional bbox division based on word lengths.
    
    Args:
        frame_boxes (dict): Dictionary of frame indices to list of boxes with text
        
    Returns:
        dict: Frame boxes where each box represents a single word
    """
    split_frame_boxes = {}
    
    for frame_idx, boxes in frame_boxes.items():
        split_boxes = []
        
        for box in boxes:
            text = box.get('text', '').strip()
            if not text:
                continue
            
            words = text.split()
            if len(words) <= 1:
                # Single word or empty - keep as is but assign unique
                # word-level track_id so it doesn't collide with siblings
                box_copy = box.copy()
                if 'bbox' in box_copy:
                    box_copy['bbox'] = _bbox_to_int_tuple(box_copy['bbox'])
                if box_copy.get('parent_box') is not None:
                    box_copy['parent_box'] = _bbox_to_int_tuple(box_copy['parent_box'])
                parent_tid = box_copy.get('track_id')
                if parent_tid is not None:
                    box_copy['track_id'] = (parent_tid, 0)
                split_boxes.append(box_copy)
                continue
            
            # Multi-word box - split into individual word boxes
            x1, y1, x2, y2 = _bbox_to_int_tuple(box['bbox'])
            parent_bbox = _bbox_to_int_tuple(box.get('original_bbox', box['bbox']))
            parent_tid = box.get('track_id', None)

            # If merge_line_boxes saved the original OCR-detected component
            # boundaries, use them as accurate split guides instead of
            # guessing proportionally by character count.
            component_boxes = box.get('component_boxes')
            if component_boxes and len(component_boxes) > 1:
                # Build a flat list of (word, component_bbox) using the
                # original per-component bboxes.  Words within a single
                # component still need proportional splitting, but the
                # inter-component boundaries come from the real OCR output.
                word_boxes = []
                global_word_idx = 0

                for comp_bbox, comp_text in component_boxes:
                    comp_words = comp_text.split()
                    if not comp_words:
                        continue
                    cx1, _, cx2, _ = _bbox_to_int_tuple(comp_bbox)
                    comp_w = max(1, cx2 - cx1)

                    if len(comp_words) == 1:
                        word_tid = (parent_tid, global_word_idx) if parent_tid is not None else None
                        word_boxes.append({
                            'bbox': (cx1, y1, cx2, y2),
                            'text': comp_words[0],
                            'confidence': box.get('confidence', 1.0),
                            'parent_box': parent_bbox,
                            'parent_box_text': text,
                            'track_id': word_tid,
                        })
                        global_word_idx += 1
                    else:
                        # Proportional split within this single component
                        total_c = sum(len(w) for w in comp_words)
                        cur_x = cx1
                        for j, cw in enumerate(comp_words):
                            if j == len(comp_words) - 1:
                                wx1, wx2 = cur_x, cx2
                            else:
                                wx1 = cur_x
                                wx2 = cx1 + int(round(sum(len(comp_words[k]) for k in range(j + 1)) / total_c * comp_w))
                            word_tid = (parent_tid, global_word_idx) if parent_tid is not None else None
                            word_boxes.append({
                                'bbox': (int(wx1), y1, int(wx2), y2),
                                'text': cw,
                                'confidence': box.get('confidence', 1.0),
                                'parent_box': parent_bbox,
                                'parent_box_text': text,
                                'track_id': word_tid,
                            })
                            cur_x = wx2
                            global_word_idx += 1

                split_boxes.extend(word_boxes)
            else:
                # No component info — fall back to proportional splitting
                box_width = x2 - x1
                padding = max(2, int(box_width * 0.02))
                total_word_chars = max(1, sum(len(word) for word in words))
                content_start = x1 + padding
                content_end = x2 - padding
                content_width = max(1, content_end - content_start)

                current_x = x1
                word_boxes = []

                for i, word in enumerate(words):
                    word_length = len(word)

                    if i == 0:
                        word_x1 = x1
                        word_proportion = word_length / total_word_chars
                        word_x2 = content_start + int(word_proportion * content_width)
                    elif i == len(words) - 1:
                        word_x1 = current_x
                        word_x2 = x2
                    else:
                        word_x1 = current_x
                        word_x2 = content_start + int(sum(len(words[j]) for j in range(i + 1)) / total_word_chars * content_width)

                    word_x1 = int(max(x1, min(x2, word_x1)))
                    word_x2 = int(max(word_x1, min(x2, word_x2)))

                    word_tid = (parent_tid, i) if parent_tid is not None else None

                    word_boxes.append({
                        'bbox': (word_x1, y1, word_x2, y2),
                        'text': word,
                        'confidence': box.get('confidence', 1.0),
                        'parent_box': parent_bbox,
                        'parent_box_text': text,
                        'track_id': word_tid,
                    })

                    current_x = word_x2

                split_boxes.extend(word_boxes)
        
        split_frame_boxes[frame_idx] = split_boxes
    
    return split_frame_boxes

def filter_boxes_by_names(frame_boxes, names_dict, similarity_threshold=0.8):
    """
    Match word-level boxes against dictionary names.
    Uses actual word boxes from split_boxes_into_words - no character estimation.
    Merges adjacent matching words for multi-word names.
    
    Args:
        frame_boxes (dict): Word-level boxes from split_boxes_into_words
        names_dict (dict): Dictionary of names to alteregos
        similarity_threshold (float): Fuzzy match threshold
        
    Returns:
        dict: Filtered boxes with matched names
    """
    import re
    
    def normalize_text(text):
        """Normalize text for matching"""
        if not text:
            return ""
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        return text.lower().strip().strip('.,;:!?()\"\'-')
    
    def word_matches_name_part(word_text, name_part):
        """Check if word matches a part of a name"""
        norm_word = normalize_text(word_text)
        norm_name = normalize_text(name_part)

        if not norm_word or not norm_name:
            return False, 0.0

        # Exact match
        if norm_word == norm_name:
            return True, 1.0

        # For short words (<=4 chars), a single OCR character substitution
        # tanks the SequenceMatcher ratio (e.g. "vlo" vs "v/o" = 0.67).
        # Use edit-distance tolerance instead: accept if at most 1 char differs
        # and the lengths are close.
        if max(len(norm_word), len(norm_name)) <= 4:
            if abs(len(norm_word) - len(norm_name)) <= 1:
                # Count character-level differences (simple edit distance approx)
                diffs = sum(1 for a, b in zip(norm_word, norm_name) if a != b)
                diffs += abs(len(norm_word) - len(norm_name))
                if diffs <= 1:
                    return True, 0.85

        # Fuzzy match
        similarity = SequenceMatcher(None, norm_word, norm_name).ratio()
        if similarity >= similarity_threshold:
            return True, similarity

        return False, 0.0
    
    def group_boxes_into_lines(boxes):
        """Group boxes on same horizontal line"""
        if not boxes:
            return []
        
        # Sort by y-position
        sorted_boxes = sorted(enumerate(boxes), key=lambda x: (x[1]['bbox'][1] + x[1]['bbox'][3]) / 2)
        
        lines = []
        current_line = [sorted_boxes[0]]
        
        for idx, box in sorted_boxes[1:]:
            prev_y_center = (current_line[-1][1]['bbox'][1] + current_line[-1][1]['bbox'][3]) / 2
            curr_y_center = (box['bbox'][1] + box['bbox'][3]) / 2
            
            # Same line if y-centers within 15 pixels
            if abs(curr_y_center - prev_y_center) < 15:
                current_line.append((idx, box))
            else:
                lines.append(current_line)
                current_line = [(idx, box)]
        
        lines.append(current_line)
        
        # Sort each line by x-position (left to right)
        for line in lines:
            line.sort(key=lambda x: x[1]['bbox'][0])
        
        return lines
    
    def find_name_sequences(line_boxes, names_dict):
        """
        Find sequences of adjacent word boxes that match names.
        
        Supports partial matching: in group chats, only the first name may appear
        even though the dictionary has the full name (e.g., "Flavia" matching
        "Flavia Fabiel"). When a partial match is found, the alterego is
        truncated to the corresponding number of words.
        """
        matches = []
        
        for name, alterego in names_dict.items():
            name_words = normalize_text(name).split()
            alterego_words = alterego.split() if alterego else []
            
            # Try to find this name in the line
            for start_idx in range(len(line_boxes)):
                matched_boxes = []
                confidence_scores = []
                
                box_idx = start_idx
                word_idx = 0
                
                while word_idx < len(name_words) and box_idx < len(line_boxes):
                    orig_idx, box = line_boxes[box_idx]
                    box_text = box.get('text', '').strip()
                    
                    if not box_text:
                        box_idx += 1
                        continue
                    
                    is_match, conf = word_matches_name_part(box_text, name_words[word_idx])
                    
                    if is_match:
                        matched_boxes.append((orig_idx, box))
                        confidence_scores.append(conf)
                        word_idx += 1
                        box_idx += 1
                    else:
                        # No match - break this attempt
                        break
                
                if not matched_boxes:
                    continue
                
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                
                # Full match: all words of the name were found
                if word_idx == len(name_words):
                    matches.append({
                        'name': name,
                        'alterego': alterego,
                        'boxes': matched_boxes,
                        'confidence': avg_confidence,
                        'full_match': True,
                    })
                # Partial match: first N words of a multi-word name matched
                # (only for names with 2+ words where at least the first word matched)
                elif word_idx >= 1 and len(name_words) > 1:
                    # Build truncated alterego with matching number of words
                    if alterego_words:
                        partial_alterego = ' '.join(alterego_words[:word_idx])
                    else:
                        partial_alterego = alterego
                    
                    # Slightly penalise partial matches so full matches win ties
                    matches.append({
                        'name': name,
                        'alterego': partial_alterego,
                        'boxes': matched_boxes,
                        'confidence': avg_confidence * 0.95,
                        'full_match': False,
                    })
        
        # Remove overlapping matches
        # Priority: full > partial, then longer > shorter, then higher confidence
        if not matches:
            return []
        
        matches.sort(
            key=lambda x: (x.get('full_match', False), len(x['boxes']), x['confidence']),
            reverse=True,
        )
        
        final_matches = []
        used_indices = set()
        
        for match in matches:
            box_indices = {idx for idx, _ in match['boxes']}
            if not box_indices.intersection(used_indices):
                final_matches.append(match)
                used_indices.update(box_indices)
        
        return final_matches
    
    def merge_matched_boxes(match):
        """Merge word boxes from a name match into single box using word boundaries"""
        boxes = [box for _, box in match['boxes']]
        
        # Use the word box boundaries from split_boxes_into_words
        # This preserves accurate word-level splitting
        x1 = min(b['bbox'][0] for b in boxes)
        y1 = min(b['bbox'][1] for b in boxes)
        x2 = max(b['bbox'][2] for b in boxes)
        y2 = max(b['bbox'][3] for b in boxes)
        
        # Get parent box if available
        parent_box = boxes[0].get('parent_box', None)
        parent_box_text = boxes[0].get('parent_box_text', None)
        # Extract the parent (integer) track_id from the word-level tuple form.
        # Word splitting turns track_id 5 into (5, 0), (5, 1), etc.
        # For merged name boxes we want the original parent track_id back.
        track_ids = [b.get('track_id') for b in boxes if b.get('track_id') is not None]
        track_id = track_ids[0] if track_ids else None
        if isinstance(track_id, tuple):
            track_id = track_id[0]
        if parent_box is not None:
            parent_box = _bbox_to_int_tuple(parent_box)
        
        return {
            'bbox': _bbox_to_int_tuple((x1, y1, x2, y2)),
            'text': ' '.join(b.get('text', '') for b in boxes),
            'name': match['name'],
            'alterego': match['alterego'],
            'confidence': match['confidence'],
            'to_show': True,
            'parent_box': parent_box,
            'parent_box_text': parent_box_text,
            'track_id': track_id,
            'match_type': 'word_level'
        }
    
    # Main processing
    filtered_mapping = {}
    
    for frame_idx, boxes in tqdm(frame_boxes.items(), desc="Matching word boxes to names", unit="frame"):
        if not boxes:
            filtered_mapping[frame_idx] = []
            continue
        
        # Group boxes into lines
        lines = group_boxes_into_lines(boxes)
        
        frame_results = []
        all_matched_indices = set()
        
        # Process each line
        for line in lines:
            matches = find_name_sequences(line, names_dict)
            
            for match in matches:
                merged_box = merge_matched_boxes(match)
                frame_results.append(merged_box)
                
                # Track which original word boxes were matched
                all_matched_indices.update(idx for idx, _ in match['boxes'])
        
        # Add unmatched word boxes as to_show=False
        for i, box in enumerate(boxes):
            if i not in all_matched_indices:
                box_copy = box.copy()
                box_copy['to_show'] = False
                frame_results.append(box_copy)
        
        filtered_mapping[frame_idx] = frame_results
    
    return filtered_mapping


def enhanced_temporal_tracking(frame_boxes, max_gap=6, position_threshold=50, size_threshold=0.4):
    """
    Enhanced temporal tracking with frame-to-frame coordinate preservation.
    
    Args:
        frame_boxes (dict): Dictionary of frame indices to list of bounding boxes
        frame_skip (int): Frame skip interval used during detection
        max_gap (int): Maximum frame gap to interpolate across
        position_threshold (int): Maximum pixel distance for tracking (more lenient for vertical)
        size_threshold (float): Maximum relative size difference for tracking
        
    Returns:
        tuple: (stabilized_boxes dict, tracks list)
            - stabilized_boxes: Enhanced frame_boxes with stabilized coordinates
            - tracks: List of tracks, each track is a list of (frame_idx, box_idx) tuples
    """
    
    def can_track(box1, box2):
        """Check if two boxes can be tracked together - prioritizes text matching"""
        bbox1, bbox2 = box1['bbox'], box2['bbox']
        
        # Check center positions
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        # Separate horizontal and vertical distances
        x_distance = abs(center1[0] - center2[0])
        y_distance = abs(center1[1] - center2[1])
        
        # Check size similarity
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        size_diff = abs(area1 - area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        
        # Check text similarity (most important for OCR)
        text1 = box1.get('text', '') or box1.get('alterego', '')
        text2 = box2.get('text', '') or box2.get('alterego', '')
        text_sim = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        # If text matches very well, be more lenient with position/size
        if text_sim >= 0.9:
            # Very similar text - allow larger vertical movement (e.g., scrolling chat)
            return x_distance < position_threshold * 1.5 and y_distance < position_threshold * 2 and size_diff < size_threshold * 1.5
        elif text_sim >= 0.75:
            # Fairly similar text - standard thresholds
            return x_distance < position_threshold and y_distance < position_threshold * 1.5 and size_diff < size_threshold
        else:
            # Low text similarity - require tighter position match
            return x_distance < position_threshold * 0.5 and y_distance < position_threshold * 0.5 and size_diff < size_threshold * 0.8
    
    if not frame_boxes:
        return {}, []

    # Build tracks and apply frame-to-frame stabilization
    frame_indices = sorted(frame_boxes.keys())
    tracks = []  # List of tracks, each track is a list of (frame_idx, box) tuples
    track_assignments = {}  # Map (frame_idx, box_idx_in_stabilized) -> track_id
    
    # Initialize stabilized_boxes with ALL frame indices to preserve empty frames
    stabilized_boxes = {}
    for frame_idx in frame_indices:
        stabilized_boxes[frame_idx] = []
    
    # Build tracks first
    for frame_idx in frame_indices:
        current_boxes = frame_boxes[frame_idx]
        unmatched_boxes = list(enumerate(current_boxes))
        
        # Skip track processing for frames with no boxes (they're already in stabilized_boxes as empty)
        if not current_boxes:
            continue
        
        # Try to extend existing tracks
        for track in tracks:
            if not track:
                continue
                
            last_frame, last_box = track[-1]
            
            # Skip if track is too old
            if frame_idx - last_frame > max_gap:
                continue
            
            # Find best matching box — pick the one with highest text similarity
            # and closest position to the previous detection.
            best_match = None
            best_idx = None
            best_score = -1.0
            
            for box_idx, box in unmatched_boxes:
                if can_track(last_box, box):
                    # Score = text similarity (dominant) + inverse position distance (minor)
                    text1 = last_box.get('text', '') or last_box.get('alterego', '')
                    text2 = box.get('text', '') or box.get('alterego', '')
                    text_sim = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
                    
                    bbox1, bbox2 = last_box['bbox'], box['bbox']
                    c1 = ((bbox1[0]+bbox1[2])/2, (bbox1[1]+bbox1[3])/2)
                    c2 = ((bbox2[0]+bbox2[2])/2, (bbox2[1]+bbox2[3])/2)
                    dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5
                    pos_score = max(0, 1.0 - dist / (position_threshold * 2))
                    
                    score = text_sim * 0.7 + pos_score * 0.3
                    if score > best_score:
                        best_score = score
                        best_match = box
                        best_idx = box_idx
            
            if best_match is not None:
                track.append((frame_idx, best_match))
                unmatched_boxes = [(idx, box) for idx, box in unmatched_boxes if idx != best_idx]
        
        # Create new tracks for unmatched boxes
        for _, box in unmatched_boxes:
            tracks.append([(frame_idx, box)])
    
    # Apply centered median smoothing to each track (no directional lag).
    for track_id, track in enumerate(tracks):
        if len(track) < 2:
            # Single detection - keep as is
            frame_idx, box = track[0]
            box_idx_in_stabilized = len(stabilized_boxes[frame_idx])
            stabilized_box = box.copy()
            stabilized_box['track_id'] = track_id
            stabilized_box['bbox'] = _bbox_to_int_tuple(stabilized_box['bbox'])
            stabilized_boxes[frame_idx].append(stabilized_box)
            track_assignments[(frame_idx, box_idx_in_stabilized)] = track_id
            continue

        # Multi-frame track - find best text representation.
        # Primary criterion: longest text (most characters).  Partial / corrupted
        # OCR detections are shorter but can have *higher* confidence, so using
        # confidence alone would propagate truncated text to every frame.
        # Secondary: most words.  Tertiary: highest confidence.
        text_variants = [
            (box.get('text', ''), box.get('confidence', 0))
            for _, box in track
        ]
        best_text = max(
            text_variants,
            key=lambda x: (len(x[0]), len(x[0].split()), x[1]),
        )[0]

        # Collect raw top-left corners and dimensions for the whole track.
        # We smooth x1/y1 (not center) because the top-left corner of a text
        # box is the most stable reference: OCR consistently finds where text
        # starts, while the bottom/right edges jitter with character detection.
        raw_x1 = []
        raw_y1 = []
        raw_w = []
        raw_h = []
        for _, box in track:
            x1, y1, x2, y2 = box['bbox']
            raw_x1.append(float(x1))
            raw_y1.append(float(y1))
            raw_w.append(float(x2 - x1))
            raw_h.append(float(y2 - y1))

        # Centered median filter on top-left corner — symmetric window, zero lag
        smooth_x1 = centered_median_smooth(raw_x1, window=5)
        smooth_y1 = centered_median_smooth(raw_y1, window=5)

        # For width: use per-track max to prevent shrinking when OCR misses
        # trailing characters (preserves intent of the old union-bbox logic)
        stable_w = float(max(raw_w))
        # For height: use per-track median (already normalized upstream)
        stable_h = float(np.median(raw_h))

        # Build stabilized boxes from smoothed top-left + stable dimensions
        for i, (frame_idx, box) in enumerate(track):
            box_idx_in_stabilized = len(stabilized_boxes[frame_idx])

            x1 = int(round(smooth_x1[i]))
            y1 = int(round(smooth_y1[i]))
            x2 = x1 + int(round(stable_w))
            y2 = y1 + int(round(stable_h))

            stabilized_box = box.copy()
            stabilized_box['bbox'] = (x1, y1, x2, y2)
            stabilized_box['text'] = best_text
            stabilized_box['track_id'] = track_id

            # Shift component_boxes by the same delta so word splitting
            # uses boundaries consistent with the stabilized parent box.
            comp = stabilized_box.get('component_boxes')
            if comp:
                orig_x1, orig_y1 = int(round(raw_x1[i])), int(round(raw_y1[i]))
                dx = x1 - orig_x1
                dy = y1 - orig_y1
                if dx != 0 or dy != 0:
                    shifted = []
                    for (cx1, cy1, cx2, cy2), ctxt in comp:
                        shifted.append(((cx1 + dx, cy1 + dy, cx2 + dx, cy2 + dy), ctxt))
                    stabilized_box['component_boxes'] = shifted

            stabilized_boxes[frame_idx].append(stabilized_box)
            track_assignments[(frame_idx, box_idx_in_stabilized)] = track_id
    
    # Convert track_assignments to track list format: [[(frame_idx, box_idx), ...], ...]
    tracks_output = [[] for _ in range(len(tracks))]
    for (frame_idx, box_idx), track_id in track_assignments.items():
        tracks_output[track_id].append((frame_idx, box_idx))
    
    return stabilized_boxes, tracks_output


def normalize_box_heights(frame_boxes, height_clusters=None):
    """
    Normalize text box heights to consistent sizes based on clustering.
    
    Args:
        frame_boxes (dict): Dictionary of frame indices to list of bounding boxes
        height_clusters (list): Predefined height clusters, if None will compute from data
        
    Returns:
        dict: Frame boxes with normalized heights
        list: Height clusters used
    """
    from sklearn.cluster import KMeans
    import numpy as np
    
    # Collect all box heights if clusters not provided
    if height_clusters is None:
        all_heights = []
        for frame_boxes_list in frame_boxes.values():
            for box in frame_boxes_list:
                if 'bbox' in box:
                    x1, y1, x2, y2 = box['bbox']
                    height = int(round(float(y2 - y1)))
                    if height > 0:
                        all_heights.append(height)
        
        if not all_heights:
            return frame_boxes, []
        
        # Cluster heights into common sizes (typically 3-4 sizes in messaging apps)
        heights_array = np.array(all_heights).reshape(-1, 1)
        n_clusters = min(4, len(set(all_heights)))  # Max 4 text sizes
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(heights_array)
        height_clusters = sorted({
            max(1, int(round(float(h))))
            for h in kmeans.cluster_centers_.flatten()
        })
    else:
        height_clusters = sorted({
            max(1, int(round(float(h))))
            for h in height_clusters
        })
    
    if not height_clusters:
        return frame_boxes, []
    
    # Normalize boxes to closest cluster height
    normalized_boxes = {}
    for frame_idx, boxes in frame_boxes.items():
        normalized_frame_boxes = []
        for box in boxes:
            if 'bbox' in box:
                x1, y1, x2, y2 = box['bbox']
                current_height = int(round(float(y2 - y1)))
                
                # Find closest cluster height
                closest_height = min(height_clusters, key=lambda h: abs(h - current_height))
                
                # Adjust box to new height (center vertically)
                height_diff = closest_height - current_height
                new_y1 = int(max(0, round(float(y1) - (height_diff / 2))))
                new_y2 = int(new_y1 + int(closest_height))
                
                normalized_box = box.copy()
                normalized_box['bbox'] = (
                    int(round(float(x1))),
                    int(new_y1),
                    int(round(float(x2))),
                    int(new_y2),
                )
                normalized_frame_boxes.append(normalized_box)
            else:
                normalized_frame_boxes.append(box)
        
        normalized_boxes[frame_idx] = normalized_frame_boxes
    
    return normalized_boxes, height_clusters

def align_boxes_on_same_line(frame_boxes, y_threshold=15):
    """
    Ensure boxes on the same horizontal line have identical y-coordinates.
    
    Args:
        frame_boxes (dict): Dictionary of frame indices to list of boxes
        y_threshold (int): Maximum vertical distance to consider boxes on same line
        
    Returns:
        dict: Frame boxes with aligned y-coordinates
    """
    aligned_boxes = {}
    
    for frame_idx, boxes in frame_boxes.items():
        if not boxes:
            aligned_boxes[frame_idx] = []
            continue
        
        # Group boxes by y-center
        boxes_with_centers = [(box, (box['bbox'][1] + box['bbox'][3]) / 2) for box in boxes]
        boxes_with_centers.sort(key=lambda x: x[1])
        
        lines = []
        current_line = [boxes_with_centers[0]]
        
        for box, y_center in boxes_with_centers[1:]:
            prev_y_center = current_line[-1][1]
            
            if abs(y_center - prev_y_center) < y_threshold:
                current_line.append((box, y_center))
            else:
                lines.append(current_line)
                current_line = [(box, y_center)]
        
        lines.append(current_line)
        
        # Align each line to median y-coordinates
        aligned_frame_boxes = []
        for line in lines:
            if not line:
                continue
            
            # Get median y1 and y2 for this line
            y1_values = [box['bbox'][1] for box, _ in line]
            y2_values = [box['bbox'][3] for box, _ in line]
            
            median_y1 = int(np.median(y1_values))
            median_y2 = int(np.median(y2_values))
            
            # Apply to all boxes on this line
            for box, _ in line:
                aligned_box = box.copy()
                x1, _, x2, _ = box['bbox']
                aligned_box['bbox'] = (x1, median_y1, x2, median_y2)
                aligned_frame_boxes.append(aligned_box)
        
        aligned_boxes[frame_idx] = aligned_frame_boxes
    
    return aligned_boxes

def ocr_boxes_to_unified(frame_boxes):
    """
    Convert OCR box mapping to unified format:
    {frame_idx: [{'bbox': (x1,y1,x2,y2), 'score': float,
                  'text': str, 'mask': None, 'source': 'ocr',
                  'to_show': bool, 'alterego': str, 'name': str, 'track_id': int or None}, ...]}

    Args:
        frame_boxes (dict): Dictionary of frame indices to list of boxes

    Returns:
        dict: Unified format dictionary with track_id field
    """
    unified = {}
    for frame_idx, boxes in frame_boxes.items():
        unified_list = []
        for i, box in enumerate(boxes):
            bbox = box.get('bbox') or box.get('original_bbox')
            if not bbox:
                continue
            x1, y1, x2, y2 = _bbox_to_int_tuple(bbox)

            parent_box = box.get('parent_box', None)
            if parent_box is not None:
                try:
                    parent_box = _bbox_to_int_tuple(parent_box)
                except Exception:
                    parent_box = None

            # Get track_id — normalise word-level tuple (parent_tid, word_idx)
            # back to the parent integer so toggle propagation works correctly.
            track_id = box.get('track_id')
            if isinstance(track_id, tuple):
                track_id = track_id[0]
            
            entry = {
                "bbox": (x1, y1, x2, y2),
                "parent_box": parent_box,
                "score": float(box.get('confidence', 1.0)),
                "text": box.get('text', '') or '',
                "alterego": box.get('alterego', '') or '',
                "mask": None,
                "source": "ocr",
                "to_show": bool(box.get('to_show', True)),
                "track_id": track_id,
            }
            unified_list.append(entry)
        unified[frame_idx] = unified_list
    return unified

def process_video_ocr(video_path, output_dir, dict_path, languages=["en", "de"], num_workers=4,
                      extract_frames=True, frame_step=1, ocr_engine="easyocr",
                      change_detection=True, change_threshold=0.985):
    """
    Process a single video for OCR detection.
    
    Args:
        video_path (str or Path): Path to the video file
        output_dir (str or Path): Directory to save outputs (frames/, ocr.pkl, boxes_<ocr_engine>.pkl)
        dict_path (str or Path): Path to JSON file with names to detect
        languages (list): OCR language hints (default: ["en", "de"])
        num_workers (int): Number of parallel workers for processing
        extract_frames (bool): Whether to extract frames from video (default: True)
        frame_step (int): Step between frames to extract (default: 1)
        ocr_engine (str): OCR backend ('easyocr' or 'paddleocr')
        change_detection (bool): Whether to skip OCR for visually unchanged frames (default: True).
            This is the main performance optimization for screen recordings.
        change_threshold (float): Similarity threshold for frame change detection.
            Higher = more aggressive skipping. Range [0, 1]. Default 0.985.
        
    Returns:
        dict: Unified OCR data with track_ids
    """
    output_dir = Path(output_dir)
    ocr_engine = (ocr_engine or "easyocr").lower().strip()
    boxes_pkl_path = output_dir / f"boxes_{ocr_engine}.pkl"
    legacy_boxes_pkl_path = output_dir / "boxes.pkl"
    frames_dir = output_dir / "frames"
    output_pkl_path = output_dir / "ocr.pkl"
    
    print(f"\n{'='*60}")
    print(f"Processing video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"OCR engine: {ocr_engine}")
    if change_detection:
        print(f"Frame change detection: ENABLED (threshold={change_threshold})")
    else:
        print(f"Frame change detection: DISABLED")
    print(f"{'='*60}\n")
    
    # Extract frames if requested
    if extract_frames:
        frames_dir = extract_video_frames(video_path, output_dir=frames_dir, frame_step=frame_step)
    else:
        print(f"Using existing frames in {frames_dir}")
    
    # Load names dictionary with UTF-8 encoding
    with open(dict_path, 'r', encoding='utf-8') as f:
        names_to_detect = json.load(f)
    
    # Process frames or load existing boxes
    loaded_boxes = False
    if boxes_pkl_path.exists():
        with open(boxes_pkl_path, "rb") as f:
            frame_boxes = pickle.load(f)
        loaded_boxes = True
        print(f"Loaded existing text boxes from {boxes_pkl_path}")
    elif ocr_engine == "easyocr" and legacy_boxes_pkl_path.exists():
        with open(legacy_boxes_pkl_path, "rb") as f:
            frame_boxes = pickle.load(f)
        loaded_boxes = True
        print(f"Loaded legacy EasyOCR text boxes from {legacy_boxes_pkl_path}")

    if not loaded_boxes:
        if change_detection:
            frame_boxes = process_frames_sequential(
                frames_dir, languages, ocr_engine=ocr_engine,
                change_threshold=change_threshold
            )
        else:
            frame_boxes = process_frames_sequential(
                frames_dir, languages, ocr_engine=ocr_engine,
                change_threshold=0.0  # Disable skipping
            )
        with open(boxes_pkl_path, "wb") as f:
            pickle.dump(frame_boxes, f)
        print(f"Saved detected text boxes to {boxes_pkl_path}")
    
    # Apply OCR pipeline
    frame_boxes = merge_line_boxes(frame_boxes)
    
    # Normalize box heights first for easier tracking
    normalized_boxes, height_clusters = normalize_box_heights(frame_boxes)
    
    # Track and stabilize normalized boxes with more lenient thresholds
    tracked_boxes, ocr_tracks = enhanced_temporal_tracking(
        normalized_boxes,
        max_gap=6,
        position_threshold=50,  # More lenient for vertical movement
        size_threshold=0.4      # More lenient for size changes
    )
    
    # Split tracked boxes into words
    frame_boxes = split_boxes_into_words(tracked_boxes)
    
    # Filter by names (merges word boxes back into name boxes)
    filtered_frame_boxes = filter_boxes_by_names(frame_boxes, names_to_detect)
    
    # Convert to unified format
    unified = ocr_boxes_to_unified(filtered_frame_boxes)
    
    # Save final output
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(unified, f)
    
    print(f"Created {len(ocr_tracks)} OCR tracks across {len(unified)} frames")
    print(f"Saved final OCR boxes with track_ids to {output_pkl_path}")
    
    # Cleanup GPU memory after processing
    try:
        import gc
        gc.collect()
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass
    
    return unified


def process_videos_batch(video_paths, output_base_dir, dict_path, languages=["en", "de"],
                        num_workers=4, extract_frames=True, frame_step=1, ocr_engine="easyocr"):
    """
    Process multiple videos for OCR detection in batch.
    
    Args:
        video_paths (list): List of paths to video files
        output_base_dir (str or Path): Base directory for outputs (each video gets a subfolder)
        dict_path (str or Path): Path to JSON file with names to detect
        languages (list): OCR language hints (default: ["en", "de"])
        num_workers (int): Number of parallel workers for processing
        extract_frames (bool): Whether to extract frames from videos (default: True)
        frame_step (int): Step between frames to extract (default: 1)
        ocr_engine (str): OCR backend ('easyocr' or 'paddleocr')
        
    Returns:
        dict: Dictionary mapping video names to their unified OCR data
    """
    output_base_dir = Path(output_base_dir)
    results = {}
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING: {len(video_paths)} videos")
    print(f"{'='*60}\n")
    
    for video_path in video_paths:
        video_path = Path(video_path)
        video_name = video_path.stem  # Filename without extension
        
        # Create video-specific output directory
        video_output_dir = output_base_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Process the video
            unified = process_video_ocr(
                video_path=video_path,
                output_dir=video_output_dir,
                dict_path=dict_path,
                languages=languages,
                num_workers=num_workers,
                extract_frames=extract_frames,
                frame_step=frame_step,
                ocr_engine=ocr_engine
            )
            results[video_name] = unified
            print(f"✓ Successfully processed {video_name}")
        except Exception as e:
            print(f"✗ Error processing {video_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[video_name] = None
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"Successful: {sum(1 for v in results.values() if v is not None)}/{len(video_paths)}")
    print(f"{'='*60}\n")
    
    return results


def main():
    video_path = "all_videos/video_laet.mp4"
    output_dir = Path("tests/video_laet")
    dict_path = "laet.json"
    
    # Use the new wrapper function
    process_video_ocr(
        video_path=video_path,
        output_dir=output_dir,
        dict_path=dict_path,
        languages=["en", "de"],
        num_workers=4,
        extract_frames=True,
        frame_step=1
    )

if __name__ == "__main__":
    main()
