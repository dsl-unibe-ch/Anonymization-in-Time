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
from utils import resolve_device

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

def _resolve_paddle_language(languages):
    """Resolve a PaddleOCR language code from a list of language hints."""
    if not languages:
        return "en"
    if isinstance(languages, str):
        return languages.lower()
    normalized = [str(lang).lower().strip() for lang in languages if str(lang).strip()]
    if not normalized:
        return "en"
    # PaddleOCR accepts a single language code. Prefer English if present.
    if "en" in normalized:
        return "en"
    return normalized[0]

def get_paddleocr_reader(languages):
    """
    Initialize PaddleOCR reader.
    
    Returns:
        paddleocr.PaddleOCR: Initialized PaddleOCR reader instance
    """
    try:
        from paddleocr import PaddleOCR
    except Exception as e:
        raise ImportError(
            "PaddleOCR is not installed. Install it or choose ocr_engine='easyocr'."
        ) from e

    device, _ = 'cuda', 0 #select_device()
    paddle_lang = _resolve_paddle_language(languages)
    # PaddleOCR>=3 uses `device` (e.g., 'cpu', 'gpu:0').
    paddle_device = "gpu:0" if device == "cuda" else "cpu"
    print(f"Initializing PaddleOCR reader (lang={paddle_lang}, device={paddle_device})...")
    # Disable Paddle's document preprocessor for chat/screen-recording frames.
    # The doc orientation + unwarping stages can distort UI screenshots.
    reader = PaddleOCR(
        lang=paddle_lang,
        device=paddle_device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    print("PaddleOCR reader initialized.")
    return reader

def get_ocr_reader(languages, ocr_engine="easyocr"):
    """
    Initialize OCR reader for the requested backend.
    """
    engine = (ocr_engine or "easyocr").lower().strip()
    if engine == "easyocr":
        return get_easyocr_reader(languages)
    if engine == "paddleocr":
        return get_paddleocr_reader(languages)
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

def _extract_detections_from_paddle_result(result, min_confidence: float, scale_factor: float) -> list:
    """Normalize PaddleOCR output into TiT detection dicts."""
    detections = []
    if not result:
        return detections
    
    def to_xyxy(bbox):
        if bbox is None:
            return None
        if hasattr(bbox, "tolist"):
            bbox = bbox.tolist()
        if not isinstance(bbox, (list, tuple)):
            return None
        if len(bbox) == 4 and all(not isinstance(v, (list, tuple)) for v in bbox):
            x1, y1, x2, y2 = bbox
            return float(x1), float(y1), float(x2), float(y2)
        if len(bbox) >= 4 and isinstance(bbox[0], (list, tuple)):
            try:
                x_coords = [float(point[0]) for point in bbox]
                y_coords = [float(point[1]) for point in bbox]
            except Exception:
                return None
            return min(x_coords), min(y_coords), max(x_coords), max(y_coords)
        return None
    
    def append_detection(text, confidence, xyxy):
        if not text:
            return
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        if confidence < min_confidence:
            return
        if xyxy is None:
            return
        
        x_start, y_start, x_end, y_end = [int(round(v)) for v in xyxy]
        if scale_factor != 1:
            x_start = int(round(x_start / scale_factor))
            y_start = int(round(y_start / scale_factor))
            x_end = int(round(x_end / scale_factor))
            y_end = int(round(y_end / scale_factor))
        
        detections.append({
            'text': str(text).strip(),
            'bbox': (x_start, y_start, x_end, y_end),
            'confidence': confidence,
            'original_bbox': (x_start, y_start, x_end, y_end)
        })

    # Paddle may return nested shapes depending on version/input type.
    candidate_lines = result
    if isinstance(candidate_lines, (list, tuple)) and len(candidate_lines) == 1 and isinstance(candidate_lines[0], (list, tuple)):
        first = candidate_lines[0]
        if first and isinstance(first[0], (list, tuple)):
            candidate_lines = first

    for item in candidate_lines:
        # PaddleOCR>=3 returns OCRResult objects (or dicts) with a nested `res`.
        item_dict = None
        if isinstance(item, dict):
            item_dict = item
        elif hasattr(item, "res"):
            try:
                item_dict = {"res": getattr(item, "res")}
            except Exception:
                item_dict = None
        elif hasattr(item, "json"):
            try:
                raw_json = getattr(item, "json")
                if callable(raw_json):
                    raw_json = raw_json()
                if isinstance(raw_json, str):
                    item_dict = json.loads(raw_json)
                elif isinstance(raw_json, dict):
                    item_dict = raw_json
            except Exception:
                item_dict = None
        
        if isinstance(item_dict, dict):
            payload = item_dict.get("res", item_dict)
            if isinstance(payload, dict):
                texts = payload.get("rec_texts") or payload.get("texts") or []
                scores = payload.get("rec_scores") or payload.get("scores") or []
                polys = (
                    payload.get("rec_polys")
                    or payload.get("dt_polys")
                    or payload.get("polys")
                    or []
                )
                
                if isinstance(texts, str):
                    texts = [texts]
                if not isinstance(texts, (list, tuple)):
                    texts = []
                if not isinstance(scores, (list, tuple)):
                    scores = []
                if not isinstance(polys, (list, tuple)):
                    polys = []
                
                for idx, text in enumerate(texts):
                    xyxy = to_xyxy(polys[idx]) if idx < len(polys) else None
                    conf = scores[idx] if idx < len(scores) else 1.0
                    append_detection(text, conf, xyxy)
                continue
        
        # Legacy PaddleOCR format: [bbox, (text, confidence)]
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        bbox = item[0]
        text_info = item[1]
        if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
            continue
        xyxy = to_xyxy(bbox)
        append_detection(text_info[0], text_info[1], xyxy)

    return detections

def _ensure_three_channel_uint8(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is uint8 HxWx3 for PaddleOCR v3 pipelines.
    """
    if image is None:
        return image

    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 1:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif channels == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif channels != 3:
            # Keep first 3 channels as a conservative fallback.
            arr = arr[:, :, :3]
    else:
        raise ValueError(f"Unsupported image shape for OCR: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr)

def extract_text_with_paddleocr(image: np.ndarray, reader: Any, min_confidence: float = 0.5) -> list:
    """
    Extract text from an image using PaddleOCR.
    """
    # processed_image, scale_factor = preprocess_image_for_ocr(image)
    processed_image = _ensure_three_channel_uint8(image)
    if hasattr(reader, "predict"):
        result = reader.predict(processed_image)
    else:
        result = reader.ocr(processed_image)
    return _extract_detections_from_paddle_result(result, min_confidence, 1.0)

def extract_text_with_backend(image: np.ndarray, reader: Any, ocr_engine="easyocr", min_confidence: float = 0.5) -> list:
    """
    Dispatch OCR extraction to selected backend.
    """
    engine = (ocr_engine or "easyocr").lower().strip()
    if engine == "easyocr":
        return extract_text_with_easyocr(image, reader, min_confidence=min_confidence)
    if engine == "paddleocr":
        return extract_text_with_paddleocr(image, reader, min_confidence=min_confidence)
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

def process_frames_sequential(frames_dir, languages, ocr_engine="easyocr"):
    """
    Process frames sequentially (GPU-safe).
    
    Args:
        frames_dir (str): Directory containing the video frames.
        languages (list): List of languages for OCR.
        
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
    for frame_path in tqdm(frame_paths, desc="Processing Frames"):
        result = process_frame(frame_path, reader, ocr_engine=ocr_engine)
        if result is not None:
            frame_idx, detections = result
            frame_boxes[frame_idx] = detections
    
    return frame_boxes


# Keep alias for backward compatibility
def process_frames_parallel(frames_dir, languages, num_workers=-1, force_cpu=False, ocr_engine="easyocr"):
    """Alias for process_frames_sequential (parallel removed due to CUDA issues)."""
    return process_frames_sequential(frames_dir, languages, ocr_engine=ocr_engine)

def merge_line_boxes(frame_boxes, x_threshold=20):
    """
    Merge horizontally adjacent boxes that likely belong to the same text line segment.
    Boxes are merged only when they are on the same line, have similar heights, and are within a horizontal gap threshold.
    
    Args:
        frame_boxes (dict): Dictionary of frame indices to list of bounding boxes
        x_threshold (int): Maximum horizontal distance between boxes to consider merging
        
    Returns:
        dict: Updated frame_boxes with merged entries
    """
    merged_frame_boxes = {}

    def _height(b):
        return float(b['bbox'][3] - b['bbox'][1])

    def _vertical_overlap_ratio(a, b):
        ay1, ay2 = a['bbox'][1], a['bbox'][3]
        by1, by2 = b['bbox'][1], b['bbox'][3]
        overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
        min_h = max(1.0, min(_height(a), _height(b)))
        return overlap / min_h

    def _height_similarity_ok(a, b, max_rel_diff=0.20):
        ha = max(1.0, _height(a))
        hb = max(1.0, _height(b))
        return abs(ha - hb) / max(ha, hb) <= max_rel_diff

    def _same_line_and_mergeable(a, b):
        # Require strong vertical overlap and similar text height.
        if _vertical_overlap_ratio(a, b) < 0.6:
            return False
        if not _height_similarity_ok(a, b):
            return False

        # b should be to the right of a (allow slight overlap from OCR jitter).
        gap = b['bbox'][0] - a['bbox'][2]
        if gap < -10 or gap > x_threshold:
            return False
        return True

    def _merge_two(a, b):
        ax1, ay1, ax2, ay2 = _bbox_to_int_tuple(a['bbox'])
        bx1, by1, bx2, by2 = _bbox_to_int_tuple(b['bbox'])
        new_x1 = min(ax1, bx1)
        new_y1 = min(ay1, by1)
        new_x2 = max(ax2, bx2)
        new_y2 = max(ay2, by2)

        merged = a.copy()
        merged['bbox'] = (new_x1, new_y1, new_x2, new_y2)
        merged['text'] = ((a.get('text', '') or '').strip() + " " + (b.get('text', '') or '').strip()).strip()
        merged['confidence'] = (float(a.get('confidence', 1.0)) + float(b.get('confidence', 1.0))) / 2.0

        # Keep original_bbox as a union too, so downstream normalization sees a stable parent.
        a_orig = a.get('original_bbox', a['bbox'])
        b_orig = b.get('original_bbox', b['bbox'])
        aox1, aoy1, aox2, aoy2 = _bbox_to_int_tuple(a_orig)
        box1, boy1, box2, boy2 = _bbox_to_int_tuple(b_orig)
        merged['original_bbox'] = (
            min(aox1, box1),
            min(aoy1, boy1),
            max(aox2, box2),
            max(aoy2, boy2),
        )
        return merged

    for frame_idx, boxes in frame_boxes.items():
        if not boxes:
            merged_frame_boxes[frame_idx] = []
            continue
            
        # Sort boxes by y (top to bottom) then x (left to right)
        # Using center y for better line alignment handling
        sorted_boxes = sorted(boxes, key=lambda b: ((b['bbox'][1] + b['bbox'][3])/2, b['bbox'][0]))
        
        merged_boxes = []
        used_indices = set()

        i = 0
        while i < len(sorted_boxes):
            if i in used_indices:
                i += 1
                continue

            current = sorted_boxes[i]
            used_indices.add(i)

            # Greedy chain-merge to the right while boxes remain on the same line and
            # have similar heights. This keeps words from the same phrase together
            # without merging across different UI text sizes.
            merged_any = True
            while merged_any:
                merged_any = False
                for j in range(i + 1, min(i + 10, len(sorted_boxes))):
                    if j in used_indices:
                        continue
                    candidate = sorted_boxes[j]
                    if _same_line_and_mergeable(current, candidate):
                        current = _merge_two(current, candidate)
                        used_indices.add(j)
                        merged_any = True
                        break

            merged_boxes.append(current)
            i += 1
            
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
                # Single word or empty - keep as is
                box_copy = box.copy()
                if 'bbox' in box_copy:
                    box_copy['bbox'] = _bbox_to_int_tuple(box_copy['bbox'])
                if box_copy.get('parent_box') is not None:
                    box_copy['parent_box'] = _bbox_to_int_tuple(box_copy['parent_box'])
                split_boxes.append(box_copy)
                continue
            
            # Multi-word box - split it proportionally based on word lengths
            x1, y1, x2, y2 = _bbox_to_int_tuple(box['bbox'])
            parent_bbox = _bbox_to_int_tuple(box.get('original_bbox', box['bbox']))
            box_width = x2 - x1
            
            # Estimate padding (typically ~2% of box width or min 2 pixels on each side)
            padding = max(2, int(box_width * 0.02))
            
            # Calculate total characters in all words
            total_word_chars = sum(len(word) for word in words)
            
            # Split the content area (box minus padding) proportionally among words
            content_start = x1 + padding
            content_end = x2 - padding
            content_width = max(1, content_end - content_start)
            
            current_x = x1  # Start from original box edge (includes padding for first word)
            word_boxes = []
            
            for i, word in enumerate(words):
                word_length = len(word)
                
                if i == 0:
                    # First word: start from parent box edge (includes left padding)
                    word_x1 = x1
                    # End at proportional position in content area
                    word_proportion = word_length / total_word_chars
                    word_x2 = content_start + int(word_proportion * content_width)
                elif i == len(words) - 1:
                    # Last word: start where previous ended, extend to parent box edge (includes right padding)
                    word_x1 = current_x
                    word_x2 = x2
                else:
                    # Middle words: split proportionally within content area
                    word_x1 = current_x
                    word_proportion = word_length / total_word_chars
                    word_x2 = content_start + int(sum(len(words[j]) for j in range(i + 1)) / total_word_chars * content_width)
                
                word_x1 = int(max(x1, min(x2, word_x1)))
                word_x2 = int(max(word_x1, min(x2, word_x2)))
                
                word_boxes.append({
                    'bbox': (word_x1, y1, word_x2, y2),
                    'text': word,
                    'confidence': box.get('confidence', 1.0),
                    'parent_box': parent_bbox,
                    'parent_box_text': text,  # Save full parent text for matching
                    'track_id': box.get('track_id', None)  # Preserve track_id from parent
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
        return text.lower().strip()
    
    def word_matches_name_part(word_text, name_part):
        """Check if word matches a part of a name"""
        norm_word = normalize_text(word_text)
        norm_name = normalize_text(name_part)
        
        # Exact match
        if norm_word == norm_name:
            return True, 1.0
        
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
        """Find sequences of adjacent word boxes that match names"""
        matches = []
        
        for name, alterego in names_dict.items():
            name_words = normalize_text(name).split()
            
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
                
                # Check if we matched all words in the name
                if word_idx == len(name_words):
                    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                    matches.append({
                        'name': name,
                        'alterego': alterego,
                        'boxes': matched_boxes,
                        'confidence': avg_confidence
                    })
        
        # Remove overlapping matches, keep higher confidence ones
        if not matches:
            return []
        
        matches.sort(key=lambda x: (len(x['boxes']), x['confidence']), reverse=True)
        
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
        track_ids = [b.get('track_id') for b in boxes if b.get('track_id') is not None]
        track_id = track_ids[0] if track_ids else None
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
    
    
    def stabilize_coordinates(current_bbox, prev_bbox, movement_threshold=3):
        """
        Stabilize coordinates based on movement direction.
        
        Args:
            current_bbox (tuple): Current frame bounding box (x1, y1, x2, y2)
            prev_bbox (tuple): Previous frame bounding box (x1, y1, x2, y2)
            movement_threshold (int): Minimum movement to consider as actual movement
            
        Returns:
            tuple: Stabilized bounding box coordinates
        """
        curr_x1, curr_y1, curr_x2, curr_y2 = current_bbox
        prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
        
        def union_bbox(a, b):
            """Keep full coverage across minor OCR size fluctuations."""
            return (
                int(min(a[0], b[0])),
                int(min(a[1], b[1])),
                int(max(a[2], b[2])),
                int(max(a[3], b[3])),
            )
        
        # Calculate centers and movements
        curr_center_x = (curr_x1 + curr_x2) / 2
        curr_center_y = (curr_y1 + curr_y2) / 2
        prev_center_x = (prev_x1 + prev_x2) / 2
        prev_center_y = (prev_y1 + prev_y2) / 2
        
        x_movement = abs(curr_center_x - prev_center_x)
        y_movement = abs(curr_center_y - prev_center_y)
        
        # Calculate current dimensions
        curr_width = curr_x2 - curr_x1
        curr_height = curr_y2 - curr_y1
        prev_width = prev_x2 - prev_x1
        prev_height = prev_y2 - prev_y1
        
        # Determine movement type and stabilize accordingly
        if x_movement < movement_threshold and y_movement < movement_threshold:
            # Static center with significant size shift often means OCR changed
            # text extent. Keep the union to avoid shrinking true text coverage.
            width_change = abs(curr_width - prev_width) / max(1, prev_width)
            height_change = abs(curr_height - prev_height) / max(1, prev_height)
            if max(width_change, height_change) > 0.15:
                return union_bbox(current_bbox, prev_bbox)
            return _bbox_to_int_tuple(prev_bbox)
            
        elif y_movement < movement_threshold or y_movement < x_movement / 2:
            # Horizontal movement: preserve Y coordinates, use current X but maintain consistent width
            stable_width = max(prev_width, curr_width)  # Prevent width collapse
            stable_y1 = prev_y1
            stable_y2 = prev_y2
            
            # Use current center X but with stable width
            stable_x1 = int(round(curr_center_x - stable_width / 2))
            stable_x2 = stable_x1 + int(stable_width)
            
            return (stable_x1, stable_y1, stable_x2, stable_y2)
            
        elif x_movement < movement_threshold or x_movement < y_movement / 2:
            # Vertical movement: preserve X coordinates, use current Y but maintain consistent height
            stable_height = max(prev_height, curr_height)  # Prevent height collapse
            stable_x1 = prev_x1
            stable_x2 = prev_x2
            
            # Use current center Y but with stable height
            stable_y1 = int(round(curr_center_y - stable_height / 2))
            stable_y2 = stable_y1 + int(stable_height)
            
            return (stable_x1, stable_y1, stable_x2, stable_y2)
            
        else:
            # Diagonal movement: use current coordinates but try to maintain consistent dimensions
            # Prefer previous dimensions if current ones are too different
            width_diff = abs(curr_width - prev_width) / prev_width if prev_width > 0 else 0
            height_diff = abs(curr_height - prev_height) / prev_height if prev_height > 0 else 0
            
            final_width = max(prev_width, curr_width) if width_diff > 0.2 else curr_width
            final_height = max(prev_height, curr_height) if height_diff > 0.2 else curr_height
            
            # Center the box with stable dimensions
            stable_x1 = int(round(curr_center_x - final_width / 2))
            stable_y1 = int(round(curr_center_y - final_height / 2))
            stable_x2 = stable_x1 + int(final_width)
            stable_y2 = stable_y1 + int(final_height)
            
            return (stable_x1, stable_y1, stable_x2, stable_y2)

    if not frame_boxes:
        return frame_boxes

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
            
            # Find best matching box
            best_match = None
            best_idx = None
            
            for box_idx, box in unmatched_boxes:
                if can_track(last_box, box):
                    if best_match is None:
                        best_match = box
                        best_idx = box_idx
            
            if best_match is not None:
                track.append((frame_idx, best_match))
                unmatched_boxes = [(idx, box) for idx, box in unmatched_boxes if idx != best_idx]
        
        # Create new tracks for unmatched boxes
        for _, box in unmatched_boxes:
            tracks.append([(frame_idx, box)])
    
    # Apply stabilization to each track
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
        
        # Multi-frame track - find best text representation
        text_variants = [(box.get('text', ''), box.get('confidence', 0)) for _, box in track]
        # Use the text with highest confidence
        best_text = max(text_variants, key=lambda x: x[1])[0]
        
        # Multi-frame track - apply frame-to-frame stabilization
        for i, (frame_idx, box) in enumerate(track):
            box_idx_in_stabilized = len(stabilized_boxes[frame_idx])
            
            if i == 0:
                # First frame in track - keep original coordinates
                stabilized_box = box.copy()
                stabilized_box['text'] = best_text  # Use consistent text
                stabilized_box['track_id'] = track_id
                stabilized_box['bbox'] = _bbox_to_int_tuple(stabilized_box['bbox'])
            else:
                # Stabilize based on previous frame
                prev_frame, prev_box = track[i-1]
                stabilized_prev_bbox = stabilized_boxes[prev_frame][-1]['bbox']  # Get the stabilized previous box
                
                stabilized_bbox = stabilize_coordinates(box['bbox'], stabilized_prev_bbox)
                
                stabilized_box = box.copy()
                stabilized_box['bbox'] = stabilized_bbox
                stabilized_box['text'] = best_text  # Use consistent text
                stabilized_box['track_id'] = track_id  # Assign track_id directly to box
            
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

def ocr_boxes_to_unified(frame_boxes, tracks=None):
    """
    Convert OCR box mapping to unified format:
    {frame_idx: [{'bbox': (x1,y1,x2,y2), 'score': float, 'label': None,
                  'text': str, 'mask': None, 'source': 'ocr',
                  'to_show': bool, 'alterego': str, 'name': str, 'track_id': int or None}, ...]}
    
    Args:
        frame_boxes (dict): Dictionary of frame indices to list of boxes
        tracks (list): List of tracks, each track is a list of (frame_idx, box_idx) tuples
    
    Returns:
        dict: Unified format dictionary with track_id field
    """
    # Build track_map: (frame_idx, box_idx) -> track_id
    track_map = {}
    if tracks:
        for track_id, track in enumerate(tracks):
            for frame_idx, box_idx in track:
                track_map[(frame_idx, box_idx)] = track_id
    
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
            
            # Get track_id for this box
            track_id = box.get('track_id')
            if track_id is None:
                track_id = track_map.get((frame_idx, i), None)
            
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
                      extract_frames=True, frame_step=1, ocr_engine="easyocr"):
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
        frame_boxes = process_frames_parallel(frames_dir, languages, num_workers, ocr_engine=ocr_engine)
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
    unified = ocr_boxes_to_unified(filtered_frame_boxes, tracks=ocr_tracks)
    
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
