"""
OCR pipeline orchestration.

Wires together:
  engine.py        — docTR word extraction
  name_matching.py — name detection
  stabilization.py — anchor-based coordinate stabilization
  format.py        — unified output format

Public API:
  process_video_ocr(...)
  process_videos_batch(...)
"""

import os
import json
import pickle
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from .engine import get_predictor, extract_words, cleanup as cleanup_engine
from .name_matching import filter_by_names
from .stabilization import stabilize
from .format import to_unified

from utils import extract_video_frames


# ---------------------------------------------------------------------------
# Frame similarity (unchanged from legacy ocr.py)
# ---------------------------------------------------------------------------

def _compute_frame_similarity(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Fast similarity score in [0, 1].  1.0 = identical frames."""
    small_a = cv2.resize(frame_a, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(frame_b, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    gray_a = cv2.cvtColor(small_a, cv2.COLOR_BGR2GRAY) if small_a.ndim == 3 else small_a
    gray_b = cv2.cvtColor(small_b, cv2.COLOR_BGR2GRAY) if small_b.ndim == 3 else small_b
    diff = cv2.absdiff(gray_a, gray_b)
    return 1.0 - float(np.mean(diff)) / 255.0


# ---------------------------------------------------------------------------
# Per-frame OCR with change-detection skip
# ---------------------------------------------------------------------------

def _run_ocr_on_frames(frames_dir: Path,
                        change_threshold: float = 0.999,
                        max_consecutive_skips: int = 30,
                        min_confidence: float = 0.4,
                        min_area: int = 100) -> tuple:
    """
    Run docTR on every frame in frames_dir, skipping visually unchanged frames.

    Returns:
        frame_boxes   {frame_idx: [word_box, ...]}
        similarities  {frame_idx: similarity_to_prev_frame}  (0.0 for first frame)
    """
    frame_paths = sorted(
        [frames_dir / f for f in os.listdir(frames_dir) if f.endswith((".jpg", ".png"))],
        key=lambda p: int("".join(filter(str.isdigit, p.stem))),
    )

    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")

    print(f"Found {len(frame_paths)} frames in {frames_dir}")
    predictor = get_predictor()

    frame_boxes: dict = {}
    similarities: dict = {}
    prev_frame = None             # previous frame (for consecutive comparison)
    prev_detections: list = []
    skipped = 0
    processed = 0
    consecutive_skips = 0

    for frame_path in tqdm(frame_paths, desc="OCR (docTR + change detection)"):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Warning: could not read {frame_path}")
            continue

        stem = frame_path.stem
        frame_idx = int("".join(filter(str.isdigit, stem)))

        if prev_frame is not None:
            # Compare consecutive frames — only skip if they are
            # virtually identical (threshold 0.999 by default).
            sim = _compute_frame_similarity(prev_frame, frame)
            similarities[frame_idx] = sim

            if sim >= change_threshold and consecutive_skips < max_consecutive_skips:
                frame_boxes[frame_idx] = [d.copy() for d in prev_detections]
                skipped += 1
                consecutive_skips += 1
                prev_frame = frame
                continue
            else:
                consecutive_skips = 0
        else:
            similarities[frame_idx] = 0.0

        detections = extract_words(frame, predictor=predictor,
                                   min_confidence=min_confidence,
                                   min_area=min_area)
        frame_boxes[frame_idx] = detections
        prev_frame = frame
        prev_detections = detections
        processed += 1

    total = skipped + processed
    if total > 0:
        print(f"\nFrame change detection summary:")
        print(f"  Total frames:     {total}")
        print(f"  OCR processed:    {processed} ({processed / total * 100:.1f}%)")
        print(f"  Skipped (static): {skipped} ({skipped / total * 100:.1f}%)")
        print(f"  Effective speedup: ~{total / max(1, processed):.1f}x")

    return frame_boxes, similarities


# ---------------------------------------------------------------------------
# Per-line height normalisation  (replaces K-means, uses docTR line_idx)
# ---------------------------------------------------------------------------

def _normalize_heights(frame_boxes: dict) -> dict:
    """
    Align all words on the same docTR line to the median y1/y2 of that line,
    and attach a ``parent_box`` (the bounding hull of the whole line) to every
    word box.

    This runs per-frame since lines are independent in each frame.
    """
    out = {}
    for frame_idx, boxes in frame_boxes.items():
        if not boxes:
            out[frame_idx] = []
            continue

        # Group by line_idx (fall back to y-center clustering if missing)
        line_map: dict = {}
        for i, box in enumerate(boxes):
            lid = box.get("line_idx", -1)
            line_map.setdefault(lid, []).append(i)

        adjusted = [b.copy() for b in boxes]
        for indices in line_map.values():
            y1_vals = [boxes[i]["bbox"][1] for i in indices]
            y2_vals = [boxes[i]["bbox"][3] for i in indices]
            med_y1 = int(np.median(y1_vals))
            med_y2 = int(np.median(y2_vals))

            # Compute parent_box = bounding hull of the full line
            line_x1 = min(boxes[i]["bbox"][0] for i in indices)
            line_x2 = max(boxes[i]["bbox"][2] for i in indices)
            parent_box = (line_x1, med_y1, line_x2, med_y2)

            for i in indices:
                x1, _, x2, _ = adjusted[i]["bbox"]
                adjusted[i] = {
                    **adjusted[i],
                    "bbox": (x1, med_y1, x2, med_y2),
                    "parent_box": parent_box,
                }

        out[frame_idx] = adjusted
    return out


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def process_video_ocr(video_path, output_dir, dict_path,
                       languages=None,           # kept for API compat, unused by docTR
                       num_workers=4,             # kept for API compat
                       extract_frames=True,
                       frame_step=1,
                       ocr_engine="doctr",        # legacy param, always uses docTR now
                       change_detection=True,
                       change_threshold=0.999):
    """
    Process a single video for OCR detection.

    Args:
        video_path:       Path to the video file.
        output_dir:       Directory for outputs (frames/, ocr.pkl, boxes_doctr.pkl).
        dict_path:        Path to JSON with names {"Full Name": "Alterego"}.
        extract_frames:   Whether to extract frames (set False to reuse existing).
        frame_step:       Extract every N-th frame.
        change_detection: Skip OCR on visually unchanged frames (default True).
        change_threshold: Similarity threshold for skipping. Default 0.999.

    Returns:
        dict: Unified OCR data {frame_idx: [box, ...]}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    boxes_pkl = output_dir / "boxes_doctr.pkl"
    sims_pkl = output_dir / "similarities_doctr.pkl"
    frames_dir = output_dir / "frames"
    output_pkl = output_dir / "ocr.pkl"

    print(f"\n{'=' * 60}")
    print(f"Processing: {video_path}")
    print(f"Output:     {output_dir}")
    print(f"Engine:     docTR")
    print(f"Change det: {'ON' if change_detection else 'OFF'} (threshold={change_threshold})")
    print(f"{'=' * 60}\n")

    # --- Step 1: extract frames ---
    if extract_frames:
        frames_dir = extract_video_frames(video_path, output_dir=frames_dir,
                                          frame_step=frame_step)
    else:
        print(f"Using existing frames in {frames_dir}")

    # --- Step 2: load or run OCR ---
    if boxes_pkl.exists() and sims_pkl.exists():
        with open(boxes_pkl, "rb") as f:
            frame_boxes = pickle.load(f)
        with open(sims_pkl, "rb") as f:
            similarities = pickle.load(f)
        print(f"Loaded existing detections from {boxes_pkl}")
    else:
        eff_threshold = change_threshold if change_detection else 0.0
        frame_boxes, similarities = _run_ocr_on_frames(
            frames_dir,
            change_threshold=eff_threshold,
        )
        with open(boxes_pkl, "wb") as f:
            pickle.dump(frame_boxes, f)
        with open(sims_pkl, "wb") as f:
            pickle.dump(similarities, f)
        print(f"Saved raw detections to {boxes_pkl}")

    # --- Step 3: per-line height normalisation ---
    frame_boxes = _normalize_heights(frame_boxes)

    # --- Step 4: tracking + height stabilization (on ALL boxes) ---
    frame_boxes = stabilize(frame_boxes)

    # --- Step 5: name matching (annotates all boxes, to_show=True/False) ---
    with open(dict_path, "r", encoding="utf-8") as f:
        names_dict = json.load(f)
    annotated_boxes = filter_by_names(frame_boxes, names_dict)

    # --- Step 7: convert to unified output format ---
    unified = to_unified(annotated_boxes)

    # --- Save ---
    with open(output_pkl, "wb") as f:
        pickle.dump(unified, f)

    n_matches = sum(len(v) for v in unified.values())
    print(f"\nDone: {n_matches} name detections across {len(unified)} frames")
    print(f"Saved OCR annotations to {output_pkl}")

    cleanup_engine()
    return unified


def process_videos_batch(video_paths, output_base_dir, dict_path,
                          languages=None, num_workers=4,
                          extract_frames=True, frame_step=1,
                          ocr_engine="doctr"):
    """
    Process multiple videos sequentially.

    Returns {video_stem: unified_dict_or_None}.
    """
    output_base_dir = Path(output_base_dir)
    results = {}

    print(f"\n{'=' * 60}")
    print(f"BATCH: {len(video_paths)} videos")
    print(f"{'=' * 60}\n")

    for video_path in video_paths:
        video_path = Path(video_path)
        video_output_dir = output_base_dir / video_path.stem
        video_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            unified = process_video_ocr(
                video_path=video_path,
                output_dir=video_output_dir,
                dict_path=dict_path,
                extract_frames=extract_frames,
                frame_step=frame_step,
            )
            results[video_path.stem] = unified
            print(f"✓  {video_path.stem}")
        except Exception as e:
            import traceback
            print(f"✗  {video_path.stem}: {e}")
            traceback.print_exc()
            results[video_path.stem] = None

    success = sum(1 for v in results.values() if v is not None)
    print(f"\nBATCH DONE: {success}/{len(video_paths)} succeeded")
    return results
