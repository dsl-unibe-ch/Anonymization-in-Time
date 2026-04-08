"""
OCR Pipeline Inspector
======================
Runs each stage of the new docTR OCR pipeline on a video (or a directory of
frames) and saves an annotated side-by-side video so you can visually verify
every step.

Output video layout — 2×2 grid:
  Panel 1 (top-left)      Raw docTR words          white boxes, confidence labels
  Panel 2 (top-right)     After name matching       green boxes for matched names
  Panel 3 (bottom-left)   Tracked names             orange boxes, track IDs
  Panel 4 (bottom-right)  Blur preview              blurred name boxes + alterego overlay

Usage
-----
Edit the CONFIG block below, then run:

    python inspect_ocr_pipeline.py

Outputs a video file at OUTPUT_VIDEO_PATH.
"""

import os
import sys
import json
import pickle
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# ===========================  CONFIG  =======================================
# ---------------------------------------------------------------------------

VIDEO_PATH      = r""           # path to input video  (or leave empty)
FRAMES_DIR      = r""           # path to existing frames dir (used if VIDEO_PATH is empty)
NAMES_JSON      = r""           # path to names dict JSON
OUTPUT_VIDEO    = r""

OUTPUT_FPS      = 20            # fps for output video (use low value to read labels)

# If you only want to inspect a specific time range (in seconds):
START_SEC       = None          # e.g. 10.0  — None = from beginning
END_SEC         = None          # e.g. 20.0  — None = to end

# Frame-change detection (same as production)
CHANGE_THRESHOLD = 0.999

# OCR engine: "doctr" or "easyocr"
OCR_ENGINE      = "easyocr"
LANGUAGES       = ["en", "de"]          # language list for EasyOCR (ignored by docTR)

# Confidence / area thresholds
MIN_CONFIDENCE  = 0.3
MIN_AREA        = 100

# ---------------------------------------------------------------------------
# ===========================  END CONFIG  ===================================
# ---------------------------------------------------------------------------

# Add project root to path so ait package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from ait.ocr.engine import get_predictor, extract_words, get_easyocr_reader, extract_words_easyocr
from ait.ocr.name_matching import filter_by_names
from ait.ocr.pipeline import _normalize_heights
from ait.ocr.stabilization import stabilize
from ait.ocr import pipeline as _pipe


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
THICKNESS  = 1

# Panel colour schemes
COLOUR_RAW    = (220, 220, 220)   # white-ish  — raw docTR words
COLOUR_LINE   = (0,   180, 180)  # cyan       — parent/line boxes
COLOUR_MATCH  = (0,   220,  0)   # green      — name matches
COLOUR_STABLE = (255, 140,  0)   # orange     — stabilized + track_id


def _draw_boxes(image: np.ndarray, boxes: list, colour, label_fn=None) -> np.ndarray:
    """Draw boxes onto a copy of image.  label_fn(box) -> str or None."""
    out = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        if label_fn:
            label = label_fn(box)
            if label:
                (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
                # Background pill
                by1 = max(0, y1 - th - 4)
                cv2.rectangle(out, (x1, by1), (x1 + tw + 4, y1), colour, -1)
                cv2.putText(out, label, (x1 + 2, y1 - 2),
                            FONT, FONT_SCALE, (0, 0, 0), THICKNESS, cv2.LINE_AA)
    return out


def _draw_parent_boxes(image: np.ndarray, boxes: list, colour) -> np.ndarray:
    """Draw unique parent_box rectangles (thin, 1px) from a list of word boxes."""
    out = image.copy()
    seen = set()
    for box in boxes:
        pb = box.get("parent_box")
        if pb is None:
            continue
        key = tuple(pb)
        if key in seen:
            continue
        seen.add(key)
        x1, y1, x2, y2 = [int(v) for v in pb]
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 1)
    return out


def _annotate_panel(frame: np.ndarray, boxes: list,
                     colour, label_fn, title: str,
                     bg_boxes: list = None,
                     show_parent_boxes: bool = False) -> np.ndarray:
    """
    Draw bg_boxes (dim gray), parent_box outlines (cyan), then boxes on top.
    """
    panel = frame.copy()
    if bg_boxes:
        panel = _draw_boxes(panel, bg_boxes, (80, 80, 80), label_fn=None)
        if show_parent_boxes:
            panel = _draw_parent_boxes(panel, bg_boxes, COLOUR_LINE)
    if show_parent_boxes:
        panel = _draw_parent_boxes(panel, boxes, COLOUR_LINE)
    panel = _draw_boxes(panel, boxes, colour, label_fn)
    cv2.putText(panel, title, (10, 22), FONT, 0.65, colour, 2, cv2.LINE_AA)
    if bg_boxes is not None:
        count_str = f"{len(bg_boxes)} words  |  {len(boxes)} matches"
    else:
        count_str = f"{len(boxes)} words"
    cv2.putText(panel, count_str, (10, 44), FONT, 0.55, colour, 1, cv2.LINE_AA)
    return panel


def _blur_and_label(frame: np.ndarray, boxes: list, title: str) -> np.ndarray:
    """
    For each to_show box: Gaussian-blur the bbox region, then draw the
    alterego name centered on top.
    """
    panel = frame.copy()
    for box in boxes:
        if not box.get("to_show"):
            continue
        x1, y1, x2, y2 = box["bbox"]
        # Clamp to frame bounds
        h, w = panel.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            continue

        # Blur the region
        roi = panel[y1c:y2c, x1c:x2c]
        ksize = max(15, ((y2c - y1c) // 2) | 1)  # odd kernel, proportional to box height
        blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
        panel[y1c:y2c, x1c:x2c] = blurred

        # Draw alterego text centered in the box
        alterego = box.get("alterego", "")
        if alterego:
            box_w = x2c - x1c
            box_h = y2c - y1c
            # Pick font scale to fit the box width
            scale = 0.5
            (tw, th), _ = cv2.getTextSize(alterego, FONT, scale, 1)
            if tw > box_w * 0.9 and tw > 0:
                scale *= (box_w * 0.9) / tw
                (tw, th), _ = cv2.getTextSize(alterego, FONT, scale, 1)
            tx = x1c + (box_w - tw) // 2
            ty = y1c + (box_h + th) // 2
            # Shadow for readability
            cv2.putText(panel, alterego, (tx + 1, ty + 1), FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(panel, alterego, (tx, ty), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(panel, title, (10, 22), FONT, 0.65, (200, 200, 255), 2, cv2.LINE_AA)
    return panel


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def _load_frames(video_path=None, frames_dir=None, start_sec=None, end_sec=None):
    """
    Load frames as {frame_idx: np.ndarray}.
    Either extracts from video or reads a directory of images.
    Returns (frame_dict, fps).
    """
    if video_path:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int((start_sec or 0) * fps)
        end_frame   = int((end_sec or total / fps) * fps)
        frames = {}
        for fi in tqdm(range(start_frame, min(end_frame, total)), desc="Loading frames"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                frames[fi] = frame
        cap.release()
        return frames, fps

    # From directory
    paths = sorted(
        Path(frames_dir).glob("*.jpg"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)))
    )
    paths += sorted(
        Path(frames_dir).glob("*.png"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)))
    )
    frames = {}
    for p in tqdm(paths, desc="Loading frames"):
        fi = int("".join(filter(str.isdigit, p.stem)))
        frame = cv2.imread(str(p))
        if frame is not None:
            frames[fi] = frame
    return frames, 25.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    assert VIDEO_PATH or FRAMES_DIR, "Set VIDEO_PATH or FRAMES_DIR in the CONFIG block."
    assert NAMES_JSON, "Set NAMES_JSON in the CONFIG block."

    print("Loading frames...")
    frames, src_fps = _load_frames(
        video_path=VIDEO_PATH or None,
        frames_dir=FRAMES_DIR or None,
        start_sec=START_SEC,
        end_sec=END_SEC,
    )
    if not frames:
        print("No frames loaded — check your paths.")
        return
    print(f"  {len(frames)} frames loaded")

    with open(NAMES_JSON, "r", encoding="utf-8") as f:
        names_dict = json.load(f)

    # ------------------------------------------------------------------
    # Stage 1: Raw OCR detections  +  similarity tracking
    # ------------------------------------------------------------------
    engine = OCR_ENGINE.lower().strip()
    print(f"\nStage 1: Running {engine} OCR...")

    if engine == "easyocr":
        reader = get_easyocr_reader(LANGUAGES)
        def _extract(frame):
            return extract_words_easyocr(frame, reader=reader,
                                         min_confidence=MIN_CONFIDENCE,
                                         min_area=MIN_AREA)
    else:
        predictor = get_predictor()
        def _extract(frame):
            return extract_words(frame, predictor=predictor,
                                 min_confidence=MIN_CONFIDENCE,
                                 min_area=MIN_AREA)

    raw_boxes: dict  = {}
    similarities: dict = {}
    sorted_ids = sorted(frames.keys())
    prev_frame = None
    prev_dets  = []
    consecutive_skips = 0

    for fi in tqdm(sorted_ids, desc=engine):
        frame = frames[fi]
        if prev_frame is not None:
            sim = _pipe._compute_frame_similarity(prev_frame, frame)
            similarities[fi] = sim
            if sim >= CHANGE_THRESHOLD and consecutive_skips < 30:
                raw_boxes[fi] = [d.copy() for d in prev_dets]
                consecutive_skips += 1
                prev_frame = frame
                continue
            consecutive_skips = 0
        else:
            similarities[fi] = 0.0

        dets = _extract(frame)
        raw_boxes[fi] = dets
        prev_frame = frame
        prev_dets  = dets

    print(f"  Total words detected across all frames: {sum(len(v) for v in raw_boxes.values())}")

    # ------------------------------------------------------------------
    # Stage 2: Height normalisation + tracking
    # ------------------------------------------------------------------
    print("\nStage 2: Height normalisation + tracking...")
    normalised = _pipe._normalize_heights(raw_boxes)
    tracked = stabilize(normalised)

    # ------------------------------------------------------------------
    # Stage 3: Name matching (annotates all boxes)
    # ------------------------------------------------------------------
    print("\nStage 3: Name matching...")
    annotated = filter_by_names(tracked, names_dict)
    n_matches = sum(sum(1 for b in v if b.get("to_show")) for v in annotated.values())
    print(f"  Name matches: {n_matches} across {sum(1 for v in annotated.values() if any(b.get('to_show') for b in v))} frames")

    # ------------------------------------------------------------------
    # Write output video
    # ------------------------------------------------------------------
    print(f"\nWriting inspection video → {OUTPUT_VIDEO}")

    sample_frame = frames[sorted_ids[0]]
    fh, fw = sample_frame.shape[:2]

    # 4 panels horizontally → output width = 4 * fw
    out_w = fw * 4
    out_h = fh
    fps_out = OUTPUT_FPS

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps_out, (out_w, out_h))

    for fi in tqdm(sorted_ids, desc="Writing"):
        frame = frames[fi]

        raw = raw_boxes.get(fi, [])
        norm = normalised.get(fi, [])
        ann = annotated.get(fi, [])
        ann_show = [b for b in ann if b.get("to_show")]
        ann_bg = [b for b in ann if not b.get("to_show")]

        # Panel 1 — raw detections
        if engine == "easyocr":
            # Show original line-level boxes (from parent_box) instead of split words
            seen_pb = {}
            line_boxes = []
            for b in raw:
                pb = b.get("parent_box")
                if pb and pb not in seen_pb:
                    # Reconstruct line text from all words sharing this parent_box
                    line_text = " ".join(w["text"] for w in raw if w.get("parent_box") == pb)
                    line_boxes.append({**b, "bbox": pb, "text": line_text})
                    seen_pb[pb] = True
            p1 = _annotate_panel(
                frame, line_boxes, COLOUR_RAW,
                label_fn=lambda b: f"{b['text']} {b['confidence']:.2f}",
                title="Stage 1: EasyOCR raw lines",
            )
        else:
            p1 = _annotate_panel(
                frame, norm, COLOUR_RAW,
                label_fn=lambda b: f"{b['text']} {b['confidence']:.2f}",
                title="Stage 1: docTR words + line boxes",
                show_parent_boxes=True,
            )

        # Panel 2 (top-right) — all words (gray) + name matches (green)
        p2 = _annotate_panel(
            frame, ann_show, COLOUR_MATCH,
            label_fn=lambda b: f"{b['text']} → {b.get('alterego', '')}" if b.get("alterego") else b["text"],
            title="Stage 2: Name matching",
            bg_boxes=ann_bg,
            show_parent_boxes=True,
        )

        # Panel 3 (bottom-left) — all words (gray) + name matches (orange) with track IDs
        p3 = _annotate_panel(
            frame, ann_show, COLOUR_STABLE,
            label_fn=lambda b: f"#{b.get('track_id', '?')} {b['text']}",
            title="Stage 3: Tracked names",
            bg_boxes=ann_bg,
            show_parent_boxes=True,
        )

        # Panel 4 (bottom-right) — blur preview with alterego overlay
        p4 = _blur_and_label(frame, ann, "Stage 4: Blur + alterego")

        # Similarity badge on panel 1
        sim = similarities.get(fi, 1.0)
        sim_colour = (0, 200, 0) if sim >= CHANGE_THRESHOLD else (0, 0, 200)
        cv2.putText(p1, f"sim={sim:.4f}", (fw - 160, 22),
                    FONT, 0.55, sim_colour, 1, cv2.LINE_AA)

        stacked = np.hstack([p1, p2, p3, p4])

        cv2.putText(stacked, f"frame {fi}", (out_w - 120, out_h - 8),
                    FONT, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        writer.write(stacked)

    writer.release()
    print(f"\nDone.  Open {OUTPUT_VIDEO} to inspect the pipeline.")
    print(f"  Panel 1 (left):           {OCR_ENGINE} words (white) + line boxes (cyan)")
    print(f"  Panel 2 (center-left):    name matches (green) on gray words")
    print(f"  Panel 3 (center-right):   tracked names (orange) + track IDs")
    print(f"  Panel 4 (right):          blur + alterego preview")


if __name__ == "__main__":
    main()
