"""
SAM3 Pipeline Inspector
=======================
Runs each stage of the SAM3 profile-picture detection pipeline on a video
(or a directory of frames) and saves an annotated side-by-side video so you
can visually verify every step.

Output video layout — 4 panels horizontally:
  Panel 1 (left)           Raw SAM3 masks           coloured overlays, confidence labels
  Panel 2 (centre-left)    After overlap merge       merged masks highlighted
  Panel 3 (centre-right)   Tracked + circularized    circle outlines, track IDs
  Panel 4 (right)          Blur preview              blurred mask regions

Usage
-----
Edit the CONFIG block below, then run:

    python inspect_sam3_pipeline.py

Outputs a video file at OUTPUT_VIDEO.
"""

import os
import sys
import pickle
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# ===========================  CONFIG  =======================================
# ---------------------------------------------------------------------------

VIDEO_PATH      = r""           # path to input video  (or leave empty)
FRAMES_DIR      = r""           # path to existing frames dir (used if VIDEO_PATH is empty)
OUTPUT_VIDEO    = r""

OUTPUT_FPS      = 20            # fps for output video
TEXT_PROMPT     = "profile image, profile picture"
CONF            = 0.25          # SAM3 confidence threshold
DEVICE          = "auto"        # "auto", "cuda", "mps", "cpu"
MODEL_PATH      = "sam3.pt"

# If you only want to inspect a specific time range (in seconds):
START_SEC       = None          # e.g. 10.0  — None = from beginning
END_SEC         = None          # e.g. 20.0  — None = to end

# Mask overlay transparency
ALPHA           = 0.45

# Blur strength for panel 4
BLUR_STRENGTH   = 51

# Circle padding (same as production)
CIRCLE_PADDING_PX = 6.0

# ---------------------------------------------------------------------------
# ===========================  END CONFIG  ===================================
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent))

from ait.segmentation.engine import setup_predictor, process_image, get_image_files
from ait.segmentation.mask_ops import (slim_results, merge_overlapping_in_results,
                           merge_overlapping_in_frame, infer_img_shape)
from ait.segmentation.tracking import match_masks_across_frames, propagate_missing_masks
from ait.segmentation.circularize import circularize_results
from ait.utils import rebuild_full_mask


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
THICKNESS  = 1

# Colour palette for masks (BGR)
MASK_COLOURS = [
    (60,  60,  220),   # red
    (60,  200, 60),    # green
    (220, 120, 40),    # blue
    (40,  220, 220),   # yellow
    (200, 60,  200),   # magenta
    (220, 200, 40),    # cyan
]


def _overlay_masks(image, frame_results, img_shape, alpha=0.45,
                   label_fn=None, draw_bbox=True):
    """Draw coloured mask overlays + optional bbox rectangles and labels."""
    out = image.copy()
    if not frame_results or 'masks' not in frame_results:
        return out

    masks = frame_results.get('masks', [])
    boxes = frame_results.get('boxes', [])
    scores = frame_results.get('scores', [])
    n = min(len(masks), len(boxes))

    # Sort by y-center for consistent colour assignment
    order = list(range(n))
    order.sort(key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)

    for colour_idx, i in enumerate(order):
        colour = MASK_COLOURS[colour_idx % len(MASK_COLOURS)]
        try:
            mask_bool = rebuild_full_mask(masks[i], img_shape)
        except Exception:
            continue

        # Coloured overlay
        overlay = out.copy()
        overlay[mask_bool] = colour
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

        # Contour outline
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, colour, 2)

        # Bounding box
        if draw_bbox:
            x1, y1, x2, y2 = [int(round(v)) for v in boxes[i]]
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 1)

        # Label
        if label_fn:
            label = label_fn(i, frame_results)
            if label:
                x1, y1 = int(round(boxes[i][0])), int(round(boxes[i][1]))
                (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
                by1 = max(0, y1 - th - 4)
                cv2.rectangle(out, (x1, by1), (x1 + tw + 4, y1), colour, -1)
                cv2.putText(out, label, (x1 + 2, y1 - 2),
                            FONT, FONT_SCALE, (0, 0, 0), THICKNESS, cv2.LINE_AA)

    return out


def _draw_circles(image, frame_results, img_shape, track_map=None):
    """Draw circle outlines for circularized masks + track ID labels."""
    out = image.copy()
    if not frame_results or 'masks' not in frame_results:
        return out

    masks = frame_results.get('masks', [])
    boxes = frame_results.get('boxes', [])
    n = min(len(masks), len(boxes))

    order = list(range(n))
    order.sort(key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)

    for colour_idx, i in enumerate(order):
        colour = MASK_COLOURS[colour_idx % len(MASK_COLOURS)]
        try:
            mask_bool = rebuild_full_mask(masks[i], img_shape)
        except Exception:
            continue

        # Draw contour (circle outline)
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, colour, 2)

        # Semi-transparent fill
        overlay = out.copy()
        overlay[mask_bool] = colour
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)

        # Track ID label
        x1, y1 = int(round(boxes[i][0])), int(round(boxes[i][1]))
        track_id = track_map.get(i, "?") if track_map else "?"
        label = f"#{track_id}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 2)
        by1 = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, by1), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 2),
                    FONT, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    return out


def _blur_masks(image, frame_results, img_shape, blur_strength=51):
    """Gaussian-blur each mask region (final output preview)."""
    out = image.copy()
    if not frame_results or 'masks' not in frame_results:
        return out

    masks = frame_results.get('masks', [])
    for mask_entry in masks:
        try:
            mask_bool = rebuild_full_mask(mask_entry, img_shape)
        except Exception:
            continue
        ks = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        blurred = cv2.GaussianBlur(out, (ks, ks), 0)
        out[mask_bool] = blurred[mask_bool]

    return out


def _add_title(panel, title, colour=(220, 220, 220)):
    """Draw title + mask count in top-left."""
    cv2.putText(panel, title, (10, 22), FONT, 0.65, colour, 2, cv2.LINE_AA)
    return panel


def _add_count(panel, count_str, colour=(220, 220, 220)):
    cv2.putText(panel, count_str, (10, 44), FONT, 0.55, colour, 1, cv2.LINE_AA)
    return panel


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def _load_frames(video_path=None, frames_dir=None, start_sec=None, end_sec=None):
    """
    Load frames as {frame_idx: np.ndarray (BGR)}.
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

    paths = sorted(
        list(Path(frames_dir).glob("*.jpg")) + list(Path(frames_dir).glob("*.png")),
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
    sorted_ids = sorted(frames.keys())
    print(f"  {len(frames)} frames loaded")

    sample = frames[sorted_ids[0]]
    fh, fw = sample.shape[:2]
    img_shape = (fh, fw)

    # ------------------------------------------------------------------
    # Stage 1: Raw SAM3 detections (every frame)
    # ------------------------------------------------------------------
    print(f"\nStage 1: Running SAM3 inference (prompt: '{TEXT_PROMPT}')...")
    predictor, device = setup_predictor(DEVICE, MODEL_PATH, CONF)

    raw_results = {}        # frame_idx -> frame_results dict

    tmp_path = Path("_inspect_sam3_tmp.jpg")
    for fi in tqdm(sorted_ids, desc="SAM3 inference"):
        frame_bgr = frames[fi]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        Image.fromarray(frame_rgb).save(tmp_path)
        results = process_image(predictor, tmp_path, TEXT_PROMPT, fi)
        results = slim_results(results, img_shape, pad=10, pack_bits=True)
        raw_results[fi] = results

    if tmp_path.exists():
        tmp_path.unlink()

    total_masks = sum(len(r.get('masks', [])) for r in raw_results.values())
    print(f"  Processed {len(sorted_ids)} frames, {total_masks} masks detected")

    # Clean up model
    del predictor
    import gc
    gc.collect()
    from ait.utils import cleanup_device
    cleanup_device(device)

    # ------------------------------------------------------------------
    # Stage 2: Overlap merging
    # ------------------------------------------------------------------
    print("\nStage 2: Merging overlapping masks...")
    merged_results = {}
    total_merged = 0
    for fi in sorted_ids:
        fr = raw_results.get(fi)
        if fr and 'masks' in fr and len(fr.get('masks', [])) > 0:
            merged_fr, mc = merge_overlapping_in_frame(fr, img_shape)
            merged_results[fi] = merged_fr
            total_merged += mc
        else:
            merged_results[fi] = fr
    print(f"  Merged {total_merged} overlapping mask(s)")

    # ------------------------------------------------------------------
    # Stage 3: Tracking + Circularization
    # ------------------------------------------------------------------
    print("\nStage 3: Tracking + circularization...")

    # Build results list format for tracking/circularize functions
    # These expect [(list_idx, frame_results, img_path), ...]
    results_list = []
    fi_to_listidx = {}
    for list_idx, fi in enumerate(sorted_ids):
        fr = merged_results.get(fi, {'boxes': [], 'masks': [], 'scores': []})
        # Use a dummy path — circularize will fall back to box-based shape inference
        results_list.append((list_idx, fr, None))
        fi_to_listidx[fi] = list_idx

    tracks = match_masks_across_frames(results_list) if len(results_list) > 1 else []
    print(f"  Found {len(tracks)} tracks")

    # Propagate masks across gaps
    if tracks and len(results_list) > 1:
        filled = propagate_missing_masks(results_list, tracks, max_gap=5)
        print(f"  Propagated {filled} mask(s) across gaps")

    # Build track map per frame: {frame_list_idx: {mask_idx: track_id}}
    track_maps_by_listidx = {}
    for track_id, track in enumerate(tracks):
        for list_idx, mask_idx in track:
            track_maps_by_listidx.setdefault(list_idx, {})[mask_idx] = track_id

    # Update merged_results from results_list (propagation may have added masks)
    for list_idx, fi in enumerate(sorted_ids):
        merged_results[fi] = results_list[list_idx][1]

    # Circularize
    circular_results_list = circularize_results(
        results_list, tracks=tracks, circle_padding_px=CIRCLE_PADDING_PX
    )

    # Convert back to dict keyed by frame_idx
    circular_results = {}
    for list_idx, (_, fr, _) in enumerate(circular_results_list):
        fi = sorted_ids[list_idx]
        circular_results[fi] = fr

    # ------------------------------------------------------------------
    # Write output video
    # ------------------------------------------------------------------
    print(f"\nWriting inspection video → {OUTPUT_VIDEO}")

    out_w = fw * 4
    out_h = fh
    fps_out = OUTPUT_FPS

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps_out, (out_w, out_h))

    for fi in tqdm(sorted_ids, desc="Writing"):
        frame = frames[fi]
        raw = raw_results.get(fi, {})
        merged = merged_results.get(fi, {})
        circ = circular_results.get(fi, {})
        list_idx = fi_to_listidx[fi]
        track_map = track_maps_by_listidx.get(list_idx, {})

        n_raw = len(raw.get('masks', []))
        n_merged = len(merged.get('masks', []))
        n_circ = len(circ.get('masks', []))

        # Panel 1 — raw SAM3 detections
        p1 = _overlay_masks(
            frame, raw, img_shape, alpha=ALPHA,
            label_fn=lambda i, r: f"{r['scores'][i]:.2f}" if i < len(r.get('scores', [])) else None,
            draw_bbox=True,
        )
        _add_title(p1, "Stage 1: Raw SAM3", (220, 220, 220))
        _add_count(p1, f"{n_raw} masks", (220, 220, 220))

        # Panel 2 — after overlap merge
        p2 = _overlay_masks(
            frame, merged, img_shape, alpha=ALPHA,
            label_fn=lambda i, r: f"{r['scores'][i]:.2f}" if i < len(r.get('scores', [])) else None,
            draw_bbox=True,
        )
        merge_note = f"  (-{n_raw - n_merged} merged)" if n_raw > n_merged else ""
        _add_title(p2, "Stage 2: Overlap merge", (60, 200, 60))
        _add_count(p2, f"{n_merged} masks{merge_note}", (60, 200, 60))

        # Panel 3 — tracked + circularized
        p3 = _draw_circles(frame, circ, img_shape, track_map=track_map)
        _add_title(p3, "Stage 3: Tracked + circular", (255, 140, 0))
        _add_count(p3, f"{n_circ} masks, {len(tracks)} tracks", (255, 140, 0))

        # Panel 4 — blur preview
        p4 = _blur_masks(frame, circ, img_shape, blur_strength=BLUR_STRENGTH)
        _add_title(p4, "Stage 4: Blur preview", (200, 200, 255))

        stacked = np.hstack([p1, p2, p3, p4])

        cv2.putText(stacked, f"frame {fi}", (out_w - 120, out_h - 8),
                    FONT, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        writer.write(stacked)

    writer.release()
    print(f"\nDone.  Open {OUTPUT_VIDEO} to inspect the pipeline.")
    print(f"  Panel 1 (left):           Raw SAM3 masks + confidence")
    print(f"  Panel 2 (centre-left):    After overlap merge")
    print(f"  Panel 3 (centre-right):   Tracked + circularized (track IDs)")
    print(f"  Panel 4 (right):          Blur preview")


if __name__ == "__main__":
    main()
