"""
Lightweight tracking for OCR boxes.

The raw docTR detections are already spatially stable — boxes track text
positions well frame-to-frame.  Per-line height normalization in the pipeline
handles within-frame height consistency.

This module only assigns track IDs so downstream consumers (SAM3, format.py)
can identify the same word across frames.  No coordinate smoothing is applied
— that was causing lag during scrolling.
"""

from __future__ import annotations

from collections import defaultdict

from .name_matching import normalize


# ---------------------------------------------------------------------------
# Track assignment (greedy nearest-neighbour matching)
# ---------------------------------------------------------------------------

def _assign_tracks(frame_boxes: dict) -> dict:
    """
    Assign integer track IDs to boxes across frames.

    Two boxes in consecutive frames get the same track_id if:
      - Their normalized text matches exactly
      - Their x-center is within 50px

    Returns:
        {frame_idx: [box_dict with track_id, ...]}
    """
    sorted_frames = sorted(frame_boxes.keys())
    if not sorted_frames:
        return {}

    next_track_id = 0
    active_tracks: dict[tuple, int] = {}
    out: dict = {}

    for fi in sorted_frames:
        boxes = frame_boxes[fi]
        new_active: dict[tuple, int] = {}
        frame_out = []

        for box in boxes:
            text = normalize(box.get("text", ""))
            x1, y1, x2, y2 = box["bbox"]
            x_center = (x1 + x2) // 2
            x_bucket = x_center // 50

            # Try same or adjacent x bucket
            matched_tid = None
            for dx in (0, -1, 1):
                key = (text, x_bucket + dx)
                if key in active_tracks:
                    matched_tid = active_tracks.pop(key)
                    break

            if matched_tid is None:
                matched_tid = next_track_id
                next_track_id += 1

            frame_out.append({**box, "track_id": matched_tid})
            new_active[(text, x_bucket)] = matched_tid

        out[fi] = frame_out
        active_tracks = new_active

    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def stabilize(frame_boxes: dict, similarities: dict = None,
              change_threshold: float = 0.999,
              smooth_window: int = 5) -> dict:
    """
    Assign track IDs to boxes across frames. No coordinate smoothing.

    Args:
        frame_boxes:      {frame_idx: [box_dict, ...]}
        similarities:     (unused, kept for API compat)
        change_threshold: (unused, kept for API compat)
        smooth_window:    (unused, kept for API compat)

    Returns:
        {frame_idx: [box_dict with track_id, ...]}
    """
    if not frame_boxes:
        return frame_boxes

    return _assign_tracks(frame_boxes)
