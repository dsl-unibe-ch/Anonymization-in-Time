"""
Lightweight tracking for OCR boxes.

The raw docTR detections are already spatially stable — boxes track text
positions well frame-to-frame.  Per-line height normalization in the pipeline
handles within-frame height consistency.

This module assigns track IDs and then locks each track to a fixed
**height** (median of y2-y1 across all frames).  The y-center from the
raw detection is preserved — only the extent above/below center is
standardized.  This eliminates height flicker without introducing any
positional lag.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict

from .name_matching import normalize


# ---------------------------------------------------------------------------
# Track assignment (greedy nearest-neighbour matching)
# ---------------------------------------------------------------------------

def _assign_tracks(frame_boxes: dict, max_dist: int = 80) -> dict:
    """
    Assign integer track IDs to boxes across frames.

    Two boxes in consecutive frames get the same track_id if:
      - Their normalized name/text matches exactly
      - They are the closest pair (by Euclidean center distance) among
        all candidates with the same text, within max_dist pixels

    Uses distance-based matching to handle duplicate names correctly
    (e.g. three "Arch" boxes at different positions).

    Returns:
        {frame_idx: [box_dict with track_id, ...]}
    """
    sorted_frames = sorted(frame_boxes.keys())
    if not sorted_frames:
        return {}

    next_track_id = 0
    # active_tracks: {norm_text: [(x_center, y_center, track_id), ...]}
    active_tracks: dict[str, list] = defaultdict(list)
    out: dict = {}

    for fi in sorted_frames:
        boxes = frame_boxes[fi]
        new_active: dict[str, list] = defaultdict(list)
        frame_out = []

        # Build list of (box, text, center) for this frame
        current = []
        for box in boxes:
            text = normalize(box.get("name", box.get("text", "")))
            x1, y1, x2, y2 = box["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            current.append((box, text, cx, cy))

        # For each unique text, match current boxes to active tracks
        # by closest distance (Hungarian-lite: greedy closest-first)
        texts_in_frame = set(t for _, t, _, _ in current)
        text_assignments: dict[int, int] = {}  # box index -> track_id

        for text in texts_in_frame:
            prev_entries = active_tracks.get(text, [])
            curr_indices = [i for i, (_, t, _, _) in enumerate(current) if t == text]

            if not prev_entries:
                # All new tracks
                for i in curr_indices:
                    text_assignments[i] = next_track_id
                    next_track_id += 1
                continue

            # Build distance pairs and sort by distance
            pairs = []
            for ci in curr_indices:
                _, _, cx, cy = current[ci]
                for pi, (px, py, ptid) in enumerate(prev_entries):
                    dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    pairs.append((dist, ci, pi, ptid))
            pairs.sort()

            used_curr = set()
            used_prev = set()
            for dist, ci, pi, ptid in pairs:
                if ci in used_curr or pi in used_prev:
                    continue
                if dist > max_dist:
                    break
                text_assignments[ci] = ptid
                used_curr.add(ci)
                used_prev.add(pi)

            # Unmatched current boxes get new tracks
            for ci in curr_indices:
                if ci not in text_assignments:
                    text_assignments[ci] = next_track_id
                    next_track_id += 1

        # Build output and new_active
        for i, (box, text, cx, cy) in enumerate(current):
            tid = text_assignments[i]
            frame_out.append({**box, "track_id": tid})
            new_active[text].append((cx, cy, tid))

        out[fi] = frame_out
        active_tracks = new_active

    return out


# ---------------------------------------------------------------------------
# Fix height per track (preserve y-center, lock height)
# ---------------------------------------------------------------------------

def _fix_track_heights(frame_boxes: dict) -> dict:
    """
    For each track_id, compute the median box height (y2-y1) across all
    frames.  Then re-apply that fixed height to every occurrence, keeping
    the original y-center so the box doesn't shift vertically.
    """
    # Collect heights per track
    track_heights: dict[int, list] = defaultdict(list)
    for fi, boxes in frame_boxes.items():
        for box in boxes:
            tid = box.get("track_id")
            if tid is None:
                continue
            _, y1, _, y2 = box["bbox"]
            track_heights[tid].append(y2 - y1)

    # Median height per track
    track_med_h: dict[int, int] = {}
    for tid, heights in track_heights.items():
        track_med_h[tid] = int(np.median(heights))

    # Apply: keep y-center, set fixed height
    out = {}
    for fi, boxes in frame_boxes.items():
        frame_out = []
        for box in boxes:
            tid = box.get("track_id")
            if tid is not None and tid in track_med_h:
                x1, y1, x2, y2 = box["bbox"]
                y_center = (y1 + y2) / 2
                half_h = track_med_h[tid] / 2
                new_y1 = int(y_center - half_h)
                new_y2 = int(y_center + half_h)
                new_box = {**box, "bbox": (x1, new_y1, x2, new_y2)}
                if box.get("parent_box"):
                    px1, _, px2, _ = box["parent_box"]
                    new_box["parent_box"] = (px1, new_y1, px2, new_y2)
                frame_out.append(new_box)
            else:
                frame_out.append({**box})
        out[fi] = frame_out
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def stabilize(frame_boxes: dict, similarities: dict = None,
              change_threshold: float = 0.999,
              smooth_window: int = 5) -> dict:
    """
    Assign track IDs and lock each track to a fixed height.

    Args:
        frame_boxes:      {frame_idx: [box_dict, ...]}
        similarities:     (unused, kept for API compat)
        change_threshold: (unused, kept for API compat)
        smooth_window:    (unused, kept for API compat)

    Returns:
        {frame_idx: [box_dict with track_id and fixed height, ...]}
    """
    if not frame_boxes:
        return frame_boxes

    tracked = _assign_tracks(frame_boxes)
    return _fix_track_heights(tracked)
