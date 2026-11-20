"""
Steady OCR boxes over time, as the last step of the OCR pipeline.

docTR's box for the same word wobbles by 1-3 px between frames (sometimes
much more) in position and size; where one name is split into word boxes,
the split point moves too. The alterego, centred in the box, jumped with it.

The page's scroll per frame is measured as the median movement of all tracks
between consecutive frames. Each box then gets the median width, height and
position relative to the page of the same track's boxes within +-window
frames, moved with the page: it follows (even uneven) scrolling without
lagging, and words of one name stay steady relative to each other. Text that
stays put while the page scrolls (a chat title) is steadied in absolute
position instead.

The steadied box becomes ``bbox`` (what the viewer shows and where the
alterego is drawn); the detection is kept as ``detected_bbox``, and the export
hides the union of both, so steadying never uncovers detected text.
"""

from __future__ import annotations

import bisect
from collections import defaultdict

import numpy as np


def steady_ocr_boxes(unified: dict, window: int = 5) -> dict:
    """
    Return a copy of unified OCR annotations ({frame_idx: [box, ...]}) with
    each tracked box's ``bbox`` steadied and the original kept in
    ``detected_bbox``.
    """
    steady = stabilized_text_boxes(unified, window=window, include_hidden=True)
    out = {}
    for frame_idx, boxes in unified.items():
        frame_out = []
        for box in boxes:
            new = steady.get(id(box))
            if new is None or tuple(new) == tuple(box["bbox"]):
                frame_out.append({**box})
            else:
                frame_out.append({**box, "bbox": new, "detected_bbox": tuple(box["bbox"])})
        out[frame_idx] = frame_out
    return out


def stabilized_text_boxes(ocr_annotations, window=5, include_hidden=False):
    """
    Steadied box per OCR annotation (see module docstring).

    Args:
        ocr_annotations: {frame_idx: [OCR annotation dict, ...]}
        window: frames on each side to take the median over.
        include_hidden: also steady boxes with to_show=False (the pipeline
            does, so a box the reviewer switches on is steady too).

    Returns:
        dict mapping id(annotation) -> (x1, y1, x2, y2). Annotations without a
        track_id (or hidden ones, unless include_hidden) are not included.
    """
    frames = sorted(ocr_annotations)
    page = _page_offsets(ocr_annotations, frames)

    by_track = defaultdict(list)
    for frame_idx in frames:
        for ann in ocr_annotations[frame_idx]:
            if ann.get("track_id") is None or not ann.get("bbox"):
                continue
            if not include_hidden and not ann.get("to_show", True):
                continue
            by_track[ann["track_id"]].append((frame_idx, ann))
    boxes = {}
    for entries in by_track.values():
        track_frames = [f for f, _ in entries]
        for frame_idx, ann in entries:
            lo = bisect.bisect_left(track_frames, frame_idx - window)
            hi = bisect.bisect_right(track_frames, frame_idx + window)
            near = entries[lo:hi]
            bb = np.array([a["bbox"] for _, a in near], dtype=np.float64)
            off = np.array([page[f] for f, _ in near])
            here = page[frame_idx]
            w = float(np.median(bb[:, 2] - bb[:, 0]))
            h = float(np.median(bb[:, 3] - bb[:, 1]))
            cx = _steady((bb[:, 0] + bb[:, 2]) / 2, off[:, 0], here[0])
            cy = _steady((bb[:, 1] + bb[:, 3]) / 2, off[:, 1], here[1])
            boxes[id(ann)] = (int(round(cx - w / 2)), int(round(cy - h / 2)),
                              int(round(cx + w / 2)), int(round(cy + h / 2)))
    return boxes


def _page_offsets(ocr_annotations, frames):
    """
    Cumulative page scroll (x, y) per frame: the median movement, between
    consecutive frames, of all OCR tracks present in both (shown or not).
    """
    page, prev, total = {}, {}, np.zeros(2)
    for frame_idx in frames:
        cur = {}
        for ann in ocr_annotations[frame_idx]:
            if ann.get("track_id") is not None and ann.get("bbox"):
                x1, y1, x2, y2 = ann["bbox"]
                cur[ann["track_id"]] = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        moves = [cur[t] - prev[t] for t in cur.keys() & prev.keys()]
        if moves:
            total = total + np.median(np.stack(moves), axis=0)
        page[frame_idx] = total.copy()
        prev = cur
    return page


def _steady(centres, page_offsets, page_here):
    """
    Steady centre for the middle frame of a window: the median position
    either relative to the page (for scrolling text) or absolute (for text
    that stays put, like a chat title over a scrolling chat), whichever
    explains the window's positions better.
    """
    relative = centres - page_offsets
    rel_med, abs_med = np.median(relative), np.median(centres)
    if np.median(np.abs(relative - rel_med)) <= np.median(np.abs(centres - abs_med)):
        return float(rel_med + page_here)
    return float(abs_med)


def hidden_box(box):
    """Region to hide for an OCR box: union of bbox and detected_bbox."""
    x1, y1, x2, y2 = box["bbox"]
    det = box.get("detected_bbox")
    if det:
        x1, y1 = min(x1, det[0]), min(y1, det[1])
        x2, y2 = max(x2, det[2]), max(y2, det[3])
    return (x1, y1, x2, y2)
