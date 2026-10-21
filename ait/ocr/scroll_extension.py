"""
Keep name boxes going for a few frames after OCR loses them in a scroll.

As a name scrolls out of the frame (or under an app's header / bottom bar)
its letters get cut off, OCR stops reading it, and the last frame or two of
the name go unblurred. The same happens, reversed, as a name scrolls in.

When a name track ends (or starts) while moving, its box is carried on by the
measured scroll: the median motion of the other text moving with it (a chat
list scrolls as one), else its own last motion. It stops once it has moved
far enough that the edge cutting it off must hide all of it, leaves the
frame, or runs into a detected name box.

First, a word matched as a name in some frames of its track is shown in all
of them (see _show_whole_tracks).

Added boxes are copies of the track's last box (same track_id, so hiding the
name in the viewer hides them too) flagged ``extrapolated=True``. Their bbox
keeps its full size and may run past the frame edge; consumers clamp it.
"""

from collections import defaultdict

import numpy as np


def _centre(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _shift(bbox, d):
    x1, y1, x2, y2 = bbox
    dx, dy = int(round(d[0])), int(round(d[1]))
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def _step_motions(frame_boxes, order):
    """motions[i]: centre displacements, order[i] -> order[i+1], of every track in both frames."""
    motions = []
    prev = {}
    for i, fi in enumerate(order):
        cur = {b["track_id"]: _centre(b["bbox"])
               for b in frame_boxes.get(fi, []) if b.get("track_id") is not None}
        if i > 0:
            motions.append(np.array([cur[t] - prev[t] for t in cur.keys() & prev.keys()]).reshape(-1, 2))
        prev = cur
    return motions


def _dominant_motion(step, min_speed, min_tracks=3):
    """Median motion of the moving tracks in one step, or None if fewer than min_tracks move."""
    moving = step[np.hypot(*step.T) >= min_speed] if len(step) else step
    if len(moving) < min_tracks:
        return None
    return np.median(moving, axis=0)


def _show_whole_tracks(frame_boxes):
    """
    Show every box of a track that was matched as a name in any frame.

    A track is one word followed across frames, so a word that was part of a
    name stays one -- but matching needs the whole name or its start, and
    when the first name gets cut off by the edge (misread "Lillian" ->
    "Liman"), the surname alone stops matching while still readable.
    """
    shown = {}
    for boxes in frame_boxes.values():
        for box in boxes:
            if box.get("to_show") and box.get("track_id") is not None:
                shown.setdefault(box["track_id"], box)
    out = {}
    for fi, boxes in frame_boxes.items():
        out[fi] = [{**box, "to_show": True,
                    **{k: shown[box["track_id"]][k] for k in ("name", "alterego", "confidence")
                       if k in shown[box["track_id"]]}}
                   if not box.get("to_show") and box.get("track_id") in shown else box
                   for box in boxes]
    return out


def _coast(frame_boxes, order, frame_size, max_frames, min_speed):
    """Extend name tracks past their last occurrence in ``order``. Returns {frame_idx: [new box]}."""
    width, height = frame_size
    motions = _step_motions(frame_boxes, order)

    seen = defaultdict(list)  # track_id -> [(pos, box), ...] in order
    for pos, fi in enumerate(order):
        for box in frame_boxes.get(fi, []):
            if box.get("track_id") is not None:
                seen[box["track_id"]].append((pos, box))

    added = defaultdict(list)
    for occurrences in seen.values():
        pos, box = occurrences[-1]
        if not box.get("to_show") or pos == len(order) - 1:
            continue
        if len(occurrences) >= 2 and pos - occurrences[-2][0] <= 3:
            # Own motion from the previous occurrence (allowing a short OCR gap)
            prev_pos, prev_box = occurrences[-2]
            v = (_centre(box["bbox"]) - _centre(prev_box["bbox"])) / (pos - prev_pos)
        else:
            # Seen once (e.g. misread "Mava" in the frame before "Maya"):
            # take the scroll of the text around it
            v = _dominant_motion(motions[pos - 1] if pos > 0 else motions[0], min_speed)
            if v is None:
                continue
        if np.hypot(*v) < min_speed:
            continue  # not scrolling: the name really went away

        x1, y1, x2, y2 = box["bbox"]
        size = np.array([x2 - x1, y2 - y1], dtype=float)
        travelled = np.zeros(2)
        for k in range(1, min(max_frames, len(order) - 1 - pos) + 1):
            step = motions[pos + k - 1]
            # Text moving with this name; tolerance lets the scroll slow down
            tol = max(3.0, 0.5 * np.hypot(*v))
            co_moving = step[np.hypot(*(step - v).T) <= tol] if len(step) else step
            if len(co_moving):
                v = np.median(co_moving, axis=0)
            travelled += v
            # The edge cutting the name off lies at most one step before its
            # last detected box; past that by the box size, nothing shows.
            if np.any(np.abs(travelled) >= size + np.abs(v)):
                break
            new_bbox = _shift(box["bbox"], travelled)
            if (new_bbox[2] <= 0 or new_bbox[0] >= width or
                    new_bbox[3] <= 0 or new_bbox[1] >= height):
                break
            fi = order[pos + k]
            if any(b.get("to_show") and _iou(b["bbox"], new_bbox) > 0.3
                   for b in frame_boxes.get(fi, [])):
                break  # OCR picked the name up again
            new_box = {**box, "bbox": new_bbox, "extrapolated": True}
            if box.get("parent_box"):
                new_box["parent_box"] = _shift(box["parent_box"], travelled)
            added[fi].append(new_box)
    return added


def extend_scrolling_names(frame_boxes: dict, frame_size: tuple,
                           max_frames: int = 10, min_speed: float = 2.0) -> dict:
    """
    Add boxes for names scrolling out of (and into) view after OCR lost them.

    Args:
        frame_boxes: {frame_idx: [box, ...]} after name matching (track_id, to_show).
        frame_size:  (width, height) of the frames in pixels.
        max_frames:  most frames a name is carried on, per end of its track.
        min_speed:   px per frame below which a track counts as static (not extended).

    Returns:
        A new {frame_idx: [box, ...]} with the added boxes appended.
    """
    order = sorted(frame_boxes)
    if len(order) < 2:
        return frame_boxes
    frame_boxes = _show_whole_tracks(frame_boxes)
    out = {fi: list(boxes) for fi, boxes in frame_boxes.items()}
    n_added = 0
    for direction in (order, order[::-1]):  # scrolling out, then scrolling in
        for fi, boxes in _coast(frame_boxes, direction, frame_size, max_frames, min_speed).items():
            out[fi].extend(boxes)
            n_added += len(boxes)
    if n_added:
        print(f"Extended scrolling names: {n_added} box(es) added past the OCR detections")
    return out
