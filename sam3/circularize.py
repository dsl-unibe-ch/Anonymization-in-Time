"""
Circle fitting and per-track stabilization for SAM3 masks.

Creates circular masks around detected profile pictures:
  1. Extract circle params from each mask (contour → minEnclosingCircle)
  2. Detect partial/clipped circles (semicircles at viewport edges)
  3. Per-track: median radius + centered median smooth for centers
  4. Generate padded circle masks

No global radius clustering — each track uses its own median radius.
"""

import math
import numpy as np
import cv2
from PIL import Image

from src.utils.mask_utils import rebuild_full_mask
from .mask_ops import mask_bbox_from_bool, make_mask_entry


def _circle_info_from_mask(full_mask, img_shape=None):
    """
    Estimate circle params and quality flags from a boolean mask.

    Returns dict with:
      - params: (cx, cy, r)
      - bbox: tight bbox of the visible mask
      - coverage: visible_area / full_circle_area
      - bbox_aspect: min(w, h) / max(w, h)
      - touches_frame_edge: whether visible mask touches video frame border
      - is_partial: likely clipped/partial circle
    """
    if full_mask is None:
        return None

    mask = np.asarray(full_mask).astype(bool)
    if mask.size == 0 or not mask.any():
        return None

    # Fill interior holes so they don't shrink the area-based radius.
    mask_u8 = mask.astype(np.uint8) * 255
    contours_info = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    if contours:
        cv2.drawContours(mask_u8, contours, -1, 255, cv2.FILLED)
        mask = mask_u8 > 0

    bbox = mask_bbox_from_bool(mask)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    bbox_aspect = float(min(bw, bh)) / float(max(bw, bh))

    # Center = centroid of filled mask pixels, radius = area-equivalent circle.
    ys, xs = np.nonzero(mask)
    cx = float(xs.mean())
    cy = float(ys.mean())

    visible_area = float(mask.sum())
    r = float(math.sqrt(visible_area / math.pi))

    # Cap radius to half the smaller bbox dimension — bbox is tight and reliable.
    max_r = float(min(bw, bh)) / 2.0
    if r > max_r:
        r = max_r

    if not np.isfinite(r) or r <= 0:
        return None

    circle_area = float(math.pi * r * r)
    coverage = visible_area / max(circle_area, 1.0)

    touches_frame_edge = False
    if img_shape is not None and len(img_shape) >= 2:
        img_h, img_w = int(img_shape[0]), int(img_shape[1])
        if img_h > 1 and img_w > 1:
            touches_frame_edge = (
                x1 <= 1 or y1 <= 1 or x2 >= (img_w - 2) or y2 >= (img_h - 2)
            )

    # Partial/clipped: non-square bbox (semicircle at viewport edge) or
    # touching frame edge with a non-square shape.
    is_partial = (
        bbox_aspect < 0.78
        or (touches_frame_edge and bbox_aspect < 0.92)
    )

    return {
        "params": (cx, cy, r),
        "bbox": bbox,
        "coverage": float(coverage),
        "bbox_aspect": float(bbox_aspect),
        "touches_frame_edge": bool(touches_frame_edge),
        "is_partial": bool(is_partial),
    }


def _circle_crop_from_params(cx, cy, r, img_shape):
    """Return (crop_mask, bbox) for a circle, or (None, None) if invalid."""
    img_h, img_w = img_shape
    x1 = int(max(0, math.floor(cx - r)))
    y1 = int(max(0, math.floor(cy - r)))
    x2 = int(min(img_w - 1, math.ceil(cx + r)))
    y2 = int(min(img_h - 1, math.ceil(cy + r)))
    if x2 < x1 or y2 < y1:
        return None, None
    yy, xx = np.ogrid[y1:y2 + 1, x1:x2 + 1]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    return circle.astype(bool), [x1, y1, x2, y2]


def circularize_results(all_results, tracks=None, circle_padding_px=6.0):
    """
    Create circular masks around each SAM3 mask and return a new results list.

    Per-track stabilization:
      - Radius: median of non-partial frames (no global clustering)
      - Centers: centered_median_smooth (symmetric window, zero lag)

    Clipped/partial masks keep the original SAM3 mask.
    """
    circular_results = []

    # Cache frame shapes
    frame_shapes = {}
    for list_idx, (frame_idx, frame_results, img_path) in enumerate(all_results):
        img_h = img_w = None
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception:
            pass
        if img_h is None or img_w is None:
            boxes = frame_results.get('boxes', []) if frame_results else []
            if boxes:
                max_x2 = max(int(round(b[2])) for b in boxes)
                max_y2 = max(int(round(b[3])) for b in boxes)
                img_w = max_x2 + 1
                img_h = max_y2 + 1
            else:
                img_w = img_h = 1
        frame_shapes[list_idx] = (img_h, img_w)

    # First pass: compute circle info per mask
    mask_info_cache = {}
    params_cache = {}
    for list_idx, (frame_idx, frame_results, img_path) in enumerate(all_results):
        if not frame_results or 'masks' not in frame_results:
            params_cache[(list_idx, None)] = None
            continue

        masks = list(frame_results.get('masks', []))
        img_shape = frame_shapes.get(list_idx, (1, 1))

        for i, mask_entry in enumerate(masks):
            full_mask = rebuild_full_mask(mask_entry, img_shape)
            info = _circle_info_from_mask(full_mask, img_shape=img_shape)
            mask_info_cache[(list_idx, i)] = info
            params_cache[(list_idx, i)] = None if info is None else info["params"]

    # Build track lookup
    track_id_by_key = {}
    if tracks:
        for track_id, track in enumerate(tracks):
            for pair in track:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                track_id_by_key[(int(pair[0]), int(pair[1]))] = int(track_id)

    # Per-track median radius (no global clustering)
    track_target_radius = {}
    if tracks:
        for track_id, track in enumerate(tracks):
            radii_good = []
            radii_all = []
            for list_idx, mask_idx in track:
                info = mask_info_cache.get((list_idx, mask_idx))
                if info is None or info.get("params") is None:
                    continue
                r = float(info["params"][2])
                radii_all.append(r)
                if not info.get("is_partial", False):
                    radii_good.append(r)
            radii = radii_good if radii_good else radii_all
            if not radii:
                continue
            track_target_radius[track_id] = float(np.median(radii))

    # Per-track params: raw centers + median radius (no center smoothing)
    stabilized_params = {}
    if tracks:
        for track_id, track in enumerate(tracks):
            target_r = track_target_radius.get(track_id)
            for list_idx, mask_idx in track:
                info = mask_info_cache.get((list_idx, mask_idx))
                if info is None or info.get("params") is None:
                    continue
                raw_cx, raw_cy, raw_r = info["params"]
                r = float(target_r if target_r is not None else raw_r)
                stabilized_params[(list_idx, mask_idx)] = (float(raw_cx), float(raw_cy), r)

    # Final pass: build circular results
    for list_idx, (frame_idx, frame_results, img_path) in enumerate(all_results):
        if not frame_results or 'masks' not in frame_results:
            circular_results.append((frame_idx, frame_results, img_path))
            continue

        boxes = list(frame_results.get('boxes', []))
        masks = list(frame_results.get('masks', []))
        scores = list(frame_results.get('scores', []))
        labels = list(frame_results.get('labels', [])) if frame_results.get('labels') is not None else []

        img_shape = frame_shapes.get(list_idx, (1, 1))

        new_boxes = []
        new_masks = []
        new_scores = []
        new_labels = []

        def _append_original(mask_idx, mask_entry):
            if mask_idx >= len(boxes):
                return
            box = boxes[mask_idx]
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            score = scores[mask_idx] if mask_idx < len(scores) else 0.0
            label = labels[mask_idx] if mask_idx < len(labels) else 1
            new_boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
            new_masks.append(mask_entry)
            new_scores.append(float(score))
            new_labels.append(int(label))

        for i, mask_entry in enumerate(masks):
            key = (list_idx, i)
            info = mask_info_cache.get(key)
            params = stabilized_params.get(key)
            if params is None:
                params = params_cache.get(key)
                # For un-tracked masks, use raw radius
                if params is not None:
                    track_id = track_id_by_key.get(key)
                    if track_id in track_target_radius:
                        params = (float(params[0]), float(params[1]),
                                  float(track_target_radius[track_id]))

            # Keep original SAM3 mask for clipped/partial circles
            if info is not None and info.get("is_partial", False):
                _append_original(i, mask_entry)
                continue

            if params is None:
                _append_original(i, mask_entry)
                continue

            padded_r = float(params[2]) + max(0.0, float(circle_padding_px))
            crop, bbox = _circle_crop_from_params(params[0], params[1], padded_r, img_shape)
            if crop is None or bbox is None:
                _append_original(i, mask_entry)
                continue

            pack_output = isinstance(mask_entry, dict) and "packed" in mask_entry
            mask_out = make_mask_entry(crop, bbox, pack_bits=pack_output)
            score = scores[i] if i < len(scores) else 0.0
            label = labels[i] if i < len(labels) else 1
            new_boxes.append(np.array(bbox, dtype=np.float32))
            new_masks.append(mask_out)
            new_scores.append(float(score))
            new_labels.append(int(label))

        circular_results.append((
            frame_idx,
            {'boxes': new_boxes, 'masks': new_masks, 'scores': new_scores, 'labels': new_labels},
            img_path
        ))

    return circular_results
