"""
Mask packing, slimming, overlap merging, and helpers.
"""

import math
import numpy as np
import cv2
from PIL import Image

from ait.utils import unpack_mask_entry, rebuild_full_mask


def _pad_box(box, pad, img_h, img_w):
    """Pad and clamp a bounding box."""
    x1, y1, x2, y2 = box
    x1 = max(0, int(math.floor(x1 - pad)))
    y1 = max(0, int(math.floor(y1 - pad)))
    x2 = min(img_w - 1, int(math.ceil(x2 + pad)))
    y2 = min(img_h - 1, int(math.ceil(y2 + pad)))
    return [x1, y1, x2, y2]


def slim_results(results, img_shape, pad=4, pack_bits=True):
    """
    Crop masks to bounding boxes and optionally pack to save memory.

    Args:
        results: dict with 'boxes', 'masks', 'scores', 'labels'
        img_shape: (H, W) of original image
        pad: pixels to expand each box before cropping
        pack_bits: if True, pack cropped mask to bytes to save RAM
    """
    if not results or 'masks' not in results or len(results['masks']) == 0:
        return results

    boxes = results.get('boxes', [])
    masks = results.get('masks', [])
    scores = results.get('scores', [])
    labels = results.get('labels', [])

    img_h, img_w = img_shape[:2]

    new_boxes = []
    new_masks = []
    new_scores = []
    new_labels = []

    for box, mask, score, label in zip(boxes, masks, scores, labels):
        box_np = np.array(box, dtype=float)
        mask_np = np.array(mask)

        if mask_np.ndim == 3:
            mask_np = mask_np[0]

        px1, py1, px2, py2 = _pad_box(box_np, pad, img_h, img_w)
        crop = mask_np[py1:py2+1, px1:px2+1].astype(np.bool_)

        if pack_bits:
            packed = np.packbits(crop.reshape(-1).astype(np.uint8))
            new_masks.append({
                "packed": packed,
                "shape": crop.shape,
                "bbox": [px1, py1, px2, py2],
            })
        else:
            new_masks.append(crop)

        new_boxes.append(np.array([px1, py1, px2, py2], dtype=np.float32))
        new_scores.append(float(score))
        new_labels.append(int(label))

    return {
        'boxes': new_boxes,
        'masks': new_masks,
        'scores': new_scores,
        'labels': new_labels,
    }


def make_mask_entry(crop_bool, bbox, pack_bits=True):
    """Pack a boolean crop with its bbox."""
    if pack_bits:
        packed = np.packbits(crop_bool.reshape(-1).astype(np.uint8))
        return {"packed": packed, "shape": crop_bool.shape, "bbox": bbox}
    return crop_bool.astype(bool)


def mask_entry_to_crop(mask_entry):
    """Return mask crop (float32) and its bbox."""
    crop, bbox = unpack_mask_entry(mask_entry)
    if bbox is None:
        h, w = crop.shape[:2]
        bbox = [0, 0, w - 1, h - 1]
    return crop.astype(np.float32), bbox


def mask_bbox_from_bool(mask_bool):
    """Return tight XYXY bbox for a boolean mask, or None if empty."""
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0 or xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def infer_img_shape(frame_results):
    """Infer a minimal image shape (H, W) from frame boxes."""
    if not frame_results or 'boxes' not in frame_results:
        return (1, 1)
    boxes = frame_results.get('boxes', [])
    if not boxes:
        return (1, 1)
    max_x2 = max(int(round(b[2])) for b in boxes)
    max_y2 = max(int(round(b[3])) for b in boxes)
    return (max(1, max_y2 + 1), max(1, max_x2 + 1))


# ---------------------------------------------------------------------------
# Overlap merging
# ---------------------------------------------------------------------------

def _boxes_intersect_xyxy(box_a, box_b):
    """Return True if two XYXY boxes overlap with positive area."""
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    return inter_w > 0.0 and inter_h > 0.0


def merge_overlapping_in_frame(frame_results, img_shape):
    """
    Merge overlapping masks within a single frame by OR-union.

    Returns (new_frame_results, merged_count).
    """
    if not frame_results or 'masks' not in frame_results or 'boxes' not in frame_results:
        return frame_results, 0

    boxes = list(frame_results.get('boxes', []))
    masks = list(frame_results.get('masks', []))
    scores = list(frame_results.get('scores', []))
    labels_present = frame_results.get('labels') is not None
    labels = list(frame_results.get('labels', [])) if labels_present else []

    n = min(len(boxes), len(masks))
    if n <= 1:
        return frame_results, 0

    boxes = boxes[:n]
    masks = masks[:n]
    has_scores = len(scores) > 0
    if has_scores:
        scores = scores[:n]
    if labels_present:
        labels = labels[:n]

    full_masks = []
    for mask_entry in masks:
        try:
            full_masks.append(rebuild_full_mask(mask_entry, img_shape).astype(bool))
        except Exception:
            full_masks.append(None)

    # Build adjacency graph
    adjacency = {i: set() for i in range(n)}
    for i in range(n):
        if full_masks[i] is None:
            continue
        for j in range(i + 1, n):
            if full_masks[j] is None:
                continue
            if not _boxes_intersect_xyxy(boxes[i], boxes[j]):
                continue
            if np.any(full_masks[i] & full_masks[j]):
                adjacency[i].add(j)
                adjacency[j].add(i)

    # Connected components via DFS
    visited = set()
    clusters = []
    for start in range(n):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        cluster = []
        while stack:
            cur = stack.pop()
            cluster.append(cur)
            for nxt in adjacency[cur]:
                if nxt in visited:
                    continue
                visited.add(nxt)
                stack.append(nxt)
        clusters.append(sorted(cluster))

    merged_count = sum(max(0, len(c) - 1) for c in clusters)
    if merged_count == 0:
        return frame_results, 0

    clusters.sort(key=lambda c: c[0])

    new_boxes = []
    new_masks = []
    new_scores = []
    new_labels = []

    for cluster in clusters:
        if len(cluster) == 1:
            idx = cluster[0]
            new_boxes.append(boxes[idx])
            new_masks.append(masks[idx])
            new_scores.append(float(scores[idx]) if has_scores and idx < len(scores) else 0.0)
            if labels_present:
                new_labels.append(int(labels[idx]) if idx < len(labels) else 1)
            continue

        union_mask = np.zeros(img_shape[:2], dtype=bool)
        pack_output = False
        for idx in cluster:
            if full_masks[idx] is not None:
                union_mask |= full_masks[idx]
            if isinstance(masks[idx], dict) and "packed" in masks[idx]:
                pack_output = True

        bbox = mask_bbox_from_bool(union_mask)
        if bbox is None:
            x1 = min(int(round(boxes[idx][0])) for idx in cluster)
            y1 = min(int(round(boxes[idx][1])) for idx in cluster)
            x2 = max(int(round(boxes[idx][2])) for idx in cluster)
            y2 = max(int(round(boxes[idx][3])) for idx in cluster)
            bbox = [x1, y1, x2, y2]
            crop = np.zeros((max(1, y2 - y1 + 1), max(1, x2 - x1 + 1)), dtype=bool)
        else:
            x1, y1, x2, y2 = bbox
            crop = union_mask[y1:y2 + 1, x1:x2 + 1]

        new_boxes.append(np.array(bbox, dtype=np.float32))
        new_masks.append(make_mask_entry(crop, bbox, pack_bits=pack_output))

        if has_scores:
            score_candidates = [float(scores[idx]) for idx in cluster if idx < len(scores)]
            new_scores.append(float(max(score_candidates)) if score_candidates else 0.0)
        else:
            new_scores.append(0.0)
        if labels_present:
            if has_scores:
                best_idx = max(cluster, key=lambda idx: float(scores[idx]) if idx < len(scores) else -1.0)
            else:
                best_idx = cluster[0]
            new_labels.append(int(labels[best_idx]) if best_idx < len(labels) else 1)

    new_frame_results = dict(frame_results)
    new_frame_results['boxes'] = new_boxes
    new_frame_results['masks'] = new_masks
    new_frame_results['scores'] = new_scores
    if labels_present:
        new_frame_results['labels'] = new_labels

    return new_frame_results, int(merged_count)


def merge_overlapping_in_results(all_results):
    """
    Merge overlapping masks for every frame in the results list.

    Returns (updated_results_list, total_merged_masks).
    """
    if not all_results:
        return all_results, 0

    total_merged = 0
    updated = []

    for frame_idx, frame_results, img_path in all_results:
        if not frame_results or 'masks' not in frame_results:
            updated.append((frame_idx, frame_results, img_path))
            continue

        img_shape = None
        try:
            with Image.open(img_path) as img:
                img_shape = (int(img.height), int(img.width))
        except Exception:
            img_shape = infer_img_shape(frame_results)

        merged_frame_results, merged_count = merge_overlapping_in_frame(
            frame_results, img_shape
        )
        total_merged += int(merged_count)
        updated.append((frame_idx, merged_frame_results, img_path))

    return updated, total_merged
