import json
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import pickle
import re
import statistics
import string
import copy
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

import cv2
from tqdm import tqdm

import ocr_postprocess as _ocr_postprocess
from utils import extract_video_frames


def _bbox_to_xyxy(box):
    """
    Convert a Paddle box/polygon to integer (x1, y1, x2, y2).
    Supports:
    - [x1, y1, x2, y2]
    - [[x, y], [x, y], [x, y], [x, y]]
    """
    if box is None:
        return None

    if hasattr(box, "tolist"):
        box = box.tolist()

    if not isinstance(box, (list, tuple)) or len(box) == 0:
        return None

    if len(box) == 4 and not isinstance(box[0], (list, tuple)):
        x1, y1, x2, y2 = box
        return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))

    if isinstance(box[0], (list, tuple)):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return (
            int(round(min(xs))),
            int(round(min(ys))),
            int(round(max(xs))),
            int(round(max(ys))),
        )

    return None


def _page_to_dict(page):
    """Convert PaddleOCR page result object to a plain dict."""
    if isinstance(page, dict):
        return page

    raw = getattr(page, "json", None)
    if callable(raw):
        raw = raw()

    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _get_paddle_device(device="auto"):
    if device and device != "auto":
        return device
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda():
            try:
                dev = paddle.device.get_device()
                if isinstance(dev, str) and dev.startswith("gpu"):
                    return dev
            except Exception:
                pass
            return "gpu:0"
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "gpu:0"
    except Exception:
        pass
    return "cpu"


def get_paddleocr_reader(lang="de", device="auto", ocr_version="PP-OCRv5", return_word_box=True):
    from paddleocr import PaddleOCR

    paddle_device = _get_paddle_device(device)
    print(f"Initializing PaddleOCR (lang={lang}, device={paddle_device}, ocr_version={ocr_version})...")

    # For UI/screen recordings, disable document preprocessing stages and use
    # stricter detection/recognition thresholds to reduce waveform/UI noise.
    reader = PaddleOCR(
        lang=lang,
        device=paddle_device,
        ocr_version=ocr_version,
        return_word_box=return_word_box,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=1536,
        text_det_limit_type="max",
        text_det_thresh=0.3,
        text_det_box_thresh=0.7,
        text_det_unclip_ratio=1.6,
        text_rec_score_thresh=0.6,
    )
    print("PaddleOCR reader initialized.")
    return reader


def _collect_boxes_from_page(page_dict):
    """
    Return a dict with:
    - parent_boxes: line-level OCR boxes
    - word_boxes: word-level OCR boxes (with parent references)
    """
    page_res = page_dict.get("res", page_dict)

    def _to_seq(v):
        if v is None:
            return []
        if hasattr(v, "tolist"):
            v = v.tolist()
        return v

    rec_texts = _to_seq(page_res.get("rec_texts", []))
    rec_scores = _to_seq(page_res.get("rec_scores", []))
    rec_boxes = _to_seq(page_res.get("rec_boxes", []))
    if len(rec_boxes) == 0:
        rec_boxes = _to_seq(page_res.get("rec_polys", []))

    text_word = _to_seq(page_res.get("text_word", []))
    text_word_boxes = _to_seq(page_res.get("text_word_boxes", []))

    parent_boxes = []
    word_boxes = []

    for i, parent_text in enumerate(rec_texts):
        parent_bbox = _bbox_to_xyxy(rec_boxes[i]) if i < len(rec_boxes) else None
        parent_score = float(rec_scores[i]) if i < len(rec_scores) else 0.0

        parent_entry = {
            "text": str(parent_text),
            "bbox": parent_bbox,
            "confidence": parent_score,
            "parent_index": i,
        }
        parent_boxes.append(parent_entry)

        words_i = text_word[i] if i < len(text_word) else []
        word_boxes_i = text_word_boxes[i] if i < len(text_word_boxes) else []

        if isinstance(words_i, str):
            words_i = [words_i]

        if isinstance(word_boxes_i, dict):
            # Some versions can wrap word boxes in a dict-like structure.
            word_boxes_i = word_boxes_i.get("boxes", [])

        if hasattr(words_i, "tolist"):
            words_i = words_i.tolist()
        if hasattr(word_boxes_i, "tolist"):
            word_boxes_i = word_boxes_i.tolist()

        if len(words_i) > 0 and len(word_boxes_i) > 0:
            for j, word_text in enumerate(words_i):
                if j >= len(word_boxes_i):
                    break
                word_bbox = _bbox_to_xyxy(word_boxes_i[j])
                if word_bbox is None:
                    continue
                word_boxes.append({
                    "text": str(word_text),
                    "bbox": word_bbox,
                    "parent_box": parent_bbox,
                    "parent_text": str(parent_text),
                    "confidence": parent_score,
                    "parent_index": i,
                    "word_index": j,
                })
        elif parent_bbox is not None and str(parent_text).strip():
            # Fallback: if no word boxes are present, keep one word entry = parent entry.
            word_boxes.append({
                "text": str(parent_text),
                "bbox": parent_bbox,
                "parent_box": parent_bbox,
                "parent_text": str(parent_text),
                "confidence": parent_score,
                "parent_index": i,
                "word_index": 0,
            })

    return {
        "parent_boxes": parent_boxes,
        "word_boxes": word_boxes,
    }


def _union_bboxes(b1, b2):
    if b1 is None:
        return b2
    if b2 is None:
        return b1
    x1 = min(b1[0], b2[0])
    y1 = min(b1[1], b2[1])
    x2 = max(b1[2], b2[2])
    y2 = max(b1[3], b2[3])
    return (int(x1), int(y1), int(x2), int(y2))


def _is_space_token(text):
    if not isinstance(text, str):
        return False
    # Paddle can emit empty strings, NBSPs, and zero-width separators as tokens.
    cleaned = (
        text.replace("\u00A0", " ")  # non-breaking space
            .replace("\u200B", "")   # zero-width space
            .replace("\u200C", "")   # zero-width non-joiner
            .replace("\u200D", "")   # zero-width joiner
            .replace("\uFEFF", "")   # zero-width no-break space / BOM
    )
    return cleaned.strip() == ""


def _is_punctuation_token(text):
    if not isinstance(text, str) or text == "":
        return False
    if text.strip() == "":
        return False
    return all((ch in string.punctuation) for ch in text)


def _merge_token_boxes(a, b, join_text=""):
    merged = a.copy()
    merged["bbox"] = _union_bboxes(a.get("bbox"), b.get("bbox"))
    merged["text"] = f"{a.get('text', '')}{join_text}{b.get('text', '')}"
    merged["parent_box"] = _union_bboxes(a.get("parent_box"), b.get("parent_box")) if a.get("parent_box") or b.get("parent_box") else a.get("parent_box") or b.get("parent_box")
    try:
        merged["confidence"] = (float(a.get("confidence", 0.0)) + float(b.get("confidence", 0.0))) / 2.0
    except Exception:
        pass
    return merged


def _clone_frame_data(frame_data):
    """Return a deep-copied frame_data dict so stage functions stay pure."""
    return copy.deepcopy(frame_data)


def _bbox_to_int_tuple(bbox):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return (
        int(round(float(x1))),
        int(round(float(y1))),
        int(round(float(x2))),
        int(round(float(y2))),
    )


def _bbox_iou(b1, b2):
    if not b1 or not b2:
        return 0.0
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(a_area + b_area - inter)


def _stabilize_bbox(curr_bbox, prev_bbox, movement_threshold=3):
    """
    Frame-to-frame bbox stabilization to reduce jitter while avoiding shrinkage.
    Returns integer bbox.
    """
    curr_bbox = _bbox_to_int_tuple(curr_bbox)
    prev_bbox = _bbox_to_int_tuple(prev_bbox)
    if not curr_bbox or not prev_bbox:
        return curr_bbox or prev_bbox

    cx1, cy1, cx2, cy2 = curr_bbox
    px1, py1, px2, py2 = prev_bbox

    curr_w = max(1, cx2 - cx1)
    curr_h = max(1, cy2 - cy1)
    prev_w = max(1, px2 - px1)
    prev_h = max(1, py2 - py1)

    curr_cx = (cx1 + cx2) / 2.0
    curr_cy = (cy1 + cy2) / 2.0
    prev_cx = (px1 + px2) / 2.0
    prev_cy = (py1 + py2) / 2.0

    dx = abs(curr_cx - prev_cx)
    dy = abs(curr_cy - prev_cy)

    # Nearly static: keep previous box unless current grows noticeably.
    if dx < movement_threshold and dy < movement_threshold:
        width_change = abs(curr_w - prev_w) / float(max(1, prev_w))
        height_change = abs(curr_h - prev_h) / float(max(1, prev_h))
        if max(width_change, height_change) > 0.15:
            return (
                int(min(cx1, px1)),
                int(min(cy1, py1)),
                int(max(cx2, px2)),
                int(max(cy2, py2)),
            )
        return prev_bbox

    # Predominantly horizontal motion: preserve Y and stable height.
    if dy < movement_threshold or dy < dx / 2.0:
        stable_h = max(prev_h, curr_h)
        y1 = py1
        y2 = y1 + stable_h
        x1 = int(round(curr_cx - max(prev_w, curr_w) / 2.0))
        x2 = x1 + max(prev_w, curr_w)
        return (int(x1), int(y1), int(x2), int(y2))

    # Predominantly vertical motion: preserve X and stable width.
    if dx < movement_threshold or dx < dy / 2.0:
        stable_w = max(prev_w, curr_w)
        x1 = px1
        x2 = x1 + stable_w
        y1 = int(round(curr_cy - max(prev_h, curr_h) / 2.0))
        y2 = y1 + max(prev_h, curr_h)
        return (int(x1), int(y1), int(x2), int(y2))

    # Diagonal / general motion: keep current center, prevent sudden shrink.
    final_w = max(prev_w, curr_w) if abs(curr_w - prev_w) / float(max(1, prev_w)) > 0.2 else curr_w
    final_h = max(prev_h, curr_h) if abs(curr_h - prev_h) / float(max(1, prev_h)) > 0.2 else curr_h
    x1 = int(round(curr_cx - final_w / 2.0))
    y1 = int(round(curr_cy - final_h / 2.0))
    return (int(x1), int(y1), int(x1 + final_w), int(y1 + final_h))

def normalize_parent_boxes_by_line(
    parent_boxes,
    x_gap_threshold=40,
    height_rel_tol=0.20,
    center_y_tol_factor=0.50,
    min_vertical_overlap_ratio=0.50,
):
    """
    Normalize parent box heights line-by-line without merging boxes.

    Boxes are grouped only if they are:
    - horizontally close (or overlapping)
    - vertically aligned (same line)
    - similar height (strict, to avoid merging timestamps with message text)

    For each group, only y1/y2 are normalized. x1/x2 stay unchanged.
    """
    if not parent_boxes:
        return []

    boxes = [b.copy() for b in parent_boxes]
    valid = [i for i, b in enumerate(boxes) if b.get("bbox")]

    def _metrics(bbox):
        x1, y1, x2, y2 = bbox
        h = max(1, y2 - y1)
        cy = (y1 + y2) / 2.0
        return x1, y1, x2, y2, h, cy

    def _compatible(i, j):
        bi = boxes[i].get("bbox")
        bj = boxes[j].get("bbox")
        if not bi or not bj:
            return False

        xi1, yi1, xi2, yi2, hi, cyi = _metrics(bi)
        xj1, yj1, xj2, yj2, hj, cyj = _metrics(bj)

        rel_h_diff = abs(hi - hj) / float(max(hi, hj))
        if rel_h_diff > height_rel_tol:
            return False

        center_y_diff = abs(cyi - cyj)
        if center_y_diff > center_y_tol_factor * min(hi, hj):
            return False

        overlap_y = max(0, min(yi2, yj2) - max(yi1, yj1))
        if overlap_y / float(min(hi, hj)) < min_vertical_overlap_ratio:
            return False

        # Horizontal distance between boxes (0 if overlapping on x).
        gap = max(0, max(xi1, xj1) - min(xi2, xj2))
        if gap > x_gap_threshold:
            return False

        return True

    visited = set()
    for start in sorted(valid, key=lambda i: ((boxes[i]["bbox"][1] + boxes[i]["bbox"][3]) / 2.0, boxes[i]["bbox"][0])):
        if start in visited:
            continue

        cluster = []
        stack = [start]
        visited.add(start)

        while stack:
            cur = stack.pop()
            cluster.append(cur)
            for j in valid:
                if j in visited:
                    continue
                if _compatible(cur, j):
                    visited.add(j)
                    stack.append(j)

        if len(cluster) < 2:
            continue

        ys1 = []
        ys2 = []
        hs = []
        cys = []
        for idx in cluster:
            x1, y1, x2, y2 = boxes[idx]["bbox"]
            h = max(1, y2 - y1)
            ys1.append(y1)
            ys2.append(y2)
            hs.append(h)
            cys.append((y1 + y2) / 2.0)

        target_h = int(round(statistics.median(hs)))
        target_cy = float(statistics.median(cys))
        target_y1 = int(round(target_cy - target_h / 2.0))
        target_y2 = int(round(target_cy + target_h / 2.0))

        # Keep target within the observed cluster vertical span to avoid drift.
        target_y1 = max(min(ys1), target_y1)
        target_y2 = min(max(ys2), target_y2)
        if target_y2 <= target_y1:
            continue

        for idx in cluster:
            x1, y1, x2, y2 = boxes[idx]["bbox"]
            boxes[idx]["bbox"] = (int(x1), int(target_y1), int(x2), int(target_y2))

    return boxes


def _sync_word_parent_boxes(word_boxes, parent_boxes):
    """Update each word box's parent_box after parent-box normalization."""
    if not word_boxes:
        return []

    parent_bbox_by_idx = {}
    parent_track_by_idx = {}
    for p in parent_boxes:
        if p.get("bbox") is None:
            continue
        parent_bbox_by_idx[p.get("parent_index")] = p.get("bbox")
        if p.get("track_id") is not None:
            parent_track_by_idx[p.get("parent_index")] = p.get("track_id")

    out = []
    for w in word_boxes:
        w2 = w.copy()
        pi = w2.get("parent_index")
        if pi in parent_bbox_by_idx:
            w2["parent_box"] = parent_bbox_by_idx[pi]
        if pi in parent_track_by_idx:
            w2["track_id"] = parent_track_by_idx[pi]
        out.append(w2)
    return out


def cluster_parent_box_heights(parent_boxes, min_confidence=0.7, height_clusters=None, max_clusters=4):
    """
    Normalize parent box heights to a small set of cluster heights.

    Cluster centers are estimated from boxes with confidence >= min_confidence.
    Boxes are then normalized to the nearest cluster height (all boxes, not just high-conf).
    """
    if not parent_boxes:
        return [], []

    boxes = [b.copy() for b in parent_boxes]

    def _sanitize_clusters(vals):
        out = []
        for v in vals:
            try:
                out.append(max(1, int(round(float(v)))))
            except Exception:
                continue
        return sorted(set(out))

    if height_clusters is None:
        heights = []
        for b in boxes:
            bbox = b.get("bbox")
            if not bbox:
                continue
            conf = float(b.get("confidence", 0.0))
            if conf < float(min_confidence):
                continue
            x1, y1, x2, y2 = bbox
            h = int(round(float(y2 - y1)))
            if h > 0:
                heights.append(h)

        if not heights:
            return boxes, []

        unique_heights = sorted(set(heights))
        if len(unique_heights) == 1:
            clusters = unique_heights
        else:
            # Prefer sklearn KMeans if available (same idea as ocr.py), otherwise
            # fall back to the observed unique heights.
            try:
                import numpy as np
                from sklearn.cluster import KMeans

                n_clusters = min(max_clusters, len(unique_heights))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(np.array(heights).reshape(-1, 1))
                clusters = _sanitize_clusters(kmeans.cluster_centers_.flatten())
            except Exception:
                clusters = unique_heights[:max_clusters]
    else:
        clusters = _sanitize_clusters(height_clusters)

    if not clusters:
        return boxes, []

    normalized = []
    for b in boxes:
        bbox = b.get("bbox")
        if not bbox:
            normalized.append(b)
            continue

        x1, y1, x2, y2 = bbox
        cur_h = int(round(float(y2 - y1)))
        if cur_h <= 0:
            normalized.append(b)
            continue

        target_h = min(clusters, key=lambda h: abs(h - cur_h))
        cy = (float(y1) + float(y2)) / 2.0
        new_y1 = int(round(cy - target_h / 2.0))
        new_y2 = int(round(cy + target_h / 2.0))
        if new_y2 <= new_y1:
            new_y2 = new_y1 + max(1, int(target_h))

        b2 = b.copy()
        b2["bbox"] = (
            int(round(float(x1))),
            int(new_y1),
            int(round(float(x2))),
            int(new_y2),
        )
        normalized.append(b2)

    return normalized, clusters


def _adjust_word_boxes_using_parent(word_boxes):
    """
    Make final word boxes inherit stable parent geometry.

    This reduces flicker after parent stabilization by:
    - forcing word y1/y2 to parent y1/y2
    - clamping word x1/x2 to parent x-range
    """
    if not word_boxes:
        return []

    out = []
    for box in word_boxes:
        b = box.copy()
        bbox = b.get("bbox")
        pbox = b.get("parent_box")
        if bbox and pbox:
            x1, y1, x2, y2 = _bbox_to_int_tuple(bbox)
            px1, py1, px2, py2 = _bbox_to_int_tuple(pbox)
            x1 = max(px1, x1)
            x2 = min(px2, x2)
            if x2 <= x1:
                x1, x2 = px1, px2
            b["bbox"] = (int(x1), int(py1), int(x2), int(py2))
        out.append(b)
    return out


def _normalize_token_for_tracking(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.strip().lower()


def _relative_x_span(word_bbox, parent_bbox):
    if not word_bbox or not parent_bbox:
        return None
    wx1, _, wx2, _ = _bbox_to_int_tuple(word_bbox)
    px1, _, px2, _ = _bbox_to_int_tuple(parent_bbox)
    pw = max(1, px2 - px1)
    rx1 = (wx1 - px1) / float(pw)
    rx2 = (wx2 - px1) / float(pw)
    return (rx1, rx2)


def _project_relative_x_span(rel_span, parent_bbox):
    if rel_span is None or parent_bbox is None:
        return None
    rx1, rx2 = rel_span
    px1, py1, px2, py2 = _bbox_to_int_tuple(parent_bbox)
    pw = max(1, px2 - px1)
    rx1 = max(0.0, min(1.0, float(rx1)))
    rx2 = max(0.0, min(1.0, float(rx2)))
    if rx2 <= rx1:
        mid = max(0.0, min(1.0, (rx1 + rx2) / 2.0))
        rx1 = max(0.0, mid - 0.005)
        rx2 = min(1.0, mid + 0.005)
    x1 = int(round(px1 + rx1 * pw))
    x2 = int(round(px1 + rx2 * pw))
    if x2 <= x1:
        x2 = min(px2, x1 + 1)
        x1 = max(px1, x2 - 1)
    return (x1, py1, x2, py2)


def _stabilize_relative_x_span(curr_rel, prev_rel, rel_movement_threshold=0.01):
    """
    1D stabilization in parent-relative coordinates.
    Parent motion is already stabilized separately, so we only stabilize x-span.
    """
    if curr_rel is None or prev_rel is None:
        return curr_rel or prev_rel

    curr_x1, curr_x2 = float(curr_rel[0]), float(curr_rel[1])
    prev_x1, prev_x2 = float(prev_rel[0]), float(prev_rel[1])
    curr_w = max(1e-6, curr_x2 - curr_x1)
    prev_w = max(1e-6, prev_x2 - prev_x1)
    curr_c = (curr_x1 + curr_x2) / 2.0
    prev_c = (prev_x1 + prev_x2) / 2.0
    movement = abs(curr_c - prev_c)

    # Static relative center: preserve previous span unless current grows a lot.
    if movement < rel_movement_threshold:
        width_change = abs(curr_w - prev_w) / max(prev_w, 1e-6)
        if width_change > 0.15:
            return (min(curr_x1, prev_x1), max(curr_x2, prev_x2))
        return (prev_x1, prev_x2)

    # Moving text within same parent: keep current center but avoid sudden shrink.
    width_diff = abs(curr_w - prev_w) / max(prev_w, 1e-6)
    final_w = max(prev_w, curr_w) if width_diff > 0.2 else curr_w
    return (curr_c - final_w / 2.0, curr_c + final_w / 2.0)


def apply_parent_constrained_word_stabilization(
    frame_data,
    max_gap=6,
    rel_movement_threshold=0.01,
    rel_position_tol=0.20,
    text_similarity_threshold=0.85,
):
    """
    Stabilize word boxes inside each stabilized parent track.

    Uses parent `track_id` as the constraint. This avoids global word tracking and
    stabilizes only x-coordinates in parent-relative space. y1/y2 stay equal to the
    stabilized parent box.
    """
    updated = _clone_frame_data(frame_data)
    frame_indices = sorted([k for k in updated.keys() if isinstance(k, int)])

    # Track per parent track_id using the last stabilized word group.
    parent_track_state = {}  # parent_track_id -> {"frame_idx": int, "words": [boxes]}

    for frame_idx in frame_indices:
        frame = updated[frame_idx]
        words = [b.copy() for b in frame.get("word_boxes", [])]
        if not words:
            continue

        groups = {}
        passthrough = []
        for w in words:
            parent_track_id = w.get("track_id", None)
            if parent_track_id is None or not w.get("parent_box") or not w.get("bbox"):
                passthrough.append(w)
                continue
            groups.setdefault(parent_track_id, []).append(w)

        stabilized_groups = []

        for parent_track_id, group_words in groups.items():
            group_words.sort(key=lambda b: (b["bbox"][0], b["bbox"][2]))
            state = parent_track_state.get(parent_track_id)

            if not state or (frame_idx - state["frame_idx"] > max_gap):
                stabilized_groups.extend(group_words)
                parent_track_state[parent_track_id] = {"frame_idx": frame_idx, "words": [g.copy() for g in group_words]}
                continue

            prev_words = [g.copy() for g in state["words"]]
            prev_words.sort(key=lambda b: (b["bbox"][0], b["bbox"][2]))

            # Fast path: same token count and same normalized sequence -> pair by index.
            curr_norm = [_normalize_token_for_tracking(w.get("text", "")) for w in group_words]
            prev_norm = [_normalize_token_for_tracking(w.get("text", "")) for w in prev_words]
            matched_pairs = []

            if len(group_words) == len(prev_words) and curr_norm == prev_norm:
                matched_pairs = list(zip(range(len(prev_words)), range(len(group_words))))
            else:
                # Greedy parent-constrained matching by text + relative x position.
                used_prev = set()
                for j, curr_w in enumerate(group_words):
                    curr_txt = _normalize_token_for_tracking(curr_w.get("text", ""))
                    curr_rel = _relative_x_span(curr_w.get("bbox"), curr_w.get("parent_box"))
                    if curr_rel is None:
                        continue
                    curr_center = (curr_rel[0] + curr_rel[1]) / 2.0

                    best_i = None
                    best_score = None
                    for i, prev_w in enumerate(prev_words):
                        if i in used_prev:
                            continue
                        prev_rel = _relative_x_span(prev_w.get("bbox"), prev_w.get("parent_box"))
                        if prev_rel is None:
                            continue
                        prev_center = (prev_rel[0] + prev_rel[1]) / 2.0
                        rel_dist = abs(curr_center - prev_center)
                        if rel_dist > rel_position_tol:
                            continue

                        prev_txt = _normalize_token_for_tracking(prev_w.get("text", ""))
                        text_sim = SequenceMatcher(None, prev_txt, curr_txt).ratio()
                        if curr_txt and prev_txt and text_sim < text_similarity_threshold:
                            continue

                        # Prefer better text match, then closer relative position.
                        score = (2.0 * text_sim) - rel_dist
                        if best_score is None or score > best_score:
                            best_score = score
                            best_i = i

                    if best_i is not None:
                        used_prev.add(best_i)
                        matched_pairs.append((best_i, j))

            # Apply x-span stabilization for matched pairs.
            stabilized_group = [w.copy() for w in group_words]
            for prev_i, curr_j in matched_pairs:
                prev_w = prev_words[prev_i]
                curr_w = stabilized_group[curr_j]
                prev_rel = _relative_x_span(prev_w.get("bbox"), prev_w.get("parent_box"))
                curr_rel = _relative_x_span(curr_w.get("bbox"), curr_w.get("parent_box"))
                stable_rel = _stabilize_relative_x_span(
                    curr_rel,
                    prev_rel,
                    rel_movement_threshold=rel_movement_threshold,
                )
                projected = _project_relative_x_span(stable_rel, curr_w.get("parent_box"))
                if projected is not None:
                    curr_w["bbox"] = projected
                    # keep a word-level track id only for debugging if parent track exists
                    curr_w["word_track_parent_id"] = parent_track_id

            stabilized_groups.extend(stabilized_group)
            parent_track_state[parent_track_id] = {"frame_idx": frame_idx, "words": [g.copy() for g in stabilized_group]}

        final_words = stabilized_groups + passthrough
        final_words.sort(
            key=lambda b: ((b["bbox"][1] + b["bbox"][3]) / 2.0, b["bbox"][0]) if b.get("bbox") else (10**9, 10**9)
        )
        frame["word_boxes"] = final_words

    return updated


def postprocess_word_boxes(word_boxes):
    """
    Clean Paddle word boxes within each parent line:
    - split whitespace box width between adjacent tokens
    - attach punctuation to previous token
    - merge time-like tokens (e.g., "19:" + "23" -> "19:23")
    """
    if not word_boxes:
        return []

    groups = {}
    for box in word_boxes:
        key = (
            tuple(box.get("parent_box")) if box.get("parent_box") else None,
            box.get("parent_index"),
            box.get("parent_text", ""),
        )
        groups.setdefault(key, []).append(box.copy())

    out = []

    for _, items in groups.items():
        items.sort(key=lambda b: (int(b.get("word_index", 10**9)), b["bbox"][0] if b.get("bbox") else 10**9))

        # 1) Whitespace handling: split box width between adjacent tokens in same parent.
        no_space = []
        for idx, tok in enumerate(items):
            txt = tok.get("text", "")
            if not _is_space_token(txt):
                no_space.append(tok)
                continue

            bbox = tok.get("bbox")
            if not bbox:
                continue
            sx1, sy1, sx2, sy2 = bbox
            mid_left = int(round((sx1 + sx2) / 2.0))
            mid_right = mid_left

            prev_tok = no_space[-1] if no_space else None
            next_tok = None
            for k in range(idx + 1, len(items)):
                if not _is_space_token(items[k].get("text", "")):
                    next_tok = items[k]
                    break

            if prev_tok and prev_tok.get("bbox"):
                px1, py1, px2, py2 = prev_tok["bbox"]
                prev_tok["bbox"] = (px1, py1, max(px2, mid_left), py2)
            if next_tok and next_tok.get("bbox"):
                nx1, ny1, nx2, ny2 = next_tok["bbox"]
                next_tok["bbox"] = (min(nx1, mid_right), ny1, nx2, ny2)
            # Drop whitespace token itself.

        # 2) Attach punctuation to previous token.
        no_punct = []
        for tok in no_space:
            txt = tok.get("text", "")
            if _is_punctuation_token(txt) and no_punct:
                no_punct[-1] = _merge_token_boxes(no_punct[-1], tok, join_text="")
            else:
                no_punct.append(tok)

        # 3) Merge time-like sequences split into adjacent tokens.
        merged = []
        i = 0
        while i < len(no_punct):
            cur = no_punct[i]
            if i + 1 < len(no_punct):
                nxt = no_punct[i + 1]
                combined = f"{cur.get('text', '')}{nxt.get('text', '')}"
                if re.match(r"^\d{1,2}:\d{2}$", combined):
                    merged.append(_merge_token_boxes(cur, nxt, join_text=""))
                    i += 2
                    continue
            merged.append(cur)
            i += 1

        # 4) Expand edge word boxes to parent box boundaries to avoid clipped
        # first/last letters produced by Paddle's internal word splitting.
        merged_with_bbox = [b for b in merged if b.get("bbox")]
        if merged_with_bbox:
            parent_bbox = merged_with_bbox[0].get("parent_box")
            if parent_bbox:
                px1, py1, px2, py2 = parent_bbox
                left_idx = min(range(len(merged)), key=lambda k: merged[k]["bbox"][0] if merged[k].get("bbox") else 10**9)
                right_idx = max(range(len(merged)), key=lambda k: merged[k]["bbox"][2] if merged[k].get("bbox") else -1)

                if merged[left_idx].get("bbox"):
                    x1, y1, x2, y2 = merged[left_idx]["bbox"]
                    merged[left_idx]["bbox"] = (int(px1), int(y1), int(x2), int(y2))
                if merged[right_idx].get("bbox"):
                    x1, y1, x2, y2 = merged[right_idx]["bbox"]
                    merged[right_idx]["bbox"] = (int(x1), int(y1), int(px2), int(y2))

        out.extend(merged)

    # Final safety cleanup: drop any blank-like tokens that survived earlier steps.
    out = [b for b in out if not _is_space_token(b.get("text", ""))]

    # Keep stable reading order across parents for plotting/debugging.
    out.sort(key=lambda b: ((b["bbox"][1] + b["bbox"][3]) / 2.0, b["bbox"][0]) if b.get("bbox") else (10**9, 10**9))
    return out


def run_paddleocr_on_frame(frame_path, reader):
    frame_name = os.path.splitext(os.path.basename(frame_path))[0]
    frame_idx = int("".join(filter(str.isdigit, frame_name)))

    page_results = reader.predict(frame_path)
    parent_boxes = []
    word_boxes_raw = []

    for page in page_results:
        page_dict = _page_to_dict(page)
        if not page_dict:
            continue
        boxes = _collect_boxes_from_page(page_dict)
        parent_boxes.extend(boxes["parent_boxes"])
        word_boxes_raw.extend(boxes["word_boxes"])

    return frame_idx, {
        "frame_path": str(frame_path),
        "parent_boxes_raw": [b.copy() for b in parent_boxes],
        "parent_boxes": [b.copy() for b in parent_boxes],
        "word_boxes_raw": word_boxes_raw,
        "word_boxes": [],
    }


def apply_parent_box_normalization(frame_data, **normalize_kwargs):
    """
    Apply line-wise parent-box normalization to all frames in a saved result dict.
    Keeps `parent_boxes_raw` untouched and writes normalized boxes to `parent_boxes`.
    """
    updated = _clone_frame_data(frame_data)
    for frame in updated.values():
        raw_parent_boxes = frame.get("parent_boxes_raw") or frame.get("parent_boxes") or []
        frame["parent_boxes"] = normalize_parent_boxes_by_line(raw_parent_boxes, **normalize_kwargs)
    return updated


def apply_parent_height_clustering(
    frame_data,
    min_confidence=0.7,
    height_clusters=None,
    max_clusters=4,
):
    """
    Cluster parent-box heights per frame and normalize to the nearest cluster.
    Uses only boxes with confidence >= min_confidence to estimate clusters.
    """
    updated = _clone_frame_data(frame_data)
    for frame in updated.values():
        src_boxes = frame.get("parent_boxes") or frame.get("parent_boxes_raw") or []
        normalized_boxes, clusters = cluster_parent_box_heights(
            src_boxes,
            min_confidence=min_confidence,
            height_clusters=height_clusters,
            max_clusters=max_clusters,
        )
        frame["parent_boxes"] = normalized_boxes
        frame["parent_height_clusters"] = clusters
    return updated


def apply_word_box_postprocessing(frame_data):
    """
    Build final `word_boxes` from `word_boxes_raw`, using the current `parent_boxes`
    (typically already normalized).
    """
    updated = _clone_frame_data(frame_data)
    for frame in updated.values():
        parent_boxes = frame.get("parent_boxes") or frame.get("parent_boxes_raw") or []
        word_boxes_raw = frame.get("word_boxes_raw", [])
        word_boxes_with_synced_parents = _sync_word_parent_boxes(word_boxes_raw, parent_boxes)
        word_boxes = postprocess_word_boxes(word_boxes_with_synced_parents)
        frame["word_boxes"] = _adjust_word_boxes_using_parent(word_boxes)
    return updated

def apply_box_stabilization(
    frame_data,
    box_key="parent_boxes",
    max_gap=6,
    position_threshold=35,
    size_threshold=0.25,
):
    """
    Temporal stabilization for boxes across frames using the same logic as
    `enhanced_temporal_tracking` in `ocr.py`.
    """
    def _enhanced_temporal_tracking_local(frame_boxes):
        def can_track(box1, box2):
            bbox1, bbox2 = box1["bbox"], box2["bbox"]

            center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
            center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
            x_distance = abs(center1[0] - center2[0])
            y_distance = abs(center1[1] - center2[1])

            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            size_diff = abs(area1 - area2) / max(area1, area2) if max(area1, area2) > 0 else 0

            text1 = box1.get("text", "") or box1.get("alterego", "")
            text2 = box2.get("text", "") or box2.get("alterego", "")
            text_sim = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

            if text_sim >= 0.9:
                return x_distance < position_threshold * 1.5 and y_distance < position_threshold * 2 and size_diff < size_threshold * 1.5
            elif text_sim >= 0.75:
                return x_distance < position_threshold and y_distance < position_threshold * 1.5 and size_diff < size_threshold
            else:
                return x_distance < position_threshold * 0.5 and y_distance < position_threshold * 0.5 and size_diff < size_threshold * 0.8

        def stabilize_coordinates(current_bbox, prev_bbox, movement_threshold=3):
            curr_x1, curr_y1, curr_x2, curr_y2 = current_bbox
            prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox

            def union_bbox(a, b):
                return (
                    int(min(a[0], b[0])),
                    int(min(a[1], b[1])),
                    int(max(a[2], b[2])),
                    int(max(a[3], b[3])),
                )

            curr_center_x = (curr_x1 + curr_x2) / 2
            curr_center_y = (curr_y1 + curr_y2) / 2
            prev_center_x = (prev_x1 + prev_x2) / 2
            prev_center_y = (prev_y1 + prev_y2) / 2

            x_movement = abs(curr_center_x - prev_center_x)
            y_movement = abs(curr_center_y - prev_center_y)

            curr_width = curr_x2 - curr_x1
            curr_height = curr_y2 - curr_y1
            prev_width = prev_x2 - prev_x1
            prev_height = prev_y2 - prev_y1

            if x_movement < movement_threshold and y_movement < movement_threshold:
                width_change = abs(curr_width - prev_width) / max(1, prev_width)
                height_change = abs(curr_height - prev_height) / max(1, prev_height)
                if max(width_change, height_change) > 0.15:
                    return union_bbox(current_bbox, prev_bbox)
                return _bbox_to_int_tuple(prev_bbox)
            elif y_movement < movement_threshold or y_movement < x_movement / 2:
                stable_width = max(prev_width, curr_width)
                stable_y1 = prev_y1
                stable_y2 = prev_y2
                stable_x1 = int(round(curr_center_x - stable_width / 2))
                stable_x2 = stable_x1 + int(stable_width)
                return (stable_x1, stable_y1, stable_x2, stable_y2)
            elif x_movement < movement_threshold or x_movement < y_movement / 2:
                stable_height = max(prev_height, curr_height)
                stable_x1 = prev_x1
                stable_x2 = prev_x2
                stable_y1 = int(round(curr_center_y - stable_height / 2))
                stable_y2 = stable_y1 + int(stable_height)
                return (stable_x1, stable_y1, stable_x2, stable_y2)
            else:
                width_diff = abs(curr_width - prev_width) / prev_width if prev_width > 0 else 0
                height_diff = abs(curr_height - prev_height) / prev_height if prev_height > 0 else 0
                final_width = max(prev_width, curr_width) if width_diff > 0.2 else curr_width
                final_height = max(prev_height, curr_height) if height_diff > 0.2 else curr_height
                stable_x1 = int(round(curr_center_x - final_width / 2))
                stable_y1 = int(round(curr_center_y - final_height / 2))
                stable_x2 = stable_x1 + int(final_width)
                stable_y2 = stable_y1 + int(final_height)
                return (stable_x1, stable_y1, stable_x2, stable_y2)

        if not frame_boxes:
            return frame_boxes, []

        frame_indices = sorted(frame_boxes.keys())
        tracks = []
        track_assignments = {}
        stabilized_boxes = {frame_idx: [] for frame_idx in frame_indices}

        for frame_idx in frame_indices:
            current_boxes = frame_boxes[frame_idx]
            unmatched_boxes = list(enumerate(current_boxes))
            if not current_boxes:
                continue

            for track in tracks:
                if not track:
                    continue
                last_frame, last_box = track[-1]
                if frame_idx - last_frame > max_gap:
                    continue

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

            for _, box in unmatched_boxes:
                tracks.append([(frame_idx, box)])

        for track_id, track in enumerate(tracks):
            if len(track) < 2:
                frame_idx, box = track[0]
                box_idx_in_stabilized = len(stabilized_boxes[frame_idx])
                stabilized_box = box.copy()
                stabilized_box["track_id"] = track_id
                stabilized_box["bbox"] = _bbox_to_int_tuple(stabilized_box["bbox"])
                stabilized_boxes[frame_idx].append(stabilized_box)
                track_assignments[(frame_idx, box_idx_in_stabilized)] = track_id
                continue

            text_variants = [(box.get("text", ""), box.get("confidence", 0)) for _, box in track]
            best_text = max(text_variants, key=lambda x: x[1])[0]

            for i, (frame_idx, box) in enumerate(track):
                box_idx_in_stabilized = len(stabilized_boxes[frame_idx])
                if i == 0:
                    stabilized_box = box.copy()
                    stabilized_box["text"] = best_text
                    stabilized_box["track_id"] = track_id
                    stabilized_box["bbox"] = _bbox_to_int_tuple(stabilized_box["bbox"])
                else:
                    prev_frame, prev_box = track[i - 1]
                    stabilized_prev_bbox = stabilized_boxes[prev_frame][-1]["bbox"]
                    stabilized_bbox = stabilize_coordinates(box["bbox"], stabilized_prev_bbox)
                    # For parent boxes, prioritize coverage over tightness, but do
                    # not propagate width from previous frames (can explode on a bad
                    # track match). Only prevent shrink relative to this frame's raw box.
                    if box_key == "parent_boxes":
                        sx1, sy1, sx2, sy2 = _bbox_to_int_tuple(stabilized_bbox)
                        cx1, cy1, cx2, cy2 = _bbox_to_int_tuple(box["bbox"])
                        stabilized_bbox = (min(sx1, cx1), sy1, max(sx2, cx2), sy2)
                    stabilized_box = box.copy()
                    stabilized_box["bbox"] = stabilized_bbox
                    stabilized_box["text"] = best_text
                    stabilized_box["track_id"] = track_id

                stabilized_boxes[frame_idx].append(stabilized_box)
                track_assignments[(frame_idx, box_idx_in_stabilized)] = track_id

        tracks_output = [[] for _ in range(len(tracks))]
        for (frame_idx, box_idx), track_id in track_assignments.items():
            tracks_output[track_id].append((frame_idx, box_idx))

        return stabilized_boxes, tracks_output

    updated = _clone_frame_data(frame_data)
    frame_boxes = {}
    for frame_idx, frame in updated.items():
        if not isinstance(frame_idx, int):
            continue
        frame_boxes[frame_idx] = [b.copy() for b in frame.get(box_key, [])]

    stabilized_boxes, _tracks = _enhanced_temporal_tracking_local(frame_boxes)
    for frame_idx, boxes in stabilized_boxes.items():
        updated[frame_idx][box_key] = boxes

    return updated


def filter_boxes_by_names(
    frame_data,
    names_dict,
    similarity_threshold=0.8,
    source_key="word_boxes",
    output_key="word_boxes",
):
    """
    Match word-level boxes against a names dictionary and add `to_show` flags.

    Reads boxes from `frame[source_key]` (default: `word_boxes`) and writes results
    to `frame[output_key]`. Matched name sequences are merged into one box with
    `to_show=True`; unmatched boxes are kept with `to_show=False`.
    """
    updated = _clone_frame_data(frame_data)

    if not names_dict:
        for frame in updated.values():
            out = []
            for b in frame.get(source_key, []) or []:
                b2 = b.copy()
                b2["to_show"] = True
                b2.setdefault("name", "")
                b2.setdefault("alterego", "")
                out.append(b2)
            frame[output_key] = out
        return updated

    def normalize_text(text):
        if not text:
            return ""
        text = unicodedata.normalize("NFD", str(text))
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        return text.lower().strip()

    def word_matches_name_part(word_text, name_part):
        norm_word = normalize_text(word_text)
        norm_name = normalize_text(name_part)
        if not norm_word or not norm_name:
            return False, 0.0
        if norm_word == norm_name:
            return True, 1.0
        similarity = SequenceMatcher(None, norm_word, norm_name).ratio()
        return (similarity >= similarity_threshold), similarity

    def group_boxes_into_lines(boxes):
        if not boxes:
            return []

        indexed = [(i, b) for i, b in enumerate(boxes) if b.get("bbox")]
        if not indexed:
            return []

        indexed.sort(key=lambda x: (((x[1]["bbox"][1] + x[1]["bbox"][3]) / 2.0), x[1]["bbox"][0]))
        lines = [[indexed[0]]]

        for idx, box in indexed[1:]:
            prev_box = lines[-1][-1][1]
            prev_cy = (prev_box["bbox"][1] + prev_box["bbox"][3]) / 2.0
            curr_cy = (box["bbox"][1] + box["bbox"][3]) / 2.0

            prev_h = max(1, prev_box["bbox"][3] - prev_box["bbox"][1])
            curr_h = max(1, box["bbox"][3] - box["bbox"][1])
            line_tol = max(15, int(round(0.5 * min(prev_h, curr_h))))

            if abs(curr_cy - prev_cy) <= line_tol:
                lines[-1].append((idx, box))
            else:
                lines.append([(idx, box)])

        for line in lines:
            line.sort(key=lambda x: x[1]["bbox"][0])
        return lines

    def find_name_sequences(line_boxes):
        matches = []

        for name, alterego in names_dict.items():
            name_words = normalize_text(name).split()
            if not name_words:
                continue

            for start_idx in range(len(line_boxes)):
                matched_boxes = []
                confidence_scores = []
                box_idx = start_idx
                word_idx = 0

                while word_idx < len(name_words) and box_idx < len(line_boxes):
                    orig_idx, box = line_boxes[box_idx]
                    box_text = str(box.get("text", "")).strip()

                    if not box_text:
                        box_idx += 1
                        continue

                    is_match, conf = word_matches_name_part(box_text, name_words[word_idx])
                    if not is_match:
                        break

                    matched_boxes.append((orig_idx, box))
                    confidence_scores.append(conf)
                    word_idx += 1
                    box_idx += 1

                if word_idx == len(name_words):
                    avg_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
                    matches.append({
                        "name": name,
                        "alterego": alterego,
                        "boxes": matched_boxes,
                        "confidence": avg_conf,
                    })

        if not matches:
            return []

        matches.sort(key=lambda x: (len(x["boxes"]), x["confidence"]), reverse=True)
        final_matches = []
        used_indices = set()
        for match in matches:
            box_indices = {idx for idx, _ in match["boxes"]}
            if box_indices.intersection(used_indices):
                continue
            final_matches.append(match)
            used_indices.update(box_indices)
        return final_matches

    def merge_matched_boxes(match):
        boxes = [box for _, box in match["boxes"]]
        x1 = min(b["bbox"][0] for b in boxes)
        y1 = min(b["bbox"][1] for b in boxes)
        x2 = max(b["bbox"][2] for b in boxes)
        y2 = max(b["bbox"][3] for b in boxes)

        parent_box = boxes[0].get("parent_box", None)
        parent_text = boxes[0].get("parent_text", None)
        track_ids = [b.get("track_id") for b in boxes if b.get("track_id") is not None]

        return {
            "bbox": _bbox_to_int_tuple((x1, y1, x2, y2)),
            "text": " ".join(str(b.get("text", "")) for b in boxes).strip(),
            "name": match["name"],
            "alterego": match["alterego"],
            "confidence": float(match.get("confidence", 0.0)),
            "to_show": True,
            "parent_box": _bbox_to_int_tuple(parent_box) if parent_box is not None else None,
            "parent_text": parent_text,
            "track_id": track_ids[0] if track_ids else None,
            "match_type": "word_level",
        }

    for frame in updated.values():
        boxes = frame.get(source_key, []) or []
        if not boxes:
            frame[output_key] = []
            continue

        lines = group_boxes_into_lines(boxes)
        frame_results = []
        all_matched_indices = set()

        for line in lines:
            matches = find_name_sequences(line)
            for match in matches:
                frame_results.append(merge_matched_boxes(match))
                all_matched_indices.update(idx for idx, _ in match["boxes"])

        for i, box in enumerate(boxes):
            if i in all_matched_indices:
                continue
            box_copy = box.copy()
            box_copy["to_show"] = False
            box_copy.setdefault("name", "")
            box_copy.setdefault("alterego", "")
            frame_results.append(box_copy)

        frame_results.sort(
            key=lambda b: (((b["bbox"][1] + b["bbox"][3]) / 2.0), b["bbox"][0]) if b.get("bbox") else (10**9, 10**9)
        )
        frame[output_key] = frame_results

    return updated


def ocr_boxes_to_unified_paddle(frame_data, source_key="word_boxes"):
    """
    Convert Paddle OCR frame_data to the same unified OCR format used by `ocr.py`.

    Output format:
    {frame_idx: [
        {
            "bbox": (x1, y1, x2, y2),
            "parent_box": (x1, y1, x2, y2) or None,
            "score": float,
            "text": str,
            "alterego": str,
            "mask": None,
            "source": "ocr",
            "to_show": bool,
            "track_id": int or None,
        }, ...
    ]}
    """
    unified = {}

    for frame_idx, frame in frame_data.items():
        # Keep only actual frame entries (int keys).
        if not isinstance(frame_idx, int):
            continue

        boxes = frame.get(source_key, []) or []
        unified_list = []

        for box in boxes:
            bbox = box.get("bbox") or box.get("original_bbox")
            if not bbox:
                continue

            bbox_int = _bbox_to_int_tuple(bbox)
            parent_box = box.get("parent_box", None)
            parent_box_int = _bbox_to_int_tuple(parent_box) if parent_box is not None else None

            unified_list.append({
                "bbox": bbox_int,
                "parent_box": parent_box_int,
                "score": float(box.get("confidence", 1.0)),
                "text": str(box.get("text", "") or ""),
                "alterego": str(box.get("alterego", "") or ""),
                "mask": None,
                "source": "ocr",
                "to_show": bool(box.get("to_show", True)),
                "track_id": box.get("track_id", None),
            })

        unified[frame_idx] = unified_list

    return unified


def _sorted_frame_paths(frames_dir):
    frame_paths = [p for p in Path(frames_dir).iterdir() if p.suffix.lower() in {".jpg", ".png"}]
    frame_paths.sort(key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))
    return frame_paths


def ensure_frames(video_path, output_dir, frame_step=1):
    """
    Use existing extracted frames if present, otherwise extract them.
    Returns the frames directory path.
    """
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"

    if frames_dir.exists() and any(frames_dir.glob("*.jpg")):
        print(f"Using existing frames in {frames_dir}")
        return frames_dir

    if frames_dir.exists() and any(frames_dir.glob("*.png")):
        print(f"Using existing frames in {frames_dir}")
        return frames_dir

    if video_path is None:
        raise ValueError("video_path is required when frames are not already extracted")

    frames_dir_str = extract_video_frames(str(video_path), output_dir=str(frames_dir), frame_step=frame_step)
    return Path(frames_dir_str)


def process_video_paddleocr(
    video_path,
    output_dir,
    lang="de",
    frame_step=1,
    device="auto",
    ocr_version="PP-OCRv5",
    save_pickle=True,
    save_raw_pickle=True,
    stabilize_boxes=True,
):
    """
    Minimal PaddleOCR pass:
    - ensure frames exist
    - run PaddleOCR on all frames
    - collect raw parent + word boxes
    - normalize parent boxes line-wise
    - postprocess word boxes
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = ensure_frames(video_path=video_path, output_dir=output_dir, frame_step=frame_step)
    frame_paths = _sorted_frame_paths(frames_dir)
    print(f"Found {len(frame_paths)} frames in {frames_dir}")

    reader = get_paddleocr_reader(lang=lang, device=device, ocr_version=ocr_version, return_word_box=True)

    frame_data = {}
    for frame_path in tqdm(frame_paths, desc="PaddleOCR Frames"):
        frame_idx, result = run_paddleocr_on_frame(str(frame_path), reader)
        frame_data[frame_idx] = result

    if save_raw_pickle:
        raw_out_path = output_dir / "boxes_paddleocr_words_raw.pkl"
        with open(raw_out_path, "wb") as f:
            pickle.dump(frame_data, f)
        print(f"Saved raw PaddleOCR parent/word boxes to {raw_out_path}")

    frame_data = apply_parent_box_normalization(frame_data)
    frame_data = apply_parent_height_clustering(frame_data, min_confidence=0.7)
    if stabilize_boxes:
        frame_data = apply_box_stabilization(frame_data, box_key="parent_boxes")
    frame_data = apply_word_box_postprocessing(frame_data)
    if stabilize_boxes:
        frame_data = apply_parent_constrained_word_stabilization(frame_data)

    if save_pickle:
        out_path = output_dir / "boxes_paddleocr_words.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(frame_data, f)
        print(f"Saved PaddleOCR word/parent boxes to {out_path}")

    return frame_data


def plot_frame_boxes(image_path, frame_result, output_path=None, show_labels=False):
    """
    Plot parent boxes (green) and word boxes (red) on an image.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = img.copy()

    # Parent boxes: green
    print(f"number of parent boxes: {len(frame_result.get('parent_boxes', []))}")
    for box in frame_result.get("parent_boxes", []):
        bbox = box.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (80, 220, 80), 2)
        # if show_labels:
        #     txt = str(box.get("text", ""))[:30]
        #     cv2.putText(img, txt, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 80), 1, cv2.LINE_AA)

    # Word boxes: red
    print(f"number of word boxes: {len(frame_result.get('word_boxes', []))}")
    word_boxes_to_plot = frame_result.get("word_boxes") or frame_result.get("word_boxes_raw", [])
    for box in word_boxes_to_plot:
        bbox = box.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 230), 1)
        if show_labels:
            txt = str(box.get("text", ""))[:20]
            cv2.putText(img, txt, (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (60, 60, 230), 1, cv2.LINE_AA)

    if output_path is not None:
        output_path = str(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, img)

    return img


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Minimal PaddleOCR pass for extracted video frames")
    parser.add_argument("--video", type=str, default=None, help="Video path (used only if frames are not extracted)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory (expects/creates frames/)")
    parser.add_argument("--lang", type=str, default="de", help="PaddleOCR language code")
    parser.add_argument("--frame_step", type=int, default=1, help="Extract every n-th frame if extraction is needed")
    parser.add_argument("--device", type=str, default="auto", help="Paddle device: auto, cpu, gpu:0")
    parser.add_argument("--ocr_version", type=str, default="PP-OCRv5", help="Paddle OCR version, e.g. PP-OCRv5")
    parser.add_argument("--plot_frame", type=int, default=None, help="Optional frame index to render debug plot")
    args = parser.parse_args()

    results = process_video_paddleocr(
        video_path=args.video,
        output_dir=args.output_dir,
        lang=args.lang,
        frame_step=args.frame_step,
        device=args.device,
        ocr_version=args.ocr_version,
    )

    if args.plot_frame is not None and args.plot_frame in results:
        frame_result = results[args.plot_frame]
        img_path = frame_result["frame_path"]
        out_plot = Path(args.output_dir) / f"frame_{args.plot_frame:04d}_paddle_boxes.jpg"
        plot_frame_boxes(img_path, frame_result, output_path=out_plot, show_labels=False)
        print(f"Saved debug plot to {out_plot}")


# Transitional re-export: use the shared postprocessing module while keeping the
# existing `ocr_paddle.py` notebook/API surface stable.
normalize_parent_boxes_by_line = _ocr_postprocess.normalize_parent_boxes_by_line
cluster_parent_box_heights = _ocr_postprocess.cluster_parent_box_heights
postprocess_word_boxes = _ocr_postprocess.postprocess_word_boxes

apply_parent_box_normalization = _ocr_postprocess.apply_parent_box_normalization
apply_parent_height_clustering = _ocr_postprocess.apply_parent_height_clustering
apply_word_box_postprocessing = _ocr_postprocess.apply_word_box_postprocessing
apply_box_stabilization = _ocr_postprocess.apply_box_stabilization
apply_parent_constrained_word_stabilization = _ocr_postprocess.apply_parent_constrained_word_stabilization

filter_boxes_by_names = _ocr_postprocess.filter_boxes_by_names
ocr_boxes_to_unified_paddle = _ocr_postprocess.ocr_boxes_to_unified_paddle


if __name__ == "__main__":
    main()
