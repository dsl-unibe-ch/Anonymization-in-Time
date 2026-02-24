import json
import os
import pickle
import re
import statistics
import string
import copy
from pathlib import Path

import cv2
from tqdm import tqdm

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
    return isinstance(text, str) and text != "" and text.strip() == ""


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
    for p in parent_boxes:
        if p.get("bbox") is None:
            continue
        parent_bbox_by_idx[p.get("parent_index")] = p.get("bbox")

    out = []
    for w in word_boxes:
        w2 = w.copy()
        pi = w2.get("parent_index")
        if pi in parent_bbox_by_idx:
            w2["parent_box"] = parent_bbox_by_idx[pi]
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
        frame["word_boxes"] = postprocess_word_boxes(word_boxes_with_synced_parents)
    return updated


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
    frame_data = apply_word_box_postprocessing(frame_data)

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
    img = cv2.imread(str(image_path)).copy()
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Parent boxes: green
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
    for box in frame_result.get("word_boxes", []):
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


if __name__ == "__main__":
    main()
