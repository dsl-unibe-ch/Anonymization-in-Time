"""
Unified output format conversion for SAM3 results.
"""

import numpy as np
import cv2
from PIL import Image


def _is_yellow_emoji(img_path, bbox):
    """
    Check if the region inside bbox is a yellow emoji.

    Opens the frame image, crops to bbox, converts to HSV, and checks
    if >50% of pixels are in the canonical emoji-yellow range.
    """
    try:
        img = Image.open(img_path).convert('RGB')
        x1, y1, x2, y2 = bbox
        crop = np.array(img.crop((x1, y1, x2, y2)))
    except Exception:
        return False

    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    # Emoji yellow: H 15-45 (OpenCV H is 0-180), high saturation, high value
    lower = np.array([15, 80, 150])
    upper = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    yellow_ratio = mask.sum() / 255.0 / max(1, mask.size)
    return yellow_ratio > 0.5


def convert_to_unified_dict(all_results, tracks=None):
    """
    Convert SAM3 results list to unified per-frame dict format.

    Yellow emoji detections are marked with to_show=False.

    Returns dict keyed by frame_idx, each value is a list of detection dicts.
    """
    track_map = {}
    if tracks:
        for track_id, track in enumerate(tracks):
            for frame_idx, mask_idx in track:
                track_map[(frame_idx, mask_idx)] = track_id

    unified = {}
    emoji_count = 0

    for frame_idx, frame_results, img_path in all_results:
        boxes = frame_results.get("boxes", [])
        masks = frame_results.get("masks", [])
        scores = frame_results.get("scores", [])
        n = min(len(boxes), len(masks), len(scores))

        for i in range(n):
            box = boxes[i]
            mask = masks[i]
            score = scores[i]

            x1, y1, x2, y2 = [int(round(v)) for v in box]

            if isinstance(mask, dict) and "packed" in mask:
                mask_data = mask
            else:
                mask_np = np.array(mask).astype(bool)
                if mask_np.size == 0:
                    continue
                mask_data = mask_np

            track_id = track_map.get((frame_idx, i), None)

            # Check for yellow emoji
            is_emoji = False
            if img_path is not None:
                is_emoji = _is_yellow_emoji(img_path, (x1, y1, x2, y2))
                if is_emoji:
                    emoji_count += 1

            unified.setdefault(frame_idx, []).append({
                "bbox": (x1, y1, x2, y2),
                "parent_box": None,
                "score": float(score),
                "text": "",
                "alterego": "",
                "mask": mask_data,
                "source": "sam3",
                "to_show": not is_emoji,
                "track_id": track_id
            })

    if emoji_count > 0:
        print(f"  Marked {emoji_count} yellow emoji detection(s) as hidden")

    return unified
