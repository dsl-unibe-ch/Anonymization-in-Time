"""
Unified output format conversion for SAM3 results.
"""

import numpy as np


def convert_to_unified_dict(all_results, tracks=None):
    """
    Convert SAM3 results list to unified per-frame dict format.

    Returns dict keyed by frame_idx, each value is a list of detection dicts.
    """
    track_map = {}
    if tracks:
        for track_id, track in enumerate(tracks):
            for frame_idx, mask_idx in track:
                track_map[(frame_idx, mask_idx)] = track_id

    unified = {}
    for frame_idx, frame_results, _img_path in all_results:
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

            unified.setdefault(frame_idx, []).append({
                "bbox": (x1, y1, x2, y2),
                "parent_box": None,
                "score": float(score),
                "text": "",
                "alterego": "",
                "mask": mask_data,
                "source": "sam3",
                "to_show": True,
                "track_id": track_id
            })

    return unified
