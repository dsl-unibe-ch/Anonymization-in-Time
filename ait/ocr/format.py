"""
Convert stabilized OCR boxes to the unified output format consumed by
the viewer (main_window_tk.py), annotation manager, and export_video.py.

Output schema per box:
    {
        "bbox":       (x1, y1, x2, y2),   int pixel coords
        "parent_box": (x1, y1, x2, y2) or None,
        "score":      float,               docTR word confidence
        "text":       str,                 detected OCR text
        "alterego":   str,                 replacement name ("" if none)
        "mask":       None,                always None for OCR source
        "source":     "ocr",
        "to_show":    bool,
        "track_id":   int or None,
    }
"""


def _to_int_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return (int(round(float(x1))), int(round(float(y1))),
            int(round(float(x2))), int(round(float(y2))))


def to_unified(frame_boxes: dict) -> dict:
    """
    Convert {frame_idx: [box_dict, ...]} to the unified output format.

    Args:
        frame_boxes: Dict from stabilization.stabilize().

    Returns:
        {frame_idx: [unified_box, ...]}
    """
    unified = {}
    for frame_idx, boxes in frame_boxes.items():
        out = []
        for box in boxes:
            bbox = box.get("bbox")
            if not bbox:
                continue

            parent = box.get("parent_box")
            if parent is not None:
                try:
                    parent = _to_int_bbox(parent)
                except Exception:
                    parent = None

            track_id = box.get("track_id")
            if isinstance(track_id, tuple):
                track_id = track_id[0]

            out.append({
                "bbox":       _to_int_bbox(bbox),
                "parent_box": parent,
                "score":      float(box.get("confidence", 1.0)),
                "text":       box.get("text", "") or "",
                "alterego":   box.get("alterego", "") or "",
                "mask":       None,
                "source":     "ocr",
                "to_show":    bool(box.get("to_show", True)),
                "track_id":   track_id,
            })
        unified[frame_idx] = out
    return unified
