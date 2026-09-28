"""
Frame-file ordering and extraction metadata (dependency-light).

Extracted frames are named ``f"{frame_idx:04d}.jpg"``: four digits up to 9999,
five from 10000 on. Plain alphabetical sorting therefore breaks for videos
longer than 9,999 frames (``10000.jpg`` sorts before ``1001.jpg``). Always order
frame files, and key per-frame results, by the number in the file name.

``video_info.json`` (written next to the frames at extraction time) records the
source frame rate and frame step, so the export can rebuild a video at the
right speed without having to find the original file.
"""

import json
from pathlib import Path

FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
VIDEO_INFO_NAME = "video_info.json"


def frame_number(path):
    """Frame index encoded in a frame file name, or None if it has no digits.

    Uses all digits of the stem, like the OCR pipeline and the export always did.
    """
    digits = "".join(ch for ch in Path(path).stem if ch.isdigit())
    return int(digits) if digits else None


def _frame_sort_key(path):
    n = frame_number(path)
    return (n is None, n if n is not None else 0, Path(path).name)


def sort_frame_files(paths):
    """Sort frame paths by frame number (files without a number go last, by name)."""
    return sorted(paths, key=_frame_sort_key)


def list_frame_files(folder, extensions=FRAME_EXTENSIONS):
    """All frame images in ``folder`` (extension match is case-insensitive), in frame order."""
    exts = {e.lower() for e in extensions}
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sort_frame_files(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def order_results_by_frame(all_results):
    """
    Put per-frame SAM3 results ``[(frame_idx, results, img_path), ...]`` in frame
    order and key each one by the frame number of its image.

    Caches written before the fix are in alphabetical file order and keyed by
    list position; this repairs them. Returns ``(results, changed)``.
    """
    ordered = sorted(all_results, key=lambda item: _frame_sort_key(item[2]))
    fixed = []
    for pos, (_, results, img_path) in enumerate(ordered):
        n = frame_number(img_path)
        fixed.append((n if n is not None else pos, results, img_path))
    changed = any(new[0] != old[0] or new[2] != old[2] for new, old in zip(fixed, all_results))
    return fixed, changed


def write_video_info(frames_dir, source, fps, frame_step=1, frame_count=None,
                     start_frame=0, end_frame=None):
    """Record how the frames in ``frames_dir`` were extracted."""
    info = {
        "source": str(source),
        "fps": float(fps) if fps else None,
        "frame_step": int(frame_step) if frame_step else 1,
        "frame_count": int(frame_count) if frame_count is not None else None,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame) if end_frame is not None else None,
    }
    path = Path(frames_dir) / VIDEO_INFO_NAME
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return path


def read_video_info(frames_dir):
    """Metadata written by :func:`write_video_info`, or None if missing or unreadable."""
    path = Path(frames_dir) / VIDEO_INFO_NAME
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) else None
