"""
OCR package — docTR-based pipeline for video name detection.

Public API (matches the old ocr.py interface used by process_videos.py):
    from ocr import process_video_ocr, process_videos_batch
"""

from .pipeline import process_video_ocr, process_videos_batch

__all__ = ["process_video_ocr", "process_videos_batch"]
