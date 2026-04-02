"""
OCR package — video name detection with docTR or EasyOCR.

Public API:
    from ocr import process_video_ocr, process_videos_batch
"""

from .pipeline import process_video_ocr, process_videos_batch

__all__ = ["process_video_ocr", "process_videos_batch"]
