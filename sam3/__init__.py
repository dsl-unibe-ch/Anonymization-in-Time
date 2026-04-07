"""
SAM3 package — profile picture detection via text-prompted segmentation.

Public API:
    from sam3 import process_video_sam3, process_videos_sam3_batch
"""

from .pipeline import process_video_sam3, process_videos_sam3_batch

__all__ = ["process_video_sam3", "process_videos_sam3_batch"]
