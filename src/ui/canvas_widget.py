"""
Canvas Widget - displays frames with annotation overlays and handles interactions.
"""
from pathlib import Path
from typing import Optional, Dict
import numpy as np
from PIL import Image

from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPixmap, QPainter, QColor, QImage, QPen
from PySide6.QtCore import Qt, QPoint, Signal, QRect

from ..utils.overlay_renderer import (
    numpy_to_qimage, create_mask_overlay, draw_bbox,
    point_in_bbox, point_in_mask
)


class CanvasWidget(QWidget):
    """Widget for displaying video frames with annotation overlays."""
    
    # Signals
    annotation_clicked = Signal(int, int)  # annotation_id, frame_idx
    
    # Color scheme
    COLOR_OCR_VISIBLE = QColor(0, 255, 0)      # Green for visible OCR
    COLOR_OCR_HIDDEN = QColor(255, 255, 0)     # Yellow for hidden OCR
    COLOR_SAM_VISIBLE = QColor(0, 120, 255)    # Blue for visible SAM
    COLOR_SAM_HIDDEN = QColor(255, 165, 0)     # Orange for hidden SAM
    COLOR_HOVER = QColor(255, 0, 0)            # Red for hover highlight
    
    def __init__(self, frames_dir: Path, parent=None):
        """
        Initialize the canvas widget.
        
        Args:
            frames_dir: Path to directory containing frame images
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.frames_dir = Path(frames_dir)
        self.current_frame_idx: Optional[int] = None
        self.current_pixmap: Optional[QPixmap] = None
        self.annotations: list = []
        self.hovered_annotation: Optional[dict] = None
        self.show_hidden_preview = False
        
        # Frame cache (simple LRU-like cache)
        self.frame_cache: Dict[int, QPixmap] = {}
        self.max_cache_size = 50
        
        # Enable mouse tracking for hover
        self.setMouseTracking(True)
        
        # Set minimum size
        self.setMinimumSize(800, 600)
    
    def load_frame(self, frame_idx: int, annotations: list) -> bool:
        """
        Load and display a frame with its annotations.
        
        Args:
            frame_idx: Frame index
            annotations: List of annotation dictionaries
            
        Returns:
            True if successful
        """
        self.current_frame_idx = frame_idx
        self.annotations = annotations
        self.hovered_annotation = None
        
        # Check cache first
        if frame_idx in self.frame_cache:
            self.current_pixmap = self.frame_cache[frame_idx]
            self.update()
            return True
        
        # Load from disk
        frame_path = self.frames_dir / f"{frame_idx:04d}.jpg"
        
        if not frame_path.exists():
            print(f"Frame not found: {frame_path}")
            return False
        
        try:
            # Load image
            self.current_pixmap = QPixmap(str(frame_path))
            
            # Add to cache
            self._add_to_cache(frame_idx, self.current_pixmap)
            
            self.update()
            return True
            
        except Exception as e:
            print(f"Error loading frame {frame_idx}: {e}")
            return False
    
    def _add_to_cache(self, frame_idx: int, pixmap: QPixmap):
        """Add frame to cache with simple size management."""
        if len(self.frame_cache) >= self.max_cache_size:
            # Remove oldest (first) item
            first_key = next(iter(self.frame_cache))
            del self.frame_cache[first_key]
        
        self.frame_cache[frame_idx] = pixmap
    
    def paintEvent(self, event):
        """Paint the canvas with frame and overlays."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.current_pixmap is None:
            # Draw empty background
            painter.fillRect(self.rect(), Qt.GlobalColor.black)
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 
                           "No frame loaded")
            return
        
        # Draw frame image
        # Scale to fit widget while maintaining aspect ratio
        scaled_pixmap = self.current_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Center the image
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)
        
        # Calculate scaling factor
        scale_x = scaled_pixmap.width() / self.current_pixmap.width()
        scale_y = scaled_pixmap.height() / self.current_pixmap.height()
        
        # Translate painter to image position
        painter.translate(x, y)
        
        # Draw annotations
        self._draw_annotations(painter, scale_x, scale_y)
    
    def _draw_annotations(self, painter: QPainter, scale_x: float, scale_y: float):
        """Draw all annotations on the painter."""
        
        # Draw masks first (below boxes)
        for ann in self.annotations:
            if ann.get('mask') is not None:
                self._draw_mask(painter, ann, scale_x, scale_y)
        
        # Draw bounding boxes
        for ann in self.annotations:
            if ann.get('bbox') is not None:
                self._draw_bbox(painter, ann, scale_x, scale_y)
    
    def _draw_mask(self, painter: QPainter, ann: dict, scale_x: float, scale_y: float):
        """Draw a single mask annotation."""
        mask = ann.get('mask')
        if mask is None:
            return
        
        is_visible = ann.get('to_show', True)
        is_hovered = self.hovered_annotation and self.hovered_annotation.get('id') == ann.get('id')
        
        # Skip hidden masks unless hovered or preview mode
        if not is_visible and not is_hovered and not self.show_hidden_preview:
            return
        
        # Choose color
        if ann.get('source') == 'sam3':
            color = self.COLOR_SAM_VISIBLE if is_visible else self.COLOR_SAM_HIDDEN
        else:
            color = self.COLOR_OCR_VISIBLE if is_visible else self.COLOR_OCR_HIDDEN
        
        # Highlight hovered
        if is_hovered:
            color = self.COLOR_HOVER
        
        # Adjust alpha for hidden items
        alpha = 76 if is_visible else 40  # Less transparent for visible
        
        # Create mask overlay
        mask_img = create_mask_overlay(mask, color, alpha)
        
        # Scale mask
        scaled_mask = mask_img.scaled(
            int(mask_img.width() * scale_x),
            int(mask_img.height() * scale_y),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Draw mask
        painter.drawImage(0, 0, scaled_mask)
    
    def _draw_bbox(self, painter: QPainter, ann: dict, scale_x: float, scale_y: float):
        """Draw a single bounding box annotation."""
        bbox = ann.get('bbox')
        if bbox is None:
            return
        
        is_visible = ann.get('to_show', True)
        is_hovered = self.hovered_annotation and self.hovered_annotation.get('id') == ann.get('id')
        
        # Skip hidden boxes unless hovered or preview mode
        if not is_visible and not is_hovered and not self.show_hidden_preview:
            return
        
        # Scale bbox
        x1, y1, x2, y2 = bbox
        scaled_bbox = (
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y)
        )
        
        # Choose color
        if is_hovered:
            color = self.COLOR_HOVER
        elif ann.get('source') == 'ocr':
            color = self.COLOR_OCR_VISIBLE if is_visible else self.COLOR_OCR_HIDDEN
        else:
            color = self.COLOR_SAM_VISIBLE if is_visible else self.COLOR_SAM_HIDDEN
        
        # Choose thickness and style
        thickness = 3 if is_hovered else 2
        dashed = not is_visible
        
        # Get text for OCR
        text = None
        if ann.get('source') == 'ocr' and ann.get('text'):
            score = ann.get('score', 0)
            text = f"{ann.get('text')} ({score:.2f})"
        
        # Draw bbox
        draw_bbox(painter, scaled_bbox, color, thickness, dashed, text)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for hover detection."""
        if self.current_pixmap is None or self.current_frame_idx is None:
            return
        
        # Get mouse position relative to image
        pos = event.pos()
        
        # Calculate image bounds
        scaled_width = self.current_pixmap.width() * min(
            self.width() / self.current_pixmap.width(),
            self.height() / self.current_pixmap.height()
        )
        scaled_height = self.current_pixmap.height() * min(
            self.width() / self.current_pixmap.width(),
            self.height() / self.current_pixmap.height()
        )
        
        x_offset = (self.width() - scaled_width) / 2
        y_offset = (self.height() - scaled_height) / 2
        
        # Convert to image coordinates
        img_x = int((pos.x() - x_offset) / scaled_width * self.current_pixmap.width())
        img_y = int((pos.y() - y_offset) / scaled_height * self.current_pixmap.height())
        
        # Find annotation at this point
        hovered = self._find_annotation_at_point(img_x, img_y)
        
        # Update hover state
        if hovered != self.hovered_annotation:
            self.hovered_annotation = hovered
            self.update()
            
            # Update cursor
            if hovered:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def mousePressEvent(self, event):
        """Handle mouse click for toggling annotations."""
        if event.button() == Qt.MouseButton.LeftButton and self.hovered_annotation:
            annotation_id = self.hovered_annotation.get('id')
            self.annotation_clicked.emit(annotation_id, self.current_frame_idx)
    
    def _find_annotation_at_point(self, x: int, y: int) -> Optional[dict]:
        """Find annotation at given point."""
        point = QPoint(x, y)
        
        # Check in reverse order (top annotations first)
        for ann in reversed(self.annotations):
            # Check bounding box
            if ann.get('bbox'):
                if point_in_bbox(point, ann['bbox']):
                    return ann
            
            # Check mask
            if ann.get('mask') is not None:
                if point_in_mask(point, ann['mask']):
                    return ann
        
        return None
    
    def toggle_hidden_preview(self):
        """Toggle preview mode for hidden annotations."""
        self.show_hidden_preview = not self.show_hidden_preview
        self.update()
    
    def clear(self):
        """Clear the canvas."""
        self.current_frame_idx = None
        self.current_pixmap = None
        self.annotations = []
        self.hovered_annotation = None
        self.update()
