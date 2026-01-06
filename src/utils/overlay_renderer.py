"""
Utility functions for rendering overlays on frames.
"""
import numpy as np
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont, QPainterPath
from PySide6.QtCore import Qt, QRect, QPoint
from typing import Tuple, Optional


def numpy_to_qimage(array: np.ndarray) -> QImage:
    """
    Convert numpy array to QImage.
    
    Args:
        array: Numpy array (H, W, 3) in RGB format
        
    Returns:
        QImage object
    """
    height, width, channel = array.shape
    bytes_per_line = 3 * width
    return QImage(array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)


def create_mask_overlay(mask: np.ndarray, color: QColor, alpha: int = 76) -> QImage:
    """
    Create a semi-transparent overlay from a binary mask.
    
    Args:
        mask: Binary mask array (H, W) or (H, W, 1)
        color: Color for the mask
        alpha: Transparency (0-255, default 76 = ~30%)
        
    Returns:
        QImage with colored semi-transparent mask
    """
    if mask.ndim == 3:
        mask = mask.squeeze()
    
    height, width = mask.shape
    
    # Create RGBA image
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Set color where mask is True
    mask_bool = mask > 0
    overlay[mask_bool, 0] = color.red()
    overlay[mask_bool, 1] = color.green()
    overlay[mask_bool, 2] = color.blue()
    overlay[mask_bool, 3] = alpha
    
    bytes_per_line = 4 * width
    return QImage(overlay.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)


def draw_bbox(painter: QPainter, bbox: Tuple[int, int, int, int], 
              color: QColor, thickness: int = 2, dashed: bool = False,
              text: Optional[str] = None) -> None:
    """
    Draw bounding box on painter.
    
    Args:
        painter: QPainter object
        bbox: Bounding box (x1, y1, x2, y2)
        color: Box color
        thickness: Line thickness
        dashed: Whether to use dashed line
        text: Optional text to display
    """
    x1, y1, x2, y2 = bbox
    
    # Set pen
    pen = QPen(color, thickness)
    if dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    
    # Draw rectangle
    painter.drawRect(x1, y1, x2 - x1, y2 - y1)
    
    # Draw text if provided
    if text:
        # Draw text background
        font = QFont("Arial", 10)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_rect = metrics.boundingRect(text)
        
        # Position text above bbox
        text_x = x1
        text_y = y1 - 5
        
        # Draw semi-transparent background
        bg_rect = QRect(text_x, text_y - text_rect.height(), 
                       text_rect.width() + 4, text_rect.height() + 2)
        painter.fillRect(bg_rect, QColor(0, 0, 0, 180))
        
        # Draw text
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.drawText(text_x + 2, text_y - 2, text)


def point_in_bbox(point: QPoint, bbox: Tuple[int, int, int, int]) -> bool:
    """
    Check if point is inside bounding box.
    
    Args:
        point: QPoint
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        True if point is inside bbox
    """
    x1, y1, x2, y2 = bbox
    return x1 <= point.x() <= x2 and y1 <= point.y() <= y2


def point_in_mask(point: QPoint, mask: np.ndarray) -> bool:
    """
    Check if point is inside mask.
    
    Args:
        point: QPoint
        mask: Binary mask array
        
    Returns:
        True if point is inside mask region
    """
    if mask is None:
        return False
    
    if mask.ndim == 3:
        mask = mask.squeeze()
    
    y, x = point.y(), point.x()
    
    # Check bounds
    if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
        return False
    
    return mask[y, x] > 0
