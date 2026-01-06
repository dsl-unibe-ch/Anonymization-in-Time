"""
Canvas Widget - displays frames with annotation overlays and handles interactions (Tkinter version).
"""
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk

from utils.mask_utils import rebuild_full_mask


class CanvasWidget(tk.Canvas):
    """Canvas widget for displaying video frames with annotation overlays."""
    
    # Color scheme
    COLOR_OCR_VISIBLE = "#00FF00"      # Green for visible OCR
    COLOR_OCR_HIDDEN = "#FFFF00"       # Yellow for hidden OCR
    COLOR_SAM_VISIBLE = "#0078FF"      # Blue for visible SAM
    COLOR_SAM_HIDDEN = "#FFA500"       # Orange for hidden SAM
    COLOR_HOVER = "#FF0000"            # Red for hover highlight
    
    def __init__(self, parent, frames_dir: Path, on_annotation_clicked=None):
        """
        Initialize the canvas widget.
        
        Args:
            parent: Parent widget
            frames_dir: Path to directory containing frame images
            on_annotation_clicked: Callback(annotation_id, frame_idx)
        """
        super().__init__(parent, bg='black', highlightthickness=0)
        
        self.frames_dir = Path(frames_dir)
        self.current_frame_idx: Optional[int] = None
        self.current_image: Optional[Image.Image] = None
        self.display_image: Optional[ImageTk.PhotoImage] = None
        self.annotations: list = []
        self.hovered_annotation: Optional[dict] = None
        self.show_hidden_preview = False
        self.on_annotation_clicked = on_annotation_clicked
        
        # Frame cache
        self.frame_cache: Dict[int, Image.Image] = {}
        self.max_cache_size = 50
        
        # Scaling info
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Bind mouse events
        self.bind('<Motion>', self._on_mouse_move)
        self.bind('<Button-1>', self._on_mouse_click)
        self.bind('<Configure>', self._on_resize)
    
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
            self.current_image = self.frame_cache[frame_idx]
            self._render()
            return True
        
        # Load from disk
        frame_path = self.frames_dir / f"{frame_idx:04d}.jpg"
        
        if not frame_path.exists():
            print(f"Frame not found: {frame_path}")
            return False
        
        try:
            self.current_image = Image.open(frame_path).convert('RGB')
            self._add_to_cache(frame_idx, self.current_image)
            self._render()
            return True
        except Exception as e:
            print(f"Error loading frame {frame_idx}: {e}")
            return False
    
    def _add_to_cache(self, frame_idx: int, image: Image.Image):
        """Add frame to cache with simple size management."""
        if len(self.frame_cache) >= self.max_cache_size:
            # Remove oldest (first) item
            first_key = next(iter(self.frame_cache))
            del self.frame_cache[first_key]
        
        self.frame_cache[frame_idx] = image.copy()
    
    def _render(self):
        """Render the frame with overlays."""
        if self.current_image is None:
            self.delete('all')
            return
        
        # Get canvas size
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Calculate scaling
        img_width, img_height = self.current_image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self.scale_factor = min(scale_x, scale_y)
        
        # Calculate display size and offsets
        display_width = int(img_width * self.scale_factor)
        display_height = int(img_height * self.scale_factor)
        self.offset_x = (canvas_width - display_width) // 2
        self.offset_y = (canvas_height - display_height) // 2
        
        # Create display image with overlays
        display_img = self.current_image.copy()
        display_img = display_img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        
        # Draw overlays
        display_img = self._draw_overlays(display_img)
        
        # Convert to PhotoImage
        self.display_image = ImageTk.PhotoImage(display_img)
        
        # Clear and draw
        self.delete('all')
        self.create_image(self.offset_x, self.offset_y, anchor='nw', image=self.display_image)
    
    def _draw_overlays(self, img: Image.Image) -> Image.Image:
        """Draw annotations on the image."""
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Store original image size for mask unpacking
        self._overlay_size = img.size
        
        # Draw masks first (below boxes)
        for ann in self.annotations:
            if ann.get('mask') is not None:
                self._draw_mask(overlay, ann)
        
        # Draw bounding boxes
        for ann in self.annotations:
            if ann.get('bbox') is not None:
                self._draw_bbox(draw, ann)
        
        # Composite overlay onto image
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        return img.convert('RGB')
    
    def _draw_mask(self, overlay: Image.Image, ann: dict):
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
            color_hex = self.COLOR_SAM_VISIBLE if is_visible else self.COLOR_SAM_HIDDEN
        else:
            color_hex = self.COLOR_OCR_VISIBLE if is_visible else self.COLOR_OCR_HIDDEN
        
        if is_hovered:
            color_hex = self.COLOR_HOVER
        
        # Convert hex to RGB
        color = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
        
        # Alpha for visibility
        alpha = 76 if is_visible else 40
        
        # Unpack mask if it's packed (from SAM3)
        try:
            # Check if mask is packed (dict with 'packed' key)
            if isinstance(mask, dict):
                # Get ORIGINAL image dimensions (H, W)
                orig_img_h, orig_img_w = self.current_image.size[1], self.current_image.size[0]
                orig_shape = (orig_img_h, orig_img_w)
                
                # Rebuild full-frame mask at ORIGINAL resolution
                mask_bool = rebuild_full_mask(mask, orig_shape)
                
                # Convert to PIL and scale to display size
                mask_img = Image.fromarray((mask_bool * 255).astype(np.uint8), mode='L')
                mask_img = mask_img.resize(overlay.size, Image.Resampling.NEAREST)
            else:
                # Regular numpy array mask
                # Ensure mask is 2D
                if mask.ndim == 3:
                    mask = mask.squeeze()
                
                # Convert to PIL and scale to display size
                mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                mask_img = mask_img.resize(overlay.size, Image.Resampling.NEAREST)
            
            # Create colored overlay
            color_overlay = Image.new('RGBA', overlay.size, color + (alpha,))
            overlay.paste(color_overlay, (0, 0), mask_img)
        except Exception as e:
            print(f"Error drawing mask: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to drawing just the bbox if available
            pass
    
    def _draw_bbox(self, draw: ImageDraw.Draw, ann: dict):
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
            int(x1 * self.scale_factor),
            int(y1 * self.scale_factor),
            int(x2 * self.scale_factor),
            int(y2 * self.scale_factor)
        )
        
        # Choose color
        if is_hovered:
            color = self.COLOR_HOVER
        elif ann.get('source') == 'ocr':
            color = self.COLOR_OCR_VISIBLE if is_visible else self.COLOR_OCR_HIDDEN
        else:
            color = self.COLOR_SAM_VISIBLE if is_visible else self.COLOR_SAM_HIDDEN
        
        # Width
        width = 2 if is_hovered else 1
        
        # Draw rectangle
        if not is_visible:
            # Dashed line for hidden
            self._draw_dashed_rectangle(draw, scaled_bbox, color, width)
        else:
            draw.rectangle(scaled_bbox, outline=color, width=width)
        
        # Draw text for OCR
        if ann.get('source') == 'ocr' and ann.get('text') and is_visible:
            text = ann.get('text', '')
            label = text
            
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Text background
            bbox_text = draw.textbbox((scaled_bbox[0], scaled_bbox[1] - 18), label, font=font)
            draw.rectangle(bbox_text, fill=(0, 0, 0, 180))
            draw.text((scaled_bbox[0], scaled_bbox[1] - 18), label, fill='white', font=font)
    
    def _draw_dashed_rectangle(self, draw: ImageDraw.Draw, bbox: Tuple[int, int, int, int], 
                               color: str, width: int, dash_length: int = 5):
        """Draw a dashed rectangle."""
        x1, y1, x2, y2 = bbox
        
        # Top edge
        for x in range(x1, x2, dash_length * 2):
            draw.line([(x, y1), (min(x + dash_length, x2), y1)], fill=color, width=width)
        
        # Bottom edge
        for x in range(x1, x2, dash_length * 2):
            draw.line([(x, y2), (min(x + dash_length, x2), y2)], fill=color, width=width)
        
        # Left edge
        for y in range(y1, y2, dash_length * 2):
            draw.line([(x1, y), (x1, min(y + dash_length, y2))], fill=color, width=width)
        
        # Right edge
        for y in range(y1, y2, dash_length * 2):
            draw.line([(x2, y), (x2, min(y + dash_length, y2))], fill=color, width=width)
    
    def _on_mouse_move(self, event):
        """Handle mouse movement for hover detection."""
        if self.current_image is None or self.current_frame_idx is None:
            return
        
        # Convert to image coordinates
        img_x = int((event.x - self.offset_x) / self.scale_factor)
        img_y = int((event.y - self.offset_y) / self.scale_factor)
        
        # Find annotation at this point
        hovered = self._find_annotation_at_point(img_x, img_y)
        
        # Update hover state
        if hovered != self.hovered_annotation:
            self.hovered_annotation = hovered
            self._render()
            
            # Update cursor
            self.config(cursor='hand2' if hovered else 'arrow')
    
    def _on_mouse_click(self, event):
        """Handle mouse click for toggling annotations."""
        if self.hovered_annotation and self.on_annotation_clicked:
            annotation_id = self.hovered_annotation.get('id')
            self.on_annotation_clicked(annotation_id, self.current_frame_idx)
    
    def _on_resize(self, event):
        """Handle window resize."""
        if self.current_image is not None:
            self._render()
    
    def _find_annotation_at_point(self, x: int, y: int) -> Optional[dict]:
        """Find annotation at given point."""
        # Check in reverse order (top annotations first)
        for ann in reversed(self.annotations):
            # Check bounding box
            if ann.get('bbox'):
                x1, y1, x2, y2 = ann['bbox']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return ann
            
            # Check mask
            if ann.get('mask') is not None:
                try:
                    mask = ann['mask']
                    
                    # Unpack if needed
                    if isinstance(mask, dict):
                        from utils.mask_utils import unpack_mask_entry
                        crop, bbox = unpack_mask_entry(mask)
                        if bbox is not None:
                            # Check if point is in bbox first
                            x1, y1, x2, y2 = bbox
                            if not (x1 <= x <= x2 and y1 <= y <= y2):
                                continue
                            # Check mask crop
                            local_x = x - x1
                            local_y = y - y1
                            if 0 <= local_y < crop.shape[0] and 0 <= local_x < crop.shape[1]:
                                if crop[local_y, local_x] > 0:
                                    return ann
                        else:
                            # Full frame mask
                            if 0 <= y < crop.shape[0] and 0 <= x < crop.shape[1]:
                                if crop[y, x] > 0:
                                    return ann
                    else:
                        # Regular numpy array mask
                        if mask.ndim == 3:
                            mask = mask.squeeze()
                        
                        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                            if mask[y, x] > 0:
                                return ann
                except Exception as e:
                    print(f"Error checking mask at point: {e}")
                    continue
        
        return None
    
    def toggle_hidden_preview(self):
        """Toggle preview mode for hidden annotations."""
        self.show_hidden_preview = not self.show_hidden_preview
        self._render()
    
    def clear(self):
        """Clear the canvas."""
        self.current_frame_idx = None
        self.current_image = None
        self.display_image = None
        self.annotations = []
        self.hovered_annotation = None
        self.delete('all')
