"""
Canvas Widget - displays frames with annotation overlays and handles interactions (Tkinter version).
"""
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
import sys

from utils.mask_utils import rebuild_full_mask


class CanvasWidget(tk.Canvas):
    """Canvas widget for displaying video frames with annotation overlays."""
    
    # Color scheme
    COLOR_OCR_VISIBLE = "#00FF00"      # Green for visible OCR
    COLOR_OCR_HIDDEN = "#FF0000"       # Red for hidden OCR
    COLOR_SAM_VISIBLE = "#0078FF"      # Blue for visible SAM
    COLOR_SAM_HIDDEN = "#FFA500"       # Orange for hidden SAM
    COLOR_HOVER = "#FF0000"            # Red for hover highlight
    
    def __init__(self, parent, frames_dir: Path, on_annotation_clicked=None):
        """
        Initialize the canvas widget.
        
        Args:
            parent: Parent widget
            frames_dir: Path to directory containing frame images
            on_annotation_clicked: Callback(annotation_id, frame_idx, is_right_click=False)
        """
        super().__init__(parent, bg='black', highlightthickness=0)
        
        self.frames_dir = Path(frames_dir)
        self.current_frame_idx: Optional[int] = None
        self.current_image: Optional[Image.Image] = None
        self.display_image: Optional[ImageTk.PhotoImage] = None
        self.annotations: list = []
        self.hovered_annotation: Optional[dict] = None
        self.show_hidden_preview = False
        self.preview_mode = False  # Blur preview mode
        self.on_annotation_clicked = on_annotation_clicked
        self.is_transition = False  # Track if current frame is in transition
        
        # Frame cache
        self.frame_cache: Dict[int, Image.Image] = {}
        self.max_cache_size = 50
        
        # Scaling info
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Bind mouse events
        self.bind('<Motion>', self._on_mouse_move)
        self.bind('<Button-1>', self._on_mouse_click)  # Left click
        self.bind('<Button-3>', self._on_right_click)  # Right click
        # macOS: right-click can be Button-2 or Control+Click
        if sys.platform == "darwin":
            self.bind('<Button-2>', self._on_right_click)
            self.bind('<Control-Button-1>', self._on_right_click)
        self.bind('<Configure>', self._on_resize)
    
    def load_frame(self, frame_idx: int, annotations: list, is_transition: bool = False) -> bool:
        """
        Load and display a frame with its annotations.
        
        Args:
            frame_idx: Frame index
            annotations: List of annotation dictionaries
            is_transition: Whether this frame is in a transition range
            
        Returns:
            True if successful
        """
        self.current_frame_idx = frame_idx
        self.annotations = annotations
        self.hovered_annotation = None
        self.is_transition = is_transition
        
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
        
        # Apply blur preview if enabled
        if self.preview_mode:
            display_img = self._apply_blur_preview(display_img)
        
        display_img = display_img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        
        # Draw overlays (only if not in preview mode)
        if not self.preview_mode:
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
        
        # Draw transition indicator if in transition
        if self.is_transition:
            # Draw thin yellow border
            border_width = 5
            width, height = img.size
            # Top border
            draw.rectangle([0, 0, width, border_width], fill=(255, 255, 0, 200))
            # Bottom border
            draw.rectangle([0, height - border_width, width, height], fill=(255, 255, 0, 200))
            # Left border
            draw.rectangle([0, 0, border_width, height], fill=(255, 255, 0, 200))
            # Right border
            draw.rectangle([width - border_width, 0, width, height], fill=(255, 255, 0, 200))
            
            # Draw text label
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            text = "TRANSITION FRAME"
            # Get text bbox for background
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position at top center
            text_x = (width - text_width) // 2
            text_y = 20
            
            # Draw background rectangle
            padding = 10
            draw.rectangle(
                [text_x - padding, text_y - padding, 
                 text_x + text_width + padding, text_y + text_height + padding],
                fill=(255, 255, 0, 220)
            )
            
            # Draw text
            draw.text((text_x, text_y), text, fill=(0, 0, 0, 255), font=font)
        
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
                mask_img = Image.fromarray(((mask_bool > 0) * 255).astype(np.uint8), mode='L')
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
        """Handle left mouse click.
        
        Normal click  -> toggle single word box
        Shift+click   -> toggle full parent line (same as right-click)
        """
        if self.hovered_annotation and self.on_annotation_clicked:
            annotation_id = self.hovered_annotation.get('id')
            # Shift held → parent-box toggle (same as right-click)
            is_shift = bool(event.state & 0x1)
            self.on_annotation_clicked(annotation_id, self.current_frame_idx, is_right_click=is_shift)
    
    def _on_right_click(self, event):
        """Handle right mouse click for toggling parent boxes."""
        if self.hovered_annotation and self.on_annotation_clicked:
            annotation_id = self.hovered_annotation.get('id')
            self.on_annotation_clicked(annotation_id, self.current_frame_idx, is_right_click=True)
    
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
    
    def _apply_blur_preview(self, img: Image.Image) -> Image.Image:
        """Apply blur and text replacement preview to visible annotations."""
        # Convert PIL to OpenCV format
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        height, width = cv_img.shape[:2]
        
        # Create blur mask
        blur_mask = np.zeros((height, width), dtype=bool)
        
        # Store OCR boxes with text replacements
        ocr_text_overlays = []
        
        # Process visible annotations
        for ann in self.annotations:
            if not ann.get('to_show', True):
                continue
            
            bbox = ann.get('bbox')
            if not bbox:
                continue
            
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # For OCR, save original patch
            if ann.get('source') == 'ocr':
                original_patch = cv_img[y1:y2, x1:x2].copy()
                ocr_text_overlays.append((ann, original_patch, x1, y1, x2, y2))
                blur_mask[y1:y2, x1:x2] = True
            
            # For SAM3, use mask
            elif ann.get('source') == 'sam3':
                mask_data = ann.get('mask')
                if mask_data is not None:
                    try:
                        from utils.mask_utils import rebuild_full_mask
                        full_mask = rebuild_full_mask(mask_data, (height, width))
                        blur_mask |= full_mask
                    except:
                        blur_mask[y1:y2, x1:x2] = True
        
        # Apply blur
        if blur_mask.any():
            kernel_size = 51  # Match export blur strength
            blurred = cv2.GaussianBlur(cv_img, (kernel_size, kernel_size), 0)
            cv_img[blur_mask] = blurred[blur_mask]
        
        # Add text overlays
        for ann, original_patch, x1, y1, x2, y2 in ocr_text_overlays:
            alterego = ann.get('alterego', '').strip()
            if alterego:
                cv_img = self._add_text_overlay(cv_img, ann, original_patch, x1, y1, x2, y2)
        
        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    
    def _add_text_overlay(self, cv_img, ann, original_patch, x1, y1, x2, y2):
        """Add replacement text with color matching."""
        text = ann.get('alterego', '').strip()
        if not text:
            return cv_img
        
        # Ensure text is properly decoded as UTF-8
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        # Ensure it's a proper Unicode string
        text = str(text)
        
        # Infer text color from original patch
        def infer_color(patch):
            if patch is None or patch.size == 0:
                return (0, 0, 0)
            try:
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                count_255 = np.count_nonzero(binary)
                count_0 = binary.size - count_255
                text_value = 0 if count_0 < count_255 else 255
                
                text_mask = binary == text_value
                if np.count_nonzero(text_mask) < 5:
                    text_mask = binary != text_value
                    if np.count_nonzero(text_mask) < 5:
                        return (0, 0, 0)
                
                text_pixels = patch[text_mask]
                if text_pixels.size == 0:
                    return (0, 0, 0)
                
                mean_bgr = np.mean(text_pixels, axis=0)
                return tuple(int(np.clip(c, 0, 255)) for c in mean_bgr)
            except:
                return (0, 0, 0)
        
        text_color_bgr = infer_color(original_patch)
        text_fill = (text_color_bgr[2], text_color_bgr[1], text_color_bgr[0], 255)
        
        # Convert to PIL for text rendering
        cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # Calculate font size
        box_height = y2 - y1
        font_size = max(14, int(box_height * 1.))
        
        try:
            # Try system fonts with UTF-8 support
            font = None
            fallback_fonts = [
                "arial.ttf", "Arial.ttf",        # Windows
                "arialuni.ttf",                  # Arial Unicode MS
                "DejaVuSans.ttf",                # Linux (excellent UTF-8)
                "NotoSans-Regular.ttf",          # Comprehensive Unicode
                "Roboto-Regular.ttf"             # Android
            ]
            for font_name in fallback_fonts:
                try:
                    font = ImageFont.truetype(font_name, font_size, encoding='utf-8')
                    break
                except:
                    continue
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Get text size and center
        bbox_text = draw.textbbox((0, 0), text, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # Account for text baseline offset from textbbox
        baseline_offset_y = bbox_text[1]
        
        text_x = x1 + (x2 - x1 - text_width) // 2 - bbox_text[0]
        text_y = y1 + (y2 - y1 - text_height) // 2 - baseline_offset_y
        
        # Draw text
        draw.text((text_x, text_y), text, font=font, fill=text_fill)
        
        # Convert back to OpenCV
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def toggle_hidden_preview(self):
        """Toggle preview mode for hidden annotations."""
        self.show_hidden_preview = not self.show_hidden_preview
        self._render()
    
    def set_preview_mode(self, enabled: bool):
        """Set blur preview mode."""
        self.preview_mode = enabled
        self._render()
    
    def clear(self):
        """Clear the canvas."""
        self.current_frame_idx = None
        self.current_image = None
        self.display_image = None
        self.annotations = []
        self.hovered_annotation = None
        self.delete('all')
