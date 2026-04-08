"""
Export Anonymized Video - Apply blurring to reviewed annotations

This script takes the reviewed OCR boxes and SAM3 masks from the annotation viewer
and creates an anonymized video by applying blurring to all visible annotations.
"""

import os
import argparse
from pathlib import Path
import pickle
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from ait.utils import rebuild_full_mask


def apply_blur_to_region(image, mask, blur_strength=51):
    """
    Apply Gaussian blur to a region of the image defined by a mask.
    
    Args:
        image: Input image (BGR)
        mask: Boolean mask indicating region to blur
        blur_strength: Kernel size for Gaussian blur (must be odd)
        
    Returns:
        Image with blur applied to masked region
    """
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    # Create blurred version of entire image
    blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    
    # Copy blurred regions to original using mask
    result = image.copy()
    result[mask] = blurred[mask]
    
    return result


def add_custom_text_to_frame(frame, bbox, original_patch, x1, y1, x2, y2, font_path=None):
    """
    Add replacement text with color matching using PIL.
    
    Args:
        frame: OpenCV frame (BGR format)
        bbox: Bounding box dictionary containing alterego text
        original_patch: Unblurred region used to infer original text color
        x1, y1, x2, y2: Text region coordinates
        font_path: Path to custom font file (optional)
        
    Returns:
        Frame with custom text overlaid
    """
    text = bbox.get("alterego", "")
    if not text or not text.strip():
        return frame
    
    # Ensure text is properly decoded as UTF-8
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    # Ensure it's a proper Unicode string
    text = str(text)
    
    def infer_text_color(patch, fallback=(0, 0, 0)):
        """Infer original text color from unblurred patch."""
        if patch is None or patch.size == 0:
            return fallback
        try:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            return fallback
        if gray.size == 0:
            return fallback

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        count_255 = np.count_nonzero(binary)
        count_0 = binary.size - count_255
        text_value = 0 if count_0 < count_255 else 255

        text_mask = binary == text_value
        if np.count_nonzero(text_mask) < 5:
            text_mask = binary != text_value
            if np.count_nonzero(text_mask) < 5:
                return fallback

        text_pixels = patch[text_mask]
        if text_pixels.size == 0:
            return fallback

        mean_bgr = np.mean(text_pixels, axis=0)
        return tuple(int(np.clip(channel, 0, 255)) for channel in mean_bgr)
    
    # Infer text color from original patch
    text_color_bgr = infer_text_color(original_patch, fallback=(0, 0, 0))
    text_fill = (text_color_bgr[2], text_color_bgr[1], text_color_bgr[0], 255)  # BGR to RGBA
    
    # Convert OpenCV frame (BGR) to PIL Image (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Calculate font size based on bounding box height
    box_height = y2 - y1
    font_size = max(14, int(box_height * 1.))
    
    try:
        # Try to load custom font
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size, encoding='utf-8')
        else:
            # Fallback to system fonts with UTF-8 support
            fallback_fonts = [
                "arial.ttf", "Arial.ttf",           # Windows
                "arialuni.ttf",                     # Arial Unicode MS (good UTF-8 support)
                "Helvetica.ttc",                    # macOS
                "DejaVuSans.ttf",                   # Linux (excellent UTF-8 support)
                "NotoSans-Regular.ttf",             # Android/Linux (comprehensive Unicode)
                "Roboto-Regular.ttf"                # Android
            ]
            
            font = None
            for font_name in fallback_fonts:
                try:
                    font = ImageFont.truetype(font_name, font_size, encoding='utf-8')
                    break
                except (OSError, IOError):
                    continue
            
            if font is None:
                font = ImageFont.load_default()
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    # Get text bounding box for centering
    bbox_text = draw.textbbox((0, 0), text, font=font)
    text_width = bbox_text[2] - bbox_text[0]
    text_height = bbox_text[3] - bbox_text[1]
    
    # Account for text baseline offset from textbbox
    baseline_offset_y = bbox_text[1]
    
    # Calculate center position
    text_x = x1 + (x2 - x1 - text_width) // 2 - bbox_text[0]
    text_y = y1 + (y2 - y1 - text_height) // 2 - baseline_offset_y
    
    # Draw main text
    draw.text((text_x, text_y), text, font=font, fill=text_fill)
    
    # Convert back to OpenCV format (RGB to BGR)
    frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return frame_bgr


def load_transitions(video_dir):
    """
    Load transition ranges from transitions.txt file.
    
    Args:
        video_dir: Path to video directory
        
    Returns:
        List of (start_frame, end_frame) tuples
    """
    transitions_file = Path(video_dir) / "transitions.txt"
    transitions = []
    
    if transitions_file.exists():
        with open(transitions_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    start, end = line.split(',')
                    transitions.append((int(start), int(end)))
                except:
                    continue
    
    return transitions


def export_anonymized_video(video_dir, output_video_path, blur_strength=51, 
                           ocr_blur=True, sam3_blur=True, transition_blur=True,
                           fps=None, codec='mp4v', font_path=None):
    """
    Export an anonymized video with blurring applied to reviewed annotations.
    
    Args:
        video_dir (str or Path): Directory containing frames/, state.pkl (or ocr.pkl/sam3.pkl), transitions.txt
        output_video_path (str or Path): Path for output video file
        blur_strength (int): Kernel size for Gaussian blur (must be odd)
        ocr_blur (bool): Whether to blur OCR boxes
        sam3_blur (bool): Whether to blur SAM3 masks
        transition_blur (bool): Whether to blur transition frames
        fps (float): Output video FPS (default: auto-detect from frames)
        codec (str): Video codec fourcc code (default: 'mp4v')
        font_path (str): Path to custom font file (default: auto-detect MYRIADPRO-REGULAR.OTF)
    """
    video_dir = Path(video_dir)
    frames_dir = video_dir / "frames"
    state_file = video_dir / "state.pkl"
    ocr_file = video_dir / "ocr.pkl"
    sam3_file = video_dir / "sam3.pkl"
    
    # Auto-detect font file if not provided
    if font_path is None:
        # Look for MYRIADPRO-REGULAR.OTF in video directory or script directory
        script_dir = Path(__file__).parent
        possible_font_names = [
            "MYRIADPRO-REGULAR.OTF",
            "MyriadPro-Regular.otf",
            "MYRIADPRO-REGULAR.otf",
            "MyriadPro-Regular.OTF"
        ]
        
        for font_name in possible_font_names:
            # Check video directory first
            font_candidate = video_dir / font_name
            if font_candidate.exists():
                font_path = str(font_candidate)
                print(f"Found font: {font_path}")
                break
            # Check script directory
            font_candidate = script_dir / font_name
            if font_candidate.exists():
                font_path = str(font_candidate)
                print(f"Found font: {font_path}")
                break
    
    # Validate inputs
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    
    # Get all frame files
    frame_files = sorted(frames_dir.glob("*.png")) or sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    
    print(f"\n{'='*70}")
    print(f"EXPORTING ANONYMIZED VIDEO")
    print(f"Video directory: {video_dir}")
    print(f"Output: {output_video_path}")
    print(f"{'='*70}\n")
    
    # Load annotations
    ocr_annotations = {}
    sam3_annotations = {}
    
    # Prefer state.pkl (reviewed annotations) over original files
    if state_file.exists():
        print(f"Loading reviewed annotations from state.pkl...")
        with open(state_file, 'rb') as f:
            all_annotations = pickle.load(f)
        
        # Separate OCR and SAM3 annotations by source
        for frame_idx, annotations in all_annotations.items():
            ocr_list = []
            sam3_list = []
            for ann in annotations:
                if ann.get('source') == 'ocr':
                    ocr_list.append(ann)
                elif ann.get('source') == 'sam3':
                    sam3_list.append(ann)
            if ocr_list:
                ocr_annotations[frame_idx] = ocr_list
            if sam3_list:
                sam3_annotations[frame_idx] = sam3_list
        
        print(f"Loaded OCR annotations: {len(ocr_annotations)} frames")
        print(f"Loaded SAM3 annotations: {len(sam3_annotations)} frames")
    else:
        # Fall back to original files
        print(f"No state.pkl found, loading original annotations...")
        if ocr_blur and ocr_file.exists():
            with open(ocr_file, 'rb') as f:
                ocr_annotations = pickle.load(f)
            print(f"Loaded OCR annotations: {len(ocr_annotations)} frames")
        
        if sam3_blur and sam3_file.exists():
            with open(sam3_file, 'rb') as f:
                sam3_annotations = pickle.load(f)
            print(f"Loaded SAM3 annotations: {len(sam3_annotations)} frames")
    
    # Load transitions
    transitions = []
    if transition_blur:
        transitions = load_transitions(video_dir)
        print(f"Loaded {len(transitions)} transition ranges")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    height, width = first_frame.shape[:2]
    
    # Auto-detect FPS if not provided
    if fps is None:
        # Try to find the original video in the parent directory to read its FPS
        detected_fps = None
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
        video_name = video_dir.name
        # Check parent dir for a video matching the folder name
        parent = video_dir.parent
        for ext in video_extensions:
            candidate = parent / f"{video_name}{ext}"
            if candidate.exists():
                cap = cv2.VideoCapture(str(candidate))
                try:
                    fps_val = cap.get(cv2.CAP_PROP_FPS)
                    if fps_val and fps_val > 0:
                        detected_fps = float(fps_val)
                        print(f"Auto-detected FPS from {candidate.name}: {detected_fps}")
                finally:
                    cap.release()
                break
        
        fps = detected_fps if detected_fps else 30.0
        if detected_fps is None:
            print(f"Could not auto-detect FPS, using default: {fps}")
    else:
        print(f"Using specified FPS: {fps}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to create video writer for {output_video_path}")
    
    print(f"\nProcessing {len(frame_files)} frames...")
    print(f"Image size: {width}x{height}")
    transition_blur_strength = blur_strength * 3
    if transition_blur_strength % 2 == 0:
        transition_blur_strength += 1
    print(f"Blur strength: {blur_strength}")
    print(f"Transition blur strength: {transition_blur_strength}")
    print(f"OCR blur: {ocr_blur}, SAM3 blur: {sam3_blur}, Transition blur: {transition_blur}\n")
    
    # Process each frame — use the frame index from the filename, not enumeration order.
    # Annotations are keyed by the actual frame index (extracted from the filename
    # during OCR/SAM3 processing), which differs from enumeration order when
    # frame_step > 1 (e.g., filenames 0000, 0002, 0004 → indices 0, 2, 4).
    for frame_file in tqdm(frame_files, desc="Exporting video"):
        # Extract actual frame index from filename
        frame_stem = frame_file.stem
        frame_idx = int(''.join(filter(str.isdigit, frame_stem)))
        
        # Read frame
        frame = cv2.imread(str(frame_file))
        if frame is None:
            print(f"Warning: Could not read frame {frame_file}")
            continue
        
        # Create combined blur mask
        blur_mask = np.zeros((height, width), dtype=bool)
        
        # Check if this frame is in a transition
        in_transition = False
        if transition_blur:
            for start, end in transitions:
                if start <= frame_idx <= end:
                    in_transition = True
                    blur_mask[:, :] = True  # Blur entire frame
                    break
        
        # Store OCR boxes with text replacements for later
        ocr_text_overlays = []
        
        # If not in transition, apply OCR and SAM3 blurring
        if not in_transition:
            # Apply OCR box blurring
            if ocr_blur and frame_idx in ocr_annotations:
                # ocr_annotations[frame_idx] is a list of annotation dicts
                for ann in ocr_annotations[frame_idx]:
                    # Only blur if to_show is True (visible in annotation viewer)
                    if ann.get('to_show', True):
                        bbox = ann.get('bbox')
                        if bbox:
                            x1, y1, x2, y2 = bbox
                            # Clamp to image bounds
                            x1 = max(0, min(x1, width))
                            x2 = max(0, min(x2, width))
                            y1 = max(0, min(y1, height))
                            y2 = max(0, min(y2, height))
                            
                            # Save original patch before blurring for color inference
                            if x2 > x1 and y2 > y1:
                                original_patch = frame[y1:y2, x1:x2].copy()
                                ocr_text_overlays.append((ann, original_patch, x1, y1, x2, y2))
                                blur_mask[y1:y2, x1:x2] = True
            
            # Apply SAM3 mask blurring
            if sam3_blur and frame_idx in sam3_annotations:
                # sam3_annotations[frame_idx] is a list of annotation dicts
                for ann in sam3_annotations[frame_idx]:
                    # Only blur if to_show is True (visible in annotation viewer)
                    if ann.get('to_show', True):
                        mask_data = ann.get('mask')
                        if mask_data is not None:
                            try:
                                full_mask = rebuild_full_mask(mask_data, (height, width))
                                blur_mask |= full_mask
                            except Exception as e:
                                print(f"Warning: Error processing mask on frame {frame_idx}: {e}")
        
        # Apply blurring if there's anything to blur
        if blur_mask.any():
            strength = transition_blur_strength if in_transition else blur_strength
            frame = apply_blur_to_region(frame, blur_mask, strength)
        
        # Add custom text overlays for OCR boxes with alterego text
        for ann, original_patch, x1, y1, x2, y2 in ocr_text_overlays:
            if ann.get('alterego', '').strip():
                frame = add_custom_text_to_frame(frame, ann, original_patch, x1, y1, x2, y2, font_path)
        
        # Write frame to video
        out.write(frame)
    
    # Release video writer
    out.release()
    
    print(f"\n{'='*70}")
    print(f"EXPORT COMPLETE")
    print(f"Output video: {output_video_path}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Export anonymized video with blurring applied to reviewed annotations'
    )
    
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Directory containing frames/, ocr.pkl, sam3.pkl, transitions.txt')
    parser.add_argument('--output', type=str, required=True,
                       help='Output video file path (e.g., output.mp4)')
    parser.add_argument('--blur_strength', type=int, default=51,
                       help='Gaussian blur kernel size (must be odd, default: 51)')
    parser.add_argument('--fps', type=float, default=None,
                       help='Output video FPS (default: auto-detect or 30)')
    parser.add_argument('--codec', type=str, default='mp4v',
                       help='Video codec fourcc (default: mp4v)')
    parser.add_argument('--no_ocr', action='store_true',
                       help='Skip OCR box blurring')
    parser.add_argument('--no_sam3', action='store_true',
                       help='Skip SAM3 mask blurring')
    parser.add_argument('--no_transitions', action='store_true',
                       help='Skip transition frame blurring')
    
    args = parser.parse_args()
    
    # Validate blur strength
    if args.blur_strength % 2 == 0:
        args.blur_strength += 1
        print(f"Adjusted blur strength to {args.blur_strength} (must be odd)")
    
    try:
        export_anonymized_video(
            video_dir=args.video_dir,
            output_video_path=args.output,
            blur_strength=args.blur_strength,
            ocr_blur=not args.no_ocr,
            sam3_blur=not args.no_sam3,
            transition_blur=not args.no_transitions,
            fps=args.fps,
            codec=args.codec
        )
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
