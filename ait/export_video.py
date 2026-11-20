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

from ait.utils import (rebuild_full_mask, shared_alterego_font_sizes,
                       track_alterego_font_sizes, load_alterego_font,
                       fit_alterego_font_size)
from ait.ocr.steady import hidden_box
from ait.frame_files import VIDEO_INFO_NAME, list_frame_files, read_video_info
from ait.obfuscation import (DEFAULT_METHOD, IMAGE_DEFAULT, METHODS, TEXT_DEFAULT,
                             infer_text_color, obfuscate_region, obfuscate_regions,
                             text_core_color, track_text_color)


def track_text_colors(ocr_annotations, frame_files):
    """
    One alterego colour per OCR track (median of its frames' ink colours,
    then snapped), read in a pass over the frames that have shown names.

    Returns:
        dict mapping track_id -> BGR colour.
    """
    files = {int(''.join(filter(str.isdigit, f.stem))): f for f in frame_files}
    samples = {}
    for frame_idx in sorted(ocr_annotations):
        anns = [a for a in ocr_annotations[frame_idx]
                if a.get('to_show', True) and a.get('bbox')
                and a.get('track_id') is not None and (a.get('alterego') or '').strip()]
        if not anns or frame_idx not in files:
            continue
        frame = cv2.imread(str(files[frame_idx]))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        for a in anns:
            x1, y1, x2, y2 = a['bbox']
            x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
            if x2 > x1 and y2 > y1:
                samples.setdefault(a['track_id'], []).append(
                    text_core_color(frame[y1:y2, x1:x2]))
    return {t: track_text_color(c) for t, c in samples.items()}


def apply_blur_to_region(image, mask, blur_strength=91, method=DEFAULT_METHOD):
    """
    Hide a masked region of the image (see ait.obfuscation).

    The default 'inpaint' method refills the region from its surroundings
    before blurring, so the output cannot be matched against candidate names
    or pictures; 'blur' is the old Gaussian blur, which can be.
    """
    return obfuscate_region(image, mask, blur_strength, method)


def add_custom_text_to_frame(frame, bbox, original_patch, x1, y1, x2, y2,
                              font_path=None, font_size=None, text_box=None,
                              text_color=None):
    """
    Add replacement text with color matching using PIL.

    Args:
        frame: OpenCV frame (BGR format)
        bbox: Bounding box dictionary containing alterego text
        original_patch: Unblurred region used to infer original text color
        x1, y1, x2, y2: Text region coordinates
        font_path: Path to custom font file (optional)
        font_size: If provided, use this exact font size (otherwise auto-fit)
        text_box: Unclamped (x1, y1, x2, y2) to position the text in, for a box
                  running past the frame edge (a name scrolling out); the text
                  is then clipped to that box within the frame
        text_color: BGR colour to use (e.g. one per track, see
                    track_text_colors); inferred from original_patch if None

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
    
    # Infer text color from original patch (coarsely, see infer_text_color)
    text_color_bgr = text_color or infer_text_color(original_patch, fallback=(0, 0, 0))
    text_fill = (text_color_bgr[2], text_color_bgr[1], text_color_bgr[0], 255)  # BGR to RGBA
    
    # Convert OpenCV frame (BGR) to PIL Image (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    tx1, ty1, tx2, ty2 = text_box if text_box is not None else (x1, y1, x2, y2)
    if font_size is None:
        font_size = fit_alterego_font_size(text, tx2 - tx1, ty2 - ty1, font_path)
    font = load_alterego_font(font_path, font_size)

    # Get text bounding box for centering
    bbox_text = draw.textbbox((0, 0), text, font=font)
    text_width = bbox_text[2] - bbox_text[0]
    text_height = bbox_text[3] - bbox_text[1]

    # Account for text baseline offset from textbbox
    baseline_offset_y = bbox_text[1]

    # Calculate center position
    text_x = tx1 + (tx2 - tx1 - text_width) // 2 - bbox_text[0]
    text_y = ty1 + (ty2 - ty1 - text_height) // 2 - baseline_offset_y

    # Draw main text
    draw.text((text_x, text_y), text, font=font, fill=text_fill)

    # Convert back to OpenCV format (RGB to BGR)
    frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    if (tx1, ty1, tx2, ty2) != (x1, y1, x2, y2):
        # Text box differs from the region (runs past the frame edge, or is a
        # steadied box): keep only the text inside the text box, in the frame
        h, w = frame.shape[:2]
        cx1, cx2 = max(0, min(tx1, w)), max(0, min(tx2, w))
        cy1, cy2 = max(0, min(ty1, h)), max(0, min(ty2, h))
        clipped = frame.copy()
        clipped[cy1:cy2, cx1:cx2] = frame_bgr[cy1:cy2, cx1:cx2]
        return clipped
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


DEFAULT_FPS = 30.0


def resolve_export_fps(video_dir, fps=None):
    """
    Frame rate for the exported video.

    Order: explicit ``fps`` -> ``frames/video_info.json`` written at extraction
    (source fps divided by frame_step, so the export keeps real-time speed) ->
    a source video named like the folder in its parent directory (legacy layout)
    -> 30 fps with a warning.
    """
    if fps:
        print(f"Using specified FPS: {fps}")
        return fps

    video_dir = Path(video_dir)
    info = read_video_info(video_dir / "frames")
    if info and info.get("fps"):
        step = max(1, int(info.get("frame_step") or 1))
        fps = float(info["fps"]) / step
        step_note = f" / frame step {step}" if step > 1 else ""
        print(f"FPS from extraction info: {info['fps']}{step_note} -> {fps:g}")
        return fps

    for ext in ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'):
        candidate = video_dir.parent / f"{video_dir.name}{ext}"
        if candidate.exists():
            cap = cv2.VideoCapture(str(candidate))
            try:
                fps_val = cap.get(cv2.CAP_PROP_FPS)
            finally:
                cap.release()
            if fps_val and fps_val > 0:
                print(f"Auto-detected FPS from {candidate.name}: {fps_val}")
                return float(fps_val)
            break

    print(f"⚠️  Could not determine the source frame rate; using {DEFAULT_FPS:g} fps. "
          f"If the source had a different rate, the export plays at the wrong speed: "
          f"pass --fps, or re-run processing so frames/{VIDEO_INFO_NAME} is written.")
    return DEFAULT_FPS


def resolve_sam3_file(video_dir, sam3_source="auto"):
    """
    Pick which SAM3 pickle to load when there is no reviewed state.pkl.

    Args:
        video_dir (Path): Processed video directory
        sam3_source (str): 'original', 'circular', or 'auto'. In auto mode the
            choice recorded by the annotation viewer (mask_choice.txt) wins,
            then sam3.pkl, then sam3_circular.pkl if it is the only one there.

    Returns:
        Path to the SAM3 pickle to use (may not exist).
    """
    video_dir = Path(video_dir)
    sam3_file = video_dir / "sam3.pkl"
    sam3_circular_file = video_dir / "sam3_circular.pkl"

    if sam3_source == "circular":
        return sam3_circular_file
    if sam3_source == "original":
        return sam3_file

    choice_file = video_dir / "mask_choice.txt"
    if choice_file.exists():
        try:
            choice = choice_file.read_text().strip().lower()
        except OSError:
            choice = ""
        if choice == "circular" and sam3_circular_file.exists():
            return sam3_circular_file
        if choice == "original" and sam3_file.exists():
            return sam3_file
    if not sam3_file.exists() and sam3_circular_file.exists():
        return sam3_circular_file
    return sam3_file


def export_anonymized_video(video_dir, output_video_path, blur_strength=91,
                           ocr_blur=True, sam3_blur=True, transition_blur=True,
                           fps=None, codec='mp4v', font_path=None,
                           sam3_source='auto', method=None,
                           text_method=TEXT_DEFAULT, image_method=IMAGE_DEFAULT):
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
        sam3_source (str): Which SAM3 masks to use when no state.pkl exists:
            'original' (sam3.pkl), 'circular' (sam3_circular.pkl), or 'auto'
            (follow the choice made in the annotation viewer, if recorded).
            Ignored when state.pkl exists — its masks are already the reviewed ones.
        text_method (str): How names (OCR boxes) are hidden: 'inpaint'
            (default), 'shade', 'flat', or 'blur' (legacy; can be undone by
            matching candidates, see tools/blur_attack.py).
        image_method (str): How profile pictures (SAM3 masks) are hidden;
            same choices, default 'shade'.
        method (str): If given, overrides both text_method and image_method.
    """
    if method:
        text_method = image_method = method
    video_dir = Path(video_dir)
    frames_dir = video_dir / "frames"
    state_file = video_dir / "state.pkl"
    ocr_file = video_dir / "ocr.pkl"
    sam3_file = resolve_sam3_file(video_dir, sam3_source)
    
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
    
    # Get all frame files, in frame order (numeric: 10000.jpg comes after 9999.jpg)
    frame_files = list_frame_files(frames_dir, (".png",)) or list_frame_files(frames_dir, (".jpg",))
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
        if sam3_source in ("original", "circular"):
            print(f"Note: --masks {sam3_source} ignored — state.pkl contains "
                  f"the masks reviewed in the annotation viewer.")
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
            print(f"Loaded SAM3 annotations from {sam3_file.name}: "
                  f"{len(sam3_annotations)} frames")
    
    # Load transitions
    transitions = []
    if transition_blur:
        transitions = load_transitions(video_dir)
        print(f"Loaded {len(transitions)} transition ranges")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    height, width = first_frame.shape[:2]
    
    fps = resolve_export_fps(video_dir, fps)

    # One font size per OCR track: fitting every frame's slightly wobbling box
    # made static fake names jump between sizes.
    track_font_sizes = track_alterego_font_sizes(
        ocr_annotations,
        lambda ann: fit_alterego_font_size(ann.get('alterego', ''),
                                           ann['bbox'][2] - ann['bbox'][0],
                                           ann['bbox'][3] - ann['bbox'][1], font_path),
    ) if ocr_blur else {}
    # One alterego colour per track: snapped per frame, video noise made it
    # flicker between shades. (Box steadying happens in the OCR pipeline.)
    track_colors = track_text_colors(ocr_annotations, frame_files) if ocr_blur else {}

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
    print(f"Obfuscation: names {text_method}, pictures {image_method}, "
          f"blur strength: {blur_strength}")
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
        
        # Masks: whole frame (transitions), names (OCR), pictures (SAM3)
        blur_mask = np.zeros((height, width), dtype=bool)
        text_mask = np.zeros((height, width), dtype=bool)
        image_mask = np.zeros((height, width), dtype=bool)
        
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
                            # Hide the steadied box and the detection it came
                            # from, so steadying never uncovers detected text
                            hx1, hy1, hx2, hy2 = hidden_box(ann)
                            hx1, hx2 = max(0, min(hx1, width)), max(0, min(hx2, width))
                            hy1, hy2 = max(0, min(hy1, height)), max(0, min(hy2, height))
                            text_mask[hy1:hy2, hx1:hx2] = True
            
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
                                image_mask |= full_mask
                            except Exception as e:
                                print(f"Warning: Error processing mask on frame {frame_idx}: {e}")
        
        # Apply blurring if there's anything to blur
        if in_transition:
            # Transitions keep the full-frame Gaussian blur (nothing around
            # a full-frame mask to refill from).
            frame = obfuscate_region(frame, blur_mask, transition_blur_strength, "blur")
        else:
            # Names under pictures' masks are hidden as pictures.
            frame = obfuscate_regions(
                frame, [(text_mask & ~image_mask, text_method),
                        (image_mask, image_method)], blur_strength)
        
        # Uniform font size per matched name — so adjacent words of one
        # alterego render at the same visual size.
        anns_only = [t[0] for t in ocr_text_overlays]
        coords_by_id = {id(t[0]): (t[2], t[3], t[4], t[5]) for t in ocr_text_overlays}
        def _per_box_size(ann):
            key = (ann.get('track_id'), (ann.get('alterego') or '').strip())
            if key in track_font_sizes:
                return track_font_sizes[key]   # stable over time
            x1, y1, x2, y2 = coords_by_id[id(ann)]
            return fit_alterego_font_size(ann.get('alterego', ''), x2 - x1, y2 - y1, font_path)
        group_sizes = shared_alterego_font_sizes(anns_only, _per_box_size)
        for ann, original_patch, x1, y1, x2, y2 in ocr_text_overlays:
            if ann.get('alterego', '').strip():
                frame = add_custom_text_to_frame(
                    frame, ann, original_patch, x1, y1, x2, y2,
                    font_path, font_size=group_sizes.get(id(ann)),
                    text_box=tuple(ann['bbox']),
                    text_color=track_colors.get(ann.get('track_id')),
                )
        
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
    parser.add_argument('--blur_strength', type=int, default=91,
                       help='Gaussian blur kernel size (must be odd, default: 91)')
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
    parser.add_argument('--masks', type=str, default='auto',
                       choices=['auto', 'original', 'circular'],
                       help='Which SAM3 masks to use when no state.pkl exists: '
                            'original (sam3.pkl), circular (sam3_circular.pkl), '
                            'or auto (follow the annotation viewer choice, default)')
    parser.add_argument('--text_method', type=str, default=TEXT_DEFAULT, choices=list(METHODS),
                       help='How names are hidden: inpaint (refill from the '
                            'surroundings, then blur; default), shade (refill, then '
                            'a shaded shape), flat (gray, then blur), or blur (legacy '
                            'Gaussian blur of the original pixels, which can be '
                            'matched against candidates)')
    parser.add_argument('--image_method', type=str, default=IMAGE_DEFAULT, choices=list(METHODS),
                       help='How profile pictures are hidden; same choices, default shade')
    parser.add_argument('--method', type=str, default=None, choices=list(METHODS),
                       help='Use this method for both names and profile pictures')
    
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
            codec=args.codec,
            sam3_source=args.masks,
            method=args.method,
            text_method=args.text_method,
            image_method=args.image_method,
        )
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
