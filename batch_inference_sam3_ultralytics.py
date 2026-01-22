"""
Batch inference script for SAM3 using Ultralytics - Process all images in a folder with text prompts
Faster and more efficient than the original SAM3 implementation, with MPS support
"""

import os
import warnings
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

import argparse
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
import math
import cv2

try:
    from ultralytics.models.sam import SAM3SemanticPredictor
except ImportError:
    print("Error: Ultralytics SAM3 not found. Install with: pip install ultralytics")
    exit(1)

from utils import resolve_device, cleanup_device


def setup_predictor(device="auto", model_path="sam3.pt", conf=0.25, half=True):
    """Initialize SAM3 Semantic Predictor with Ultralytics"""
    device = resolve_device(device)
    
    # Convert device string for ultralytics
    if device == "mps":
        device_str = "mps"
    elif device == "cuda":
        device_str = "0"  # First CUDA device
    else:
        device_str = "cpu"
    
    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model=model_path,
        device=device_str,
        half=half and device in ["cuda", "mps"],  # Use FP16 for faster inference
        save=False,  # Don't auto-save, we'll handle that
        verbose=False,
    )
    
    predictor = SAM3SemanticPredictor(overrides=overrides)
    print(f"Loaded Ultralytics SAM3 on device: {device_str}")
    
    return predictor, device


def process_image(predictor, image_path, text_prompt, frame_idx=0):
    """
    Process a single image with SAM3 predictor
    
    Args:
        predictor: SAM3SemanticPredictor instance
        image_path: Path to image file
        text_prompt: Text prompt or list of text prompts
        frame_idx: Frame index for tracking
        
    Returns:
        dict with 'boxes', 'masks', 'scores', 'labels'
    """
    # Set the image
    predictor.set_image(str(image_path))
    
    # Run inference with text prompt(s)
    if isinstance(text_prompt, str):
        text_prompt = [text_prompt]
    
    results = predictor(text=text_prompt)
    
    # Extract results - Ultralytics returns a list of Results objects
    if not results or len(results) == 0:
        return {'boxes': [], 'masks': [], 'scores': [], 'labels': []}
    
    result = results[0]  # First (and only) result
    
    # Extract boxes, masks, and scores
    boxes = []
    masks = []
    scores = []
    labels = []
    
    if result.masks is not None and result.boxes is not None:
        # Get the data
        boxes_data = result.boxes.xyxy.cpu().numpy()  # XYXY format
        masks_data = result.masks.data.cpu().numpy()  # (N, H, W)
        scores_data = result.boxes.conf.cpu().numpy()  # Confidence scores
        
        for i in range(len(boxes_data)):
            boxes.append(boxes_data[i])
            masks.append(masks_data[i])
            scores.append(float(scores_data[i]))
            labels.append(1)  # Default label
    
    return {
        'boxes': boxes,
        'masks': masks,
        'scores': scores,
        'labels': labels
    }


def _pad_box(box, pad, img_h, img_w):
    """Pad and clamp a bounding box"""
    x1, y1, x2, y2 = box
    x1 = max(0, int(math.floor(x1 - pad)))
    y1 = max(0, int(math.floor(y1 - pad)))
    x2 = min(img_w - 1, int(math.ceil(x2 + pad)))
    y2 = min(img_h - 1, int(math.ceil(y2 + pad)))
    return [x1, y1, x2, y2]


def slim_results(results, img_shape, pad=4, pack_bits=True):
    """
    Crop masks to bounding boxes and optionally pack to save memory
    
    Args:
        results: dict with 'boxes', 'masks', 'scores', 'labels'
        img_shape: (H, W) of original image
        pad: pixels to expand each box before cropping
        pack_bits: if True, pack cropped mask to bytes to save RAM
    """
    if not results or 'masks' not in results or len(results['masks']) == 0:
        return results
    
    boxes = results.get('boxes', [])
    masks = results.get('masks', [])
    scores = results.get('scores', [])
    labels = results.get('labels', [])
    
    img_h, img_w = img_shape[:2]
    
    new_boxes = []
    new_masks = []
    new_scores = []
    new_labels = []
    
    for box, mask, score, label in zip(boxes, masks, scores, labels):
        box_np = np.array(box, dtype=float)
        mask_np = np.array(mask)
        
        if mask_np.ndim == 3:  # (1, H, W) -> (H, W)
            mask_np = mask_np[0]
        
        # Pad box
        px1, py1, px2, py2 = _pad_box(box_np, pad, img_h, img_w)
        
        # Crop mask
        crop = mask_np[py1:py2+1, px1:px2+1].astype(np.bool_)
        
        if pack_bits:
            packed = np.packbits(crop.reshape(-1).astype(np.uint8))
            new_masks.append({
                "packed": packed,
                "shape": crop.shape,
                "bbox": [px1, py1, px2, py2],
            })
        else:
            new_masks.append(crop)
        
        new_boxes.append(np.array([px1, py1, px2, py2], dtype=np.float32))
        new_scores.append(float(score))
        new_labels.append(int(label))
    
    return {
        'boxes': new_boxes,
        'masks': new_masks,
        'scores': new_scores,
        'labels': new_labels,
    }


def unpack_mask_entry(mask_entry):
    """Rebuild a cropped mask from a packed entry"""
    if isinstance(mask_entry, dict) and "packed" in mask_entry:
        h, w = mask_entry["shape"]
        flat = np.unpackbits(mask_entry["packed"])[: h * w]
        return flat.reshape((h, w)).astype(bool), mask_entry["bbox"]
    return np.array(mask_entry, dtype=bool), None


def rebuild_full_mask(mask_entry, img_shape):
    """Unpack cropped/packed mask into a full-frame boolean mask"""
    crop, bbox = unpack_mask_entry(mask_entry)
    
    if bbox is None:
        # Already full-frame
        if crop.shape[:2] == img_shape[:2]:
            return crop.astype(bool)
        return cv2.resize(
            crop.astype(np.float32),
            (img_shape[1], img_shape[0]),
            interpolation=cv2.INTER_NEAREST
        ) > 0.5
    
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Clamp bbox to image bounds
    x1 = max(0, min(x1, img_shape[1] - 1))
    x2 = max(0, min(x2, img_shape[1] - 1))
    y1 = max(0, min(y1, img_shape[0] - 1))
    y2 = max(0, min(y2, img_shape[0] - 1))
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    
    if crop.shape[0] != h or crop.shape[1] != w:
        crop = cv2.resize(crop.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
    
    full = np.zeros(img_shape[:2], dtype=bool)
    full[y1:y2+1, x1:x2+1] = crop.astype(bool)
    return full


def mask_entry_to_crop(mask_entry):
    """Return mask crop (float32) and its bbox"""
    crop, bbox = unpack_mask_entry(mask_entry)
    if bbox is None:
        h, w = crop.shape[:2]
        bbox = [0, 0, w - 1, h - 1]
    return crop.astype(np.float32), bbox


def make_mask_entry(crop_bool, bbox, pack_bits=True):
    """Pack a boolean crop with its bbox"""
    if pack_bits:
        packed = np.packbits(crop_bool.reshape(-1).astype(np.uint8))
        return {"packed": packed, "shape": crop_bool.shape, "bbox": bbox}
    return crop_bool.astype(bool)


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in XYXY format"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def match_masks_across_frames(results_list, iou_threshold=0.3):
    """
    Match masks across frames by IoU and position similarity
    Returns list of tracks where each track is a list of (frame_idx, mask_idx) pairs
    """
    if not results_list or len(results_list) == 0:
        return []
    
    tracks = []
    
    # Start tracks from first frame
    first_results = results_list[0][1]
    if first_results and 'boxes' in first_results:
        for mask_idx in range(len(first_results['boxes'])):
            tracks.append([(0, mask_idx)])
    
    # Match masks in subsequent frames
    for frame_idx in range(1, len(results_list)):
        curr_results = results_list[frame_idx][1]
        
        if not curr_results or 'boxes' not in curr_results:
            continue
        
        curr_boxes = curr_results['boxes']
        matched_tracks = set()
        matched_masks = set()
        
        # Try to match each current mask to existing tracks
        for mask_idx, curr_box in enumerate(curr_boxes):
            best_track_idx = -1
            best_iou = iou_threshold
            
            # Find best matching track
            for track_idx, track in enumerate(tracks):
                if track_idx in matched_tracks:
                    continue
                
                # Get last box in this track
                last_frame, last_mask_idx = track[-1]
                last_results = results_list[last_frame][1]
                last_box = last_results['boxes'][last_mask_idx]
                
                iou = calculate_iou(curr_box, last_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = track_idx
            
            if best_track_idx >= 0:
                tracks[best_track_idx].append((frame_idx, mask_idx))
                matched_tracks.add(best_track_idx)
                matched_masks.add(mask_idx)
            else:
                # Start new track
                tracks.append([(frame_idx, mask_idx)])
                matched_masks.add(mask_idx)
    
    return tracks


def propagate_missing_masks(results_list, tracks, max_gap=5):
    """Fill gaps in tracks by interpolating masks"""
    filled_count = 0
    
    for track in tracks:
        if len(track) < 2:
            continue
        
        # Check for gaps in this track
        for i in range(len(track) - 1):
            curr_frame, curr_mask = track[i]
            next_frame, next_mask = track[i + 1]
            gap_size = next_frame - curr_frame - 1
            
            if gap_size > 0 and gap_size <= max_gap:
                # Fill the gap by interpolating
                curr_results = results_list[curr_frame][1]
                next_results = results_list[next_frame][1]
                
                curr_box = curr_results['boxes'][curr_mask]
                next_box = next_results['boxes'][next_mask]
                curr_mask_data = curr_results['masks'][curr_mask]
                next_mask_data = next_results['masks'][next_mask]
                
                # Convert mask entries to crops
                curr_crop, curr_bbox = mask_entry_to_crop(curr_mask_data)
                next_crop, next_bbox = mask_entry_to_crop(next_mask_data)
                pack_output = isinstance(curr_mask_data, dict) and "packed" in curr_mask_data
                
                # Interpolate for each missing frame
                for gap_idx in range(1, gap_size + 1):
                    missing_frame = curr_frame + gap_idx
                    alpha = gap_idx / (gap_size + 1)
                    
                    # Interpolate box
                    interp_box = curr_box * (1 - alpha) + next_box * alpha
                    
                    # Interpolate mask
                    x1, y1, x2, y2 = interp_box
                    target_w = max(1, int(round(x2 - x1 + 1)))
                    target_h = max(1, int(round(y2 - y1 + 1)))
                    curr_resized = cv2.resize(curr_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    next_resized = cv2.resize(next_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    interp_mask = curr_resized * (1 - alpha) + next_resized * alpha
                    interp_bool = interp_mask > 0.5
                    bbox_int = [int(round(v)) for v in interp_box]
                    mask_entry = make_mask_entry(interp_bool, bbox_int, pack_bits=pack_output)
                    
                    # Add to results
                    missing_results = results_list[missing_frame][1]
                    if not missing_results or 'boxes' not in missing_results:
                        missing_results = {'boxes': [], 'masks': [], 'scores': []}
                        results_list[missing_frame] = (
                            results_list[missing_frame][0],
                            missing_results,
                            results_list[missing_frame][2]
                        )
                    
                    # Convert to lists if needed
                    if not isinstance(missing_results['boxes'], list):
                        missing_results['boxes'] = list(missing_results['boxes'])
                        missing_results['masks'] = list(missing_results['masks'])
                        missing_results['scores'] = list(missing_results['scores'])
                    
                    missing_results['boxes'].append(np.array(interp_box, dtype=np.float32))
                    missing_results['masks'].append(mask_entry)
                    missing_results['scores'].append(0.5)
                    
                    filled_count += 1
    
    return filled_count


def save_images_with_masks(img, results, output_path, mode='color', alpha=0.5, blur_strength=51):
    """Save image with mask overlays - either colored or blurred"""
    # Simple color palette
    COLORS = [
        np.array([1.0, 0.0, 0.0]),  # Red
        np.array([0.0, 1.0, 0.0]),  # Green
        np.array([0.0, 0.0, 1.0]),  # Blue
        np.array([1.0, 1.0, 0.0]),  # Yellow
        np.array([1.0, 0.0, 1.0]),  # Magenta
        np.array([0.0, 1.0, 1.0]),  # Cyan
    ]
    
    if not results or 'masks' not in results or len(results['masks']) == 0:
        return False
    
    # Convert PIL to numpy
    if isinstance(img, Image.Image):
        img_array = np.array(img)
    else:
        img_array = img.copy()
    
    # Ensure RGB
    if img_array.dtype == np.float32 or img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    img_array = img_array[..., :3]
    
    overlay = img_array.copy()
    
    # Calculate positions and sort masks
    mask_positions = []
    for i, (mask, box) in enumerate(zip(results['masks'], results['boxes'])):
        # Use box center for sorting
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        mask_positions.append((y_center, x_center, i))
    
    # Sort by position
    mask_positions.sort()
    
    # Process each mask in sorted order
    for color_idx, (_, _, original_idx) in enumerate(mask_positions):
        mask_entry = results['masks'][original_idx]
        mask_bool = rebuild_full_mask(mask_entry, img_array.shape)
        
        if mode == 'blur':
            # Blur the masked region
            blurred = cv2.GaussianBlur(overlay, (blur_strength, blur_strength), 0)
            overlay[mask_bool] = blurred[mask_bool]
        else:  # mode == 'color'
            # Get color based on sorted position
            color = COLORS[color_idx % len(COLORS)]
            color255 = (color * 255).astype(np.uint8)
            
            # Apply colored mask overlay
            for c in range(3):
                overlay[..., c][mask_bool] = (
                    alpha * color255[c] + (1 - alpha) * overlay[..., c][mask_bool]
                ).astype(np.uint8)
    
    # Save
    Image.fromarray(overlay).save(output_path)
    return True


def convert_results_to_unified_dict(all_results, tracks=None):
    """
    Convert SAM3 results list to unified per-frame dict format for compatibility
    """
    track_map = {}
    if tracks:
        for track_id, track in enumerate(tracks):
            for frame_idx, mask_idx in track:
                track_map[(frame_idx, mask_idx)] = track_id
    
    unified = {}
    for frame_idx, frame_results, _img_path in all_results:
        boxes = frame_results.get("boxes", [])
        masks = frame_results.get("masks", [])
        scores = frame_results.get("scores", [])
        n = min(len(boxes), len(masks), len(scores))
        
        for i in range(n):
            box = boxes[i]
            mask = masks[i]
            score = scores[i]
            
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            
            # Keep packed format to save space
            if isinstance(mask, dict) and "packed" in mask:
                mask_data = mask
            else:
                mask_np = np.array(mask).astype(bool)
                if mask_np.size == 0:
                    continue
                mask_data = mask_np
            
            # Get track_id if available
            track_id = track_map.get((frame_idx, i), None)
            
            unified.setdefault(frame_idx, []).append({
                "bbox": (x1, y1, x2, y2),
                "parent_box": None,
                "score": float(score),
                "text": "",
                "alterego": "",
                "mask": mask_data,
                "source": "sam3",
                "to_show": True,
                "track_id": track_id
            })
    
    return unified


def get_image_files(folder_path):
    """Get all image files from folder"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = set()
    
    for ext in image_extensions:
        image_files.update(Path(folder_path).glob(f'*{ext}'))
        image_files.update(Path(folder_path).glob(f'*{ext.upper()}'))
    
    return sorted(image_files)


def process_video_sam3_ultralytics(frames_folder, output_folder, 
                                   text_prompt="profile image, profile picture",
                                   device='auto', model_path="sam3.pt", conf=0.25,
                                   mask_mode='color', blur_strength=51,
                                   masks_propagation=True, max_gap=5, 
                                   save_images=False, pad=10):
    """
    Process frames from a single video with SAM3 using Ultralytics.
    
    Args:
        frames_folder: Path to folder containing extracted frames
        output_folder: Path to folder for saving results
        text_prompt: Text prompt to segment (can be a list)
        device: Device to use ('auto', 'cuda', 'mps', or 'cpu')
        model_path: Path to SAM3 model file
        conf: Confidence threshold
        mask_mode: Mask visualization mode ('color' or 'blur')
        blur_strength: Blur kernel size for blur mode (must be odd)
        masks_propagation: Enable temporal mask propagation
        max_gap: Maximum frame gap to fill when propagating masks
        save_images: Save images with masks overlaid
        pad: Padding for mask cropping
        
    Returns:
        dict: Unified SAM3 data with track_ids
    """
    frames_folder = Path(frames_folder)
    output_folder = Path(output_folder)
    
    print(f"\n{'='*60}")
    print(f"Processing SAM3 (Ultralytics) for: {frames_folder}")
    print(f"Output folder: {output_folder}")
    print(f"{'='*60}\n")
    
    # Validate blur strength
    if blur_strength % 2 == 0:
        blur_strength += 1
        print(f"Blur strength adjusted to {blur_strength} (must be odd)")
    
    # Setup predictor
    print("Loading Ultralytics SAM3 model...")
    predictor, device = setup_predictor(device, model_path, conf)
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all images
    image_files = get_image_files(frames_folder)
    print(f"Found {len(image_files)} images in {frames_folder}")
    
    if len(image_files) == 0:
        print("No images found!")
        return None
    
    print(f"Processing with text prompt: '{text_prompt}'")
    print(f"Device: {device}")
    print(f"Confidence threshold: {conf}")
    print(f"Mask mode: {mask_mode}")
    if mask_mode == 'blur':
        print(f"Blur strength: {blur_strength}")
    if not masks_propagation:
        print(f"Temporal propagation disabled")
    else:
        print(f"Temporal propagation enabled (max gap: {max_gap} frames)")
    
    pickle_path = output_folder / 'detected_masks.pkl'
    try:
        with open(pickle_path, 'rb') as f:
            all_results = pickle.load(f)
        print(f"Loaded existing results from {pickle_path}, skipping processing")
    except FileNotFoundError:
        all_results = []
        print("\nStarting processing...")
        
        # Process each image
        for frame_idx, img_path in enumerate(tqdm(image_files, desc="Processing frames")):
            try:
                # Get image shape for slimming
                img = Image.open(img_path).convert('RGB')
                img_shape = (img.height, img.width)
                
                # Process image
                results = process_image(predictor, img_path, text_prompt, frame_idx)
                
                # Slim results to save memory
                results = slim_results(results, img_shape, pad=pad, pack_bits=True)
                
                all_results.append((frame_idx, results, img_path))
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # Add empty results for this frame
                all_results.append((frame_idx, {'boxes': [], 'masks': [], 'scores': []}, img_path))
                continue
        
        print(f"\nSaving masks to pickle: {pickle_path}")
        with open(pickle_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Saved {len(all_results)} frames with masks")
    
    pickle_path_propagated = output_folder / 'detected_masks_propagated.pkl'
    tracks = None
    try:
        with open(pickle_path_propagated, 'rb') as f:
            all_results = pickle.load(f)
        print(f"Loaded existing propagated results from {pickle_path_propagated}, skipping propagation")
        
        tracks_path = output_folder / 'mask_tracks.pkl'
        try:
            with open(tracks_path, 'rb') as f:
                tracks = pickle.load(f)
            print(f"Loaded {len(tracks)} mask tracks")
        except FileNotFoundError:
            if len(all_results) > 1:
                tracks = match_masks_across_frames(all_results)
                print(f"Regenerated {len(tracks)} mask tracks")
    except FileNotFoundError:
        if masks_propagation and len(all_results) > 1:
            print(f"\nApplying temporal mask propagation...")
            tracks = match_masks_across_frames(all_results)
            print(f"Found {len(tracks)} mask tracks across {len(all_results)} frames")
            filled_count = propagate_missing_masks(all_results, tracks, max_gap)
            print(f"Filled {filled_count} missing mask(s)")
        
        print(f"\nSaving masks to pickle: {pickle_path_propagated}")
        with open(pickle_path_propagated, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Saved {len(all_results)} frames with propagated masks")
        
        if tracks:
            tracks_path = output_folder / 'mask_tracks.pkl'
            with open(tracks_path, 'wb') as f:
                pickle.dump(tracks, f)
            print(f"Saved {len(tracks)} mask tracks to {tracks_path}")
    
    # Save unified dict format to sam3.pkl
    unified_path = output_folder / 'sam3.pkl'
    try:
        with open(unified_path, 'rb') as f:
            unified = pickle.load(f)
        print(f"Unified detections already exist at {unified_path}, skipping conversion")
    except FileNotFoundError:
        unified = convert_results_to_unified_dict(all_results, tracks)
        with open(unified_path, 'wb') as f:
            pickle.dump(unified, f)
        print(f"Saved unified detections to {unified_path} ({len(unified)} frames)")
    
    if save_images:
        print(f"\nSaving processed images...")
        for frame_idx, frame_results, img_path in tqdm(all_results, desc="Saving images"):
            output_path = output_folder / f"{img_path.stem}_result.png"
            img = Image.open(img_path).convert('RGB')
            saved = save_images_with_masks(
                img, frame_results, output_path,
                mode=mask_mode, alpha=0.5, blur_strength=blur_strength
            )
            if not saved:
                img.save(output_path)
    
    print(f"SAM3 (Ultralytics) processing complete for {frames_folder.parent.name}")
    
    # Cleanup
    print("Cleaning up SAM3 model...")
    try:
        del predictor
        import gc
        gc.collect()
        cleanup_device(device)
        print("SAM3 model unloaded")
    except Exception as e:
        print(f"Warning: Model cleanup error: {e}")
    
    return unified


def process_videos_sam3_ultralytics_batch(video_folders, text_prompt="profile image, profile picture",
                                          device='auto', model_path="sam3.pt", conf=0.25,
                                          mask_mode='color', blur_strength=51,
                                          masks_propagation=True, max_gap=5, 
                                          save_images=False, pad=10):
    """
    Process multiple video folders with SAM3 using Ultralytics in batch.
    
    Args:
        video_folders: List of paths to video output folders
        text_prompt: Text prompt to segment
        device: Device to use ('auto', 'cuda', 'mps', or 'cpu')
        model_path: Path to SAM3 model file
        conf: Confidence threshold
        mask_mode: Mask visualization mode ('color' or 'blur')
        blur_strength: Blur kernel size for blur mode
        masks_propagation: Enable temporal mask propagation
        max_gap: Maximum frame gap to fill
        save_images: Save images with masks overlaid
        pad: Padding for mask cropping
        
    Returns:
        dict: Dictionary mapping video names to their unified SAM3 data
    """
    results = {}
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING SAM3 (Ultralytics): {len(video_folders)} videos")
    print(f"{'='*60}\n")
    
    device = resolve_device(device)
    print(f"Resolved SAM3 device for batch: {device}")
    
    for video_folder in video_folders:
        video_folder = Path(video_folder)
        frames_folder = video_folder / "frames"
        video_name = video_folder.name
        
        if not frames_folder.exists():
            print(f"✗ Skipping {video_name}: frames folder not found at {frames_folder}")
            results[video_name] = None
            continue
        
        try:
            unified = process_video_sam3_ultralytics(
                frames_folder=frames_folder,
                output_folder=video_folder,
                text_prompt=text_prompt,
                device=device,
                model_path=model_path,
                conf=conf,
                mask_mode=mask_mode,
                blur_strength=blur_strength,
                masks_propagation=masks_propagation,
                max_gap=max_gap,
                save_images=save_images,
                pad=pad
            )
            results[video_name] = unified
            print(f"✓ Successfully processed {video_name}")
        except Exception as e:
            print(f"✗ Error processing {video_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[video_name] = None
    
    print(f"\n{'='*60}")
    print(f"BATCH SAM3 (Ultralytics) PROCESSING COMPLETE")
    print(f"Successful: {sum(1 for v in results.values() if v is not None)}/{len(video_folders)}")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='SAM3 Batch Inference using Ultralytics')
    parser.add_argument('--input_folder', type=str, required=True,
                      help='Path to folder containing images')
    parser.add_argument('--output_folder', type=str, required=True,
                      help='Path to folder for saving results')
    parser.add_argument('--text_prompt', type=str, default="profile image, profile picture",
                      help='Text prompt to segment (e.g., "person", "car")')
    parser.add_argument('--model_path', type=str, default='sam3.pt',
                      help='Path to SAM3 model file (default: sam3.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='Confidence threshold (default: 0.25)')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use: auto|cuda|mps|cpu (default: auto)')
    parser.add_argument('--mask_mode', type=str, default='color', choices=['color', 'blur'],
                      help='Mask visualization mode: "color" or "blur"')
    parser.add_argument('--blur_strength', type=int, default=51,
                      help='Blur kernel size for blur mode (must be odd, default: 51)')
    parser.add_argument('--no_masks_propagation', action='store_true',
                      help='Disable temporal mask propagation')
    parser.add_argument('--max_gap', type=int, default=5,
                      help='Maximum frame gap to fill when propagating masks (default: 5)')
    parser.add_argument('--save_images', action='store_true',
                      help='Save images with masks overlaid')
    parser.add_argument('--pad', type=int, default=10,
                      help='Padding for mask cropping (default: 10)')
    
    args = parser.parse_args()
    
    # Validate blur strength
    if args.blur_strength % 2 == 0:
        args.blur_strength += 1
        print(f"Blur strength adjusted to {args.blur_strength} (must be odd)")
    
    # Process the video
    unified = process_video_sam3_ultralytics(
        frames_folder=args.input_folder,
        output_folder=args.output_folder,
        text_prompt=args.text_prompt,
        device=args.device,
        model_path=args.model_path,
        conf=args.conf,
        mask_mode=args.mask_mode,
        blur_strength=args.blur_strength,
        masks_propagation=not args.no_masks_propagation,
        max_gap=args.max_gap,
        save_images=args.save_images,
        pad=args.pad
    )
    
    if unified:
        print(f"\nDone! Results saved to {args.output_folder}")
        print(f"Processed {len(unified)} frames with detections")
    else:
        print(f"\nFailed to process video")


if __name__ == "__main__":
    main()
