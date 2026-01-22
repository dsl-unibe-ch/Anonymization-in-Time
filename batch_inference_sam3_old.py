"""
Batch inference script for SAM3 - Process all images in a folder with a single text prompt
"""

import os
import warnings
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

import sys
sys.path.append("sam3")

import argparse
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
import math
import cv2
import sam3
from sam3 import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from sam3.eval.postprocessors import PostProcessImage
from utils import resolve_device, cleanup_device

# Get SAM3 root directory
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")


def setup_model(device="auto"):
    """Load and configure the SAM3 model"""
    device = resolve_device(device)
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)
    
    if device != "cpu":
        model = model.to(device)
    
    model.eval()
    
    # Enable optimizations only if CUDA
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Note: Don't use global autocast here - it breaks other models like EasyOCR
        # Use autocast locally in inference functions instead
    
    return model


def setup_transforms():
    """Create preprocessing transforms"""
    return ComposeAPI(
        transforms=[
            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def setup_postprocessor():
    """Create postprocessor for results"""
    return PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.5,
        to_cpu=True,  # Move to CPU for saving
    )

def create_empty_datapoint():
    """ A datapoint is a single image on which we can apply several queries at once. """
    return Datapoint(find_queries=[], images=[])

def set_image(datapoint, pil_image):
    """ Add the image to be processed to the datapoint """
    w,h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h,w])]

def create_datapoint(pil_image, text_prompt, query_id):
    """Create a datapoint with text prompt"""

    datapoint = create_empty_datapoint()
    set_image(datapoint, pil_image)

    w, h = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_prompt,
            image_id=0,
            object_ids_output=[], # unused for inference
            is_exhaustive=True, # unused for inference
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=query_id,
                original_image_id=query_id,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )
    )
    
    return datapoint

def _pad_box(box, pad, img_h, img_w):
    x1, y1, x2, y2 = box
    x1 = max(0, int(math.floor(x1 - pad)))
    y1 = max(0, int(math.floor(y1 - pad)))
    x2 = min(img_w - 1, int(math.ceil(x2 + pad)))
    y2 = min(img_h - 1, int(math.ceil(y2 + pad)))
    return [x1, y1, x2, y2]

def slim_batch_results(results, pad=4, pack_bits=True):
    """
    Args:
        results: dict from process_batch() -> {img_id: {'boxes','masks','scores','labels'}}
        pad: pixels to expand each box before cropping the mask
        pack_bits: if True, pack cropped mask to bytes to save RAM
    Returns:
        dict with same top-level keys; boxes are padded/clamped, masks are cropped (and packed if requested)
        Mask entries become either np.bool_ arrays (if pack_bits=False) or dicts with packed data.
    """
    slimmed = {}
    for img_id, det in results.items():
        if not det:
            slimmed[img_id] = det
            continue

        boxes = det.get("boxes", [])
        masks = det.get("masks", [])
        scores = det.get("scores", [])
        labels = det.get("labels", [])

        new_boxes, new_masks, new_scores, new_labels = [], [], [], []

        for box, mask, score, label in zip(boxes, masks, scores, labels):
            box_np = box.detach().cpu().numpy() if torch.is_tensor(box) else np.array(box, dtype=float)
            mask_np = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.array(mask)
            if mask_np.ndim == 3:  # (1,H,W) -> (H,W)
                mask_np = mask_np[0]
            img_h, img_w = mask_np.shape[-2], mask_np.shape[-1]

            px1, py1, px2, py2 = _pad_box(box_np, pad, img_h, img_w)
            crop = mask_np[py1:py2+1, px1:px2+1].astype(np.bool_)

            if pack_bits:
                packed = np.packbits(crop.reshape(-1).astype(np.uint8))
                new_masks.append({
                    "packed": packed,
                    "shape": crop.shape,
                    "bbox": [px1, py1, px2, py2],  # location in full frame
                })
            else:
                new_masks.append(crop)

            new_boxes.append(np.array([px1, py1, px2, py2], dtype=np.float32))
            new_scores.append(float(score.detach().cpu().item()) if torch.is_tensor(score) else float(score))
            new_labels.append(int(label.detach().cpu().item()) if torch.is_tensor(label) else int(label))

        slimmed[img_id] = {
            "boxes": new_boxes,
            "masks": new_masks,
            "scores": new_scores,
            "labels": new_labels,
        }
    return slimmed

def unpack_mask_entry(mask_entry):
    """Rebuild a cropped mask from a packed entry."""
    if isinstance(mask_entry, dict) and "packed" in mask_entry:
        h, w = mask_entry["shape"]
        flat = np.unpackbits(mask_entry["packed"])[: h * w]
        return flat.reshape((h, w)).astype(bool), mask_entry["bbox"]
    return np.array(mask_entry, dtype=bool), None

def rebuild_full_mask(mask_entry, img_shape):
    """Unpack cropped/packed mask into a full-frame boolean mask aligned to the original image."""
    crop, bbox = unpack_mask_entry(mask_entry)
    if bbox is None:
        # Already full-frame; if shape mismatches, resize as a fallback
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
    """Return mask crop (float32) and its bbox. If no bbox, assume full-frame starting at 0,0."""
    crop, bbox = unpack_mask_entry(mask_entry)
    if bbox is None:
        h, w = crop.shape[:2]
        bbox = [0, 0, w - 1, h - 1]
    return crop.astype(np.float32), bbox

def make_mask_entry(crop_bool, bbox, pack_bits=True):
    """Pack a boolean crop with its bbox, mirroring slim_batch_results format."""
    if pack_bits:
        packed = np.packbits(crop_bool.reshape(-1).astype(np.uint8))
        return {"packed": packed, "shape": crop_bool.shape, "bbox": bbox}
    return crop_bool.astype(bool)

def save_images_with_masks(img, results, output_path, mode='color', alpha=0.5, blur_strength=51):
    """Save image with mask overlays - either colored or blurred
    
    Args:
        img: PIL Image
        results: Detection results with masks and boxes
        output_path: Path to save output
        mode: 'color' for colored overlay or 'blur' for blurring
        alpha: Transparency for colored masks (0-1)
        blur_strength: Kernel size for blur (must be odd number)
    """
    from sam3.visualization_utils import COLORS
    import cv2
    
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
    
    # Calculate positions and sort masks by position (left-to-right, top-to-bottom)
    mask_positions = []
    for i, (mask, box) in enumerate(zip(results['masks'], results['boxes'])):
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        # Use box center for sorting
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        # Sort primarily by y (top to bottom), then by x (left to right)
        mask_positions.append((y_center, x_center, i))
    
    # Sort by position
    mask_positions.sort()
    
    # Process each mask in sorted order
    for color_idx, (_, _, original_idx) in enumerate(mask_positions):
        mask_entry = results['masks'][original_idx]
        # Rebuild full-frame mask from packed/cropped format
        mask_bool = rebuild_full_mask(mask_entry, img_array.shape)
        
        if mode == 'blur':
            # Blur the masked region
            blurred = cv2.GaussianBlur(overlay, (blur_strength, blur_strength), 0)
            overlay[mask_bool] = blurred[mask_bool]
        else:  # mode == 'color'
            # Get color based on sorted position (consistent across frames)
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
    """Match masks across frames by IoU and position similarity
    
    Returns list of tracks where each track is a list of (frame_idx, mask_idx) pairs
    """
    if not results_list or len(results_list) == 0:
        return []
    
    tracks = []  # Each track: [(frame_idx, mask_idx), ...]
    
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
            if isinstance(curr_box, torch.Tensor):
                curr_box = curr_box.cpu().numpy()
            
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
                
                if isinstance(last_box, torch.Tensor):
                    last_box = last_box.cpu().numpy()
                
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
    """Fill gaps in tracks by interpolating masks
    
    Args:
        results_list: List of (frame_idx, results_dict, image_path, image)
        tracks: List of tracks from match_masks_across_frames
        max_gap: Maximum number of missing frames to fill
    """
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
                
                if isinstance(curr_box, torch.Tensor):
                    curr_box = curr_box.cpu().numpy()
                if isinstance(next_box, torch.Tensor):
                    next_box = next_box.cpu().numpy()

                # Convert mask entries to crops (float32) and bbox
                curr_crop, curr_bbox = mask_entry_to_crop(curr_mask_data)
                next_crop, next_bbox = mask_entry_to_crop(next_mask_data)
                pack_output = isinstance(curr_mask_data, dict) and "packed" in curr_mask_data
                
                # Interpolate for each missing frame
                for gap_idx in range(1, gap_size + 1):
                    missing_frame = curr_frame + gap_idx
                    alpha = gap_idx / (gap_size + 1)
                    
                    # Interpolate box
                    interp_box = curr_box * (1 - alpha) + next_box * alpha
                    
                    # Interpolate mask crops (resize to target box then blend)
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
                        # Update tuple with new results dict
                        results_list[missing_frame] = (
                            results_list[missing_frame][0],
                            missing_results,
                            results_list[missing_frame][2]
                        )
                    
                    # Convert existing tensors to lists if needed
                    if not isinstance(missing_results['boxes'], list):
                        missing_results['boxes'] = list(missing_results['boxes'])
                        missing_results['masks'] = list(missing_results['masks'])
                        missing_results['scores'] = list(missing_results['scores'])
                    
                    missing_results['boxes'].append(np.array(interp_box, dtype=np.float32))
                    missing_results['masks'].append(mask_entry)
                    missing_results['scores'].append(0.5)  # Interpolated score
                    
                    filled_count += 1
    
    return filled_count


def get_image_files(folder_path):
    """Get all image files from folder"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = set()  # Use set to avoid duplicates
    
    for ext in image_extensions:
        image_files.update(Path(folder_path).glob(f'*{ext}'))
        image_files.update(Path(folder_path).glob(f'*{ext.upper()}'))
    
    return sorted(image_files)


def process_batch(model, datapoints, image_ids, postprocessor, device="auto"):
    """Process a batch of images"""
    try:
        device = resolve_device(device)
        # Collate into batch
        batch = collate(datapoints, dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, device, non_blocking=True)
        
        # Forward pass with local autocast for bfloat16 (only affects this scope)
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(batch)
            elif device == "mps":
                with torch.autocast("mps", dtype=torch.float16):
                    output = model(batch)
            else:
                output = model(batch)
        
        # Post-process
        processed_results = postprocessor.process_results(output, batch.find_metadatas)
        
        # Clear GPU cache after processing
        if device in ("cuda", "mps"):
            del batch, output
            cleanup_device(device)
        
        return processed_results
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\nOUT OF MEMORY ERROR! Try reducing --batch_size to 1")
            if device == "cuda":
                torch.cuda.empty_cache()
        raise


def convert_results_to_unified_dict(all_results, tracks=None):
    """
    Convert SAM3 results list to unified per-frame dict:
    {frame_idx: [{'bbox': (x1,y1,x2,y2), 'parent_box': None, 'score': float, 'text': '', 'alterego': '',
                  'mask': np.bool_ or crop, 'source': 'sam3', 'to_show': True, 'track_id': int}, ...]}
    
    Args:
        all_results: List of (frame_idx, results_dict, img_path)
        tracks: Optional list of tracks from match_masks_across_frames
    """
    # Build track_id mapping: (frame_idx, mask_idx) -> track_id
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

            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()
            if isinstance(score, torch.Tensor):
                score = float(score.detach().cpu().item())
            else:
                score = float(score)

            x1, y1, x2, y2 = [int(round(v)) for v in box]

            # Keep packed format to save space
            if isinstance(mask, dict) and "packed" in mask:
                # Keep as packed dict
                mask_data = mask
            else:
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.detach().cpu().numpy()
                else:
                    mask_np = np.array(mask)
                mask_np = np.squeeze(mask_np).astype(bool)
                if mask_np.size == 0:
                    continue
                mask_data = mask_np

            # Get track_id if available
            track_id = track_map.get((frame_idx, i), None)

            unified.setdefault(frame_idx, []).append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "parent_box": None,
                    "score": score,
                    "text": "",
                    "alterego": "",
                    "mask": mask_data,  # Keep packed format
                    "source": "sam3",
                    "to_show": True,
                    "track_id": track_id
                }
            )
    return unified


def process_video_sam3(frames_folder, output_folder, text_prompt="profile image, profile picture",
                      batch_size=2, device='auto', mask_mode='color', blur_strength=51,
                      masks_propagation=True, max_gap=5, save_images=False):
    """
    Process frames from a single video with SAM3.
    
    Args:
        frames_folder (str or Path): Path to folder containing extracted frames
        output_folder (str or Path): Path to folder for saving results (should be same as frames parent)
        text_prompt (str): Text prompt to segment
        batch_size (int): Batch size for inference
        device (str): Device to use ('cuda' or 'cpu')
        mask_mode (str): Mask visualization mode ('color' or 'blur')
        blur_strength (int): Blur kernel size for blur mode (must be odd)
        masks_propagation (bool): Enable temporal mask propagation
        max_gap (int): Maximum frame gap to fill when propagating masks
        save_images (bool): Save images with masks overlaid
        
    Returns:
        dict: Unified SAM3 data with track_ids
    """
    frames_folder = Path(frames_folder)
    output_folder = Path(output_folder)
    
    print(f"\n{'='*60}")
    print(f"Processing SAM3 for: {frames_folder}")
    print(f"Output folder: {output_folder}")
    print(f"{'='*60}\n")
    
    # Validate blur strength
    if blur_strength % 2 == 0:
        blur_strength += 1
        print(f"Blur strength adjusted to {blur_strength} (must be odd)")
    
    # Resolve device
    device = resolve_device(device)
    if device == "cuda":
        print("Using CUDA for SAM3")
    elif device == "mps":
        print("Using Apple MPS for SAM3")
    else:
        print("Using CPU for SAM3")
    
    # Setup
    print("Loading model...")
    model = setup_model(device)
    transform = setup_transforms()
    postprocessor = setup_postprocessor()
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all images
    image_files = get_image_files(frames_folder)
    print(f"Found {len(image_files)} images in {frames_folder}")
    
    if len(image_files) == 0:
        print("No images found!")
        return None
    
    print(f"Processing with text prompt: '{text_prompt}'")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
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
        print(f"Mask mode: {mask_mode}")
        if mask_mode == 'blur':
            print(f"Blur strength: {blur_strength}")
        print("\nStarting processing...")
        
        query_id = 1
        for batch_start in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
            batch_files = image_files[batch_start:batch_start + batch_size]
            
            datapoints = []
            image_ids = []
            
            for img_path in batch_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    datapoint = create_datapoint(img, text_prompt, query_id)
                    image_ids.append(query_id)
                    query_id += 1
                    datapoint = transform(datapoint)
                    datapoints.append(datapoint)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
            
            if len(datapoints) == 0:
                continue
            
            try:
                results = process_batch(model, datapoints, image_ids, postprocessor, device)
                results = slim_batch_results(results, pad=10, pack_bits=True)
                for img_path, img_id in zip(batch_files, image_ids):
                    frame_idx = len(all_results)
                    all_results.append((frame_idx, results.get(img_id, {}), img_path))
            except Exception as e:
                print(f"Error processing batch: {e}")
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
    
    print(f"SAM3 processing complete for {frames_folder.parent.name}")
    
    # Cleanup model from GPU after processing
    print("Cleaning up SAM3 model...")
    try:
        del model
        del transform
        del postprocessor
        
        import gc
        gc.collect()
        
        cleanup_device(device)
        print("SAM3 model unloaded")
    except Exception as e:
        print(f"Warning: Model cleanup error: {e}")
    
    return unified


def process_videos_sam3_batch(video_folders, text_prompt="profile image, profile picture",
                              batch_size=4, device='auto', mask_mode='color', blur_strength=51,
                              masks_propagation=True, max_gap=5, save_images=False):
    """
    Process multiple video folders with SAM3 in batch.
    
    Args:
        video_folders (list): List of paths to video output folders (each should contain frames/ subdirectory)
        text_prompt (str): Text prompt to segment
        batch_size (int): Batch size for inference
        device (str): Device to use ('auto', 'cuda', 'mps', or 'cpu')
        mask_mode (str): Mask visualization mode ('color' or 'blur')
        blur_strength (int): Blur kernel size for blur mode (must be odd)
        masks_propagation (bool): Enable temporal mask propagation
        max_gap (int): Maximum frame gap to fill when propagating masks
        save_images (bool): Save images with masks overlaid
        
    Returns:
        dict: Dictionary mapping video names to their unified SAM3 data
    """
    results = {}
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING SAM3: {len(video_folders)} videos")
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
            unified = process_video_sam3(
                frames_folder=frames_folder,
                output_folder=video_folder,
                text_prompt=text_prompt,
                batch_size=batch_size,
                device=device,
                mask_mode=mask_mode,
                blur_strength=blur_strength,
                masks_propagation=masks_propagation,
                max_gap=max_gap,
                save_images=save_images
            )
            results[video_name] = unified
            print(f"✓ Successfully processed {video_name}")
        except Exception as e:
            print(f"✗ Error processing {video_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[video_name] = None
    
    print(f"\n{'='*60}")
    print(f"BATCH SAM3 PROCESSING COMPLETE")
    print(f"Successful: {sum(1 for v in results.values() if v is not None)}/{len(video_folders)}")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='SAM3 Batch Inference on Image Folder')
    parser.add_argument('--input_folder', type=str, required=True,
                      help='Path to folder containing images')
    parser.add_argument('--output_folder', type=str, required=True,
                      help='Path to folder for saving results')
    parser.add_argument('--text_prompt', type=str, default="profile image, profile picture",
                      help='Text prompt to segment (e.g., "person", "car")')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for inference (default: 1, increase with caution!)')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use: auto|cuda|mps|cpu (default: auto)')
    parser.add_argument('--mask_mode', type=str, default='color', choices=['color', 'blur'],
                      help='Mask visualization mode: "color" for colored overlay, "blur" for blurring')
    parser.add_argument('--blur_strength', type=int, default=51,
                      help='Blur kernel size for blur mode (must be odd, default: 51)')
    parser.add_argument('--no_masks_propagation', action='store_true',
                      help='Disable temporal mask propagation that fills missing detections')
    parser.add_argument('--max_gap', type=int, default=5,
                      help='Maximum frame gap to fill when propagating masks (default: 5)')
    parser.add_argument('--save_images', action='store_true',
                      help='Save images with masks overlaid (default: False)')
    
    args = parser.parse_args()
    
    # Validate blur strength
    if args.blur_strength % 2 == 0:
        args.blur_strength += 1  # Make it odd
        print(f"Blur strength adjusted to {args.blur_strength} (must be odd)")
    
    # Resolve device
    args.device = resolve_device(args.device)
    print(f"Resolved device: {args.device}")
    
    # Setup
    print("Loading model...")
    model = setup_model(args.device)
    transform = setup_transforms()
    postprocessor = setup_postprocessor()
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Get all images
    image_files = get_image_files(args.input_folder)
    print(f"Found {len(image_files)} images in {args.input_folder}")
    
    if len(image_files) == 0:
        print("No images found!")
        return
    
    print(f"Processing with text prompt: '{args.text_prompt}'")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Mask mode: {args.mask_mode}")
    if args.mask_mode == 'blur':
        print(f"Blur strength: {args.blur_strength}")
    propagate_masks = True
    if args.no_masks_propagation:
        propagate_masks = False
        print(f"Temporal propagation disabled")
    else:
        print(f"Temporal propagation enabled (max gap: {args.max_gap} frames)")
    
    pickle_path = os.path.join(args.output_folder, 'detected_masks.pkl')
    try:
        # Try loading existing results
        with open(pickle_path, 'rb') as f:
            all_results = pickle.load(f)
        print(f"Loaded existing results from {pickle_path}, skipping processing")
    except FileNotFoundError:
        # Store all results for temporal propagation
        all_results = []  # List of (frame_idx, results_dict, img_path, img)
        print(f"Mask mode: {args.mask_mode}")
        if args.mask_mode == 'blur':
            print(f"Blur strength: {args.blur_strength}")
        print("\nStarting processing...")
        
        # Process in batches
        query_id = 1
        for batch_start in tqdm(range(0, len(image_files), args.batch_size), desc="Processing batches"):
            batch_files = image_files[batch_start:batch_start + args.batch_size]
            
            # Create datapoints for this batch
            datapoints = []
            image_ids = []
            
            for img_path in batch_files:
                try:
                    # Load image
                    img = Image.open(img_path).convert('RGB')
                    
                    # Create datapoint
                    datapoint = create_datapoint(img, args.text_prompt, query_id)
                    image_ids.append(query_id)
                    query_id += 1
                    
                    # Transform
                    datapoint = transform(datapoint)
                    datapoints.append(datapoint)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
            
            if len(datapoints) == 0:
                continue
            
            # Process batch
            try:
                results = process_batch(model, datapoints, image_ids, postprocessor, args.device)
                results = slim_batch_results(results, pad=10, pack_bits=True)
                for img_path, img_id in zip(batch_files, image_ids):
                    frame_idx = len(all_results)
                    all_results.append((frame_idx, results.get(img_id, {}), img_path))
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

        # Save detected masks to pickle (after propagation so it includes filled masks)
        print(f"\nSaving masks to pickle: {pickle_path}")
        with open(pickle_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Saved {len(all_results)} frames with masks")
    
    pickle_path_propagated = os.path.join(args.output_folder, 'detected_masks_propagated.pkl')
    tracks = None
    try:
        # Try loading existing results
        with open(pickle_path_propagated, 'rb') as f:
            all_results = pickle.load(f)
        print(f"Loaded existing propagated results from {pickle_path_propagated}, skipping propagation")
        # Try loading tracks
        tracks_path = os.path.join(args.output_folder, 'mask_tracks.pkl')
        try:
            with open(tracks_path, 'rb') as f:
                tracks = pickle.load(f)
            print(f"Loaded {len(tracks)} mask tracks")
        except FileNotFoundError:
            print(f"No tracks file found, will regenerate")
            if len(all_results) > 1:
                tracks = match_masks_across_frames(all_results)
                print(f"Regenerated {len(tracks)} mask tracks")
    except FileNotFoundError:
        # Apply temporal mask propagation if enabled
        if propagate_masks and len(all_results) > 1:
            print(f"\nApplying temporal mask propagation...")
            tracks = match_masks_across_frames(all_results)
            print(f"Found {len(tracks)} mask tracks across {len(all_results)} frames")
            filled_count = propagate_missing_masks(all_results, tracks, args.max_gap)
            print(f"Filled {filled_count} missing mask(s)")
        
        # Save detected masks to pickle (after propagation so it includes filled masks)
        print(f"\nSaving masks to pickle: {pickle_path_propagated}")
        with open(pickle_path_propagated, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Saved {len(all_results)} frames with propagated masks")
        
        # Save tracks
        if tracks:
            tracks_path = os.path.join(args.output_folder, 'mask_tracks.pkl')
            with open(tracks_path, 'wb') as f:
                pickle.dump(tracks, f)
            print(f"Saved {len(tracks)} mask tracks to {tracks_path}")

    # Save unified dict format
    unified_path = os.path.join(args.output_folder, 'detected_masks_final.pkl')
    try:
        with open(unified_path, 'rb') as f:
            unified = pickle.load(f)
        print(f"Unified detections already exist at {unified_path}, skipping conversion")
    except FileNotFoundError:
        unified = convert_results_to_unified_dict(all_results, tracks)
        with open(unified_path, 'wb') as f:
            pickle.dump(unified, f)
        print(f"Saved unified detections to {unified_path} ({len(unified)} frames)")
        
    # Save all results (original or propagated)
    if args.save_images:
        print(f"\nSaving processed images...")
        for frame_idx, frame_results, img_path in tqdm(all_results, desc="Saving images"):
            output_path = os.path.join(args.output_folder, f"{img_path.stem}_result.png")
            # Load original image
            img = Image.open(img_path).convert('RGB')
            
            # Save with mask overlays (colored or blurred)
            saved = save_images_with_masks(
                img, 
                frame_results, 
                output_path, 
                mode=args.mask_mode,
                alpha=0.5,
                blur_strength=args.blur_strength
            )
            
            if not saved:
                # No detections, just save original image
                img.save(output_path)
    
    print(f"\nDone! Results saved to {args.output_folder}")


if __name__ == "__main__":
    main()
