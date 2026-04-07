"""
SAM3 pipeline orchestration.

Wires together:
  engine.py      — SAM3 model setup + per-frame inference
  mask_ops.py    — mask packing, slimming, overlap merging
  tracking.py    — IoU-based temporal tracking + gap propagation
  circularize.py — circle fitting + per-track stabilization
  format.py      — unified output format

Public API:
  process_video_sam3(...)
  process_videos_sam3_batch(...)
"""

import os
import pickle
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from .engine import setup_predictor, process_image, get_image_files
from .mask_ops import slim_results, merge_overlapping_in_results
from .tracking import match_masks_across_frames, propagate_missing_masks
from .circularize import circularize_results
from .format import convert_to_unified_dict

from src.utils.mask_utils import rebuild_full_mask
from utils import resolve_device, cleanup_device

import copy


def _compute_frame_similarity(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Fast similarity score in [0, 1].  1.0 = identical frames."""
    small_a = cv2.resize(frame_a, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(frame_b, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    gray_a = cv2.cvtColor(small_a, cv2.COLOR_BGR2GRAY) if small_a.ndim == 3 else small_a
    gray_b = cv2.cvtColor(small_b, cv2.COLOR_BGR2GRAY) if small_b.ndim == 3 else small_b
    diff = cv2.absdiff(gray_a, gray_b)
    return 1.0 - float(np.mean(diff)) / 255.0


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _save_images_with_masks(img, results, output_path, mode='color', alpha=0.5, blur_strength=51):
    """Save image with mask overlays — either colored or blurred."""
    COLORS = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
    ]

    if not results or 'masks' not in results or len(results['masks']) == 0:
        return False

    if isinstance(img, Image.Image):
        img_array = np.array(img)
    else:
        img_array = img.copy()

    if img_array.dtype == np.float32 or img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    img_array = img_array[..., :3]

    overlay = img_array.copy()

    mask_positions = []
    for i, (mask, box) in enumerate(zip(results['masks'], results['boxes'])):
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        mask_positions.append((y_center, x_center, i))

    mask_positions.sort()

    for color_idx, (_, _, original_idx) in enumerate(mask_positions):
        mask_entry = results['masks'][original_idx]
        mask_bool = rebuild_full_mask(mask_entry, img_array.shape)

        if mode == 'blur':
            blurred = cv2.GaussianBlur(overlay, (blur_strength, blur_strength), 0)
            overlay[mask_bool] = blurred[mask_bool]
        else:
            color = COLORS[color_idx % len(COLORS)]
            color255 = (color * 255).astype(np.uint8)
            for c in range(3):
                overlay[..., c][mask_bool] = (
                    alpha * color255[c] + (1 - alpha) * overlay[..., c][mask_bool]
                ).astype(np.uint8)

    Image.fromarray(overlay).save(output_path)
    return True


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def _cleanup_intermediate_pickles(output_folder):
    """Remove transient SAM3 cache files after final outputs are generated."""
    output_folder = Path(output_folder)
    removable = [
        output_folder / "detected_masks_propagated.pkl",
        output_folder / "mask_tracks.pkl",
        output_folder / "detected_masks_circular.pkl",
    ]
    for path in removable:
        try:
            if path.exists():
                path.unlink()
                print(f"Removed intermediate file: {path.name}")
        except Exception as e:
            print(f"Warning: Could not remove intermediate file {path}: {e}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_video_sam3(frames_folder, output_folder,
                       text_prompt="profile image, profile picture",
                       device='auto', model_path="sam3.pt", conf=0.25,
                       mask_mode='color', blur_strength=51,
                       masks_propagation=True, max_gap=5,
                       save_images=False, pad=10):
    """
    Process frames from a single video with SAM3.

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
    print(f"Processing SAM3 for: {frames_folder}")
    print(f"Output folder: {output_folder}")
    print(f"{'='*60}\n")

    if blur_strength % 2 == 0:
        blur_strength += 1
        print(f"Blur strength adjusted to {blur_strength} (must be odd)")

    # Setup
    print("Loading SAM3 model...")
    predictor, device = setup_predictor(device, model_path, conf)

    os.makedirs(output_folder, exist_ok=True)

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

    # Stage 1: Per-frame inference
    pickle_path = output_folder / 'detected_masks.pkl'
    try:
        with open(pickle_path, 'rb') as f:
            all_results = pickle.load(f)
        print(f"Loaded existing results from {pickle_path}, skipping processing")
    except FileNotFoundError:
        all_results = []
        print("\nStarting processing...")

        change_threshold = 0.999
        max_consecutive_skips = 30
        prev_frame = None
        prev_results = None
        skipped = 0
        processed = 0
        consecutive_skips = 0

        for frame_idx, img_path in enumerate(tqdm(image_files, desc="Processing frames")):
            try:
                img = Image.open(img_path).convert('RGB')
                img_shape = (img.height, img.width)
                frame_array = np.array(img)

                if prev_frame is not None:
                    sim = _compute_frame_similarity(prev_frame, frame_array)
                    if sim >= change_threshold and consecutive_skips < max_consecutive_skips:
                        all_results.append((frame_idx, copy.deepcopy(prev_results), img_path))
                        skipped += 1
                        consecutive_skips += 1
                        prev_frame = frame_array
                        continue
                    else:
                        consecutive_skips = 0

                results = process_image(predictor, img_path, text_prompt, frame_idx)
                results = slim_results(results, img_shape, pad=pad, pack_bits=True)

                all_results.append((frame_idx, results, img_path))
                prev_frame = frame_array
                prev_results = results
                processed += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                all_results.append((frame_idx, {'boxes': [], 'masks': [], 'scores': []}, img_path))
                continue

        total = skipped + processed
        if total > 0:
            print(f"\nFrame skip summary: {processed} processed, {skipped} skipped"
                  f" (~{total / max(1, processed):.1f}x speedup)")

        print(f"\nSaving masks to pickle: {pickle_path}")
        with open(pickle_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Saved {len(all_results)} frames with masks")

    # Stage 2: Merge overlapping masks
    all_results, merged_overlap_count = merge_overlapping_in_results(all_results)
    if merged_overlap_count > 0:
        print(f"Merged {merged_overlap_count} overlapping SAM3 mask(s)")

    # Stage 3: Temporal tracking + propagation
    pickle_path_propagated = output_folder / 'detected_masks_propagated.pkl'
    tracks = None
    try:
        with open(pickle_path_propagated, 'rb') as f:
            all_results = pickle.load(f)
        print(f"Loaded existing propagated results from {pickle_path_propagated}")

        all_results, merged_overlap_count = merge_overlapping_in_results(all_results)
        if merged_overlap_count > 0:
            print(f"Merged {merged_overlap_count} overlapping SAM3 mask(s) in propagated cache")

        tracks_path = output_folder / 'mask_tracks.pkl'
        try:
            with open(tracks_path, 'rb') as f:
                tracks = pickle.load(f)
            print(f"Loaded {len(tracks)} mask tracks")
        except FileNotFoundError:
            if len(all_results) > 1:
                tracks = match_masks_across_frames(all_results)
                print(f"Regenerated {len(tracks)} mask tracks")

        if merged_overlap_count > 0 and len(all_results) > 1:
            tracks = match_masks_across_frames(all_results)
            print(f"Regenerated {len(tracks)} mask tracks after overlap merge")
    except FileNotFoundError:
        if len(all_results) > 1:
            tracks = match_masks_across_frames(all_results)
            print(f"Found {len(tracks)} mask tracks across {len(all_results)} frames")

        if masks_propagation and tracks and len(all_results) > 1:
            print(f"\nApplying temporal mask propagation...")
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

    # Stage 4: Unified format
    unified_path = output_folder / 'sam3.pkl'
    try:
        with open(unified_path, 'rb') as f:
            unified = pickle.load(f)
        print(f"Unified detections already exist at {unified_path}, skipping conversion")
    except FileNotFoundError:
        unified = convert_to_unified_dict(all_results, tracks)
        with open(unified_path, 'wb') as f:
            pickle.dump(unified, f)
        print(f"Saved unified detections to {unified_path} ({len(unified)} frames)")

    # Stage 5: Circularization
    circular_pickle_path = output_folder / 'detected_masks_circular.pkl'
    circular_unified_path = output_folder / 'sam3_circular.pkl'
    try:
        with open(circular_unified_path, 'rb') as f:
            _ = pickle.load(f)
        print(f"Circular masks already exist at {circular_unified_path}, skipping")
    except FileNotFoundError:
        circular_results = circularize_results(all_results, tracks=tracks)
        with open(circular_pickle_path, 'wb') as f:
            pickle.dump(circular_results, f)
        circular_unified = convert_to_unified_dict(circular_results, tracks)
        with open(circular_unified_path, 'wb') as f:
            pickle.dump(circular_unified, f)
        print(f"Saved circular masks to {circular_unified_path} ({len(circular_unified)} frames)")

    # Optional: save visualization images
    if save_images:
        print(f"\nSaving processed images...")
        for frame_idx, frame_results, img_path in tqdm(all_results, desc="Saving images"):
            output_path = output_folder / f"{img_path.stem}_result.png"
            img = Image.open(img_path).convert('RGB')
            saved = _save_images_with_masks(
                img, frame_results, output_path,
                mode=mask_mode, alpha=0.5, blur_strength=blur_strength
            )
            if not saved:
                img.save(output_path)

    # Cleanup
    _cleanup_intermediate_pickles(output_folder)

    print(f"SAM3 processing complete for {frames_folder.parent.name}")

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


def process_videos_sam3_batch(video_folders, text_prompt="profile image, profile picture",
                              device='auto', model_path="sam3.pt", conf=0.25,
                              mask_mode='color', blur_strength=51,
                              masks_propagation=True, max_gap=5,
                              save_images=False, pad=10):
    """
    Process multiple video folders with SAM3 in batch.

    Returns dict mapping video names to their unified SAM3 data.
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
    print(f"BATCH SAM3 PROCESSING COMPLETE")
    print(f"Successful: {sum(1 for v in results.values() if v is not None)}/{len(video_folders)}")
    print(f"{'='*60}\n")

    return results
