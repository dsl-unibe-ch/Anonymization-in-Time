"""
Pipeline script to process videos through the complete workflow:
1. Extract frames from video
2. Run OCR detection
3. Run SAM3 mask detection
4. Detect scene transitions

Each video gets its own folder with:
- frames/ (extracted frames)
- ocr.pkl (OCR detections with track_ids)
- sam3.pkl (SAM3 masks with track_ids)
- transitions.txt (scene transition ranges)
- boxes_<ocr_engine>.pkl (intermediate OCR boxes, backend-specific cache)
- detected_masks*.pkl (intermediate SAM3 results)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
from pathlib import Path
import json
import gc
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from utils import extract_video_frames
from utils import export_ocr_text_timeline
from ocr import process_video_ocr, process_videos_batch
from batch_inference_sam3 import process_video_sam3, process_videos_sam3_batch
from transition_detection import detect_scene_transitions
from utils import resolve_device, cleanup_device


def cleanup_gpu_memory(device: str | None = None):
    """
    Clean up accelerator memory between videos (CUDA or MPS).
    """
    if not TORCH_AVAILABLE:
        return
    
    try:
        print("Starting accelerator cleanup...")
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        if device:
            cleanup_device(device)
        else:
            cleanup_device("cuda")
            cleanup_device("mps")
        
        gc.collect()
        time.sleep(1.0)
        print("✓ Accelerator cleanup complete")
    except Exception as e:
        print(f"Warning: GPU cleanup failed: {e}")


def reset_cuda_on_error(device: str | None = None):
    """
    Attempt to reset accelerator state after an error.
    """
    if not TORCH_AVAILABLE:
        return
    
    try:
        if device:
            cleanup_device(device)
        else:
            cleanup_device("cuda")
            cleanup_device("mps")
        print("✓ Accelerator reset attempt completed - consider restarting if errors persist")
    except Exception as e:
        print(f"Warning: GPU reset failed: {e}")


def process_single_video(video_path, output_base_dir, dict_path, 
                        ocr_languages=["en", "de"], ocr_workers=4,
                        sam3_prompt="profile image, profile picture", sam3_batch_size=2,
                        sam3_device='auto', frame_step=1, ocr_engine="easyocr",
                        extract_frames=True, run_ocr=True, run_sam3=True, run_transitions=True):
    """
    Process a single video through the complete pipeline.
    
    Args:
        video_path (str or Path): Path to the video file
        output_base_dir (str or Path): Base directory for all outputs
        dict_path (str or Path): Path to JSON file with names to detect
        ocr_languages (list): OCR language hints
        ocr_workers (int): Number of parallel workers for OCR
        sam3_prompt (str): Text prompt for SAM3 segmentation
        sam3_batch_size (int): Batch size for SAM3 inference
        sam3_device (str): Device for SAM3 ('auto', 'cuda', 'mps', or 'cpu')
        frame_step (int): Step between frames to extract
        ocr_engine (str): OCR backend ('easyocr' or 'paddleocr')
        extract_frames (bool): Whether to extract frames from video
        run_ocr (bool): Whether to run OCR processing
        run_sam3 (bool): Whether to run SAM3 processing
        run_transitions (bool): Whether to run transition detection
        
    Returns:
        dict: Results containing paths to all generated files
    """
    video_path = Path(video_path)
    output_base_dir = Path(output_base_dir)
    
    # Resolve accelerator once for the full video
    sam3_device_resolved = resolve_device(sam3_device)
    
    # Create video-specific output directory
    video_name = video_path.stem
    video_output_dir = output_base_dir / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'video_name': video_name,
        'output_dir': str(video_output_dir),
        'frames_dir': None,
        'ocr_raw_boxes_pkl': None,
        'ocr_pkl': None,
        'ocr_text_timeline': None,
        'sam3_raw_masks_pkl': None,
        'sam3_pkl': None,
        'sam3_circular_pkl': None,
        'transitions_txt': None
    }
    
    print(f"\n{'='*70}")
    print(f"PROCESSING VIDEO: {video_name}")
    print(f"Output directory: {video_output_dir}")
    print(f"{'='*70}\n")
    print(f"SAM3 device: {sam3_device_resolved}")
    
    # Step 1: Extract frames
    frames_dir = video_output_dir / "frames"
    if extract_frames:
        print(f"\n[1/4] Extracting frames...")
        frames_dir = extract_video_frames(video_path, output_dir=frames_dir, frame_step=frame_step)
        results['frames_dir'] = str(frames_dir)
        print(f"✓ Frames extracted to {frames_dir}")
    else:
        if frames_dir.exists():
            results['frames_dir'] = str(frames_dir)
            print(f"[1/4] Using existing frames in {frames_dir}")
        else:
            print(f"[1/4] Frames directory not found, will extract frames")
            frames_dir = extract_video_frames(video_path, output_dir=frames_dir, frame_step=frame_step)
            results['frames_dir'] = str(frames_dir)
    
    # Step 2: Run OCR
    if run_ocr:
        print(f"\n[2/4] Running OCR detection ({ocr_engine})...")
        try:
            process_video_ocr(
                video_path=video_path,
                output_dir=video_output_dir,
                dict_path=dict_path,
                languages=ocr_languages,
                num_workers=ocr_workers,
                extract_frames=False,  # Already extracted
                frame_step=frame_step,
                ocr_engine=ocr_engine
            )
            raw_ocr_boxes = video_output_dir / f"boxes_{ocr_engine}.pkl"
            if raw_ocr_boxes.exists():
                results['ocr_raw_boxes_pkl'] = str(raw_ocr_boxes)
            elif ocr_engine == "easyocr":
                legacy_ocr_boxes = video_output_dir / "boxes.pkl"
                if legacy_ocr_boxes.exists():
                    results['ocr_raw_boxes_pkl'] = str(legacy_ocr_boxes)
            ocr_pkl = video_output_dir / "ocr.pkl"
            results['ocr_pkl'] = str(ocr_pkl)
            # Export collapsed text timeline right after OCR
            try:
                timeline_path = video_output_dir / "ocr_text_timeline.txt"
                export_ocr_text_timeline(
                    ocr_pkl,
                    timeline_path,
                    video_path=video_path,
                )
                results['ocr_text_timeline'] = str(timeline_path)
                print(f"✓ OCR text timeline saved: {timeline_path}")
            except Exception as e:
                print(f"Warning: Failed to export OCR text timeline: {e}")
            print(f"✓ OCR processing complete: {ocr_pkl}")
        except Exception as e:
            print(f"✗ Error in OCR processing: {str(e)}")
            import traceback
            traceback.print_exc()
            if 'CUDA' in str(e) or 'cuda' in str(e):
                reset_cuda_on_error(sam3_device_resolved)
    else:
        print(f"[2/4] Skipping OCR processing")
    
    # Step 3: Run SAM3
    if run_sam3:
        print(f"\n[3/4] Running SAM3 mask detection...")
        try:
            process_video_sam3(
                frames_folder=frames_dir,
                output_folder=video_output_dir,
                text_prompt=sam3_prompt,
                device=sam3_device_resolved,
                mask_mode='color',
                blur_strength=51,
                masks_propagation=True,
                max_gap=5,
                save_images=False
            )
            sam3_raw_masks = video_output_dir / "detected_masks.pkl"
            if sam3_raw_masks.exists():
                results['sam3_raw_masks_pkl'] = str(sam3_raw_masks)
            sam3_pkl = video_output_dir / "sam3.pkl"
            results['sam3_pkl'] = str(sam3_pkl)
            sam3_circular_pkl = video_output_dir / "sam3_circular.pkl"
            if sam3_circular_pkl.exists():
                results['sam3_circular_pkl'] = str(sam3_circular_pkl)
            print(f"✓ SAM3 processing complete: {sam3_pkl}")
        except Exception as e:
            print(f"✗ Error in SAM3 processing: {str(e)}")
            import traceback
            traceback.print_exc()
            if 'CUDA' in str(e) or 'cuda' in str(e):
                reset_cuda_on_error(sam3_device_resolved)
    else:
        print(f"[3/4] Skipping SAM3 processing")
    
    # Step 4: Detect transitions
    if run_transitions:
        print(f"\n[4/4] Detecting scene transitions...")
        try:
            transitions, transitions_file = detect_scene_transitions(
                video_path=str(video_path),
                threshold=0.25,
                min_transition_frames=2,
                edge_weight=0.3,
                histogram_weight=0.7,
                min_duration_seconds=0.2,
                save_to_file=True,
                output_dir=str(video_output_dir)
            )
            results['transitions_txt'] = transitions_file
            print(f"✓ Transition detection complete: {transitions_file}")
        except Exception as e:
            print(f"✗ Error in transition detection: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[4/4] Skipping transition detection")
    
    print(f"\n{'='*70}")
    print(f"COMPLETED: {video_name}")
    print(f"{'='*70}\n")
    
    return results


def process_multiple_videos(video_paths, output_base_dir, dict_path,
                           ocr_languages=["en", "de"], ocr_workers=4,
                           sam3_prompt="profile image, profile picture", sam3_batch_size=2,
                           sam3_device='auto', frame_step=1, ocr_engine="easyocr",
                           extract_frames=True, run_ocr=True, run_sam3=True, run_transitions=True):
    """
    Process multiple videos through the complete pipeline.
    
    Args:
        video_paths (list): List of paths to video files
        output_base_dir (str or Path): Base directory for all outputs
        dict_path (str or Path): Path to JSON file with names to detect
        ocr_languages (list): OCR language hints
        ocr_workers (int): Number of parallel workers for OCR
        sam3_prompt (str): Text prompt for SAM3 segmentation
        sam3_batch_size (int): Batch size for SAM3 inference
        sam3_device (str): Device for SAM3 ('auto', 'cuda', 'mps', or 'cpu')
        frame_step (int): Step between frames to extract
        ocr_engine (str): OCR backend ('easyocr' or 'paddleocr')
        extract_frames (bool): Whether to extract frames from videos
        run_ocr (bool): Whether to run OCR processing
        run_sam3 (bool): Whether to run SAM3 processing
        run_transitions (bool): Whether to run transition detection
        
    Returns:
        list: List of result dictionaries for each video
    """
    all_results = []
    sam3_device_resolved = resolve_device(sam3_device)
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING: {len(video_paths)} videos")
    print(f"Output base directory: {output_base_dir}")
    print(f"Device: {sam3_device_resolved}")
    print(f"{'='*70}\n")
    
    for idx, video_path in enumerate(video_paths):
        # Clean up GPU memory before processing each video (especially important after first video)
        if idx > 0:
            print(f"\n{'='*70}")
            print(f"Cleaning up accelerator before video {idx+1}/{len(video_paths)}...")
            print(f"{'='*70}")
            cleanup_gpu_memory(sam3_device_resolved)
        
        try:
            result = process_single_video(
                video_path=video_path,
                output_base_dir=output_base_dir,
                dict_path=dict_path,
                ocr_languages=ocr_languages,
                ocr_workers=ocr_workers,
                ocr_engine=ocr_engine,
                sam3_prompt=sam3_prompt,
                sam3_batch_size=sam3_batch_size,
                sam3_device=sam3_device_resolved,
                frame_step=frame_step,
                extract_frames=extract_frames,
                run_ocr=run_ocr,
                run_sam3=run_sam3,
                run_transitions=run_transitions
            )
            all_results.append(result)
        except Exception as e:
            print(f"✗ Fatal error processing {video_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'video_name': Path(video_path).stem,
                'error': str(e)
            })
            if 'CUDA' in str(e) or 'cuda' in str(e):
                reset_cuda_on_error(sam3_device_resolved)
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETE")
    successful = sum(1 for r in all_results if 'error' not in r)
    print(f"Successful: {successful}/{len(video_paths)}")
    print(f"{'='*70}\n")
    
    # Final accelerator cleanup
    print("Final accelerator cleanup...")
    cleanup_gpu_memory(sam3_device_resolved)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Process videos through OCR, SAM3, and transition detection pipeline'
    )
    
    # Input/output arguments
    parser.add_argument('--video', type=str, help='Path to a single video file')
    parser.add_argument('--video_folder', type=str, help='Path to folder containing multiple videos')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Base output directory (each video gets a subfolder)')
    parser.add_argument('--dict_path', type=str, required=True,
                       help='Path to JSON file with names to detect for OCR')
    
    # Processing options
    parser.add_argument('--frame_step', type=int, default=1,
                       help='Step between frames to extract (default: 1 = all frames)')
    parser.add_argument('--skip_frames', action='store_true',
                       help='Skip frame extraction (use existing frames)')
    parser.add_argument('--skip_ocr', action='store_true',
                       help='Skip OCR processing')
    parser.add_argument('--skip_sam3', action='store_true',
                       help='Skip SAM3 processing')
    parser.add_argument('--skip_transitions', action='store_true',
                       help='Skip transition detection')
    
    # OCR options
    parser.add_argument('--ocr_languages', type=str, nargs='+', default=["en", "de"],
                       help='Languages for OCR (default: en de)')
    parser.add_argument('--ocr_workers', type=int, default=4,
                       help='Number of parallel workers for OCR (default: 4)')
    parser.add_argument('--ocr_engine', type=str, default='easyocr', choices=['easyocr', 'paddleocr'],
                       help='OCR backend to use (default: easyocr)')
    
    # SAM3 options
    parser.add_argument('--sam3_prompt', type=str, default="profile image, profile picture",
                       help='Text prompt for SAM3 segmentation')
    parser.add_argument('--sam3_batch_size', type=int, default=2,
                       help='Batch size for SAM3 inference (default: 2)')
    parser.add_argument('--sam3_device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device for SAM3 (default: auto)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.video and not args.video_folder:
        parser.error("Must provide either --video or --video_folder")
    
    if args.video and args.video_folder:
        parser.error("Cannot provide both --video and --video_folder")
    
    # Collect video paths
    video_paths = []
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            return
        video_paths = [video_path]
    else:
        video_folder = Path(args.video_folder)
        if not video_folder.exists():
            print(f"Error: Video folder not found: {video_folder}")
            return
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
        for ext in video_extensions:
            video_paths.extend(video_folder.glob(f'*{ext}'))
            video_paths.extend(video_folder.glob(f'*{ext.upper()}'))
        
        if not video_paths:
            print(f"Error: No video files found in {video_folder}")
            return
    
    print(f"Found {len(video_paths)} video(s) to process")
    
    # Process videos
    if len(video_paths) == 1:
        process_single_video(
            video_path=video_paths[0],
            output_base_dir=args.output_dir,
            dict_path=args.dict_path,
            ocr_languages=args.ocr_languages,
            ocr_workers=args.ocr_workers,
            ocr_engine=args.ocr_engine,
            sam3_prompt=args.sam3_prompt,
            sam3_batch_size=args.sam3_batch_size,
            sam3_device=args.sam3_device,
            frame_step=args.frame_step,
            extract_frames=not args.skip_frames,
            run_ocr=not args.skip_ocr,
            run_sam3=not args.skip_sam3,
            run_transitions=not args.skip_transitions
        )
    else:
        process_multiple_videos(
            video_paths=video_paths,
            output_base_dir=args.output_dir,
            dict_path=args.dict_path,
            ocr_languages=args.ocr_languages,
            ocr_workers=args.ocr_workers,
            ocr_engine=args.ocr_engine,
            sam3_prompt=args.sam3_prompt,
            sam3_batch_size=args.sam3_batch_size,
            sam3_device=args.sam3_device,
            frame_step=args.frame_step,
            extract_frames=not args.skip_frames,
            run_ocr=not args.skip_ocr,
            run_sam3=not args.skip_sam3,
            run_transitions=not args.skip_transitions
        )


if __name__ == "__main__":
    main()
