"""
Scene Transition Detection Module

This module provides functionality to detect scene transitions in videos
using multi-metric analysis including edge detection and histogram comparison.
"""

import cv2
import numpy as np
from tqdm import tqdm
import pickle
import os
import utils


def detect_scene_transitions(video_path, threshold=0.25, min_transition_frames=2, 
                           edge_weight=0.3, histogram_weight=0.7, min_duration_seconds=0.2, 
                           save_to_file=True, output_dir=None):
    """
    Detect scene transitions using improved multi-metric analysis.
    
    Args:
        video_path (str): Path to the video file
        threshold (float): Threshold for detecting significant changes (0-1)
        min_transition_frames (int): Minimum consecutive frames to consider a transition
        edge_weight (float): Weight for edge-based detection
        histogram_weight (float): Weight for histogram-based detection
        min_duration_seconds (float): Minimum duration in seconds to consider a valid transition
        save_to_file (bool): Whether to save detected transitions to a text file
        output_dir (str): Directory to save the transition file (if None, saves in current directory)
        
    Returns:
        list: List of frame ranges where transitions occur [(start_frame, end_frame), ...]
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return [], None
    
    # Prepare previous frame data
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    prev_edges = cv2.Canny(prev_gray, 50, 150)
    prev_hist = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
    
    transitions = []
    high_change_frames = []
    change_scores = []
    frame_idx = 1
    
    print("Analyzing video for scene transitions...")
    
    with tqdm(total=total_frames-1, desc="Detecting transitions") as pbar:
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            # Prepare current frame data
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
            curr_edges = cv2.Canny(curr_gray, 50, 150)
            curr_hist = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
            
            # Calculate multiple change metrics
            
            # 1. Edge-based change detection
            edge_diff = cv2.absdiff(prev_edges, curr_edges)
            edge_change_ratio = np.count_nonzero(edge_diff) / edge_diff.size
            
            # 2. Histogram comparison (using correlation - lower means more different)
            hist_correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
            hist_change_ratio = 1 - hist_correlation  # Convert to change metric
            
            # 3. Combined change score
            combined_score = (edge_weight * edge_change_ratio + 
                            histogram_weight * hist_change_ratio)
            
            change_scores.append(combined_score)
            
            # Detect significant changes using adaptive threshold
            if combined_score > threshold:
                high_change_frames.append(frame_idx)
            else:
                # Check if we have a sequence of high-change frames
                if len(high_change_frames) >= min_transition_frames:
                    # Calculate duration in seconds
                    duration_frames = len(high_change_frames)
                    duration_seconds = duration_frames / fps
                    
                    # Only consider transitions longer than minimum duration
                    if duration_seconds >= min_duration_seconds:
                        # Convert to processed frame indices
                        start_original = high_change_frames[0]
                        end_original = high_change_frames[-1]
                        
                        # Add buffer frames - increased to cover full transition
                        start_frame = max(0, start_original - 6)
                        end_frame = min(total_frames - 1, end_original + 15)
                        
                        # Calculate average change score for this transition
                        transition_start_idx = max(0, len(change_scores) - len(high_change_frames) - 1)
                        avg_score = np.mean(change_scores[transition_start_idx:])
                        
                        transitions.append((start_frame, end_frame, avg_score))
                        
                        print(f"  Detected transition: original frames {start_original}-{end_original} "
                              f"({duration_seconds:.2f}s, score={avg_score:.3f}) -> "
                              f"expanded frames {start_frame}-{end_frame}")
                
                high_change_frames = []
            
            # Update previous frame data
            prev_gray = curr_gray
            prev_edges = curr_edges
            prev_hist = curr_hist
            frame_idx += 1
            pbar.update(1)
    
    # Handle case where video ends with high change
    if len(high_change_frames) >= min_transition_frames:
        duration_frames = len(high_change_frames)
        duration_seconds = duration_frames / fps
        
        if duration_seconds >= min_duration_seconds:
            start_original = high_change_frames[0]
            end_original = high_change_frames[-1]
            
            start_frame = max(0, start_original)
            end_frame = min(total_frames - 1, end_original)

            transition_start_idx = max(0, len(change_scores) - len(high_change_frames))
            avg_score = np.mean(change_scores[transition_start_idx:])
            
            transitions.append((start_frame, end_frame, avg_score))
            
            print(f"  Detected transition: original frames {start_original}-{end_original} "
                  f"({duration_seconds:.2f}s, score={avg_score:.3f}) -> "
                  f"expanded frames {start_frame}-{end_frame}")
    
    cap.release()
    
    # Just keep all detected transitions without score filtering
    filtered_transitions = [(start, end) for start, end, score in transitions]
    
    # Merge overlapping transitions
    merged_transitions = merge_overlapping_ranges(filtered_transitions)

    print(f"\nDetected {len(merged_transitions)} scene transitions (in expanded frame space):")
    for start, end in merged_transitions:
        duration_processed = (end - start) + 1
        duration_seconds = duration_processed / fps
        print(f"  Expanded frames {start}-{end} ({duration_processed} frames, {duration_seconds:.2f}s)")
    
    # Save transitions to text file for manual review/editing
    output_file = None
    if save_to_file:
        output_file = save_transitions_to_file(merged_transitions, video_path, output_dir)
    
    return merged_transitions, output_file

def merge_overlapping_ranges(ranges):
    """Merge overlapping or adjacent frame ranges."""
    if not ranges:
        return []
    
    # Sort by start frame
    sorted_ranges = sorted(ranges)
    merged = [sorted_ranges[0]]
    
    for current in sorted_ranges[1:]:
        last = merged[-1]
        
        # If current range overlaps or is adjacent to the last one, merge them
        if current[0] <= last[1] + 5:  # 5-frame buffer for merging
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged


def create_transition_boxes(transition_ranges, video_path):
    """
    Create full-screen blur boxes for transition periods.
    
    Args:
        transition_ranges (list): List of (start_frame, end_frame) tuples
        video_path (str): Path to video to get dimensions
        
    Returns:
        dict: Frame boxes for full-screen blur during transitions
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    transition_boxes = {}
    
    for start_frame, end_frame in transition_ranges:
        full_screen_box = {
            'bbox': [0, 0, width, height],
            'alterego': '',  # No text replacement needed
            'confidence': 1.0,
            'transition_blur': True  # Flag to indicate this is a transition blur
        }
        
        for frame_idx in range(start_frame, end_frame + 1):
            transition_boxes[frame_idx] = [full_screen_box]
    
    return transition_boxes


def merge_transition_and_text_boxes(text_boxes, transition_boxes):
    """
    Merge transition boxes with text boxes, prioritizing transitions.
    
    Args:
        text_boxes (dict): Regular text detection boxes
        transition_boxes (dict): Full-screen transition boxes
        
    Returns:
        dict: Combined boxes with transitions taking priority
    """
    merged_boxes = text_boxes.copy()
    
    for frame_idx, boxes in transition_boxes.items():
        if frame_idx in merged_boxes:
            # During transitions, only use full-screen blur
            merged_boxes[frame_idx] = boxes
        else:
            merged_boxes[frame_idx] = boxes
    
    return merged_boxes


def save_transitions(transition_ranges, output_path):
    """Save transition ranges to pickle file."""
    with open(output_path, 'wb') as f:
        pickle.dump(transition_ranges, f)
    print(f"Saved transitions to {output_path}")


def load_transitions(input_path):
    """Load transition ranges from pickle file."""
    with open(input_path, 'rb') as f:
        transition_ranges = pickle.load(f)
    print(f"Loaded transitions from {input_path}")
    return transition_ranges


def save_transitions_to_file(transitions, video_path, output_dir=None):
    """
    Save detected transition ranges to a text file for manual review/editing.
    
    Args:
        transitions (list): List of (start_frame, end_frame) tuples
        video_path (str): Path to the video file (used to generate output filename)
        output_dir (str): Directory to save the file (should be the video's output directory)
    """
    # Always save as 'transitions.txt' in the output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "transitions.txt")
    else:
        # If no output_dir provided, save in current directory with video name prefix
        video_name = os.path.splitext(os.path.basename(video_path))[0].lower()
        output_file = f"{video_name}_transitions.txt"
    
    with open(output_file, 'w') as f:
        f.write("# Scene Transition Ranges\n")
        f.write("# Format: start_frame,end_frame\n")
        f.write("# You can manually edit this file to add missing transitions or remove false positives\n")
        f.write("# Lines starting with # are comments and will be ignored\n\n")
        
        for start, end in transitions:
            f.write(f"{start},{end}\n")
    
    print(f"Transition ranges saved to: {output_file}")
    print("You can manually edit this file to correct transitions before running OCR processing.")
    return output_file

def load_transitions_from_file(video_path, output_dir=None):
    """
    Load transition ranges from a text file.
    
    Args:
        video_path (str): Path to the video file (used to find corresponding transition file)
        output_dir (str): Directory where the transition file might be saved
        
    Returns:
        list: List of (start_frame, end_frame) tuples, or empty list if file doesn't exist
    """
    # Generate filename based on video filename
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    filename = f"{video_name}_transitions.txt"
    
    # Try output directory first, then current directory
    possible_paths = []
    if output_dir:
        possible_paths.append(os.path.join(output_dir, filename))
    possible_paths.append(filename)
    
    transition_file = None
    for path in possible_paths:
        if os.path.exists(path):
            transition_file = path
            break
    
    if not transition_file:
        print(f"No transition file found. Searched: {possible_paths}")
        return None, None
    
    transitions = []
    with open(transition_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            try:
                parts = line.split(',')
                if len(parts) != 2:
                    print(f"Warning: Invalid format on line {line_num}: {line}")
                    continue
                
                start_frame = int(parts[0].strip())
                end_frame = int(parts[1].strip())
                
                if start_frame >= end_frame:
                    print(f"Warning: Invalid range on line {line_num}: start >= end ({start_frame},{end_frame})")
                    continue
                
                transitions.append((start_frame, end_frame))
                
            except ValueError as e:
                print(f"Warning: Could not parse line {line_num}: {line} - {e}")
                continue
    
    print(f"Loaded {len(transitions)} transition ranges from {transition_file}")
    return transitions, transition_file


def is_frame_in_transitions(frame_idx, transitions):
    """
    Check if a frame is within any transition range.
    
    Args:
        frame_idx (int): Frame index to check
        transitions (list): List of (start_frame, end_frame) tuples
        
    Returns:
        bool: True if frame is within any transition range
    """
    for start, end in transitions:
        if start <= frame_idx <= end:
            return True
    return False


def main():
    """Example usage of the transition detection module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect scene transitions in video")
    parser.add_argument("--video-path", required=True, help="Path to input video file (overrides config)")
    parser.add_argument("--output", "-o", help="Output txt file path")
    parser.add_argument("--threshold", type=float, help="Detection threshold (overrides config)")
    parser.add_argument("--min-frames", type=int, help="Minimum transition frames (overrides config)")

    args = parser.parse_args()
    
    # Use command line video path and parameters
    transitions = detect_scene_transitions(
        args.video_path,
        threshold=args.threshold or 0.25,
        min_transition_frames=args.min_frames or 2
    )
    
    # Save results if output path provided
    if args.output:
        save_transitions(transitions, args.output)
    
    return transitions


if __name__ == "__main__":
    main()