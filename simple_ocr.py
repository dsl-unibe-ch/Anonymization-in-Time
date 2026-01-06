import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ocr import process_frames_parallel
import pickle
import json
import numpy as np
import unicodedata
from tqdm import tqdm
from difflib import SequenceMatcher

boxes_pkl_path = "tests\\ocr testvideo laet_boxes.pkl"
frames_dir = "tests\\ocr testvideo laet_frames"
dict_path = "laet.json"
output_pkl_path = "tests\\ocr testvideo laet_final_frame_boxes.pkl"

languages = ["en", "de"]
num_workers = 4


def merge_split_names(frame_boxes, names_dict, x_threshold=50):
    """
    Merge horizontally adjacent boxes if their combined text matches a name in the dictionary.
    This helps with composite names that EasyOCR might split into multiple boxes.
    
    Args:
        frame_boxes (dict): Dictionary of frame indices to list of bounding boxes
        names_dict (dict): Dictionary of names to match against
        x_threshold (int): Maximum horizontal distance between boxes to consider merging
        
    Returns:
        dict: Updated frame_boxes with merged entries
    """
    merged_frame_boxes = {}
    
    # Pre-process names for faster matching
    normalized_names = set()
    for name in names_dict.keys():
        # Add various normalized forms
        normalized_names.add(name.lower().strip())
        normalized_names.add(unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode("utf-8").lower().strip())

    for frame_idx, boxes in frame_boxes.items():
        if not boxes:
            merged_frame_boxes[frame_idx] = []
            continue
            
        # Sort boxes by y (top to bottom) then x (left to right)
        # Using center y for better line alignment handling
        sorted_boxes = sorted(boxes, key=lambda b: ((b['bbox'][1] + b['bbox'][3])/2, b['bbox'][0]))
        
        merged_boxes = []
        used_indices = set()
        
        i = 0
        while i < len(sorted_boxes):
            if i in used_indices:
                i += 1
                continue
                
            box1 = sorted_boxes[i]
            merged = False
            
            # Look ahead for a merge candidate
            # We only check the next few boxes to avoid O(N^2) on large sets, 
            # though usually N is small per frame
            for j in range(i + 1, min(i + 10, len(sorted_boxes))):
                if j in used_indices:
                    continue
                    
                box2 = sorted_boxes[j]
                
                # Check vertical alignment (y-overlap)
                y1_a, y2_a = box1['bbox'][1], box1['bbox'][3]
                y1_b, y2_b = box2['bbox'][1], box2['bbox'][3]
                
                # Calculate vertical overlap
                overlap_y1 = max(y1_a, y1_b)
                overlap_y2 = min(y2_a, y2_b)
                overlap_h = max(0, overlap_y2 - overlap_y1)
                
                min_height = min(y2_a - y1_a, y2_b - y1_b)
                
                # Require significant vertical overlap (they should be on the same line)
                if min_height > 0 and overlap_h / min_height < 0.5:
                    continue
                    
                # Check horizontal proximity
                # box2 should be to the right of box1
                x2_a = box1['bbox'][2]
                x1_b = box2['bbox'][0]
                
                dist = x1_b - x2_a
                
                # Allow small negative distance (slight overlap) or positive distance up to threshold
                if -10 <= dist <= x_threshold:
                    # Candidate for merge
                    combined_text = (box1['text'] + " " + box2['text']).strip()
                    
                    # Check if combined text matches a name
                    is_match = False
                    norm_combined = combined_text.lower().strip()
                    norm_combined_ascii = unicodedata.normalize('NFD', norm_combined).encode('ascii', 'ignore').decode("utf-8")
                    
                    # Check if the combined text is a substring of any name or vice versa
                    # This is a loose check, strict filtering happens later
                    for name in normalized_names:
                        if name in norm_combined or norm_combined in name or \
                           name in norm_combined_ascii or norm_combined_ascii in name:
                            is_match = True
                            break
                    
                    if is_match:
                        # Merge them
                        new_x1 = min(box1['bbox'][0], box2['bbox'][0])
                        new_y1 = min(box1['bbox'][1], box2['bbox'][1])
                        new_x2 = max(box1['bbox'][2], box2['bbox'][2])
                        new_y2 = max(box1['bbox'][3], box2['bbox'][3])
                        
                        new_box = {
                            'bbox': (new_x1, new_y1, new_x2, new_y2),
                            'text': combined_text,
                            'confidence': (box1['confidence'] + box2['confidence']) / 2
                        }
                        
                        # Update box1 to be the merged box and continue checking for more merges with this new box
                        # This allows merging "Mario" + "Rossi" + "Verdi"
                        box1 = new_box
                        used_indices.add(j)
                        merged = True
                        # Don't break, keep looking for more parts of the name
            
            merged_boxes.append(box1)
            used_indices.add(i)
            i += 1
            
        merged_frame_boxes[frame_idx] = merged_boxes
        
    return merged_frame_boxes

def split_boxes_into_words(frame_boxes):
    """
    Split multi-word text boxes into individual word boxes.
    Uses proportional bbox division based on character positions.
    
    Args:
        frame_boxes (dict): Dictionary of frame indices to list of boxes with text
        
    Returns:
        dict: Frame boxes where each box represents a single word
    """
    import numpy as np
    
    split_frame_boxes = {}
    
    for frame_idx, boxes in frame_boxes.items():
        split_boxes = []
        
        for box in boxes:
            text = box.get('text', '').strip()
            if not text:
                continue
            
            words = text.split()
            if len(words) <= 1:
                # Single word or empty - keep as is
                split_boxes.append(box)
                continue
            
            # Multi-word box - split it
            x1, y1, x2, y2 = box['bbox']
            box_width = x2 - x1
            
            # Find character positions for each word in the original text
            char_positions = []
            current_pos = 0
            for word in words:
                start = text.find(word, current_pos)
                end = start + len(word)
                char_positions.append((word, start, end))
                current_pos = end
            
            # Create bbox for each word based on proportional character positions
            # Make boxes overlap at borders to avoid gaps
            total_chars = len(text)
            word_boxes = []
            
            for i, (word, char_start, char_end) in enumerate(char_positions):
                # Calculate proportional x positions
                start_ratio = char_start / total_chars
                end_ratio = char_end / total_chars
                
                word_x1 = x1 + int(start_ratio * box_width)
                word_x2 = x1 + int(end_ratio * box_width)
                
                # First word: start at box beginning
                if i == 0:
                    word_x1 = x1
                
                # Last word: end at box end
                if i == len(char_positions) - 1:
                    word_x2 = x2
                else:
                    # Extend to overlap slightly with next word (avoid gaps)
                    # Add 2 pixels to create small overlap
                    word_x2 = min(x2, word_x2 + 2)
                
                word_boxes.append({
                    'bbox': (word_x1, y1, word_x2, y2),
                    'text': word,
                    'confidence': box.get('confidence', 1.0),
                    'parent_box': box['bbox']
                })
            
            # Ensure adjacent boxes overlap: each box starts where previous ended
            for i in range(1, len(word_boxes)):
                prev_x2 = word_boxes[i-1]['bbox'][2]
                curr_x1, curr_y1, curr_x2, curr_y2 = word_boxes[i]['bbox']
                # Start current box at previous box's end (creates overlap)
                word_boxes[i]['bbox'] = (prev_x2 - 1, curr_y1, curr_x2, curr_y2)
            
            split_boxes.extend(word_boxes)
        
        split_frame_boxes[frame_idx] = split_boxes
    
    return split_frame_boxes

def filter_boxes_by_names(frame_boxes, names_dict, similarity_threshold=0.8):
    """
    Match word-level boxes against dictionary names.
    Uses actual word boxes from split_boxes_into_words - no character estimation.
    Merges adjacent matching words for multi-word names.
    
    Args:
        frame_boxes (dict): Word-level boxes from split_boxes_into_words
        names_dict (dict): Dictionary of names to alteregos
        similarity_threshold (float): Fuzzy match threshold
        
    Returns:
        dict: Filtered boxes with matched names
    """
    import re
    
    def normalize_text(text):
        """Normalize text for matching"""
        if not text:
            return ""
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        return text.lower().strip()
    
    def word_matches_name_part(word_text, name_part):
        """Check if word matches a part of a name"""
        norm_word = normalize_text(word_text)
        norm_name = normalize_text(name_part)
        
        # Exact match
        if norm_word == norm_name:
            return True, 1.0
        
        # Fuzzy match
        similarity = SequenceMatcher(None, norm_word, norm_name).ratio()
        if similarity >= similarity_threshold:
            return True, similarity
            
        return False, 0.0
    
    def group_boxes_into_lines(boxes):
        """Group boxes on same horizontal line"""
        if not boxes:
            return []
        
        # Sort by y-position
        sorted_boxes = sorted(enumerate(boxes), key=lambda x: (x[1]['bbox'][1] + x[1]['bbox'][3]) / 2)
        
        lines = []
        current_line = [sorted_boxes[0]]
        
        for idx, box in sorted_boxes[1:]:
            prev_y_center = (current_line[-1][1]['bbox'][1] + current_line[-1][1]['bbox'][3]) / 2
            curr_y_center = (box['bbox'][1] + box['bbox'][3]) / 2
            
            # Same line if y-centers within 15 pixels
            if abs(curr_y_center - prev_y_center) < 15:
                current_line.append((idx, box))
            else:
                lines.append(current_line)
                current_line = [(idx, box)]
        
        lines.append(current_line)
        
        # Sort each line by x-position (left to right)
        for line in lines:
            line.sort(key=lambda x: x[1]['bbox'][0])
        
        return lines
    
    def find_name_sequences(line_boxes, names_dict):
        """Find sequences of adjacent word boxes that match names"""
        matches = []
        
        for name, alterego in names_dict.items():
            name_words = normalize_text(name).split()
            
            # Try to find this name in the line
            for start_idx in range(len(line_boxes)):
                matched_boxes = []
                confidence_scores = []
                
                box_idx = start_idx
                word_idx = 0
                
                while word_idx < len(name_words) and box_idx < len(line_boxes):
                    orig_idx, box = line_boxes[box_idx]
                    box_text = box.get('text', '').strip()
                    
                    if not box_text:
                        box_idx += 1
                        continue
                    
                    is_match, conf = word_matches_name_part(box_text, name_words[word_idx])
                    
                    if is_match:
                        matched_boxes.append((orig_idx, box))
                        confidence_scores.append(conf)
                        word_idx += 1
                        box_idx += 1
                    else:
                        # No match - break this attempt
                        break
                
                # Check if we matched all words in the name
                if word_idx == len(name_words):
                    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                    matches.append({
                        'name': name,
                        'alterego': alterego,
                        'boxes': matched_boxes,
                        'confidence': avg_confidence
                    })
        
        # Remove overlapping matches, keep higher confidence ones
        if not matches:
            return []
        
        matches.sort(key=lambda x: (len(x['boxes']), x['confidence']), reverse=True)
        
        final_matches = []
        used_indices = set()
        
        for match in matches:
            box_indices = {idx for idx, _ in match['boxes']}
            if not box_indices.intersection(used_indices):
                final_matches.append(match)
                used_indices.update(box_indices)
        
        return final_matches
    
    def merge_matched_boxes(match):
        """Merge word boxes from a name match into single box with exact boundaries"""
        boxes = [box for _, box in match['boxes']]
        
        # Calculate bounding box of all matched word boxes
        x1 = min(b['bbox'][0] for b in boxes)
        y1 = min(b['bbox'][1] for b in boxes)
        x2 = max(b['bbox'][2] for b in boxes)
        y2 = max(b['bbox'][3] for b in boxes)
        
        # Get parent box if available
        parent_box = boxes[0].get('parent_box', None)
        
        return {
            'bbox': (x1, y1, x2, y2),
            'text': ' '.join(b.get('text', '') for b in boxes),
            'name': match['name'],
            'alterego': match['alterego'],
            'confidence': match['confidence'],
            'to_show': True,
            'parent_box': parent_box,
            'match_type': 'word_level'
        }
    
    # Main processing
    filtered_mapping = {}
    
    for frame_idx, boxes in tqdm(frame_boxes.items(), desc="Matching word boxes to names", unit="frame"):
        if not boxes:
            filtered_mapping[frame_idx] = []
            continue
        
        # Group boxes into lines
        lines = group_boxes_into_lines(boxes)
        
        frame_results = []
        all_matched_indices = set()
        
        # Process each line
        for line in lines:
            matches = find_name_sequences(line, names_dict)
            
            for match in matches:
                merged_box = merge_matched_boxes(match)
                frame_results.append(merged_box)
                
                # Track which original word boxes were matched
                all_matched_indices.update(idx for idx, _ in match['boxes'])
        
        # Add unmatched word boxes as to_show=False
        for i, box in enumerate(boxes):
            if i not in all_matched_indices:
                box_copy = box.copy()
                box_copy['to_show'] = False
                frame_results.append(box_copy)
        
        filtered_mapping[frame_idx] = frame_results
    
    return filtered_mapping


def enhanced_temporal_tracking(frame_boxes, max_gap=6, position_threshold=20, size_threshold=0.3):
    """
    Enhanced temporal tracking with frame-to-frame coordinate preservation.
    
    Args:
        frame_boxes (dict): Dictionary of frame indices to list of bounding boxes
        frame_skip (int): Frame skip interval used during detection
        max_gap (int): Maximum frame gap to interpolate across
        position_threshold (int): Maximum pixel distance for tracking
        size_threshold (float): Maximum relative size difference for tracking
        
    Returns:
        tuple: (stabilized_boxes dict, tracks list)
            - stabilized_boxes: Enhanced frame_boxes with stabilized coordinates
            - tracks: List of tracks, each track is a list of (frame_idx, box_idx) tuples
    """
    
    def can_track(box1, box2):
        """Check if two boxes can be tracked together"""
        bbox1, bbox2 = box1['bbox'], box2['bbox']
        
        # Check center distance
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Check size similarity
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        size_diff = abs(area1 - area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        
        # Check text similarity
        text_sim = SequenceMatcher(None, box1.get('alterego', '').lower(), 
                                 box2.get('alterego', '').lower()).ratio()
        
        return (distance < position_threshold and 
                size_diff < size_threshold and 
                text_sim > 0.8)
    
    def stabilize_coordinates(current_bbox, prev_bbox, movement_threshold=3):
        """
        Stabilize coordinates based on movement direction.
        
        Args:
            current_bbox (tuple): Current frame bounding box (x1, y1, x2, y2)
            prev_bbox (tuple): Previous frame bounding box (x1, y1, x2, y2)
            movement_threshold (int): Minimum movement to consider as actual movement
            
        Returns:
            tuple: Stabilized bounding box coordinates
        """
        curr_x1, curr_y1, curr_x2, curr_y2 = current_bbox
        prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
        
        # Calculate centers and movements
        curr_center_x = (curr_x1 + curr_x2) / 2
        curr_center_y = (curr_y1 + curr_y2) / 2
        prev_center_x = (prev_x1 + prev_x2) / 2
        prev_center_y = (prev_y1 + prev_y2) / 2
        
        x_movement = abs(curr_center_x - prev_center_x)
        y_movement = abs(curr_center_y - prev_center_y)
        
        # Calculate current dimensions
        curr_width = curr_x2 - curr_x1
        curr_height = curr_y2 - curr_y1
        prev_width = prev_x2 - prev_x1
        prev_height = prev_y2 - prev_y1
        
        # Determine movement type and stabilize accordingly
        if x_movement < movement_threshold and y_movement < movement_threshold:
            # Static: keep previous frame coordinates exactly
            return prev_bbox
            
        elif y_movement < movement_threshold or y_movement < x_movement / 2:
            # Horizontal movement: preserve Y coordinates, use current X but maintain consistent width
            stable_width = prev_width  # Keep consistent width
            stable_y1 = prev_y1
            stable_y2 = prev_y2
            
            # Use current center X but with stable width
            stable_x1 = int(curr_center_x - stable_width / 2)
            stable_x2 = stable_x1 + stable_width
            
            return (stable_x1, stable_y1, stable_x2, stable_y2)
            
        elif x_movement < movement_threshold or x_movement < y_movement / 2:
            # Vertical movement: preserve X coordinates, use current Y but maintain consistent height
            stable_height = prev_height  # Keep consistent height
            stable_x1 = prev_x1
            stable_x2 = prev_x2
            
            # Use current center Y but with stable height
            stable_y1 = int(curr_center_y - stable_height / 2)
            stable_y2 = stable_y1 + stable_height
            
            return (stable_x1, stable_y1, stable_x2, stable_y2)
            
        else:
            # Diagonal movement: use current coordinates but try to maintain consistent dimensions
            # Prefer previous dimensions if current ones are too different
            width_diff = abs(curr_width - prev_width) / prev_width if prev_width > 0 else 0
            height_diff = abs(curr_height - prev_height) / prev_height if prev_height > 0 else 0
            
            final_width = prev_width if width_diff > 0.2 else curr_width
            final_height = prev_height if height_diff > 0.2 else curr_height
            
            # Center the box with stable dimensions
            stable_x1 = int(curr_center_x - final_width / 2)
            stable_y1 = int(curr_center_y - final_height / 2)
            stable_x2 = stable_x1 + final_width
            stable_y2 = stable_y1 + final_height
            
            return (stable_x1, stable_y1, stable_x2, stable_y2)

    if not frame_boxes:
        return frame_boxes

    # Build tracks and apply frame-to-frame stabilization
    frame_indices = sorted(frame_boxes.keys())
    tracks = []  # List of tracks, each track is a list of (frame_idx, box) tuples
    track_assignments = {}  # Map (frame_idx, box_idx_in_stabilized) -> track_id
    
    # Initialize stabilized_boxes with ALL frame indices to preserve empty frames
    stabilized_boxes = {}
    for frame_idx in frame_indices:
        stabilized_boxes[frame_idx] = []
    
    # Build tracks first
    for frame_idx in frame_indices:
        current_boxes = frame_boxes[frame_idx]
        unmatched_boxes = list(enumerate(current_boxes))
        
        # Skip track processing for frames with no boxes (they're already in stabilized_boxes as empty)
        if not current_boxes:
            continue
        
        # Try to extend existing tracks
        for track in tracks:
            if not track:
                continue
                
            last_frame, last_box = track[-1]
            
            # Skip if track is too old
            if frame_idx - last_frame > max_gap:
                continue
            
            # Find best matching box
            best_match = None
            best_idx = None
            
            for box_idx, box in unmatched_boxes:
                if can_track(last_box, box):
                    if best_match is None:
                        best_match = box
                        best_idx = box_idx
            
            if best_match is not None:
                track.append((frame_idx, best_match))
                unmatched_boxes = [(idx, box) for idx, box in unmatched_boxes if idx != best_idx]
        
        # Create new tracks for unmatched boxes
        for _, box in unmatched_boxes:
            tracks.append([(frame_idx, box)])
    
    # Apply stabilization to each track
    for track_id, track in enumerate(tracks):
        if len(track) < 2:
            # Single detection - keep as is
            frame_idx, box = track[0]
            box_idx_in_stabilized = len(stabilized_boxes[frame_idx])
            stabilized_boxes[frame_idx].append(box)
            track_assignments[(frame_idx, box_idx_in_stabilized)] = track_id
            continue
        
        # Multi-frame track - apply frame-to-frame stabilization
        for i, (frame_idx, box) in enumerate(track):
            box_idx_in_stabilized = len(stabilized_boxes[frame_idx])
            
            if i == 0:
                # First frame in track - keep original coordinates
                stabilized_box = box.copy()
            else:
                # Stabilize based on previous frame
                prev_frame, prev_box = track[i-1]
                stabilized_prev_bbox = stabilized_boxes[prev_frame][-1]['bbox']  # Get the stabilized previous box
                
                stabilized_bbox = stabilize_coordinates(box['bbox'], stabilized_prev_bbox)
                
                stabilized_box = box.copy()
                stabilized_box['bbox'] = stabilized_bbox
            
            stabilized_boxes[frame_idx].append(stabilized_box)
            track_assignments[(frame_idx, box_idx_in_stabilized)] = track_id
    
    # Convert track_assignments to track list format: [[(frame_idx, box_idx), ...], ...]
    tracks_output = [[] for _ in range(len(tracks))]
    for (frame_idx, box_idx), track_id in track_assignments.items():
        tracks_output[track_id].append((frame_idx, box_idx))
    
    return stabilized_boxes, tracks_output


def normalize_box_heights(frame_boxes, height_clusters=None):
    """
    Normalize text box heights to consistent sizes based on clustering.
    
    Args:
        frame_boxes (dict): Dictionary of frame indices to list of bounding boxes
        height_clusters (list): Predefined height clusters, if None will compute from data
        
    Returns:
        dict: Frame boxes with normalized heights
        list: Height clusters used
    """
    from sklearn.cluster import KMeans
    import numpy as np
    
    # Collect all box heights if clusters not provided
    if height_clusters is None:
        all_heights = []
        for frame_boxes_list in frame_boxes.values():
            for box in frame_boxes_list:
                if 'bbox' in box:
                    x1, y1, x2, y2 = box['bbox']
                    height = y2 - y1
                    all_heights.append(height)
        
        if not all_heights:
            return frame_boxes, []
        
        # Cluster heights into common sizes (typically 3-4 sizes in messaging apps)
        heights_array = np.array(all_heights).reshape(-1, 1)
        n_clusters = min(4, len(set(all_heights)))  # Max 4 text sizes
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(heights_array)
        height_clusters = sorted(kmeans.cluster_centers_.flatten())
    
    # Normalize boxes to closest cluster height
    normalized_boxes = {}
    for frame_idx, boxes in frame_boxes.items():
        normalized_frame_boxes = []
        for box in boxes:
            if 'bbox' in box:
                x1, y1, x2, y2 = box['bbox']
                current_height = y2 - y1
                
                # Find closest cluster height
                closest_height = min(height_clusters, key=lambda h: abs(h - current_height))
                
                # Adjust box to new height (center vertically)
                height_diff = closest_height - current_height
                new_y1 = max(0, y1 - height_diff // 2)
                new_y2 = new_y1 + int(closest_height)
                
                normalized_box = box.copy()
                normalized_box['bbox'] = (x1, new_y1, x2, new_y2)
                normalized_frame_boxes.append(normalized_box)
            else:
                normalized_frame_boxes.append(box)
        
        normalized_boxes[frame_idx] = normalized_frame_boxes
    
    return normalized_boxes, height_clusters

def ocr_boxes_to_unified(frame_boxes, tracks=None):
    """
    Convert OCR box mapping to unified format:
    {frame_idx: [{'bbox': (x1,y1,x2,y2), 'score': float, 'label': None,
                  'text': str, 'mask': None, 'source': 'ocr',
                  'to_show': bool, 'alterego': str, 'name': str, 'track_id': int or None}, ...]}
    
    Args:
        frame_boxes (dict): Dictionary of frame indices to list of boxes
        tracks (list): List of tracks, each track is a list of (frame_idx, box_idx) tuples
    
    Returns:
        dict: Unified format dictionary with track_id field
    """
    # Build track_map: (frame_idx, box_idx) -> track_id
    track_map = {}
    if tracks:
        for track_id, track in enumerate(tracks):
            for frame_idx, box_idx in track:
                track_map[(frame_idx, box_idx)] = track_id
    
    unified = {}
    for frame_idx, boxes in frame_boxes.items():
        unified_list = []
        for i, box in enumerate(boxes):
            bbox = box.get('bbox') or box.get('original_bbox')
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            
            # Get track_id for this box
            track_id = track_map.get((frame_idx, i), None)
            
            entry = {
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "parent_box": box.get('parent_box', None),
                "score": float(box.get('confidence', 1.0)),
                "text": box.get('text', '') or '',
                "alterego": box.get('alterego', '') or '',
                "mask": None,
                "source": "ocr",
                "to_show": bool(box.get('to_show', True)),
                "track_id": track_id,
            }
            unified_list.append(entry)
        unified[frame_idx] = unified_list
    return unified

with open(dict_path, 'r') as f:
    names_to_detect = json.load(f)

try:
    with open(boxes_pkl_path, "rb") as f:
        frame_boxes = pickle.load(f)
except:
    # Process frames to detect text boxes
    frame_boxes = process_frames_parallel(frames_dir, languages, num_workers)
    with open(boxes_pkl_path, "wb") as f:
        pickle.dump(frame_boxes, f)
    print(f"Saved detected text boxes to {boxes_pkl_path}")

frame_boxes = merge_split_names(frame_boxes, names_to_detect)
frame_boxes = split_boxes_into_words(frame_boxes)
filtered_frame_boxes = filter_boxes_by_names(frame_boxes, names_to_detect)

enhanced_boxes, ocr_tracks = enhanced_temporal_tracking(
    filtered_frame_boxes,
    max_gap=6,
    position_threshold=20,
    size_threshold=0.3
)

# Normalize box heights for consistency
enhanced_boxes, height_clusters = normalize_box_heights(enhanced_boxes)

unified = ocr_boxes_to_unified(enhanced_boxes, tracks=ocr_tracks)
with open(output_pkl_path, 'wb') as f:
    pickle.dump(unified, f)

print(f"Created {len(ocr_tracks)} OCR tracks across {len(unified)} frames")
print(f"Saved final OCR boxes with track_ids to {output_pkl_path}")