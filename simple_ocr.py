from ocr import process_frames_parallel
import pickle
import json

boxes_pkl_path = "data/frame_boxes.pkl"
frames_dir = "data/frames"
dict_path = "laet.json"
output_pkl_path = "data/final_frame_boxes.pkl"

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

def filter_boxes_by_names(frame_boxes, names, similarity_threshold=0.7):
    """
    Annotate boxes with match info instead of dropping non-matches.
    Boxes that match a name are marked with to_show=True and use a sub-bbox around the match.
    Non-matching boxes are kept with to_show=False.
    """
    
    def normalize_text(text):
        """Normalize text for better matching - removes accents, standardizes case"""
        if not text:
            return ""
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        return text.lower().strip()
    
    def calculate_word_similarity(text1, text2):
        """Calculate similarity between two texts, handling word-level differences"""
        norm_text1, norm_text2 = normalize_text(text1), normalize_text(text2)
        direct_sim = SequenceMatcher(None, norm_text1, norm_text2).ratio()
        
        words1, words2 = norm_text1.split(), norm_text2.split()
        if len(words1) <= 1 and len(words2) <= 1:
            return direct_sim
        
        word_similarities = []
        for word1 in words1:
            best_sim = max(SequenceMatcher(None, word1, word2).ratio() for word2 in words2)
            word_similarities.append(best_sim)
        
        word_avg_sim = sum(word_similarities) / len(word_similarities) if word_similarities else 0
        return max(direct_sim, word_avg_sim)
    
    def find_exact_matches(text, name, norm_text, norm_name):
        """Find exact matches using word boundaries and normalization"""
        matches = []
        
        # Method 1: Exact word boundary match
        pattern = r'\b' + re.escape(name.lower().strip()) + r'\b'
        for match in re.finditer(pattern, text.lower().strip()):
            matches.append({
                'name': name, 'start': match.start(), 'end': match.end(),
                'match_type': 'exact_word', 'confidence': 1.0
            })
        
        # Method 2: Normalized exact match
        if not matches and norm_name in norm_text:
            start_pos = norm_text.find(norm_name)
            matches.append({
                'name': name, 'start': start_pos, 'end': start_pos + len(norm_name),
                'match_type': 'normalized_exact', 'confidence': 0.95
            })
        
        return matches
    
    def find_fuzzy_matches(text, name, norm_text, norm_name):
        """Find fuzzy matches for multi-word names and similar text"""
        matches = []
        name_words = norm_name.split()
        
        # Method 3: Multi-word fuzzy matching
        if len(name_words) > 1:
            word_positions = []
            for word in name_words:
                if word in norm_text:
                    word_positions.append(norm_text.find(word))
                else:
                    # Try fuzzy matching for individual words
                    text_words = norm_text.split()
                    best_pos = -1
                    best_sim = 0
                    
                    for text_word in text_words:
                        sim = SequenceMatcher(None, word, text_word).ratio()
                        if sim > best_sim and sim >= 0.6:
                            best_sim = sim
                            best_pos = norm_text.find(text_word)
                    
                    if best_pos != -1:
                        word_positions.append(best_pos)
                    else:
                        break
            else:  # All words found
                start_pos = min(word_positions)
                end_pos = min(start_pos + len(norm_name), len(norm_text))
                matches.append({
                    'name': name, 'start': start_pos, 'end': end_pos,
                    'match_type': 'multi_word_fuzzy', 'confidence': 0.8
                })
        
        # Method 4: Simple substring match
        elif norm_name in norm_text:
            start_pos = norm_text.find(norm_name)
            matches.append({
                'name': name, 'start': start_pos, 'end': start_pos + len(norm_name),
                'match_type': 'substring', 'confidence': 0.8
            })
        
        # Method 5: Overall fuzzy matching
        else:
            similarity = calculate_word_similarity(text, name)
            if similarity >= similarity_threshold:
                matches.append({
                    'name': name, 'start': 0, 'end': len(text),
                    'match_type': 'fuzzy_overall', 'confidence': similarity
                })
        
        return matches
    
    def find_name_positions_in_text(text, name_list):
        """Find all occurrences of names in the text and return their positions"""
        if not text:
            return []
        
        text_lower = text.lower().strip()
        norm_text = normalize_text(text)
        all_matches = []
        
        for name in name_list:
            norm_name = normalize_text(name)
            
            # Try exact matches first
            matches = find_exact_matches(text_lower, name, norm_text, norm_name)
            
            # If no exact matches, try fuzzy matching
            if not matches:
                matches = find_fuzzy_matches(text, name, norm_text, norm_name)
            
            all_matches.extend(matches)
        
        # Remove overlapping matches, keeping longer/higher confidence ones
        if not all_matches:
            return []
            
        # Sort by length (descending) then confidence (descending)
        # This prioritizes "Mario Rossi" over "Mario"
        all_matches.sort(key=lambda x: (x['end'] - x['start'], x['confidence']), reverse=True)
        
        final_matches = []
        for match in all_matches:
            is_overlapping = False
            for kept in final_matches:
                # Check for overlap
                if (match['start'] < kept['end'] and match['end'] > kept['start']):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                final_matches.append(match)
                
        return final_matches
    
    def calculate_char_positions_to_pixels(bbox, text, char_start, char_end):
        """Convert character positions to pixel coordinates within a bounding box"""
        x1, y1, x2, y2 = bbox
        box_width, box_height = x2 - x1, y2 - y1
        
        if len(text) == 0:
            return bbox
        
        # Calculate pixel positions with padding
        char_width = box_width / len(text)
        padding_x = max(3, int(char_width * 0.15))
        padding_y = max(3, int(box_height * 0.15))
        
        sub_x1 = max(x1, x1 + int(char_start * char_width) - padding_x)
        sub_x2 = min(x2, x1 + int(char_end * char_width) + padding_x)
        sub_y1 = max(y1, y1 - padding_y)
        sub_y2 = min(y2, y2 + padding_y)
        
        return (sub_x1, sub_y1, sub_x2, sub_y2)
    
    def process_frame_boxes(frame_id, boxes, names):
        """Process boxes for a single frame"""
        frame_results = []
        
        for box_info in boxes:
            text = box_info.get('text', '')
            if not text:
                # Keep empty text boxes but mark as not to show
                box_copy = box_info.copy()
                box_copy['to_show'] = False
                frame_results.append(box_copy)
                continue
            
            name_matches = find_name_positions_in_text(text, list(names.keys()))
            
            if name_matches:
                for match in name_matches:
                    sub_bbox = calculate_char_positions_to_pixels(
                        box_info['bbox'], text, match['start'], match['end']
                    )
                    
                    frame_results.append({
                        'name': match['name'],
                        'alterego': names[match['name']],
                        'text': text,
                        'char_start': match['start'],
                        'char_end': match['end'],
                        'bbox': sub_bbox,
                        'match_type': match['match_type'],
                        'confidence': match['confidence'],
                        'to_show': True,
                        'parent_box': box_info.get('bbox')
                    })
            else:
                box_copy = box_info.copy()
                box_copy['to_show'] = False
                frame_results.append(box_copy)
        
        return frame_results
    
    # Main processing
    filtered_mapping = {}
    
    for frame_id, boxes in tqdm(frame_boxes.items(), desc="Filtering Boxes", unit="frame"):
        frame_results = process_frame_boxes(frame_id, boxes, names)
        filtered_mapping[frame_id] = frame_results
    
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