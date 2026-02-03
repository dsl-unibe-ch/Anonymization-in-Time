"""
Annotation Manager - handles loading, saving, and managing annotation data.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from utils.pickle_loader import load_pickle, save_pickle, merge_annotations, assign_unique_ids


class AnnotationManager:
    """Manages video frame annotations from OCR and SAM3."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize the annotation manager.
        
        Args:
            data_dir: Path to data directory containing pickle files
        """
        self.data_dir = Path(data_dir)
        self.frames: Dict[int, List[dict]] = {}
        self.original_frames: Dict[int, List[dict]] = {}  # For reset functionality
        self.frame_indices: List[int] = []
        self.modified = False
        
    def load_annotations(self, sam_path_override: Optional[Path] = None) -> bool:
        """
        Load annotations from pickle files.
        Tries to load combined file first, then separate OCR/SAM3 files.
        
        Args:
            sam_path_override: Optional path to SAM3 pickle to use instead of sam3.pkl

        Returns:
            True if successful, False otherwise
        """
        combined_path = self.data_dir / "annotations.pkl"
        ocr_path = self.data_dir / "ocr.pkl"
        sam_path = sam_path_override if sam_path_override else (self.data_dir / "sam3.pkl")
        
        try:
            # Try combined file first
            if combined_path.exists():
                print(f"Loading combined annotations from {combined_path}")
                self.frames = load_pickle(combined_path)
            # Try separate files
            elif ocr_path.exists() or sam_path.exists():
                print("Loading separate OCR and SAM3 files...")
                ocr_data = load_pickle(ocr_path) if ocr_path.exists() else {}
                sam_data = load_pickle(sam_path) if sam_path.exists() else {}
                self.frames = merge_annotations(ocr_data, sam_data)
            else:
                print(f"No annotation files found in {self.data_dir}")
                return False
            
            # Assign unique IDs if not present
            self.frames = assign_unique_ids(self.frames)
            
            # Ensure all annotations have to_show field
            for frame_idx, annotations in self.frames.items():
                for ann in annotations:
                    if 'to_show' not in ann:
                        ann['to_show'] = True
            
            # Store original state for reset
            import copy
            self.original_frames = copy.deepcopy(self.frames)
            
            # Get sorted frame indices
            self.frame_indices = sorted(self.frames.keys())
            
            print(f"Loaded {len(self.frame_indices)} frames with annotations")
            return True
            
        except Exception as e:
            print(f"Error loading annotations: {e}")
            return False
    
    def get_frame_annotations(self, frame_idx: int) -> List[dict]:
        """
        Get all annotations for a specific frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            List of annotation dictionaries
        """
        return self.frames.get(frame_idx, [])
    
    def toggle_annotation(self, annotation_id: int, frame_idx: int) -> bool:
        """
        Toggle the visibility of an annotation and propagate across frames with same track_id.
        
        Args:
            annotation_id: Unique annotation ID
            frame_idx: Frame index
            
        Returns:
            New to_show state
        """
        annotations = self.frames.get(frame_idx, [])
        
        # Find the annotation and get its track_id
        target_ann = None
        for ann in annotations:
            if ann.get('id') == annotation_id:
                target_ann = ann
                break
        
        if not target_ann:
            return False
        
        # Toggle the annotation
        new_state = not target_ann['to_show']
        target_ann['to_show'] = new_state
        self.modified = True
        
        # Propagate to all annotations with same track_id (if available)
        track_id = target_ann.get('track_id')
        if track_id is not None:
            propagated_count = 0
            for frame_idx_iter, frame_annotations in self.frames.items():
                for ann in frame_annotations:
                    # Skip the original annotation
                    if ann.get('id') == annotation_id:
                        continue
                    # Toggle if same track_id
                    if ann.get('track_id') == track_id:
                        ann['to_show'] = new_state
                        propagated_count += 1
            
            if propagated_count > 0:
                print(f"Toggled {propagated_count + 1} annotations across frames (track_id={track_id})")
        
        return new_state
    
    def toggle_parent_box(self, annotation_id: int, frame_idx: int) -> bool:
        """
        Toggle the visibility of all annotations that share the same parent_box
        as the clicked annotation, propagating across all frames.
        Uses fuzzy position matching on parent_box coordinates.
        
        Args:
            annotation_id: Unique annotation ID of clicked annotation
            frame_idx: Frame index where click occurred
            
        Returns:
            New to_show state
        """
        annotations = self.frames.get(frame_idx, [])
        
        # Find the clicked annotation and get its parent_box
        target_ann = None
        for ann in annotations:
            if ann.get('id') == annotation_id:
                target_ann = ann
                break
        
        if not target_ann:
            return False
        
        parent_box = target_ann.get('parent_box')
        parent_box_text = target_ann.get('parent_box_text', '')
        
        if parent_box is None:
            # No parent box - fall back to regular toggle
            return self.toggle_annotation(annotation_id, frame_idx)
        
        # Determine new state from the clicked annotation
        new_state = not target_ann['to_show']
        self.modified = True
        
        def parent_boxes_match(pb1, pb2, text1, text2, threshold=30):
            """Check if two parent boxes are similar enough to be considered the same."""
            if pb1 is None or pb2 is None:
                return False
            
            # First check text - must match exactly (text is stabilized)
            if text1 and text2:
                if text1.lower() != text2.lower():
                    return False
            
            # Check position similarity (center point)
            x1_center = (pb1[0] + pb1[2]) / 2
            y1_center = (pb1[1] + pb1[3]) / 2
            x2_center = (pb2[0] + pb2[2]) / 2
            y2_center = (pb2[1] + pb2[3]) / 2
            
            x_dist = abs(x1_center - x2_center)
            y_dist = abs(y1_center - y2_center)
            
            # More lenient on vertical movement (chat scrolling)
            return x_dist < threshold and y_dist < threshold * 2
        
        # Toggle all annotations across all frames that have similar parent_box
        toggled_count = 0
        for frame_idx_iter, frame_annotations in self.frames.items():
            for ann in frame_annotations:
                ann_parent = ann.get('parent_box')
                ann_parent_text = ann.get('parent_box_text', '')
                
                # Check if this annotation has a similar parent_box
                if parent_boxes_match(parent_box, ann_parent, parent_box_text, ann_parent_text):
                    ann['to_show'] = new_state
                    toggled_count += 1
        
        print(f"Toggled {toggled_count} annotations with matching parent_box across all frames")
        
        return new_state
    
    def get_annotation_by_id(self, annotation_id: int, frame_idx: int) -> Optional[dict]:
        """
        Get annotation by ID.
        
        Args:
            annotation_id: Unique annotation ID
            frame_idx: Frame index
            
        Returns:
            Annotation dictionary or None
        """
        annotations = self.frames.get(frame_idx, [])
        
        for ann in annotations:
            if ann.get('id') == annotation_id:
                return ann
        
        return None
    
    def find_annotation_at_point(self, frame_idx: int, x: int, y: int) -> Optional[dict]:
        """
        Find annotation at given point (considering both bbox and mask).
        NOTE: This method is not used by the tkinter version but kept for compatibility.
        
        Args:
            frame_idx: Frame index
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Annotation dictionary or None
        """
        annotations = self.frames.get(frame_idx, [])
        
        # Simple bbox and mask checking without Qt dependencies
        for ann in reversed(annotations):
            # Check bounding box
            if 'bbox' in ann and ann['bbox']:
                x1, y1, x2, y2 = ann['bbox']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return ann
            
            # Check mask
            if 'mask' in ann and ann['mask'] is not None:
                mask = ann['mask']
                if mask.ndim == 3:
                    mask = mask.squeeze()
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    if mask[y, x] > 0:
                        return ann
        
        return None
    
    def save_state(self, filepath: Optional[Path] = None) -> bool:
        """
        Save current annotation state to file.
        
        Args:
            filepath: Optional custom path, defaults to data_dir/state.pkl
            
        Returns:
            True if successful
        """
        if filepath is None:
            filepath = self.data_dir / "state.pkl"
        
        try:
            save_pickle(self.frames, filepath)
            self.modified = False
            print(f"State saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving state: {e}")
            return False
    
    def export_visibility_state(self, filepath: Optional[Path] = None) -> bool:
        """
        Export only the visibility state (id -> to_show mapping).
        
        Args:
            filepath: Optional custom path, defaults to data_dir/visibility.pkl
            
        Returns:
            True if successful
        """
        if filepath is None:
            filepath = self.data_dir / "visibility.pkl"
        
        visibility_state = {}
        
        for frame_idx, annotations in self.frames.items():
            frame_visibility = {}
            for ann in annotations:
                frame_visibility[ann['id']] = ann['to_show']
            visibility_state[frame_idx] = frame_visibility
        
        try:
            save_pickle(visibility_state, filepath)
            print(f"Visibility state exported to {filepath}")
            return True
        except Exception as e:
            print(f"Error exporting visibility state: {e}")
            return False
    
    def get_frame_count(self) -> int:
        """Get total number of frames."""
        return len(self.frame_indices)
    
    def get_frame_index(self, idx: int) -> Optional[int]:
        """
        Get frame index by position.
        
        Args:
            idx: Position in sorted frame list
            
        Returns:
            Frame index or None
        """
        if 0 <= idx < len(self.frame_indices):
            return self.frame_indices[idx]
        return None
    
    def get_statistics(self, frame_idx: int) -> Dict[str, int]:
        """
        Get statistics for a frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Dictionary with counts
        """
        annotations = self.frames.get(frame_idx, [])
        
        stats = {
            'total': len(annotations),
            'visible': sum(1 for a in annotations if a['to_show']),
            'hidden': sum(1 for a in annotations if not a['to_show']),
            'ocr': sum(1 for a in annotations if a.get('source') == 'ocr'),
            'sam3': sum(1 for a in annotations if a.get('source') == 'sam3'),
        }
        
        return stats
