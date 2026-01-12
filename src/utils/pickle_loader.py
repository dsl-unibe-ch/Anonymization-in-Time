"""
Utility functions for loading and saving pickle files.
"""
import pickle
from pathlib import Path
from typing import Dict, Any


def load_pickle(filepath: Path) -> Dict[int, list]:
    """
    Load pickle file containing frame annotations.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Dictionary mapping frame indices to annotation lists
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='utf-8')
        
        # Ensure all text fields are proper UTF-8 strings
        for frame_idx, annotations in data.items():
            if isinstance(annotations, list):
                for ann in annotations:
                    if isinstance(ann, dict):
                        # Fix alterego and text fields
                        for key in ['alterego', 'text', 'name']:
                            if key in ann and ann[key]:
                                if isinstance(ann[key], bytes):
                                    ann[key] = ann[key].decode('utf-8')
                                else:
                                    ann[key] = str(ann[key])
        
        return data


def save_pickle(data: Dict[int, list], filepath: Path) -> None:
    """
    Save annotations to pickle file with UTF-8 encoding.
    
    Args:
        data: Dictionary mapping frame indices to annotation lists
        filepath: Path to save pickle file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure all text fields are proper UTF-8 strings before saving
    for frame_idx, annotations in data.items():
        if isinstance(annotations, list):
            for ann in annotations:
                if isinstance(ann, dict):
                    for key in ['alterego', 'text', 'name']:
                        if key in ann and ann[key]:
                            if isinstance(ann[key], bytes):
                                ann[key] = ann[key].decode('utf-8')
                            else:
                                ann[key] = str(ann[key])
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def merge_annotations(ocr_data: Dict[int, list], sam_data: Dict[int, list]) -> Dict[int, list]:
    """
    Merge OCR and SAM3 annotations into a single dictionary.
    
    Args:
        ocr_data: OCR annotations
        sam_data: SAM3 annotations
        
    Returns:
        Merged annotations dictionary
    """
    merged = {}
    
    # Get all frame indices
    all_frames = set(ocr_data.keys()) | set(sam_data.keys())
    
    for frame_idx in all_frames:
        frame_annotations = []
        
        # Add OCR annotations
        if frame_idx in ocr_data:
            frame_annotations.extend(ocr_data[frame_idx])
        
        # Add SAM annotations
        if frame_idx in sam_data:
            frame_annotations.extend(sam_data[frame_idx])
        
        merged[frame_idx] = frame_annotations
    
    return merged


def assign_unique_ids(frames: Dict[int, list]) -> Dict[int, list]:
    """
    Assign unique IDs to all annotations for tracking.
    
    Args:
        frames: Annotations dictionary
        
    Returns:
        Updated annotations with unique IDs
    """
    unique_id = 0
    
    for frame_idx in sorted(frames.keys()):
        for annotation in frames[frame_idx]:
            if 'id' not in annotation:
                annotation['id'] = unique_id
                unique_id += 1
    
    return frames
