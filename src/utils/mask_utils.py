"""
Utility functions for unpacking SAM3 masks.
"""
import numpy as np
import cv2


def unpack_mask_entry(mask_entry):
    """
    Rebuild a cropped mask from a packed entry.
    
    Args:
        mask_entry: Either a dict with 'packed' data or a numpy array
        
    Returns:
        Tuple of (mask_crop, bbox) where bbox is [x1, y1, x2, y2] or None
    """
    if isinstance(mask_entry, dict) and "packed" in mask_entry:
        h, w = mask_entry["shape"]
        flat = np.unpackbits(mask_entry["packed"])[: h * w]
        return flat.reshape((h, w)).astype(bool), mask_entry["bbox"]
    return np.array(mask_entry, dtype=bool), None


def rebuild_full_mask(mask_entry, img_shape):
    """
    Unpack cropped/packed mask into a full-frame boolean mask aligned to the original image.
    
    Args:
        mask_entry: Packed mask dictionary or numpy array
        img_shape: Target image shape (H, W) or (H, W, C)
        
    Returns:
        Full-frame boolean mask array
    """
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
