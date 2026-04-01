"""
docTR OCR engine wrapper.

Provides word-level text detection with automatic device selection
(CUDA > MPS > CPU).
"""

import numpy as np
import cv2

_predictor = None
_device = None


def _select_device():
    """Auto-select best available device."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_predictor(device=None):
    """
    Lazy-init and return the docTR OCR predictor.
    Cached globally so it's only loaded once per process.
    """
    global _predictor, _device
    if _predictor is not None:
        return _predictor

    from doctr.models import ocr_predictor
    _device = device or _select_device()
    _predictor = ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True).to(_device)
    print(f"docTR predictor loaded on {_device}")
    return _predictor


def extract_words(image_bgr: np.ndarray, predictor=None,
                  min_confidence: float = 0.4,
                  min_area: int = 100) -> list:
    """
    Run docTR on a single BGR frame and return word-level detections.

    Each detection dict has:
        bbox:       (x1, y1, x2, y2)  pixel coords, ints
        text:       str
        confidence: float
        line_idx:   int  — which docTR line this word belongs to (for grouping)

    Args:
        image_bgr:      OpenCV BGR frame.
        predictor:      docTR predictor (if None, uses global singleton).
        min_confidence: Drop words below this confidence. Default 0.4.
        min_area:       Drop boxes smaller than this pixel area. Default 100.
    """
    if predictor is None:
        predictor = get_predictor()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = predictor([image_rgb])
    h, w = image_bgr.shape[:2]

    detections = []
    line_counter = 0

    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    if word.confidence < min_confidence:
                        continue
                    (xmin, ymin), (xmax, ymax) = word.geometry
                    x1 = int(round(xmin * w))
                    y1 = int(round(ymin * h))
                    x2 = int(round(xmax * w))
                    y2 = int(round(ymax * h))

                    if (x2 - x1) * (y2 - y1) < min_area:
                        continue

                    text = word.value.strip()
                    if not text:
                        continue

                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "text": text,
                        "confidence": word.confidence,
                        "line_idx": line_counter,
                    })
                line_counter += 1

    return detections


def cleanup():
    """Release GPU memory held by the predictor."""
    global _predictor, _device
    _predictor = None
    _device = None
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
