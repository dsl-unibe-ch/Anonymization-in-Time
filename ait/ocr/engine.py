"""
OCR engine wrapper supporting docTR and EasyOCR.

Both engines produce the same output format:
    {bbox, text, confidence, line_idx, parent_box (EasyOCR only)}

- docTR:   word-level boxes natively. parent_box added later by _normalize_heights.
- EasyOCR: line-level boxes split into word boxes here. parent_box = original line box.
"""

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Singleton caches
# ---------------------------------------------------------------------------
_doctr_predictor = None
_doctr_device = None
_easyocr_reader = None


def _select_device():
    """Auto-select best available device."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# docTR
# ---------------------------------------------------------------------------

def get_predictor(device=None):
    """Lazy-init and return the docTR OCR predictor."""
    global _doctr_predictor, _doctr_device
    if _doctr_predictor is not None:
        return _doctr_predictor

    from doctr.models import ocr_predictor
    _doctr_device = device or _select_device()
    _doctr_predictor = ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True).to(_doctr_device)
    print(f"docTR predictor loaded on {_doctr_device}")
    return _doctr_predictor


def extract_words(image_bgr: np.ndarray, predictor=None,
                  min_confidence: float = 0.4,
                  min_area: int = 100) -> list:
    """
    Run docTR on a single BGR frame and return word-level detections.

    Each detection dict has:
        bbox:       (x1, y1, x2, y2)  pixel coords, ints
        text:       str
        confidence: float
        line_idx:   int  — which docTR line this word belongs to
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

                    # Pad vertically — docTR boxes are tight and can clip
                    # ascenders/descenders. 15% of box height on each side.
                    box_h = y2 - y1
                    pad = max(1, int(round(box_h * 0.15)))
                    y1 = max(0, y1 - pad)
                    y2 = min(h, y2 + pad)

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


# ---------------------------------------------------------------------------
# EasyOCR
# ---------------------------------------------------------------------------

def get_easyocr_reader(languages=None):
    """Lazy-init and return the EasyOCR reader."""
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader

    import easyocr
    if languages is None:
        languages = ["en"]
    _, use_gpu = _select_device().type == "cuda", True
    import torch
    use_gpu = torch.cuda.is_available()
    print(f"Initializing EasyOCR reader (languages={languages}, gpu={use_gpu})...")
    _easyocr_reader = easyocr.Reader(languages, gpu=use_gpu)
    print("EasyOCR reader initialized.")
    return _easyocr_reader


def extract_words_easyocr(image_bgr: np.ndarray, reader=None,
                          min_confidence: float = 0.4,
                          min_area: int = 100,
                          languages: list = None) -> list:
    """
    Run EasyOCR on a single BGR frame and return word-level detections.

    EasyOCR returns line-level boxes. Each line is split into word boxes
    by proportionally dividing the x-span based on character count.
    The original line box is stored as parent_box on each word.

    Output format matches extract_words():
        bbox:       (x1, y1, x2, y2)  pixel coords
        text:       str
        confidence: float
        line_idx:   int
        parent_box: (x1, y1, x2, y2)  original line-level box
    """
    if reader is None:
        reader = get_easyocr_reader(languages)

    results = reader.readtext(
        image_bgr,
        paragraph=False,
        text_threshold=0.6,
        low_text=0.35,
        canvas_size=2560,
        adjust_contrast=0.5,
    )

    detections = []
    line_idx = 0

    for (bbox_pts, text, confidence) in results:
        if confidence < min_confidence:
            continue

        text = text.strip()
        if not text:
            continue

        x_coords = [pt[0] for pt in bbox_pts]
        y_coords = [pt[1] for pt in bbox_pts]
        lx1 = int(min(x_coords))
        ly1 = int(min(y_coords))
        lx2 = int(max(x_coords))
        ly2 = int(max(y_coords))

        if (lx2 - lx1) * (ly2 - ly1) < min_area:
            continue

        parent_box = (lx1, ly1, lx2, ly2)
        words = text.split()

        if len(words) <= 1:
            # Single word — line box is the word box
            detections.append({
                "bbox": parent_box,
                "text": text,
                "confidence": confidence,
                "line_idx": line_idx,
                "parent_box": parent_box,
            })
        else:
            # Split line box into word boxes proportionally by char length
            total_chars = sum(len(w) for w in words)
            # Add spacing: each gap between words counts as ~0.5 char
            total_with_gaps = total_chars + 0.5 * (len(words) - 1)
            line_w = lx2 - lx1
            cursor = lx1

            for wi, word in enumerate(words):
                word_chars = len(word)
                # Add half-gap before (except first) and after (except last)
                gap_before = 0.25 if wi > 0 else 0
                gap_after = 0.25 if wi < len(words) - 1 else 0
                word_span = (word_chars + gap_before + gap_after) / total_with_gaps * line_w

                wx1 = int(round(cursor + gap_before / total_with_gaps * line_w)) if wi > 0 else lx1
                cursor_end = cursor + word_span
                wx2 = int(round(cursor_end - gap_after / total_with_gaps * line_w)) if wi < len(words) - 1 else lx2

                detections.append({
                    "bbox": (wx1, ly1, wx2, ly2),
                    "text": word,
                    "confidence": confidence,
                    "line_idx": line_idx,
                    "parent_box": parent_box,
                })
                cursor = cursor_end

        line_idx += 1

    return detections


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup():
    """Release GPU memory held by loaded models."""
    global _doctr_predictor, _doctr_device, _easyocr_reader
    _doctr_predictor = None
    _doctr_device = None
    _easyocr_reader = None
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
