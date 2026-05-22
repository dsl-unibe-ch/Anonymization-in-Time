"""
SAM3 model setup and per-frame inference.
"""

import os
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

import numpy as np
from pathlib import Path
from PIL import Image

try:
    from ultralytics.models.sam import SAM3SemanticPredictor
except ImportError:
    print("Error: Ultralytics SAM3 not found. Install with: pip install ultralytics")
    exit(1)

from ait.utils import resolve_device
from ait.config import get_sam3_model_path, CONFIG_FILE


def resolve_model_path(model_path=None):
    """
    Resolve the SAM3 checkpoint path.

    Lookup order:
        1. ``model_path`` argument if it points to an existing file
        2. Saved config (``~/.ait/config.json`` → ``sam3_model_path``)
        3. ``./sam3.pt`` in the current working directory (legacy behavior)

    Raises FileNotFoundError with instructions if none resolve.
    """
    candidates = []
    if model_path:
        candidates.append(Path(model_path))
    saved = get_sam3_model_path()
    if saved:
        candidates.append(Path(saved))
    candidates.append(Path.cwd() / "sam3.pt")

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "SAM3 model checkpoint not found.\n\n"
        "Searched:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\n\n"
        "To fix:\n"
        "  1. Download `sam3.1_multiplex.pt` from https://huggingface.co/facebook/sam3.1\n"
        "  2. Rename it to `sam3.pt`\n"
        "  3. Either:\n"
        "     - Select it in the Video Processor GUI (SAM3 Model field), or\n"
        "     - Pass --sam3_model /path/to/sam3.pt to ait-process, or\n"
        f"     - Edit {CONFIG_FILE} to set `sam3_model_path`\n"
    )


def setup_predictor(device="auto", model_path=None, conf=0.25, half=True):
    """Initialize SAM3 Semantic Predictor with Ultralytics."""
    device = resolve_device(device)
    resolved_model_path = resolve_model_path(model_path)

    if device == "mps":
        device_str = "mps"
    elif device == "cuda":
        device_str = "0"
    else:
        device_str = "cpu"

    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model=resolved_model_path,
        device=device_str,
        imgsz=644,  # divisible by SAM3 stride (14)
        half=half and device in ["cuda", "mps"],
        save=False,
        verbose=False,
    )

    predictor = SAM3SemanticPredictor(overrides=overrides)
    print(f"Loaded Ultralytics SAM3 on device: {device_str}")

    return predictor, device


def process_image(predictor, image_path, text_prompt, frame_idx=0):
    """
    Process a single image with SAM3 predictor.

    Returns dict with 'boxes', 'masks', 'scores', 'labels'.
    """
    predictor.set_image(str(image_path))

    if isinstance(text_prompt, str):
        text_prompt = [text_prompt]

    results = predictor(text=text_prompt)

    if not results or len(results) == 0:
        return {'boxes': [], 'masks': [], 'scores': [], 'labels': []}

    result = results[0]

    boxes = []
    masks = []
    scores = []
    labels = []

    if result.masks is not None and result.boxes is not None:
        boxes_data = result.boxes.xyxy.cpu().numpy()
        masks_data = result.masks.data.cpu().numpy()
        scores_data = result.boxes.conf.cpu().numpy()

        for i in range(len(boxes_data)):
            boxes.append(boxes_data[i])
            masks.append(masks_data[i])
            scores.append(float(scores_data[i]))
            labels.append(1)

    return {
        'boxes': boxes,
        'masks': masks,
        'scores': scores,
        'labels': labels
    }


def get_image_files(folder_path):
    """Get all image files from folder, sorted."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = set()

    for ext in image_extensions:
        image_files.update(Path(folder_path).glob(f'*{ext}'))
        image_files.update(Path(folder_path).glob(f'*{ext.upper()}'))

    return sorted(image_files)
