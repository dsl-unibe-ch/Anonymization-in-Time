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


def setup_predictor(device="auto", model_path="sam3.pt", conf=0.25, half=True):
    """Initialize SAM3 Semantic Predictor with Ultralytics."""
    device = resolve_device(device)

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
        model=model_path,
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
