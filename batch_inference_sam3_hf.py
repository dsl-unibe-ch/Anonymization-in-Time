"""
Batch inference for SAM3 using Hugging Face transformers (facebook/sam3).

This mirrors the original batch_inference_sam3 flow but uses the HF checkpoint
and processor instead of the local repo import. Outputs a unified pickle
compatible with the annotation tooling (sam3_hf.pkl).
"""

import argparse
import os
from pathlib import Path
import pickle

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Sam3Processor, Sam3Model

from device_utils import resolve_device


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def get_image_files(folder_path: Path):
    """Return sorted list of image files in a folder."""
    files = []
    for ext in IMAGE_EXTS:
        files.extend(folder_path.glob(f"*{ext}"))
        files.extend(folder_path.glob(f"*{ext.upper()}"))
    return sorted(files)


def pack_mask(mask: np.ndarray, bbox):
    """Pack a boolean mask crop for compact storage."""
    packed = np.packbits(mask.reshape(-1).astype(np.uint8))
    return {
        "packed": packed,
        "shape": mask.shape,
        "bbox": bbox,
    }


def run_sam3_hf_on_image(image: Image.Image, text_prompt: str, model, processor, device: str, threshold: float):
    """Run SAM3 inference on a single PIL image and return processed results."""
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    processed = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    return processed


def convert_to_unified(all_results):
    """
    Convert HF SAM3 results to the unified dict format expected by the viewer.
    {frame_idx: [ {bbox, parent_box, score, text, alterego, mask, source, to_show, track_id}, ... ]}
    """
    unified = {}
    for frame_idx, res, _img_path in all_results:
        boxes = res.get("boxes", [])
        masks = res.get("masks", [])
        scores = res.get("scores", [])

        for box, mask, score in zip(boxes, masks, scores):
            box_np = box.cpu().numpy() if torch.is_tensor(box) else np.array(box)
            x1, y1, x2, y2 = [int(round(v)) for v in box_np.tolist()]

            mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else np.array(mask)
            mask_np = np.squeeze(mask_np).astype(bool)

            mask_entry = pack_mask(mask_np, [x1, y1, x2, y2])

            unified.setdefault(frame_idx, []).append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "parent_box": None,
                    "score": float(score.detach().cpu().item() if torch.is_tensor(score) else float(score)),
                    "text": "",
                    "alterego": "",
                    "mask": mask_entry,
                    "source": "sam3_hf",
                    "to_show": True,
                    "track_id": None,
                }
            )
    return unified


def process_folder(input_folder: Path, output_folder: Path, text_prompt: str, device: str, threshold: float):
    """Process all images in a folder and write sam3_hf.pkl."""
    output_folder.mkdir(parents=True, exist_ok=True)

    device = resolve_device(device)
    print(f"Using device: {device}")

    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    image_files = get_image_files(input_folder)
    print(f"Found {len(image_files)} images in {input_folder}")
    if not image_files:
        print("No images to process.")
        return None

    all_results = []
    for idx, img_path in enumerate(tqdm(image_files, desc="SAM3 HF")):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue

        try:
            res = run_sam3_hf_on_image(image, text_prompt, model, processor, device, threshold)
            all_results.append((idx, res, img_path))
        except Exception as e:
            print(f"Error on {img_path}: {e}")
            continue

    unified = convert_to_unified(all_results)
    out_path = output_folder / "sam3_hf.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(unified, f)
    print(f"Saved unified detections to {out_path} ({len(unified)} frames)")
    return unified


def main():
    parser = argparse.ArgumentParser(description="SAM3 (HF transformers) batch inference on an image folder")
    parser.add_argument("--input_folder", required=True, type=str, help="Folder with input images")
    parser.add_argument("--output_folder", required=True, type=str, help="Folder to save outputs (sam3_hf.pkl)")
    parser.add_argument("--text_prompt", default="profile image, profile picture", type=str, help="Text prompt to segment")
    parser.add_argument("--device", default="auto", type=str, help="Device: auto|cuda|mps|cpu (default: auto)")
    parser.add_argument("--threshold", default=0.5, type=float, help="Score threshold for masks")

    args = parser.parse_args()
    process_folder(
        input_folder=Path(args.input_folder),
        output_folder=Path(args.output_folder),
        text_prompt=args.text_prompt,
        device=args.device,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
