"""
Render one frame of a processed video with every obfuscation technique side
by side (original, current Gaussian blur, pixelation, flat gray + blur,
inpaint + blur, full-frame transition blur).

Usage:
    python tools/compare_obfuscation.py --video_dir <processed dir> --frame 10 \
        --out comparison.png [--strength 91]
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ait.export_video import add_custom_text_to_frame, resolve_sam3_file  # noqa: E402
from ait.frame_files import list_frame_files  # noqa: E402
from ait.obfuscation import obfuscate_region  # noqa: E402
from ait.utils import rebuild_full_mask  # noqa: E402


def load_annotations(video_dir, frame_idx):
    state = video_dir / "state.pkl"
    if state.exists():
        anns = pickle.load(open(state, "rb")).get(frame_idx, [])
    else:
        anns = []
        for f in (video_dir / "ocr.pkl", resolve_sam3_file(video_dir)):
            if f.exists():
                anns += pickle.load(open(f, "rb")).get(frame_idx, [])
    return [a for a in anns if a.get("to_show", True)]


def pixelate(frame, mask, block):
    small = cv2.resize(frame, (max(1, frame.shape[1] // block), max(1, frame.shape[0] // block)),
                       interpolation=cv2.INTER_AREA)
    big = cv2.resize(small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = frame.copy()
    out[mask] = big[mask]
    return out


def render(frame, anns, method, strength):
    h, w = frame.shape[:2]
    if method == "original":
        return frame.copy()
    if method == "fullframe":
        return obfuscate_region(frame, np.ones((h, w), bool), 3 * strength, "blur")
    mask = np.zeros((h, w), bool)
    texts = []
    for a in anns:
        if a.get("source") == "ocr" and a.get("bbox"):
            x1, y1, x2, y2 = [int(v) for v in a["bbox"]]
            x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
            if x2 > x1 and y2 > y1:
                texts.append((a, frame[y1:y2, x1:x2].copy(), x1, y1, x2, y2))
                mask[y1:y2, x1:x2] = True
        elif a.get("source") == "sam3" and a.get("mask") is not None:
            mask |= rebuild_full_mask(a["mask"], (h, w))
    if method == "pixelate":
        out = pixelate(frame, mask, 8)
    else:
        out = obfuscate_region(frame, mask, strength, method)
    for a, patch, x1, y1, x2, y2 in texts:
        if (a.get("alterego") or "").strip():
            out = add_custom_text_to_frame(out, a, patch, x1, y1, x2, y2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=int, default=91)
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--only", default="",
                    help="comma-separated panels to show, e.g. original,blur,shade")
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    frame_file = next(f for f in list_frame_files(video_dir / "frames", (".jpg", ".png"))
                      if int("".join(filter(str.isdigit, f.stem))) == args.frame)
    frame = cv2.imread(str(frame_file))
    anns = load_annotations(video_dir, args.frame)
    k = args.strength
    panels = [("original", "Original"),
              ("blur", f"Current: Gaussian k={k}"),
              ("pixelate", "Pixelate 8 px"),
              ("flat", f"Flat gray + blur k={k}"),
              ("inpaint", f"Inpaint + blur k={k}"),
              ("shade", "Shade (new default)"),
              ("fullframe", f"Transition: full frame k={3 * k}")]
    if args.only:
        keep = args.only.split(",")
        panels = [p for p in panels if p[0] in keep]
    tiles = []
    for method, title in panels:
        img = render(frame, anns, method, k)
        if args.zoom != 1.0:
            img = cv2.resize(img, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_NEAREST)
        bar = np.full((34, img.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, title, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * max(1, args.zoom ** 0.5),
                    (0, 0, 0), 1, cv2.LINE_AA)
        tile = np.vstack([bar, img])
        tiles.append(np.pad(tile, ((0, 0), (0, 6), (0, 0)), constant_values=200))
    cv2.imwrite(args.out, np.hstack(tiles))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
