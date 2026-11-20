"""
Region obfuscation used by the export and the viewer preview.

A plain Gaussian blur is NOT anonymization: every blurred pixel is a fixed,
public, linear function of the original pixels, so an attacker who has a list
of candidates (the names in a chat, the members' profile pictures) can render
each candidate, apply the same blur and pick the closest match. See
``tools/blur_attack.py`` for a measurement.

The default method here throws the content away first: the masked region is
refilled from the pixels around it (which the viewer sees anyway), and only
then blurred, so the output inside the mask carries no information about what
was there. Only the mask's shape and position remain.
"""

import cv2
import numpy as np

METHODS = ("inpaint", "flat", "blur")
DEFAULT_METHOD = "inpaint"

# Pixels next to the mask (anti-aliased glyph edges, avatar rims) are not used
# as fill sources, so they do not smear streaks into the region.
_FILL_MARGIN = 2
_INPAINT_RADIUS = 5
_FLAT_GRAY = (160, 160, 160)


def _odd(k):
    k = max(1, int(k))
    return k if k % 2 else k + 1


def _bounds(mask, pad, shape):
    ys, xs = np.nonzero(mask)
    h, w = shape[:2]
    return (max(0, ys.min() - pad), min(h, ys.max() + pad + 1),
            max(0, xs.min() - pad), min(w, xs.max() + pad + 1))


def _fill_from_surroundings(image, mask):
    """Replace the masked pixels using only pixels outside the mask."""
    grown = cv2.dilate(mask.astype(np.uint8),
                       np.ones((2 * _FILL_MARGIN + 1,) * 2, np.uint8)) > 0
    if grown.all():
        # Nothing outside to fill from (full frame): a neutral constant.
        out = image.copy()
        out[:] = _FLAT_GRAY
        return out
    out = image.copy()
    # Zero the hidden pixels first so the fill cannot depend on them, whatever
    # the inpainting algorithm does internally.
    out[grown] = 0
    y0, y1, x0, x1 = _bounds(grown, 4 * _INPAINT_RADIUS, image.shape)
    crop = np.ascontiguousarray(out[y0:y1, x0:x1])
    m = grown[y0:y1, x0:x1].astype(np.uint8) * 255
    out[y0:y1, x0:x1] = cv2.inpaint(crop, m, _INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return out


_COLOR_STEP = 32


def infer_text_color(patch, fallback=(0, 0, 0)):
    """
    BGR colour of the text in an (unobfuscated) name patch, for the alterego.

    It must not reveal the name: a mean over all text pixels shifts with the
    amount of anti-aliasing, i.e. with the letters, and the attacker in
    tools/blur_attack.py ranked candidates by it. So only the strongest ink
    pixels are used (solid text has one colour there whatever the letters)
    and the result is snapped to a coarse grid.
    """
    if patch is None or patch.size == 0:
        return fallback
    try:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return fallback
    if gray.size == 0:
        return fallback
    _, binary = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    is_text = binary == (0 if np.count_nonzero(binary == 0) < binary.size / 2 else 255)
    if np.count_nonzero(is_text) < 5 or np.count_nonzero(~is_text) < 1:
        return fallback
    background = float(np.median(gray[~is_text]))
    contrast = np.abs(gray[is_text].astype(np.float32) - background)
    core = patch[is_text][contrast >= np.percentile(contrast, 90)]
    color = np.median(core, axis=0)
    snapped = np.clip(np.round(color / _COLOR_STEP) * _COLOR_STEP, 0, 255)
    return tuple(int(c) for c in snapped)


def obfuscate_region(image, mask, strength=91, method=DEFAULT_METHOD):
    """
    Hide the masked region of a BGR image.

    Args:
        image: BGR uint8 image (not modified).
        mask: Boolean (H, W) mask of the pixels to hide.
        strength: Gaussian kernel size (made odd) for the soft look.
        method: 'inpaint' (default: refill from the surroundings, then blur),
            'flat' (fill with a neutral gray, then blur), or 'blur' (legacy
            Gaussian blur of the original pixels; can be matched against
            candidates, kept only for comparison).

    Returns:
        New image. For 'inpaint' and 'flat' the pixels inside the mask do not
        depend on the original pixels inside the mask.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown obfuscation method {method!r}; use one of {METHODS}")
    mask = np.asarray(mask, dtype=bool)
    result = image.copy()
    if not mask.any():
        return result
    k = _odd(strength)

    if method == "blur":
        source = image
    elif method == "flat":
        source = image.copy()
        source[mask] = _FLAT_GRAY
    else:
        source = _fill_from_surroundings(image, mask)

    # Blur only a window around the mask (the kernel reaches k//2 px).
    y0, y1, x0, x1 = _bounds(mask, k // 2 + 1, image.shape)
    window = cv2.GaussianBlur(np.ascontiguousarray(source[y0:y1, x0:x1]), (k, k), 0)
    local = mask[y0:y1, x0:x1]
    result[y0:y1, x0:x1][local] = window[local]
    return result
