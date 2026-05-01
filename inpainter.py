"""
inpainter.py  –  Inpainting strategies (all 100 % free)

Priority order:
  1. LaMa  (simple-lama-inpainting)  – best quality, deep-learning model
  2. TELEA  (cv2.INPAINT_TELEA)      – fast, good for small logos
  3. NS     (cv2.INPAINT_NS)         – Navier-Stokes, good for textured BGs
  4. Patch  (custom)                 – texture copy from surrounding area
"""

import cv2
import numpy as np
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# OpenCV built-in methods
# ──────────────────────────────────────────────────────────────────────────────

def inpaint_telea(image_bgr: np.ndarray,
                  mask: np.ndarray,
                  radius: int = 4) -> np.ndarray:
    """Fast Marching Method (Telea 2004) – great for small watermarks."""
    return cv2.inpaint(image_bgr, mask, inpaintRadius=radius,
                       flags=cv2.INPAINT_TELEA)


def inpaint_ns(image_bgr: np.ndarray,
               mask: np.ndarray,
               radius: int = 4) -> np.ndarray:
    """Navier-Stokes based inpainting – good for smooth gradients."""
    return cv2.inpaint(image_bgr, mask, inpaintRadius=radius,
                       flags=cv2.INPAINT_NS)


# ──────────────────────────────────────────────────────────────────────────────
# Custom patch-based fill (fallback when OpenCV leaves artefacts)
# ──────────────────────────────────────────────────────────────────────────────

def inpaint_patch(image_bgr: np.ndarray,
                  mask: np.ndarray) -> np.ndarray:
    """
    Fill the masked area by mirroring texture from the region directly
    above (or left if at top edge). Works best for uniform backgrounds.
    """
    result = image_bgr.copy()
    h, w   = image_bgr.shape[:2]

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return result

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    rh = y2 - y1 + 1
    rw = x2 - x1 + 1

    # Try to sample from directly above the logo
    src_y1 = max(0, y1 - rh * 2)
    src_y2 = y1

    if src_y2 - src_y1 > 4:
        patch = image_bgr[src_y1:src_y2, x1: x2 + 1]
        # Tile vertically until we have enough rows
        reps  = (rh // patch.shape[0]) + 2
        tiled = np.tile(patch, (reps, 1, 1))
        fill  = tiled[-rh:, :rw]
    else:
        # Fallback: use median colour of surrounding pixels
        border = _border_pixels(image_bgr, y1, y2, x1, x2, margin=20)
        med    = np.median(border, axis=0).astype(np.uint8)
        fill   = np.full((rh, rw, 3), med, dtype=np.uint8)

    # Apply fill only where mask is set
    roi_mask = mask[y1: y2 + 1, x1: x2 + 1]
    roi      = result[y1: y2 + 1, x1: x2 + 1]
    roi[roi_mask > 0] = fill[roi_mask > 0]
    result[y1: y2 + 1, x1: x2 + 1] = roi
    return result


def _border_pixels(img, y1, y2, x1, x2, margin=20):
    h, w = img.shape[:2]
    ry1 = max(0, y1 - margin)
    ry2 = min(h, y2 + margin)
    rx1 = max(0, x1 - margin)
    rx2 = min(w, x2 + margin)
    region = img[ry1:ry2, rx1:rx2]
    return region.reshape(-1, 3)


# ──────────────────────────────────────────────────────────────────────────────
# LaMa deep-learning inpainting  (optional – best quality)
# ──────────────────────────────────────────────────────────────────────────────

_lama_model = None   # lazy-loaded singleton


def _load_lama():
    global _lama_model
    if _lama_model is None:
        from simple_lama_inpainting import SimpleLama   # type: ignore
        _lama_model = SimpleLama()
    return _lama_model


def inpaint_lama(image_bgr: np.ndarray,
                 mask: np.ndarray) -> np.ndarray | None:
    """
    LaMa (Large Mask inpainting) – best free quality for large regions.
    Returns None if the library is not installed.
    """
    try:
        lama  = _load_lama()
        image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image  = Image.fromarray(image_rgb)
        pil_mask   = Image.fromarray(mask)
        result_pil = lama(pil_image, pil_mask)
        result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        return result_bgr
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Unified entry-point
# ──────────────────────────────────────────────────────────────────────────────

METHODS = ("auto", "lama", "telea", "ns", "patch")


def inpaint(image_bgr: np.ndarray,
            mask: np.ndarray,
            method: str = "auto") -> tuple[np.ndarray, str]:
    """
    Remove the masked region from image_bgr.

    Parameters
    ----------
    method : one of METHODS
        'auto'  → try lama → telea → patch
        'lama'  → LaMa deep model (best; requires simple-lama-inpainting)
        'telea' → OpenCV TELEA (fast, good quality)
        'ns'    → OpenCV Navier-Stokes
        'patch' → texture copy from surroundings

    Returns
    -------
    (result_bgr, method_used)
    """
    if method == "auto":
        # Try best-to-fastest
        result = inpaint_lama(image_bgr, mask)
        if result is not None:
            return result, "lama"
        return inpaint_telea(image_bgr, mask), "telea"

    if method == "lama":
        result = inpaint_lama(image_bgr, mask)
        if result is not None:
            return result, "lama"
        print("[warn] LaMa not available – falling back to TELEA")
        return inpaint_telea(image_bgr, mask), "telea (fallback)"

    if method == "telea":
        return inpaint_telea(image_bgr, mask), "telea"

    if method == "ns":
        return inpaint_ns(image_bgr, mask), "ns"

    if method == "patch":
        return inpaint_patch(image_bgr, mask), "patch"

    raise ValueError(f"Unknown method '{method}'. Choose from {METHODS}")
