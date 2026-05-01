"""
detector.py  –  Gemini watermark / logo detector  (template-matching edition)

Detection pipeline (in order of precision):

  Stage 1 — Template matching
    • Load synthetic sparkle templates (5 shape variants × 26 sizes = 130 templates)
    • Run cv2.matchTemplate at every scale inside the bottom-right corner window
    • Also run edge-based matching (Canny) for gradient-filled logos
    • Accept the best hit above a confidence threshold

  Stage 2 — Pixel-level HSV mask inside the matched box
    • Within the small matched region, threshold the Gemini colour range
    • Only the actual logo-coloured pixels become the mask
    • Minimal morphological cleanup (tiny kernel, 1 iteration only)
    • Result: a mask that is only a few hundred pixels, not a big rectangle

  Stage 3 — HSV-only fallback (colour detection without template)
    • Used when template matching finds nothing
    • Dramatically reduced dilation vs old code (5×5, 1 iter vs 22×22, 2 iter)
    • Bounding box with small padding (6 px) instead of 18 px

  Stage 4 — Last resort: fixed adaptive corner region
    • Smallest possible fallback region

KEY IMPROVEMENT over the old code
  Old: kernel_dilate 22×22, 2 iterations → expands mask by ~88 px each side
  New: Stage 1+2 → mask covers only actual logo pixels (precise)
       Stage 3   → kernel 5×5, 1 iteration → expands by ~5 px only
"""

from __future__ import annotations
import cv2
import numpy as np
from .templates import get_templates, get_edge_templates


# ─────────────────────────────────────────────────────────────────────────────
# Gemini colour signature (HSV)
# ─────────────────────────────────────────────────────────────────────────────

_COLOUR_RANGES = [
    # Blue core
    (np.array([100, 50, 50]),  np.array([140, 255, 255])),
    # Cyan tones
    (np.array([85,  50, 50]),  np.array([100, 255, 255])),
    # Purple / violet
    (np.array([140, 40, 40]),  np.array([170, 255, 255])),
    # Bright white nucleus
    (np.array([0,   0, 210]),  np.array([180,  35, 255])),
    # Lighter blue-purple (some Gemini versions)
    (np.array([95,  30, 100]), np.array([145, 255, 255])),
]

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded template cache
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATES:      list[np.ndarray] | None = None
_EDGE_TEMPLATES: list[np.ndarray] | None = None


def _load_templates():
    global _TEMPLATES, _EDGE_TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES      = get_templates(min_size=16, max_size=120, step=4)
        _EDGE_TEMPLATES = get_edge_templates(min_size=16, max_size=120, step=4)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _colour_mask(region_bgr: np.ndarray) -> np.ndarray:
    """Return binary mask of all Gemini-coloured pixels in region_bgr."""
    hsv  = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _COLOUR_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)
    return mask


def _match_templates(region_gray: np.ndarray,
                     templates: list[np.ndarray],
                     threshold: float) -> tuple[int, int, int, int, float] | None:
    """
    Find the best matching template inside region_gray.
    Returns (x, y, w, h, score) in region coordinates, or None.
    """
    rh, rw  = region_gray.shape
    best_score = threshold
    best_box   = None

    for tmpl in templates:
        th, tw = tmpl.shape[:2]
        if th >= rh or tw >= rw:
            continue

        res = cv2.matchTemplate(region_gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            best_box   = (max_loc[0], max_loc[1], tw, th, max_val)

    return best_box


def _tight_colour_mask(region_bgr: np.ndarray,
                       rx1: int, ry1: int,
                       rx2: int, ry2: int) -> np.ndarray:
    """
    Return a full-region mask with colour pixels set only inside [ry1:ry2, rx1:rx2].
    Uses minimal morphology to avoid expanding beyond the actual logo pixels.
    """
    rh, rw = region_bgr.shape[:2]
    out    = np.zeros((rh, rw), dtype=np.uint8)

    sub = region_bgr[ry1:ry2, rx1:rx2]
    if sub.size == 0:
        return out

    cm = _colour_mask(sub)

    # Tiny cleanup: close small gaps, dilate by just 3 px to catch anti-aliased edges
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cm = cv2.morphologyEx(cm, cv2.MORPH_CLOSE, k3)
    cm = cv2.dilate(cm, k3, iterations=1)

    out[ry1:ry2, rx1:rx2] = cm
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect(image_bgr: np.ndarray,
           padding:  int = 8,
           min_area: int = 60) -> tuple[np.ndarray, bool]:
    """
    Detect the Gemini sparkle logo.

    Returns
    -------
    mask : uint8 ndarray  –  255 = logo pixels, 0 = background
    auto : bool           –  True when a real detection succeeded
    """
    _load_templates()

    h, w = image_bgr.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)

    # ── Search window: bottom-right 22 % × 28 % ──────────────────────────────
    sh = int(h * 0.22)
    sw = int(w * 0.28)
    oy = h - sh          # offset y into full image
    ox = w - sw          # offset x into full image
    region     = image_bgr[oy:h, ox:w]
    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # ── Stage 1: Template matching (pixel-accurate) ───────────────────────────
    # Try grayscale templates first, then edge-based templates
    for tmpl_list, t_thresh in [
        (_TEMPLATES,      0.32),
        (_EDGE_TEMPLATES, 0.28),
    ]:
        # Preprocess region to match template type
        if tmpl_list is _EDGE_TEMPLATES:
            query = cv2.Canny(region_gray, 40, 120)
        else:
            query = region_gray

        box = _match_templates(query, tmpl_list, threshold=t_thresh)

        if box is not None:
            rx, ry, tw, th, score = box
            print(f"[detect]   Template match score={score:.3f}  "
                  f"loc=({rx},{ry}) size={tw}×{th}")

            # Expand matched box by a small padding
            rx1 = max(0,  rx  - padding)
            ry1 = max(0,  ry  - padding)
            rx2 = min(sw, rx  + tw + padding)
            ry2 = min(sh, ry  + th + padding)

            # ── Stage 2: pixel-level HSV mask inside the matched box ─────────
            colour_mask = _tight_colour_mask(region, rx1, ry1, rx2, ry2)

            n_logo_px = int(colour_mask.sum() // 255)
            if n_logo_px >= min_area:
                full_mask[oy:h, ox:w] = colour_mask
                print(f"[detect]   Logo pixels found: {n_logo_px}  "
                      f"(template + HSV)")
                return full_mask, True

            # Template matched but HSV didn't confirm — fall back to tight rect
            if n_logo_px > 0 or score >= 0.45:
                full_mask[oy + ry1: oy + ry2, ox + rx1: ox + rx2] = 255
                print(f"[detect]   Template rect  ({n_logo_px} colour px, "
                      f"score={score:.3f})")
                return full_mask, True

    # ── Stage 3: HSV-only, tight morphology ──────────────────────────────────
    colour_mask_full = _colour_mask(region)

    k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    colour_mask_full = cv2.morphologyEx(colour_mask_full,
                                        cv2.MORPH_CLOSE, k_close)
    colour_mask_full = cv2.dilate(colour_mask_full, k_dilate, iterations=1)

    contours, _ = cv2.findContours(colour_mask_full,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= min_area:
            rx, ry, rw_c, rh_c = cv2.boundingRect(largest)

            rx1 = max(0,  rx  - padding)
            ry1 = max(0,  ry  - padding)
            rx2 = min(sw, rx  + rw_c + padding)
            ry2 = min(sh, ry  + rh_c + padding)

            # Use the actual colour pixel mask (not just bounding rect)
            roi = np.zeros((sh, sw), dtype=np.uint8)
            roi[ry1:ry2, rx1:rx2] = colour_mask_full[ry1:ry2, rx1:rx2]
            full_mask[oy:h, ox:w] = roi

            n_px = int(roi.sum() // 255)
            print(f"[detect]   HSV-only  ({n_px} px, tight morphology)")
            return full_mask, True

    # ── Stage 4: Last resort — minimal fixed corner ───────────────────────────
    logo_h = max(40,  int(h * 0.06))
    logo_w = max(120, int(w * 0.13))
    full_mask[h - logo_h:h, w - logo_w:w] = 255
    print("[detect]   Fallback corner region used")
    return full_mask, False


def refine_mask(mask: np.ndarray, feather: int = 2) -> np.ndarray:
    """
    Gently expand and soften mask edges for seamless inpainting blending.

    Feather is intentionally small now (default 2 px) because the mask is
    already pixel-accurate — we only need to catch anti-aliased edge pixels.
    """
    if feather == 0:
        return mask

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (feather * 2 + 1, feather * 2 + 1))
    expanded = cv2.dilate(mask, k, iterations=1)
    blurred  = cv2.GaussianBlur(expanded,
                                 (feather * 4 + 1, feather * 4 + 1), 0)
    _, refined = cv2.threshold(blurred, 64, 255, cv2.THRESH_BINARY)
    return refined
