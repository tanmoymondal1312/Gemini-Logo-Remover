"""
detector.py  –  Gemini watermark / logo detector
Finds the sparkle icon in the bottom-right corner using HSV colour analysis.
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────
# HSV colour ranges that match the Gemini sparkle
# (blue, cyan, purple/violet gradient)
# ──────────────────────────────────────────────
GEMINI_COLOUR_RANGES = [
    # Blue
    (np.array([100, 40, 40]),  np.array([140, 255, 255])),
    # Cyan
    (np.array([85,  40, 40]),  np.array([100, 255, 255])),
    # Purple / violet
    (np.array([140, 40, 40]),  np.array([170, 255, 255])),
    # Bright white core of sparkle
    (np.array([0,   0, 220]),  np.array([180,  40, 255])),
]


def _colour_mask(region_bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask of Gemini-coloured pixels inside *region_bgr*."""
    hsv   = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    mask  = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in GEMINI_COLOUR_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)
    return mask


def _adaptive_logo_size(h: int, w: int) -> tuple[int, int]:
    """
    Estimate logo bounding box from image dimensions.
    Gemini embeds a logo that scales roughly with image size.
    Returns (logo_h, logo_w).
    """
    logo_h = max(50,  int(h * 0.07))
    logo_w = max(160, int(w * 0.16))
    return logo_h, logo_w


def detect(image_bgr: np.ndarray,
           padding: int = 18,
           min_area: int = 80) -> tuple[np.ndarray, bool]:
    """
    Detect the Gemini logo region.

    Returns
    -------
    mask : np.ndarray  –  uint8 binary mask (255 = logo, 0 = background)
    auto : bool        –  True if colour-based detection succeeded
    """
    h, w = image_bgr.shape[:2]

    # ── 1. Define the search window (bottom-right 18 % × 22 %) ──────────────
    sh = int(h * 0.18)
    sw = int(w * 0.22)
    region = image_bgr[h - sh:h, w - sw:w]

    # ── 2. Colour-based detection ────────────────────────────────────────────
    colour_mask = _colour_mask(region)

    # Noise removal + dilation to group sparkle pixels into one blob
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,   (22, 22))
    colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_CLOSE,  kernel_close)
    colour_mask = cv2.dilate(colour_mask, kernel_dilate, iterations=2)

    contours, _ = cv2.findContours(colour_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    full_mask = np.zeros((h, w), dtype=np.uint8)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= min_area:
            rx, ry, rw_c, rh_c = cv2.boundingRect(largest)

            # Expand bounding box by padding
            rx  = max(0, rx  - padding)
            ry  = max(0, ry  - padding)
            rx2 = min(sw, rx + rw_c + 2 * padding)
            ry2 = min(sh, ry + rh_c + 2 * padding)

            # Map back to full-image coordinates
            oy = h - sh
            ox = w - sw
            full_mask[oy + ry : oy + ry2, ox + rx : ox + rx2] = 255
            return full_mask, True

    # ── 3. Fallback: fixed adaptive corner region ────────────────────────────
    logo_h, logo_w = _adaptive_logo_size(h, w)
    full_mask[h - logo_h:h, w - logo_w:w] = 255
    return full_mask, False


def refine_mask(mask: np.ndarray, feather: int = 4) -> np.ndarray:
    """
    Slightly expand and soften the mask edges so inpainting blends cleanly.
    Returns the refined uint8 mask.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather * 2 + 1,
                                                       feather * 2 + 1))
    expanded = cv2.dilate(mask, k, iterations=1)
    # Gaussian blur → re-threshold to keep binary
    blurred  = cv2.GaussianBlur(expanded, (feather * 4 + 1, feather * 4 + 1), 0)
    _, refined = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    return refined
