"""
templates.py  –  Gemini sparkle template bank

Template types generated (all at sizes 12–150 px):
  ① PIL-rendered ✦ / ✧ / ✴ characters (most accurate to real Gemini sparkle)
  ② 4-pointed star polygon  — inner_ratio 0.08 / 0.14 / 0.22
  ③ Diamond cross           — arm_ratio 0.14 / 0.20 / 0.26
  ④ Nucleus circle          — bright centre glow
  ⑤ Blurred versions of each (catches anti-aliased / gradient-filled logos)

Edge-map versions (Canny) are also provided for gradient-fill logo matching.
"""

from __future__ import annotations
import os
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# PIL character rendering  (✦ ✧ ✴)
# ─────────────────────────────────────────────────────────────────────────────

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSans-Regular.otf",
]
_SPARKLE_CHARS = ["✦", "✧", "✴", "✺"]


def _render_char(char: str, size: int, font_path: str) -> np.ndarray | None:
    """
    Render a Unicode character at the given size using PIL.
    Returns a cropped grayscale numpy array, or None if rendering failed.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
        fnt  = ImageFont.truetype(font_path, size)
        bbox = fnt.getbbox(char)
        if bbox is None:
            return None
        cw = max(1, bbox[2] - bbox[0])
        ch = max(1, bbox[3] - bbox[1])
        pad = max(4, size // 8)
        img  = Image.new("L", (cw + pad * 2, ch + pad * 2), 0)
        draw = ImageDraw.Draw(img)
        draw.text((-bbox[0] + pad, -bbox[1] + pad), char, font=fnt, fill=255)
        arr = np.array(img)
        ys, xs = np.where(arr > 10)
        if len(ys) == 0:
            return None
        cropped = arr[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
        # Normalise to full 0-255 range
        if cropped.max() > 0:
            cropped = (cropped.astype(np.float32) * 255 / cropped.max()).astype(np.uint8)
        return cropped
    except Exception:
        return None


def _char_templates(sizes: list[int]) -> list[np.ndarray]:
    """Try every font × every sparkle char and collect valid templates."""
    seen: set[tuple] = set()
    out: list[np.ndarray] = []

    for font_path in _FONT_PATHS:
        if not os.path.exists(font_path):
            continue
        for char in _SPARKLE_CHARS:
            for size in sizes:
                tmpl = _render_char(char, size, font_path)
                if tmpl is None or tmpl.size < 9:
                    continue
                key = (tmpl.shape, int(tmpl.mean()))
                if key not in seen:
                    seen.add(key)
                    out.append(tmpl)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Geometric primitives
# ─────────────────────────────────────────────────────────────────────────────

def _star_polygon(size: int, inner_ratio: float = 0.12) -> np.ndarray:
    """4-pointed star polygon."""
    img = np.zeros((size, size), dtype=np.uint8)
    cx = cy = size // 2
    r_out = max(1, size // 2 - 1)
    r_in  = max(1, int(r_out * inner_ratio))
    pts   = []
    for i in range(8):
        angle = i * np.pi / 4 - np.pi / 2
        r = r_out if i % 2 == 0 else r_in
        pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], 255)
    return img


def _diamond_cross(size: int, arm_ratio: float = 0.18) -> np.ndarray:
    """Two perpendicular thin diamonds → 4-pointed star."""
    img = np.zeros((size, size), dtype=np.uint8)
    cx = cy = size // 2
    w  = max(1, int(size * arm_ratio))
    h_pts = np.array([[0, cy-w//2],[cx, cy-w],[size-1, cy-w//2],
                       [size-1, cy+w//2],[cx, cy+w],[0, cy+w//2]], np.int32)
    v_pts = np.array([[cx-w//2, 0],[cx-w, cy],[cx-w//2, size-1],
                       [cx+w//2, size-1],[cx+w, cy],[cx+w//2, 0]], np.int32)
    cv2.fillPoly(img, [h_pts], 255)
    cv2.fillPoly(img, [v_pts], 255)
    return img


def _nucleus(size: int, r_ratio: float = 0.22) -> np.ndarray:
    """Bright centre circle (nucleus glow of the sparkle)."""
    img = np.zeros((size, size), dtype=np.uint8)
    r   = max(2, int(size * r_ratio))
    cv2.circle(img, (size // 2, size // 2), r, 255, -1)
    return img


def _filled_cross(size: int, arm_ratio: float = 0.22) -> np.ndarray:
    """Wide + cross — matches the thick-arm Gemini style."""
    img = np.zeros((size, size), dtype=np.uint8)
    w   = max(1, int(size * arm_ratio))
    cy  = cx = size // 2
    cv2.rectangle(img, (0, cy - w), (size - 1, cy + w), 255, -1)
    cv2.rectangle(img, (cx - w, 0), (cx + w, size - 1), 255, -1)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Blurred variants  (catches anti-aliased / soft-edge logos)
# ─────────────────────────────────────────────────────────────────────────────

def _blur_variants(templates: list[np.ndarray],
                   sigmas: tuple[float, ...] = (1.0, 1.8)) -> list[np.ndarray]:
    blurred = []
    for t in templates:
        for sigma in sigmas:
            k = max(3, int(sigma * 6) | 1)
            b = cv2.GaussianBlur(t, (k, k), sigma)
            if b.max() > 0:
                blurred.append(b)
    return blurred


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_templates(min_size: int = 16,
                  max_size: int = 130,
                  step:     int = 6) -> list[np.ndarray]:
    """
    Full template bank: PIL chars + geometric shapes + blurred variants.

    step=6 keeps the template count manageable (~400) for fast matching.
    PIL templates are returned first so early-stopping hits them before
    falling through to the geometric variants.
    """
    sizes = list(range(min_size, max_size + 1, step))

    # ① PIL-rendered sparkle characters (most accurate)
    char_tmpls = _char_templates(sizes)

    # ② Geometric variants
    geo_tmpls: list[np.ndarray] = []
    for size in sizes:
        for ir in (0.08, 0.14, 0.22):
            geo_tmpls.append(_star_polygon(size, inner_ratio=ir))
        for ar in (0.14, 0.20, 0.26):
            geo_tmpls.append(_diamond_cross(size, arm_ratio=ar))
        geo_tmpls.append(_nucleus(size, r_ratio=0.22))
        geo_tmpls.append(_filled_cross(size, arm_ratio=0.22))

    # ③ Blurred versions of both sets
    all_sharp   = char_tmpls + geo_tmpls
    all_blurred = _blur_variants(all_sharp, sigmas=(1.0, 1.8))

    return all_sharp + all_blurred


def get_edge_templates(min_size: int = 16,
                       max_size: int = 130,
                       step:     int = 6) -> list[np.ndarray]:
    """
    Canny-edge versions of the template bank.
    More robust when the logo has a gradient fill (edges are sharper than fill).
    """
    base = get_templates(min_size, max_size, step)
    out  = []
    for t in base:
        e = cv2.Canny(t, 30, 100)
        if e.any():
            out.append(e)
    return out
