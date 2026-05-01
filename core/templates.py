"""
templates.py  –  Synthetic Gemini sparkle template generator

The Gemini logo is a 4-pointed star (sparkle ✦) with a distinctive shape:
  - Four elongated diamond arms pointing N / S / E / W
  - Arms are very thin at the tips, slightly wider near the centre
  - A small bright nucleus at the centre

We generate many templates at different sizes and aspect ratios so that
cv2.matchTemplate can find the logo at any resolution.

No external files required — all templates are computed at runtime.
"""

from __future__ import annotations
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shape primitives
# ─────────────────────────────────────────────────────────────────────────────

def _diamond_cross(size: int, arm_ratio: float = 0.18) -> np.ndarray:
    """
    Two perpendicular thin diamonds overlapping at the centre → 4-pointed star.

    arm_ratio controls how wide each arm is relative to size.
    Smaller = thinner, sharper star. Range 0.10 – 0.30.
    """
    img = np.zeros((size, size), dtype=np.uint8)
    cx = cy = size // 2
    w  = max(1, int(size * arm_ratio))

    # Horizontal arm  ◇ rotated 90°
    h_pts = np.array([
        [0,        cy - w // 2],
        [cx,       cy - w],
        [size - 1, cy - w // 2],
        [size - 1, cy + w // 2],
        [cx,       cy + w],
        [0,        cy + w // 2],
    ], dtype=np.int32)

    # Vertical arm
    v_pts = np.array([
        [cx - w // 2, 0],
        [cx - w,      cy],
        [cx - w // 2, size - 1],
        [cx + w // 2, size - 1],
        [cx + w,      cy],
        [cx + w // 2, 0],
    ], dtype=np.int32)

    cv2.fillPoly(img, [h_pts], 255)
    cv2.fillPoly(img, [v_pts], 255)
    return img


def _star_polygon(size: int, n_points: int = 4,
                  inner_ratio: float = 0.15) -> np.ndarray:
    """
    Classic n-pointed star polygon.

    inner_ratio: radius of inner vertices relative to outer.
    0.10 = very spiky (like the Gemini icon), 0.50 = rounder.
    """
    img = np.zeros((size, size), dtype=np.uint8)
    cx = cy = size // 2
    r_out = size // 2 - 1
    r_in  = max(1, int(r_out * inner_ratio))

    pts = []
    for i in range(n_points * 2):
        angle = i * np.pi / n_points - np.pi / 2
        r = r_out if i % 2 == 0 else r_in
        pts.append([int(cx + r * np.cos(angle)),
                    int(cy + r * np.sin(angle))])

    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], 255)
    return img


def _nucleus(size: int, radius_ratio: float = 0.25) -> np.ndarray:
    """Small bright circle — the centre glow of the sparkle."""
    img = np.zeros((size, size), dtype=np.uint8)
    r   = max(2, int(size * radius_ratio))
    cv2.circle(img, (size // 2, size // 2), r, 255, -1)
    return img


def _cross_plus(size: int, arm_ratio: float = 0.20) -> np.ndarray:
    """Simple + / cross shape (wider arms than diamond_cross)."""
    img = np.zeros((size, size), dtype=np.uint8)
    w   = max(1, int(size * arm_ratio))
    cy = size // 2
    cx = size // 2
    cv2.rectangle(img, (0, cy - w), (size - 1, cy + w), 255, -1)
    cv2.rectangle(img, (cx - w, 0), (cx + w, size - 1), 255, -1)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_templates(min_size: int = 18,
                  max_size: int = 120,
                  step:     int = 4) -> list[np.ndarray]:
    """
    Return a list of grayscale template images at every size from
    min_size to max_size (step px apart).

    Each size generates 5 variants to improve match robustness:
      1. Spiky 4-point star  (inner_ratio=0.10)
      2. Medium 4-point star (inner_ratio=0.20)
      3. Diamond cross       (arm_ratio=0.16)
      4. Diamond cross wide  (arm_ratio=0.22)
      5. Small nucleus only  (centre glow)
    """
    templates: list[np.ndarray] = []

    for size in range(min_size, max_size + 1, step):
        templates.append(_star_polygon(size, inner_ratio=0.10))
        templates.append(_star_polygon(size, inner_ratio=0.20))
        templates.append(_diamond_cross(size, arm_ratio=0.16))
        templates.append(_diamond_cross(size, arm_ratio=0.22))
        templates.append(_nucleus(size, radius_ratio=0.22))

    return templates


def get_edge_templates(min_size: int = 18,
                       max_size: int = 120,
                       step:     int = 4) -> list[np.ndarray]:
    """
    Edge-map versions of the templates (Canny).
    Used for edge-based matching which is more robust when the
    logo has a gradient fill (colour varies but edges stay sharp).
    """
    base = get_templates(min_size, max_size, step)
    edge_templates: list[np.ndarray] = []
    for t in base:
        edges = cv2.Canny(t, 50, 150)
        if edges.any():
            edge_templates.append(edges)
    return edge_templates
