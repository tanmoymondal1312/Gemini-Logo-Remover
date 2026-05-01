"""
inpainter.py  –  Inpainting strategies (all 100 % free)

Priority order (auto mode):
  1. LaMa        (simple-lama-inpainting)  – deep-learning, best quality
  2. Biharmonic  (scikit-image)             – no neural net, excellent smooth fill
  3. TELEA       (mirror-padded cv2)
     + Poisson   (cv2.seamlessClone)        – seamless edge blending
     + soft-alpha blend                     – feathered boundary
  4. Gradient    (polynomial 2-D fit)       – perfect for solid/blurred backgrounds
  5. NS          (mirror-padded cv2)        – Navier-Stokes smooth diffusion
  6. Patch       (custom texture copy)      – last-resort fallback

KEY FIX: all classical cv2.inpaint calls are wrapped with mirror-padding so the
algorithm has reflected context on the right/bottom image edges — this eliminates
the "draw bug" visible in the bottom-right corner.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Internal utilities
# ─────────────────────────────────────────────────────────────────────────────

def _mirror_pad(image_bgr: np.ndarray,
                mask: np.ndarray,
                pad: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Reflect-pad the image and zero-pad the mask.

    Mirror-reflection gives classical inpainting algorithms synthetic background
    pixels beyond all four image edges.  This is critical for corner/edge masks:
    without padding TELEA/NS have no reference pixels on two sides and produce
    streaky artifacts.
    """
    img_p  = cv2.copyMakeBorder(image_bgr, pad, pad, pad, pad,
                                cv2.BORDER_REFLECT_101)
    mask_p = cv2.copyMakeBorder(mask, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=0)
    return img_p, mask_p


def _crop_pad(result: np.ndarray, h: int, w: int, pad: int) -> np.ndarray:
    return result[pad: pad + h, pad: pad + w]


def _soft_blend(original: np.ndarray,
                inpainted: np.ndarray,
                mask: np.ndarray,
                blur_sigma: float = 14.0) -> np.ndarray:
    """
    Alpha-blend inpainted → original using a Gaussian-feathered mask.

    Inside the mask the result is 100 % inpainted; the alpha fades to 0 at the
    mask boundary so the seam is invisible.  Handles right/bottom image-edge
    cases gracefully because we only blend where the mask exists.
    """
    k    = max(3, int(blur_sigma * 6) | 1)          # odd, >= 3
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (k, k), blur_sigma)
    a    = soft[:, :, np.newaxis]
    out  = (inpainted.astype(np.float32) * a
            + original.astype(np.float32) * (1.0 - a))
    return out.clip(0, 255).astype(np.uint8)


def _poisson_blend(original: np.ndarray,
                   inpainted: np.ndarray,
                   mask: np.ndarray,
                   pad: int = 70) -> np.ndarray | None:
    """
    Poisson seamless clone with mirror-padding to handle corner/edge masks.

    cv2.seamlessClone requires the mask to be interior to the source image
    (not touching any edge).  Padding with 70 px of reflected content pushes
    the Gemini corner mask safely away from the boundaries.

    Returns None when blending fails (e.g. very large mask, OpenCV error).
    """
    h, w = original.shape[:2]
    orig_p, mask_p = _mirror_pad(original,   mask, pad)
    inp_p,  _      = _mirror_pad(inpainted,  mask, pad)

    ys, xs = np.where(mask_p > 0)
    if len(ys) == 0:
        return None

    cy = int((int(ys.min()) + int(ys.max())) // 2)
    cx = int((int(xs.min()) + int(xs.max())) // 2)

    try:
        result_p = cv2.seamlessClone(inp_p, orig_p, mask_p,
                                     (cx, cy), cv2.MIXED_CLONE)
        return _crop_pad(result_p, h, w, pad)
    except cv2.error:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Classical inpainting – mirror-padded wrappers
# ─────────────────────────────────────────────────────────────────────────────

def inpaint_telea(image_bgr: np.ndarray,
                  mask: np.ndarray,
                  radius: int = 6) -> np.ndarray:
    """
    Fast Marching Method (Telea 2004) with mirror-padding.

    The increased default radius (6 vs old 4) + padding produces much smoother
    fills for the bottom-right corner where two image edges meet.
    """
    h, w = image_bgr.shape[:2]
    pad  = max(70, radius * 12)
    img_p, mask_p = _mirror_pad(image_bgr, mask, pad)
    result_p = cv2.inpaint(img_p, mask_p, inpaintRadius=radius,
                           flags=cv2.INPAINT_TELEA)
    return _crop_pad(result_p, h, w, pad)


def inpaint_ns(image_bgr: np.ndarray,
               mask: np.ndarray,
               radius: int = 6) -> np.ndarray:
    """Navier-Stokes inpainting with mirror-padding — good for gradient backgrounds."""
    h, w = image_bgr.shape[:2]
    pad  = max(70, radius * 12)
    img_p, mask_p = _mirror_pad(image_bgr, mask, pad)
    result_p = cv2.inpaint(img_p, mask_p, inpaintRadius=radius,
                           flags=cv2.INPAINT_NS)
    return _crop_pad(result_p, h, w, pad)


# ─────────────────────────────────────────────────────────────────────────────
# Polynomial gradient fill
# ─────────────────────────────────────────────────────────────────────────────

def inpaint_gradient(image_bgr: np.ndarray,
                     mask: np.ndarray) -> np.ndarray:
    """
    Reconstruct background by fitting a 2-D degree-3 polynomial to background
    pixels (outside the mask) and predicting the masked area.

    Achieves near-perfect results for solid colours, linear gradients, radial
    gradients, and blurred backgrounds — the most common background types in
    AI-generated images where the Gemini logo appears.
    """
    result = image_bgr.copy()
    h, w   = image_bgr.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return result

    by, bx = np.where(mask == 0)
    if len(by) < 10:
        return result

    rng = np.random.RandomState(42)
    if len(by) > 8000:
        idx    = rng.choice(len(by), 8000, replace=False)
        by, bx = by[idx], bx[idx]

    ny = by.astype(np.float64) / max(h - 1, 1)
    nx = bx.astype(np.float64) / max(w - 1, 1)
    A_bg = _poly2d(nx, ny)

    mny = ys.astype(np.float64) / max(h - 1, 1)
    mnx = xs.astype(np.float64) / max(w - 1, 1)
    A_mk = _poly2d(mnx, mny)

    for i, ch in enumerate(cv2.split(image_bgr)):
        z      = ch[by, bx].astype(np.float64)
        coeffs, _, _, _ = np.linalg.lstsq(A_bg, z, rcond=None)
        pred   = A_mk @ coeffs
        ch_out = ch.copy().astype(np.float64)
        ch_out[ys, xs] = pred
        result[:, :, i] = np.clip(ch_out, 0, 255).astype(np.uint8)

    return result


def _poly2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Degree-3 polynomial basis: 1, x, y, x², xy, y², x³, x²y, xy², y³."""
    return np.column_stack([
        np.ones_like(x),
        x,     y,
        x**2,  x*y,   y**2,
        x**3,  x**2*y, x*y**2, y**3,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Biharmonic inpainting  (scikit-image – high quality, no neural net)
# ─────────────────────────────────────────────────────────────────────────────

def inpaint_biharmonic(image_bgr: np.ndarray,
                       mask: np.ndarray) -> np.ndarray | None:
    """
    Biharmonic (∇⁴ = 0) inpainting via scikit-image.

    Operates on a cropped ROI (masked region + 50 px context) so it stays fast
    even on high-resolution images.  Produces mathematically smooth fills with
    no halo or streak artifacts.  Returns None if scikit-image is not installed.
    """
    try:
        from skimage.restoration import inpaint_biharmonic as _bih  # type: ignore
    except ImportError:
        return None

    h, w   = image_bgr.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return image_bgr.copy()

    ctx  = 50
    y1 = max(0, int(ys.min()) - ctx)
    y2 = min(h, int(ys.max()) + ctx + 1)
    x1 = max(0, int(xs.min()) - ctx)
    x2 = min(w, int(xs.max()) + ctx + 1)

    roi_img  = image_bgr[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]

    rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    out = _bih(rgb, roi_mask > 0, channel_axis=-1)
    out = (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)

    result = image_bgr.copy()
    result[y1:y2, x1:x2] = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# LaMa deep-learning inpainting  (optional – best quality)
# ─────────────────────────────────────────────────────────────────────────────

_lama_model = None


def _load_lama():
    global _lama_model
    if _lama_model is None:
        from simple_lama_inpainting import SimpleLama   # type: ignore
        _lama_model = SimpleLama()
    return _lama_model


def inpaint_lama(image_bgr: np.ndarray,
                 mask: np.ndarray) -> np.ndarray | None:
    """LaMa large-mask deep inpainting.  Returns None if library not installed."""
    try:
        lama       = _load_lama()
        rgb        = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img    = Image.fromarray(rgb)
        pil_mask   = Image.fromarray(mask)
        result_pil = lama(pil_img, pil_mask)
        return cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Patch-based texture copy  (legacy fallback)
# ─────────────────────────────────────────────────────────────────────────────

def inpaint_patch(image_bgr: np.ndarray,
                  mask: np.ndarray) -> np.ndarray:
    """Copy texture from the region above the logo (last-resort fallback)."""
    result = image_bgr.copy()
    h, w   = image_bgr.shape[:2]

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return result

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    rh = y2 - y1 + 1
    rw = x2 - x1 + 1

    src_y1 = max(0, y1 - rh * 2)
    src_y2 = y1

    if src_y2 - src_y1 > 4:
        patch = image_bgr[src_y1:src_y2, x1: x2 + 1]
        reps  = (rh // patch.shape[0]) + 2
        tiled = np.tile(patch, (reps, 1, 1))
        fill  = tiled[-rh:, :rw]
    else:
        border = image_bgr[
            max(0, y1 - 30): min(h, y2 + 30),
            max(0, x1 - 30): min(w, x2 + 30),
        ].reshape(-1, 3)
        med  = np.median(border, axis=0).astype(np.uint8)
        fill = np.full((rh, rw, 3), med, dtype=np.uint8)

    roi_mask = mask[y1: y2 + 1, x1: x2 + 1]
    roi      = result[y1: y2 + 1, x1: x2 + 1]
    roi[roi_mask > 0] = fill[roi_mask > 0]
    result[y1: y2 + 1, x1: x2 + 1] = roi
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry-point
# ─────────────────────────────────────────────────────────────────────────────

METHODS = ("auto", "lama", "telea", "ns", "gradient", "biharmonic", "patch")


def inpaint(image_bgr: np.ndarray,
            mask: np.ndarray,
            method: str = "auto") -> tuple[np.ndarray, str]:
    """
    Remove the masked region from image_bgr using the best available method.

    Parameters
    ----------
    method : one of METHODS
        'auto'       → lama → biharmonic → telea+poisson+blend  (progressive)
        'lama'       → LaMa deep model   (best; needs simple-lama-inpainting)
        'telea'      → mirror-padded TELEA + Poisson + soft blend
        'ns'         → mirror-padded Navier-Stokes   + Poisson + soft blend
        'gradient'   → polynomial 2-D fit  (ideal for blurred/solid backgrounds)
        'biharmonic' → biharmonic PDE      (needs scikit-image)
        'patch'      → copy-from-above texture

    Returns
    -------
    (result_bgr, method_name_used)
    """
    if method == "auto":
        # 1. LaMa – deep learning, best quality
        result = inpaint_lama(image_bgr, mask)
        if result is not None:
            return result, "lama"

        # 2. Biharmonic – excellent, no neural net
        result = inpaint_biharmonic(image_bgr, mask)
        if result is not None:
            return _soft_blend(image_bgr, result, mask), "biharmonic"

        # 3. Mirror-padded TELEA + Poisson seamless clone + soft blend
        raw    = inpaint_telea(image_bgr, mask)
        result = _poisson_blend(image_bgr, raw, mask)
        if result is None:
            result = _soft_blend(image_bgr, raw, mask)
        return result, "telea+blend"

    if method == "lama":
        result = inpaint_lama(image_bgr, mask)
        if result is not None:
            return result, "lama"
        print("[warn] LaMa unavailable – falling back to auto chain")
        return inpaint(image_bgr, mask, "auto")

    if method == "telea":
        raw    = inpaint_telea(image_bgr, mask)
        result = _poisson_blend(image_bgr, raw, mask)
        if result is None:
            result = _soft_blend(image_bgr, raw, mask)
        return result, "telea+blend"

    if method == "ns":
        raw    = inpaint_ns(image_bgr, mask)
        result = _poisson_blend(image_bgr, raw, mask)
        if result is None:
            result = _soft_blend(image_bgr, raw, mask)
        return result, "ns+blend"

    if method == "gradient":
        result = inpaint_gradient(image_bgr, mask)
        return _soft_blend(image_bgr, result, mask), "gradient"

    if method == "biharmonic":
        result = inpaint_biharmonic(image_bgr, mask)
        if result is None:
            print("[warn] scikit-image unavailable – falling back to telea")
            return inpaint(image_bgr, mask, "telea")
        return _soft_blend(image_bgr, result, mask), "biharmonic"

    if method == "patch":
        return inpaint_patch(image_bgr, mask), "patch"

    raise ValueError(f"Unknown method '{method}'. Choose from {METHODS}")
