"""
remover.py  –  High-level API  (mirrors how rembg exposes remove())

Quick usage
-----------
    from core.remover import remove

    with open("photo.png", "rb") as f:
        result_bytes = remove(f.read())

    with open("photo_clean.png", "wb") as f:
        f.write(result_bytes)
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .detector import detect, refine_mask
from .inpainter import inpaint, METHODS


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bytes_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes. "
                         "Make sure the input is a valid PNG/JPG/WEBP.")
    return img


def _bgr_to_bytes(image_bgr: np.ndarray, fmt: str = ".png") -> bytes:
    success, buf = cv2.imencode(fmt, image_bgr)
    if not success:
        raise RuntimeError("Image encoding failed.")
    return buf.tobytes()


def _path_to_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return img


# ──────────────────────────────────────────────────────────────────────────────
# rembg-style byte-level API
# ──────────────────────────────────────────────────────────────────────────────

def remove(
    data: bytes,
    *,
    method: str = "auto",
    feather: int = 4,
    return_mask: bool = False,
) -> bytes | tuple[bytes, bytes]:
    """
    Remove the Gemini logo from raw image bytes.

    Parameters
    ----------
    data         : raw image bytes (PNG / JPG / WEBP)
    method       : inpainting method – one of ('auto','lama','telea','ns','patch')
    feather      : mask edge softness (pixels)
    return_mask  : if True, also return the mask as PNG bytes

    Returns
    -------
    result_bytes           – cleaned image as PNG bytes
    (result_bytes, mask_bytes) – if return_mask=True
    """
    image = _bytes_to_bgr(data)
    result, mask = _process(image, method=method, feather=feather)
    result_bytes = _bgr_to_bytes(result)
    if return_mask:
        return result_bytes, _bgr_to_bytes(mask)
    return result_bytes


# ──────────────────────────────────────────────────────────────────────────────
# File-level API
# ──────────────────────────────────────────────────────────────────────────────

def remove_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    method: str = "auto",
    feather: int = 4,
    save_mask: bool = False,
) -> Path:
    """
    Remove Gemini logo from a file on disk.

    Parameters
    ----------
    input_path  : source image path
    output_path : destination path (default: <name>_clean.<ext>)
    method      : inpainting method
    feather     : mask edge softness
    save_mask   : also save the detection mask alongside the result

    Returns
    -------
    Path to the saved clean image.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / (input_path.stem + "_clean"
                                           + input_path.suffix)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = _path_to_bgr(input_path)
    result, mask = _process(image, method=method, feather=feather)

    cv2.imwrite(str(output_path), result)

    if save_mask:
        mask_path = output_path.parent / (output_path.stem + "_mask.png")
        cv2.imwrite(str(mask_path), mask)
        print(f"Mask saved → {mask_path}")

    print(f"Clean image saved → {output_path}")
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# NumPy array API
# ──────────────────────────────────────────────────────────────────────────────

def remove_array(
    image_bgr: np.ndarray,
    *,
    method: str = "auto",
    feather: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove Gemini logo from a BGR numpy array.

    Returns
    -------
    (result_bgr, mask)
    """
    return _process(image_bgr, method=method, feather=feather)


# ──────────────────────────────────────────────────────────────────────────────
# Core processing pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _process(
    image_bgr: np.ndarray,
    method: str,
    feather: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Internal pipeline: detect → refine → inpaint."""
    # 1. Detect
    raw_mask, auto = detect(image_bgr)
    mode = "auto-detected" if auto else "fallback corner"
    print(f"[detect]   Logo region {mode}")

    # 2. Refine mask edges
    mask = refine_mask(raw_mask, feather=feather)

    # 3. Inpaint
    result, method_used = inpaint(image_bgr, mask, method=method)
    print(f"[inpaint]  Method used: {method_used}")

    return result, mask


# ──────────────────────────────────────────────────────────────────────────────
# Batch helper
# ──────────────────────────────────────────────────────────────────────────────

def remove_folder(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    method: str = "auto",
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> list[Path]:
    """
    Process every image in *input_dir* and save results to *output_dir*.

    Returns list of output paths.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir / "cleaned"
    output_dir.mkdir(parents=True, exist_ok=True)

    files   = [p for p in input_dir.iterdir()
               if p.suffix.lower() in extensions]
    results = []

    print(f"Found {len(files)} image(s) in {input_dir}")
    for i, fp in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {fp.name}")
        out = remove_file(fp, output_dir / fp.name, method=method)
        results.append(out)

    print(f"\n✓ Done. {len(results)} image(s) cleaned → {output_dir}")
    return results
