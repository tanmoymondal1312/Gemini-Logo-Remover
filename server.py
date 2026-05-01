"""
server.py  –  Production-grade FastAPI web server for Gemini Logo Remover

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
    # or for development:
    python server.py
"""

from __future__ import annotations

import io
import logging
import time
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from core.remover import remove as core_remove

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Gemini Logo Remover",
    description="Remove the Gemini ✦ watermark from images — free, local, no API keys.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MAX_FILE_BYTES = 30 * 1024 * 1024  # 30 MB
ALLOWED_TYPES  = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
ValidMethod    = Literal["auto", "lama", "telea", "ns", "gradient", "biharmonic", "patch"]


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/remove")
async def remove_logo(
    image: UploadFile = File(..., description="PNG / JPG / WEBP image"),
    method: ValidMethod = Form("auto"),
    feather: int = Form(4, ge=0, le=20),
    return_mask: bool = Form(False),
):
    """
    Remove the Gemini logo from an uploaded image.

    Returns the cleaned image as PNG (or JSON with base64 mask when return_mask=true).
    """
    # ── Validate content type ────────────────────────────────────────────────
    ct = (image.content_type or "").lower()
    if ct not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ct}'. Use PNG, JPG, or WEBP.",
        )

    # ── Read & size-check ────────────────────────────────────────────────────
    raw = await image.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw)//1024} KB). Max is 30 MB.",
        )

    # ── Validate it's actually an image ─────────────────────────────────────
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    # Re-encode as PNG for consistent processing (handles JPEG artefacts, EXIF, etc.)
    _, png_buf = cv2.imencode(".png", bgr)
    png_bytes = png_buf.tobytes()

    log.info(
        "Processing %s | %dx%d | method=%s feather=%d mask=%s",
        image.filename, bgr.shape[1], bgr.shape[0], method, feather, return_mask,
    )

    t0 = time.perf_counter()

    try:
        if return_mask:
            result_bytes, mask_bytes = core_remove(
                png_bytes, method=method, feather=feather, return_mask=True
            )
        else:
            result_bytes = core_remove(
                png_bytes, method=method, feather=feather, return_mask=False
            )
    except Exception as exc:
        log.exception("Processing failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - t0
    log.info("Done in %.2fs", elapsed)

    if return_mask:
        import base64
        from fastapi.responses import JSONResponse
        return JSONResponse({
            "result": base64.b64encode(result_bytes).decode(),
            "mask":   base64.b64encode(mask_bytes).decode(),
            "elapsed_ms": round(elapsed * 1000),
            "method": method,
        })

    return Response(
        content=result_bytes,
        media_type="image/png",
        headers={
            "X-Elapsed-Ms": str(round(elapsed * 1000)),
            "X-Method": method,
            "Content-Disposition": (
                f'attachment; filename="{_clean_name(image.filename)}"'
            ),
        },
    )


@app.post("/api/remove-with-mask")
async def remove_logo_with_mask(
    image: UploadFile = File(...),
    method: ValidMethod = Form("auto"),
    feather: int = Form(4, ge=0, le=20),
):
    """Same as /api/remove but always returns both result + mask as JSON+base64."""
    return await remove_logo(
        image=image, method=method, feather=feather, return_mask=True
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean_name(filename: str | None) -> str:
    if not filename:
        return "cleaned.png"
    stem = filename.rsplit(".", 1)[0]
    return f"{stem}_clean.png"


# ──────────────────────────────────────────────────────────────────────────────
# Dev entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True, workers=1)
