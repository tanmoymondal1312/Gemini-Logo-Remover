"""
app.py  –  Gradio web interface for Gemini Logo Remover

Run:
    python app.py          # opens browser at http://127.0.0.1:7860
    python cli.py --ui     # same, via CLI shortcut
"""

from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Processing bridge
# ──────────────────────────────────────────────────────────────────────────────

def _pil_to_bytes(pil_image: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def _bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def process_image(
    pil_image: Image.Image | None,
    method: str,
    feather: int,
    show_mask: bool,
):
    """Gradio callback – called on every 'Remove Logo' click."""
    if pil_image is None:
        return None, None, "⚠️  Please upload an image first."

    from core.remover import remove

    # Ensure RGB (Gradio may give RGBA)
    pil_image = pil_image.convert("RGB")

    raw_bytes = _pil_to_bytes(pil_image)

    result_bytes, mask_bytes = remove(
        raw_bytes,
        method=method,
        feather=feather,
        return_mask=True,
    )

    result_pil = _bytes_to_pil(result_bytes)
    mask_pil   = _bytes_to_pil(mask_bytes) if show_mask else None

    status = (
        f"✅  Done!  Method: **{method}** | "
        f"Feather: {feather}px | "
        f"Output size: {result_pil.size[0]}×{result_pil.size[1]}"
    )
    return result_pil, mask_pil, status


# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────

CSS = """
#title { text-align: center; }
#subtitle { text-align: center; color: #888; margin-top: -10px; }
.gr-button-primary { background: linear-gradient(135deg,#4f8ef7,#a855f7) !important; }
"""

DESCRIPTION = """
**Gemini Logo Remover** automatically finds the Gemini ✦ sparkle watermark in
the bottom-right corner of your image and reconstructs the background as if the
logo was never there — **100 % free, runs locally, no API keys, no charges.**
"""

METHOD_INFO = {
    "auto":  "🔄 Auto – tries LaMa first, falls back to TELEA",
    "lama":  "🧠 LaMa  – deep-learning, best quality  (requires simple-lama-inpainting)",
    "telea": "⚡ TELEA – OpenCV fast marching, great for small logos",
    "ns":    "🌊 NS    – Navier-Stokes, good for smooth/gradient backgrounds",
    "patch": "🧩 Patch – copies texture from surrounding pixels",
}


def launch_ui(share: bool = False) -> None:
    try:
        import gradio as gr
    except ImportError:
        print("[error] Gradio is not installed. Run:  pip install gradio")
        return

    with gr.Blocks(css=CSS, title="Gemini Logo Remover") as demo:

        gr.Markdown("# ✦ Gemini Logo Remover", elem_id="title")
        gr.Markdown(DESCRIPTION, elem_id="subtitle")

        with gr.Row():
            with gr.Column(scale=1):
                inp_image = gr.Image(
                    type="pil",
                    label="📤  Upload Image  (PNG / JPG / WEBP)",
                    height=380,
                )

                with gr.Accordion("⚙️  Settings", open=False):
                    method = gr.Radio(
                        choices=list(METHOD_INFO.keys()),
                        value="auto",
                        label="Inpainting Method",
                        info="auto is recommended for most images",
                    )
                    gr.Markdown(
                        "\n".join(f"- `{k}` – {v}"
                                  for k, v in METHOD_INFO.items())
                    )
                    feather = gr.Slider(
                        minimum=0, maximum=20, value=4, step=1,
                        label="Mask Feather (px)",
                        info="Higher = softer edges around removed area",
                    )
                    show_mask = gr.Checkbox(
                        value=False,
                        label="Show detection mask",
                    )

                btn = gr.Button("🗑️  Remove Logo", variant="primary")

            with gr.Column(scale=1):
                out_image = gr.Image(
                    type="pil",
                    label="✅  Cleaned Image",
                    height=380,
                )
                mask_image = gr.Image(
                    type="pil",
                    label="🔍  Detection Mask",
                    height=180,
                    visible=True,
                )
                status_box = gr.Markdown("_Status will appear here after processing._")

        # Wire up
        btn.click(
            fn=process_image,
            inputs=[inp_image, method, feather, show_mask],
            outputs=[out_image, mask_image, status_box],
        )

        gr.Markdown(
            "---\n"
            "**Tip:** For best results use `auto` method.  "
            "If the background is complex (photo/art), try `lama` for deep-learning quality.\n\n"
            "All processing happens **locally** on your machine — "
            "no image ever leaves your computer."
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
        show_error=True,
    )


if __name__ == "__main__":
    launch_ui()
