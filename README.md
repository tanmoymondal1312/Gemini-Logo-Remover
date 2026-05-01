# ✦ Gemini Logo Remover

A **100 % free, local, open-source** Python tool that automatically detects and
removes the **Gemini AI watermark** (the sparkle ✦ logo in the bottom-right
corner) from images — reconstructing the background so naturally that no one can
tell the logo was ever there.

> Built on the same philosophy as **rembg** — zero API keys, zero cloud fees,
> runs entirely on your machine.

---

## 🧠 How It Works (3-Layer Pipeline)

```
┌──────────────────────────────────────────────────────────────────┐
│  INPUT IMAGE                                                     │
│       ↓                                                          │
│  ① DETECT  ── HSV colour analysis (blue/cyan/purple sparkle)     │
│               + contour bounding-box                             │
│               + fallback: adaptive corner region                 │
│       ↓                                                          │
│  ② REFINE  ── morphological ops + Gaussian feathering            │
│               → clean binary mask                                │
│       ↓                                                          │
│  ③ INPAINT ── auto: LaMa (deep-learning) → TELEA (OpenCV)        │
│               result blends seamlessly with background           │
│       ↓                                                          │
│  OUTPUT IMAGE  (logo gone, background reconstructed)             │
└──────────────────────────────────────────────────────────────────┘
```

### Detection
The Gemini sparkle icon uses a **distinctive HSV colour signature**
(blue #4f8ef7, cyan, purple/violet gradient). We:
1. Crop the bottom-right 18 % × 22 % search window
2. Run HSV colour thresholding across 4 colour ranges
3. Close + dilate to group sparkle pixels into one blob
4. Extract the tightest bounding box + padding

### Inpainting Methods

| Method | Quality | Speed | Requirement |
|--------|---------|-------|-------------|
| `lama`  | ⭐⭐⭐⭐⭐ | Slow  | `pip install simple-lama-inpainting` |
| `telea` | ⭐⭐⭐⭐  | Fast  | Built-in (OpenCV) |
| `ns`    | ⭐⭐⭐   | Fast  | Built-in (OpenCV) |
| `patch` | ⭐⭐    | Fast  | Built-in (numpy)  |
| `auto`  | Best available | — | Default |

---

## 🚀 Quick Start

### 1. Install

```bash
git clone <this-repo>
cd gemini_logo_remover
pip install -r requirements.txt

# Optional – enable LaMa deep-learning inpainting (best quality):
pip install simple-lama-inpainting
```

### 2. Web UI (easiest)

```bash
python app.py
# Opens  http://127.0.0.1:7860  in your browser
```

### 3. Command Line

```bash
# Single image (auto method)
python cli.py photo.jpg

# Choose method + save detection mask
python cli.py photo.jpg --method telea --save-mask

# Custom output path
python cli.py input.png --output result.png

# Batch: whole folder
python cli.py ./my_images/ --batch --output ./cleaned/

# Launch web UI via CLI
python cli.py --ui
```

### 4. Python API  (like rembg)

```python
# ── Byte-level API ──────────────────────────────────────────────
from core import remove

with open("image.png", "rb") as f:
    clean_bytes = remove(f.read())

with open("clean.png", "wb") as f:
    f.write(clean_bytes)

# ── File API ────────────────────────────────────────────────────
from core import remove_file

remove_file("photo.jpg", "photo_clean.jpg", method="lama")

# ── NumPy array API ─────────────────────────────────────────────
import cv2
from core import remove_array

image = cv2.imread("photo.png")
result, mask = remove_array(image, method="telea")
cv2.imwrite("result.png", result)

# ── Batch folder API ────────────────────────────────────────────
from core import remove_folder

remove_folder("./input_dir/", "./output_dir/", method="auto")
```

---

## 📁 Project Structure

```
gemini_logo_remover/
├── core/
│   ├── __init__.py       # Public API: remove, remove_file, remove_array, remove_folder
│   ├── detector.py       # Logo detection (HSV + contours + fallback)
│   ├── inpainter.py      # Inpainting engines (LaMa / TELEA / NS / Patch)
│   └── remover.py        # Pipeline orchestration
├── app.py                # Gradio web UI
├── cli.py                # Command-line interface
├── requirements.txt
└── README.md
```

---

## 💡 Tips

- **Solid / gradient background?** → `telea` is fast and excellent.
- **Complex photo/art background?** → `lama` gives the best seamless result.
- **Transparent PNG?** → Convert to RGB first; inpainting does not support alpha.
- **Logo not detected?** → The fallback region will still cover the Gemini position.
  You can also increase `--feather` for a larger blend zone.

---

## 🆓 Cost

| Component | Cost |
|-----------|------|
| opencv-python | Free |
| numpy | Free |
| Pillow | Free |
| gradio | Free |
| simple-lama-inpainting | Free |
| **Total** | **$0 forever** |

No API keys. No cloud. No subscriptions. Everything runs on your CPU/GPU.

---

## 📜 License

MIT
