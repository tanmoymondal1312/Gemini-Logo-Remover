# ✦ Gemini Logo Remover

> Automatically removes the **Gemini AI watermark** from images and reconstructs the background so cleanly that no one can tell it was ever there.

**Free • Local • No API keys • No cloud • Runs on your machine**

---

## What it does

When you generate images with Google Gemini, a small sparkle logo `✦` appears in the bottom-right corner. This tool detects and removes it, then fills the background back — perfectly.

| Before | After |
|--------|-------|
| Image with Gemini logo in corner | Clean image, logo gone, background intact |

---

## Setup ( 3 commands )

```bash
# 1. Download the project
git clone https://github.com/tanmoymondal1312/Gemini-Logo-Remover.git
cd Gemini-Logo-Remover

# 2. Install all dependencies at once
pip install -r requirements.txt

# 3. Run the web app
python server.py
```

Then open your browser and go to → **http://localhost:8000**

That's it. Drag your image, click **Remove Logo**, download the result.

---

## Ways to use it

### Option A — Web App (recommended, easiest)

```bash
python server.py
```

Open **http://localhost:8000** in your browser.  
Drag & drop your image → pick a method → click **Remove Logo** → download.

---

### Option B — Command Line

```bash
# Remove logo from one image
python cli.py photo.jpg

# Specify output file
python cli.py photo.jpg --output clean_photo.jpg

# Process a whole folder at once
python cli.py ./my_images/ --batch --output ./cleaned/

# Also save the detection mask (see exactly what was removed)
python cli.py photo.jpg --save-mask
```

The cleaned image is saved as `photo_clean.jpg` by default.

---

### Option C — Python code

```python
from core import remove_file

# Remove logo and save result
remove_file("photo.jpg", "photo_clean.jpg")
```

```python
# More control
from core import remove_file

remove_file(
    "photo.jpg",
    "photo_clean.jpg",
    method="auto",   # see methods table below
    feather=4        # how soft the edges are (0–20)
)
```

```python
# Work with image bytes (useful in web apps / APIs)
from core import remove

with open("photo.jpg", "rb") as f:
    clean_bytes = remove(f.read())

with open("clean.jpg", "wb") as f:
    f.write(clean_bytes)
```

```python
# Work with OpenCV arrays
import cv2
from core import remove_array

image = cv2.imread("photo.jpg")
result, mask = remove_array(image)
cv2.imwrite("clean.jpg", result)
```

---

## Inpainting methods

The tool fills the removed logo area using one of these methods.  
**`auto` is the default** — it picks the best one available automatically.

| Method | Quality | Speed | When to use |
|--------|---------|-------|-------------|
| `auto` | Best available | — | Always start here (default) |
| `lama` | ⭐⭐⭐⭐⭐ | Slow | Best quality, complex backgrounds (needs extra install) |
| `biharmonic` | ⭐⭐⭐⭐ | Medium | Smooth fill, no extra install needed |
| `gradient` | ⭐⭐⭐⭐ | Fast | Solid colours, gradients, blurred backgrounds |
| `telea` | ⭐⭐⭐ | Fast | General purpose, always available |
| `ns` | ⭐⭐⭐ | Fast | Smooth gradient backgrounds |
| `patch` | ⭐⭐ | Fast | Last resort fallback |

### Want the best possible quality? Install LaMa:

```bash
pip install simple-lama-inpainting
```

Then use `method="lama"` or just leave it on `auto` — it will use LaMa automatically.  
*(Downloads a ~200 MB model file on first run, then it's cached forever.)*

---

## Project structure

```
Gemini-Logo-Remover/
│
├── core/
│   ├── detector.py     ← finds the Gemini logo in the image
│   ├── inpainter.py    ← fills the logo area with background
│   └── remover.py      ← connects detection + inpainting
│
├── server.py           ← web app  (python server.py)
├── cli.py              ← command line  (python cli.py photo.jpg)
├── app.py              ← Gradio UI alternative
├── static/
│   └── index.html      ← the web page served by server.py
├── requirements.txt    ← all dependencies in one file
└── README.md
```

---

## Requirements

- Python 3.10 or newer
- Works on Windows, macOS, Linux
- No GPU needed (CPU is fine)

---

## Common questions

**The logo was not detected — what do I do?**  
The tool falls back to a fixed corner region automatically, so it still works. Try increasing `--feather` for a larger blend zone: `python cli.py photo.jpg --feather 8`

**The background looks slightly off after removal?**  
Switch to a better method: `python cli.py photo.jpg --method gradient` (great for solid/gradient backgrounds) or install LaMa for the best results.

**Can I use this in my own Python project?**  
Yes. `from core import remove_file` — see the Python code examples above.

**Does this send my images anywhere?**  
No. Everything runs 100% locally on your machine. No internet required after setup.

---

## License

MIT — free to use, modify, and distribute.
