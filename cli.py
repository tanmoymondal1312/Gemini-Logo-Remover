#!/usr/bin/env python3
"""
cli.py  –  Command-line interface for Gemini Logo Remover

Examples
--------
    # Single file
    python cli.py image.png

    # Single file, pick method, save mask
    python cli.py photo.jpg --method telea --save-mask

    # Custom output path
    python cli.py input.png --output result.png

    # Batch (whole folder)
    python cli.py ./my_images/ --batch --output ./cleaned/

    # Gradio web UI
    python cli.py --ui
"""

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gemini-remove",
        description="Remove the Gemini watermark logo from AI-generated images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "input",
        nargs="?",
        help="Path to an image file or folder (for --batch).",
    )
    p.add_argument(
        "-o", "--output",
        default=None,
        help="Output file/folder path. Default: <input>_clean.<ext>",
    )
    p.add_argument(
        "-m", "--method",
        choices=("auto", "lama", "telea", "ns", "patch"),
        default="auto",
        help=(
            "Inpainting method:\n"
            "  auto   – try LaMa → TELEA  (default)\n"
            "  lama   – deep-learning inpainting (best quality)\n"
            "  telea  – OpenCV TELEA (fast, good quality)\n"
            "  ns     – OpenCV Navier-Stokes\n"
            "  patch  – texture copy from surroundings"
        ),
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Process all images inside the INPUT folder.",
    )
    p.add_argument(
        "--save-mask",
        action="store_true",
        help="Also save the detection mask (single-file mode only).",
    )
    p.add_argument(
        "--feather",
        type=int,
        default=4,
        metavar="N",
        help="Mask edge feather size in pixels (default: 4).",
    )
    p.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Gradio web interface instead of CLI mode.",
    )
    return p


def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Web UI mode ──────────────────────────────────────────────────────────
    if args.ui:
        from app import launch_ui
        launch_ui()
        return 0

    # ── Require input in CLI mode ────────────────────────────────────────────
    if not args.input:
        parser.print_help()
        return 1

    from core.remover import remove_file, remove_folder

    inp = Path(args.input)

    # ── Batch folder mode ────────────────────────────────────────────────────
    if args.batch:
        if not inp.is_dir():
            print(f"[error] --batch requires a directory, got: {inp}")
            return 1
        remove_folder(inp, args.output, method=args.method)
        return 0

    # ── Single file mode ─────────────────────────────────────────────────────
    if not inp.is_file():
        print(f"[error] File not found: {inp}")
        return 1

    remove_file(
        inp,
        args.output,
        method=args.method,
        feather=args.feather,
        save_mask=args.save_mask,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
