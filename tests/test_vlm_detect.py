"""Small test harness to check CLIP (VLM) detection of glass-related prompts on an image.

Usage:
    python tests/test_vlm_detect.py --image pathtoimage.jpg

Optional flags:
    --device cuda|cpu (lets the VLM choose by default)
    --prompts "prompt1|prompt2|prompt3" (pipe-separated list)

The script prints similarity scores and writes an annotated copy of the image with the top label.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List
import pathlib as _pathlib

# Ensure project root is on sys.path so local modules (vlm.py) can be imported when
# running the test script from the repo root or another working directory.
_proj_root = _pathlib.Path(__file__).resolve().parents[1]
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    print("Pillow is required. Install with: pip install Pillow")
    raise

import numpy as np

try:
    from vlm import VLM
except Exception as e:  # pragma: no cover - helpful error for missing deps
    print("Could not import `vlm`. Make sure you installed dependencies from requirements.txt and that `vlm.py` exists.")
    print("Original error:", e)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None, help="torch device override, e.g. cpu or cuda")
    return p.parse_args()


def main():
    args = parse_args()

    # Hard-coded image path (user-provided file in repository)
    img_p = Path("data/rgb/glazen-binnendeuren-op-maat.jpg")
    if not img_p.exists():
        print(f"Image not found: {img_p}. Please place the image at this path or update the script.")
        sys.exit(1)

    # Hard-coded prompts focused on glass/transparent objects
    prompts = [
        "a glass window",
        "a transparent surface",
        "a glass bottle",
        "a mirror",
        "an open window",
        "a wooden door",
        "a concrete wall",
    ]

    # load image
    img = Image.open(img_p).convert("RGB")

    # instantiate VLM
    print("Loading VLM model (this may download weights if not cached)...")
    vlm = VLM(device=args.device)

    print("Computing image embedding...")
    img_emb = vlm.image_embedding(img)  # shape (1, D)

    print("Computing text embeddings...")
    txt_embs = vlm.text_embeddings(prompts)  # shape (N, D)

    sims = VLM.cosine_similarity_matrix(img_emb, txt_embs)[0]  # (N,)

    ranking = sorted(list(zip(prompts, sims.tolist())), key=lambda x: x[1], reverse=True)

    print("Scores (highest first):")
    for label, score in ranking:
        print(f"  {label:25s}  {float(score):.4f}")

    # annotate and save image with top label
    top_label, top_score = ranking[0]
    out_img = img.copy()
    draw = ImageDraw.Draw(out_img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text = f"Top: {top_label} ({top_score:.3f})"
    # draw a semi-transparent rectangle behind text for readability
    w, h = out_img.size
    margin = 8
    tw, th = draw.textsize(text, font=font)
    rect_h = th + margin * 2
    rect_w = tw + margin * 2
    rect_x0 = 10
    rect_y0 = 10
    rect_x1 = rect_x0 + rect_w
    rect_y1 = rect_y0 + rect_h
    draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=(0, 0, 0, 160))
    draw.text((rect_x0 + margin, rect_y0 + margin), text, fill=(255, 255, 255), font=font)

    out_path = img_p.parent / (img_p.stem + "_vlm_result.png")
    out_img.save(out_path)
    print(f"Annotated image saved to: {out_path}")


if __name__ == "__main__":
    main()
