"""Batch OWL-ViT detector for a folder of images.

Usage:
  python batch_owlvit_detect.py --input data/rgb --output output/boxed
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from detector import OwlDetectorbase32


QUERIES = [
    "window",
    "glass",
    "a glass window",
    "a transparent surface",
    "a glass wall",
    "a glass door",
]


def annotate(image: Image.Image, detections):
    out = image.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        x0, y0, x1, y1 = map(int, det["box"])
        label = det.get("label", "object")
        score = det.get("score", 0.0)
        text = f"{label} {score:.2f}"

        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        if font is not None:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            pad = 3
            top = max(0, y0 - th - pad * 2)
            draw.rectangle([x0, top, x0 + tw + pad * 2, top + th + pad * 2], fill="red")
            draw.text((x0 + pad, top + pad), text, fill="white", font=font)

    return out


def main():
    parser = argparse.ArgumentParser(description="Run OWL-ViT over a folder and save boxed images")
    parser.add_argument("--input", required=True, help="Folder with input images")
    parser.add_argument("--output", default="output/boxed", help="Folder to write annotated images")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection threshold")
    parser.add_argument("--device", default=None, help="torch device override, e.g. cpu or cuda")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = OwlDetectorbase32(device=args.device)

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        detections = detector.detect(image, QUERIES, threshold=args.threshold)
        annotated = annotate(image, detections)
        out_path = output_dir / f"{img_path.stem}_boxed{img_path.suffix}"
        annotated.save(out_path)
        print(f"Saved {out_path} ({len(detections)} detections)")


if __name__ == "__main__":
    main()