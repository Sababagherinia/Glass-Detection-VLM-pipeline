"""
vln-drone-pipline.py

Orchestrator script for a simple RGB-D -> VLM -> mapping pipeline.

Features:
- Read RGB-D frames (folder playback or RealSense if available)
- Run a CLIP-style VLM to score image regions/frames against text prompts
- Convert RGB-D to a point cloud and add to a map (pyoctomap if installed, else Open3D fallback)

This is a lightweight, portable skeleton. See README.md for setup and usage.
"""
import argparse
import time
from pathlib import Path
import numpy as np

from camera import RGBDSource
from vlm import VLM
from mapping import MapBuilder


def parse_args():
    p = argparse.ArgumentParser(description="RGB-D -> VLM -> mapping pipeline")
    p.add_argument("--mode", choices=["folder", "realsense"], default="folder",
                   help="Frame source mode: folder playback or realsense live")
    p.add_argument("--data-dir", type=str, default="data",
                   help="Folder with subfolders 'rgb' and 'depth' when using folder mode")
    p.add_argument("--out", type=str, default="output",
                   help="Output directory for saved maps and results")
    p.add_argument("--frames", type=int, default=50, help="Max frames to process")
    p.add_argument("--device", type=str, default=None, help="Torch device override, e.g. cpu or cuda")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing frame source...")
    src = RGBDSource(mode=args.mode, folder_path=args.data_dir)

    device = args.device
    if device is None:
        device = None  # let VLM choose (cuda if available)

    print("Loading VLM model (CLIP via transformers)...")
    vlm = VLM(device=device)

    print("Creating map builder (pyoctomap if available, else Open3D fallback)")
    mapper = MapBuilder(voxel_size=0.05)

    # Example textual queries relevant to glass/transparent object detection
    queries = ["a glass window", "a transparent surface", "a glass bottle", "a mirror", "an open window"]
    text_embs = vlm.text_embeddings(queries)

    print("Starting main loop. Press Ctrl-C to stop.")
    processed = 0
    try:
        for rgb, depth, meta in src.frame_iter():
            t0 = time.time()
            # compute a frame-level image embedding
            img_emb = vlm.image_embedding(rgb)
            # compute cosine similarity with the query texts
            sims = VLM.cosine_similarity_matrix(img_emb, text_embs)[0]
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            best_label = queries[best_idx]

            print(f"Frame {processed:04d}: best match='{best_label}' score={best_score:.3f}")

            # add RGB-D frame to the map
            mapper.add_frame(rgb, depth, meta.get("intrinsics", None))

            processed += 1
            if processed >= args.frames:
                break

            dt = time.time() - t0
            # throttle a bit for folder playback readability
            if args.mode == "folder":
                time.sleep(min(0.1, max(0.01, 0.05)))

    except KeyboardInterrupt:
        print("Interrupted by user — finishing up...")

    # finalize and save the map
    map_path = out_dir / "map.ply"
    print(f"Saving final map to {map_path}")
    mapper.save_map(map_path)

    print("Done.")


if __name__ == "__main__":
    main()
