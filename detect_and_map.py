"""Detect glass regions with a VLM (CLIP) and produce a simple planar map (script version).

Workflow (single-image):
- Load RGB image (required). Optional depth image (aligned, same size) can be provided.
- Use OWL-ViT detector (optional) or split the image into a grid of patches and compute CLIP embeddings per patch.
- Compare embeddings to hard-coded glass-related text prompts to find glass patches or boxes.
- Produce a binary mask of detected glass regions and save an overlay image.
- If depth is provided: backproject masked pixels into 3D points (using provided intrinsics or sensible defaults) and
  either insert them into a pyoctomap octree (if pyoctomap is installed) or save a PLY point cloud as fallback.

This is a pragmatic, simple detection-to-map prototype for testing before integrating camera / SLAM data.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import warnings

import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageFilter  # NEW

# Ensure repo root on path for local imports
import pathlib as _pathlib
_proj_root = _pathlib.Path(__file__).resolve().parents[2]
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from vlm import VLM
try:
    from detector import OwlDetector
    OWL_AVAILABLE = True
except Exception:
    OwlDetector = None
    OWL_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser(description="Detect glass regions and create a planar map (single-image)")
    p.add_argument("--image", type=str, required=True, help="Path to RGB image")
    p.add_argument("--depth", type=str, default=None, help="Optional aligned depth image (uint16 mm or float meters)")
    p.add_argument("--device", type=str, default=None, help="torch device override, e.g. cpu or cuda")
    p.add_argument("--grid", type=int, default=8, help="Split image into grid x grid patches")
    p.add_argument("--use-detector", action="store_true", help="Use OWL-ViT open-vocabulary detector for bounding boxes (if available)")
    p.add_argument("--threshold", type=float, default=0.28, help="Similarity threshold for patch to be considered glass")
    p.add_argument("--out", type=str, default="output", help="Output directory")
    p.add_argument("--fx", type=float, default=None, help="Camera fx (optional)")
    p.add_argument("--fy", type=float, default=None, help="Camera fy (optional)")
    p.add_argument("--cx", type=float, default=None, help="Camera cx (optional)")
    p.add_argument("--cy", type=float, default=None, help="Camera cy (optional)")
    return p.parse_args()


GLASS_PROMPTS = [
    "a glass window",
    "a transparent surface",
    "a glass wall",
    "a mirror",
    "window",
]


def make_patches(img: Image.Image, grid: int):
    w, h = img.size
    gw = int(np.ceil(w / grid))
    gh = int(np.ceil(h / grid))
    patches = []
    boxes = []
    for r in range(grid):
        for c in range(grid):
            x0 = c * gw
            y0 = r * gh
            x1 = min(w, x0 + gw)
            y1 = min(h, y0 + gh)
            boxes.append((x0, y0, x1, y1))
            patches.append(img.crop((x0, y0, x1, y1)))
    return patches, boxes


def overlay_mask_on_image(img: Image.Image, mask: np.ndarray) -> Image.Image:
    # mask is HxW boolean
    overlay = img.convert("RGBA")
    red = Image.new("RGBA", img.size, (255, 0, 0, 90))  # slightly lower alpha
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
    overlay.paste(red, (0, 0), mask_img)
    return overlay

def draw_detections(image: Image.Image, detections, color=(0, 255, 0), width=3) -> Image.Image:
    """Draw OWL-ViT detection boxes and labels on top of image."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for d in detections or []:
        x0, y0, x1, y1 = list(map(int, d["box"]))[:4]
        label = d.get("label", "glass")
        score = d.get("score", 0.0)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        txt = f"{label}: {score:.2f}"
        # background for text for readability
        tw, th = draw.textlength(txt), 12
        draw.rectangle([x0, max(0, y0 - th - 2), x0 + tw + 6, y0], fill=(0, 0, 0, 160))
        draw.text((x0 + 3, y0 - th - 1), txt, fill=(255, 255, 255))
    return out

def scores_to_heatmap(best_sims: np.ndarray, grid: int, size: tuple[int, int]) -> np.ndarray:
    """
    Convert per-patch scores (length grid*grid, row-major) to a pixel heatmap HxW in [0,1]
    via bilinear upsampling to smooth blockiness.
    """
    w, h = size
    # arrange to (grid, grid)
    heat_grid = best_sims.reshape(grid, grid)
    # scale to [0,1] across present scores (robust min/max)
    vmin = float(np.percentile(heat_grid, 1))
    vmax = float(np.percentile(heat_grid, 99))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = (heat_grid - vmin) / (vmax - vmin)
    small = (norm * 255).astype(np.uint8)
    # upsample smoothly to image resolution
    heat_img = Image.fromarray(small, mode="L").resize((w, h), resample=Image.BILINEAR)
    heat = np.asarray(heat_img).astype(np.float32) / 255.0
    return heat

def morph_open_mask(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Light morphological opening using PIL Min/Max filters to reduce blockiness and
    break thin connections. ksize should be odd (3, 5).
    """
    if ksize < 3 or ksize % 2 == 0:
        return mask
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    eroded = img.filter(ImageFilter.MinFilter(ksize))
    opened = eroded.filter(ImageFilter.MaxFilter(ksize))
    return (np.asarray(opened) > 0).astype(np.uint8)

def main():
    args = parse_args()
    img_p = Path(args.image)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_p.exists():
        print(f"Image not found: {img_p}")
        return

    img = Image.open(img_p).convert("RGB")
    w, h = img.size

    vlm = VLM(device=args.device)
    txt_embs = vlm.text_embeddings(GLASS_PROMPTS)

    # Prefer OWL-ViT detector as the primary detector; fall back to patch-grid VLM if it's not available
    mask = np.zeros((h, w), dtype=np.uint8)
    used_detector = False
    dets = []
    if OWL_AVAILABLE:
        try:
            detector = OwlDetector(device=args.device)
            dets = detector.detect(img, GLASS_PROMPTS, threshold=args.threshold)
            print(f"OWL-ViT detector returned {len(dets)} detections")
            if len(dets) > 0:
                used_detector = True
                for d in dets:
                    x0, y0, x1, y1 = list(map(int, d["box"]))[:4]
                    # clamp
                    x0 = max(0, min(w - 1, x0))
                    x1 = max(0, min(w, x1))
                    y0 = max(0, min(h - 1, y0))
                    y1 = max(0, min(h, y1))
                    if x1 > x0 and y1 > y0:
                        mask[y0:y1, x0:x1] = 1
        except Exception as e:
            print("OWL-ViT detector failed — falling back to grid-based VLM:", e)

    # If detector not available or returned no detections, use grid-based VLM fallback
    if not used_detector:
        if args.use_detector and not OWL_AVAILABLE:
            print("OWL-ViT detector requested but not available (install transformers with OWL-ViT support). Using grid-based VLM.")
        grid = max(2, args.grid)  # ensure >=2 to allow smoothing
        patches, boxes = make_patches(img, grid)

        # Batch compute image embeddings for patches
        batch_size = 32
        img_embs_list = []
        for i in range(0, len(patches), batch_size):
            batch = patches[i : i + batch_size]
            inputs = vlm.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(vlm.device) for k, v in inputs.items()}
            with __import__("torch").no_grad():
                feats = vlm.model.get_image_features(**inputs).cpu().numpy()
            # normalize
            feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
            img_embs_list.append(feats)
        img_embs = np.vstack(img_embs_list)

        sims = img_embs @ txt_embs.T  # (num_patches, num_texts)
        best_sims = sims.max(axis=1)  # length grid*grid

        # Build a smooth heatmap and threshold it to a pixel-accurate mask
        heat = scores_to_heatmap(best_sims, grid, (w, h))  # [0,1]
        mask = (heat >= args.threshold).astype(np.uint8)

        # Light morphological opening to reduce blockiness and separate objects
        mask = morph_open_mask(mask, ksize=3)

    # Save overlay (for detections, also draw box outlines for clarity)
    overlay_base = overlay_mask_on_image(img, mask.astype(bool))
    if used_detector and len(dets) > 0:
        overlay = draw_detections(overlay_base.convert("RGB"), dets, color=(0, 255, 0), width=3)
    else:
        overlay = overlay_base
    overlay_path = out_dir / (img_p.stem + "_glass_overlay.png")
    overlay.save(overlay_path)
    print(f"Saved overlay image to: {overlay_path}")

    # Optional: save the continuous heatmap for analysis when using grid fallback
    if not used_detector:
        heat_vis = (heat * 255).astype(np.uint8)
        heatmap_path = out_dir / (img_p.stem + "_glass_heatmap.png")
        Image.fromarray(heat_vis, mode="L").save(heatmap_path)
        print(f"Saved glass heatmap to: {heatmap_path}")

    # Save raw mask
    mask_path = out_dir / (img_p.stem + "_glass_mask.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
    print(f"Saved mask to: {mask_path}")

    # If depth provided, backproject masked pixels into 3D
    if args.depth:
        depth_p = Path(args.depth)
        if not depth_p.exists():
            print(f"Depth image not found: {depth_p}")
            return
        depth_img = Image.open(depth_p)
        depth_np = np.asarray(depth_img)
        # convert to meters
        if depth_np.dtype == np.uint16:
            depth_m = depth_np.astype(np.float32) / 1000.0
        else:
            depth_m = depth_np.astype(np.float32)

        # intrinsics
        if args.fx and args.fy and args.cx is not None and args.cy is not None:
            fx = args.fx
            fy = args.fy
            cx = args.cx
            cy = args.cy
        else:
            # sensible defaults: assume 60 deg horizontal FOV
            fov = np.deg2rad(60.0)
            fx = fy = 0.5 * w / np.tan(0.5 * fov)
            cx = 0.5 * w
            cy = 0.5 * h
            warnings.warn("No intrinsics provided: using default focal estimate. Results will be approximate.")

        pts = backproject_mask_to_points(depth_m, mask.astype(bool), fx, fy, cx, cy)
        print(f"Backprojected {pts.shape[0]} points from masked depth")

        # Try pyoctomap integration
        try:
            import pyoctomap  # type: ignore

            print("pyoctomap available — inserting points into octree")
            tree = pyoctomap.OcTree(0.1)  # resolution 0.1m default
            # pyoctomap expects iterable of points; sensor origin at (0,0,0)
            for p in pts:
                tree.updateNode(p.tolist(), True)
            out_ot = out_dir / (img_p.stem + ".ot")
            tree.write(str(out_ot))
            print(f"Saved OctoMap to: {out_ot}")
        except Exception:
            print("pyoctomap not available — saving PLY point cloud as fallback")
            try:
                import open3d as o3d

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                out_ply = out_dir / (img_p.stem + "_glass_points.ply")
                o3d.io.write_point_cloud(str(out_ply), pcd)
                print(f"Saved PLY point cloud: {out_ply}")
            except Exception as e:
                print("Failed to save PLY — is open3d installed?", e)
    else:
        # No depth: produce a simple planar occupancy image as the 'map'
        occ = mask.astype(np.uint8) * 255
        occ_path = out_dir / (img_p.stem + "_planar_map.png")
        Image.fromarray(occ).save(occ_path)
        print(f"Saved planar occupancy map (pixel-space) to: {occ_path}")


if __name__ == "__main__":
    main()
