"""
Standalone diagnostic for depth-geometry shape distortion.

This script compares monocular estimated depth against real depth on a TUM-style
RGB-D sequence and reports whether estimated geometry tends to compress/stretch
axes (for example, making rectangular rooms look squarer).

It does not modify the main mapping pipeline.

Example:
    python test_depth_shape_distortion.py \
      --dataset data/rgbd_dataset_freiburg1_360 \
      --model-id depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf \
      --mono-depth-scale 0.9332 \
      --max-frames 150 --frame-step 5
"""

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from util import (
    associate_depth,
    depth_to_points,
    estimate_depth_from_rgb,
    interpolated_pose,
    load_groundtruth,
    load_tum_list,
    transform_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth shape distortion diagnostic")
    parser.add_argument("--dataset", type=str, default="data/rgbd_dataset_freiburg1_360")
    parser.add_argument(
        "--model-id",
        type=str,
        default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        help="Hugging Face depth model id",
    )
    parser.add_argument("--mono-depth-scale", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--max-assoc-dt", type=float, default=0.03)
    parser.add_argument("--depth-scale", type=float, default=5000.0)
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=4.0)
    parser.add_argument("--point-stride", type=int, default=4)
    parser.add_argument("--fx", type=float, default=517.3)
    parser.add_argument("--fy", type=float, default=516.5)
    parser.add_argument("--cx", type=float, default=318.6)
    parser.add_argument("--cy", type=float, default=255.3)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--outdir", type=str, default="output/depth_shape_diagnostic")
    return parser.parse_args()


def get_device_id(device: str) -> int:
    if device == "cpu":
        return -1
    if device == "cuda":
        return 0
    return 0 if torch.cuda.is_available() else -1


def extents_xyz(points: np.ndarray) -> Optional[np.ndarray]:
    if points is None or points.shape[0] == 0:
        return None
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return np.maximum(maxs - mins, 1e-9)


def safe_ratio(a: float, b: float) -> float:
    if b <= 1e-9:
        return float("nan")
    return float(a / b)


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rgb_txt = os.path.join(args.dataset, "rgb.txt")
    depth_txt = os.path.join(args.dataset, "depth.txt")
    gt_txt = os.path.join(args.dataset, "groundtruth.txt")

    if not os.path.exists(rgb_txt) or not os.path.exists(depth_txt):
        raise FileNotFoundError("Dataset must include rgb.txt and depth.txt")

    rgb_list = load_tum_list(rgb_txt)
    depth_list = load_tum_list(depth_txt)
    poses = load_groundtruth(gt_txt) if os.path.exists(gt_txt) else None

    print(f"Loading model: {args.model_id}")
    pipe = pipeline(
        task="depth-estimation",
        model=args.model_id,
        device=get_device_id(args.device),
        use_fast=True,
    )

    rows: List[Dict[str, float]] = []
    world_pts_gt_all: List[np.ndarray] = []
    world_pts_est_all: List[np.ndarray] = []

    processed = 0
    evaluated = 0

    for i, (ts, rgb_rel) in enumerate(rgb_list):
        if i % args.frame_step != 0:
            continue
        if args.max_frames is not None and processed >= args.max_frames:
            break
        processed += 1

        dep_item = associate_depth(ts, depth_list, max_dt=args.max_assoc_dt)
        if dep_item is None:
            continue

        _, depth_rel = dep_item
        rgb_path = os.path.join(args.dataset, rgb_rel)
        depth_path = os.path.join(args.dataset, depth_rel)

        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        gt_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if bgr is None or gt_raw is None:
            continue

        rgb_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        depth_gt_m = gt_raw.astype(np.float32) / args.depth_scale
        depth_est_m = estimate_depth_from_rgb(
            rgb_image=rgb_pil,
            pipe=pipe,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            scale_factor=args.mono_depth_scale,
        )

        if depth_est_m.shape != depth_gt_m.shape:
            depth_est_m = cv2.resize(
                depth_est_m,
                (depth_gt_m.shape[1], depth_gt_m.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Per-frame scalar scale estimate to diagnose residual global bias.
        valid = (
            np.isfinite(depth_gt_m)
            & np.isfinite(depth_est_m)
            & (depth_gt_m >= args.min_depth)
            & (depth_gt_m <= args.max_depth)
            & (depth_est_m > 1e-6)
        )
        if not np.any(valid):
            continue
        frame_scale = float(np.median(depth_gt_m[valid] / np.clip(depth_est_m[valid], 1e-6, None)))

        pts_gt_cam = depth_to_points(
            depth_gt_m,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            stride=args.point_stride,
            z_min=args.min_depth,
            z_max=args.max_depth,
        )
        pts_est_cam = depth_to_points(
            depth_est_m,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            stride=args.point_stride,
            z_min=args.min_depth,
            z_max=args.max_depth,
        )

        if pts_gt_cam.shape[0] == 0 or pts_est_cam.shape[0] == 0:
            continue

        ext_gt_cam = extents_xyz(pts_gt_cam)
        ext_est_cam = extents_xyz(pts_est_cam)

        ratio_xy_gt_cam = safe_ratio(ext_gt_cam[0], ext_gt_cam[1])
        ratio_xy_est_cam = safe_ratio(ext_est_cam[0], ext_est_cam[1])
        anisotropy_error_cam = float(abs(np.log((ratio_xy_est_cam + 1e-9) / (ratio_xy_gt_cam + 1e-9))))

        world_used = 0
        if poses:
            t_world, q_xyzw = interpolated_pose(ts, poses)
            pts_gt_world = transform_points(pts_gt_cam, t_world, q_xyzw)
            pts_est_world = transform_points(pts_est_cam, t_world, q_xyzw)
            world_pts_gt_all.append(pts_gt_world)
            world_pts_est_all.append(pts_est_world)
            world_used = 1

        rows.append(
            {
                "frame_idx": float(i),
                "timestamp": float(ts),
                "n_gt_pts": float(pts_gt_cam.shape[0]),
                "n_est_pts": float(pts_est_cam.shape[0]),
                "frame_scale_to_gt": frame_scale,
                "cam_extent_x_gt": float(ext_gt_cam[0]),
                "cam_extent_y_gt": float(ext_gt_cam[1]),
                "cam_extent_z_gt": float(ext_gt_cam[2]),
                "cam_extent_x_est": float(ext_est_cam[0]),
                "cam_extent_y_est": float(ext_est_cam[1]),
                "cam_extent_z_est": float(ext_est_cam[2]),
                "cam_ratio_xy_gt": ratio_xy_gt_cam,
                "cam_ratio_xy_est": ratio_xy_est_cam,
                "cam_anisotropy_error": anisotropy_error_cam,
                "has_world_pose": float(world_used),
            }
        )

        evaluated += 1
        if evaluated % 10 == 0:
            print(f"Evaluated {evaluated} frames...")

    if not rows:
        print("No valid frames evaluated.")
        return

    csv_path = os.path.join(args.outdir, "frame_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    frame_scales = np.array([r["frame_scale_to_gt"] for r in rows], dtype=np.float32)
    anis_err = np.array([r["cam_anisotropy_error"] for r in rows], dtype=np.float32)

    print("\n=== Per-frame Summary ===")
    print(f"Frames evaluated: {len(rows)}")
    print(f"Median frame scale-to-GT: {float(np.median(frame_scales)):.4f}")
    print(f"Mean frame scale-to-GT: {float(np.mean(frame_scales)):.4f}")
    print(f"Median cam anisotropy error: {float(np.median(anis_err)):.4f}")
    print(f"Mean cam anisotropy error: {float(np.mean(anis_err)):.4f}")
    print(f"Saved: {csv_path}")

    if world_pts_gt_all and world_pts_est_all:
        gt_world = np.concatenate(world_pts_gt_all, axis=0)
        est_world = np.concatenate(world_pts_est_all, axis=0)
        ext_gt_w = extents_xyz(gt_world)
        ext_est_w = extents_xyz(est_world)

        ratio_xy_gt_w = safe_ratio(ext_gt_w[0], ext_gt_w[1])
        ratio_xy_est_w = safe_ratio(ext_est_w[0], ext_est_w[1])
        ratio_xz_gt_w = safe_ratio(ext_gt_w[0], ext_gt_w[2])
        ratio_xz_est_w = safe_ratio(ext_est_w[0], ext_est_w[2])

        print("\n=== Global World Geometry Summary ===")
        print(f"GT extents xyz (m):  {ext_gt_w[0]:.3f}, {ext_gt_w[1]:.3f}, {ext_gt_w[2]:.3f}")
        print(f"EST extents xyz (m): {ext_est_w[0]:.3f}, {ext_est_w[1]:.3f}, {ext_est_w[2]:.3f}")
        print(f"GT xy ratio:  {ratio_xy_gt_w:.4f}")
        print(f"EST xy ratio: {ratio_xy_est_w:.4f}")
        print(f"GT xz ratio:  {ratio_xz_gt_w:.4f}")
        print(f"EST xz ratio: {ratio_xz_est_w:.4f}")

        summary_path = os.path.join(args.outdir, "global_summary.txt")
        with open(summary_path, "w") as f:
            f.write("Global world geometry summary\n")
            f.write(f"GT extents xyz (m): {ext_gt_w.tolist()}\n")
            f.write(f"EST extents xyz (m): {ext_est_w.tolist()}\n")
            f.write(f"GT xy ratio: {ratio_xy_gt_w}\n")
            f.write(f"EST xy ratio: {ratio_xy_est_w}\n")
            f.write(f"GT xz ratio: {ratio_xz_gt_w}\n")
            f.write(f"EST xz ratio: {ratio_xz_est_w}\n")

        print(f"Saved: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
