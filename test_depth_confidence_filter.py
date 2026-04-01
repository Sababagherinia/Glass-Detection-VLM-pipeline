"""
Standalone confidence-filter experiment for monocular depth.

This script estimates pseudo-confidence (no model-native uncertainty available)
from:
1) Flip consistency: original prediction vs horizontally-flipped prediction.
2) Temporal consistency: frame-to-frame consistency score.

It then compares depth metrics before and after confidence-based filtering.

Example:
    python test_depth_confidence_filter.py \
      --dataset data/rgbd_dataset_freiburg1_360 \
      --model-id depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf \
      --mono-depth-scale 0.9332 \
      --max-frames 150 --frame-step 5 --conf-threshold 0.55
"""

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from util import associate_depth, estimate_depth_from_rgb, load_tum_list


@dataclass
class RunningMetrics:
    n_valid: int = 0
    abs_err_sum: float = 0.0
    sq_err_sum: float = 0.0
    abs_rel_sum: float = 0.0
    d1_count: int = 0
    d2_count: int = 0
    d3_count: int = 0

    def update(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> None:
        p = pred[mask]
        g = gt[mask]
        if p.size == 0:
            return

        err = p - g
        abs_err = np.abs(err)
        self.n_valid += p.size
        self.abs_err_sum += float(np.sum(abs_err))
        self.sq_err_sum += float(np.sum(err ** 2))
        self.abs_rel_sum += float(np.sum(abs_err / np.clip(g, 1e-6, None)))

        ratio = np.maximum(
            np.clip(g, 1e-6, None) / np.clip(p, 1e-6, None),
            np.clip(p, 1e-6, None) / np.clip(g, 1e-6, None),
        )
        self.d1_count += int(np.sum(ratio < 1.25))
        self.d2_count += int(np.sum(ratio < 1.25 ** 2))
        self.d3_count += int(np.sum(ratio < 1.25 ** 3))

    def as_dict(self) -> Dict[str, float]:
        if self.n_valid == 0:
            return {
                "n_valid": 0,
                "mae": float("nan"),
                "rmse": float("nan"),
                "abs_rel": float("nan"),
                "delta1": float("nan"),
                "delta2": float("nan"),
                "delta3": float("nan"),
            }
        n = float(self.n_valid)
        return {
            "n_valid": self.n_valid,
            "mae": self.abs_err_sum / n,
            "rmse": float(np.sqrt(self.sq_err_sum / n)),
            "abs_rel": self.abs_rel_sum / n,
            "delta1": self.d1_count / n,
            "delta2": self.d2_count / n,
            "delta3": self.d3_count / n,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pseudo-confidence filtering for depth")
    parser.add_argument("--dataset", type=str, default="data/rgbd_dataset_freiburg1_360")
    parser.add_argument("--model-id", type=str, default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    parser.add_argument("--mono-depth-scale", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--max-assoc-dt", type=float, default=0.03)
    parser.add_argument("--depth-scale", type=float, default=5000.0)
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--flip-tau", type=float, default=0.15, help="Lower => stricter flip-consistency confidence")
    parser.add_argument("--temporal-tau", type=float, default=0.20, help="Lower => stricter temporal confidence")
    parser.add_argument("--conf-threshold", type=float, default=0.55)
    parser.add_argument("--save-vis-every", type=int, default=25)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--outdir", type=str, default="output/depth_confidence_filter")
    return parser.parse_args()


def get_device_id(device: str) -> int:
    if device == "cpu":
        return -1
    if device == "cuda":
        return 0
    return 0 if torch.cuda.is_available() else -1


def flip_consistency_conf(depth_est: np.ndarray, depth_est_flip_back: np.ndarray, tau: float) -> np.ndarray:
    rel = np.abs(depth_est - depth_est_flip_back) / np.clip(np.maximum(depth_est, depth_est_flip_back), 1e-6, None)
    return np.exp(-rel / max(tau, 1e-6)).astype(np.float32)


def temporal_consistency_score(curr: np.ndarray, prev: Optional[np.ndarray], tau: float) -> float:
    if prev is None:
        return 1.0
    valid = np.isfinite(curr) & np.isfinite(prev) & (curr > 1e-6) & (prev > 1e-6)
    if not np.any(valid):
        return 1.0

    # Align previous prediction to current by robust scalar.
    s = float(np.median(curr[valid] / np.clip(prev[valid], 1e-6, None)))
    prev_aligned = prev * s
    rel = np.abs(curr - prev_aligned) / np.clip(np.maximum(curr, prev_aligned), 1e-6, None)
    med_rel = float(np.median(rel[valid]))
    return float(np.exp(-med_rel / max(tau, 1e-6)))


def print_metrics(title: str, m: Dict[str, float]) -> None:
    print(f"\n{title}")
    print(f"  valid_pixels: {int(m['n_valid'])}")
    print(f"  MAE (m):      {m['mae']:.4f}")
    print(f"  RMSE (m):     {m['rmse']:.4f}")
    print(f"  AbsRel:       {m['abs_rel']:.4f}")
    print(f"  delta<1.25:   {m['delta1']:.4f}")
    print(f"  delta<1.25^2: {m['delta2']:.4f}")
    print(f"  delta<1.25^3: {m['delta3']:.4f}")


def save_conf_vis(conf: np.ndarray, path: str) -> None:
    img = np.clip(conf * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    cv2.imwrite(path, heat)


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    vis_dir = os.path.join(args.outdir, "confidence_vis")
    os.makedirs(vis_dir, exist_ok=True)

    rgb_txt = os.path.join(args.dataset, "rgb.txt")
    depth_txt = os.path.join(args.dataset, "depth.txt")
    if not os.path.exists(rgb_txt) or not os.path.exists(depth_txt):
        raise FileNotFoundError("Dataset must include rgb.txt and depth.txt")

    rgb_list = load_tum_list(rgb_txt)
    depth_list = load_tum_list(depth_txt)

    print(f"Loading model: {args.model_id}")
    pipe = pipeline("depth-estimation", model=args.model_id, device=get_device_id(args.device), use_fast=True)

    raw_metrics = RunningMetrics()
    filtered_metrics = RunningMetrics()

    rows = []
    prev_est = None
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

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_pil = Image.fromarray(rgb)
        rgb_flip_pil = Image.fromarray(np.ascontiguousarray(np.fliplr(rgb)))

        gt_m = gt_raw.astype(np.float32) / args.depth_scale
        est = estimate_depth_from_rgb(
            rgb_image=rgb_pil,
            pipe=pipe,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            scale_factor=args.mono_depth_scale,
        )
        est_flip = estimate_depth_from_rgb(
            rgb_image=rgb_flip_pil,
            pipe=pipe,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            scale_factor=args.mono_depth_scale,
        )

        if est.shape != gt_m.shape:
            est = cv2.resize(est, (gt_m.shape[1], gt_m.shape[0]), interpolation=cv2.INTER_LINEAR)
        if est_flip.shape != gt_m.shape:
            est_flip = cv2.resize(est_flip, (gt_m.shape[1], gt_m.shape[0]), interpolation=cv2.INTER_LINEAR)

        est_flip_back = np.ascontiguousarray(np.fliplr(est_flip))
        conf_flip = flip_consistency_conf(est, est_flip_back, args.flip_tau)
        conf_temp_scalar = temporal_consistency_score(est, prev_est, args.temporal_tau)
        conf = conf_flip * conf_temp_scalar

        valid = (
            np.isfinite(gt_m)
            & np.isfinite(est)
            & (gt_m >= args.min_depth)
            & (gt_m <= args.max_depth)
            & (est > 1e-6)
        )
        if not np.any(valid):
            prev_est = est.copy()
            continue

        keep = valid & (conf >= args.conf_threshold)

        raw_metrics.update(est, gt_m, valid)
        filtered_metrics.update(est, gt_m, keep)

        retain_ratio = float(np.sum(keep) / max(np.sum(valid), 1))
        rows.append(
            {
                "frame_idx": float(i),
                "timestamp": float(ts),
                "temporal_conf": float(conf_temp_scalar),
                "mean_flip_conf": float(np.mean(conf_flip[valid])),
                "mean_combined_conf": float(np.mean(conf[valid])),
                "retain_ratio": retain_ratio,
            }
        )

        if args.save_vis_every > 0 and evaluated % args.save_vis_every == 0:
            save_conf_vis(conf, os.path.join(vis_dir, f"conf_{i:05d}.png"))

        prev_est = est.copy()
        evaluated += 1
        if evaluated % 10 == 0:
            print(f"Evaluated {evaluated} frames...")

    if not rows:
        print("No valid frames evaluated.")
        return

    csv_path = os.path.join(args.outdir, "confidence_frame_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    raw = raw_metrics.as_dict()
    fil = filtered_metrics.as_dict()

    print("\n=== Confidence Filter Summary ===")
    print(f"Frames evaluated: {len(rows)}")
    print(f"Confidence threshold: {args.conf_threshold:.2f}")
    print(f"Mean retain ratio: {float(np.mean([r['retain_ratio'] for r in rows])):.4f}")
    print_metrics("Raw metrics (no confidence filtering)", raw)
    print_metrics("Filtered metrics (high-confidence pixels only)", fil)

    summary_path = os.path.join(args.outdir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Confidence filter summary\n")
        f.write(f"Frames evaluated: {len(rows)}\n")
        f.write(f"Confidence threshold: {args.conf_threshold}\n")
        f.write(f"Mean retain ratio: {float(np.mean([r['retain_ratio'] for r in rows]))}\n")
        f.write(f"Raw: {raw}\n")
        f.write(f"Filtered: {fil}\n")

    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved confidence maps in: {vis_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
