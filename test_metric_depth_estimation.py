"""
Evaluate metric depth estimation quality on a TUM-style RGB-D dataset.

This script helps decide whether a metric depth model is good enough to
integrate into the mapping pipeline by comparing predicted depth against
ground-truth depth maps.

Example:
    python test_metric_depth_estimation.py \
      --dataset data/rgbd_dataset_freiburg1_360 \
      --model-id depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf \
      --max-frames 100 --frame-step 5
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from util import associate_depth, load_tum_list


@dataclass
class RunningMetrics:
    """Accumulates depth metrics across all valid pixels."""

    n_valid: int = 0
    abs_err_sum: float = 0.0
    sq_err_sum: float = 0.0
    abs_rel_sum: float = 0.0
    d1_count: int = 0
    d2_count: int = 0
    d3_count: int = 0

    def update(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> None:
        pred_v = pred[mask]
        gt_v = gt[mask]

        if pred_v.size == 0:
            return

        err = pred_v - gt_v
        abs_err = np.abs(err)

        self.n_valid += pred_v.size
        self.abs_err_sum += float(np.sum(abs_err))
        self.sq_err_sum += float(np.sum(err ** 2))
        self.abs_rel_sum += float(np.sum(abs_err / np.clip(gt_v, 1e-6, None)))

        ratio = np.maximum(
            np.clip(gt_v, 1e-6, None) / np.clip(pred_v, 1e-6, None),
            np.clip(pred_v, 1e-6, None) / np.clip(gt_v, 1e-6, None),
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
            "rmse": np.sqrt(self.sq_err_sum / n),
            "abs_rel": self.abs_rel_sum / n,
            "delta1": self.d1_count / n,
            "delta2": self.d2_count / n,
            "delta3": self.d3_count / n,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test metric depth estimation on TUM data")

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/rgbd_dataset_freiburg1_360",
        help="Path to TUM dataset directory containing rgb.txt and depth.txt",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        help="Hugging Face model id to evaluate",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum RGB frames to process",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help="Process every Nth frame",
    )
    parser.add_argument(
        "--max-assoc-dt",
        type=float,
        default=0.03,
        help="Max RGB-depth timestamp association gap (seconds)",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=5000.0,
        help="Convert raw depth png to meters via depth_m = raw / depth_scale",
    )
    parser.add_argument("--min-depth", type=float, default=0.2, help="Min valid GT depth (m)")
    parser.add_argument("--max-depth", type=float, default=5.0, help="Max valid GT depth (m)")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device",
    )
    parser.add_argument(
        "--compare-relative",
        action="store_true",
        help="Also evaluate relative model for side-by-side comparison",
    )
    parser.add_argument(
        "--relative-model-id",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Relative depth model id used when --compare-relative is enabled",
    )

    return parser.parse_args()


def get_device_id(device: str) -> int:
    if device == "cpu":
        return -1
    if device == "cuda":
        return 0
    return 0 if torch.cuda.is_available() else -1


def load_depth_prediction(pipe, rgb_pil: Image.Image) -> np.ndarray:
    """Run HF depth pipeline and return float32 depth array."""
    out = pipe(rgb_pil)

    if "predicted_depth" in out:
        pred = out["predicted_depth"]
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        else:
            pred = np.array(pred)
        if pred.ndim == 3:
            pred = pred[0]
    else:
        pred = np.array(out["depth"])

    return pred.astype(np.float32)


def prepare_paths(dataset: str) -> Tuple[str, str]:
    rgb_txt = os.path.join(dataset, "rgb.txt")
    depth_txt = os.path.join(dataset, "depth.txt")

    if not os.path.exists(rgb_txt):
        raise FileNotFoundError(f"Missing file: {rgb_txt}")
    if not os.path.exists(depth_txt):
        raise FileNotFoundError(f"Missing file: {depth_txt}")

    return rgb_txt, depth_txt


def evaluate_model(
    model_name: str,
    model_id: str,
    dataset: str,
    rgb_list,
    depth_list,
    args: argparse.Namespace,
) -> Optional[Dict[str, Dict[str, float]]]:
    print(f"\n[{model_name}] Loading model: {model_id}")

    try:
        pipe = pipeline(
            task="depth-estimation",
            model=model_id,
            device=get_device_id(args.device),
            use_fast=True,
        )
    except Exception as exc:
        print(f"[{model_name}] Failed to load model: {exc}")
        print("Hint: verify model id and Hugging Face access.")
        return None

    raw_metrics = RunningMetrics()
    aligned_metrics = RunningMetrics()
    scale_values = []

    processed = 0
    evaluated = 0

    for i, (ts, rgb_rel) in enumerate(rgb_list):
        if i % args.frame_step != 0:
            continue
        if args.max_frames is not None and processed >= args.max_frames:
            break

        processed += 1

        depth_item = associate_depth(ts, depth_list, args.max_assoc_dt)
        if depth_item is None:
            continue

        _, depth_rel = depth_item
        rgb_path = os.path.join(dataset, rgb_rel)
        depth_path = os.path.join(dataset, depth_rel)

        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        gt_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if bgr is None or gt_raw is None:
            continue

        gt_m = gt_raw.astype(np.float32) / args.depth_scale
        rgb_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        pred = load_depth_prediction(pipe, rgb_pil)
        if pred.shape != gt_m.shape:
            pred = cv2.resize(pred, (gt_m.shape[1], gt_m.shape[0]), interpolation=cv2.INTER_LINEAR)

        valid = (
            np.isfinite(gt_m)
            & np.isfinite(pred)
            & (gt_m >= args.min_depth)
            & (gt_m <= args.max_depth)
            & (pred > 1e-6)
        )
        if not np.any(valid):
            continue

        raw_metrics.update(pred, gt_m, valid)

        scale = float(np.median(gt_m[valid] / np.clip(pred[valid], 1e-6, None)))
        pred_aligned = pred * scale
        aligned_metrics.update(pred_aligned, gt_m, valid)
        scale_values.append(scale)

        evaluated += 1
        if evaluated % 10 == 0:
            print(f"[{model_name}] Evaluated {evaluated} frames...")

    if evaluated == 0:
        print(f"[{model_name}] No valid frames were evaluated.")
        return None

    raw = raw_metrics.as_dict()
    aligned = aligned_metrics.as_dict()
    median_scale = float(np.median(np.array(scale_values, dtype=np.float32)))

    print(f"[{model_name}] Frames used: {evaluated}")
    print(f"[{model_name}] Median per-frame scale-to-GT: {median_scale:.4f}")

    return {
        "raw": raw,
        "aligned": aligned,
        "frames": {"count": float(evaluated), "median_scale": median_scale},
    }


def print_metrics(title: str, m: Dict[str, float]) -> None:
    print(f"\n{title}")
    print(f"  valid_pixels: {int(m['n_valid'])}")
    print(f"  MAE (m):      {m['mae']:.4f}")
    print(f"  RMSE (m):     {m['rmse']:.4f}")
    print(f"  AbsRel:       {m['abs_rel']:.4f}")
    print(f"  delta<1.25:   {m['delta1']:.4f}")
    print(f"  delta<1.25^2: {m['delta2']:.4f}")
    print(f"  delta<1.25^3: {m['delta3']:.4f}")


def main() -> None:
    args = parse_args()

    rgb_txt, depth_txt = prepare_paths(args.dataset)
    rgb_list = load_tum_list(rgb_txt)
    depth_list = load_tum_list(depth_txt)

    metric_result = evaluate_model(
        model_name="METRIC",
        model_id=args.model_id,
        dataset=args.dataset,
        rgb_list=rgb_list,
        depth_list=depth_list,
        args=args,
    )

    if metric_result is not None:
        print("\n=== Metric Model Summary ===")
        print(f"Frames evaluated: {int(metric_result['frames']['count'])}")
        print(f"Median scale-to-GT: {metric_result['frames']['median_scale']:.4f} (ideal metric is near 1.0)")
        print_metrics("Raw prediction metrics", metric_result["raw"])
        print_metrics("Scale-aligned metrics", metric_result["aligned"])

    if args.compare_relative:
        relative_result = evaluate_model(
            model_name="RELATIVE",
            model_id=args.relative_model_id,
            dataset=args.dataset,
            rgb_list=rgb_list,
            depth_list=depth_list,
            args=args,
        )

        if relative_result is not None:
            print("\n=== Relative Model Summary ===")
            print(f"Frames evaluated: {int(relative_result['frames']['count'])}")
            print(f"Median scale-to-GT: {relative_result['frames']['median_scale']:.4f}")
            print_metrics("Raw prediction metrics", relative_result["raw"])
            print_metrics("Scale-aligned metrics", relative_result["aligned"])

    print("\nDone.")


if __name__ == "__main__":
    main()
