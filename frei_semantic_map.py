"""module frei_semantic_map.py

Build a semantic voxel map from the TUM Freiburg 1 360 RGB-D dataset using OWL-ViT detections.

Usage:
    python frei_semantic_map.py
This will process the dataset, build a semantic voxel map, and save it as a PLY file for visualization.
"""

from __future__ import annotations
import os
import cv2
import numpy as np
import pickle
from collections import defaultdict
from typing import List,Tuple
from detector import OwlDetectorbase32

from PIL import Image
from scipy.spatial.transform import Rotation as R, Slerp

# -----------------------------
# TUM Freiburg 1 intrinsics
# -----------------------------
fx, fy = 517.3, 516.5
cx, cy = 318.6, 255.3
depth_scale = 5000.0  # depth_raw / 5000 = meters

# -----------------------------
# Semantic voxel settings
# -----------------------------
RES = 0.05          # must match your octomap resolution
DEPTH_MIN = 0.2
DEPTH_MAX = 4.0
PIX_STRIDE = 4     # sample pixels inside bbox
MIN_SAMPLES = 80   # require enough valid depth samples
ASSOC_MAX_DT = 0.03 # 30ms max timestamp mismatch

# Reduce compute: run detector on every Nth RGB frame
RGB_FRAME_STEP = 5

# Detections
QUERIES = [
    "door", "window","glass","a glass window",
    "a transparent surface",
    "a glass wall",
    "a mirror",
    "a glass door",
]
DET_THRESHOLD = 0.35

DATASET_DIR = "data/rgbd_dataset_freiburg1_360"
RGB_TXT   = os.path.join(DATASET_DIR, "rgb.txt")
DEPTH_TXT = os.path.join(DATASET_DIR, "depth.txt")
GT_TXT    = os.path.join(DATASET_DIR, "groundtruth.txt")


def load_tum_list(path_txt: str) -> List[Tuple[float, str]]:
    items = []
    with open(path_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ts, rel = line.split()[:2]
            items.append((float(ts), rel))
    items.sort(key=lambda x: x[0])
    return items


def load_groundtruth(path_txt: str):
    gt = []
    with open(path_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tstamp, tx, ty, tz, qx, qy, qz, qw = map(float, line.split())
            gt.append((tstamp,
                       np.array([tx, ty, tz], dtype=np.float64),
                       np.array([qx, qy, qz, qw], dtype=np.float64)))
    gt.sort(key=lambda x: x[0])
    return gt


def interpolated_pose(ts: float, gt_list):
    times = np.array([g[0] for g in gt_list], dtype=np.float64)

    if ts <= times[0]:
        return gt_list[0][1], gt_list[0][2]
    if ts >= times[-1]:
        return gt_list[-1][1], gt_list[-1][2]

    idx = np.searchsorted(times, ts) - 1
    t0, p0, q0 = gt_list[idx]
    t1, p1, q1 = gt_list[idx + 1]

    a = (ts - t0) / (t1 - t0)
    p = (1 - a) * p0 + a * p1

    rots = R.from_quat([q0, q1])  # xyzw
    slerp = Slerp([t0, t1], rots)
    r = slerp([ts])[0]

    return p, r.as_quat()


def associate_depth(ts: float, depth_list: List[Tuple[float, str]]):
    times = np.array([t for t, _ in depth_list], dtype=np.float64)
    idx = np.searchsorted(times, ts)
    if idx <= 0:
        j = 0
    elif idx >= len(times):
        j = len(times) - 1
    else:
        j = idx - 1 if abs(times[idx-1]-ts) < abs(times[idx]-ts) else idx

    dt = abs(times[j] - ts)
    if dt > ASSOC_MAX_DT:
        return None
    return depth_list[j]


def bbox_points_cam(depth_m: np.ndarray, box, stride=PIX_STRIDE, fill_interior=True):
    """Extract 3D points from bounding box region. stride=1 fills entire region.
    If fill_interior=True, fills missing interior depth using boundary median."""
    x0, y0, x1, y1 = map(int, box)
    x0 = max(x0, 0); y0 = max(y0, 0)
    x1 = min(x1, depth_m.shape[1]-1); y1 = min(y1, depth_m.shape[0]-1)
    if x1 <= x0 or y1 <= y0:
        return None

    us = np.arange(x0, x1, stride)  # +1 to include boundary
    vs = np.arange(y0, y1, stride)  # +1 to include boundary
    if len(us) == 0 or len(vs) == 0:
        return None

    u_grid, v_grid = np.meshgrid(us, vs)
    z = depth_m[v_grid, u_grid].copy()

    valid = (z > DEPTH_MIN) & (z < DEPTH_MAX) & np.isfinite(z)
    
    if fill_interior and valid.sum() >= MIN_SAMPLES:
        # Fill invalid interior points with median depth from valid boundary points
        # Extract boundary pixels
        h, w = u_grid.shape
        boundary_mask = np.zeros_like(valid, dtype=bool)
        # Top and bottom rows
        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
        # Left and right columns
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True
        
        boundary_valid = valid & boundary_mask
        if boundary_valid.sum() > 0:
            median_depth = np.median(z[boundary_valid])
            # Fill invalid interior pixels with median boundary depth
            invalid = ~valid
            z[invalid] = median_depth
            # Update valid mask to include filled pixels
            valid = (z > DEPTH_MIN) & (z < DEPTH_MAX) & np.isfinite(z)
    
    if valid.sum() < MIN_SAMPLES:
        return None

    u = u_grid[valid].astype(np.float64)
    v = v_grid[valid].astype(np.float64)
    z = z[valid].astype(np.float64)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def cam_to_world(points_cam: np.ndarray, t_world: np.ndarray, q_xyzw: np.ndarray):
    rot = R.from_quat(q_xyzw)
    return rot.apply(points_cam) + t_world


def world_to_voxel_idx(p_world: np.ndarray, res: float = RES):
    return (int(np.floor(p_world[0] / res)),
            int(np.floor(p_world[1] / res)),
            int(np.floor(p_world[2] / res)))


def label_to_color(label: str):
    # stable-ish hash -> RGB
    h = abs(hash(label)) % (256**3)
    r = (h // (256**2)) % 256
    g = (h // 256) % 256
    b = h % 256
    return r, g, b


def export_semantic_ply(semantic_map, ply_out="semantic_voxels.ply", res: float = RES, min_weight: float = 2.0):
    pts = []
    cols = []
    labels = []

    for (vx, vy, vz), dist in semantic_map.items():
        label, weight = max(dist.items(), key=lambda kv: kv[1])
        if weight < min_weight:
            continue

        x = (vx + 0.5) * res
        y = (vy + 0.5) * res
        z = (vz + 0.5) * res
        r, g, b = label_to_color(label)

        pts.append((x, y, z))
        cols.append((r, g, b))
        labels.append(label)

    with open(ply_out, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(pts, cols):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")

    print("Wrote:", ply_out, "voxels:", len(pts))


def main():
    rgb_frames = load_tum_list(RGB_TXT)
    depth_frames = load_tum_list(DEPTH_TXT)
    gt_list = load_groundtruth(GT_TXT)

    detector = OwlDetectorbase32()

    # semantic voxel map: (vx,vy,vz) -> {label: weight}
    semantic = defaultdict(lambda: defaultdict(float))

    for i, (ts, rgb_rel) in enumerate(rgb_frames):
        if i % RGB_FRAME_STEP != 0:
            continue

        depth_item = associate_depth(ts, depth_frames)
        if depth_item is None:
            continue
        _, depth_rel = depth_item

        rgb_path = os.path.join(DATASET_DIR, rgb_rel)
        depth_path = os.path.join(DATASET_DIR, depth_rel)

        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            continue

        # PIL expects RGB
        rgb_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        detections = detector.detect(rgb_pil, QUERIES, threshold=DET_THRESHOLD)
        if not detections:
            continue

        depth_m = depth_raw.astype(np.float32) / depth_scale
        t_world, q_xyzw = interpolated_pose(ts, gt_list)

        # Fuse detections
        for det in detections:
            box = det["box"]
            score = float(det["score"])
            label = det["label"]

            pts_cam = bbox_points_cam(depth_m, box, stride=PIX_STRIDE, fill_interior=True)
            if pts_cam is None:
                continue

            pts_world = cam_to_world(pts_cam, t_world, q_xyzw)

            w = max(0.0, min(1.0, score))
            for p in pts_world:
                key = world_to_voxel_idx(p)
                semantic[key][label] += w

        if i % (RGB_FRAME_STEP * 50) == 0:
            print(f"Processed rgb idx {i}/{len(rgb_frames)} | semantic voxels: {len(semantic)}")

    # Save semantic layer
    semantic_out = "freiburg1_semantic_voxels.pkl"
    with open(semantic_out, "wb") as f:
        pickle.dump({k: dict(v) for k, v in semantic.items()}, f)
    print("Saved:", semantic_out, "voxels:", len(semantic))

    # Export PLY for visualization
    export_semantic_ply(semantic, ply_out="freiburg1_semantic_voxels.ply", min_weight=0.2)


if __name__ == "__main__":
    main()
