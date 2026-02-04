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

# def inset_box(box, frac=0.08):
#     """Inset bbox by a fraction to reduce overfilling outside the actual window frame."""
#     x0, y0, x1, y1 = map(float, box)
#     w = max(1.0, x1 - x0)
#     h = max(1.0, y1 - y0)
#     dx = w * frac
#     dy = h * frac
#     return (x0 + dx, y0 + dy, x1 - dx, y1 - dy)

# def ray_dir_from_pixel(u: float, v: float):
#     # Ray direction in camera frame (origin at camera)
#     return np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)

# def intersect_ray_plane(n: np.ndarray, d: float, r: np.ndarray):
#     # plane: n·p + d = 0, ray: p(t)=t*r (camera origin)
#     denom = float(n @ r)
#     if abs(denom) < 1e-8:
#         return None
#     t = -d / denom
#     if t <= 0:
#         return None
#     return t * r

# def fit_plane_svd(P: np.ndarray):
#     """Plane fit via SVD. Returns unit normal n and offset d in n·p + d = 0."""
#     c = P.mean(axis=0)
#     X = P - c
#     _, _, Vt = np.linalg.svd(X, full_matrices=False)
#     n = Vt[-1]
#     n = n / (np.linalg.norm(n) + 1e-12)
#     d = -float(n @ c)
#     return n, d, c

# def sample_border_points_cam(depth_m: np.ndarray, box, border_px=6, stride_px=3):
#     """Backproject valid depth points from a bbox border band (fast, robust for windows)."""
#     x0, y0, x1, y1 = map(int, box)
#     x0 = max(0, x0); y0 = max(0, y0)
#     x1 = min(depth_m.shape[1]-1, x1); y1 = min(depth_m.shape[0]-1, y1)
#     if x1 <= x0 or y1 <= y0:
#         return None

#     b = max(1, int(border_px))
#     us = np.arange(x0, x1 + 1, stride_px)
#     vs = np.arange(y0, y1 + 1, stride_px)

#     coords = []
#     # top/bottom
#     for v in range(y0, min(y0 + b, y1 + 1)):
#         for u in us: coords.append((u, v))
#     for v in range(max(y1 - b + 1, y0), y1 + 1):
#         for u in us: coords.append((u, v))
#     # left/right
#     for u in range(x0, min(x0 + b, x1 + 1)):
#         for v in vs: coords.append((u, v))
#     for u in range(max(x1 - b + 1, x0), x1 + 1):
#         for v in vs: coords.append((u, v))

#     if not coords:
#         return None

#     coords = np.array(coords, dtype=np.int32)
#     u = coords[:, 0]
#     v = coords[:, 1]
#     z = depth_m[v, u]

#     valid = (z > DEPTH_MIN) & (z < DEPTH_MAX) & np.isfinite(z)
#     if valid.sum() < 40:  # need enough border depth
#         return None

#     u = u[valid].astype(np.float64)
#     v = v[valid].astype(np.float64)
#     z = z[valid].astype(np.float64)

#     x = (u - cx) * z / fx
#     y = (v - cy) * z / fy
#     return np.stack([x, y, z], axis=1)

# def fill_parallelogram(p00, p10, p01, step):
#     """Fill parallelogram in 3D: p(u,v)=p00 + u*(p10-p00) + v*(p01-p00)."""
#     a = p10 - p00
#     b = p01 - p00
#     la = float(np.linalg.norm(a))
#     lb = float(np.linalg.norm(b))
#     if la < 1e-6 or lb < 1e-6:
#         return None
#     na = max(1, int(np.ceil(la / step)))
#     nb = max(1, int(np.ceil(lb / step)))

#     pts = []
#     for i in range(na + 1):
#         ui = i / na
#         for j in range(nb + 1):
#             vj = j / nb
#             pts.append(p00 + ui * a + vj * b)
#     return np.asarray(pts, dtype=np.float64)

# def bbox_filled_on_plane_points_cam(depth_m: np.ndarray, box,
#                                    step=RES,
#                                    inset_frac=0.08,
#                                    border_px=6,
#                                    border_stride_px=3,
#                                    thickness=0.0,
#                                    thickness_layers=3):
#     """
#     Make interior-filled window points even when depth inside is missing:
#     1) sample valid depth on bbox border band
#     2) fit plane
#     3) intersect bbox corner rays with plane
#     4) fill rectangle on plane (fast grid at `step`)
#     5) optional thickness along plane normal
#     """
#     box2 = inset_box(box, frac=inset_frac)

#     border_pts = sample_border_points_cam(depth_m, box2, border_px=border_px, stride_px=border_stride_px)
#     if border_pts is None:
#         return None

#     n, d, c = fit_plane_svd(border_pts)

#     # Reject unstable planes (helps avoid scattered junk)
#     # If border depth varies too much, it’s likely not a real planar window surface.
#     if np.std(border_pts[:, 2]) > 0.25:
#         return None

#     x0, y0, x1, y1 = map(float, box2)

#     # intersect 3 rays for a parallelogram basis
#     p00 = intersect_ray_plane(n, d, ray_dir_from_pixel(x0, y0))
#     p10 = intersect_ray_plane(n, d, ray_dir_from_pixel(x1, y0))
#     p01 = intersect_ray_plane(n, d, ray_dir_from_pixel(x0, y1))
#     if p00 is None or p10 is None or p01 is None:
#         return None

#     pts = fill_parallelogram(p00, p10, p01, step=step)
#     if pts is None:
#         return None

#     if thickness and thickness > 0:
#         layers = max(1, int(thickness_layers))
#         offs = np.linspace(-thickness/2.0, thickness/2.0, layers)
#         pts = np.vstack([pts + o * n for o in offs])

#     return pts

# def fill_window_world_from_border(depth_m, box, t_world, q_xyzw,
#                                   step=RES, border_px=6, border_stride_px=3,
#                                   thickness=0.0, thickness_layers=3):
#     border_cam = sample_border_points_cam(depth_m, box, border_px=border_px, stride_px=border_stride_px)
#     if border_cam is None:
#         return None

#     border_world = cam_to_world(border_cam, t_world, q_xyzw)  # uses your existing convention

#     n, d, c = fit_plane_svd(border_world)

#     # Build plane basis (u,v) from normal
#     # pick a stable axis not parallel to n
#     a = np.array([1.0, 0.0, 0.0])
#     if abs(n @ a) > 0.9:
#         a = np.array([0.0, 1.0, 0.0])
#     u = np.cross(n, a); u /= (np.linalg.norm(u) + 1e-12)
#     v = np.cross(n, u); v /= (np.linalg.norm(v) + 1e-12)

#     # Project border points to plane coords to get extents
#     X = border_world - c
#     uv = np.stack([X @ u, X @ v], axis=1)
#     mn = uv.min(axis=0)
#     mx = uv.max(axis=0)

#     # Fill rectangle in plane coords
#     xs = np.arange(mn[0], mx[0] + step, step)
#     ys = np.arange(mn[1], mx[1] + step, step)

#     pts = []
#     for yy in ys:
#         for xx in xs:
#             pts.append(c + xx * u + yy * v)
#     pts = np.asarray(pts, dtype=np.float64)

#     if thickness and thickness > 0:
#         offs = np.linspace(-thickness/2.0, thickness/2.0, max(1, int(thickness_layers)))
#         pts = np.vstack([pts + o * n for o in offs])

#     return pts


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
