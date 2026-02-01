"""
module 3d_octomap.py
Build a 3D occupancy map from the TUM Freiburg 1 360 RGB-D dataset using pyoctomap.
Usage:
    python 3d_octomap.py
    This will process the dataset, build a 3D occupancy map, and save it as a binary OctoMap file.
"""

import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

import pyoctomap

# ---------- TUM Freiburg1 intrinsics ----------
fx, fy = 517.3, 516.5
cx, cy = 318.6, 255.3
depth_scale = 5000.0  # depth_raw / 5000 = meters

# ---------- OctoMap params ----------
resolution = 0.05  # meters (5cm voxels)
max_range = 5.0    # optional clamp

# ---------- Paths ----------
DATASET_DIR = "data/rgbd_dataset_freiburg1_360"  # set to rgbd_dataset_freiburg1_room if running elsewhere
DEPTH_DIR = os.path.join(DATASET_DIR, "depth")
DEPTH_LIST = os.path.join(DATASET_DIR, "depth.txt")
GT_FILE = os.path.join(DATASET_DIR, "groundtruth.txt")

def load_tum_list(path_txt):
    """Returns list of (timestamp, relative_path). Skips comments."""
    items = []
    with open(path_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ts = float(parts[0])
            rel = parts[1]
            items.append((ts, rel))
    return items

def load_groundtruth(path_txt):
    """Returns list of (timestamp, t(3,), q(x,y,z,w))"""
    gt = []
    with open(path_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tstamp, tx, ty, tz, qx, qy, qz, qw = map(float, line.split())
            gt.append((tstamp, np.array([tx, ty, tz], dtype=np.float64),
                       np.array([qx, qy, qz, qw], dtype=np.float64)))
    return gt

def nearest_pose(ts, gt_list):
    """Pick closest timestamp pose (simple + effective for TUM)."""
    # gt_list is sorted
    idx = np.searchsorted([g[0] for g in gt_list], ts)
    if idx <= 0:
        return gt_list[0][1], gt_list[0][2]
    if idx >= len(gt_list):
        return gt_list[-1][1], gt_list[-1][2]
    before = gt_list[idx - 1]
    after = gt_list[idx]
    return (before[1], before[2]) if abs(before[0]-ts) < abs(after[0]-ts) else (after[1], after[2])

def depth_to_points(depth_m, stride=4, z_min=0.2, z_max=5.0):
    """
    Convert depth image (meters) -> Nx3 points in camera frame.
    Stride skips pixels to speed up.
    """
    h, w = depth_m.shape
    us = np.arange(0, w, stride)
    vs = np.arange(0, h, stride)
    u_grid, v_grid = np.meshgrid(us, vs)
    z = depth_m[v_grid, u_grid]

    valid = (z > z_min) & (z < z_max) & np.isfinite(z)
    u = u_grid[valid].astype(np.float64)
    v = v_grid[valid].astype(np.float64)
    z = z[valid].astype(np.float64)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    return pts

def transform_points(pts_cam, t_world, q_xyzw):
    """Camera frame points -> world frame using pose (world_T_cam)."""
    rot = R.from_quat(q_xyzw)         # xyzw
    pts_world = rot.apply(pts_cam) + t_world
    return pts_world


def interpolated_pose(ts, gt_list):
    times = np.array([g[0] for g in gt_list])

    if ts <= times[0]:
        return gt_list[0][1], gt_list[0][2]
    if ts >= times[-1]:
        return gt_list[-1][1], gt_list[-1][2]

    idx = np.searchsorted(times, ts) - 1

    t0, p0, q0 = gt_list[idx]
    t1, p1, q1 = gt_list[idx + 1]

    alpha = (ts - t0) / (t1 - t0)

    # Interpolate translation
    p = (1 - alpha) * p0 + alpha * p1

    # Interpolate rotation (SLERP)
    r0 = R.from_quat(q0)
    r1 = R.from_quat(q1)
    slerp = Slerp([t0, t1], R.from_quat([q0, q1]))
    r = slerp(ts)

    return p, r.as_quat()


def main():
    depth_list = load_tum_list(DEPTH_LIST)
    gt_list = load_groundtruth(GT_FILE)

    tree = pyoctomap.OcTree(resolution)
    tree.setClampingThresMin(0.12)
    tree.setClampingThresMax(0.97)

    # Optional: tighter probability model
    tree.setProbHit(0.7)
    tree.setProbMiss(0.4)

    for i, (ts, relpath) in enumerate(depth_list):
        # Process only every 5th frame
        if i % 5 != 0:
            continue
            
        depth_path = os.path.join(DATASET_DIR, relpath)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            print("Missing depth:", depth_path)
            continue

        depth_m = depth_raw.astype(np.float32) / depth_scale

        # t_world, q = nearest_pose(ts, gt_list)
        t_world, q = interpolated_pose(ts, gt_list)

        pts_cam = depth_to_points(depth_m, stride=4, z_min=0.2, z_max=max_range)
        if pts_cam.shape[0] == 0:
            continue

        pts_world = transform_points(pts_cam, t_world, q)

        # Sensor origin is the camera position in world
        origin = np.array([float(t_world[0]), float(t_world[1]), float(t_world[2])], dtype=float)

        # Insert points directly as numpy array
        tree.insertPointCloud(pts_world, origin, max_range, False)

        if i % 50 == 0:
            print(f"Inserted {i}/{len(depth_list)} frames, points={len(pts_world)}")

    tree.updateInnerOccupancy()

    out_bt = "freiburg1_room.bt"
    tree.writeBinary(out_bt)
    print("Saved:", out_bt)

if __name__ == "__main__":
    main()
