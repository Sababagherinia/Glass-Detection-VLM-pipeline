"""
Utility functions for RGB-D processing and mapping.

Reusable functions extracted from various pipeline modules for:
- TUM dataset loading
- Pose interpolation
- 3D point transformations
- Voxel mapping
- Semantic map export
"""

import os
import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R, Slerp
from PIL import Image


def load_tum_list(path_txt: str) -> List[Tuple[float, str]]:
    """
    Load TUM dataset list file (rgb.txt or depth.txt).
    
    Args:
        path_txt: Path to TUM list file
        
    Returns:
        List of (timestamp, relative_path) tuples, sorted by timestamp
    """
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
    items.sort(key=lambda x: x[0])
    return items


def load_groundtruth(path_txt: str) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """
    Load TUM groundtruth poses.
    
    Args:
        path_txt: Path to groundtruth.txt
        
    Returns:
        List of (timestamp, translation[3], quaternion_xyzw[4]) tuples, sorted by timestamp
    """
    gt = []
    with open(path_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tstamp, tx, ty, tz, qx, qy, qz, qw = map(float, line.split())
            gt.append((
                tstamp,
                np.array([tx, ty, tz], dtype=np.float64),
                np.array([qx, qy, qz, qw], dtype=np.float64)
            ))
    gt.sort(key=lambda x: x[0])
    return gt


def interpolated_pose(ts: float, gt_list: List[Tuple[float, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate camera pose at given timestamp using SLERP.
    
    Args:
        ts: Target timestamp
        gt_list: List of groundtruth poses from load_groundtruth()
        
    Returns:
        Tuple of (translation[3], quaternion_xyzw[4])
    """
    times = np.array([g[0] for g in gt_list], dtype=np.float64)

    # Handle boundary cases
    if ts <= times[0]:
        return gt_list[0][1], gt_list[0][2]
    if ts >= times[-1]:
        return gt_list[-1][1], gt_list[-1][2]

    # Find surrounding poses
    idx = np.searchsorted(times, ts) - 1
    t0, p0, q0 = gt_list[idx]
    t1, p1, q1 = gt_list[idx + 1]

    # Linear interpolation for translation
    alpha = (ts - t0) / (t1 - t0)
    p = (1 - alpha) * p0 + alpha * p1

    # SLERP for rotation
    rots = R.from_quat([q0, q1])  # xyzw format
    slerp = Slerp([t0, t1], rots)
    r = slerp([ts])[0]

    return p, r.as_quat()


def associate_depth(ts: float, depth_list: List[Tuple[float, str]], 
                   max_dt: float = 0.03) -> Optional[Tuple[float, str]]:
    """
    Find closest depth frame to RGB timestamp.
    
    Args:
        ts: RGB timestamp
        depth_list: List of depth frames from load_tum_list()
        max_dt: Maximum time difference in seconds
        
    Returns:
        (timestamp, relative_path) or None if no match within max_dt
    """
    times = np.array([t for t, _ in depth_list], dtype=np.float64)
    idx = np.searchsorted(times, ts)
    
    if idx <= 0:
        j = 0
    elif idx >= len(times):
        j = len(times) - 1
    else:
        j = idx - 1 if abs(times[idx-1] - ts) < abs(times[idx] - ts) else idx

    dt = abs(times[j] - ts)
    if dt > max_dt:
        return None
    return depth_list[j]


def depth_to_points(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                   stride: int = 4, z_min: float = 0.2, z_max: float = 5.0) -> np.ndarray:
    """
    Convert depth image to 3D point cloud in camera frame.
    
    Args:
        depth_m: Depth map in meters (H, W)
        fx, fy, cx, cy: Camera intrinsics
        stride: Sample every Nth pixel (for efficiency)
        z_min, z_max: Valid depth range
        
    Returns:
        Points in camera frame (N, 3) [x, y, z]
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


def transform_points(pts_cam: np.ndarray, t_world: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """
    Transform points from camera frame to world frame.
    
    Args:
        pts_cam: Points in camera frame (N, 3)
        t_world: Translation (3,)
        q_xyzw: Quaternion in xyzw format (4,)
        
    Returns:
        Points in world frame (N, 3)
    """
    rot = R.from_quat(q_xyzw)
    pts_world = rot.apply(pts_cam) + t_world
    return pts_world


def bbox_points_cam(depth_m: np.ndarray, box: Tuple[float, float, float, float],
                   fx: float, fy: float, cx: float, cy: float,
                   stride: int = 4, z_min: float = 0.2, z_max: float = 4.0,
                   fill_interior: bool = True, min_samples: int = 80) -> Optional[np.ndarray]:
    """
    Extract 3D points from bounding box region in depth map.
    
    Args:
        depth_m: Depth map in meters (H, W)
        box: Bounding box (x0, y0, x1, y1) in pixels
        fx, fy, cx, cy: Camera intrinsics
        stride: Sample every Nth pixel
        z_min, z_max: Valid depth range
        fill_interior: Fill missing interior depth with boundary median
        min_samples: Minimum number of valid points required
        
    Returns:
        Points in camera frame (N, 3) or None if insufficient samples
    """
    x0, y0, x1, y1 = map(int, box)
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, depth_m.shape[1] - 1)
    y1 = min(y1, depth_m.shape[0] - 1)
    
    if x1 <= x0 or y1 <= y0:
        return None

    us = np.arange(x0, x1, stride)
    vs = np.arange(y0, y1, stride)
    if len(us) == 0 or len(vs) == 0:
        return None

    u_grid, v_grid = np.meshgrid(us, vs)
    z = depth_m[v_grid, u_grid].copy()

    valid = (z > z_min) & (z < z_max) & np.isfinite(z)
    
    # Fill interior missing depth using boundary median
    if fill_interior and valid.sum() >= min_samples:
        h, w = u_grid.shape
        boundary_mask = np.zeros_like(valid, dtype=bool)
        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True
        
        boundary_valid = valid & boundary_mask
        if boundary_valid.sum() > 0:
            median_depth = np.median(z[boundary_valid])
            invalid = ~valid
            z[invalid] = median_depth
            valid = (z > z_min) & (z < z_max) & np.isfinite(z)
    
    if valid.sum() < min_samples:
        return None

    u = u_grid[valid].astype(np.float64)
    v = v_grid[valid].astype(np.float64)
    z = z[valid].astype(np.float64)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def world_to_voxel_idx(p_world: np.ndarray, resolution: float = 0.05) -> Tuple[int, int, int]:
    """
    Convert world coordinate to voxel grid index.
    
    Args:
        p_world: 3D point in world frame (3,)
        resolution: Voxel size in meters
        
    Returns:
        Voxel index (vx, vy, vz)
    """
    return (
        int(np.floor(p_world[0] / resolution)),
        int(np.floor(p_world[1] / resolution)),
        int(np.floor(p_world[2] / resolution))
    )


def label_to_color(label: str) -> Tuple[int, int, int]:
    """
    Generate consistent red variation color for a label using hash.
    Detected objects are colored in red variations.
    
    Args:
        label: Label string
        
    Returns:
        RGB tuple (r, g, b) in range [0, 255], with red variations
    """
    h = abs(hash(label)) % (256**2)  # Only vary G and B channels
    r = 255  # Keep red channel at maximum
    g = (h // 256) % 128  # Green: 0-127 (keep it lower to maintain red tint)
    b = h % 128  # Blue: 0-127 (keep it lower to maintain red tint)
    return r, g, b


def export_colored_points_to_ply(colored_points: dict, ply_path: str):
    """
    Export colored points dictionary to PLY file format.
    
    Args:
        colored_points: Dict mapping (x, y, z) -> (r, g, b)
        ply_path: Output PLY file path
    """
    if not colored_points:
        print(f"Warning: No colored points to export to {ply_path}")
        return
    
    # Write PLY file
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(colored_points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in colored_points.items():
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
    
    print(f"Exported colored PLY: {ply_path} ({len(colored_points)} points)")


def estimate_depth_from_rgb(rgb_image: Image.Image, pipe, 
                            min_depth: float = 0.3, max_depth: float = 5.0) -> np.ndarray:
    """
    Estimate depth from RGB image using Depth-Anything model.
    
    Args:
        rgb_image: PIL RGB image
        pipe: HuggingFace depth estimation pipeline
        min_depth: Minimum depth in meters
        max_depth: Maximum depth in meters
        
    Returns:
        Depth map in meters (H, W) as float32
    """
    # Run inference
    depth_output = pipe(rgb_image)["depth"]
    
    # Convert PIL depth image to numpy
    depth_raw = np.array(depth_output, dtype=np.float32)
    
    # Normalize and convert to metric depth
    # Depth-Anything outputs inverted depth (bright=close, dark=far)
    depth_norm = depth_raw / 255.0
    inverse_depth = depth_norm + 1e-6
    depth_m = min_depth + (max_depth - min_depth) * (1.0 - inverse_depth)
    
    return depth_m


def estimate_intrinsics_from_image(width: int, height: int) -> Tuple[float, float, float, float]:
    """
    Estimate camera intrinsics from image dimensions (rough approximation).
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Tuple of (fx, fy, cx, cy)
    """
    fx = fy = 0.8 * width  # Typical field of view assumption
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy
