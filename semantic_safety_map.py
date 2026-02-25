"""
not applicable yet
semantic_safety_map.py

Builds a planar semantic danger map for a UAV using ONLY:
    - OWL-ViT detections (no depth required)

Usage:
    from semantic_safety_map import build_safety_map

    danger_grid, decision = build_safety_map(
        detections,
        image_width=1280,
        image_height=720
    )
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

CAMERA_HFOV_DEG = 90.0        # horizontal FOV of your camera (adjust if needed)

MAX_RANGE = 10.0              # forward range of the map (m)
MAX_WIDTH = 6.0               # left-right span (m)

GRID_X_CELLS = 100
GRID_Y_CELLS = 60

HAZARD_KEYWORDS = [
    "glass", "window", "mirror",
    "person", "human", "pedestrian",
    "tree", "branch", "trunk",
    "wall", "door", "building"
]

# -------------------------------------------------------
# Utility Functions
# -------------------------------------------------------

def is_hazard(label: str) -> bool:
    label = label.lower()
    return any(k in label for k in HAZARD_KEYWORDS)


def estimate_distance_band(h_px: int, image_height: int):
    frac = h_px / (image_height + 1e-6)

    if frac > 0.5:
        return 0.0, 3.0
    elif frac > 0.2:
        return 3.0, 6.0
    else:
        return 6.0, MAX_RANGE


def pixel_to_angle(u: float, image_width: int):
    hfov_rad = math.radians(CAMERA_HFOV_DEG)
    c_rel = (u / (image_width - 1)) * 2.0 - 1.0
    return c_rel * (hfov_rad / 2.0)


def world_to_grid(x: float, y: float):
    if x < 0 or x > MAX_RANGE:
        return None, None
    if y < -MAX_WIDTH/2 or y > MAX_WIDTH/2:
        return None, None

    ix = int((x / MAX_RANGE) * (GRID_X_CELLS - 1))
    iy = int(((y + MAX_WIDTH/2) / MAX_WIDTH) * (GRID_Y_CELLS - 1))
    return ix, iy


# -------------------------------------------------------
# Core Function
# -------------------------------------------------------

def build_safety_map(detections, image_width, image_height):
    """Builds a 2D semantic danger grid and returns (grid, decision)."""

    grid = np.zeros((GRID_X_CELLS, GRID_Y_CELLS), dtype=np.float32)

    for det in detections:
        x0, y0, x1, y1 = det["box"]
        score = float(det["score"])
        label = det["label"]

        if not is_hazard(label):
            continue

        h_px = (y1 - y0)
        d_min, d_max = estimate_distance_band(h_px, image_height)

        u_center = 0.5*(x0 + x1)
        theta_center = pixel_to_angle(u_center, image_width)

        angle_spread = math.radians(6.0)
        num_dist = 20
        num_ang = 9

        for i in range(num_dist):
            d = d_min + (d_max - d_min)*(i/(num_dist-1))

            for j in range(num_ang):
                frac = (j/(num_ang-1)) - 0.5
                theta = theta_center + frac*angle_spread

                x = d * math.cos(theta)
                y = d * math.sin(theta)

                ix, iy = world_to_grid(x, y)
                if ix is None:
                    continue

                w = max(0.1, min(1.0, score))
                grid[ix, iy] = min(1.0, grid[ix, iy] + 0.3*w)

    # Simple smoothing
    kernel = np.array([[0.05,0.1,0.05],[0.1,0.4,0.1],[0.05,0.1,0.05]])
    padded = np.pad(grid, 1, mode="edge")
    smoothed = grid.copy()

    for i in range(GRID_X_CELLS):
        for j in range(GRID_Y_CELLS):
            region = padded[i:i+3, j:j+3]
            smoothed[i, j] = np.sum(region * kernel)

    smoothed = np.clip(smoothed, 0.0, 1.0)

    # Decision logic
    close_band = int((3.0/MAX_RANGE)*(GRID_X_CELLS-1))
    mid_band   = int((6.0/MAX_RANGE)*(GRID_X_CELLS-1))

    if np.any(smoothed[:close_band] > 0.4):
        decision = "STOP"
    elif np.any(smoothed[:mid_band] > 0.2):
        decision = "CAUTIOUS"
    else:
        decision = "OK"

    return smoothed, decision


# -------------------------------------------------------
# Visualization
# -------------------------------------------------------

def plot_safety_map(grid, decision=""):
    xs = np.linspace(0, MAX_RANGE, GRID_X_CELLS)
    ys = np.linspace(-MAX_WIDTH/2, MAX_WIDTH/2, GRID_Y_CELLS)

    plt.figure(figsize=(6,6))
    plt.imshow(grid.T, origin="lower",
               extent=[xs[0], xs[-1], ys[0], ys[-1]],
               vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Semantic Danger Map ({decision})")
    plt.xlabel("Forward (m)")
    plt.ylabel("Lateral (m)")
    plt.show()
