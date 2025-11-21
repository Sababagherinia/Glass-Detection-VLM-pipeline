# flowchart ==> VLM detection → pixel bbox → depth map → project to 3D → insert into octree
                                                                            #   ↓
                                                                #  slice z range → 2D planar map

# visulalization
import matplotlib.pyplot as plt
import pyoctomap as p
import numpy as np
from pathlib import Path

# Create an octree with 0.1m resolution
tree = p.Octree(0.1)

def pixel_to_3d(u, v, depth, intrinsics):
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    z = depth[v, u]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])

# Load depth image
camera_origin = p.point3d(0,0,0)

# Assume depth is a 2D numpy array with depth values in meters
depth = np.load("depth_image.npy")  # shape (H, W)
for det in detections:
    bbox = det["bbox"]  # (x_min, y_min, x_max, y_max)
    x_min, y_min, x_max, y_max = map(int, bbox)

    for u in range(x_min, x_max):
        for v in range(y_min, y_max):
            pt_3d = pixel_to_3d(u, v, depth, intrinsics)
            pt = p.point3d(pt_3d[0], pt_3d[1], pt_3d[2])
            tree.insertRay(camera_origin, pt)
    
# Estimate depth for missing values
z_est = estimate_distance_from_surrounding(depth, det["bbox"])

# Fill in the bounding box area with estimated depth
for u in range(x_min, x_max):
    for v in range(y_min, y_max):
        pt_3d = pixel_to_3d(u, v, z_est, intrinsics)
        pt = p.point3d(*pt_3d)
        tree.updateNode(pt, True) # mark as occupied

# Visualize a slice of the octree at z=1.0
z_min = 0.9
z_max = 1.1
planar = []

# Create an iterator for leaf nodes within the bounding box
it = tree.begin_leafs_bbx(p.point3d(-20,-20,z_min), p.point3d(20,20,z_max))

# Iterate through the leaf nodes in the bounding box
for leaf in it:
    x = round(leaf.getX(),1)
    y = round(leaf.getY(),1)
    occ = leaf.getOccupancy()
    planar.append((x,y,occ))

xs = sorted(set([p[0] for p in planar.keys()]))
ys = sorted(set([p[1] for p in planar.keys()]))
occ_grid = np.zeros((len(ys), len(xs)))

for (x,y,occ) in planar:
    xi = xs.index(x)
    yi = ys.index(y)
    occ_grid[yi, xi] = occ > 0.5 # occupied if occupancy > 0.5