"""mapping.py

Helpers to convert RGB-D frames to a point cloud and accumulate a scene map.

use pyoctomap to create the map.
"""
import numpy as np
import cv2
import open3d as o3d
import pyoctomap as pom

# ------------------------------------------------------
# 1) Load RGB and depth image
# ------------------------------------------------------
rgb = cv2.imread("/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/data/rgb/5237818140983496476.jpg", cv2.IMREAD_COLOR)
depth_raw = cv2.imread("/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/data/depth/5237818140983496476.jpg_depth.png", cv2.IMREAD_UNCHANGED).astype(np.float32)

# print(f"Original dtype: {depth_raw.dtype}")
# print(f"Original shape: {depth_raw.shape}")
# print(f"Value range: [{depth_raw.min()}, {depth_raw.max()}]")
# print(f"Dimensions: {depth_raw.ndim}D")

if rgb is None or depth_raw is None:
    raise RuntimeError("Could not load rgb.png or depth.png")


print(f"Depth dtype: {depth_raw.dtype}, shape: {depth_raw.shape}, range: [{depth_raw.min()}, {depth_raw.max()}]")

if depth_raw.ndim == 3:
    depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)

# Normalize to [0, 1]
depth_norm = depth_raw.astype(np.float32) / 255.0

# Invert: closer objects should have SMALLER depth values
# depth_norm is [0,1] where 1=close, 0=far
# We want metric depth where small=close, large=far
inverse_depth = depth_norm + 1e-6  # avoid div by zero

# Convert to metric depth with scale calibration
# Adjust max_depth based on your scene (e.g., 10m for indoor, 50m for outdoor)
max_depth = 10.0  # meters - maximum scene depth
min_depth = 0.1   # meters - minimum valid depth

# Linear mapping: high inverse_depth (bright) → small metric depth (close)
depth_m = min_depth + (max_depth - min_depth) * (1.0 - inverse_depth)

# Clamp to valid range
depth_m = np.clip(depth_m, min_depth, max_depth)

print(f"Metric depth range: [{depth_m.min():.2f}, {depth_m.max():.2f}] meters")

# 3) Prepare images for Open3D
H, W = depth_m.shape
rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
depth_o3d = o3d.geometry.Image(depth_m)

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    rgb_o3d,
    depth_o3d,
    depth_scale=1.0,      # depth is already in "meters"
    depth_trunc=max_depth,    # max distance to keep; adapt to your scene
    convert_rgb_to_intensity=False
)


# ------------------------------------------------------
# 3) Generic pinhole intrinsics from image size
# ------------------------------------------------------
fx = fy = 0.8 * W
cx = W / 2.0
cy = H / 2.0

intr = o3d.camera.PinholeCameraIntrinsic(
    W, H, fx, fy, cx, cy
)
print(f"Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

# ------------------------------------------------------
# 4) Create point cloud from RGB-D
# ------------------------------------------------------
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)

# Don't flip coordinates - keep original orientation
# Open3D default: X right, Y down, Z forward (camera frame)
points = np.asarray(pcd.points)
print("Number of points:", points.shape[0])

# Find floor plane (lowest Z values = floor in camera frame)
z_vals = points[:, 2]
floor_z = np.percentile(z_vals, 5)  # Bottom 5% is likely floor
print(f"Detected floor at Z = {floor_z:.2f}m")

# Remove statistical outliers (optional but recommended)
pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"After outlier removal: {len(pcd_clean.points)} points")

# # o3d.visualization.draw_geometries([pcd])
# output_ply = "/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/output/pointcloud_3d_2.ply"
# o3d.io.write_point_cloud(output_ply, pcd)
# print(f"Saved point cloud to {output_ply}")

# Also save a downsampled version for faster viewing
pcd_downsampled = pcd_clean.voxel_down_sample(voxel_size=0.05)  # 5cm voxels
# output_ply_down = "/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/output/pointcloud_3d_downsampled_2.ply"
# o3d.io.write_point_cloud(output_ply_down, pcd_downsampled)
# print(f"Saved downsampled point cloud ({len(pcd_downsampled.points)} points) to {output_ply_down}")


# # Optional: estimate normals for better visualization
# pcd_downsampled.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
# )
# output_ply_normals = "/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/output/pointcloud_3d_with_normals.ply"
# o3d.io.write_point_cloud(output_ply_normals, pcd_downsampled)
# print(f"Saved with normals to {output_ply_normals}")
# # Downsample using Open3D voxel filter to reduce memory
# print(f"Downsampled to {len(pcd_downsampled.points)} points")
# # ------------------------------------------------------
# 5) Build OctoMap and insert point cloud
# ------------------------------------------------------
points = np.asarray(pcd_downsampled.points)
print(f"Using {points.shape[0]} cleaned points for octree")
tree = pom.OcTree(0.10)  # 10 cm resolution
origin = np.array([0.0, 0.0, 0.0], dtype=float)

# cloud = pom.Pointcloud()
# for x, y, z in points:
#     cloud.push_back(x, y, z)
batch_size = 10000  # Smaller batches
num_batches = (len(points) - 1) // batch_size + 1

print(f"Inserting {len(points)} points in {num_batches} batches...")
for i in range(0, len(points), batch_size):
    batch = points[i:i+batch_size]
    tree.insertPointCloud(batch, origin)
    print(f"  Batch {i//batch_size + 1}/{num_batches} ({100*(i+len(batch))/len(points):.1f}%)")

print("Point cloud insertion complete!")


# ------------------------------------------------------
# 6) Project to 2D occupancy map
# ------------------------------------------------------
# ------------------------------------------------------
# 6) Save OctoMap
# ------------------------------------------------------
tree.writeBinary("/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/trusted_depth_3d_map.bt")
print("Saved 3D OctoMap as trusted_depth_3d_map.bt")

# ------------------------------------------------------
# 7) Project to 2D occupancy map (if needed)
# ------------------------------------------------------
# # Note: projectDownTo2D may not be available in all pyoctomap builds
# # If it fails, just skip this step and use the 3D map
# try:
#     proj_map = pom.OccupancyGrid()
#     tree.projectDownTo2D(proj_map)
#     proj_map.writeBinary("/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/trusted_depth_2d_map.bt")
#     print("Saved 2D planar map as trusted_depth_2d_map.bt")
# except AttributeError:
#     print("projectDownTo2D not available in this pyoctomap build - using 3D map only")
# # ------------------------------------------------------
# Create 2D occupancy grid for navigation (slice at obstacle height)
print("\nCreating 2D occupancy grid from 3D octree...")

# Define height range relative to floor for obstacle detection
height_above_floor_min = 0.3  # meters above floor
height_above_floor_max = 2.5  # meters above floor

z_min = floor_z + height_above_floor_min
z_max = floor_z + height_above_floor_max

print(f"Slicing at {height_above_floor_min:.1f}-{height_above_floor_max:.1f}m above floor")
print(f"Absolute Z range: [{z_min:.2f}, {z_max:.2f}]m")

# Collect occupied voxels in height range
x_coords = []
y_coords = []

for leaf in tree.begin_leafs():
    if tree.isNodeOccupied(leaf):
        coord = leaf.getCoordinate()
        x, y, z = coord[0], coord[1], coord[2]
        
        # Only include voxels at navigation height (obstacles, not floor/ceiling)
        if z_min <= z <= z_max:
            x_coords.append(x)
            y_coords.append(y)

# Grid parameters
resolution_2d = 0.05  # 5cm grid cells for finer detail

if len(x_coords) > 0:
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    
    print(f"Found {len(x_coords)} occupied voxels at navigation height")
    
    # Create 2D grid
    x_min_grid, x_max_grid = x_coords.min(), x_coords.max()
    y_min_grid, y_max_grid = y_coords.min(), y_coords.max()
    
    grid_width = int((x_max_grid - x_min_grid) / resolution_2d) + 1
    grid_height = int((y_max_grid - y_min_grid) / resolution_2d) + 1
    
    print(f"Grid size: {grid_width} x {grid_height}")
    
    occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

    # Fill grid
    for x, y in zip(x_coords, y_coords):
        grid_x = int((x - x_min_grid) / resolution_2d)
        grid_y = int((y - y_min_grid) / resolution_2d)
        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            occupancy_grid[grid_y, grid_x] = 255  # white = occupied
    
    # Save as image (AFTER filling the entire grid)
    output_2d = "/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/output/occupancy_grid_2d.png"
    cv2.imwrite(output_2d, occupancy_grid)
    print(f"Saved 2D occupancy grid to {output_2d}")
    
    # Also save metadata
    output_2d_npy = "/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/output/occupancy_grid_2d.npy"
    np.save(output_2d_npy, {
        'grid': occupancy_grid,
        'resolution': resolution_2d,
        'x_min': x_min_grid,
        'y_min': y_min_grid,
        'x_max': x_max_grid,
        'y_max': y_max_grid,
        'z_range': (z_min, z_max)
    })
    print(f"Saved 2D grid metadata to {output_2d_npy}")
    
    # Visualize
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 10))
    plt.imshow(occupancy_grid, cmap='gray_r', origin='lower', interpolation='nearest')
    plt.colorbar(label='Occupancy (0=free, 255=occupied)')
    plt.xlabel(f'X (grid cells @ {resolution_2d}m resolution)')
    plt.ylabel(f'Y (grid cells @ {resolution_2d}m resolution)')
    plt.title(f'2D Occupancy Grid (Navigation Map)\nHeight: {height_above_floor_min:.1f}-{height_above_floor_max:.1f}m above floor | {len(x_coords)} obstacles')
    plt.tight_layout()
    output_plot = "/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/output/occupancy_grid_2d_plot.png"
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_plot}")
else:
    print("No occupied voxels found at navigation height!")

print("\nDone!")
