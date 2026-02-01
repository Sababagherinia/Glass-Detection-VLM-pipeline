"""visualize_3dmap.py
Visualize a 3D OctoMap from a .bt file using Open3D.
Generates color-coded point cloud based on distance from origin.

"""


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import pyoctomap as pom
# import numpy as np

# # Load the octree
# tree = pom.OcTree(0.1)
# success = tree.readBinary("output/trusted_depth_v2_3d_map.bt")

# if not success:
#     raise RuntimeError("Failed to read octree file")

# print(f"Loaded octree with resolution {tree.getResolution()}")

# # Extract occupied voxel centers and gather statistics
# total_nodes = 0
# occupied_points = []
# for leaf in tree.begin_leafs():
#     total_nodes += 1
#     if tree.isNodeOccupied(leaf):
#         coord = leaf.getCoordinate()
#         occupied_points.append(coord)

# occupied_points = np.array(occupied_points)
# print(f"\n=== Octree Statistics ===")
# print(f"Total leaf nodes: {total_nodes}")
# print(f"Occupied voxels: {len(occupied_points)}")
# print(f"Occupancy rate: {100*len(occupied_points)/total_nodes:.1f}%")

# if len(occupied_points) > 0:
#     print(f"\n=== Spatial Extent ===")
#     print(f"X range: [{occupied_points[:, 0].min():.2f}, {occupied_points[:, 0].max():.2f}] m")
#     print(f"Y range: [{occupied_points[:, 1].min():.2f}, {occupied_points[:, 1].max():.2f}] m")
#     print(f"Z range: [{occupied_points[:, 2].min():.2f}, {occupied_points[:, 2].max():.2f}] m")
    
#     # Check if coordinates look reasonable
#     x_span = occupied_points[:, 0].max() - occupied_points[:, 0].min()
#     y_span = occupied_points[:, 1].max() - occupied_points[:, 1].min()
#     z_span = occupied_points[:, 2].max() - occupied_points[:, 2].min()
#     print(f"\n=== Scene Dimensions ===")
#     print(f"Width (X): {x_span:.2f} m")
#     print(f"Depth (Y): {y_span:.2f} m")
#     print(f"Height (Z): {z_span:.2f} m")

# if len(occupied_points) == 0:
#     print("No occupied voxels to visualize")
# else:
#     # 3D scatter plot
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Color by height (z-coordinate)
#     colors = occupied_points[:, 2]
    
#     scatter = ax.scatter(
#         occupied_points[:, 0],
#         occupied_points[:, 1], 
#         occupied_points[:, 2],
#         c=colors,
#         cmap='viridis',
#         s=1,
#         alpha=0.6
#     )
    
#     ax.set_xlabel('X (m)')
#     ax.set_ylabel('Y (m)')
#     ax.set_zlabel('Z (m)')
#     ax.set_title(f'OctoMap 3D Visualization\n{len(occupied_points)} occupied voxels | Resolution: {tree.getResolution()}m')
    
#     # Set better viewing angle
#     ax.view_init(elev=20, azim=45)
    
#     plt.colorbar(scatter, label='Height (m)', shrink=0.8)
    
#     # Save multiple views
#     output_path = "/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/output/octree_visualization1.png"
#     plt.savefig(output_path, dpi=150, bbox_inches='tight')
#     print(f"\n=== Saved Visualizations ===")
#     print(f"Main view: {output_path}")
    
#     # Save top-down view
#     ax.view_init(elev=90, azim=0)
#     output_topdown = "/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/output/octree_topdown.png"
#     plt.savefig(output_topdown, dpi=150, bbox_inches='tight')
#     print(f"Top-down view: {output_topdown}")
    
#     # Save side view
#     ax.view_init(elev=0, azim=0)
#     output_side = "/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/output/octree_sideview.png"
#     plt.savefig(output_side, dpi=150, bbox_inches='tight')
#     print(f"Side view: {output_side}")
    
#     print("\n✓ Octree appears to be created correctly!")
#     print("  Check the visualizations to verify structure matches your scene.")

import pyoctomap as octomap
import open3d as o3d
import numpy as np

BT_FILE = "/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/freiburg1_room.bt"
OCC_THRESHOLD = 0.5  # occupancy probability

# Load OctoMap
tree = octomap.OcTree(BT_FILE.encode('utf-8'))

points = []

# Iterate over leaf nodes
for it in tree.begin_leafs():
    if tree.isNodeOccupied(it):
        coord = it.getCoordinate()
        points.append([coord[0], coord[1], coord[2]])

points = np.array(points)
print(f"Loaded {len(points)} occupied voxels")

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Optional: voxel visualization clarity
pcd = pcd.voxel_down_sample(voxel_size=tree.getResolution())

# Compute distance from origin for each point
points_downsampled = np.asarray(pcd.points)
distances = np.linalg.norm(points_downsampled, axis=1)

# Normalize distances to [0, 1] for color mapping
min_dist = distances.min()
max_dist = distances.max()
normalized_distances = (distances - min_dist) / (max_dist - min_dist)

# Create colormap: blue (close) -> green -> yellow -> red (far)
colors = np.zeros((len(normalized_distances), 3))
colors[:, 0] = normalized_distances  # Red channel increases with distance
colors[:, 1] = 1.0 - np.abs(normalized_distances - 0.5) * 2  # Green peaks at mid distance
colors[:, 2] = 1.0 - normalized_distances  # Blue channel decreases with distance

pcd.colors = o3d.utility.Vector3dVector(colors)

print(f"Distance range: {min_dist:.2f}m to {max_dist:.2f}m")

# Save to file instead of visualizing (WSL doesn't have display)
output_file = "freiburg1_room_visualization.ply"
o3d.io.write_point_cloud(output_file, pcd)
print(f"Saved point cloud with distance colors to {output_file}")
print(f"Color scheme: Blue (close) -> Green (mid) -> Red (far)")

# Try to visualize (will fail in WSL without X11, but works if display is available)
try:
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="OctoMap (.bt)",
        width=1000,
        height=800
    )
except Exception as e:
    print(f"Visualization not available (expected in WSL): {e}")
