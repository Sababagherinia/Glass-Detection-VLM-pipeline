import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyoctomap as pom
import numpy as np

# Load the octree
tree = pom.OcTree(0.1)
success = tree.readBinary("/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/trusted_depth_3d_map.bt")

if not success:
    raise RuntimeError("Failed to read octree file")

print(f"Loaded octree with resolution {tree.getResolution()}")

# Extract occupied voxel centers
occupied_points = []
for leaf in tree.begin_leafs():
    if tree.isNodeOccupied(leaf):
        coord = leaf.getCoordinate()
        occupied_points.append(coord)

occupied_points = np.array(occupied_points)
print(f"Found {len(occupied_points)} occupied voxels")

if len(occupied_points) == 0:
    print("No occupied voxels to visualize")
else:
    # 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by height (z-coordinate)
    colors = occupied_points[:, 2]
    
    scatter = ax.scatter(
        occupied_points[:, 0],
        occupied_points[:, 1], 
        occupied_points[:, 2],
        c=colors,
        cmap='viridis',
        s=1,
        alpha=0.6
    )
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'OctoMap Visualization ({len(occupied_points)} voxels)')
    
    plt.colorbar(scatter, label='Height (m)')
    # plt.show()

    output_path = "/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/output/octree_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")