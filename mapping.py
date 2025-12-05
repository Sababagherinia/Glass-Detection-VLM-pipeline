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
depth_raw = cv2.imread("/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/data/depth/5237818140983496476.jpg_depth.png", cv2.IMREAD_UNCHANGED)

if rgb is None or depth_raw is None:
    raise RuntimeError("Could not load rgb.png or depth.png")

# Make sure they have the same size
H, W = depth_raw.shape[:2]
rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)

# Convert BGR -> RGB for Open3D
rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))


# ------------------------------------------------------
# 2) Use depth as-is
# ------------------------------------------------------
# If it's not already float32, convert it.
depth_m = depth_raw.astype(np.float32)

# Optionally, you can mask invalid values (e.g. 0) but don't rescale:
depth_m[depth_m <= 0] = 0.0   # Open3D treats 0 as "no depth"

depth_o3d = o3d.geometry.Image(depth_m)

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    rgb_o3d,
    depth_o3d,
    depth_scale=1.0,      # depth is already in "meters"
    depth_trunc=100.0,    # max distance to keep; adapt to your scene
    convert_rgb_to_intensity=False
)


# ------------------------------------------------------
# 3) Generic pinhole intrinsics from image size
# ------------------------------------------------------
fx = fy = 0.5 * W
cx = W / 2.0
cy = H / 2.0

intr = o3d.camera.PinholeCameraIntrinsic(
    W, H, fx, fy, cx, cy
)


# ------------------------------------------------------
# 4) Create point cloud from RGB-D
# ------------------------------------------------------
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)

# Optional: flip to a more conventional coordinate frame
pcd.transform([[1, 0, 0, 0],
               [0,-1, 0, 0],
               [0, 0,-1, 0],
               [0, 0, 0, 1]])

points = np.asarray(pcd.points)
print("Number of points:", points.shape[0])

# Downsample using Open3D voxel filter to reduce memory
pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.1)  # 10cm voxels
points = np.asarray(pcd_downsampled.points)
print(f"Downsampled to {points.shape[0]} points")
# ------------------------------------------------------
# 5) Build OctoMap and insert point cloud
# ------------------------------------------------------
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
# Note: projectDownTo2D may not be available in all pyoctomap builds
# If it fails, just skip this step and use the 3D map
try:
    proj_map = pom.OccupancyGrid()
    tree.projectDownTo2D(proj_map)
    proj_map.writeBinary("/mnt/d/uni_vub/thesis/Glass-Detection-VLM-pipeline/trusted_depth_2d_map.bt")
    print("Saved 2D planar map as trusted_depth_2d_map.bt")
except AttributeError:
    print("projectDownTo2D not available in this pyoctomap build - using 3D map only")


# # class MapBuilder:
#     def __init__(self, voxel_size: float = 0.05):
#         self.voxel_size = voxel_size
#         # keep a list of point arrays (N x 3) and colors
#         self._points = []
#         self._colors = []

#     def add_frame(self, rgb_image, depth_m: np.ndarray, intrinsics: Optional[object] = None):
#         """Add an RGB-D frame to the accumulated map.

#         rgb_image: PIL.Image or numpy array (H,W,3)
#         depth_m: numpy array (H,W) in meters
#         intrinsics: optional camera intrinsics (ignored if None)
#         """
#         # convert RGB to numpy
#         if hasattr(rgb_image, "convert"):
#             rgb = np.asarray(rgb_image)
#         else:
#             rgb = rgb_image

#         # Prepare Open3D RGBD image
#         color_o3d = o3d.geometry.Image((rgb).astype(np.uint8))
#         # Open3D expects depth in either uint16 (mm) or float in meters; we pass float32 meters
#         depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))

#         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#             color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
#         )

#         # Choose intrinsics
#         if intrinsics is None:
#             # assume a default pinhole model for typical 640x480 frames
#             h, w = depth_m.shape
#             fx = fy = 525.0
#             cx = w / 2.0
#             cy = h / 2.0
#             intrinsic = o3d.camera.PinholeCameraIntrinsic(int(w), int(h), fx, fy, cx, cy)
#         else:
#             # intrinsics from RealSense wrapper will be an object with width/height/fx/fy/ppx/ppy
#             try:
#                 intrinsic = o3d.camera.PinholeCameraIntrinsic(
#                     intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
#                 )
#             except Exception:
#                 h, w = depth_m.shape
#                 intrinsic = o3d.camera.PinholeCameraIntrinsic(int(w), int(h), 525.0, 525.0, w / 2.0, h / 2.0)

#         # create point cloud
#         pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
#         # transform to standard coords (optional)
#         pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

#         pts = np.asarray(pcd.points)
#         cols = np.asarray(pcd.colors)
#         if pts.size == 0:
#             return

#         self._points.append(pts)
#         self._colors.append(cols)

#     def save_map(self, out_path: str):
#         """Save combined map to a PLY file. If pyoctomap is available, a small attempt is made
#         to create an octree (user may change this code to match their pyoctomap API).
#         """
#         out_p = str(out_path)
#         if len(self._points) == 0:
#             print("No points to save")
#             return

#         pts = np.vstack(self._points)
#         cols = np.vstack(self._colors)
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(pts)
#         pcd.colors = o3d.utility.Vector3dVector(cols)

#         # downsample for smaller file
#         pcd = pcd.voxel_down_sample(self.voxel_size)

#         try:
#             # optional: try to use pyoctomap if installed
#             import pyoctomap  # type: ignore
#             # pyoctomap integration depends on the library API; leave a placeholder
#             print("pyoctomap detected — if desired, integrate point cloud into an OctoMap here.")
#         except Exception:
#             # fallback: write point cloud PLY
#             o3d.io.write_point_cloud(out_p, pcd)
#             print(f"Saved PLY point cloud: {out_p}")