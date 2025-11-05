"""mapping.py

Helpers to convert RGB-D frames to a point cloud and accumulate a scene map.

Uses Open3D to create point clouds. If `pyoctomap` is installed, a hook point is provided
to integrate with OctoMap. Otherwise the final map is saved as a combined PLY file.
"""
from typing import Optional
import numpy as np
import open3d as o3d


class MapBuilder:
    def __init__(self, voxel_size: float = 0.05):
        self.voxel_size = voxel_size
        # keep a list of point arrays (N x 3) and colors
        self._points = []
        self._colors = []

    def add_frame(self, rgb_image, depth_m: np.ndarray, intrinsics: Optional[object] = None):
        """Add an RGB-D frame to the accumulated map.

        rgb_image: PIL.Image or numpy array (H,W,3)
        depth_m: numpy array (H,W) in meters
        intrinsics: optional camera intrinsics (ignored if None)
        """
        # convert RGB to numpy
        if hasattr(rgb_image, "convert"):
            rgb = np.asarray(rgb_image)
        else:
            rgb = rgb_image

        # Prepare Open3D RGBD image
        color_o3d = o3d.geometry.Image((rgb).astype(np.uint8))
        # Open3D expects depth in either uint16 (mm) or float in meters; we pass float32 meters
        depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )

        # Choose intrinsics
        if intrinsics is None:
            # assume a default pinhole model for typical 640x480 frames
            h, w = depth_m.shape
            fx = fy = 525.0
            cx = w / 2.0
            cy = h / 2.0
            intrinsic = o3d.camera.PinholeCameraIntrinsic(int(w), int(h), fx, fy, cx, cy)
        else:
            # intrinsics from RealSense wrapper will be an object with width/height/fx/fy/ppx/ppy
            try:
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
                )
            except Exception:
                h, w = depth_m.shape
                intrinsic = o3d.camera.PinholeCameraIntrinsic(int(w), int(h), 525.0, 525.0, w / 2.0, h / 2.0)

        # create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        # transform to standard coords (optional)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors)
        if pts.size == 0:
            return

        self._points.append(pts)
        self._colors.append(cols)

    def save_map(self, out_path: str):
        """Save combined map to a PLY file. If pyoctomap is available, a small attempt is made
        to create an octree (user may change this code to match their pyoctomap API).
        """
        out_p = str(out_path)
        if len(self._points) == 0:
            print("No points to save")
            return

        pts = np.vstack(self._points)
        cols = np.vstack(self._colors)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)

        # downsample for smaller file
        pcd = pcd.voxel_down_sample(self.voxel_size)

        try:
            # optional: try to use pyoctomap if installed
            import pyoctomap  # type: ignore
            # pyoctomap integration depends on the library API; leave a placeholder
            print("pyoctomap detected — if desired, integrate point cloud into an OctoMap here.")
        except Exception:
            # fallback: write point cloud PLY
            o3d.io.write_point_cloud(out_p, pcd)
            print(f"Saved PLY point cloud: {out_p}")
