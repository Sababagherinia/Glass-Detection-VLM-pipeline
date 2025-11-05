"""camera.py

Simple RGB-D frame source helper.

Supports two modes:
- folder playback: expects a folder with subfolders 'rgb' and 'depth' containing matching files
- realsense: uses pyrealsense2 if installed (optional)

The module yields tuples (PIL.Image RGB, depth numpy array in meters, meta dict)
"""
from pathlib import Path
from typing import Iterator, Tuple, Optional
import numpy as np
from PIL import Image


class RGBDSource:
    def __init__(self, mode: str = "folder", folder_path: str = "data"):
        self.mode = mode
        self.folder = Path(folder_path)

        if self.mode == "folder":
            rgb_dir = self.folder / "rgb"
            depth_dir = self.folder / "depth"
            if not rgb_dir.exists() or not depth_dir.exists():
                raise FileNotFoundError(f"Expected '{rgb_dir}' and '{depth_dir}' for folder mode")
            self._rgb_files = sorted([p for p in rgb_dir.iterdir() if p.is_file()])
            self._depth_files = sorted([p for p in depth_dir.iterdir() if p.is_file()])
            if len(self._rgb_files) != len(self._depth_files):
                # allow unequal lengths but will iterate until one runs out
                pass

        elif self.mode == "realsense":
            try:
                import pyrealsense2 as rs  # type: ignore
            except Exception as e:
                raise RuntimeError("pyrealsense2 is required for realsense mode") from e
            self._rs = rs

        else:
            raise ValueError("mode must be 'folder' or 'realsense'")

    def frame_iter(self) -> Iterator[Tuple[Image.Image, np.ndarray, dict]]:
        """Yield (rgb_pil, depth_meters_np, meta)

        Depth arrays are returned as float32 meters.
        Meta may include 'intrinsics' when available.
        """
        if self.mode == "folder":
            for rgb_p, depth_p in zip(self._rgb_files, self._depth_files):
                rgb = Image.open(rgb_p).convert("RGB")
                # load depth as numpy
                d = Image.open(depth_p)
                d_np = np.asarray(d)
                # common convention: 16-bit PNG with depth in millimeters
                if d_np.dtype == np.uint16:
                    depth_m = d_np.astype(np.float32) / 1000.0
                else:
                    # if float depth encoded, assume meters already
                    depth_m = d_np.astype(np.float32)

                meta = {"path": str(rgb_p), "depth_path": str(depth_p)}
                yield rgb, depth_m, meta

        else:
            # very small RealSense wrapper — yields frames continuously
            rs = self._rs
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = pipe.start(cfg)
            try:
                while True:
                    frames = pipe.wait_for_frames()
                    color = frames.get_color_frame()
                    depth = frames.get_depth_frame()
                    if not color or not depth:
                        continue
                    # convert
                    import cv2  # local import
                    color_arr = np.asanyarray(color.get_data())
                    color_pil = Image.fromarray(cv2.cvtColor(color_arr, cv2.COLOR_BGR2RGB))
                    depth_arr = np.asanyarray(depth.get_data()).astype(np.float32)
                    depth_m = depth_arr / 1000.0
                    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                    meta = {"intrinsics": intrinsics}
                    yield color_pil, depth_m, meta
            finally:
                pipe.stop()
