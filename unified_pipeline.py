"""
Unified RGB-D Semantic Mapping Pipeline

A complete pipeline that processes RGB-D image streams and generates:
1. Geometric OctoMap from depth data (pure reconstruction, no semantics)
2. Combined OctoMap with full geometry + semantic labels on detected regions
3. Semantic-only OctoMap with ONLY detected glass/transparent regions
4. Semantic voxel map (dict) mapping detected regions to glass/transparent labels

Supports both real depth sensor data (TUM format) and estimated depth from monocular RGB.

Usage:
    python unified_pipeline.py --dataset data/rgbd_dataset_freiburg1_360 --output output/unified
    
Outputs:
    - geometric_map.bt: Pure depth-based 3D reconstruction (all geometry)
    - combined_map.bt: Same reconstruction with semantic regions tracked
    - semantic_only_map.bt: Only detected glass/transparent regions (sparse)
    - semantic_voxels.pkl: Voxel coordinates -> labels mapping for detected regions
    - combined_voxels.ply: PLY visualization of all geometry with semantic regions colored
    
Note: All .bt files can be visualized with: octovis <filename>.bt
"""

import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from collections import defaultdict
import pickle
from pathlib import Path

# Image processing
import cv2
from PIL import Image

# OctoMap
try:
    import pyoctomap
    OCTOMAP_AVAILABLE = True
except ImportError:
    OCTOMAP_AVAILABLE = False
    print("Warning: pyoctomap not available. Install with: pip install pyoctomap")

# Detector
from detector import OwlDetectorbase32

# Depth estimation (for monocular mode)
try:
    from transformers import pipeline
    DEPTH_ESTIMATION_AVAILABLE = True
except ImportError:
    DEPTH_ESTIMATION_AVAILABLE = False
    print("Warning: transformers not available for depth estimation")

# Open3D for live visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: open3d not available for live visualization. Install with: pip install open3d")

# Import utilities
from util import (
    load_tum_list,
    load_groundtruth,
    interpolated_pose,
    associate_depth,
    depth_to_points,
    transform_points,
    bbox_points_cam,
    world_to_voxel_idx,
    label_to_color,
    estimate_depth_from_rgb,
    estimate_intrinsics_from_image,
    quaternion_to_rotation_matrix
)


@dataclass
class PipelineConfig:
    """Configuration for the unified pipeline."""
    
    # Dataset paths
    dataset_dir: str = "data/rgbd_dataset_freiburg1_360"
    output_dir: str = "output/unified"
    
    # Depth source
    use_real_depth: bool = True  # If False, estimate from RGB
    depth_source_auto_detect: bool = True  # Auto-detect if depth/ exists
    
    # Camera intrinsics (TUM Freiburg1 defaults)
    fx: float = 517.3
    fy: float = 516.5
    cx: float = 318.6
    cy: float = 255.3
    depth_scale: float = 5000.0  # for real depth: meters = raw / depth_scale
    
    # OctoMap parameters
    octomap_resolution: float = 0.05  # meters (5cm voxels)
    octomap_max_range: float = 5.0
    
    # Depth processing
    depth_min: float = 0.2  # meters
    depth_max: float = 4.0  # meters
    point_stride: int = 4  # sample every Nth pixel
    
    # Detection parameters
    detection_queries: List[str] = field(default_factory=lambda: [
        "door", "window", "glass", "a glass window",
        "a transparent surface", "a glass wall",
        "a mirror", "a glass door"
    ])
    detection_threshold: float = 0.35
    detector_model: str = "base32"
    
    # Semantic mapping
    voxel_resolution: float = 0.05  # must match octomap_resolution
    min_detection_samples: int = 80  # min valid depth points in bbox
    fill_bbox_interior: bool = True  # fill missing depth inside detections
    
    # Processing controls
    frame_step: int = 5  # process every Nth frame
    max_frames: Optional[int] = None  # limit processing (None = all)
    start_frame: int = 0  # start from frame N
    
    # Timestamp association (for TUM format)
    max_timestamp_diff: float = 0.03  # 30ms max for RGB-depth association
    
    # Output options
    save_geometric_map: bool = True
    save_combined_map: bool = True  # geometry + semantic labels
    save_semantic_only_map: bool = True  # only detected regions
    save_semantic_pickle: bool = True  # save voxel label weights
    save_debug_images: bool = False  # save detection visualizations
    verbose: bool = True
    
    # Live visualization
    live_view: bool = False  # Enable real-time Open3D visualization
    live_view_update_freq: int = 5  # Update viewer every N frames
    live_view_follow: bool = False  # Follow camera mode: track sensor position
    live_view_compare: bool = False  # Dual window: geometric vs combined (with detections)
    live_view_sync: bool = True  # Sync camera views in compare mode (default: True)


class UnifiedPipeline:
    """
    Unified pipeline for RGB-D semantic mapping.
    
    Processes RGB-D streams and generates:
    - Geometric OctoMap: Pure depth-based reconstruction (all geometry)
    - Combined OctoMap: Full geometry with semantic detections tracked
    - Semantic-only OctoMap: Only detected glass/transparent regions
    - Semantic voxel dict: Labels for detected glass/transparent regions
    """
    
    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline with configuration."""
        self.config = config
        
        # Initialize detector
        if self.config.verbose:
            print(f"Initializing OWL-ViT detector ({self.config.detector_model})...")
        self.detector = self._init_detector()
        
        # Initialize depth estimator (if needed)
        self.depth_estimator = None
        if not self.config.use_real_depth:
            if self.config.verbose:
                print("Initializing depth estimation model...")
            self.depth_estimator = self._init_depth_estimator()
        
        # Initialize OctoMaps
        self.geometric_map = None
        self.combined_map = None  # geometry + semantic (ColorOcTree)
        self.semantic_only_map = None  # only detected regions
        if OCTOMAP_AVAILABLE:
            if self.config.verbose:
                print(f"Initializing OctoMaps (resolution: {self.config.octomap_resolution}m)...")
            self.geometric_map = pyoctomap.OcTree(self.config.octomap_resolution)
            self.combined_map = pyoctomap.ColorOcTree(self.config.octomap_resolution)
            self.semantic_only_map = pyoctomap.OcTree(self.config.octomap_resolution)
        
        # Semantic voxel storage (for label accumulation)
        self.semantic_voxels = defaultdict(lambda: defaultdict(float))
        
        # Colored points storage for PLY export (point -> (r, g, b))
        self.colored_points = {}  # Dict mapping (x, y, z) tuple to (r, g, b)
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'detections_total': 0,
            'points_geometric': 0,
            'points_semantic': 0
        }
        
        # Live visualization
        self.vis = None
        self.pcd = None
        # For dual-view comparison mode
        self.vis_geometric = None
        self.pcd_geometric = None
        self.geometric_points = {}  # Geometry-only points (no detections)
        self.current_camera_pose = None  # For follow mode: (t_world, q_xyzw)
        if self.config.live_view:
            self._init_live_viewer()
    
    def _init_detector(self):
        """Initialize the object detector."""
        if self.config.detector_model == "base32":
            return OwlDetectorbase32()
        else:
            raise ValueError(f"Unknown detector model: {self.config.detector_model}")
    
    def _init_depth_estimator(self):
        """Initialize depth estimation model for monocular RGB."""
        if not DEPTH_ESTIMATION_AVAILABLE:
            raise RuntimeError("transformers library required for depth estimation")
        # Depth-Anything-V2-Small from HuggingFace
        return pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    
    def _init_live_viewer(self):
        """Initialize Open3D live visualization window."""
        if not OPEN3D_AVAILABLE:
            print("Warning: Open3D not available, live view disabled")
            self.config.live_view = False
            return
        
        if self.config.verbose:
            if self.config.live_view_compare:
                print("Initializing dual 3D viewers (comparison mode)...")
            else:
                print("Initializing live 3D viewer...")
        
        if self.config.live_view_compare:
            # Dual window mode: geometric (left) and combined (right)
            self.vis_geometric = o3d.visualization.Visualizer()
            self.vis_geometric.create_window(window_name="Geometric Map (No Detections)", width=640, height=720, left=0, top=0)
            self.pcd_geometric = o3d.geometry.PointCloud()
            self.vis_geometric.add_geometry(self.pcd_geometric)
            opt1 = self.vis_geometric.get_render_option()
            opt1.point_size = 3.0
            opt1.background_color = np.asarray([0.1, 0.1, 0.1])
            
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="Combined Map (With Detections)", width=640, height=720, left=650, top=0)
            self.pcd = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.pcd)
            opt2 = self.vis.get_render_option()
            opt2.point_size = 3.0
            opt2.background_color = np.asarray([0.1, 0.1, 0.1])
        else:
            # Single window mode
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="Live Map Construction", width=1280, height=720)
            self.pcd = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.pcd)
            opt = self.vis.get_render_option()
            opt.point_size = 3.0
            opt.background_color = np.asarray([0.5, 0.5, 0.5])
        
        self.view_initialized = False
    
    def _update_live_viewer(self):
        """Update the live viewer with current colored points."""
        if not self.config.live_view or self.vis is None:
            return
        
        if self.config.live_view_compare:
            # Dual window mode: update both views
            self._update_dual_viewers()
        else:
            # Single window mode
            self._update_single_viewer()
    
    def _update_single_viewer(self):
        """Update single window view."""
        if not self.colored_points:
            if self.config.verbose and self.stats['frames_processed'] <= 3:
                print(f"  [Live View] No points yet (frame {self.stats['frames_processed']})")
            return
        
        points = np.array(list(self.colored_points.keys()))
        colors = np.array(list(self.colored_points.values())) / 255.0
        
        if self.config.verbose and self.stats['frames_processed'] <= 3:
            print(f"  [Live View] Updating with {len(points)} points")
        
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(self.pcd)
        
        # Reset view on first update to frame the geometry with better zoom
        if not self.view_initialized and len(points) > 0:
            if not self.config.live_view_follow:
                # Static view mode
                self.vis.reset_view_point(True)
                # Zoom out for better overview
                ctr = self.vis.get_view_control()
                ctr.set_zoom(2)  # Zoom out (bigger = more zoomed out)
            else:
                # Follow mode: do initial reset but we'll override it immediately after
                self.vis.reset_view_point(True)
            self.view_initialized = True
        
        # Follow camera mode: update camera to track sensor position
        if self.config.live_view_follow and self.current_camera_pose is not None:
            t_world, q_xyzw = self.current_camera_pose
            ctr = self.vis.get_view_control()
            
            # Convert quaternion to rotation matrix for camera orientation
            R = quaternion_to_rotation_matrix(q_xyzw)
            
            # Camera coordinate system: X=right, Y=down, Z=forward (into scene)
            # Forward direction: where the sensor is looking
            forward_dir = R @ np.array([0, 0, 1])
            
            # Up direction in world frame
            up_dir = R @ np.array([0, -1, 0])
            
            # Position viewer camera BEHIND the sensor so we can see what it captures
            # Offset camera back by 0.5m and slightly up for better view
            back_offset = 0.0
            up_offset = 0.0
            cam_pos = t_world + forward_dir * back_offset + up_dir * (-up_offset)
            
            # Look at point ahead of the sensor (where sensor is looking)
            lookat = t_world + forward_dir * 2.0
            
            # Debug output on first few updates
            if self.stats['frames_processed'] <= 3 and self.config.verbose:
                print(f"  [Follow Cam] Sensor pos: {t_world}, lookat: {lookat}")
                print(f"  [Follow Cam] Points in cloud: {len(points)}")
            
            # Set camera parameters
            ctr.set_lookat(lookat)
            ctr.set_front(forward_dir)  # View direction from camera
            ctr.set_up(-up_dir)  # Up is opposite of down vector
            ctr.set_zoom(0.7)  # Adjust zoom for good visibility
        
        if not self.config.live_view_follow:
            self.vis.poll_events()
            self.vis.update_renderer()
        else:
            # Follow mode updates handled above
            self.vis.poll_events()
            self.vis.update_renderer()
        
        # Brief pause to allow viewer to render
        import time
        time.sleep(0.01)
    
    def _update_dual_viewers(self):
        """Update both geometric and combined viewers."""
        if not self.geometric_points and not self.colored_points:
            return
        
        # Update geometric window (gray points, no detections)
        if self.geometric_points and self.vis_geometric:
            geo_points = np.array(list(self.geometric_points.keys()))
            geo_colors = np.array(list(self.geometric_points.values())) / 255.0
            self.pcd_geometric.points = o3d.utility.Vector3dVector(geo_points)
            self.pcd_geometric.colors = o3d.utility.Vector3dVector(geo_colors)
            self.vis_geometric.update_geometry(self.pcd_geometric)
            
            if not self.view_initialized:
                self.vis_geometric.reset_view_point(True)
                ctr = self.vis_geometric.get_view_control()
                ctr.set_zoom(2.0)
        
        # Update combined window (with red detections)
        if self.colored_points and self.vis:
            combined_points = np.array(list(self.colored_points.keys()))
            combined_colors = np.array(list(self.colored_points.values())) / 255.0
            self.pcd.points = o3d.utility.Vector3dVector(combined_points)
            self.pcd.colors = o3d.utility.Vector3dVector(combined_colors)
            self.vis.update_geometry(self.pcd)
            
            if not self.view_initialized:
                self.vis.reset_view_point(True)
                ctr = self.vis.get_view_control()
                ctr.set_zoom(2.0)
                self.view_initialized = True
        
        # Poll and render both windows
        if self.vis_geometric:
            self.vis_geometric.poll_events()
            self.vis_geometric.update_renderer()
        if self.vis:
            self.vis.poll_events()
            self.vis.update_renderer()
        
        import time
        time.sleep(0.01)
    
    def process_stream(self, dataset_path: Optional[str] = None):
        """
        Process a complete RGB-D dataset stream.
        
        Args:
            dataset_path: Path to TUM-format dataset (overrides config if provided)
        """
        if dataset_path is None:
            dataset_path = self.config.dataset_dir
        
        if self.config.verbose:
            print(f"\nProcessing dataset: {dataset_path}")
            print("=" * 60)
        
        # Auto-detect depth source if enabled
        if self.config.depth_source_auto_detect:
            depth_dir = os.path.join(dataset_path, "depth")
            if os.path.exists(depth_dir):
                self.config.use_real_depth = True
                if self.config.verbose:
                    print("Detected real depth data")
            else:
                self.config.use_real_depth = False
                if self.config.verbose:
                    print("No depth data found, will estimate from RGB")
        
        # Load dataset
        rgb_list, depth_list, gt_list = self._load_dataset(dataset_path)
        
        if self.config.verbose:
            print(f"Found {len(rgb_list)} RGB frames")
            if depth_list:
                print(f"Found {len(depth_list)} depth frames")
            if gt_list:
                print(f"Found {len(gt_list)} groundtruth poses")
        
        # Process frames
        self._process_frames(dataset_path, rgb_list, depth_list, gt_list)
        
        # Final live viewer update
        if self.config.live_view:
            self._update_live_viewer()
            if self.config.verbose:
                if self.config.live_view_compare:
                    print("\nLive viewers active. Close any window to continue...")
                else:
                    print("\nLive viewer active. Close the window to continue...")
            
            # Run and destroy windows
            if self.config.live_view_compare:
                # Dual window mode - run both in non-blocking loop
                if self.config.verbose and self.config.live_view_sync:
                    print("  Camera views are synchronized. Move one to move both!")
                
                last_view_params = None
                while self.vis_geometric and self.vis:
                    # Poll both windows
                    if not self.vis_geometric.poll_events():
                        break
                    if not self.vis.poll_events():
                        break
                    
                    # Synchronize camera views if enabled
                    if self.config.live_view_sync:
                        # Get camera parameters from left window (geometric)
                        ctr_geo = self.vis_geometric.get_view_control()
                        ctr_combined = self.vis.get_view_control()
                        
                        # Extract camera parameters
                        cam_params_geo = ctr_geo.convert_to_pinhole_camera_parameters()
                        
                        # Apply to combined window
                        ctr_combined.convert_from_pinhole_camera_parameters(cam_params_geo)
                    
                    self.vis_geometric.update_renderer()
                    self.vis.update_renderer()
                    import time
                    time.sleep(0.01)
                
                # Cleanup
                if self.vis_geometric:
                    self.vis_geometric.destroy_window()
                if self.vis:
                    self.vis.destroy_window()
            else:
                # Single window mode
                self.vis.run()
                self.vis.destroy_window()
        
        # Save outputs
        self._save_outputs()
        
        # Print statistics
        self._print_statistics()
    
    def process_frame(self, rgb: np.ndarray, depth: Optional[np.ndarray],
                     pose: Tuple[np.ndarray, np.ndarray], timestamp: float):
        """
        Process a single RGB-D frame.
        
        Args:
            rgb: RGB image (H, W, 3) uint8
            depth: Depth map (H, W) float32 in meters, or None to estimate
            pose: Tuple of (translation, quaternion_xyzw)
            timestamp: Frame timestamp
        """
        # Convert to PIL if needed
        if isinstance(rgb, np.ndarray):
            rgb_pil = Image.fromarray(rgb)
        else:
            rgb_pil = rgb
        
        # Estimate depth if not provided
        if depth is None and self.depth_estimator:
            depth = estimate_depth_from_rgb(rgb_pil, self.depth_estimator,
                                           self.config.depth_min, self.config.depth_max)
        
        if depth is None:
            raise ValueError("No depth data provided or estimated")
        
        t_world, q_xyzw = pose
        self._process_single_frame(rgb_pil, depth, t_world, q_xyzw, timestamp, 0)
    
    def _load_dataset(self, dataset_path: str):
        """Load RGB, depth, and groundtruth lists from TUM dataset."""
        rgb_list = None
        depth_list = None
        gt_list = None
        
        # Load RGB list
        rgb_txt = os.path.join(dataset_path, "rgb.txt")
        if os.path.exists(rgb_txt):
            rgb_list = load_tum_list(rgb_txt)
        else:
            # Fallback: scan rgb/ directory
            rgb_dir = os.path.join(dataset_path, "rgb")
            if os.path.exists(rgb_dir):
                files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg'))])
                rgb_list = [(float(i), f"rgb/{f}") for i, f in enumerate(files)]
        
        # Load depth list (if using real depth)
        if self.config.use_real_depth:
            depth_txt = os.path.join(dataset_path, "depth.txt")
            if os.path.exists(depth_txt):
                depth_list = load_tum_list(depth_txt)
            else:
                depth_dir = os.path.join(dataset_path, "depth")
                if os.path.exists(depth_dir):
                    files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
                    depth_list = [(float(i), f"depth/{f}") for i, f in enumerate(files)]
        
        # Load groundtruth poses
        gt_txt = os.path.join(dataset_path, "groundtruth.txt")
        if os.path.exists(gt_txt):
            gt_list = load_groundtruth(gt_txt)
        
        return rgb_list, depth_list, gt_list
    
    def _process_frames(self, dataset_path: str, rgb_list: List,
                       depth_list: Optional[List], gt_list: Optional[List]):
        """Process all frames in the dataset."""
        if not rgb_list:
            print("Error: No RGB frames found")
            return
        
        # Limit frame range
        start_idx = self.config.start_frame
        end_idx = len(rgb_list) if self.config.max_frames is None else min(start_idx + self.config.max_frames, len(rgb_list))
        
        if self.config.verbose:
            print(f"\nProcessing frames {start_idx} to {end_idx} (step={self.config.frame_step})")
        
        for i in range(start_idx, end_idx):
            # Skip frames based on frame_step
            if (i - start_idx) % self.config.frame_step != 0:
                continue
            
            ts, rgb_rel = rgb_list[i]
            
            # Load RGB image
            rgb_path = os.path.join(dataset_path, rgb_rel)
            bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if bgr is None:
                if self.config.verbose:
                    print(f"  Warning: Could not load RGB {rgb_path}")
                continue
            
            rgb_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            
            # Load or estimate depth
            depth_m = None
            if self.config.use_real_depth and depth_list:
                # Associate depth frame
                depth_item = associate_depth(ts, depth_list, self.config.max_timestamp_diff)
                if depth_item:
                    _, depth_rel = depth_item
                    depth_path = os.path.join(dataset_path, depth_rel)
                    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                    if depth_raw is not None:
                        depth_m = depth_raw.astype(np.float32) / self.config.depth_scale
            elif self.depth_estimator:
                # Estimate depth from RGB
                depth_m = estimate_depth_from_rgb(rgb_pil, self.depth_estimator, 
                                                 self.config.depth_min, self.config.depth_max)
            
            if depth_m is None:
                if self.config.verbose:
                    print(f"  Warning: No depth data for frame {i}")
                continue
            
            # Get pose
            if gt_list:
                t_world, q_xyzw = interpolated_pose(ts, gt_list)
            else:
                # Identity pose fallback
                t_world = np.array([0.0, 0.0, 0.0])
                q_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
            
            # Process frame
            self._process_single_frame(rgb_pil, depth_m, t_world, q_xyzw, ts, i)
            
            if self.config.verbose and i % (self.config.frame_step * 10) == 0:
                print(f"  Processed frame {i}/{end_idx} | Voxels: {len(self.semantic_voxels)}")
    
    def _process_single_frame(self, rgb_pil: Image.Image, depth_m: np.ndarray,
                             t_world: np.ndarray, q_xyzw: np.ndarray, 
                             timestamp: float, frame_idx: int):
        """
        Core processing for a single frame: build geometric map and detect semantics.
        """
        # 1. Build geometric OctoMap from full depth (pure reconstruction)
        pts_cam_full = None
        pts_world_full = None
        origin = np.array([float(t_world[0]), float(t_world[1]), float(t_world[2])], dtype=float)
        
        if self.geometric_map is not None or self.combined_map is not None:
            pts_cam_full = depth_to_points(
                depth_m, 
                self.config.fx, self.config.fy, self.config.cx, self.config.cy,
                stride=self.config.point_stride,
                z_min=self.config.depth_min,
                z_max=self.config.depth_max
            )
            
            if pts_cam_full.shape[0] > 0:
                pts_world_full = transform_points(pts_cam_full, t_world, q_xyzw)
                
                # Insert into geometric map (pure geometry, no semantics)
                if self.geometric_map is not None:
                    self.geometric_map.insertPointCloud(pts_world_full, origin, self.config.octomap_max_range, False)
                
                # Insert into combined map with default white/gray color (background)
                if self.combined_map is not None:
                    # Insert points with white color for regular geometry
                    for pt in pts_world_full:
                        self.combined_map.updateNode(pt, True)  # occupied
                        self.combined_map.integrateNodeColor(pt, 144, 238, 180)  # light green
                        # Track colored point for PLY export
                        pt_key = (float(pt[0]), float(pt[1]), float(pt[2]))
                        self.colored_points[pt_key] = (144, 238, 180)
                        # Also track in geometric-only view (for comparison mode)
                        if self.config.live_view_compare:
                            self.geometric_points[pt_key] = (200, 200, 200)  # gray for geometric
                
                self.stats['points_geometric'] += len(pts_world_full)
        
        # 2. Run detection and build semantic map
        detections = self.detector.detect(
            rgb_pil, 
            self.config.detection_queries, 
            threshold=self.config.detection_threshold
        )
        
        if detections:
            self.stats['detections_total'] += len(detections)
            
            for det in detections:
                box = det["box"]
                score = float(det["score"])
                label = det["label"]
                
                # Extract 3D points from detection bbox
                pts_cam = bbox_points_cam(
                    depth_m, box,
                    self.config.fx, self.config.fy, self.config.cx, self.config.cy,
                    stride=self.config.point_stride,
                    z_min=self.config.depth_min,
                    z_max=self.config.depth_max,
                    fill_interior=self.config.fill_bbox_interior,
                    min_samples=self.config.min_detection_samples
                )
                
                if pts_cam is None:
                    continue
                
                pts_world = transform_points(pts_cam, t_world, q_xyzw)
                
                # Accumulate semantic labels in voxel map
                weight = max(0.0, min(1.0, score))
                for p in pts_world:
                    voxel_key = world_to_voxel_idx(p, self.config.voxel_resolution)
                    self.semantic_voxels[voxel_key][label] += weight
                
                # Update colors in combined map for semantic regions (BRIGHT RED)
                if self.combined_map is not None:
                    # Use bright red for detected glass/transparent regions
                    r, g, b = 255, 0, 0  # Bright red
                    for pt in pts_world:
                        self.combined_map.updateNode(pt, True)
                        self.combined_map.integrateNodeColor(pt, r, g, b)
                        # Update colored point for PLY export
                        pt_key = (float(pt[0]), float(pt[1]), float(pt[2]))
                        self.colored_points[pt_key] = (r, g, b)
                
                self.stats['points_semantic'] += len(pts_world)
                
                # Note: Combined map already has all geometry inserted above
                # Semantic information is stored in semantic_voxels dict
                
                # Insert into semantic-only map (only detected regions)
                if self.semantic_only_map is not None:
                    self.semantic_only_map.insertPointCloud(pts_world, origin, self.config.octomap_max_range, False)
            
            # Save debug image with detections
            if self.config.save_debug_images:
                self._save_debug_image(rgb_pil, detections, frame_idx)
        
        self.stats['frames_processed'] += 1
        
        # Store current camera pose for follow mode
        if self.config.live_view_follow:
            self.current_camera_pose = (t_world, q_xyzw)
        
        # Update live viewer
        if self.config.live_view and self.stats['frames_processed'] % self.config.live_view_update_freq == 0:
            self._update_live_viewer()
    
    def _save_debug_image(self, rgb_pil: Image.Image, detections: List[Dict], frame_idx: int):
        """Save RGB image with detection boxes overlaid."""
        import cv2
        from PIL import ImageDraw
        
        img_draw = rgb_pil.copy()
        draw = ImageDraw.Draw(img_draw)
        
        for det in detections:
            box = det["box"]
            label = det["label"]
            score = det["score"]
            
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1] - 10), f"{label}: {score:.2f}", fill="red")
        
        debug_dir = os.path.join(self.config.output_dir, "debug_images")
        os.makedirs(debug_dir, exist_ok=True)
        img_draw.save(os.path.join(debug_dir, f"frame_{frame_idx:05d}.jpg"))
    
    def _save_outputs(self):
        """Save all generated maps and data."""
        # Update OctoMaps before saving
        if self.geometric_map is not None:
            self.geometric_map.updateInnerOccupancy()
        if self.combined_map is not None:
            # Average colors before updating occupancy for ColorOcTree
            self.combined_map.updateInnerOccupancy()
        if self.semantic_only_map is not None:
            self.semantic_only_map.updateInnerOccupancy()
        
        # Save maps
        self.save_maps()
        
        # Export combined PLY from tracked colored points
        if self.combined_map is not None and self.colored_points:
            from util import export_colored_points_to_ply
            ply_path = os.path.join(self.config.output_dir, "combined_voxels.ply")
            export_colored_points_to_ply(self.colored_points, ply_path)
            if self.config.verbose:
                print(f"  ✓ Combined PLY: {ply_path}")
    
    def save_maps(self, output_dir: Optional[str] = None):
        """
        Save geometric, combined, and semantic-only maps.
        
        Args:
            output_dir: Output directory (overrides config if provided)
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.config.verbose:
            print(f"\nSaving maps to: {output_dir}")
        
        # Save geometric map (pure reconstruction)
        if self.config.save_geometric_map and self.geometric_map:
            geo_path = os.path.join(output_dir, "geometric_map.bt")
            self.geometric_map.writeBinary(geo_path)
            if self.config.verbose:
                print(f"  ✓ Geometric map: {geo_path}")
        
        # Save combined map (geometry + semantic regions)
        if self.config.save_combined_map and self.combined_map:
            combined_path = os.path.join(output_dir, "combined_map.bt")
            self.combined_map.writeBinary(combined_path)
            if self.config.verbose:
                print(f"  ✓ Combined map (geometry + semantic): {combined_path}")
        
        # Save semantic-only map (detected regions only)
        if self.config.save_semantic_only_map and self.semantic_only_map:
            semantic_path = os.path.join(output_dir, "semantic_only_map.bt")
            self.semantic_only_map.writeBinary(semantic_path)
            if self.config.verbose:
                print(f"  ✓ Semantic-only map (detected regions): {semantic_path}")
        
        # Save semantic voxel weights
        if self.config.save_semantic_pickle and self.semantic_voxels:
            pkl_path = os.path.join(output_dir, "semantic_voxels.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump({k: dict(v) for k, v in self.semantic_voxels.items()}, f)
            if self.config.verbose:
                print(f"  ✓ Semantic voxels: {pkl_path}")
    
    def _print_statistics(self):
        """Print processing statistics."""
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("Processing Statistics")
            print("=" * 60)
            print(f"Frames processed:      {self.stats['frames_processed']}")
            print(f"Total detections:      {self.stats['detections_total']}")
            print(f"Geometric points:      {self.stats['points_geometric']}")
            print(f"Semantic points:       {self.stats['points_semantic']}")
            if self.semantic_voxels:
                print(f"Semantic voxels:       {len(self.semantic_voxels)}")
            print("=" * 60)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified RGB-D Semantic Mapping Pipeline"
    )
    parser.add_argument(
        "--dataset", type=str,
        default="data/rgbd_dataset_freiburg1_360",
        help="Path to TUM-format RGB-D dataset"
    )
    parser.add_argument(
        "--output", type=str,
        default="output/unified",
        help="Output directory for maps"
    )
    parser.add_argument(
        "--estimate-depth", action="store_true",
        help="Estimate depth from RGB (ignore real depth data)"
    )
    parser.add_argument(
        "--frame-step", type=int, default=5,
        help="Process every Nth frame (default: 5)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--detector", type=str, default="base32",
        choices=["base32"],
        help="OWL-ViT detector model"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.35,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--debug-images", action="store_true",
        help="Save debug images with detection overlays"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--live-view", action="store_true",
        help="Enable real-time 3D visualization with Open3D"
    )
    parser.add_argument(
        "--live-view-freq", type=int, default=5,
        help="Update live view every N frames (default: 5)"
    )
    parser.add_argument(
        "--follow-camera", action="store_true",
        help="Follow camera mode: viewer tracks sensor position (requires --live-view)"
    )
    parser.add_argument(
        "--compare-view", action="store_true",
        help="Dual window comparison: show geometric map vs combined map side by side (requires --live-view)"
    )
    parser.add_argument(
        "--no-sync", action="store_true",
        help="Disable camera synchronization in compare-view mode (allows independent camera control)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create configuration
    config = PipelineConfig(
        dataset_dir=args.dataset,
        output_dir=args.output,
        use_real_depth=not args.estimate_depth,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        detector_model=args.detector,
        detection_threshold=args.threshold,
        save_debug_images=args.debug_images,
        verbose=not args.quiet,
        live_view=args.live_view,
        live_view_update_freq=args.live_view_freq,
        live_view_follow=args.follow_camera,
        live_view_compare=args.compare_view,
        live_view_sync=not args.no_sync  # Default to synced unless --no-sync is used
    )
    
    # Check dependencies
    if not OCTOMAP_AVAILABLE:
        print("ERROR: pyoctomap is required. Install with: pip install pyoctomap")
        sys.exit(1)
    
    # Create and run pipeline
    pipeline = UnifiedPipeline(config)
    pipeline.process_stream()
    
    if config.verbose:
        print("\n✓ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
