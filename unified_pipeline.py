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
    - combined_map.ot: Same reconstruction with semantic regions tracked (preserves color)
    - semantic_only_map.bt: Only detected glass/transparent regions (sparse)
    - semantic_voxels.pkl: Voxel coordinates -> labels mapping for detected regions
    - combined_voxels.ply: PLY visualization of all geometry with semantic regions colored
    
Note: .bt and .ot files can be visualized with octovis.
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

# Visualization: colormaps for confidence-based rendering
try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available for advanced visualization")

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
    depth_model: str = "da2"  # da2|da3
    monocular_depth_scale: float = 1.0  # global multiplier for estimated monocular depth

    # Depth nonlinearity correction (depth-dependent scaling)
    enable_depth_dependent_scale: bool = False  # enable nonlinear depth correction
    scale_near: float = 1.3088  # scale factor at near depth (Freiburg1 calibration)
    scale_far: float = 0.9184   # scale factor at far depth (Freiburg1 calibration)
    scale_depth_range: float = 3.0  # depth range for linear interpolation (meters)

    # Monocular scale updater placeholders (future use)
    enable_scale_updater: bool = False
    scale_updater_mode: str = "none"  # placeholder: none|floor|object|ema
    scale_ema_alpha: float = 0.02
    
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
    transparent_min_detection_samples: int = 20  # lower threshold for transparent objects
    fill_bbox_interior: bool = True  # fill missing depth inside detections
    enable_transparent_depth_fallback: bool = True  # infer transparent depth from bbox context
    transparent_context_ring_px: int = 16  # context ring width around bbox (pixels)
    transparent_fallback_min_samples: int = 12  # min valid context samples for fallback
    transparent_label_keywords: List[str] = field(default_factory=lambda: [
        "glass", "transparent", "window", "mirror"
    ])
    
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
    save_semantic_sidebyside: bool = False  # Save one final top-down RGB + semantic side-by-side visualization
    topdown_resolution: float = 0.05  # meters per pixel for final top-down export
    verbose: bool = True
    
    # Visualization options (for live view)
    vis_confidence_colormapping: bool = True  # Use colormap based on detection confidence
    vis_confidence_colormap: str = "magma"  # Options: "coolwarm", "viridis", "plasma", "magma", "jet"
    vis_confidence_min: float = 0.3  # Minimum confidence for visualization (0.0-1.0)
    vis_transparency_enabled: bool = True  # Enable semi-transparent rendering
    vis_transparency_alpha: float = 0.7  # Point transparency (0=invisible, 1=opaque)
    vis_show_statistics: bool = True  # Show real-time statistics text overlay
    vis_point_size_scale: float = 1.0  # Scale for point size in visualization
    
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
        self.current_mono_depth_scale = float(self.config.monocular_depth_scale)
        self._scale_updater_notice_printed = False
        self.depth_backend = self.config.depth_model
        
        # Initialize detector
        if self.config.verbose:
            print(f"Initializing OWL-ViT detector ({self.config.detector_model})...")
        self.detector = self._init_detector()
        
        # Initialize depth estimator (if needed)
        self.depth_estimator = None
        if not self.config.use_real_depth:
            if self.config.verbose:
                print("Initializing depth estimation model...")
                print(f"Monocular depth scale: {self.current_mono_depth_scale:.4f}")
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
        
        # Live-view semantic overlay points (rendered above the geometry layer)
        self.semantic_points = {}  # Dict mapping (x, y, z) tuple to (r, g, b)
        
        # Point confidence tracking for visualization (for confidence-based colormapping)
        self.point_confidences = {}  # Dict mapping (x, y, z) tuple to confidence score (0-1)

        # Final top-down RGB orthomosaic accumulators
        self.topdown_rgb_accum = defaultdict(lambda: np.zeros(4, dtype=np.float64))
        self.topdown_bounds = [np.inf, np.inf, -np.inf, -np.inf]  # x_min, z_min, x_max, z_max
        
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
        self.pcd_overlay = None
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
        if self.config.depth_model == "da3":
            try:
                import torch
                from depth_anything_3.api import DepthAnything3  # type: ignore[import-not-found]

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to(device=device)
                if self.config.verbose:
                    print("Using Depth Anything 3 metric model (DA3METRIC-LARGE)")
                return {"backend": "da3", "model": model}
            except Exception as exc:
                raise RuntimeError(f"DA3 requested but could not be initialized: {exc}") from exc

        if not DEPTH_ESTIMATION_AVAILABLE:
            raise RuntimeError("transformers library required for depth estimation")
        if self.config.verbose:
            print("Using Depth Anything V2 metric model")
        return pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-small-hf")

    def _estimate_depth_quick(self, rgb_pil: Image.Image) -> np.ndarray:
        """Estimate depth with the active backend (DA2 default, DA3 opt-in)."""
        if self.depth_backend == "da3":
            pred = self.depth_estimator["model"].inference([np.array(rgb_pil)])
            depth_m = np.asarray(pred.depth[0], dtype=np.float32)
            depth_m = depth_m * float(self.current_mono_depth_scale)
            depth_m = np.clip(depth_m, self.config.depth_min, self.config.depth_max)
            return depth_m

        return estimate_depth_from_rgb(
            rgb_pil,
            self.depth_estimator,
            self.config.depth_min,
            self.config.depth_max,
            scale_factor=self.current_mono_depth_scale,
            enable_depth_dependent_scale=self.config.enable_depth_dependent_scale,
            scale_near=self.config.scale_near,
            scale_far=self.config.scale_far,
            scale_depth_range=self.config.scale_depth_range,
        )
    
    def _confidence_to_color(self, confidence: float) -> Tuple[int, int, int]:
        """
        Convert detection confidence (0-1) to RGB color using a colormap.
        Uses matplotlib colormaps for professional visualization.
        
        Args:
            confidence: Detection confidence score (0-1)
        
        Returns:
            (r, g, b) tuple with values 0-255
        """
        if not MATPLOTLIB_AVAILABLE:
            # Fallback to simple red-yellow-green gradient if matplotlib not available
            if confidence < 0.5:
                # Red to Yellow
                r, g, b = 255, int(confidence * 2 * 255), 0
            else:
                # Yellow to Green
                r, g, b = int((1 - confidence) * 2 * 255), 255, 0
            return (r, g, b)
        
        # Get colormap from matplotlib
        cmap_name = self.config.vis_confidence_colormap
        cmap = cm.get_cmap(cmap_name)
        
        # Apply colormap (confidence is normalized 0-1)
        rgba = cmap(confidence)
        
        # Convert RGBA [0,1] to RGB [0,255]
        r = int(rgba[0] * 255)
        g = int(rgba[1] * 255)
        b = int(rgba[2] * 255)
        
        return (r, g, b)
    
    def _accumulate_topdown_rgb(self, rgb_pil: Image.Image, depth_m: np.ndarray,
                                t_world: np.ndarray, q_xyzw: np.ndarray):
        """Accumulate true RGB colors onto a top-down world grid from the current frame."""
        if depth_m is None:
            return

        rgb_np = np.asarray(rgb_pil)
        h, w = depth_m.shape
        stride = max(1, int(self.config.point_stride))
        res = float(self.config.topdown_resolution)

        ys = np.arange(0, h, stride)
        xs = np.arange(0, w, stride)
        uu, vv = np.meshgrid(xs, ys)
        z = depth_m[vv, uu]

        valid = np.isfinite(z) & (z >= self.config.depth_min) & (z <= self.config.depth_max)
        if not np.any(valid):
            return

        u = uu[valid].astype(np.float32)
        v = vv[valid].astype(np.float32)
        d = z[valid].astype(np.float32)

        x_cam = (u - self.config.cx) * d / self.config.fx
        y_cam = (v - self.config.cy) * d / self.config.fy
        pts_cam = np.stack([x_cam, y_cam, d], axis=1)
        pts_world = transform_points(pts_cam, t_world, q_xyzw)
        colors = rgb_np[vv[valid], uu[valid]].astype(np.float64)

        xw = pts_world[:, 0]
        zw = pts_world[:, 2]
        self.topdown_bounds[0] = min(self.topdown_bounds[0], float(np.min(xw)))
        self.topdown_bounds[1] = min(self.topdown_bounds[1], float(np.min(zw)))
        self.topdown_bounds[2] = max(self.topdown_bounds[2], float(np.max(xw)))
        self.topdown_bounds[3] = max(self.topdown_bounds[3], float(np.max(zw)))

        ix = np.floor(xw / res).astype(np.int32)
        iz = np.floor(zw / res).astype(np.int32)
        for i in range(len(ix)):
            key = (int(ix[i]), int(iz[i]))
            acc = self.topdown_rgb_accum[key]
            acc[0:3] += colors[i]
            acc[3] += 1.0

    def _save_sidebyside_visualization(self, output_dir: str):
        """Save one final top-down side-by-side image: RGB orthomosaic (left), semantic map (right)."""
        if len(self.topdown_rgb_accum) == 0 and len(self.semantic_points) == 0:
            if self.config.verbose:
                print("  No accumulated data for final top-down visualization")
            return

        res = float(self.config.topdown_resolution)
        x_min, z_min, x_max, z_max = self.topdown_bounds

        # Expand bounds with semantic points if needed
        for pt_key in self.semantic_points.keys():
            x, _, z = pt_key
            x_min = min(x_min, x)
            z_min = min(z_min, z)
            x_max = max(x_max, x)
            z_max = max(z_max, z)

        if not np.isfinite([x_min, z_min, x_max, z_max]).all():
            if self.config.verbose:
                print("  Invalid bounds for top-down visualization")
            return

        margin = max(res, 0.1)
        x_min -= margin
        z_min -= margin
        x_max += margin
        z_max += margin

        width = max(64, int(np.ceil((x_max - x_min) / res)))
        height = max(64, int(np.ceil((z_max - z_min) / res)))

        rgb_topdown = np.ones((height, width, 3), dtype=np.uint8) * 235
        semantic_map = np.ones((height, width, 3), dtype=np.uint8) * 12

        # Render true RGB orthomosaic from accumulated per-cell averages
        for (ix, iz), acc in self.topdown_rgb_accum.items():
            if acc[3] <= 0:
                continue
            x = (ix + 0.5) * res
            z = (iz + 0.5) * res
            px = int((x - x_min) / res)
            pz = int((z - z_min) / res)
            if 0 <= px < width and 0 <= pz < height:
                rgb_topdown[pz, px] = np.clip(acc[0:3] / acc[3], 0, 255).astype(np.uint8)

        # Render semantic map with confidence-prioritized color per cell
        semantic_cells = {}
        for pt_key, color in self.semantic_points.items():
            x, _, z = pt_key
            ix = int(np.floor(x / res))
            iz = int(np.floor(z / res))
            conf = float(self.point_confidences.get(pt_key, 0.0))
            existing = semantic_cells.get((ix, iz))
            if existing is None or conf > existing[0]:
                semantic_cells[(ix, iz)] = (conf, color)

        for (ix, iz), (_, color) in semantic_cells.items():
            x = (ix + 0.5) * res
            z = (iz + 0.5) * res
            px = int((x - x_min) / res)
            pz = int((z - z_min) / res)
            if 0 <= px < width and 0 <= pz < height:
                r, g, b = color
                semantic_map[pz, px] = np.array([r, g, b], dtype=np.uint8)

        # Tight-crop around occupied content so the map does not appear tiny in a large blank canvas.
        rgb_occ = np.any(rgb_topdown != 235, axis=2)
        sem_occ = np.any(semantic_map != 12, axis=2)
        occ = rgb_occ | sem_occ
        if np.any(occ):
            ys, xs = np.where(occ)
            pad = 8
            y0 = max(0, int(np.min(ys)) - pad)
            y1 = min(height, int(np.max(ys)) + pad + 1)
            x0 = max(0, int(np.min(xs)) - pad)
            x1 = min(width, int(np.max(xs)) + pad + 1)
            rgb_topdown = rgb_topdown[y0:y1, x0:x1]
            semantic_map = semantic_map[y0:y1, x0:x1]

        # Flip horizontally and vertically for true top-down view (correct both X and Z axes)
        rgb_topdown = cv2.flip(rgb_topdown, -1)
        semantic_map = cv2.flip(semantic_map, -1)

        # Upscale small crops for readability in reports/presentations.
        h_crop, w_crop = rgb_topdown.shape[:2]
        min_display = 500
        scale = max(1.0, min_display / float(max(h_crop, w_crop)))
        if scale > 1.0:
            new_w = max(1, int(round(w_crop * scale)))
            new_h = max(1, int(round(h_crop * scale)))
            rgb_topdown = cv2.resize(rgb_topdown, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            semantic_map = cv2.resize(semantic_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        sidebyside = np.hstack([rgb_topdown, semantic_map])

        viz_dir = os.path.join(output_dir, "topdown_viz")
        os.makedirs(viz_dir, exist_ok=True)
        save_path = os.path.join(viz_dir, "topdown_sidebyside_final.png")
        Image.fromarray(sidebyside).save(save_path)

        if self.config.verbose:
            print(f"  ✓ Final top-down visualization: {save_path}")
            print(f"    Canvas: {width}x{height} px | Resolution: {res:.3f} m/px")


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
            self.pcd_overlay = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.pcd)
            self.vis.add_geometry(self.pcd_overlay)
            opt = self.vis.get_render_option()
            opt.point_size = 2.0
            opt.background_color = np.asarray([0.06, 0.07, 0.10])
        
        self.view_initialized = False

    def _update_monocular_scale_placeholder(self, rgb_pil: Image.Image, depth_m: np.ndarray, frame_idx: int) -> float:
        """
        Placeholder hook for future online scale updaters.

        Current behavior intentionally keeps a fixed user-provided scale.
        """
        if self.config.enable_scale_updater and not self._scale_updater_notice_printed and self.config.verbose:
            print(
                f"Scale updater placeholder active (mode={self.config.scale_updater_mode}, "
                f"ema_alpha={self.config.scale_ema_alpha}). Using fixed scale for now."
            )
            self._scale_updater_notice_printed = True
        return self.current_mono_depth_scale
    
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
        """Update single window view with a geometry layer and semantic overlay."""
        if not self.geometric_points and not self.semantic_points:
            if self.config.verbose and self.stats['frames_processed'] <= 3:
                print(f"  [Live View] No points yet (frame {self.stats['frames_processed']})")
            return
        
        geo_points = np.array(list(self.geometric_points.keys())) if self.geometric_points else np.empty((0, 3))
        geo_colors = np.array(list(self.geometric_points.values())) / 255.0 if self.geometric_points else np.empty((0, 3))
        sem_points = np.array(list(self.semantic_points.keys())) if self.semantic_points else np.empty((0, 3))
        sem_colors = np.array(list(self.semantic_points.values())) / 255.0 if self.semantic_points else np.empty((0, 3))
        
        if self.config.verbose and self.stats['frames_processed'] <= 3:
            print(f"  [Live View] Geo: {len(geo_points)} | Semantic: {len(sem_points)}")
        
        if len(geo_points) > 0:
            self.pcd.points = o3d.utility.Vector3dVector(geo_points)
            self.pcd.colors = o3d.utility.Vector3dVector(geo_colors)
            self.vis.update_geometry(self.pcd)
        
        if self.pcd_overlay is not None and len(sem_points) > 0:
            # Bright overlay points for detections only
            overlay_colors = sem_colors.copy()
            if self.config.vis_transparency_enabled and self.point_confidences:
                confidences = np.array([self.point_confidences.get(tuple(p), 0.5) for p in sem_points])
                bg_color = np.array([0.06, 0.07, 0.10])
                alpha_values = np.clip(confidences, self.config.vis_confidence_min, 1.0) * self.config.vis_transparency_alpha
                for i in range(len(overlay_colors)):
                    overlay_colors[i] = overlay_colors[i] * alpha_values[i] + bg_color * (1 - alpha_values[i])
            
            self.pcd_overlay.points = o3d.utility.Vector3dVector(sem_points)
            self.pcd_overlay.colors = o3d.utility.Vector3dVector(overlay_colors)
            self.vis.update_geometry(self.pcd_overlay)
        
        # Reset view on first update
        if not self.view_initialized and (len(geo_points) > 0 or len(sem_points) > 0):
            self.vis.reset_view_point(True)
            ctr = self.vis.get_view_control()
            ctr.set_zoom(1.6)
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
        
        # Generate final top-down visualization if enabled
        if self.config.save_semantic_sidebyside:
            if self.config.verbose:
                print("\nGenerating final top-down visualization...")
            self._save_sidebyside_visualization(self.config.output_dir)
        
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
            self.current_mono_depth_scale = self._update_monocular_scale_placeholder(rgb_pil, None, 0)
            depth = self._estimate_depth_quick(rgb_pil)
        
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
                self.current_mono_depth_scale = self._update_monocular_scale_placeholder(rgb_pil, None, i)
                depth_m = self._estimate_depth_quick(rgb_pil)
            
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

                if self.config.save_semantic_sidebyside:
                    self._accumulate_topdown_rgb(rgb_pil, depth_m, t_world, q_xyzw)
                
                # Insert into geometric map (pure geometry, no semantics)
                if self.geometric_map is not None:
                    self.geometric_map.insertPointCloud(pts_world_full, origin, self.config.octomap_max_range, False)
                
                # Insert into combined map with default white/gray color (background)
                if self.combined_map is not None:
                    # Insert points with white color for regular geometry
                    for pt in pts_world_full:
                        self.combined_map.updateNode(pt, True)  # occupied
                        self.combined_map.integrateNodeColor(pt, 112, 128, 144)  # slate gray
                        # Track colored point for PLY export
                        pt_key = (float(pt[0]), float(pt[1]), float(pt[2]))
                        self.colored_points[pt_key] = (112, 128, 144)
                        # Track geometry for live visualization
                        self.geometric_points[pt_key] = (120, 120, 120)
                
                self.stats['points_geometric'] += len(pts_world_full)
        
        # 2. Run detection and build semantic map
            # Skip detection if no queries configured (geometry-only baseline)
            detections = None
            if self.config.detection_queries:
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
                label_lc = label.lower()
                is_transparent = any(k in label_lc for k in self.config.transparent_label_keywords)
                min_samples = (
                    self.config.transparent_min_detection_samples
                    if is_transparent else self.config.min_detection_samples
                )
                
                # Extract 3D points from detection bbox
                pts_cam = bbox_points_cam(
                    depth_m, box,
                    self.config.fx, self.config.fy, self.config.cx, self.config.cy,
                    stride=self.config.point_stride,
                    z_min=self.config.depth_min,
                    z_max=self.config.depth_max,
                    fill_interior=self.config.fill_bbox_interior,
                    min_samples=min_samples,
                    enable_boundary_fallback=(
                        is_transparent and self.config.enable_transparent_depth_fallback
                    ),
                    boundary_ring_px=self.config.transparent_context_ring_px,
                    boundary_min_samples=self.config.transparent_fallback_min_samples,
                )
                
                if pts_cam is None:
                    continue
                
                pts_world = transform_points(pts_cam, t_world, q_xyzw)
                
                # Accumulate semantic labels in voxel map
                weight = max(0.0, min(1.0, score))
                for p in pts_world:
                    voxel_key = world_to_voxel_idx(p, self.config.voxel_resolution)
                    self.semantic_voxels[voxel_key][label] += weight
                
                # Update colors in combined map for semantic regions
                if self.combined_map is not None:
                    # Use confidence-based colormapping if enabled
                    if self.config.vis_confidence_colormapping:
                        # Map confidence to color using colormap
                        r, g, b = self._confidence_to_color(score)
                    else:
                        # Fallback to bright red for detected glass/transparent regions
                        r, g, b = 255, 0, 0  # Bright red
                    
                    for pt in pts_world:
                        self.combined_map.updateNode(pt, True)
                        self.combined_map.integrateNodeColor(pt, r, g, b)
                        # Update colored point for PLY export and live view
                        pt_key = (float(pt[0]), float(pt[1]), float(pt[2]))
                        self.colored_points[pt_key] = (r, g, b)
                        # Track confidence for visualization
                        self.point_confidences[pt_key] = score
                        self.semantic_points[pt_key] = (r, g, b)
                
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
            combined_path = os.path.join(output_dir, "combined_map.ot")
            # ColorOcTree should be written as .ot to retain voxel color data.
            self.combined_map.write(combined_path)
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
        "--depth-model", type=str, default="da2",
        choices=["da2", "da3"],
        help="Depth model to use for monocular estimation (default: da2)"
    )
    parser.add_argument(
        "--mono-depth-scale", type=float, default=1.0,
        help="Global multiplier for monocular estimated depth (default: 1.0)"
    )
    parser.add_argument(
        "--enable-scale-updater", action="store_true",
        help="Enable placeholder hook for future online monocular scale updater"
    )
    parser.add_argument(
        "--scale-updater", type=str, default="none",
        choices=["none", "floor", "object", "ema"],
        help="Placeholder updater mode for future online scale adaptation"
    )
    parser.add_argument(
        "--scale-ema-alpha", type=float, default=0.02,
        help="Placeholder EMA alpha for future updater (default: 0.02)"
    )
    parser.add_argument(
        "--enable-depth-dependent-scale", action="store_true",
        help="Enable depth-dependent nonlinear scale correction (fixes geometry distortion)"
    )
    parser.add_argument(
        "--scale-near", type=float, default=1.3088,
        help="Scale factor at near depth (d=0) for depth-dependent correction (default: 1.3088)"
    )
    parser.add_argument(
        "--scale-far", type=float, default=0.9184,
        help="Scale factor at far depth for depth-dependent correction (default: 0.9184)"
    )
    parser.add_argument(
        "--scale-depth-range", type=float, default=3.0,
        help="Depth range for linear interpolation in depth-dependent correction (default: 3.0m)"
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
        "--min-detection-samples", type=int, default=80,
        help="Minimum valid depth samples for non-transparent detections (default: 80)"
    )
    parser.add_argument(
        "--transparent-min-samples", type=int, default=20,
        help="Minimum valid depth samples for transparent detections (default: 20)"
    )
    parser.add_argument(
        "--disable-transparent-fallback", action="store_true",
        help="Disable boundary-context depth fallback for transparent detections"
    )
    parser.add_argument(
        "--transparent-ring-px", type=int, default=16,
        help="Ring width in pixels around transparent detections for depth fallback (default: 16)"
    )
    parser.add_argument(
        "--transparent-ring-min-samples", type=int, default=12,
        help="Minimum valid ring samples required for transparent fallback (default: 12)"
    )
    parser.add_argument(
        "--no-fill-bbox-interior", action="store_true",
        help="Disable interior depth filling inside detection bounding boxes"
    )
    parser.add_argument(
        "--debug-images", action="store_true",
        help="Save debug images with detection overlays"
    )
    parser.add_argument(
        "--sidebyside", action="store_true",
        help="Save one final top-down RGB orthomosaic + semantic map side-by-side image"
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
        depth_source_auto_detect=not args.estimate_depth,  # Disable auto-detect when forcing estimation
        depth_model=args.depth_model,
        monocular_depth_scale=args.mono_depth_scale,
        enable_scale_updater=args.enable_scale_updater,
        scale_updater_mode=args.scale_updater,
        scale_ema_alpha=args.scale_ema_alpha,
        enable_depth_dependent_scale=args.enable_depth_dependent_scale,
        scale_near=args.scale_near,
        scale_far=args.scale_far,
        scale_depth_range=args.scale_depth_range,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        detector_model=args.detector,
        detection_threshold=args.threshold,
        min_detection_samples=args.min_detection_samples,
        transparent_min_detection_samples=args.transparent_min_samples,
        fill_bbox_interior=not args.no_fill_bbox_interior,
        enable_transparent_depth_fallback=not args.disable_transparent_fallback,
        transparent_context_ring_px=args.transparent_ring_px,
        transparent_fallback_min_samples=args.transparent_ring_min_samples,
        save_debug_images=args.debug_images,
        save_semantic_sidebyside=args.sidebyside,
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
