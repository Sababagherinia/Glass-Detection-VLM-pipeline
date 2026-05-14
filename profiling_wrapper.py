#!/usr/bin/env python3
"""
Lightweight profiling wrapper for UnifiedPipeline.
This file monkey-patches the detector call and the frame processor to collect
per-frame wall time, detection time, and memory (RSS/VMS). It writes results
to a CSV and does not modify `unified_pipeline.py` on disk.

Usage:
    python profiling_wrapper.py --dataset data/rgbd_dataset_freiburg1_360 --output output/unified --profile-out output/unified/profile.csv

Note: This wrapper imports and constructs `PipelineConfig` and `UnifiedPipeline`
from `unified_pipeline` and then runs `process_stream()` after applying runtime
wrapping. It intentionally does not change the original source file.
"""
import argparse
import time
import csv
import os
import sys
from typing import Dict

# Optional memory tools
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False

try:
    import tracemalloc
    TRACEMALLOC_AVAILABLE = True
except Exception:
    TRACEMALLOC_AVAILABLE = False

# Import the pipeline module
try:
    import unified_pipeline
except Exception as e:
    print(f"Error importing unified_pipeline: {e}")
    raise

# Try to import pyoctomap for availability check
try:
    import pyoctomap
    _PYOCTOMAP_AVAILABLE = True
except Exception:
    _PYOCTOMAP_AVAILABLE = False


def make_profiled_pipeline(config: unified_pipeline.PipelineConfig, profile_out: str, use_tracemalloc: bool = False):
    """Create a UnifiedPipeline instance and apply runtime wrappers for profiling."""
    os.makedirs(os.path.dirname(profile_out) or '.', exist_ok=True)
    fp = open(profile_out, "w", newline="")
    writer = csv.writer(fp)
    header = [
        "frame_idx", "timestamp", "wall_time_s", "detection_time_s",
        "points_geometric", "points_semantic", "rss_mb", "vms_mb",
        "tracemalloc_current_kb", "tracemalloc_peak_kb"
    ]
    writer.writerow(header)
    fp.flush()

    # Instantiate pipeline
    pipeline = unified_pipeline.UnifiedPipeline(config)

    # If user requested tracemalloc and available, start it
    if use_tracemalloc and TRACEMALLOC_AVAILABLE:
        tracemalloc.start()
    elif use_tracemalloc and not TRACEMALLOC_AVAILABLE:
        print("Warning: tracemalloc not available; continuing without it")

    # Save originals
    orig_proc = unified_pipeline.UnifiedPipeline._process_single_frame

    def _install_map_timers(pipeline_obj):
        """Install OctoMap wrappers once and attach a mutable timing state to the pipeline instance."""
        if hasattr(pipeline_obj, "_profile_map_timing_state"):
            return pipeline_obj._profile_map_timing_state

        timing_state = {"geometric": 0.0, "combined": 0.0}
        pipeline_obj._profile_map_timing_state = timing_state

        if getattr(pipeline_obj, "geometric_map", None) is not None and hasattr(pipeline_obj.geometric_map, "insertPointCloud"):
            pipeline_obj._profile_orig_geometric_insert = pipeline_obj.geometric_map.insertPointCloud

            def insert_wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return pipeline_obj._profile_orig_geometric_insert(*args, **kwargs)
                finally:
                    timing_state["geometric"] += (time.perf_counter() - t0)

            pipeline_obj.geometric_map.insertPointCloud = insert_wrapper

        if getattr(pipeline_obj, "combined_map", None) is not None:
            if hasattr(pipeline_obj.combined_map, "updateNode"):
                pipeline_obj._profile_orig_combined_update = pipeline_obj.combined_map.updateNode

                def update_node_wrapper(*args, **kwargs):
                    t0 = time.perf_counter()
                    try:
                        return pipeline_obj._profile_orig_combined_update(*args, **kwargs)
                    finally:
                        timing_state["combined"] += (time.perf_counter() - t0)

                pipeline_obj.combined_map.updateNode = update_node_wrapper

            if hasattr(pipeline_obj.combined_map, "integrateNodeColor"):
                pipeline_obj._profile_orig_combined_color = pipeline_obj.combined_map.integrateNodeColor

                def integrate_color_wrapper(*args, **kwargs):
                    t0 = time.perf_counter()
                    try:
                        return pipeline_obj._profile_orig_combined_color(*args, **kwargs)
                    finally:
                        timing_state["combined"] += (time.perf_counter() - t0)

                pipeline_obj.combined_map.integrateNodeColor = integrate_color_wrapper

        return timing_state

    def _wrapped_process_single_frame(self, rgb_pil, depth_m, t_world, q_xyzw, timestamp, frame_idx):
        # Prepare detection wrapper to capture detection time
        detect_time = {"t": 0.0}

        if hasattr(self, 'detector') and hasattr(self.detector, 'detect'):
            orig_detect = self.detector.detect

            def detect_wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    res = orig_detect(*args, **kwargs)
                finally:
                    t1 = time.perf_counter()
                    detect_time['t'] += (t1 - t0)
                return res

            # Replace detector method for this call
            self.detector.detect = detect_wrapper

        # Wall-clock for frame
        t0 = time.perf_counter()
        try:
            # Call original processing (will use our wrapped detect)
            orig_proc(self, rgb_pil, depth_m, t_world, q_xyzw, timestamp, frame_idx)
        finally:
            t1 = time.perf_counter()
            wall = t1 - t0

            # Memory stats
            rss_mb = vms_mb = 0.0
            trac_cur = trac_peak = 0
            if PSUTIL_AVAILABLE:
                try:
                    proc = psutil.Process()
                    mi = proc.memory_info()
                    rss_mb = mi.rss / (1024.0 * 1024.0)
                    vms_mb = mi.vms / (1024.0 * 1024.0)
                except Exception:
                    pass
            if use_tracemalloc and TRACEMALLOC_AVAILABLE:
                try:
                    cur, peak = tracemalloc.get_traced_memory()
                    trac_cur = int(cur / 1024)
                    trac_peak = int(peak / 1024)
                except Exception:
                    pass

            # Points counters
            pts_geo = int(self.stats.get('points_geometric', 0))
            pts_sem = int(self.stats.get('points_semantic', 0))

            row = [
                int(frame_idx), float(timestamp), f"{wall:.6f}", f"{detect_time['t']:.6f}",
                pts_geo, pts_sem, f"{rss_mb:.3f}", f"{vms_mb:.3f}", int(trac_cur), int(trac_peak)
            ]
            try:
                writer.writerow(row)
                fp.flush()
            except Exception:
                pass

            # Restore original detect method if we replaced it
            if hasattr(self, 'detector') and hasattr(self.detector, 'detect') and 'orig_detect' in locals():
                try:
                    self.detector.detect = orig_detect
                except Exception:
                    pass

    # Patch the pipeline class method
    unified_pipeline.UnifiedPipeline._process_single_frame = _wrapped_process_single_frame

    return pipeline, fp


def map_output_sizes(output_dir: str) -> Dict[str, int]:
    """Return on-disk sizes for the saved map files, if present."""
    files = {
        "geometric_map_bytes": os.path.join(output_dir, "geometric_map.bt"),
        "combined_map_bytes": os.path.join(output_dir, "combined_map.ot"),
        "semantic_only_map_bytes": os.path.join(output_dir, "semantic_only_map.bt"),
        "combined_voxels_ply_bytes": os.path.join(output_dir, "combined_voxels.ply"),
    }
    sizes = {}
    for key, path in files.items():
        try:
            sizes[key] = os.path.getsize(path) if os.path.exists(path) else 0
        except Exception:
            sizes[key] = 0
    return sizes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/rgbd_dataset_freiburg1_360')
    parser.add_argument('--output', type=str, default='output/unified')
    parser.add_argument('--profile-out', type=str, default='output/unified/profile.csv')
    parser.add_argument('--tracemalloc', action='store_true')
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--frame-step', type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    # Build config using PipelineConfig from module (keeps main code untouched)
    cfg = unified_pipeline.PipelineConfig(
        dataset_dir=args.dataset,
        output_dir=args.output,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        verbose=True
    )

    pipeline, fp = make_profiled_pipeline(cfg, args.profile_out, use_tracemalloc=args.tracemalloc)

    try:
        pipeline.process_stream(args.dataset)
    finally:
        # Close CSV and stop tracemalloc if used
        try:
            fp.close()
        except Exception:
            pass
        if args.tracemalloc and TRACEMALLOC_AVAILABLE:
            try:
                tracemalloc.stop()
            except Exception:
                pass


if __name__ == '__main__':
    main()
