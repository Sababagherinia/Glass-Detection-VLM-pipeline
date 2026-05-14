#!/usr/bin/env python3
"""
Batch profiling experiments with different detection query sets.

This script runs the pipeline multiple times with different sets of detection queries,
collects profiling stats (wall time, memory, detections), and generates a comparison
summary so you can see how different semantic integrations perform on the same data.

Usage:
    python batch_profiling_experiments.py --dataset data/rgbd_dataset_freiburg1_360

Output:
    - experiments/ directory with subdirs for each query set
    - experiment_summary.csv with aggregated stats across all runs
    - Per-experiment profile.csv with per-frame timings
"""

import os
import sys
import argparse
import subprocess
import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Define different query sets to test (completely different semantic domains)
QUERY_SETS = {
    "geometry_baseline": {
        "description": "Geometry only: NO detections (baseline for timing comparison)",
        "queries": []  # Empty queries = pure geometric map build
    },
    "glass_transparent": {
        "description": "Glass & Transparent: windows, mirrors, glass surfaces",
        "queries": [
            "glass", "transparent", "window", "mirror", "reflective",
            "a glass window", "a glass door", "a transparent surface",
            "shiny surface", "glossy surface", "a mirror"
        ]
    },
    "people": {
        "description": "People: humans, faces, bodies, persons",
        "queries": [
            "a person", "a human", "a man", "a woman", "a child",
            "a face", "a head", "a body", "a people", "crowd of people",
            "a person standing", "a human figure"
        ]
    },
    "furniture": {
        "description": "Furniture: chairs, desks, tables, sofas",
        "queries": [
            "a chair", "a desk", "a table", "sofa", "couch", "a shelf", "a bookshelf", "a piece of furniture",
            "a wooden chair", "office furniture", "a computer desk", "a computer table", "a computer", "a white board"
        ]
    },
    "plants_nature": {
        "description": "Nature: trees, plants, leaves, grass, flowers",
        "queries": [
            "tree", "plant", "leaf", "leaves", "grass", "flower", "vegetation",
            "a potted plant", "a tree trunk", "a flowering plant"
        ]
    },
    # "vehicles": {
    #     "description": "Vehicles: cars, bikes, motorcycles, vehicles",
    #     "queries": [
    #         "car", "bicycle", "bike", "motorcycle", "vehicle",
    #         "wheel", "tire", "vehicle", "truck", "van",
    #         "a parked car", "a bicycle"
    #     ]
    # },
    "walls_floors": {
        "description": "Surfaces: walls, floors, ceilings",
        "queries": [
            "walls","a wall", "floor", "ceiling", "wooden floor", "wooden wall",
            "brick", "wood", "carpet", "a tiled floor", "a brick wall"
        ]
    },
    # "electronics": {
    #     "description": "Electronics: screens, monitors, keyboards, devices",
    #     "queries": [
    #         "screen", "monitor", "keyboard", "mouse", "device",
    #         "computer", "laptop", "phone", "tablet",
    #         "a computer monitor", "a smartphone", "a keyboard"
    #     ]
    # },
    "minimal_single": {
        "description": "Minimal: single object class (glass only)",
        "queries": ["glass"]
    },
    "comprehensive": {
        "description": "Comprehensive: all objects across domains",
        "queries": [
            "glass", "a window", "a person", "a people","a human", "chair", "desk", "tree",
            "car", "wall", "floor", "door", "screen", "plant",
            "table", "flower", "window", "transparent", "a glass window", "a person standing", "a wooden chair", "a white board"
        ]
    }
}


def parse_profile_csv(csv_path: str) -> Dict:
    """Parse a profile.csv and compute aggregate statistics."""
    if not os.path.exists(csv_path):
        return None
    
    stats = {
        "frame_count": 0,
        "wall_time": [],
        "detection_time": [],
        "points_geometric": [],
        "points_semantic": [],
        "rss_mb": [],
        "vms_mb": []
    }
    
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats["frame_count"] += 1
                stats["wall_time"].append(float(row.get("wall_time_s", 0)))
                stats["detection_time"].append(float(row.get("detection_time_s", 0)))
                stats["points_geometric"].append(int(float(row.get("points_geometric", 0))))
                stats["points_semantic"].append(int(float(row.get("points_semantic", 0))))
                stats["rss_mb"].append(float(row.get("rss_mb", 0)))
                stats["vms_mb"].append(float(row.get("vms_mb", 0)))
    except Exception as e:
        print(f"Warning: Could not parse {csv_path}: {e}")
        return None
    
    return stats


def compute_aggregate_stats(stats: Dict) -> Dict:
    """Compute mean, std, min, max for aggregated stats."""
    agg = {}
    for key in ["wall_time", "detection_time", "rss_mb", "vms_mb"]:
        if stats[key]:
            arr = np.array(stats[key])
            agg[f"{key}_total"] = float(np.sum(arr))
            agg[f"{key}_mean"] = float(np.mean(arr))
            agg[f"{key}_std"] = float(np.std(arr))
            agg[f"{key}_min"] = float(np.min(arr))
            agg[f"{key}_max"] = float(np.max(arr))
    
    # Points: total and average
    if stats["points_geometric"]:
        agg["points_geometric_total"] = int(np.sum(stats["points_geometric"]))
        agg["points_geometric_mean"] = float(np.mean(stats["points_geometric"]))
    if stats["points_semantic"]:
        agg["points_semantic_total"] = int(np.sum(stats["points_semantic"]))
        agg["points_semantic_mean"] = float(np.mean(stats["points_semantic"]))
    
    agg["frame_count"] = stats["frame_count"]
    return agg


def compute_overall_row(results: List[Dict]) -> Dict:
    """Compute one overall row across all tests (weighted by frame count)."""
    overall = {
        "experiment_name": "ALL_TESTS_WEIGHTED",
        "description": "Frame-weighted overall averages across successful experiments",
        "num_queries": int(sum(r.get("num_queries", 0) for r in results)),
        "frame_count": int(sum(r.get("frame_count", 0) for r in results)),
    }

    total_frames = overall["frame_count"]
    if total_frames <= 0:
        return overall

    # Totals across tests
    wall_total = float(sum(r.get("wall_time_total", 0.0) for r in results))
    det_total = float(sum(r.get("detection_time_total", 0.0) for r in results))
    overall["wall_time_total"] = wall_total
    overall["detection_time_total"] = det_total

    # Weighted means from totals
    overall["wall_time_mean"] = wall_total / total_frames
    overall["detection_time_mean"] = det_total / total_frames

    # Range across tests (not per-frame-global, but useful bound)
    overall["wall_time_min"] = float(min(r.get("wall_time_min", np.inf) for r in results))
    overall["wall_time_max"] = float(max(r.get("wall_time_max", -np.inf) for r in results))
    overall["detection_time_min"] = float(min(r.get("detection_time_min", np.inf) for r in results))
    overall["detection_time_max"] = float(max(r.get("detection_time_max", -np.inf) for r in results))

    # Weighted average memory usage
    overall["rss_mb_mean"] = float(
        sum(r.get("rss_mb_mean", 0.0) * r.get("frame_count", 0) for r in results) / total_frames
    )
    overall["vms_mb_mean"] = float(
        sum(r.get("vms_mb_mean", 0.0) * r.get("frame_count", 0) for r in results) / total_frames
    )

    # Keep totals for points; weighted means derived from totals
    overall["points_geometric_total"] = int(sum(r.get("points_geometric_total", 0) for r in results))
    overall["points_semantic_total"] = int(sum(r.get("points_semantic_total", 0) for r in results))
    overall["points_geometric_mean"] = float(overall["points_geometric_total"] / total_frames)
    overall["points_semantic_mean"] = float(overall["points_semantic_total"] / total_frames)

    return overall


def collect_map_sizes(output_dir: str) -> Dict[str, int]:
    """Collect saved map file sizes in bytes for one experiment output directory."""
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


def run_experiment(dataset_path: str, output_dir: str, query_set_name: str, 
                   queries: List[str], frame_step: int, max_frames: int = None,
                   tracemalloc: bool = False, verbose: bool = True) -> bool:
    """
    Run profiling_wrapper.py with a specific query set.
    
    Returns True if successful, False otherwise.
    """
    exp_output = os.path.join(output_dir, query_set_name)
    os.makedirs(exp_output, exist_ok=True)
    profile_csv = os.path.join(exp_output, "profile.csv")
    
    # Build command to modify config and run
    cmd = [
        sys.executable, "profiling_wrapper.py",
        "--dataset", dataset_path,
        "--output", exp_output,
        "--profile-out", profile_csv,
        "--frame-step", str(frame_step)
    ]
    
    if max_frames:
        cmd.extend(["--max-frames", str(max_frames)])
    
    if tracemalloc:
        cmd.append("--tracemalloc")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Running experiment: {query_set_name}")
        print(f"Queries: {queries}")
        print(f"Output: {exp_output}")
        print(f"{'='*70}")
    
    try:
        # Note: We need to patch the queries in the config before running.
        # Since profiling_wrapper.py doesn't expose a --queries flag yet,
        # we'll modify the run to create a modified wrapper on the fly.
        # For now, this runs with default queries. We'll extend this below.
        
        # Create a temporary wrapper that uses the specified queries
        project_root = os.path.dirname(os.path.abspath(__file__))
        wrapper_code = _create_temp_wrapper(queries, project_root=project_root)
        temp_wrapper = os.path.join(exp_output, f"_wrapper_{query_set_name}.py")
        with open(temp_wrapper, "w") as f:
            f.write(wrapper_code)
        
        # Run the temporary wrapper
        result = subprocess.run(
            [sys.executable, temp_wrapper, "--dataset", dataset_path, 
             "--output", exp_output, "--profile-out", profile_csv, 
             "--frame-step", str(frame_step)] + 
            (["--max-frames", str(max_frames)] if max_frames else []) +
            (["--tracemalloc"] if tracemalloc else []),
            capture_output=True, text=True, timeout=3600
        )
        
        if result.returncode != 0:
            print(f"Error running experiment {query_set_name}:")
            print(result.stderr)
            return False
        
        if verbose:
            print(result.stdout)
        
        return True
    
    except Exception as e:
        print(f"Error running experiment {query_set_name}: {e}")
        return False


def _create_temp_wrapper(queries: List[str], project_root: str = None) -> str:
    """Create a temporary wrapper script with specific queries."""
    queries_str = ", ".join([f'"{q}"' for q in queries])
    
    # Determine project root (where unified_pipeline.py lives)
    if project_root is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
    
    code = f'''#!/usr/bin/env python3
import sys
import os
# Add project root to path so we can import unified_pipeline and profiling_wrapper
sys.path.insert(0, "{project_root}")

import unified_pipeline
import profiling_wrapper
import argparse
import time
import csv

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

    # Build config with custom queries
    cfg = unified_pipeline.PipelineConfig(
        dataset_dir=args.dataset,
        output_dir=args.output,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        detection_queries=[{queries_str}],
        verbose=True
    )

    pipeline, fp = profiling_wrapper.make_profiled_pipeline(
        cfg, args.profile_out, use_tracemalloc=args.tracemalloc
    )

    try:
        pipeline.process_stream(args.dataset)
    finally:
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
'''
    return code


def generate_summary_report(output_dir: str, experiments: Dict[str, Dict]) -> str:
    """Generate a summary report of all experiments."""
    summary_csv = os.path.join(output_dir, "experiment_summary.csv")
    
    # Collect all experiment results
    results = []
    geometry_baseline_time = None
    
    for exp_name, exp_config in experiments.items():
        exp_dir = os.path.join(output_dir, exp_name)
        profile_csv = os.path.join(exp_dir, "profile.csv")
        
        stats = parse_profile_csv(profile_csv)
        if stats is None:
            print(f"Warning: No profiling data for {exp_name}")
            continue
        
        agg = compute_aggregate_stats(stats)
        agg["experiment_name"] = exp_name
        agg["description"] = exp_config["description"]
        agg["num_queries"] = len(exp_config["queries"])
        agg.update(collect_map_sizes(exp_dir))
        results.append(agg)
        
        # Store geometry baseline time if this is the baseline experiment
        if exp_name == "geometry_baseline":
            geometry_baseline_time = agg.get("wall_time_mean", 0)
    
    # Add semantic overhead for each experiment (vs geometry baseline)
    if geometry_baseline_time is not None:
        for r in results:
            if r["experiment_name"] != "geometry_baseline":
                r["semantic_overhead_ms"] = (r.get("wall_time_mean", 0) - geometry_baseline_time) * 1000
            else:
                r["semantic_overhead_ms"] = 0.0
    
    if not results:
        print("No experiment results to summarize")
        return None
    
    # Add overall row (weighted by frame count)
    overall_row = compute_overall_row(results)
    results_with_overall = results + [overall_row]

    # Write summary CSV
    if results:
        # Get all fieldnames
        preferred_order = [
            "experiment_name", "description", "num_queries", "frame_count",
            "wall_time_total", "wall_time_mean", "semantic_overhead_ms",
            "detection_time_total", "detection_time_mean",
            "rss_mb_mean", "rss_mb_std", "rss_mb_min", "rss_mb_max",
            "vms_mb_mean", "vms_mb_std", "vms_mb_min", "vms_mb_max",
            "points_geometric_total", "points_geometric_mean",
            "points_semantic_total", "points_semantic_mean",
            "geometric_map_bytes", "combined_map_bytes", "semantic_only_map_bytes", "combined_voxels_ply_bytes",
        ]
        all_keys = set()
        for r in results_with_overall:
            all_keys.update(r.keys())
        fieldnames = [k for k in preferred_order if k in all_keys] + [k for k in sorted(all_keys) if k not in preferred_order]
        
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_with_overall)
        
        print(f"\n{'='*70}")
        print(f"Summary report written to: {summary_csv}")
        print(f"{'='*70}")
        print_summary_table(results_with_overall)
        
        return summary_csv
    
    return None


def print_summary_table(results: List[Dict]):
    """Print a nice ASCII table of results."""
    print("\nExperiment Results Summary")
    print("-" * 160)
    
    # Header
    header = f"{'Experiment':<20} {'Queries':<8} {'Wall(ms)':<12} {'Overhead(ms)':<14} {'Detect(ms)':<12} {'GeoMB':<10} {'CombMB':<10} {'RSS(MB)':<12}"
    print(header)
    print("-" * 160)
    
    for r in results:
        name = r.get("experiment_name", "?")[:19]
        nq = r.get("num_queries", 0)
        wall_mean = r.get("wall_time_mean", 0) * 1000
        overhead = r.get("semantic_overhead_ms", 0)
        det_mean = r.get("detection_time_mean", 0) * 1000
        geo_mb = r.get("geometric_map_bytes", 0) / (1024.0 * 1024.0)
        comb_mb = r.get("combined_map_bytes", 0) / (1024.0 * 1024.0)
        rss_mean = r.get("rss_mb_mean", 0)
        
        row = f"{name:<20} {nq:<8} {wall_mean:<12.2f} {overhead:<14.2f} {det_mean:<12.2f} {geo_mb:<10.2f} {comb_mb:<10.2f} {rss_mean:<12.2f}"
        print(row)
    
    print("-" * 160)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batch profiling experiments with different query sets"
    )
    parser.add_argument(
        "--dataset", type=str, default="data/rgbd_dataset_freiburg1_360",
        help="Path to dataset"
    )
    parser.add_argument(
        "--output", type=str, default="output/experiments",
        help="Base output directory for all experiments"
    )
    parser.add_argument(
        "--frame-step", type=int, default=5,
        help="Frame sampling stride (default: 5)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit number of frames per experiment"
    )
    parser.add_argument(
        "--experiments", type=str, nargs="+", 
        default=list(QUERY_SETS.keys()),
        help=f"Which experiments to run (default: all). Choices: {list(QUERY_SETS.keys())}"
    )
    parser.add_argument(
        "--tracemalloc", action="store_true",
        help="Enable Python allocation profiling"
    )
    parser.add_argument(
        "--no-summary", action="store_true",
        help="Skip generating summary report"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\nBatch Profiling Experiments")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Frame step: {args.frame_step}")
    if args.max_frames:
        print(f"Max frames per experiment: {args.max_frames}")
    
    # Select experiments
    experiments = {
        name: QUERY_SETS[name]
        for name in args.experiments
        if name in QUERY_SETS
    }
    
    if not experiments:
        print("Error: No valid experiments selected")
        sys.exit(1)
    
    print(f"Experiments to run: {list(experiments.keys())}")
    
    # Run each experiment
    successful = {}
    for exp_name, exp_config in experiments.items():
        success = run_experiment(
            args.dataset, args.output, exp_name,
            exp_config["queries"],
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            tracemalloc=args.tracemalloc
        )
        if success:
            successful[exp_name] = exp_config
    
    # Generate summary
    if not args.no_summary and successful:
        generate_summary_report(args.output, successful)
    
    print(f"\n{'='*70}")
    print(f"Batch profiling complete!")
    print(f"Results in: {args.output}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
