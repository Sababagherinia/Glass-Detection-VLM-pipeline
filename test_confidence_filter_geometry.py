#!/usr/bin/env python3
"""
Analyze geometry (XY/XZ axis ratios) before and after confidence filtering.
Compares room proportions to see if filtering fixes distortion.
"""

import os
import csv
import numpy as np
import argparse
from pathlib import Path

def load_confidence_frame_metrics(csv_path):
    """Load per-frame confidence scores from diagnostic output."""
    metrics = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(float(row['frame_idx']))
            metrics[frame_idx] = {
                'combined_conf': float(row['mean_combined_conf']),
                'retain_ratio': float(row['retain_ratio'])
            }
    return metrics

def load_frame_metrics_csv(csv_path):
    """Load per-frame geometry metrics from shape distortion diagnostic."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

def compute_summary_stats(values):
    """Compute min, p10, median, mean, p90, max."""
    if not values:
        return None
    arr = sorted(values)
    n = len(arr)
    def q(p):
        return arr[int((n-1)*p)]
    
    return {
        'n': n,
        'min': arr[0],
        'p10': q(0.1),
        'median': np.median(arr),
        'mean': np.mean(arr),
        'p90': q(0.9),
        'max': arr[-1]
    }

def main():
    parser = argparse.ArgumentParser(description='Compare geometry before/after confidence filtering')
    parser.add_argument('--conf-csv', default='output/depth_confidence_filter/confidence_frame_metrics.csv',
                        help='Confidence metrics CSV from test_depth_confidence_filter.py')
    parser.add_argument('--geom-csv', default='output/depth_shape_diagnostic/frame_metrics.csv',
                        help='Geometry metrics CSV from test_depth_shape_distortion.py')
    parser.add_argument('--conf-threshold', type=float, default=0.55,
                        help='Confidence threshold used during filtering')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.conf_csv):
        print(f"Error: Confidence CSV not found: {args.conf_csv}")
        return
    if not os.path.exists(args.geom_csv):
        print(f"Error: Geometry CSV not found: {args.geom_csv}")
        return
    
    print("Loading confidence scores...")
    conf_metrics = load_confidence_frame_metrics(args.conf_csv)
    
    print("Loading geometry metrics...")
    geom_rows = load_frame_metrics_csv(args.geom_csv)
    
    # Filter rows to only those with confidence data
    frame_indices = list(conf_metrics.keys())
    filtered_rows = [r for r in geom_rows if int(float(r['frame_idx'])) in frame_indices]
    
    print(f"Analyzing {len(filtered_rows)} frames with confidence data")
    
    # Extract metrics for ALL frames (treat as "raw")
    # Compute ratios from extents since they're not pre-computed
    raw_xy = []
    raw_xz = []
    raw_yz = []
    raw_xy_gt = []
    raw_xz_gt = []
    raw_scale = []
    
    for r in filtered_rows:
        x_est = float(r['cam_extent_x_est'])
        y_est = float(r['cam_extent_y_est'])
        z_est = float(r['cam_extent_z_est'])
        x_gt = float(r['cam_extent_x_gt'])
        y_gt = float(r['cam_extent_y_gt'])
        z_gt = float(r['cam_extent_z_gt'])
        
        if y_est > 1e-6:
            raw_xy.append(x_est / y_est)
        if z_est > 1e-6:
            raw_xz.append(x_est / z_est)
        if z_est > 1e-6 and y_est > 1e-6:
            raw_yz.append(y_est / z_est)
        
        if y_gt > 1e-6:
            raw_xy_gt.append(x_gt / y_gt)
        if z_gt > 1e-6:
            raw_xz_gt.append(x_gt / z_gt)
        
        raw_scale.append(float(r['frame_scale_to_gt']))
    
    # Extract metrics for HIGH-CONFIDENCE frames only
    high_conf_rows = [r for r in filtered_rows 
                      if conf_metrics[int(float(r['frame_idx']))]['combined_conf'] >= args.conf_threshold]
    
    print(f"High-confidence frames (conf >= {args.conf_threshold}): {len(high_conf_rows)} / {len(filtered_rows)}")
    
    if not high_conf_rows:
        print("No high-confidence frames found!")
        return
    
    filt_xy = []
    filt_xz = []
    filt_yz = []
    filt_scale = []
    
    for r in high_conf_rows:
        x_est = float(r['cam_extent_x_est'])
        y_est = float(r['cam_extent_y_est'])
        z_est = float(r['cam_extent_z_est'])
        
        if y_est > 1e-6:
            filt_xy.append(x_est / y_est)
        if z_est > 1e-6:
            filt_xz.append(x_est / z_est)
        if z_est > 1e-6 and y_est > 1e-6:
            filt_yz.append(y_est / z_est)
        
        filt_scale.append(float(r['frame_scale_to_gt']))
    
    # Compute statistics
    raw_xy_stats = compute_summary_stats(raw_xy)
    raw_xz_stats = compute_summary_stats(raw_xz)
    raw_yz_stats = compute_summary_stats(raw_yz)
    raw_scale_stats = compute_summary_stats(raw_scale)
    raw_xy_gt_stats = compute_summary_stats(raw_xy_gt)
    raw_xz_gt_stats = compute_summary_stats(raw_xz_gt)
    
    filt_xy_stats = compute_summary_stats(filt_xy)
    filt_xz_stats = compute_summary_stats(filt_xz)
    filt_yz_stats = compute_summary_stats(filt_yz)
    filt_scale_stats = compute_summary_stats(filt_scale)
    
    # Output report
    output_dir = Path('output/depth_confidence_filter')
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'geometry_comparison.txt'
    
    with open(report_path, 'w') as f:
        f.write("=== GEOMETRY COMPARISON: RAW vs HIGH-CONFIDENCE FILTERED ===\n\n")
        f.write(f"Confidence threshold: {args.conf_threshold}\n")
        f.write(f"Raw frames: {len(filtered_rows)}\n")
        f.write(f"Filtered frames (high-conf only): {len(high_conf_rows)}\n\n")
        
        def write_axis(axis_name, raw_stats, filt_stats, indent=""):
            f.write(f"{indent}{axis_name} Ratio\n")
            f.write(f"{indent}  RAW:      median={raw_stats['median']:.4f}  mean={raw_stats['mean']:.4f}  "
                   f"p10={raw_stats['p10']:.4f}  p90={raw_stats['p90']:.4f}\n")
            f.write(f"{indent}  FILTERED: median={filt_stats['median']:.4f}  mean={filt_stats['mean']:.4f}  "
                   f"p10={filt_stats['p10']:.4f}  p90={filt_stats['p90']:.4f}\n")
            med_delta = ((filt_stats['median'] - raw_stats['median']) / raw_stats['median']) * 100
            mean_delta = ((filt_stats['mean'] - raw_stats['mean']) / raw_stats['mean']) * 100
            f.write(f"{indent}  Δ (median): {med_delta:+.1f}% | Δ (mean): {mean_delta:+.1f}%\n\n")
        
        f.write("AXIS RATIOS (In Camera Frame):\n\n")
        write_axis("XY", raw_xy_stats, filt_xy_stats, "  ")
        write_axis("XZ", raw_xz_stats, filt_xz_stats, "  ")
        write_axis("YZ", raw_yz_stats, filt_yz_stats, "  ")
        
        f.write("\nGROUND TRUTH REFERENCE (for comparison):\n")
        f.write(f"  XY (GT): median={raw_xy_gt_stats['median']:.4f}  mean={raw_xy_gt_stats['mean']:.4f}\n")
        f.write(f"  XZ (GT): median={raw_xz_gt_stats['median']:.4f}  mean={raw_xz_gt_stats['mean']:.4f}\n")
        
        f.write("\nDEPTH SCALE (Frame Median / GT Median):\n")
        f.write(f"  RAW:      median={raw_scale_stats['median']:.4f}  mean={raw_scale_stats['mean']:.4f}  "
               f"p10={raw_scale_stats['p10']:.4f}  p90={raw_scale_stats['p90']:.4f}\n")
        f.write(f"  FILTERED: median={filt_scale_stats['median']:.4f}  mean={filt_scale_stats['mean']:.4f}  "
               f"p10={filt_scale_stats['p10']:.4f}  p90={filt_scale_stats['p90']:.4f}\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("  • XY Ratio: Should be stable at ~1.0 (square aspect ratio in camera frame)\n")
        f.write("  • XZ Ratio: Distortion here indicates depth nonlinearity (your problem was XZ ~ 2.23)\n")
        f.write("  • YZ Ratio: Complements XZ for aspect ratio validation\n")
        f.write("  • Scale: Should be ~1.0 (with --mono-depth-scale 0.9332 already applied)\n")
        f.write("  • If XZ improves (decreases toward ~1.0), filtering is reducing depth outliers\n")
        f.write("  • If scale variance (p90-p10) decreases, filtering stabilizes depth consistency\n")
    
    print(f"\nReport saved: {report_path}")
    print("\n=== QUICK SUMMARY ===")
    print(f"\nXZ Ratio (YOUR KEY METRIC):")
    print(f"  Raw:      median={raw_xz_stats['median']:.4f}")
    print(f"  Filtered: median={filt_xz_stats['median']:.4f}")
    xz_delta = ((filt_xz_stats['median'] - raw_xz_stats['median']) / raw_xz_stats['median']) * 100
    print(f"  Δ: {xz_delta:+.1f}%")
    print(f"\nDepth Scale Variance:")
    print(f"  Raw range:      p10={raw_scale_stats['p10']:.4f} → p90={raw_scale_stats['p90']:.4f}")
    print(f"  Filtered range: p10={filt_scale_stats['p10']:.4f} → p90={filt_scale_stats['p90']:.4f}")

if __name__ == '__main__':
    main()
