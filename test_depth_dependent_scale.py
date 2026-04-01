#!/usr/bin/env python3
"""
Analyze depth-dependent scale errors.
Find if different depth ranges need different corrections.
"""

import os
import csv
import json
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

def load_depth_data(dataset_path, rgb_dir='rgb', depth_dir='depth', max_frames=150):
    """Load RGB-D frames from TUM dataset."""
    rgb_path = os.path.join(dataset_path, f'{rgb_dir}.txt')
    depth_path = os.path.join(dataset_path, f'{depth_dir}.txt')
    gt_path = os.path.join(dataset_path, 'groundtruth.txt')
    
    if not all(os.path.exists(p) for p in [rgb_path, depth_path, gt_path]):
        raise FileNotFoundError(f"Dataset files not found in {dataset_path}")
    
    rgb_frames = []
    with open(rgb_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                rgb_frames.append({'ts': float(parts[0]), 'path': parts[1]})
    
    depth_frames = []
    with open(depth_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                depth_frames.append({'ts': float(parts[0]), 'path': parts[1]})
    
    gt_poses = {}
    with open(gt_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                ts = float(parts[0])
                gt_poses[ts] = list(map(float, parts[1:8]))
    
    return rgb_frames[:max_frames], depth_frames[:max_frames], gt_poses

def load_csv_frame_metrics(csv_path):
    """Load per-frame scale metrics from distortion diagnostic."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'frame_idx': int(float(row['frame_idx'])),
                'frame_scale_to_gt': float(row['frame_scale_to_gt']),
                'cam_extent_z_est': float(row['cam_extent_z_est']),
                'cam_extent_z_gt': float(row['cam_extent_z_gt']),
            })
    return rows

def analyze_depth_dependence(frame_metrics, dataset_path, depth_dir='depth'):
    """Analyze per-pixel depth-scale relationship."""
    
    # Load actual depth images to get per-pixel depths
    depth_frames = []
    depth_txt = os.path.join(dataset_path, f'{depth_dir}.txt')
    with open(depth_txt) as f:
        for line in f:
            if not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    depth_frames.append(parts[1])
    
    # Bin frame metrics by depth range
    depth_bins = defaultdict(list)
    
    for frame_data in frame_metrics:
        frame_idx = frame_data['frame_idx']
        
        if frame_idx >= len(depth_frames):
            continue
        
        # Load depth image
        depth_file = os.path.join(dataset_path, depth_frames[frame_idx])
        try:
            depth_img = np.load(depth_file) if depth_file.endswith('.npy') else \
                        (cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) if False else None)
            
            # If load fails, use frame metadata as proxy
            if depth_img is None:
                z_est = frame_data['cam_extent_z_est']
                scale = frame_data['frame_scale_to_gt']
                
                if z_est > 0.1:
                    depth_bin = int(z_est)  # Bin by integer meter
                    depth_bins[depth_bin].append(scale)
        except:
            # Fallback: use frame extent as proxy for depth
            z_est = frame_data['cam_extent_z_est']
            scale = frame_data['frame_scale_to_gt']
            
            if z_est > 0.1:
                depth_bin = int(z_est * 2) / 2.0  # 0.5m bins
                depth_bins[depth_bin].append(scale)
    
    # Summarize per bin
    bin_summary = []
    for depth_bin in sorted(depth_bins.keys()):
        scales = depth_bins[depth_bin]
        if len(scales) >= 3:
            bin_summary.append({
                'depth_bin': depth_bin,
                'n_frames': len(scales),
                'mean_scale': np.mean(scales),
                'median_scale': np.median(scales),
                'std_scale': np.std(scales),
            })
    
    return bin_summary

def main():
    parser = argparse.ArgumentParser(description='Analyze depth-dependent scale errors')
    parser.add_argument('--dataset', default='data/rgbd_dataset_freiburg1_360',
                        help='Path to TUM RGB-D dataset')
    parser.add_argument('--geom-csv', default='output/depth_shape_diagnostic/frame_metrics.csv',
                        help='Geometry metrics CSV from test_depth_shape_distortion.py')
    parser.add_argument('--max-frames', type=int, default=150,
                        help='Maximum frames to analyze')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.geom_csv):
        print(f"Error: Geometry CSV not found: {args.geom_csv}")
        return
    
    print("Loading frame metrics...")
    frame_metrics = load_csv_frame_metrics(args.geom_csv)
    
    print(f"Analyzing {len(frame_metrics)} frames for depth-dependent scaling...")
    bin_summary = analyze_depth_dependence(frame_metrics, args.dataset, depth_dir='depth')
    
    # Output report
    output_dir = Path('output/depth_confidence_filter')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    csv_path = output_dir / 'depth_dependent_scale_analysis.csv'
    with open(csv_path, 'w') as f:
        f.write('depth_bin_m,n_frames,mean_scale,median_scale,std_scale\n')
        for item in bin_summary:
            f.write(f"{item['depth_bin']},{item['n_frames']},{item['mean_scale']:.4f},"
                   f"{item['median_scale']:.4f},{item['std_scale']:.4f}\n")
    
    # Write summary
    report_path = output_dir / 'depth_dependent_scale_report.txt'
    with open(report_path, 'w') as f:
        f.write("=== DEPTH-DEPENDENT SCALE ANALYSIS ===\n\n")
        f.write(f"Global scale (median across all frames): {np.median([m['frame_scale_to_gt'] for m in frame_metrics]):.4f}\n\n")
        
        f.write("Scale Factor by Depth Range:\n")
        f.write("(Lower = depth is overestimated in that range; Higher = depth is underestimated)\n\n")
        
        f.write("Depth Range (m) | Median Scale | Mean Scale | Std Dev | Samples\n")
        f.write("-" * 70 + "\n")
        
        for item in bin_summary:
            f.write(f"  {item['depth_bin']:>5.1f}       | {item['median_scale']:>10.4f}  | "
                   f"{item['mean_scale']:>10.4f}  | {item['std_scale']:>7.4f} | {item['n_frames']:>6}\n")
        
        f.write("\n\nINTERPRETATION:\n")
        f.write("• Scale = median(GT depth) / median(estimated depth) for frames in this range\n")
        f.write("• Scale > 1.0 = model underestimates depth (needs upward correction)\n")
        f.write("• Scale < 1.0 = model overestimates depth (needs downward correction)\n")
        f.write("• If scale varies significantly across depth, apply depth-dependent correction\n")
        f.write("• Example: if near=0.95, far=1.10, use 2-point linear interpolation\n\n")
        
        if len(bin_summary) >= 2:
            near = bin_summary[0]['median_scale']
            far = bin_summary[-1]['median_scale']
            delta = abs(far - near)
            f.write(f"Current variation (near to far): {near:.4f} → {far:.4f} (Δ {delta:+.4f})\n")
            
            if delta > 0.2:
                f.write(f"\n✓ RECOMMENDATION: Use depth-dependent scaling\n")
                f.write(f"  Near scale: {near:.4f}\n")
                f.write(f"  Far scale:  {far:.4f}\n")
                f.write(f"  Linear interpolation: scale(d) = {near:.4f} + ({far:.4f} - {near:.4f}) * d/{bin_summary[-1]['depth_bin']:.1f}\n")
            else:
                f.write(f"\n✗ Low variation (Δ={delta:.4f}): Global scale sufficient\n")
    
    print(f"\nAnalysis saved:")
    print(f"  CSV:    {csv_path}")
    print(f"  Report: {report_path}")
    
    print("\n=== SUMMARY ===")
    if bin_summary:
        print(f"Depth range: {bin_summary[0]['depth_bin']:.1f}m - {bin_summary[-1]['depth_bin']:.1f}m")
        print(f"Near-depth scale:  {bin_summary[0]['median_scale']:.4f}")
        print(f"Far-depth scale:   {bin_summary[-1]['median_scale']:.4f}")
        print(f"Variation: {abs(bin_summary[-1]['median_scale'] - bin_summary[0]['median_scale']):+.4f}")
    else:
        print("No sufficient depth variation in dataset")

if __name__ == '__main__':
    main()
