"""
# not done yet
Combine glass detections with depth-based occupancy grid to create a complete safety map.
Glass areas detected by VLM are marked as obstacles even if depth sees through them.
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from detector import OwlDetectorbase32

def project_detections_to_occupancy_grid(
    detections,
    image_width,
    image_height,
    grid_metadata,
    depth_image_path,
    max_depth=10.0
):
    """
    Project 2D bounding box detections onto the 3D occupancy grid.
    
    Args:
        detections: List of detection dicts with 'box' key (x0, y0, x1, y1)
        image_width, image_height: RGB image dimensions
        grid_metadata: Dict with grid parameters (resolution, x_min, y_min, etc.)
        depth_image_path: Path to corresponding depth image
        max_depth: Maximum depth value in meters
        
    Returns:
        glass_mask: 2D boolean array marking glass obstacles in the grid
    """
    # Load depth image
    depth_raw = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    if depth_raw.ndim == 3:
        depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
    
    # Resize depth to match RGB
    depth_raw = cv2.resize(depth_raw, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    
    # Convert to metric depth (same as mapping.py)
    depth_norm = depth_raw.astype(np.float32) / 255.0
    inverse_depth = depth_norm + 1e-6
    min_depth = 0.1
    depth_m = min_depth + (max_depth - min_depth) * (1.0 - inverse_depth)
    depth_m = np.clip(depth_m, min_depth, max_depth)
    
    # Camera intrinsics (same as mapping.py)
    fx = fy = 0.8 * image_width
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    # Grid parameters
    resolution = grid_metadata['resolution']
    x_min_grid = grid_metadata['x_min']
    y_min_grid = grid_metadata['y_min']
    grid_height, grid_width = grid_metadata['grid'].shape
    
    # Create glass mask
    glass_mask = np.zeros((grid_height, grid_width), dtype=bool)
    
    print(f"\nProjecting {len(detections)} glass detections to occupancy grid...")
    
    # After: for det in detections:
    for det in detections:
        x0, y0, x1, y1 = det['box']
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        
        # DEBUG: Print detection info
        print(f"\n  Detection box: [{x0}, {y0}, {x1}, {y1}]")
        
        projected_count = 0
        sample_depths = []
        sample_3d_points = []
        
        # Sample points within bounding box
        for py in range(y0, y1, 2):
            for px in range(x0, x1, 2):
                if 0 <= py < image_height and 0 <= px < image_width:
                    # Get depth at this pixel
                    Z = depth_m[py, px]
                    
                    # DEBUG: Collect depth values
                    sample_depths.append(Z)
                    
                    if Z > 0.1 and Z < max_depth:
                        # Back-project to 3D (camera coordinates)
                        X = (px - cx) * Z / fx
                        Y = (py - cy) * Z / fy
                        
                        # DEBUG: Collect 3D points
                        sample_3d_points.append((X, Y, Z))
                        
                        # Convert to grid coordinates (top-down view)
                        # Assuming camera looking forward: X=right, Y=down, Z=forward
                        grid_x = int((X - x_min_grid) / resolution)
                        grid_y = int((Z - y_min_grid) / resolution)  # Z becomes Y in top-down
                        
                        # DEBUG: Check if in bounds
                        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                            glass_mask[grid_y, grid_x] = True
                            projected_count += 1
        
        # DEBUG: Print statistics
        print(f"    Depth values: min={min(sample_depths) if sample_depths else 0:.2f}, "
              f"max={max(sample_depths) if sample_depths else 0:.2f}, "
              f"mean={np.mean(sample_depths) if sample_depths else 0:.2f}")
        print(f"    Valid 3D points: {len(sample_3d_points)}")
        if sample_3d_points:
            X_vals = [p[0] for p in sample_3d_points]
            Y_vals = [p[1] for p in sample_3d_points]
            Z_vals = [p[2] for p in sample_3d_points]
            print(f"    3D X range: [{min(X_vals):.2f}, {max(X_vals):.2f}]")
            print(f"    3D Y range: [{min(Y_vals):.2f}, {max(Y_vals):.2f}]")
            print(f"    3D Z range: [{min(Z_vals):.2f}, {max(Z_vals):.2f}]")
        print(f"    Grid bounds: X=[0, {grid_width}], Y=[0, {grid_height}]")
        print(f"    Grid world bounds: X=[{x_min_grid:.2f}, {x_min_grid + grid_width*resolution:.2f}], "
              f"Y=[{y_min_grid:.2f}, {y_min_grid + grid_height*resolution:.2f}]")
        print(f"    Projected to grid: {projected_count} cells")
    
    print(f"Glass mask covers {glass_mask.sum()} grid cells")
    return glass_mask


def combine_maps(depth_occupancy_grid, glass_mask):
    """
    Combine depth-based occupancy with glass detections.
    
    Returns:
        combined_grid: Occupancy values (0=free, 128=glass, 255=solid obstacle)
    """
    combined = depth_occupancy_grid.copy()
    
    # Mark glass areas as obstacles (value 128 to distinguish from solid obstacles)
    combined[glass_mask] = 128
    
    # Solid obstacles from depth (already 255) take precedence
    combined[depth_occupancy_grid == 255] = 255
    
    return combined


def main():
    # Paths
    rgb_path = "data/rgb/5237818140983496476.jpg"
    depth_path = "data/depth/5237818140983496476.jpg_depth.png"
    occupancy_grid_path = "output/occupancy_grid_2d_v2.npy"
    
    # 1. Load existing depth-based occupancy grid
    print("Loading depth-based occupancy grid...")
    grid_data = np.load(occupancy_grid_path, allow_pickle=True).item()
    depth_grid = grid_data['grid']
    
    print(f"Grid shape: {depth_grid.shape}")
    print(f"Resolution: {grid_data['resolution']}m")
    print(f"X range: [{grid_data['x_min']:.2f}, {grid_data['x_max']:.2f}]m")
    print(f"Y range: [{grid_data['y_min']:.2f}, {grid_data['y_max']:.2f}]m")
    
    # 2. Detect glass in RGB image
    print("\nDetecting glass objects...")
    img = Image.open(rgb_path).convert("RGB")
    detector = OwlDetectorbase32()
    
    queries = [
        "a glass window",
        "a transparent surface",
        "a glass wall",
        "a mirror",
        "window"
    ]
    
    detections = detector.detect(img, queries, threshold=0.3)
    print(f"Found {len(detections)} glass detections")
    
    for det in detections:
        print(f"  - {det['label']}: score {det['score']:.2f}, box {det['box']}")
    
    if len(detections) == 0:
        print("No glass detected, using depth-only occupancy grid")
        combined_grid = depth_grid
        glass_mask = np.zeros_like(depth_grid, dtype=bool)
    else:
        # 3. Project glass detections to occupancy grid
        glass_mask = project_detections_to_occupancy_grid(
            detections,
            img.size[0],
            img.size[1],
            grid_data,
            depth_path,
            max_depth=10.0
        )
        
        # 4. Combine maps
        print("\nCombining depth and glass maps...")
        combined_grid = combine_maps(depth_grid, glass_mask)
        
        glass_cells = glass_mask.sum()
        depth_cells = (depth_grid == 255).sum()
        print(f"  - Depth obstacles: {depth_cells} cells")
        print(f"  - Glass obstacles: {glass_cells} cells")
        print(f"  - Total obstacles: {(combined_grid > 0).sum()} cells")
    
    # 5. Visualize combined map
    print("\nVisualizing combined safety map...")
    
    # Create annotated RGB image with detections
    img_annotated = img.copy()
    draw = ImageDraw.Draw(img_annotated)
    for det in detections:
        x0, y0, x1, y1 = det['box']
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        draw.text((x0, y0-10), f"{det['label']}: {det['score']:.2f}", fill="red")
    
    # Save annotated image
    annotated_path = "output/glass_detections_annotated.png"
    img_annotated.save(annotated_path)
    print(f"Saved annotated image to {annotated_path}")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Depth-only occupancy
    axes[0].imshow(depth_grid, cmap='gray_r', origin='lower', interpolation='nearest')
    axes[0].set_title('Depth-Only Occupancy\n(from point cloud)')
    axes[0].set_xlabel(f'X (grid cells @ {grid_data["resolution"]}m)')
    axes[0].set_ylabel(f'Y (grid cells @ {grid_data["resolution"]}m)')
    
    # 2. Glass detections overlay
    glass_overlay = np.zeros((depth_grid.shape[0], depth_grid.shape[1], 3))
    glass_overlay[:, :, 0] = depth_grid / 255.0  # Red channel = depth obstacles
    glass_overlay[:, :, 1] = glass_mask.astype(float)  # Green channel = glass
    axes[1].imshow(glass_overlay, origin='lower', interpolation='nearest')
    axes[1].set_title(f'Glass Detection Overlay\n({len(detections)} detections, {glass_mask.sum()} cells)')
    axes[1].set_xlabel(f'X (grid cells @ {grid_data["resolution"]}m)')
    axes[1].set_ylabel(f'Y (grid cells @ {grid_data["resolution"]}m)')
    
    # 3. Combined safety map
    colors = ['white', 'yellow', 'orange', 'red']  # 0=free, 128=glass, 255=solid
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('safety', colors, N=n_bins)
    
    im = axes[2].imshow(combined_grid, cmap=cmap, origin='lower', interpolation='nearest', vmin=0, vmax=255)
    axes[2].set_title('Combined Safety Map\n(white=free, yellow=glass, red=solid)')
    axes[2].set_xlabel(f'X (grid cells @ {grid_data["resolution"]}m)')
    axes[2].set_ylabel(f'Y (grid cells @ {grid_data["resolution"]}m)')
    
    plt.colorbar(im, ax=axes[2], label='Obstacle type (0=free, 128=glass, 255=solid)')
    
    plt.tight_layout()
    output_path = "output/combined_safety_map.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined safety map to {output_path}")
    
    # 6. Save combined grid
    combined_grid_path = "output/combined_safety_map.npy"
    np.save(combined_grid_path, {
        'grid': combined_grid,
        'resolution': grid_data['resolution'],
        'x_min': grid_data['x_min'],
        'y_min': grid_data['y_min'],
        'x_max': grid_data['x_max'],
        'y_max': grid_data['y_max'],
        'z_range': grid_data.get('z_range', (0, 0)),
        'glass_detections': len(detections),
        'glass_cells': int(glass_mask.sum()),
        'depth_cells': int((depth_grid == 255).sum())
    })
    print(f"Saved combined grid metadata to {combined_grid_path}")
    
    print("\n✓ Complete! Files created:")
    print(f"  - {annotated_path} (RGB with glass bounding boxes)")
    print(f"  - {output_path} (3-panel comparison)")
    print(f"  - {combined_grid_path} (numpy grid + metadata)")


if __name__ == "__main__":
    main()