# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 18:59:24 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
roi_diagnosis.py

This script diagnoses issues with ROI assignment by:
1. Loading both ROIs and trajectories
2. Visualizing their relative scales and positions
3. Providing scale conversion options
4. Testing different coordinate transformations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pickle
import read_roi
from datetime import datetime
import json

# Global parameters that can be modified
# =====================================
# Output directory name format
OUTPUT_DIR_FORMAT = 'roi_diagnosis_%Y%m%d_%H%M%S'
# =====================================

def load_rois(roi_file):
    """
    Load ROIs from ImageJ ZIP file maintaining original ImageJ coordinate system.
    """
    try:
        rois = read_roi.read_roi_zip(roi_file)
        print(f"Loaded {len(rois)} ROIs")
        return rois
    except Exception as e:
        print(f"Error loading ROIs from {roi_file}: {e}")
        return None

def load_trajectory_data(file_path):
    """
    Load trajectory data from pickle file.
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded data from {file_path}")
        print(f"Number of trajectories: {len(data['trajectories'])}")
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def get_roi_bounds(rois):
    """Get the bounding box of all ROIs."""
    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
    
    for roi in rois.values():
        if 'x' in roi and 'y' in roi:
            x, y = roi['x'], roi['y']
            min_x = min(min_x, np.min(x))
            max_x = max(max_x, np.max(x))
            min_y = min(min_y, np.min(y))
            max_y = max(max_y, np.max(y))
    
    return min_x, max_x, min_y, max_y

def get_trajectory_bounds(trajectories):
    """Get the bounding box of all trajectories."""
    all_x = []
    all_y = []
    
    for traj in trajectories:
        all_x.extend(traj['x'])
        all_y.extend(traj['y'])
    
    min_x = np.min(all_x)
    max_x = np.max(all_x)
    min_y = np.min(all_y)
    max_y = np.max(all_y)
    
    return min_x, max_x, min_y, max_y

def visualize_scale_comparison(rois, trajectories, output_path):
    """
    Visualize ROIs and trajectories on separate plots to compare their scales.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot ROIs on the first plot
    roi_min_x, roi_max_x, roi_min_y, roi_max_y = get_roi_bounds(rois)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, (roi_id, roi) in enumerate(list(rois.items())[:20]):  # Limit to 20 for clarity
        if 'x' in roi and 'y' in roi:
            color_idx = i % len(colors)
            ax1.plot(roi['x'], roi['y'], '-', color=colors[color_idx], linewidth=1)
    
    ax1.set_title("ROI Coordinates")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.grid(alpha=0.3)
    ax1.invert_yaxis()  # ImageJ coordinates have origin at top-left
    
    # Plot trajectories on the second plot
    traj_min_x, traj_max_x, traj_min_y, traj_max_y = get_trajectory_bounds(trajectories)
    
    # Plot a subset of trajectories
    for i, traj in enumerate(trajectories[:100]):  # Limit to 100 for clarity
        ax2.plot(traj['x'], traj['y'], 'o-', markersize=2, linewidth=1, alpha=0.5)
    
    ax2.set_title("Trajectory Coordinates")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")
    ax2.grid(alpha=0.3)
    
    # Print scale information
    print("\nScale Information:")
    print(f"ROI X range: {roi_min_x:.2f} to {roi_max_x:.2f} (width: {roi_max_x - roi_min_x:.2f})")
    print(f"ROI Y range: {roi_min_y:.2f} to {roi_max_y:.2f} (height: {roi_max_y - roi_min_y:.2f})")
    print(f"Trajectory X range: {traj_min_x:.2f} to {traj_max_x:.2f} (width: {traj_max_x - traj_min_x:.2f})")
    print(f"Trajectory Y range: {traj_min_y:.2f} to {traj_max_y:.2f} (height: {traj_max_y - traj_min_y:.2f})")
    
    # Calculate potential scale factors
    x_scale = (roi_max_x - roi_min_x) / (traj_max_x - traj_min_x) if (traj_max_x - traj_min_x) != 0 else 0
    y_scale = (roi_max_y - roi_min_y) / (traj_max_y - traj_min_y) if (traj_max_y - traj_min_y) != 0 else 0
    
    print(f"\nPotential scale factors:")
    print(f"X scale (ROI width / trajectory width): {x_scale:.4f}")
    print(f"Y scale (ROI height / trajectory height): {y_scale:.4f}")
    
    # Calculate potential offsets
    x_offset = roi_min_x - traj_min_x * x_scale
    y_offset = roi_min_y - traj_min_y * y_scale
    
    print(f"\nPotential offsets after scaling:")
    print(f"X offset: {x_offset:.4f}")
    print(f"Y offset: {y_offset:.4f}")
    
    # Save scale information to a file
    scale_info = {
        "roi": {
            "x_min": float(roi_min_x),
            "x_max": float(roi_max_x),
            "y_min": float(roi_min_y),
            "y_max": float(roi_max_y),
            "width": float(roi_max_x - roi_min_x),
            "height": float(roi_max_y - roi_min_y)
        },
        "trajectory": {
            "x_min": float(traj_min_x),
            "x_max": float(traj_max_x),
            "y_min": float(traj_min_y),
            "y_max": float(traj_max_y),
            "width": float(traj_max_x - traj_min_x),
            "height": float(traj_max_y - traj_min_y)
        },
        "scale_factors": {
            "x_scale": float(x_scale),
            "y_scale": float(y_scale),
            "x_offset": float(x_offset),
            "y_offset": float(y_offset)
        }
    }
    
    scale_info_path = os.path.splitext(output_path)[0] + "_scale_info.json"
    with open(scale_info_path, 'w') as f:
        json.dump(scale_info, f, indent=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Scale comparison visualization saved to {output_path}")
    print(f"Scale information saved to {scale_info_path}")
    
    return scale_info

def test_transformed_assignment(rois, trajectories, scale_info, output_path):
    """
    Test trajectory assignment with different coordinate transformations.
    """
    # Extract scale factors
    x_scale = scale_info["scale_factors"]["x_scale"]
    y_scale = scale_info["scale_factors"]["y_scale"]
    x_offset = scale_info["scale_factors"]["x_offset"]
    y_offset = scale_info["scale_factors"]["y_offset"]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot ROIs
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, (roi_id, roi) in enumerate(rois.items()):
        if 'x' in roi and 'y' in roi:
            color_idx = i % len(colors)
            poly = Polygon(list(zip(roi['x'], roi['y'])), 
                          closed=True, 
                          fill=False, 
                          edgecolor=colors[color_idx], 
                          alpha=0.7,
                          linewidth=1.5)
            plt.gca().add_patch(poly)
    
    # Plot transformed trajectories
    for i, traj in enumerate(trajectories[:100]):  # Limit to 100 for clarity
        # Apply transformation
        x_transformed = np.array(traj['x']) * x_scale + x_offset
        y_transformed = np.array(traj['y']) * y_scale + y_offset
        
        plt.plot(x_transformed, y_transformed, 'o-', color='red', alpha=0.5, markersize=3, linewidth=1)
    
    # Set plot parameters
    plt.title("Transformed Trajectories with ROIs")
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.grid(alpha=0.3)
    
    # Set origin at top-left corner to match ImageJ coordinates
    plt.gca().invert_yaxis()
    
    # Set equal aspect to prevent distortion
    plt.axis('equal')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Transformed trajectories visualization saved to {output_path}")

def main():
    """
    Main function to diagnose ROI assignment issues.
    """
    # Ask for input paths
    roi_file = input("Enter path to ImageJ ROI ZIP file: ")
    
    # Load ROIs
    rois = load_rois(roi_file)
    if rois is None:
        print("Failed to load ROIs. Exiting.")
        return
    
    # Create output directory
    output_dir_name = datetime.now().strftime(OUTPUT_DIR_FORMAT)
    output_base_dir = os.path.dirname(roi_file)
    output_dir = os.path.join(output_base_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Ask for trajectory data file
    traj_file = input("Enter path to trajectory data file (.pkl): ")
    traj_data = load_trajectory_data(traj_file)
    if traj_data is None:
        print("Failed to load trajectory data. Exiting.")
        return
    
    # Get trajectories
    trajectories = traj_data['trajectories']
    
    # Visualize scale comparison
    scale_info = visualize_scale_comparison(
        rois, 
        trajectories, 
        os.path.join(output_dir, 'scale_comparison.png')
    )
    
    # Test transformed assignment
    test_transformed_assignment(
        rois,
        trajectories,
        scale_info,
        os.path.join(output_dir, 'transformed_trajectories.png')
    )
    
    print("\nDiagnosis complete. Check the visualizations to determine appropriate transformations.")
    print(f"Results saved to {output_dir}")
    
    # Return the scale information for use in the fixed script
    return scale_info

if __name__ == "__main__":
    main()