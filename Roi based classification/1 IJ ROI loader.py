#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
roi_loader_improved.py

This script loads ImageJ ROIs and trajectory data, and assigns trajectories to ROIs
with proper coordinate transformation between the two systems.

Input:
- ImageJ ROI ZIP files
- Processed trajectory data (.pkl files)

Output:
- ROI-assigned trajectories saved as .pkl files for further analysis
- Diagnostic plots visualizing ROI assignments

Usage:
python roi_loader_improved.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pickle
import glob
from pathlib import Path
import read_roi
import pandas as pd
from datetime import datetime

# Global parameters that can be modified
# =====================================
# Minimum number of trajectories per ROI to be considered valid
MIN_TRAJECTORIES_PER_ROI = 1
# Diagnostic visualization parameters
MAX_ROIS_TO_DISPLAY = 20
# Output directory name format
OUTPUT_DIR_FORMAT = 'roi_diffusion_%Y%m%d_%H%M%S'

# Pixel-to-micrometer conversion factor
# This is the key parameter for coordinate transformation
# Default value: 0.09 µm/pixel (means 1 pixel = 0.09 µm, or 1 µm = 11.11 pixels)
PIXEL_TO_MICRON = 0.09  # µm/pixel

# Offset parameters (typically should be close to 0 if both coordinate systems have the same origin)
X_OFFSET = 0.0
Y_OFFSET = 0.0
# =====================================


def load_rois(roi_file):
    """
    Load ROIs from ImageJ ROI file using read_roi library with fallback options.
    """
    try:
        # Attempt to load ROIs using read_roi library
        if roi_file.lower().endswith('.zip'):
            rois = read_roi.read_roi_zip(roi_file)
        else:
            rois = read_roi.read_roi_file(roi_file)
        print(f"Loaded {len(rois)} ROIs using read_roi library.")
        return rois
    except Exception as e:
        print(f"Error loading ROIs using read_roi library: {e}")
        return None

def is_inside_roi(x, y, roi):
    """Check if a point (x, y) is inside a ROI."""
    if 'x' in roi and 'y' in roi:
        # Create a polygon from the ROI coordinates
        polygon = np.column_stack((roi['x'], roi['y']))
        # Check if the point is inside the polygon
        return np.logical_or(np.cross(polygon[1:] - polygon[:-1], [x,y] - polygon[:-1]).sum() != 0,
                             np.cross(polygon[-1:] - polygon[:1], [x,y] - polygon[:1]).sum() != 0)
    return False

def load_trajectory_data(file_path):
    """
    Load processed trajectory data from pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary containing the trajectory data
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

def is_point_in_roi(x, y, roi_coords):
    """
    Check if a point is inside a polygon ROI using ray casting algorithm.
    
    Args:
        x, y: Coordinates of the point (in transformed coordinates)
        roi_coords: List of (x, y) coordinates defining the polygon
        
    Returns:
        Boolean indicating if the point is inside the ROI
    """
    n = len(roi_coords)
    inside = False
    
    p1x, p1y = roi_coords[0]
    for i in range(1, n + 1):
        p2x, p2y = roi_coords[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def transform_coordinates(x, y):
    """
    Transform trajectory coordinates (in µm) to pixel coordinates.
    
    Args:
        x, y: Original trajectory coordinates (in µm)
        
    Returns:
        Transformed x, y coordinates (in pixels)
    """
    # Calculate the scaling factor (pixels per µm)
    scale_factor = 1.0 / PIXEL_TO_MICRON
    
    # Apply the transformation
    x_transformed = x * scale_factor + X_OFFSET
    y_transformed = y * scale_factor + Y_OFFSET
    
    return x_transformed, y_transformed

def assign_trajectories_to_rois(trajectories, rois, show_progress=True):
    """
    Assign each trajectory to an ROI based on the mean position of the trajectory.
    Transforms trajectory coordinates to match ROI coordinate system.
    
    Args:
        trajectories: List of trajectory dictionaries
        rois: Dictionary of ROIs
        show_progress: Whether to show progress updates
        
    Returns:
        Dictionary mapping ROI IDs to lists of trajectory indices
    """
    roi_assignments = {roi_id: [] for roi_id in rois.keys()}
    roi_assignments['unassigned'] = []
    
    total_trajectories = len(trajectories)
    if show_progress:
        progress_step = max(1, total_trajectories // 20)  # Show progress every 5%
    
    for traj_idx, trajectory in enumerate(trajectories):
        # Show progress
        if show_progress and traj_idx % progress_step == 0:
            print(f"Processing trajectory {traj_idx}/{total_trajectories} ({traj_idx/total_trajectories*100:.1f}%)")
        
        # Calculate mean position of trajectory
        mean_x = np.mean(trajectory['x'])
        mean_y = np.mean(trajectory['y'])
        
        # Transform coordinates to match ROI coordinate system
        mean_x_transformed, mean_y_transformed = transform_coordinates(mean_x, mean_y)
        
        assigned = False
        
        # Check which ROI this trajectory belongs to
        for roi_id, roi in rois.items():
            if 'x' in roi and 'y' in roi:
                # Create list of coordinates for polygon
                roi_coords = list(zip(roi['x'], roi['y']))
                
                if is_point_in_roi(mean_x_transformed, mean_y_transformed, roi_coords):
                    roi_assignments[roi_id].append(traj_idx)
                    assigned = True
                    break
        
        # If not assigned to any ROI
        if not assigned:
            roi_assignments['unassigned'].append(traj_idx)
    
    # Print number of trajectories in each ROI
    print("\nAssigning trajectories to ROIs...")
    total_assigned = 0
    for roi_id, traj_indices in roi_assignments.items():
        num_traj = len(traj_indices)
        if roi_id != 'unassigned':
            total_assigned += num_traj
        print(f"{roi_id}: {num_traj} trajectories")
    
    # Print summary
    print(f"\nSummary: {total_assigned} of {total_trajectories} trajectories assigned to ROIs")
    print(f"{len(roi_assignments['unassigned'])} trajectories unassigned")
    
    return roi_assignments

def visualize_rois(rois, output_path, max_display=MAX_ROIS_TO_DISPLAY):
    """
    Create visualization of ROIs using ImageJ coordinate system (origin at top-left).
    
    Args:
        rois: Dictionary of ROIs
        output_path: Path to save the visualization
        max_display: Maximum number of ROIs to display for clarity
        
    Returns:
        None (saves visualization to file)
    """
    plt.figure(figsize=(10, 8))
    
    # Use different colors for ROIs
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Determine how many ROIs to show
    roi_ids = list(rois.keys())
    num_rois_to_show = min(len(roi_ids), max_display)
    
    # Plot selected ROIs
    for i in range(num_rois_to_show):
        roi_id = roi_ids[i]
        roi = rois[roi_id]
        
        if 'x' in roi and 'y' in roi:
            # Create polygon patch
            color_idx = i % len(colors)
            poly = Polygon(list(zip(roi['x'], roi['y'])), 
                          closed=True, 
                          fill=False, 
                          edgecolor=colors[color_idx], 
                          linewidth=2,
                          label=f"ROI {i+1}")
            plt.gca().add_patch(poly)
    
    # Set plot parameters
    plt.title(f"ROI Visualization (showing {num_rois_to_show} of {len(rois)} ROIs)")
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.grid(alpha=0.3)
    
    # Set origin at top-left corner to match ImageJ coordinates
    # This makes Y increase downward
    plt.gca().invert_yaxis()
    
    # Set equal aspect to prevent distortion
    plt.axis('equal')
    
    # Add legend for ROIs
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"ROI visualization saved to {output_path}")

def visualize_assigned_trajectories(trajectories, roi_assignments, rois, output_path):
    """
    Visualize trajectory assignments to ROIs.
    Transforms trajectory coordinates to match ROI coordinate system.
    
    Args:
        trajectories: List of trajectory dictionaries
        roi_assignments: Dictionary mapping ROI IDs to lists of trajectory indices
        rois: Dictionary of ROIs
        output_path: Path to save the visualization
        
    Returns:
        None (saves visualization to file)
    """
    plt.figure(figsize=(12, 10))
    
    # Use different colors for ROIs
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot ROIs
    for i, (roi_id, roi) in enumerate(rois.items()):
        if 'x' in roi and 'y' in roi and roi_id != 'unassigned':
            # Create polygon patch
            color_idx = i % len(colors)
            poly = Polygon(list(zip(roi['x'], roi['y'])), 
                          closed=True, 
                          fill=False, 
                          edgecolor=colors[color_idx], 
                          alpha=0.7,
                          linewidth=1.5)
            plt.gca().add_patch(poly)
    
    # Plot assigned trajectories
    for roi_idx, (roi_id, traj_indices) in enumerate(roi_assignments.items()):
        if roi_id == 'unassigned':
            # Plot unassigned trajectories in gray
            for traj_idx in traj_indices[:min(len(traj_indices), 100)]:  # Limit to 100 for clarity
                traj = trajectories[traj_idx]
                # Transform coordinates
                x_transformed, y_transformed = transform_coordinates(traj['x'], traj['y'])
                plt.plot(x_transformed, y_transformed, 'o-', color='gray', alpha=0.3, markersize=2, linewidth=1)
        else:
            # Plot assigned trajectories in ROI color
            color_idx = roi_idx % len(colors)
            num_to_show = min(len(traj_indices), 20)  # Limit to 20 per ROI for clarity
            for traj_idx in traj_indices[:num_to_show]:
                traj = trajectories[traj_idx]
                # Transform coordinates
                x_transformed, y_transformed = transform_coordinates(traj['x'], traj['y'])
                plt.plot(x_transformed, y_transformed, 'o-', color=colors[color_idx], alpha=0.7, markersize=3, linewidth=1)
    
    # Set plot parameters
    plt.title("Assigned Trajectories")
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
    
    print(f"Assigned trajectories visualization saved to {output_path}")

def prepare_output_data(trajectories, analyzed_data, roi_assignments):
    """
    Prepare data structure for output file with ROI assignments.
    
    Args:
        trajectories: List of trajectory dictionaries
        analyzed_data: Dictionary containing analyzed trajectory data
        roi_assignments: Dictionary mapping ROI IDs to lists of trajectory indices
        
    Returns:
        Dictionary with ROI-specific trajectory data
    """
    output_data = {
        'roi_assignments': roi_assignments,
        'roi_trajectories': {},
        'roi_statistics': {},
        'coordinate_transform': {
            'pixel_to_micron': PIXEL_TO_MICRON,
            'scale_factor': 1.0 / PIXEL_TO_MICRON,
            'x_offset': X_OFFSET,
            'y_offset': Y_OFFSET
        }
    }
    
    # Group trajectories by ROI
    for roi_id, traj_indices in roi_assignments.items():
        roi_trajectories = [trajectories[idx] for idx in traj_indices]
        
        # Get analyzed data for these trajectories if available
        if analyzed_data and 'trajectories' in analyzed_data:
            # Map trajectory IDs to their indices in analyzed_data
            traj_id_to_idx = {traj['id']: i for i, traj in enumerate(analyzed_data['trajectories'])}
            
            # Get analyzed data for each trajectory in this ROI
            roi_analyzed_trajs = []
            for traj in roi_trajectories:
                if traj['id'] in traj_id_to_idx:
                    analyzed_idx = traj_id_to_idx[traj['id']]
                    roi_analyzed_trajs.append(analyzed_data['trajectories'][analyzed_idx])
            
            # Calculate ROI statistics
            if roi_analyzed_trajs:
                # Get valid diffusion coefficients
                D_values = [traj['D'] for traj in roi_analyzed_trajs if not np.isnan(traj['D'])]
                
                if D_values:
                    output_data['roi_statistics'][roi_id] = {
                        'n': len(D_values),
                        'mean_D': np.mean(D_values),
                        'median_D': np.median(D_values),
                        'std_D': np.std(D_values),
                        'sem_D': np.std(D_values) / np.sqrt(len(D_values)),
                        'min_D': np.min(D_values),
                        'max_D': np.max(D_values)
                    }
                else:
                    output_data['roi_statistics'][roi_id] = {
                        'n': 0,
                        'mean_D': np.nan,
                        'median_D': np.nan,
                        'std_D': np.nan,
                        'sem_D': np.nan,
                        'min_D': np.nan,
                        'max_D': np.nan
                    }
            
            output_data['roi_trajectories'][roi_id] = roi_analyzed_trajs
        else:
            # If no analyzed data available, just store raw trajectories
            output_data['roi_trajectories'][roi_id] = roi_trajectories
    
    return output_data

def main():
    """
    Main function to load ROIs and assign trajectories.
    """
    print("\nROI Trajectory Loader (Improved Version)")
    print("=====================================")
    
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
    
    # Visualize ROIs
    visualize_rois(rois, os.path.join(output_dir, 'rois.png'))
    
    # Ask for trajectory data file
    traj_file = input("Enter path to trajectory data file (.pkl): ")
    traj_data = load_trajectory_data(traj_file)
    if traj_data is None:
        print("Failed to load trajectory data. Exiting.")
        return
    
    # Get trajectories
    trajectories = traj_data['trajectories']
    
    # Ask for analyzed diffusion data file
    analyzed_file = input("Enter path to analyzed diffusion data file (.pkl): ")
    analyzed_data = load_trajectory_data(analyzed_file)
    
    # Ask for pixel-to-micron conversion factor
    global PIXEL_TO_MICRON, X_OFFSET, Y_OFFSET
    
    try:
        pixel_to_micron_input = input(f"Pixel-to-micrometer conversion factor [{PIXEL_TO_MICRON} µm/pixel]: ")
        if pixel_to_micron_input.strip():
            PIXEL_TO_MICRON = float(pixel_to_micron_input)
        
        # Calculate and display the inverse (pixels per µm) for clarity
        scale_factor = 1.0 / PIXEL_TO_MICRON
        print(f"This corresponds to {scale_factor:.4f} pixels/µm")
        
        # Ask for offset values
        x_offset_input = input(f"X offset (pixels) [{X_OFFSET}]: ")
        if x_offset_input.strip():
            X_OFFSET = float(x_offset_input)
        
        y_offset_input = input(f"Y offset (pixels) [{Y_OFFSET}]: ")
        if y_offset_input.strip():
            Y_OFFSET = float(y_offset_input)
        
        print(f"\nUsing coordinate transformation:")
        print(f"pixel_to_micron = {PIXEL_TO_MICRON} µm/pixel")
        print(f"scale_factor = {scale_factor} pixels/µm")
        print(f"x_offset = {X_OFFSET} pixels")
        print(f"y_offset = {Y_OFFSET} pixels")
        print(f"\nTransformation equations:")
        print(f"x_transformed = x * {scale_factor} + {X_OFFSET}")
        print(f"y_transformed = y * {scale_factor} + {Y_OFFSET}")
    
    except ValueError as e:
        print(f"Error parsing input: {e}")
        print("Using default values instead.")
    
    # Assign trajectories to ROIs
    roi_assignments = assign_trajectories_to_rois(trajectories, rois)
    
    # Visualize assigned trajectories
    visualize_assigned_trajectories(trajectories, roi_assignments, rois, 
                                  os.path.join(output_dir, 'assigned_trajectories.png'))
    
    # Prepare output data
    output_data = prepare_output_data(trajectories, analyzed_data, roi_assignments)
    
    # Save output data
    output_file = os.path.join(output_dir, 'roi_trajectory_data.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"ROI trajectory data saved to {output_file}")
    
    # Save transformation parameters separately for reference
    transform_file = os.path.join(output_dir, 'coordinate_transform.txt')
    with open(transform_file, 'w') as f:
        f.write("Coordinate transformation parameters:\n")
        f.write(f"Pixel-to-micrometer conversion: {PIXEL_TO_MICRON} µm/pixel\n")
        f.write(f"Scale factor (pixels/µm): {1.0 / PIXEL_TO_MICRON}\n")
        f.write(f"X offset: {X_OFFSET} pixels\n")
        f.write(f"Y offset: {Y_OFFSET} pixels\n")
        f.write("\nTransformation equations:\n")
        f.write(f"x_transformed = x * {1.0 / PIXEL_TO_MICRON} + {X_OFFSET}\n")
        f.write(f"y_transformed = y * {1.0 / PIXEL_TO_MICRON} + {Y_OFFSET}\n")
    
    print(f"Transformation parameters saved to {transform_file}")
    
    return output_data

if __name__ == "__main__":
    main()