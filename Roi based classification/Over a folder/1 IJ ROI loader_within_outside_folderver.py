# -*- coding: utf-8 -*-
"""
roi_loader_batch.py

This script loads ImageJ ROIs and trajectory data from matched file sets,
and assigns trajectories to ROIs with proper coordinate transformation.

Input:
- matched_files.pkl (containing matched sets of ROI, processed traj, analyzed traj files)

Output:
- ROI-assigned trajectories saved as .pkl files in same folder as ROI files
- Diagnostic plots visualizing ROI assignments
- Assigns trajectories as being within or outside the ROI

Usage:
python roi_loader_batch.py
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

def visualize_rois(rois, output_path):
    """
    Visualize ROIs on a plot.
    
    Args:
        rois: Dictionary of ROIs
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 10))
    
    # Define colormap for ROIs
    colors = plt.cm.tab20(np.linspace(0, 1, len(rois)))
    
    num_rois_displayed = 0
    
    for roi_idx, (roi_id, roi) in enumerate(rois.items()):
        if 'x' in roi and 'y' in roi:
            # Create polygon
            polygon = Polygon(np.column_stack((roi['x'], roi['y'])), 
                            closed=True, 
                            edgecolor=colors[roi_idx], 
                            facecolor=colors[roi_idx],
                            alpha=0.3,
                            linewidth=2)
            plt.gca().add_patch(polygon)
            
            # Add label at centroid
            centroid_x = np.mean(roi['x'])
            centroid_y = np.mean(roi['y'])
            plt.text(centroid_x, centroid_y, roi_id, 
                    ha='center', va='center', 
                    fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            num_rois_displayed += 1
            
            if num_rois_displayed >= MAX_ROIS_TO_DISPLAY:
                break
    
    # Set plot parameters
    plt.title("ROIs")
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
    
    print(f"ROI visualization saved to {output_path}")

def visualize_assigned_trajectories(trajectories, roi_assignments, rois, output_path):
    """
    Visualize trajectories assigned to ROIs.
    
    Args:
        trajectories: List of trajectory dictionaries
        roi_assignments: Dictionary mapping ROI IDs to lists of trajectory indices
        rois: Dictionary of ROIs
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(14, 12))
    
    # Define colormap
    colors = plt.cm.tab20(np.linspace(0, 1, len(rois)))
    
    # Plot ROIs first
    for roi_idx, (roi_id, roi) in enumerate(rois.items()):
        if 'x' in roi and 'y' in roi:
            polygon = Polygon(np.column_stack((roi['x'], roi['y'])), 
                            closed=True, 
                            edgecolor=colors[roi_idx], 
                            facecolor='none',
                            linewidth=2,
                            label=roi_id)
            plt.gca().add_patch(polygon)
    
    # Plot trajectories
    for roi_idx, (roi_id, traj_indices) in enumerate(roi_assignments.items()):
        if roi_id == 'unassigned':
            # Plot unassigned trajectories in gray
            for traj_idx in traj_indices[:50]:  # Limit to 50 for clarity
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

def process_file_set(roi_file, process_traj_file, analyzed_traj_file):
    """
    Process a single set of matched files.
    
    Args:
        roi_file: Path to ROI zip file
        process_traj_file: Path to processed trajectory pkl file
        analyzed_traj_file: Path to analyzed trajectory pkl file
    """
    print("\n" + "="*70)
    print(f"Processing: {os.path.basename(roi_file)}")
    print("="*70)
    
    # Load ROIs
    rois = load_rois(roi_file)
    if rois is None:
        print(f"Failed to load ROIs from {roi_file}. Skipping this file set.")
        return
    
    # Get output directory (same folder as ROI file)
    output_dir = os.path.dirname(roi_file)
    
    # Extract base name for output files
    roi_filename = os.path.basename(roi_file)
    base_name = roi_filename.rsplit("_", 3)[0]  # Remove "_rois.zip"
    
    # Visualize ROIs
    rois_output_path = os.path.join(output_dir, f'rois_{base_name}.png')
    visualize_rois(rois, rois_output_path)
    
    # Load trajectory data
    traj_data = load_trajectory_data(process_traj_file)
    if traj_data is None:
        print(f"Failed to load trajectory data from {process_traj_file}. Skipping this file set.")
        return
    
    # Get trajectories
    trajectories = traj_data['trajectories']
    
    # Load analyzed diffusion data
    analyzed_data = load_trajectory_data(analyzed_traj_file)
    
    # Assign trajectories to ROIs
    roi_assignments = assign_trajectories_to_rois(trajectories, rois)
    
    # Visualize assigned trajectories
    assigned_traj_output_path = os.path.join(output_dir, f'assigned_trajectories_{base_name}.png')
    visualize_assigned_trajectories(trajectories, roi_assignments, rois, assigned_traj_output_path)
    
    # Prepare output data
    output_data = prepare_output_data(trajectories, analyzed_data, roi_assignments)
    
    # Save output data
    output_file = os.path.join(output_dir, f'roi_trajectory_data_{base_name}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"ROI trajectory data saved to {output_file}")
    
    # Save transformation parameters separately for reference
    transform_file = os.path.join(output_dir, f'coordinate_transform_{base_name}.txt')
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
    print(f"Completed processing: {base_name}")

def main():
    """
    Main function to process all matched file sets.
    """
    print("\nROI Trajectory Loader - Batch Processing")
    print("=========================================")
    
    # Ask for matched files pickle
    matched_files_pkl = input("Enter path to matched_files.pkl: ")
    
    # Load matched files
    try:
        with open(matched_files_pkl, 'rb') as f:
            matched_files = pickle.load(f)
        print(f"\nLoaded {len(matched_files)} matched file sets.")
    except Exception as e:
        print(f"Error loading matched files: {e}")
        return
    
    # Display coordinate transformation parameters
    print(f"\nUsing coordinate transformation:")
    print(f"pixel_to_micron = {PIXEL_TO_MICRON} µm/pixel")
    print(f"scale_factor = {1.0 / PIXEL_TO_MICRON} pixels/µm")
    print(f"x_offset = {X_OFFSET} pixels")
    print(f"y_offset = {Y_OFFSET} pixels")
    print(f"\nYou can modify these values in the global parameters section of the code.")
    
    # Process each file set
    for i, (roi_file, process_traj_file, analyzed_traj_file) in enumerate(matched_files, 1):
        print(f"\n\nProcessing file set {i}/{len(matched_files)}")
        process_file_set(roi_file, process_traj_file, analyzed_traj_file)
    
    print("\n" + "="*70)
    print("All file sets processed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()