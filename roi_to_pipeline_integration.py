#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
roi_to_pipeline_integration.py

Integrates ROI-based classification with the main diffusion analysis pipeline.

This script takes ROI classification results and outputs tracked_*.pkl files
that are compatible with the main pipeline (starting from Step 2: 2traj_analyze_v1.py).

Workflow:
1. Load ROI-classified trajectory data
2. Split trajectories into "inside_roi" and "outside_roi" groups
3. Create tracked_*.pkl files in the format expected by main pipeline
4. Organize output in proper directory structure
5. Preserve existing ROI classification images

Input:
- ROI classification output (roi_trajectory_data.pkl from ROI classification step)

Output:
- Directory structure:
    condition_name/
        ├── inside_roi/
        │   └── tracked_inside_roi.pkl
        ├── outside_roi/
        │   └── tracked_outside_roi.pkl
        └── roi_classification_images/
            └── [preserved images from ROI classification]

Usage:
python roi_to_pipeline_integration.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shutil
from pathlib import Path
from datetime import datetime

# Global parameters - MUST match main pipeline!
# =====================================
DT = 0.1  # Time step in seconds (must match main pipeline)
CONVERSION = 0.094  # Conversion factor from pixels to μm (must match main pipeline)
MIN_TRACK_LENGTH = 10  # Minimum track length (in frames)
# =====================================

def load_roi_data(file_path):
    """
    Load ROI classification results from pickle file.

    Args:
        file_path: Path to the roi_trajectory_data.pkl file

    Returns:
        Dictionary containing ROI classification results
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded ROI data from {file_path}")

        # Display summary
        if 'roi_assignments' in data:
            total_trajectories = sum(len(indices) for indices in data['roi_assignments'].values())
            print(f"Total trajectories: {total_trajectories}")
            for roi_id, indices in data['roi_assignments'].items():
                print(f"  {roi_id}: {len(indices)} trajectories")

        return data
    except Exception as e:
        print(f"Error loading ROI data from {file_path}: {e}")
        return None

def check_parameter_consistency(roi_data):
    """
    Check if coordinate transformation parameters match the main pipeline.

    Args:
        roi_data: Dictionary containing ROI classification results

    Returns:
        Boolean indicating if parameters are consistent
    """
    if 'coordinate_transform' in roi_data:
        pixel_to_micron = roi_data['coordinate_transform'].get('pixel_to_micron', None)

        if pixel_to_micron is not None:
            # Check if it matches CONVERSION from main pipeline
            if abs(pixel_to_micron - CONVERSION) > 0.001:
                print(f"\n⚠️  WARNING: Parameter mismatch detected!")
                print(f"   ROI classification used: {pixel_to_micron} μm/pixel")
                print(f"   Main pipeline uses: {CONVERSION} μm/pixel")
                print(f"   This mismatch may affect coordinate consistency.\n")

                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    return False

    return True

def split_trajectories_by_roi(roi_data):
    """
    Split trajectories into inside_roi and outside_roi groups.

    Args:
        roi_data: Dictionary containing ROI classification results

    Returns:
        Tuple of (inside_trajectories, outside_trajectories)
    """
    roi_assignments = roi_data['roi_assignments']
    roi_trajectories = roi_data.get('roi_trajectories', {})

    # Collect inside ROI trajectories (all ROIs except 'unassigned')
    inside_trajectories = []
    for roi_id, trajs in roi_trajectories.items():
        if roi_id != 'unassigned':
            inside_trajectories.extend(trajs)

    # Collect outside ROI trajectories (unassigned)
    outside_trajectories = roi_trajectories.get('unassigned', [])

    print(f"\nSplit results:")
    print(f"  Inside ROI: {len(inside_trajectories)} trajectories")
    print(f"  Outside ROI: {len(outside_trajectories)} trajectories")

    return inside_trajectories, outside_trajectories

def create_tracked_pkl_data(trajectories):
    """
    Convert trajectory list to tracked_*.pkl format expected by main pipeline.

    The main pipeline expects this structure:
    {
        'trajectories': [...],  # List of trajectory dicts
        'trajectory_lengths': [...],
        'msd_data': [...],
        'time_data': [...]
    }

    Args:
        trajectories: List of trajectory dictionaries from ROI classification

    Returns:
        Dictionary in tracked_*.pkl format
    """
    if not trajectories:
        print("Warning: No trajectories to process")
        return {
            'trajectories': [],
            'trajectory_lengths': [],
            'msd_data': [],
            'time_data': []
        }

    tracked_data = {
        'trajectories': [],
        'trajectory_lengths': [],
        'msd_data': [],
        'time_data': []
    }

    for traj in trajectories:
        # Skip if trajectory is too short
        if len(traj['x']) < MIN_TRACK_LENGTH:
            continue

        # Extract coordinates
        x = traj['x']
        y = traj['y']
        time = traj.get('time', np.arange(len(x)) * DT)

        # Calculate displacements
        dx = np.diff(x)
        dy = np.diff(y)
        dr2 = dx**2 + dy**2

        # Calculate MSD for different time lags
        max_dt_index = len(x) // 2
        msd = np.zeros(max_dt_index)

        for dt_index in range(1, max_dt_index + 1):
            # Calculate all possible displacements for this time lag
            total_displacement = 0
            count = 0

            for i in range(len(x) - dt_index):
                dx_lag = x[i + dt_index] - x[i]
                dy_lag = y[i + dt_index] - y[i]
                total_displacement += dx_lag**2 + dy_lag**2
                count += 1

            if count > 0:
                msd[dt_index - 1] = total_displacement / count
            else:
                msd[dt_index - 1] = np.nan

        # Store trajectory data in main pipeline format
        tracked_data['trajectories'].append({
            'id': traj['id'],
            'x': x,
            'y': y,
            'time': time,
            'dx': dx,
            'dy': dy,
            'dr2': dr2
        })

        tracked_data['trajectory_lengths'].append(len(x))
        tracked_data['msd_data'].append(msd)
        tracked_data['time_data'].append(np.arange(1, max_dt_index + 1) * DT)

    print(f"  Processed {len(tracked_data['trajectories'])} valid trajectories")

    return tracked_data

def save_tracked_pkl(data, output_path, name):
    """
    Save data in tracked_*.pkl format.

    Args:
        data: Dictionary containing tracked trajectory data
        output_path: Directory to save the file
        name: Base name for the file (e.g., 'inside_roi', 'outside_roi')
    """
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, f"tracked_{name}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved tracked data to: {output_file}")

    return output_file

def copy_roi_images(source_dir, target_dir):
    """
    Copy ROI classification images to output directory.

    Args:
        source_dir: Source directory containing ROI classification results
        target_dir: Target directory to copy images to
    """
    # Create target directory
    image_dir = os.path.join(target_dir, 'roi_classification_images')
    os.makedirs(image_dir, exist_ok=True)

    # Find and copy image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.pdf', '.svg']
    copied_count = 0

    for ext in image_extensions:
        for image_file in Path(source_dir).glob(f'*{ext}'):
            shutil.copy2(image_file, image_dir)
            copied_count += 1

    if copied_count > 0:
        print(f"Copied {copied_count} image(s) to: {image_dir}")
    else:
        print("No images found to copy")

    return image_dir

def create_summary_report(output_dir, inside_data, outside_data, roi_data):
    """
    Create a summary report of the integration.

    Args:
        output_dir: Output directory
        inside_data: Inside ROI tracked data
        outside_data: Outside ROI tracked data
        roi_data: Original ROI classification data
    """
    report_file = os.path.join(output_dir, 'integration_summary.txt')

    with open(report_file, 'w') as f:
        f.write("ROI-to-Pipeline Integration Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Parameters:\n")
        f.write(f"  DT = {DT} s\n")
        f.write(f"  CONVERSION = {CONVERSION} μm/pixel\n")
        f.write(f"  MIN_TRACK_LENGTH = {MIN_TRACK_LENGTH} frames\n\n")

        if 'coordinate_transform' in roi_data:
            f.write("Original ROI Classification Parameters:\n")
            transform = roi_data['coordinate_transform']
            f.write(f"  PIXEL_TO_MICRON = {transform.get('pixel_to_micron', 'N/A')} μm/pixel\n")
            f.write(f"  X_OFFSET = {transform.get('x_offset', 'N/A')} pixels\n")
            f.write(f"  Y_OFFSET = {transform.get('y_offset', 'N/A')} pixels\n\n")

        f.write("Trajectory Counts:\n")
        f.write(f"  Inside ROI: {len(inside_data['trajectories'])} trajectories\n")
        f.write(f"  Outside ROI: {len(outside_data['trajectories'])} trajectories\n")
        f.write(f"  Total: {len(inside_data['trajectories']) + len(outside_data['trajectories'])} trajectories\n\n")

        f.write("Output Files:\n")
        f.write(f"  inside_roi/tracked_inside_roi.pkl\n")
        f.write(f"  outside_roi/tracked_outside_roi.pkl\n\n")

        f.write("Next Steps:\n")
        f.write("  1. Run main pipeline Step 2 (2traj_analyze_v1.py) on inside_roi/tracked_inside_roi.pkl\n")
        f.write("  2. Run main pipeline Step 2 (2traj_analyze_v1.py) on outside_roi/tracked_outside_roi.pkl\n")
        f.write("  3. Continue with subsequent pipeline steps (3, 4, 5, ...) for each condition\n")
        f.write("  4. Use comparison scripts to compare inside vs outside ROI results\n")

    print(f"Summary report saved to: {report_file}")

def main():
    """
    Main function to integrate ROI classification with main pipeline.
    """
    print("\n" + "=" * 60)
    print("ROI-to-Pipeline Integration Tool")
    print("=" * 60 + "\n")

    # Get input file
    roi_data_file = input("Enter path to ROI classification results (roi_trajectory_data.pkl): ")

    if not os.path.exists(roi_data_file):
        print(f"Error: File not found: {roi_data_file}")
        return

    # Load ROI data
    roi_data = load_roi_data(roi_data_file)
    if roi_data is None:
        return

    # Check parameter consistency
    if not check_parameter_consistency(roi_data):
        print("Integration aborted due to parameter mismatch.")
        return

    # Split trajectories
    inside_trajectories, outside_trajectories = split_trajectories_by_roi(roi_data)

    if len(inside_trajectories) == 0 and len(outside_trajectories) == 0:
        print("Error: No trajectories found in ROI data")
        return

    # Get output directory
    default_output = os.path.join(os.path.dirname(roi_data_file), "pipeline_integrated")
    output_dir = input(f"Enter output directory [{default_output}]: ")

    if output_dir.strip() == "":
        output_dir = default_output

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Convert to tracked_*.pkl format
    print("\nConverting inside ROI trajectories...")
    inside_tracked = create_tracked_pkl_data(inside_trajectories)

    print("\nConverting outside ROI trajectories...")
    outside_tracked = create_tracked_pkl_data(outside_trajectories)

    # Save tracked files
    print("\nSaving tracked pickle files...")
    inside_dir = os.path.join(output_dir, "inside_roi")
    outside_dir = os.path.join(output_dir, "outside_roi")

    save_tracked_pkl(inside_tracked, inside_dir, "inside_roi")
    save_tracked_pkl(outside_tracked, outside_dir, "outside_roi")

    # Copy ROI classification images
    print("\nCopying ROI classification images...")
    source_dir = os.path.dirname(roi_data_file)
    copy_roi_images(source_dir, output_dir)

    # Create summary report
    print("\nCreating summary report...")
    create_summary_report(output_dir, inside_tracked, outside_tracked, roi_data)

    print("\n" + "=" * 60)
    print("Integration Complete!")
    print("=" * 60)
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    ├── inside_roi/")
    print(f"    │   └── tracked_inside_roi.pkl")
    print(f"    ├── outside_roi/")
    print(f"    │   └── tracked_outside_roi.pkl")
    print(f"    ├── roi_classification_images/")
    print(f"    └── integration_summary.txt")
    print(f"\nNext steps:")
    print(f"  1. Run Step 2 of main pipeline on inside_roi/tracked_inside_roi.pkl")
    print(f"  2. Run Step 2 of main pipeline on outside_roi/tracked_outside_roi.pkl")
    print(f"  3. Continue with subsequent analysis steps")
    print(f"  4. Compare results using existing comparison tools\n")

if __name__ == "__main__":
    main()
