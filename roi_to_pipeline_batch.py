#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
roi_to_pipeline_batch.py

Batch version: Integrates multiple ROI-classified datasets with the main diffusion
analysis pipeline. Supports both simple inside/outside splitting and multi-ROI mode.

This script processes multiple ROI classification results and outputs tracked_*.pkl
files that are compatible with the main pipeline (starting from Step 2: 2traj_analyze_v1.py).

Workflow:
1. Scan directory for roi_trajectory_data.pkl files
2. For each file, split trajectories by ROI
3. Create tracked_*.pkl files in the format expected by main pipeline
4. Organize output in proper directory structure
5. Preserve existing ROI classification images

Modes:
- Simple mode: inside_roi/ and outside_roi/ only
- Multi-ROI mode: separate directory for each individual ROI

Input:
- Directory containing roi_trajectory_data.pkl files from ROI classification

Output:
- Directory structure (Simple mode):
    batch_output/
        ├── condition1/
        │   ├── inside_roi/tracked_inside_roi.pkl
        │   ├── outside_roi/tracked_outside_roi.pkl
        │   └── roi_classification_images/
        ├── condition2/
        │   ├── inside_roi/tracked_inside_roi.pkl
        │   └── ...
        └── integration_summary.txt

- Directory structure (Multi-ROI mode):
    batch_output/
        ├── condition1/
        │   ├── roi_0001/tracked_roi_0001.pkl
        │   ├── roi_0002/tracked_roi_0002.pkl
        │   ├── unassigned/tracked_unassigned.pkl
        │   ├── inside_roi/tracked_inside_roi.pkl  (combined)
        │   ├── outside_roi/tracked_outside_roi.pkl
        │   └── roi_classification_images/
        └── ...

Usage:
python roi_to_pipeline_batch.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shutil
from pathlib import Path
from datetime import datetime
import glob

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

        # Display summary
        if 'roi_assignments' in data:
            total_trajectories = sum(len(indices) for indices in data['roi_assignments'].values())
            print(f"  Total trajectories: {total_trajectories}")
            for roi_id, indices in data['roi_assignments'].items():
                if len(indices) > 0:
                    print(f"    {roi_id}: {len(indices)} trajectories")

        return data
    except Exception as e:
        print(f"  Error loading ROI data from {file_path}: {e}")
        return None

def check_parameter_consistency(roi_data, verbose=False):
    """
    Check if coordinate transformation parameters match the main pipeline.

    Args:
        roi_data: Dictionary containing ROI classification results
        verbose: Whether to print detailed warnings

    Returns:
        Boolean indicating if parameters are consistent
    """
    if 'coordinate_transform' in roi_data:
        pixel_to_micron = roi_data['coordinate_transform'].get('pixel_to_micron', None)

        if pixel_to_micron is not None:
            # Check if it matches CONVERSION from main pipeline
            if abs(pixel_to_micron - CONVERSION) > 0.001:
                if verbose:
                    print(f"  ⚠️  WARNING: Parameter mismatch detected!")
                    print(f"     ROI classification used: {pixel_to_micron} μm/pixel")
                    print(f"     Main pipeline uses: {CONVERSION} μm/pixel")
                return False

    return True

def create_tracked_pkl_data(trajectories):
    """
    Convert trajectory list to tracked_*.pkl format expected by main pipeline.

    Args:
        trajectories: List of trajectory dictionaries from ROI classification

    Returns:
        Dictionary in tracked_*.pkl format
    """
    if not trajectories:
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

    return tracked_data

def save_tracked_pkl(data, output_path, name):
    """
    Save data in tracked_*.pkl format.

    Args:
        data: Dictionary containing tracked trajectory data
        output_path: Directory to save the file
        name: Base name for the file (e.g., 'inside_roi', 'roi_0001')
    """
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, f"tracked_{name}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

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

    return copied_count

def process_single_condition(roi_data_file, output_base_dir, condition_name, multi_roi_mode=False):
    """
    Process a single ROI classification file.

    Args:
        roi_data_file: Path to roi_trajectory_data.pkl
        output_base_dir: Base directory for all outputs
        condition_name: Name for this condition
        multi_roi_mode: If True, create separate directories for each ROI

    Returns:
        Dictionary with processing results
    """
    print(f"\n{'='*70}")
    print(f"Processing: {condition_name}")
    print(f"{'='*70}")

    # Load ROI data
    print(f"Loading {os.path.basename(roi_data_file)}...")
    roi_data = load_roi_data(roi_data_file)
    if roi_data is None:
        return {'status': 'failed', 'reason': 'Failed to load ROI data'}

    # Check parameter consistency
    if not check_parameter_consistency(roi_data, verbose=True):
        print("  Continuing despite parameter mismatch...")

    # Get ROI assignments and trajectories
    roi_assignments = roi_data['roi_assignments']
    roi_trajectories = roi_data.get('roi_trajectories', {})

    # Create output directory for this condition
    condition_dir = os.path.join(output_base_dir, condition_name)
    os.makedirs(condition_dir, exist_ok=True)

    results = {
        'status': 'success',
        'condition': condition_name,
        'roi_counts': {},
        'files_created': []
    }

    # Process in multi-ROI mode
    if multi_roi_mode:
        print("\n  Multi-ROI mode: Creating separate directories for each ROI")

        # Process each ROI individually
        for roi_id, trajs in roi_trajectories.items():
            if len(trajs) == 0:
                continue

            print(f"    Processing {roi_id}: {len(trajs)} trajectories")

            # Convert to tracked format
            tracked_data = create_tracked_pkl_data(trajs)

            if len(tracked_data['trajectories']) > 0:
                # Create directory for this ROI
                roi_dir = os.path.join(condition_dir, roi_id)

                # Save tracked file
                output_file = save_tracked_pkl(tracked_data, roi_dir, roi_id)

                results['roi_counts'][roi_id] = len(tracked_data['trajectories'])
                results['files_created'].append(output_file)
                print(f"      Saved: {os.path.relpath(output_file, output_base_dir)}")
            else:
                print(f"      Skipped: No valid trajectories after filtering")

    # Also create combined inside/outside directories (always)
    print("\n  Creating combined inside/outside ROI directories")

    # Collect inside ROI trajectories (all ROIs except 'unassigned')
    inside_trajectories = []
    for roi_id, trajs in roi_trajectories.items():
        if roi_id != 'unassigned':
            inside_trajectories.extend(trajs)

    # Collect outside ROI trajectories (unassigned)
    outside_trajectories = roi_trajectories.get('unassigned', [])

    print(f"    Inside ROI: {len(inside_trajectories)} trajectories")
    print(f"    Outside ROI: {len(outside_trajectories)} trajectories")

    # Convert to tracked format
    inside_tracked = create_tracked_pkl_data(inside_trajectories)
    outside_tracked = create_tracked_pkl_data(outside_trajectories)

    # Save inside ROI
    if len(inside_tracked['trajectories']) > 0:
        inside_dir = os.path.join(condition_dir, "inside_roi")
        inside_file = save_tracked_pkl(inside_tracked, inside_dir, "inside_roi")
        results['roi_counts']['inside_roi'] = len(inside_tracked['trajectories'])
        results['files_created'].append(inside_file)
        print(f"      Saved: {os.path.relpath(inside_file, output_base_dir)}")
    else:
        print(f"      Inside ROI: No valid trajectories after filtering")

    # Save outside ROI
    if len(outside_tracked['trajectories']) > 0:
        outside_dir = os.path.join(condition_dir, "outside_roi")
        outside_file = save_tracked_pkl(outside_tracked, outside_dir, "outside_roi")
        results['roi_counts']['outside_roi'] = len(outside_tracked['trajectories'])
        results['files_created'].append(outside_file)
        print(f"      Saved: {os.path.relpath(outside_file, output_base_dir)}")
    else:
        print(f"      Outside ROI: No valid trajectories after filtering")

    # Copy ROI classification images
    print("\n  Copying ROI classification images...")
    source_dir = os.path.dirname(roi_data_file)
    copied_count = copy_roi_images(source_dir, condition_dir)
    if copied_count > 0:
        print(f"    Copied {copied_count} image(s)")
    else:
        print(f"    No images found to copy")

    return results

def create_batch_summary(output_base_dir, all_results, multi_roi_mode):
    """
    Create a summary report for the batch processing.

    Args:
        output_base_dir: Output directory
        all_results: List of processing results for each condition
        multi_roi_mode: Whether multi-ROI mode was used
    """
    summary_file = os.path.join(output_base_dir, 'batch_integration_summary.txt')

    with open(summary_file, 'w') as f:
        f.write("ROI-to-Pipeline Batch Integration Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mode: {'Multi-ROI' if multi_roi_mode else 'Simple (Inside/Outside only)'}\n\n")

        f.write("Parameters:\n")
        f.write(f"  DT = {DT} s\n")
        f.write(f"  CONVERSION = {CONVERSION} μm/pixel\n")
        f.write(f"  MIN_TRACK_LENGTH = {MIN_TRACK_LENGTH} frames\n\n")

        f.write("Processed Conditions:\n")
        f.write("-" * 70 + "\n")

        total_conditions = 0
        total_files = 0

        for result in all_results:
            if result['status'] == 'success':
                total_conditions += 1
                f.write(f"\n{result['condition']}:\n")

                for roi_id, count in result['roi_counts'].items():
                    f.write(f"  {roi_id}: {count} trajectories\n")

                f.write(f"  Files created: {len(result['files_created'])}\n")
                total_files += len(result['files_created'])
            else:
                f.write(f"\n{result.get('condition', 'Unknown')}: FAILED\n")
                f.write(f"  Reason: {result.get('reason', 'Unknown error')}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Total conditions processed: {total_conditions}\n")
        f.write(f"Total tracked_*.pkl files created: {total_files}\n\n")

        f.write("Next Steps:\n")
        f.write("  1. Navigate to each condition directory\n")
        f.write("  2. For each ROI subdirectory, run main pipeline Step 2:\n")
        f.write("     cd <condition>/<roi_dir>/\n")
        f.write("     python ../../2traj_analyze_v1.py\n")
        f.write("  3. Continue with subsequent pipeline steps (3, 4, 5, ...) for each ROI\n")
        f.write("  4. Use comparison scripts to compare different ROI conditions\n")

    print(f"\nBatch summary saved to: {summary_file}")

def find_roi_data_files(directory):
    """
    Find all roi_trajectory_data.pkl files in a directory.

    Args:
        directory: Directory to search

    Returns:
        List of file paths
    """
    search_pattern = os.path.join(directory, "**", "roi_trajectory_data.pkl")
    files = glob.glob(search_pattern, recursive=True)
    return files

def extract_condition_name(file_path, base_dir):
    """
    Extract a meaningful condition name from the file path.

    Args:
        file_path: Path to roi_trajectory_data.pkl
        base_dir: Base directory for comparison

    Returns:
        Condition name string
    """
    # Get the parent directory name
    parent_dir = os.path.dirname(file_path)

    # Try to extract condition name from directory structure
    rel_path = os.path.relpath(parent_dir, base_dir)

    # Clean up the path
    condition_name = rel_path.replace(os.sep, "_")

    # If it's just ".", use the parent directory name
    if condition_name == ".":
        condition_name = os.path.basename(parent_dir)

    # Remove common prefixes
    condition_name = condition_name.replace("roi_diffusion_", "")

    return condition_name

def main():
    """
    Main function for batch processing of ROI classification results.
    """
    print("\n" + "=" * 70)
    print("ROI-to-Pipeline Batch Integration Tool")
    print("=" * 70 + "\n")

    # Get input directory
    input_dir = input("Enter directory containing ROI classification results: ")

    if not os.path.exists(input_dir):
        print(f"Error: Directory not found: {input_dir}")
        return

    # Find all roi_trajectory_data.pkl files
    print(f"\nSearching for roi_trajectory_data.pkl files in {input_dir}...")
    roi_files = find_roi_data_files(input_dir)

    if len(roi_files) == 0:
        print(f"No roi_trajectory_data.pkl files found in {input_dir}")
        return

    print(f"Found {len(roi_files)} file(s) to process:")
    for i, file in enumerate(roi_files, 1):
        print(f"  {i}. {os.path.relpath(file, input_dir)}")

    # Ask for multi-ROI mode
    print("\nProcessing mode:")
    print("  1. Simple mode (inside_roi and outside_roi only)")
    print("  2. Multi-ROI mode (separate directory for each ROI)")
    mode_choice = input("Select mode [1]: ").strip()

    multi_roi_mode = (mode_choice == "2")

    if multi_roi_mode:
        print("\n✓ Multi-ROI mode selected")
    else:
        print("\n✓ Simple mode selected (inside/outside only)")

    # Get output directory
    default_output = os.path.join(input_dir, "pipeline_batch_integrated")
    output_dir = input(f"\nEnter output directory [{default_output}]: ").strip()

    if output_dir == "":
        output_dir = default_output

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Process each file
    all_results = []

    for roi_file in roi_files:
        # Extract condition name
        condition_name = extract_condition_name(roi_file, input_dir)

        # Process this condition
        result = process_single_condition(roi_file, output_dir, condition_name, multi_roi_mode)
        all_results.append(result)

    # Create batch summary
    print("\n" + "=" * 70)
    print("Creating batch summary...")
    create_batch_summary(output_dir, all_results, multi_roi_mode)

    # Print final summary
    print("\n" + "=" * 70)
    print("Batch Integration Complete!")
    print("=" * 70)

    successful = sum(1 for r in all_results if r['status'] == 'success')
    total_files = sum(len(r.get('files_created', [])) for r in all_results if r['status'] == 'success')

    print(f"\nSuccessfully processed: {successful}/{len(roi_files)} conditions")
    print(f"Total tracked_*.pkl files created: {total_files}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Summary report: {os.path.join(output_dir, 'batch_integration_summary.txt')}\n")

if __name__ == "__main__":
    main()
