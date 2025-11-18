#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_roi_integration.py

Test script to verify the ROI-to-pipeline integration works correctly.
Creates mock data and runs the integration to verify output format.
"""

import os
import numpy as np
import pickle
import tempfile
import shutil

# Parameters matching main pipeline
DT = 0.1
CONVERSION = 0.094

def create_mock_trajectory(traj_id, num_points=50, center_x=10.0, center_y=10.0, noise=0.5):
    """Create a mock trajectory with random walk."""
    # Random walk in μm
    dx = np.random.randn(num_points) * noise
    dy = np.random.randn(num_points) * noise

    x = center_x + np.cumsum(dx)
    y = center_y + np.cumsum(dy)

    time = np.arange(num_points) * DT

    return {
        'id': traj_id,
        'x': x,
        'y': y,
        'time': time,
        'dx': dx,
        'dy': dy,
        'dr2': dx**2 + dy**2
    }

def create_mock_roi_data(num_rois=3, trajectories_per_roi=10):
    """Create mock ROI classification data."""

    roi_assignments = {}
    roi_trajectories = {}
    roi_statistics = {}

    traj_id_counter = 0

    # Create trajectories for each ROI
    for roi_idx in range(num_rois):
        roi_id = f"roi_{roi_idx:04d}"

        trajectories = []
        traj_indices = []

        for _ in range(trajectories_per_roi):
            # Create trajectory with center near this ROI
            center_x = 10.0 + roi_idx * 5.0
            center_y = 10.0 + roi_idx * 5.0

            traj = create_mock_trajectory(
                traj_id=traj_id_counter,
                center_x=center_x,
                center_y=center_y
            )

            trajectories.append(traj)
            traj_indices.append(traj_id_counter)
            traj_id_counter += 1

        roi_assignments[roi_id] = traj_indices
        roi_trajectories[roi_id] = trajectories

        # Mock statistics
        roi_statistics[roi_id] = {
            'n': len(trajectories),
            'mean_D': 0.1 + roi_idx * 0.05,
            'median_D': 0.1 + roi_idx * 0.05,
            'std_D': 0.02,
            'sem_D': 0.01,
            'min_D': 0.05,
            'max_D': 0.20
        }

    # Add unassigned trajectories
    unassigned_trajectories = []
    unassigned_indices = []

    for _ in range(5):
        traj = create_mock_trajectory(
            traj_id=traj_id_counter,
            center_x=50.0,
            center_y=50.0
        )
        unassigned_trajectories.append(traj)
        unassigned_indices.append(traj_id_counter)
        traj_id_counter += 1

    roi_assignments['unassigned'] = unassigned_indices
    roi_trajectories['unassigned'] = unassigned_trajectories
    roi_statistics['unassigned'] = {
        'n': len(unassigned_trajectories),
        'mean_D': np.nan,
        'median_D': np.nan,
        'std_D': np.nan,
        'sem_D': np.nan,
        'min_D': np.nan,
        'max_D': np.nan
    }

    # Complete ROI data structure
    roi_data = {
        'roi_assignments': roi_assignments,
        'roi_trajectories': roi_trajectories,
        'roi_statistics': roi_statistics,
        'coordinate_transform': {
            'pixel_to_micron': CONVERSION,
            'scale_factor': 1.0 / CONVERSION,
            'x_offset': 0.0,
            'y_offset': 0.0
        }
    }

    return roi_data

def verify_tracked_pkl_format(file_path):
    """Verify that a tracked_*.pkl file has the correct format."""

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Check required keys
        required_keys = ['trajectories', 'trajectory_lengths', 'msd_data', 'time_data']
        for key in required_keys:
            if key not in data:
                return False, f"Missing key: {key}"

        # Check data consistency
        num_trajs = len(data['trajectories'])

        if len(data['trajectory_lengths']) != num_trajs:
            return False, "Inconsistent trajectory_lengths"

        if len(data['msd_data']) != num_trajs:
            return False, "Inconsistent msd_data"

        if len(data['time_data']) != num_trajs:
            return False, "Inconsistent time_data"

        # Check trajectory structure
        for i, traj in enumerate(data['trajectories']):
            required_traj_keys = ['id', 'x', 'y', 'time', 'dx', 'dy', 'dr2']
            for key in required_traj_keys:
                if key not in traj:
                    return False, f"Trajectory {i} missing key: {key}"

        return True, f"Valid format with {num_trajs} trajectories"

    except Exception as e:
        return False, f"Error loading file: {e}"

def test_simple_integration():
    """Test the simple (single condition) integration."""

    print("\n" + "="*70)
    print("Test 1: Simple Integration (Single Condition)")
    print("="*70)

    # Create temporary directory
    test_dir = tempfile.mkdtemp(prefix="roi_integration_test_")
    print(f"Test directory: {test_dir}")

    try:
        # Create mock ROI data
        print("\nCreating mock ROI data...")
        roi_data = create_mock_roi_data(num_rois=3, trajectories_per_roi=10)

        # Save to pickle file
        roi_file = os.path.join(test_dir, "roi_trajectory_data.pkl")
        with open(roi_file, 'wb') as f:
            pickle.dump(roi_data, f)
        print(f"Saved mock data to: {roi_file}")

        # Import and run integration
        print("\nRunning integration...")
        import sys
        sys.path.insert(0, '/home/user/GEM_analysis')
        import roi_to_pipeline_integration as integration

        # Process the data
        output_dir = os.path.join(test_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Split trajectories
        inside_trajectories, outside_trajectories = integration.split_trajectories_by_roi(roi_data)

        # Convert to tracked format
        inside_tracked = integration.create_tracked_pkl_data(inside_trajectories)
        outside_tracked = integration.create_tracked_pkl_data(outside_trajectories)

        # Save tracked files
        inside_file = integration.save_tracked_pkl(inside_tracked, os.path.join(output_dir, "inside_roi"), "inside_roi")
        outside_file = integration.save_tracked_pkl(outside_tracked, os.path.join(output_dir, "outside_roi"), "outside_roi")

        print(f"\nVerifying output files...")

        # Verify inside ROI file
        is_valid, msg = verify_tracked_pkl_format(inside_file)
        if is_valid:
            print(f"✓ inside_roi: {msg}")
        else:
            print(f"✗ inside_roi: {msg}")
            return False

        # Verify outside ROI file
        is_valid, msg = verify_tracked_pkl_format(outside_file)
        if is_valid:
            print(f"✓ outside_roi: {msg}")
        else:
            print(f"✗ outside_roi: {msg}")
            return False

        print("\n✓ Test 1 PASSED: Simple integration works correctly")
        return True

    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory")

def test_multi_roi_integration():
    """Test the multi-ROI batch integration."""

    print("\n" + "="*70)
    print("Test 2: Multi-ROI Batch Integration")
    print("="*70)

    # Create temporary directory
    test_dir = tempfile.mkdtemp(prefix="roi_batch_test_")
    print(f"Test directory: {test_dir}")

    try:
        # Create multiple mock conditions
        print("\nCreating mock data for 3 conditions...")

        conditions = {
            'condition_A': create_mock_roi_data(num_rois=2, trajectories_per_roi=15),
            'condition_B': create_mock_roi_data(num_rois=3, trajectories_per_roi=10),
            'condition_C': create_mock_roi_data(num_rois=2, trajectories_per_roi=12)
        }

        # Save each condition
        for cond_name, roi_data in conditions.items():
            cond_dir = os.path.join(test_dir, cond_name)
            os.makedirs(cond_dir, exist_ok=True)

            roi_file = os.path.join(cond_dir, "roi_trajectory_data.pkl")
            with open(roi_file, 'wb') as f:
                pickle.dump(roi_data, f)
            print(f"  Created: {cond_name}/roi_trajectory_data.pkl")

        # Import batch processor
        print("\nRunning batch integration...")
        import sys
        sys.path.insert(0, '/home/user/GEM_analysis')
        import roi_to_pipeline_batch as batch_integration

        # Find ROI files
        roi_files = batch_integration.find_roi_data_files(test_dir)
        print(f"  Found {len(roi_files)} ROI data files")

        # Process each condition
        output_dir = os.path.join(test_dir, "batch_output")
        os.makedirs(output_dir, exist_ok=True)

        all_results = []
        for roi_file in roi_files:
            condition_name = batch_integration.extract_condition_name(roi_file, test_dir)
            result = batch_integration.process_single_condition(
                roi_file, output_dir, condition_name, multi_roi_mode=True
            )
            all_results.append(result)

        # Verify all conditions were processed successfully
        successful = sum(1 for r in all_results if r['status'] == 'success')

        if successful == len(conditions):
            print(f"\n✓ All {successful} conditions processed successfully")
        else:
            print(f"\n✗ Only {successful}/{len(conditions)} conditions processed")
            return False

        # Verify output files exist and are valid
        print("\nVerifying output files...")
        all_valid = True

        for result in all_results:
            if result['status'] == 'success':
                for file_path in result['files_created']:
                    is_valid, msg = verify_tracked_pkl_format(file_path)
                    rel_path = os.path.relpath(file_path, output_dir)

                    if is_valid:
                        print(f"  ✓ {rel_path}: {msg}")
                    else:
                        print(f"  ✗ {rel_path}: {msg}")
                        all_valid = False

        if all_valid:
            print("\n✓ Test 2 PASSED: Multi-ROI batch integration works correctly")
            return True
        else:
            print("\n✗ Test 2 FAILED: Some output files are invalid")
            return False

    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory")

def main():
    """Run all tests."""

    print("\n" + "="*70)
    print("ROI-to-Pipeline Integration Test Suite")
    print("="*70)

    # Run tests
    test1_passed = test_simple_integration()
    test2_passed = test_multi_roi_integration()

    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Test 1 (Simple Integration): {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Test 2 (Multi-ROI Batch):     {'PASSED' if test2_passed else 'FAILED'}")

    if test1_passed and test2_passed:
        print("\n✓ All tests PASSED")
        return 0
    else:
        print("\n✗ Some tests FAILED")
        return 1

if __name__ == "__main__":
    exit(main())
