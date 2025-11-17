#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diffusion_analyzer_v3_tau_filtering.py

VERSION 3: Adds tau-specific filtering based on trajectory length

This script analyzes processed trajectory data using tau-aware filtering
to ensure sufficient averaging at each time lag.

NEW IN V3 (builds on V2):
- Dynamic max_tau calculation: max_tau = trajectory_length / 4
- Ensures sufficient number of displacement pairs for MSD averaging
- Addresses "poor averaging at long lags" issue from review
- Reports number of points used for fitting
- Warning system for trajectories that are too short

Theory:
- At tau = N/4, we have ~N/4 non-overlapping intervals for averaging
- This ensures reliable statistics while maximizing the fitting window
- Prevents overfitting to noisy long-lag MSD values

Input:
- Processed trajectory data (.pkl files) from trajectory_processor.py

Output:
- Analyzed trajectory data with tau-filtered fits
- Diagnostic plots showing filtering effects
- CSV with fitting statistics

Usage:
python diffusion_analyzer_v3_tau_filtering.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import glob
import pickle
from pathlib import Path
import warnings

# Global parameters that can be modified
# =====================================
# Time step in seconds (default: 0.05s = 50ms)
DT = 0.1
# Conversion factor from pixels to μm (default: 1.0 for TrackMate which outputs in μm)
CONVERSION = 0.094
# Fraction of trajectory to use for max tau (default: 1/4)
TAU_FRACTION = 0.25
# Minimum number of points required for fitting
MIN_FIT_POINTS = 5
# Number of bootstrap iterations
N_BOOTSTRAP = 1000
# =====================================

def load_processed_data(file_path):
    """
    Load processed trajectory data from a pickle file.

    Args:
        file_path: Path to the pickle file

    Returns:
        Dictionary containing the processed trajectory data
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading processed data from {file_path}: {e}")
        return None

def linear_msd(t, D, offset):
    """
    Linear MSD model: MSD = 4*D*t + offset

    Args:
        t: Time lag
        D: Diffusion coefficient
        offset: Y-intercept = 4*σ²_loc

    Returns:
        MSD values according to the model
    """
    return 4 * D * t + offset

def calculate_max_tau_index(trajectory_length, tau_fraction=TAU_FRACTION):
    """
    Calculate the maximum tau index to use for fitting based on trajectory length.

    Theory: At tau = N * tau_fraction, we have approximately N * tau_fraction
    non-overlapping displacement pairs, ensuring good averaging statistics.

    Args:
        trajectory_length: Number of frames in the trajectory
        tau_fraction: Fraction of trajectory length to use (default: 0.25)

    Returns:
        max_tau_index: Maximum index to use for MSD fitting
        warning: Warning message if trajectory is too short
    """
    max_tau_index = int(trajectory_length * tau_fraction)

    # Check if we have enough points
    warning = None
    if max_tau_index < MIN_FIT_POINTS:
        warning = f"Trajectory too short ({trajectory_length} frames) - only {max_tau_index} points available for fitting"
        max_tau_index = min(trajectory_length - 1, MIN_FIT_POINTS)

    return max_tau_index, warning

def bootstrap_fit_msd_with_tau_filter(time_data, msd_data, trajectory_length, n_bootstrap=N_BOOTSTRAP):
    """
    Fit MSD curve using bootstrap with tau-aware filtering.

    Args:
        time_data: Array of time lag values
        msd_data: Array of MSD values
        trajectory_length: Number of frames in original trajectory
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dictionary with fitting parameters and bootstrap confidence intervals
    """
    # Calculate max tau index based on trajectory length
    max_tau_index, tau_warning = calculate_max_tau_index(trajectory_length)

    # Extract data up to max_tau
    t_fit = time_data[:max_tau_index]
    msd_fit = msd_data[:max_tau_index]

    # Filter out any NaN values
    valid_indices = ~np.isnan(msd_fit)
    t_fit = t_fit[valid_indices]
    msd_fit = msd_fit[valid_indices]

    # Check if we have enough points after filtering
    if len(t_fit) < 3:
        return {
            'D': np.nan,
            'offset': np.nan,
            'sigma_loc': np.nan,
            'D_CI_low': np.nan,
            'D_CI_high': np.nan,
            'sigma_loc_CI_low': np.nan,
            'sigma_loc_CI_high': np.nan,
            'r_squared': np.nan,
            't_fit': t_fit,
            'msd_fit': msd_fit,
            'fit_values': np.nan,
            'n_fit_points': len(t_fit),
            'max_tau_used': np.nan,
            'tau_warning': tau_warning
        }

    try:
        # First, get the main fit
        popt, pcov = curve_fit(linear_msd, t_fit, msd_fit)
        D_main, offset_main = popt

        # Calculate localization error from offset
        if offset_main > 0:
            sigma_loc_main = np.sqrt(offset_main / 4.0)
        else:
            sigma_loc_main = 0.0

        # Calculate fit quality (R²)
        fit_values = linear_msd(t_fit, D_main, offset_main)
        residuals = msd_fit - fit_values
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((msd_fit - np.mean(msd_fit))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Bootstrap resampling
        D_bootstrap = []
        sigma_loc_bootstrap = []

        n_points = len(t_fit)

        for _ in range(n_bootstrap):
            # Resample indices with replacement
            indices = np.random.choice(n_points, size=n_points, replace=True)
            t_boot = t_fit[indices]
            msd_boot = msd_fit[indices]

            try:
                # Fit the resampled data
                popt_boot, _ = curve_fit(linear_msd, t_boot, msd_boot)
                D_boot, offset_boot = popt_boot

                D_bootstrap.append(D_boot)

                # Calculate sigma_loc for this bootstrap sample
                if offset_boot > 0:
                    sigma_loc_bootstrap.append(np.sqrt(offset_boot / 4.0))
                else:
                    sigma_loc_bootstrap.append(0.0)
            except:
                continue

        # Calculate 95% confidence intervals
        if len(D_bootstrap) > 0:
            D_CI_low, D_CI_high = np.percentile(D_bootstrap, [2.5, 97.5])
            sigma_loc_CI_low, sigma_loc_CI_high = np.percentile(sigma_loc_bootstrap, [2.5, 97.5])
        else:
            D_CI_low = D_CI_high = np.nan
            sigma_loc_CI_low = sigma_loc_CI_high = np.nan

        return {
            'D': D_main,
            'offset': offset_main,
            'sigma_loc': sigma_loc_main,
            'D_CI_low': D_CI_low,
            'D_CI_high': D_CI_high,
            'sigma_loc_CI_low': sigma_loc_CI_low,
            'sigma_loc_CI_high': sigma_loc_CI_high,
            'r_squared': r_squared,
            't_fit': t_fit,
            'msd_fit': msd_fit,
            'fit_values': fit_values,
            'n_fit_points': len(t_fit),  # NEW: Number of points used
            'max_tau_used': t_fit[-1] if len(t_fit) > 0 else np.nan,  # NEW: Max tau value used
            'tau_warning': tau_warning  # NEW: Warning message if any
        }
    except Exception as e:
        print(f"Error during MSD fitting: {e}")
        return {
            'D': np.nan,
            'offset': np.nan,
            'sigma_loc': np.nan,
            'D_CI_low': np.nan,
            'D_CI_high': np.nan,
            'sigma_loc_CI_low': np.nan,
            'sigma_loc_CI_high': np.nan,
            'r_squared': np.nan,
            't_fit': t_fit,
            'msd_fit': msd_fit,
            'fit_values': np.nan,
            'n_fit_points': len(t_fit),
            'max_tau_used': np.nan,
            'tau_warning': tau_warning
        }

def calculate_radius_of_gyration(trajectory):
    """
    Calculate the radius of gyration for a trajectory.

    Args:
        trajectory: Dictionary containing trajectory data

    Returns:
        Radius of gyration value
    """
    x = trajectory['x']
    y = trajectory['y']

    x_cm = np.mean(x)
    y_cm = np.mean(y)

    r2 = (x - x_cm)**2 + (y - y_cm)**2
    rg = np.sqrt(np.mean(r2))

    return rg

def analyze_trajectories(processed_data):
    """
    Analyze processed trajectory data using tau-aware filtering.

    Args:
        processed_data: Dictionary containing processed trajectory data

    Returns:
        Dictionary with analysis results
    """
    analyzed_data = {
        'trajectories': [],
        'diffusion_coefficients': [],
        'localization_errors': [],
        'D_CI_low': [],
        'D_CI_high': [],
        'n_fit_points': [],  # NEW: Track number of points used
        'max_tau_used': [],  # NEW: Track max tau used
        'radius_of_gyration': [],
        'track_lengths': [],
        'r_squared_values': [],
        'warnings': []  # NEW: Track warnings
    }

    n_warnings = 0

    for i, trajectory in enumerate(processed_data['trajectories']):
        if i % 10 == 0:
            print(f"  Processing trajectory {i+1}/{len(processed_data['trajectories'])}...")

        # Get MSD data for this trajectory
        time_data = processed_data['time_data'][i]
        msd_data = processed_data['msd_data'][i]
        trajectory_length = len(trajectory['x'])

        # Fit MSD curve with tau filtering
        fit_results = bootstrap_fit_msd_with_tau_filter(time_data, msd_data, trajectory_length)

        # Track warnings
        if fit_results['tau_warning'] is not None:
            n_warnings += 1
            analyzed_data['warnings'].append({
                'trajectory_id': trajectory['id'],
                'warning': fit_results['tau_warning']
            })

        # Calculate radius of gyration
        rg = calculate_radius_of_gyration(trajectory)

        # Store analysis results
        trajectory_analysis = {
            'id': trajectory['id'],
            'D': fit_results['D'],
            'D_CI_low': fit_results['D_CI_low'],
            'D_CI_high': fit_results['D_CI_high'],
            'offset': fit_results['offset'],
            'sigma_loc': fit_results['sigma_loc'],
            'sigma_loc_CI_low': fit_results['sigma_loc_CI_low'],
            'sigma_loc_CI_high': fit_results['sigma_loc_CI_high'],
            'r_squared': fit_results['r_squared'],
            'n_fit_points': fit_results['n_fit_points'],
            'max_tau_used': fit_results['max_tau_used'],
            'tau_warning': fit_results['tau_warning'],
            'radius_of_gyration': rg,
            'track_length': trajectory_length,
            'msd_data': msd_data,
            'time_data': time_data,
            't_fit': fit_results['t_fit'],
            'msd_fit': fit_results['msd_fit'],
            'fit_values': fit_results['fit_values'],
            'x': trajectory['x'],
            'y': trajectory['y']
        }

        analyzed_data['trajectories'].append(trajectory_analysis)

        # Store in separate lists
        if not np.isnan(fit_results['D']):
            analyzed_data['diffusion_coefficients'].append(fit_results['D'])
            analyzed_data['localization_errors'].append(fit_results['sigma_loc'])
            analyzed_data['D_CI_low'].append(fit_results['D_CI_low'])
            analyzed_data['D_CI_high'].append(fit_results['D_CI_high'])
            analyzed_data['n_fit_points'].append(fit_results['n_fit_points'])
            analyzed_data['max_tau_used'].append(fit_results['max_tau_used'])
            analyzed_data['radius_of_gyration'].append(rg)
            analyzed_data['track_lengths'].append(trajectory_length)
            analyzed_data['r_squared_values'].append(fit_results['r_squared'])

    if n_warnings > 0:
        print(f"  ⚠️  {n_warnings} trajectories generated warnings (too short for optimal fitting)")

    return analyzed_data

def create_diagnostic_plots(analyzed_data, output_path, filename):
    """
    Create diagnostic plots showing tau filtering effects.

    Args:
        analyzed_data: Dictionary containing analyzed trajectory data
        output_path: Directory to save plots
        filename: Base filename for plots
    """
    if not analyzed_data['trajectories']:
        print("No trajectories to plot")
        return

    # Figure 1: Number of fit points vs trajectory length
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    ax = axs[0, 0]
    if analyzed_data['track_lengths']:
        ax.scatter(analyzed_data['track_lengths'], analyzed_data['n_fit_points'], alpha=0.5)
        # Add theoretical line (n_fit = track_length * TAU_FRACTION)
        x_theory = np.array([min(analyzed_data['track_lengths']), max(analyzed_data['track_lengths'])])
        y_theory = x_theory * TAU_FRACTION
        ax.plot(x_theory, y_theory, 'r--', linewidth=2, label=f'Theory: n_fit = N × {TAU_FRACTION}')
        ax.set_xlabel('Trajectory length (frames)')
        ax.set_ylabel('Number of points used for fitting')
        ax.set_title('Tau Filtering: Fit Points vs Trajectory Length')
        ax.legend()
        ax.grid(True)

    # Plot 2: Max tau used vs trajectory length
    ax = axs[0, 1]
    if analyzed_data['max_tau_used']:
        ax.scatter(analyzed_data['track_lengths'], analyzed_data['max_tau_used'], alpha=0.5)
        ax.set_xlabel('Trajectory length (frames)')
        ax.set_ylabel('Max tau used (seconds)')
        ax.set_title('Max Tau Used vs Trajectory Length')
        ax.grid(True)

    # Plot 3: D histogram
    ax = axs[1, 0]
    if analyzed_data['diffusion_coefficients']:
        ax.hist(analyzed_data['diffusion_coefficients'], bins=20, alpha=0.7)
        median_D = np.median(analyzed_data['diffusion_coefficients'])
        ax.axvline(median_D, color='r', linestyle='--', linewidth=2,
                   label=f'Median: {median_D:.4f} µm²/s')
        ax.set_xlabel('Diffusion coefficient (µm²/s)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of D (with tau filtering)')
        ax.legend()
        ax.grid(True)

    # Plot 4: σ_loc histogram
    ax = axs[1, 1]
    if analyzed_data['localization_errors']:
        sigma_loc_nm = [s*1000 for s in analyzed_data['localization_errors'] if not np.isnan(s)]
        if sigma_loc_nm:
            ax.hist(sigma_loc_nm, bins=20, alpha=0.7)
            median_sigma = np.median(sigma_loc_nm)
            ax.axvline(median_sigma, color='r', linestyle='--', linewidth=2,
                       label=f'Median: {median_sigma:.1f} nm')
            ax.set_xlabel('Localization error σ_loc (nm)')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of σ_loc')
            ax.legend()
            ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{filename}_tau_filtering_summary.png"), dpi=300)
    plt.close()

def main():
    """Main function to analyze with tau filtering."""
    print("="*60)
    print("Diffusion Analyzer V3 - Tau-Aware Filtering")
    print("="*60)
    print(f"Tau fraction: {TAU_FRACTION} (using first {TAU_FRACTION*100:.0f}% of trajectory)")
    print(f"Minimum fit points: {MIN_FIT_POINTS}")

    # Ask for input directory
    input_dir = input("\nEnter the directory containing processed trajectory files (press Enter for processed_trajectories): ")

    if input_dir == "":
        input_dir = os.path.join(os.getcwd(), "processed_trajectories")

    if not os.path.isdir(input_dir):
        print(f"Directory {input_dir} does not exist")
        return

    # Get list of processed files
    file_paths = glob.glob(os.path.join(input_dir, "tracked_*.pkl"))

    if not file_paths:
        print(f"No processed trajectory files found in {input_dir}")
        return

    print(f"Found {len(file_paths)} files to analyze")

    # Create output directory
    output_dir = os.path.join(os.path.dirname(input_dir), "analyzed_trajectories_v3")
    os.makedirs(output_dir, exist_ok=True)

    # Summary data
    summary_data = []

    # Process each file
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]

        processed_data = load_processed_data(file_path)

        if processed_data is None:
            print(f"Skipping {filename} due to loading errors")
            continue

        print(f"\nAnalyzing {filename} with {len(processed_data['trajectories'])} trajectories")

        # Analyze with tau filtering
        analyzed_data = analyze_trajectories(processed_data)

        # Print summary
        if analyzed_data['diffusion_coefficients']:
            median_D = np.median(analyzed_data['diffusion_coefficients'])
            mean_fit_points = np.mean(analyzed_data['n_fit_points'])
            median_fit_points = np.median(analyzed_data['n_fit_points'])

            sigma_loc_nm = [s*1000 for s in analyzed_data['localization_errors'] if not np.isnan(s)]
            median_sigma = np.median(sigma_loc_nm) if sigma_loc_nm else np.nan

            print(f"  Median D: {median_D:.6f} µm²/s")
            print(f"  Mean fit points: {mean_fit_points:.1f}")
            print(f"  Median fit points: {median_fit_points:.0f}")
            print(f"  Median σ_loc: {median_sigma:.1f} nm")
            print(f"  Valid trajectories: {len(analyzed_data['diffusion_coefficients'])}")

            summary_data.append({
                'filename': base_name,
                'median_D': median_D,
                'mean_fit_points': mean_fit_points,
                'median_fit_points': median_fit_points,
                'median_sigma_loc_nm': median_sigma,
                'n_trajectories': len(analyzed_data['diffusion_coefficients']),
                'n_warnings': len(analyzed_data['warnings'])
            })

        # Save analyzed data
        output_file = os.path.join(output_dir, f"analyzed_{base_name}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(analyzed_data, f)

        # Create plots
        create_diagnostic_plots(analyzed_data, output_dir, base_name)

    # Save summary CSV
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(output_dir, "tau_filtering_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nSummary saved to {summary_csv}")

    print(f"\nAll files analyzed. Results saved in {output_dir}")

if __name__ == "__main__":
    main()
