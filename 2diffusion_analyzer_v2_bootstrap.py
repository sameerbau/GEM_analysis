#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diffusion_analyzer_v2_bootstrap.py

VERSION 2: Adds bootstrap confidence intervals for D and σ_loc

This script analyzes processed trajectory data using bootstrap resampling
to calculate robust confidence intervals for diffusion coefficients and
localization error.

NEW IN V2 (builds on V1):
- Bootstrap resampling (1000 iterations) for each trajectory
- 95% confidence intervals for D and σ_loc
- More robust error estimates than curve_fit covariance
- Following Carlini et al. methodology from the review

Input:
- Processed trajectory data (.pkl files) from trajectory_processor.py

Output:
- Analyzed trajectory data with bootstrap CI saved as .pkl files
- Diagnostic plots showing confidence intervals
- CSV with bootstrap statistics

Usage:
python diffusion_analyzer_v2_bootstrap.py
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

# Global parameters that can be modified
# =====================================
# Time step in seconds (default: 0.05s = 50ms)
DT = 0.1
# Conversion factor from pixels to μm (default: 1.0 for TrackMate which outputs in μm)
CONVERSION = 0.094
# Maximum number of points to use for linear fitting of MSD curves
MAX_POINTS_FOR_FIT = 11
# Fraction of the MSD curve to use for linear fit (0.2 = first 20%)
MSD_FIT_FRACTION = 0.8
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
    For 2D diffusion, the factor is 4 (2*dimensions)
    The offset = 4*σ²_loc where σ_loc is the localization error

    Args:
        t: Time lag
        D: Diffusion coefficient
        offset: Y-intercept = 4*σ²_loc (localization error contribution)

    Returns:
        MSD values according to the model
    """
    return 4 * D * t + offset

def bootstrap_fit_msd(time_data, msd_data, n_bootstrap=N_BOOTSTRAP):
    """
    Fit MSD curve using bootstrap resampling to get robust confidence intervals.

    Theory:
    - Resample (tau, MSD) pairs with replacement
    - Fit linear model to each bootstrap sample
    - Calculate percentile-based confidence intervals

    Args:
        time_data: Array of time lag values
        msd_data: Array of MSD values
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dictionary with fitting parameters and bootstrap confidence intervals
    """
    # Determine how many points to use for the fit
    fit_length = min(
        int(len(time_data) * MSD_FIT_FRACTION),
        MAX_POINTS_FOR_FIT,
        len(time_data)
    )

    # Ensure we have at least 3 points for fitting
    fit_length = max(fit_length, min(3, len(time_data)))

    # Extract data for fitting
    t_fit = time_data[:fit_length]
    msd_fit = msd_data[:fit_length]

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
            'fit_values': np.nan
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
                # Skip failed fits
                continue

        # Calculate 95% confidence intervals from bootstrap distributions
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
            'D_CI_low': D_CI_low,  # NEW: Bootstrap 95% CI lower bound
            'D_CI_high': D_CI_high,  # NEW: Bootstrap 95% CI upper bound
            'sigma_loc_CI_low': sigma_loc_CI_low,  # NEW: Bootstrap CI for σ_loc
            'sigma_loc_CI_high': sigma_loc_CI_high,  # NEW: Bootstrap CI for σ_loc
            'r_squared': r_squared,
            't_fit': t_fit,
            'msd_fit': msd_fit,
            'fit_values': fit_values,
            'D_bootstrap': D_bootstrap,  # Store full bootstrap distribution
            'sigma_loc_bootstrap': sigma_loc_bootstrap
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
            'fit_values': np.nan
        }

def calculate_radius_of_gyration(trajectory):
    """
    Calculate the radius of gyration for a trajectory.

    Args:
        trajectory: Dictionary containing trajectory data

    Returns:
        Radius of gyration value
    """
    # Extract coordinates
    x = trajectory['x']
    y = trajectory['y']

    # Calculate center of mass
    x_cm = np.mean(x)
    y_cm = np.mean(y)

    # Calculate squared distances from center of mass
    r2 = (x - x_cm)**2 + (y - y_cm)**2

    # Calculate radius of gyration
    rg = np.sqrt(np.mean(r2))

    return rg

def analyze_trajectories(processed_data):
    """
    Analyze processed trajectory data using bootstrap method.

    Args:
        processed_data: Dictionary containing processed trajectory data

    Returns:
        Dictionary with analysis results including bootstrap CI
    """
    analyzed_data = {
        'trajectories': [],
        'diffusion_coefficients': [],
        'localization_errors': [],
        'D_CI_low': [],  # NEW: Lower CI for D
        'D_CI_high': [],  # NEW: Upper CI for D
        'radius_of_gyration': [],
        'track_lengths': [],
        'r_squared_values': []
    }

    for i, trajectory in enumerate(processed_data['trajectories']):
        if i % 10 == 0:
            print(f"  Processing trajectory {i+1}/{len(processed_data['trajectories'])}...")

        # Get MSD data for this trajectory
        time_data = processed_data['time_data'][i]
        msd_data = processed_data['msd_data'][i]

        # Fit MSD curve with bootstrap
        fit_results = bootstrap_fit_msd(time_data, msd_data)

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
            'radius_of_gyration': rg,
            'track_length': len(trajectory['x']),
            'msd_data': msd_data,
            'time_data': time_data,
            't_fit': fit_results['t_fit'],
            'msd_fit': fit_results['msd_fit'],
            'fit_values': fit_results['fit_values'],
            'x': trajectory['x'],
            'y': trajectory['y']
        }

        analyzed_data['trajectories'].append(trajectory_analysis)

        # Also store in separate lists for easy access
        if not np.isnan(fit_results['D']):
            analyzed_data['diffusion_coefficients'].append(fit_results['D'])
            analyzed_data['localization_errors'].append(fit_results['sigma_loc'])
            analyzed_data['D_CI_low'].append(fit_results['D_CI_low'])
            analyzed_data['D_CI_high'].append(fit_results['D_CI_high'])
            analyzed_data['radius_of_gyration'].append(rg)
            analyzed_data['track_lengths'].append(len(trajectory['x']))
            analyzed_data['r_squared_values'].append(fit_results['r_squared'])

    return analyzed_data

def create_diagnostic_plots(analyzed_data, output_path, filename):
    """
    Create diagnostic plots including bootstrap confidence intervals.

    Args:
        analyzed_data: Dictionary containing the analyzed trajectory data
        output_path: Directory to save the plot
        filename: Base filename for the plot
    """
    if not analyzed_data['trajectories']:
        print("No trajectories to plot")
        return

    # Select a few trajectories for individual plots (up to 3)
    num_trajectories = min(3, len(analyzed_data['trajectories']))

    # Select trajectories with good fits (high R²) if possible
    if len(analyzed_data['trajectories']) > num_trajectories:
        r_squared_values = [traj['r_squared'] for traj in analyzed_data['trajectories']]
        best_indices = np.argsort(r_squared_values)[-num_trajectories:]
        selected_trajectories = [analyzed_data['trajectories'][i] for i in best_indices]
    else:
        selected_trajectories = analyzed_data['trajectories'][:num_trajectories]

    # Figure 1: MSD fits for selected trajectories with CI
    plt.figure(figsize=(15, 5*num_trajectories))

    for i, traj in enumerate(selected_trajectories):
        plt.subplot(num_trajectories, 1, i+1)

        # Plot full MSD curve
        plt.plot(traj['time_data'], traj['msd_data'], 'o', label='MSD data')

        # If fit is available, plot it with confidence band
        if not np.isnan(traj['D']):
            t_extended = np.linspace(0, traj['time_data'][-1], 100)

            # Main fit
            msd_extended = linear_msd(t_extended, traj['D'], traj['offset'])
            plt.plot(t_extended, msd_extended, '-',
                    label=f'D = {traj["D"]:.4f} [{traj["D_CI_low"]:.4f}, {traj["D_CI_high"]:.4f}] µm²/s')

            # Confidence bands
            msd_low = linear_msd(t_extended, traj['D_CI_low'], traj['offset'])
            msd_high = linear_msd(t_extended, traj['D_CI_high'], traj['offset'])
            plt.fill_between(t_extended, msd_low, msd_high, alpha=0.2, label='95% CI')

            # Highlight fitted region
            plt.plot(traj['t_fit'], traj['msd_fit'], 'o', color='red', label='Fitted points')

        plt.xlabel('Time lag (s)')
        plt.ylabel('MSD (µm²)')
        plt.title(f'Trajectory {int(traj["id"])} - σ_loc = {traj["sigma_loc"]*1000:.1f} nm [{traj["sigma_loc_CI_low"]*1000:.1f}, {traj["sigma_loc_CI_high"]*1000:.1f}], R² = {traj["r_squared"]:.3f}')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{filename}_msd_fits.png"), dpi=300)
    plt.close()

    # Figure 2: Summary plots with error bars
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: D with error bars (first 20 trajectories for clarity)
    ax = axs[0, 0]
    if analyzed_data['diffusion_coefficients']:
        n_show = min(20, len(analyzed_data['diffusion_coefficients']))
        x_pos = np.arange(n_show)
        D_vals = analyzed_data['diffusion_coefficients'][:n_show]
        D_low = analyzed_data['D_CI_low'][:n_show]
        D_high = analyzed_data['D_CI_high'][:n_show]

        # Calculate error bar sizes
        err_low = [D_vals[i] - D_low[i] for i in range(n_show)]
        err_high = [D_high[i] - D_vals[i] for i in range(n_show)]

        ax.errorbar(x_pos, D_vals, yerr=[err_low, err_high], fmt='o', capsize=3)
        ax.set_xlabel('Trajectory index')
        ax.set_ylabel('Diffusion coefficient (µm²/s)')
        ax.set_title(f'D with 95% Bootstrap CI (first {n_show} trajectories)')
        ax.grid(True)

    # Plot 2: Histogram of D with median and CI
    ax = axs[0, 1]
    if analyzed_data['diffusion_coefficients']:
        ax.hist(analyzed_data['diffusion_coefficients'], bins=20, alpha=0.7)
        median_D = np.median(analyzed_data['diffusion_coefficients'])
        ax.axvline(median_D, color='r', linestyle='--', linewidth=2,
                   label=f'Median: {median_D:.4f} µm²/s')
        ax.set_xlabel('Diffusion coefficient (µm²/s)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of diffusion coefficients')
        ax.legend()
        ax.grid(True)

    # Plot 3: Histogram of localization errors
    ax = axs[1, 0]
    if analyzed_data['localization_errors']:
        sigma_loc_nm = [s*1000 for s in analyzed_data['localization_errors'] if not np.isnan(s)]
        if sigma_loc_nm:
            ax.hist(sigma_loc_nm, bins=20, alpha=0.7)
            median_sigma = np.median(sigma_loc_nm)
            ax.axvline(median_sigma, color='r', linestyle='--', linewidth=2,
                       label=f'Median: {median_sigma:.1f} nm')
            ax.set_xlabel('Localization error σ_loc (nm)')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of localization errors')
            ax.legend()
            ax.grid(True)

    # Plot 4: R² distribution
    ax = axs[1, 1]
    if analyzed_data['r_squared_values']:
        ax.hist(analyzed_data['r_squared_values'], bins=20, alpha=0.7)
        median_r2 = np.median(analyzed_data['r_squared_values'])
        ax.axvline(median_r2, color='r', linestyle='--', linewidth=2,
                   label=f'Median: {median_r2:.3f}')
        ax.set_xlabel('R² value')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of fit quality (R²)')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{filename}_diffusion_summary.png"), dpi=300)
    plt.close()

def main():
    """Main function to analyze processed trajectory data with bootstrap."""
    print("="*60)
    print("Diffusion Analyzer V2 - Bootstrap Confidence Intervals")
    print("="*60)

    # Ask for input directory
    input_dir = input("Enter the directory containing processed trajectory files (press Enter for processed_trajectories): ")

    if input_dir == "":
        input_dir = os.path.join(os.getcwd(), "processed_trajectories")

    # Check if directory exists
    if not os.path.isdir(input_dir):
        print(f"Directory {input_dir} does not exist")
        return

    # Get list of processed files
    file_paths = glob.glob(os.path.join(input_dir, "tracked_*.pkl"))

    if not file_paths:
        print(f"No processed trajectory files found in {input_dir}")
        return

    print(f"Found {len(file_paths)} files to analyze")
    print(f"Bootstrap iterations per trajectory: {N_BOOTSTRAP}")

    # Create output directory
    output_dir = os.path.join(os.path.dirname(input_dir), "analyzed_trajectories_v2")
    os.makedirs(output_dir, exist_ok=True)

    # Summary data for CSV
    summary_data = []

    # Process each file
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]

        # Load processed data
        processed_data = load_processed_data(file_path)

        if processed_data is None:
            print(f"Skipping {filename} due to loading errors")
            continue

        print(f"\nAnalyzing {filename} with {len(processed_data['trajectories'])} trajectories")

        # Analyze trajectories with bootstrap
        analyzed_data = analyze_trajectories(processed_data)

        # Print summary statistics
        if analyzed_data['diffusion_coefficients']:
            median_D = np.median(analyzed_data['diffusion_coefficients'])
            D_CI_low_median = np.median(analyzed_data['D_CI_low'])
            D_CI_high_median = np.median(analyzed_data['D_CI_high'])

            sigma_loc_nm = [s*1000 for s in analyzed_data['localization_errors'] if not np.isnan(s)]
            median_sigma = np.median(sigma_loc_nm) if sigma_loc_nm else np.nan

            print(f"  Median D: {median_D:.6f} µm²/s")
            print(f"  Median D 95% CI: [{D_CI_low_median:.6f}, {D_CI_high_median:.6f}] µm²/s")
            print(f"  Median localization error: {median_sigma:.1f} nm")
            print(f"  Number of valid trajectories: {len(analyzed_data['diffusion_coefficients'])}")

            summary_data.append({
                'filename': base_name,
                'median_D': median_D,
                'median_D_CI_low': D_CI_low_median,
                'median_D_CI_high': D_CI_high_median,
                'median_sigma_loc_nm': median_sigma,
                'n_trajectories': len(analyzed_data['diffusion_coefficients'])
            })
        else:
            print("  No valid diffusion coefficients calculated")

        # Save analyzed data
        output_file = os.path.join(output_dir, f"analyzed_{base_name}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(analyzed_data, f)

        print(f"  Analyzed data saved to {output_file}")

        # Create diagnostic plots
        create_diagnostic_plots(analyzed_data, output_dir, base_name)

    # Save summary CSV
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(output_dir, "bootstrap_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nBootstrap summary statistics saved to {summary_csv}")

    print(f"\nAll files analyzed. Results saved in {output_dir}")

if __name__ == "__main__":
    main()
