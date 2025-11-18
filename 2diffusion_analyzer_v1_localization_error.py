#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diffusion_analyzer_v1_localization_error.py

VERSION 1: Adds explicit localization error (σ_loc) reporting

This script analyzes processed trajectory data to calculate diffusion coefficients
and localization error. It performs MSD curve fitting and generates diagnostic plots.

NEW IN V1:
- Explicit calculation and reporting of localization error (σ_loc) from MSD intercept
- Theory: MSD(τ) = 4Dτ + 4σ²_loc, so σ_loc = sqrt(offset/4)
- Localization error reported in nm for each trajectory
- Added to summary statistics and diagnostic plots

Input:
- Processed trajectory data (.pkl files) from trajectory_processor.py

Output:
- Analyzed trajectory data with diffusion coefficients and σ_loc saved as .pkl files
- Diagnostic plots showing MSD fitting and localization error
- CSV with localization error statistics

Usage:
python diffusion_analyzer_v1_localization_error.py
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

def fit_msd_curve(time_data, msd_data):
    """
    Fit a linear model to the MSD curve to extract diffusion coefficient and localization error.

    Theory:
    MSD(τ) = 4Dτ + 4σ²_loc
    - Slope = 4D → D = slope/4
    - Intercept = 4σ²_loc → σ_loc = sqrt(intercept/4)

    Args:
        time_data: Array of time lag values
        msd_data: Array of MSD values

    Returns:
        Dictionary with fitting parameters and metrics including σ_loc
    """
    # Determine how many points to use for the fit
    fit_length = min(
        int(len(time_data) * MSD_FIT_FRACTION),  # First fraction of the curve
        MAX_POINTS_FOR_FIT,                       # Maximum number of points
        len(time_data)                            # All available points
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
            'D_err': np.nan,
            'offset_err': np.nan,
            'sigma_loc': np.nan,
            'sigma_loc_err': np.nan,
            'r_squared': np.nan,
            't_fit': t_fit,
            'msd_fit': msd_fit,
            'fit_values': np.nan
        }

    try:
        # Fit the linear model
        popt, pcov = curve_fit(linear_msd, t_fit, msd_fit)
        D, offset = popt
        D_err, offset_err = np.sqrt(np.diag(pcov))

        # Calculate localization error from offset
        # offset = 4*σ²_loc → σ_loc = sqrt(offset/4)
        if offset > 0:
            sigma_loc = np.sqrt(offset / 4.0)
            # Error propagation: if offset = 4*σ², then dσ = (1/8σ)*d(offset)
            sigma_loc_err = offset_err / (8.0 * sigma_loc) if sigma_loc > 0 else np.nan
        else:
            sigma_loc = 0.0
            sigma_loc_err = np.nan

        # Calculate fit quality (R²)
        fit_values = linear_msd(t_fit, D, offset)
        residuals = msd_fit - fit_values
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((msd_fit - np.mean(msd_fit))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            'D': D,
            'offset': offset,
            'D_err': D_err,
            'offset_err': offset_err,
            'sigma_loc': sigma_loc,  # NEW: Localization error in μm
            'sigma_loc_err': sigma_loc_err,  # NEW: Error in σ_loc
            'r_squared': r_squared,
            't_fit': t_fit,
            'msd_fit': msd_fit,
            'fit_values': fit_values
        }
    except Exception as e:
        print(f"Error during MSD fitting: {e}")
        return {
            'D': np.nan,
            'offset': np.nan,
            'D_err': np.nan,
            'offset_err': np.nan,
            'sigma_loc': np.nan,
            'sigma_loc_err': np.nan,
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
    Analyze processed trajectory data to extract diffusion coefficients and localization error.

    Args:
        processed_data: Dictionary containing processed trajectory data

    Returns:
        Dictionary with analysis results
    """
    analyzed_data = {
        'trajectories': [],
        'diffusion_coefficients': [],
        'localization_errors': [],  # NEW: List of σ_loc values
        'radius_of_gyration': [],
        'track_lengths': [],
        'r_squared_values': []
    }

    for i, trajectory in enumerate(processed_data['trajectories']):
        # Get MSD data for this trajectory
        time_data = processed_data['time_data'][i]
        msd_data = processed_data['msd_data'][i]

        # Fit MSD curve
        fit_results = fit_msd_curve(time_data, msd_data)

        # Calculate radius of gyration
        rg = calculate_radius_of_gyration(trajectory)

        # Store analysis results
        trajectory_analysis = {
            'id': trajectory['id'],
            'D': fit_results['D'],
            'D_err': fit_results['D_err'],
            'offset': fit_results['offset'],
            'sigma_loc': fit_results['sigma_loc'],  # NEW: Localization error
            'sigma_loc_err': fit_results['sigma_loc_err'],  # NEW: Error in σ_loc
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
            analyzed_data['localization_errors'].append(fit_results['sigma_loc'])  # NEW
            analyzed_data['radius_of_gyration'].append(rg)
            analyzed_data['track_lengths'].append(len(trajectory['x']))
            analyzed_data['r_squared_values'].append(fit_results['r_squared'])

    return analyzed_data

def create_diagnostic_plots(analyzed_data, output_path, filename):
    """
    Create diagnostic plots for the analyzed trajectory data.

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
        # Get R² values for all trajectories
        r_squared_values = [traj['r_squared'] for traj in analyzed_data['trajectories']]

        # Select trajectories with best R² values
        best_indices = np.argsort(r_squared_values)[-num_trajectories:]
        selected_trajectories = [analyzed_data['trajectories'][i] for i in best_indices]
    else:
        selected_trajectories = analyzed_data['trajectories'][:num_trajectories]

    # Figure 1: MSD fits for selected trajectories
    plt.figure(figsize=(15, 5*num_trajectories))

    for i, traj in enumerate(selected_trajectories):
        plt.subplot(num_trajectories, 1, i+1)

        # Plot full MSD curve
        plt.plot(traj['time_data'], traj['msd_data'], 'o', label='MSD data')

        # If fit is available, plot it
        if not np.isnan(traj['D']):
            # Plot the fit line
            t_extended = np.linspace(0, traj['time_data'][-1], 100)
            msd_extended = linear_msd(t_extended, traj['D'], traj['offset'])
            plt.plot(t_extended, msd_extended, '--',
                    label=f'Fit: D = {traj["D"]:.4f} µm²/s, σ_loc = {traj["sigma_loc"]*1000:.1f} nm')

            # Highlight fitted region
            plt.plot(traj['t_fit'], traj['msd_fit'], 'o', color='red', label='Fitted points')

        plt.xlabel('Time lag (s)')
        plt.ylabel('MSD (µm²)')
        plt.title(f'Trajectory {int(traj["id"])} - D = {traj["D"]:.4f} µm²/s, σ_loc = {traj["sigma_loc"]*1000:.1f} nm, R² = {traj["r_squared"]:.3f}')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{filename}_msd_fits.png"), dpi=300)
    plt.close()

    # Figure 2: Diffusion and localization error summary plots
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))

    # Plot 1: Histogram of diffusion coefficients
    ax = axs[0, 0]
    if analyzed_data['diffusion_coefficients']:
        ax.hist(analyzed_data['diffusion_coefficients'], bins=20)
        ax.axvline(np.mean(analyzed_data['diffusion_coefficients']), color='r', linestyle='--',
                   label=f'Mean: {np.mean(analyzed_data["diffusion_coefficients"]):.4f} µm²/s')
        ax.axvline(np.median(analyzed_data['diffusion_coefficients']), color='g', linestyle='--',
                    label=f'Median: {np.median(analyzed_data["diffusion_coefficients"]):.4f} µm²/s')
    ax.set_xlabel('Diffusion coefficient (µm²/s)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of diffusion coefficients')
    ax.legend()
    ax.grid(True)

    # Plot 2: NEW - Histogram of localization errors
    ax = axs[0, 1]
    if analyzed_data['localization_errors']:
        # Convert to nm for plotting
        sigma_loc_nm = [s*1000 for s in analyzed_data['localization_errors'] if not np.isnan(s)]
        if sigma_loc_nm:
            ax.hist(sigma_loc_nm, bins=20)
            ax.axvline(np.mean(sigma_loc_nm), color='r', linestyle='--',
                       label=f'Mean: {np.mean(sigma_loc_nm):.1f} nm')
            ax.axvline(np.median(sigma_loc_nm), color='g', linestyle='--',
                        label=f'Median: {np.median(sigma_loc_nm):.1f} nm')
    ax.set_xlabel('Localization error σ_loc (nm)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of localization errors')
    ax.legend()
    ax.grid(True)

    # Plot 3: Histogram of radius of gyration
    ax = axs[0, 2]
    if analyzed_data['radius_of_gyration']:
        ax.hist(analyzed_data['radius_of_gyration'], bins=20)
        ax.axvline(np.mean(analyzed_data['radius_of_gyration']), color='r', linestyle='--',
                   label=f'Mean: {np.mean(analyzed_data["radius_of_gyration"]):.4f} µm')
        ax.axvline(np.median(analyzed_data['radius_of_gyration']), color='g', linestyle='--',
                    label=f'Median: {np.median(analyzed_data["radius_of_gyration"]):.4f} µm')
    ax.set_xlabel('Radius of gyration (µm)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of radius of gyration')
    ax.legend()
    ax.grid(True)

    # Plot 4: D vs track length
    ax = axs[1, 0]
    if analyzed_data['diffusion_coefficients']:
        ax.scatter(analyzed_data['track_lengths'], analyzed_data['diffusion_coefficients'])

        # Add trend line if enough points
        if len(analyzed_data['track_lengths']) > 2:
            z = np.polyfit(analyzed_data['track_lengths'], analyzed_data['diffusion_coefficients'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(analyzed_data['track_lengths']), max(analyzed_data['track_lengths']), 100)
            ax.plot(x_trend, p(x_trend), "r--", label=f'Trend: y = {z[0]:.2e}x + {z[1]:.2e}')

    ax.set_xlabel('Track length (frames)')
    ax.set_ylabel('Diffusion coefficient (µm²/s)')
    ax.set_title('Diffusion coefficient vs track length')
    ax.legend()
    ax.grid(True)

    # Plot 5: NEW - D vs σ_loc (should be uncorrelated if properly measured)
    ax = axs[1, 1]
    if analyzed_data['diffusion_coefficients'] and analyzed_data['localization_errors']:
        sigma_loc_nm = [s*1000 for s in analyzed_data['localization_errors'] if not np.isnan(s)]
        if sigma_loc_nm and len(sigma_loc_nm) == len(analyzed_data['diffusion_coefficients']):
            ax.scatter(sigma_loc_nm, analyzed_data['diffusion_coefficients'])
            ax.set_xlabel('Localization error σ_loc (nm)')
            ax.set_ylabel('Diffusion coefficient (µm²/s)')
            ax.set_title('D vs σ_loc (should be uncorrelated)')
            ax.grid(True)

    # Plot 6: D vs Rg
    ax = axs[1, 2]
    if analyzed_data['diffusion_coefficients']:
        ax.scatter(analyzed_data['radius_of_gyration'], analyzed_data['diffusion_coefficients'])

        # Add trend line if enough points
        if len(analyzed_data['radius_of_gyration']) > 2:
            z = np.polyfit(analyzed_data['radius_of_gyration'], analyzed_data['diffusion_coefficients'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(analyzed_data['radius_of_gyration']), max(analyzed_data['radius_of_gyration']), 100)
            ax.plot(x_trend, p(x_trend), "r--", label=f'Trend: y = {z[0]:.2e}x + {z[1]:.2e}')

    ax.set_xlabel('Radius of gyration (µm)')
    ax.set_ylabel('Diffusion coefficient (µm²/s)')
    ax.set_title('Diffusion coefficient vs radius of gyration')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{filename}_diffusion_summary.png"), dpi=300)
    plt.close()

def main():
    """Main function to analyze processed trajectory data."""
    # Ask for input directory
    input_dir = input("Enter the directory containing processed trajectory files (press Enter for processed_trajectories): ")

    if input_dir == "":
        # Default to the processed_trajectories directory in the current folder
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

    # Create output directory for analyzed files
    output_dir = os.path.join(os.path.dirname(input_dir), "analyzed_trajectories_v1")
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

        print(f"Analyzing {filename} with {len(processed_data['trajectories'])} trajectories")

        # Analyze trajectories
        analyzed_data = analyze_trajectories(processed_data)

        # Print summary statistics
        if analyzed_data['diffusion_coefficients']:
            mean_D = np.mean(analyzed_data['diffusion_coefficients'])
            median_D = np.median(analyzed_data['diffusion_coefficients'])
            std_D = np.std(analyzed_data['diffusion_coefficients'])

            # NEW: Localization error statistics
            sigma_loc_nm = [s*1000 for s in analyzed_data['localization_errors'] if not np.isnan(s)]
            mean_sigma = np.mean(sigma_loc_nm) if sigma_loc_nm else np.nan
            median_sigma = np.median(sigma_loc_nm) if sigma_loc_nm else np.nan

            print(f"  Mean diffusion coefficient: {mean_D:.6f} µm²/s")
            print(f"  Median diffusion coefficient: {median_D:.6f} µm²/s")
            print(f"  Standard deviation: {std_D:.6f} µm²/s")
            print(f"  Mean localization error: {mean_sigma:.1f} nm")  # NEW
            print(f"  Median localization error: {median_sigma:.1f} nm")  # NEW
            print(f"  Number of valid trajectories: {len(analyzed_data['diffusion_coefficients'])}")

            # Store for summary CSV
            summary_data.append({
                'filename': base_name,
                'mean_D': mean_D,
                'median_D': median_D,
                'std_D': std_D,
                'mean_sigma_loc_nm': mean_sigma,
                'median_sigma_loc_nm': median_sigma,
                'n_trajectories': len(analyzed_data['diffusion_coefficients'])
            })
        else:
            print("  No valid diffusion coefficients calculated")

        # Save analyzed data
        output_file = os.path.join(output_dir, f"analyzed_{base_name}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(analyzed_data, f)

        print(f"Analyzed data saved to {output_file}")

        # Create diagnostic plots
        create_diagnostic_plots(analyzed_data, output_dir, base_name)

    # Save summary CSV
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(output_dir, "localization_error_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nSummary statistics saved to {summary_csv}")

    print(f"All files analyzed. Results saved in {output_dir}")

if __name__ == "__main__":
    main()
