#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
msd_overlapping_vs_nonoverlapping_comparison.py

Compare overlapping vs non-overlapping MSD calculations

This script analyzes the same trajectory data using both overlapping and
non-overlapping methods for MSD calculation to quantify the differences
in diffusion coefficient estimates and error bars.

Theory:
-------
OVERLAPPING MSD (standard):
- Uses all possible pairs (i, i+τ) for each time lag τ
- More statistical power (more pairs)
- Measurements are correlated
- Example for τ=2: pairs (0,2), (1,3), (2,4), (3,5), ...

NON-OVERLAPPING MSD:
- Uses only non-overlapping intervals
- Independent measurements (no correlation)
- Fewer pairs → larger error bars
- Example for τ=2: pairs (0,2), (2,4), (4,6), ...

Key Questions:
1. How much does D differ between methods?
2. How do error bars compare?
3. Does localization error estimate change?
4. Which method is more robust?

From the review:
"Most papers use overlapping intervals for better statistics, but this
introduces correlation between measurements at different time lags."

Output:
- Side-by-side comparison of D values
- Error bar comparison
- Correlation analysis
- Recommendation for method choice

Usage:
python msd_overlapping_vs_nonoverlapping_comparison.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import pickle
from pathlib import Path

# Global parameters
DT = 0.1  # Time step (s)
TAU_FRACTION = 0.25  # Use first 25% of trajectory
N_BOOTSTRAP = 1000

def load_processed_data(file_path):
    """Load processed trajectory data."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_msd_overlapping(x, y, dt):
    """
    Calculate MSD using overlapping intervals (standard method).

    For each time lag τ, uses all possible pairs (i, i+τ).

    Args:
        x, y: Coordinate arrays
        dt: Time step

    Returns:
        msd: Array of MSD values
        time_lags: Array of time lag values
        n_pairs: Number of pairs used for each lag
    """
    n = len(x)
    max_lag = n - 1
    msd = np.zeros(max_lag)
    n_pairs = np.zeros(max_lag)

    # For each time lag
    for lag in range(1, max_lag + 1):
        # Use all possible pairs with this lag
        for i in range(n - lag):
            dx = x[i + lag] - x[i]
            dy = y[i + lag] - y[i]
            msd[lag - 1] += dx**2 + dy**2
            n_pairs[lag - 1] += 1

    # Average over all pairs
    msd = msd / n_pairs
    time_lags = np.arange(1, max_lag + 1) * dt

    return msd, time_lags, n_pairs

def calculate_msd_nonoverlapping(x, y, dt):
    """
    Calculate MSD using non-overlapping intervals.

    For each time lag τ, uses only non-overlapping pairs:
    (0, τ), (τ, 2τ), (2τ, 3τ), etc.

    Args:
        x, y: Coordinate arrays
        dt: Time step

    Returns:
        msd: Array of MSD values
        time_lags: Array of time lag values
        n_pairs: Number of pairs used for each lag
    """
    n = len(x)
    max_lag = n - 1
    msd = np.zeros(max_lag)
    n_pairs = np.zeros(max_lag)

    # For each time lag
    for lag in range(1, max_lag + 1):
        # Use only non-overlapping pairs
        # Start at i=0, lag, 2*lag, 3*lag, ...
        i = 0
        while i + lag < n:
            dx = x[i + lag] - x[i]
            dy = y[i + lag] - y[i]
            msd[lag - 1] += dx**2 + dy**2
            n_pairs[lag - 1] += 1
            i += lag  # Jump by lag to avoid overlap

    # Average over pairs (if any)
    valid = n_pairs > 0
    msd[valid] = msd[valid] / n_pairs[valid]
    msd[~valid] = np.nan  # Mark invalid lags

    time_lags = np.arange(1, max_lag + 1) * dt

    return msd, time_lags, n_pairs

def linear_msd(t, D, offset):
    """Linear MSD model: MSD = 4*D*t + offset"""
    return 4 * D * t + offset

def fit_msd_with_bootstrap(time_data, msd_data, max_tau_idx, n_bootstrap=N_BOOTSTRAP):
    """
    Fit MSD using linear regression with bootstrap CI.

    Args:
        time_data: Time lag array
        msd_data: MSD array
        max_tau_idx: Maximum index to use for fitting
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dictionary with D, sigma_loc, and confidence intervals
    """
    # Extract fitting region
    t_fit = time_data[:max_tau_idx]
    msd_fit = msd_data[:max_tau_idx]

    # Remove NaN values
    valid = ~np.isnan(msd_fit)
    t_fit = t_fit[valid]
    msd_fit = msd_fit[valid]

    if len(t_fit) < 3:
        return {'D': np.nan, 'sigma_loc': np.nan, 'D_CI_low': np.nan,
                'D_CI_high': np.nan, 'r_squared': np.nan, 'n_points': len(t_fit)}

    try:
        # Main fit
        popt, pcov = curve_fit(linear_msd, t_fit, msd_fit)
        D_main, offset_main = popt

        # Calculate sigma_loc
        sigma_loc_main = np.sqrt(offset_main / 4.0) if offset_main > 0 else 0.0

        # R²
        fit_values = linear_msd(t_fit, D_main, offset_main)
        ss_res = np.sum((msd_fit - fit_values)**2)
        ss_tot = np.sum((msd_fit - np.mean(msd_fit))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Bootstrap
        D_bootstrap = []
        n_points = len(t_fit)

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_points, size=n_points, replace=True)
            try:
                popt_boot, _ = curve_fit(linear_msd, t_fit[indices], msd_fit[indices])
                D_bootstrap.append(popt_boot[0])
            except:
                continue

        # CI
        if len(D_bootstrap) > 0:
            D_CI_low, D_CI_high = np.percentile(D_bootstrap, [2.5, 97.5])
        else:
            D_CI_low = D_CI_high = np.nan

        return {
            'D': D_main,
            'sigma_loc': sigma_loc_main,
            'D_CI_low': D_CI_low,
            'D_CI_high': D_CI_high,
            'r_squared': r_squared,
            'n_points': len(t_fit)
        }
    except Exception as e:
        return {'D': np.nan, 'sigma_loc': np.nan, 'D_CI_low': np.nan,
                'D_CI_high': np.nan, 'r_squared': np.nan, 'n_points': len(t_fit)}

def compare_methods_single_trajectory(trajectory, dt=DT):
    """
    Compare overlapping vs non-overlapping for a single trajectory.

    Returns:
        Dictionary with comparison results
    """
    x = trajectory['x']
    y = trajectory['y']
    traj_length = len(x)

    # Calculate max tau index (using first 25% of trajectory)
    max_tau_idx = max(int(traj_length * TAU_FRACTION), 3)

    # Method 1: Overlapping
    msd_overlap, tau_overlap, n_pairs_overlap = calculate_msd_overlapping(x, y, dt)
    fit_overlap = fit_msd_with_bootstrap(tau_overlap, msd_overlap, max_tau_idx)

    # Method 2: Non-overlapping
    msd_nonoverlap, tau_nonoverlap, n_pairs_nonoverlap = calculate_msd_nonoverlapping(x, y, dt)
    fit_nonoverlap = fit_msd_with_bootstrap(tau_nonoverlap, msd_nonoverlap, max_tau_idx)

    return {
        'trajectory_id': trajectory['id'],
        'trajectory_length': traj_length,
        'max_tau_idx': max_tau_idx,
        # Overlapping results
        'D_overlap': fit_overlap['D'],
        'D_overlap_CI_low': fit_overlap['D_CI_low'],
        'D_overlap_CI_high': fit_overlap['D_CI_high'],
        'sigma_loc_overlap': fit_overlap['sigma_loc'],
        'r_squared_overlap': fit_overlap['r_squared'],
        'n_points_overlap': fit_overlap['n_points'],
        # Non-overlapping results
        'D_nonoverlap': fit_nonoverlap['D'],
        'D_nonoverlap_CI_low': fit_nonoverlap['D_CI_low'],
        'D_nonoverlap_CI_high': fit_nonoverlap['D_CI_high'],
        'sigma_loc_nonoverlap': fit_nonoverlap['sigma_loc'],
        'r_squared_nonoverlap': fit_nonoverlap['r_squared'],
        'n_points_nonoverlap': fit_nonoverlap['n_points'],
        # Raw data for plotting
        'msd_overlap': msd_overlap[:max_tau_idx],
        'msd_nonoverlap': msd_nonoverlap[:max_tau_idx],
        'tau': tau_overlap[:max_tau_idx],
        'n_pairs_overlap': n_pairs_overlap[:max_tau_idx],
        'n_pairs_nonoverlap': n_pairs_nonoverlap[:max_tau_idx]
    }

def analyze_file(file_path):
    """Analyze all trajectories in a file."""
    processed_data = load_processed_data(file_path)
    if processed_data is None:
        return None

    results = []
    for traj in processed_data['trajectories']:
        result = compare_methods_single_trajectory(traj)
        results.append(result)

    return results

def create_comparison_plots(results, output_dir, filename):
    """Create comprehensive comparison plots."""

    # Extract valid results
    valid_results = [r for r in results if not np.isnan(r['D_overlap']) and not np.isnan(r['D_nonoverlap'])]

    if len(valid_results) == 0:
        print("No valid results to plot")
        return

    # Figure 1: D comparison scatter plot
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: D overlap vs D non-overlap
    ax = axs[0, 0]
    D_overlap = [r['D_overlap'] for r in valid_results]
    D_nonoverlap = [r['D_nonoverlap'] for r in valid_results]

    ax.scatter(D_overlap, D_nonoverlap, alpha=0.5)

    # Add y=x line
    min_d = min(min(D_overlap), min(D_nonoverlap))
    max_d = max(max(D_overlap), max(D_nonoverlap))
    ax.plot([min_d, max_d], [min_d, max_d], 'r--', linewidth=2, label='y=x')

    ax.set_xlabel('D overlapping (μm²/s)')
    ax.set_ylabel('D non-overlapping (μm²/s)')
    ax.set_title('Diffusion Coefficient Comparison')
    ax.legend()
    ax.grid(True)

    # Plot 2: Relative difference
    ax = axs[0, 1]
    rel_diff = [(r['D_nonoverlap'] - r['D_overlap']) / r['D_overlap'] * 100
                for r in valid_results]
    ax.hist(rel_diff, bins=30, alpha=0.7)
    ax.axvline(np.median(rel_diff), color='r', linestyle='--', linewidth=2,
               label=f'Median: {np.median(rel_diff):.1f}%')
    ax.set_xlabel('Relative difference (%)')
    ax.set_ylabel('Count')
    ax.set_title('(D_nonoverlap - D_overlap) / D_overlap')
    ax.legend()
    ax.grid(True)

    # Plot 3: CI width comparison
    ax = axs[0, 2]
    ci_width_overlap = [(r['D_overlap_CI_high'] - r['D_overlap_CI_low']) / r['D_overlap']
                        for r in valid_results if not np.isnan(r['D_overlap_CI_low'])]
    ci_width_nonoverlap = [(r['D_nonoverlap_CI_high'] - r['D_nonoverlap_CI_low']) / r['D_nonoverlap']
                           for r in valid_results if not np.isnan(r['D_nonoverlap_CI_low'])]

    ax.boxplot([ci_width_overlap, ci_width_nonoverlap], labels=['Overlapping', 'Non-overlapping'])
    ax.set_ylabel('Relative CI width (ΔD/D)')
    ax.set_title('Uncertainty Comparison')
    ax.grid(True)

    # Plot 4: Number of pairs comparison
    ax = axs[1, 0]
    # Average number of pairs at tau = max_tau_idx/2
    n_pairs_overlap_avg = [np.mean(r['n_pairs_overlap']) for r in valid_results]
    n_pairs_nonoverlap_avg = [np.mean(r['n_pairs_nonoverlap']) for r in valid_results]

    ax.scatter(n_pairs_overlap_avg, n_pairs_nonoverlap_avg, alpha=0.5)
    ax.set_xlabel('Avg pairs (overlapping)')
    ax.set_ylabel('Avg pairs (non-overlapping)')
    ax.set_title('Number of Displacement Pairs Used')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True)

    # Plot 5: R² comparison
    ax = axs[1, 1]
    r2_overlap = [r['r_squared_overlap'] for r in valid_results]
    r2_nonoverlap = [r['r_squared_nonoverlap'] for r in valid_results]

    ax.scatter(r2_overlap, r2_nonoverlap, alpha=0.5)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x')
    ax.set_xlabel('R² overlapping')
    ax.set_ylabel('R² non-overlapping')
    ax.set_title('Fit Quality Comparison')
    ax.legend()
    ax.grid(True)

    # Plot 6: Localization error comparison
    ax = axs[1, 2]
    sigma_overlap = [r['sigma_loc_overlap'] * 1000 for r in valid_results]  # Convert to nm
    sigma_nonoverlap = [r['sigma_loc_nonoverlap'] * 1000 for r in valid_results]

    ax.scatter(sigma_overlap, sigma_nonoverlap, alpha=0.5)
    if len(sigma_overlap) > 0:
        min_s = min(min(sigma_overlap), min(sigma_nonoverlap))
        max_s = max(max(sigma_overlap), max(sigma_nonoverlap))
        ax.plot([min_s, max_s], [min_s, max_s], 'r--', linewidth=2, label='y=x')
    ax.set_xlabel('σ_loc overlapping (nm)')
    ax.set_ylabel('σ_loc non-overlapping (nm)')
    ax.set_title('Localization Error Comparison')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_method_comparison.png"), dpi=300)
    plt.close()

    # Figure 2: Example trajectories showing MSD curves
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Select 4 representative trajectories
    selected_idx = np.linspace(0, len(valid_results)-1, min(4, len(valid_results)), dtype=int)

    for i, idx in enumerate(selected_idx):
        ax = axs[i // 2, i % 2]
        r = valid_results[idx]

        # Plot both MSD curves
        ax.plot(r['tau'], r['msd_overlap'], 'o-', label='Overlapping', alpha=0.7)
        ax.plot(r['tau'], r['msd_nonoverlap'], 's-', label='Non-overlapping', alpha=0.7)

        ax.set_xlabel('Time lag (s)')
        ax.set_ylabel('MSD (μm²)')
        ax.set_title(f"Trajectory {int(r['trajectory_id'])} (N={r['trajectory_length']})\n"
                    f"D_overlap={r['D_overlap']:.4f}, D_nonoverlap={r['D_nonoverlap']:.4f}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_msd_curves_comparison.png"), dpi=300)
    plt.close()

def main():
    """Main comparison analysis."""
    print("="*70)
    print("MSD Calculation Method Comparison: Overlapping vs Non-Overlapping")
    print("="*70)

    input_dir = input("\nEnter directory (press Enter for processed_trajectories): ")
    if input_dir == "":
        input_dir = os.path.join(os.getcwd(), "processed_trajectories")

    if not os.path.isdir(input_dir):
        print(f"Directory {input_dir} does not exist")
        return

    file_paths = glob.glob(os.path.join(input_dir, "tracked_*.pkl"))
    if not file_paths:
        print("No processed trajectory files found")
        return

    print(f"Found {len(file_paths)} files")

    # Create output directory
    output_dir = os.path.join(os.path.dirname(input_dir), "msd_method_comparison")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    # Analyze each file
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]

        print(f"\nAnalyzing {filename}...")
        results = analyze_file(file_path)

        if results is None:
            continue

        all_results.extend(results)

        # Create plots for this file
        create_comparison_plots(results, output_dir, base_name)

        # Save detailed results
        df = pd.DataFrame(results)
        csv_file = os.path.join(output_dir, f"{base_name}_comparison.csv")
        df.to_csv(csv_file, index=False)

    # Overall summary
    if all_results:
        print("\n" + "="*70)
        print("OVERALL SUMMARY")
        print("="*70)

        valid = [r for r in all_results if not np.isnan(r['D_overlap']) and not np.isnan(r['D_nonoverlap'])]

        if valid:
            D_overlap = np.array([r['D_overlap'] for r in valid])
            D_nonoverlap = np.array([r['D_nonoverlap'] for r in valid])
            rel_diff = (D_nonoverlap - D_overlap) / D_overlap * 100

            print(f"\nTotal trajectories analyzed: {len(valid)}")
            print(f"\nMedian D (overlapping): {np.median(D_overlap):.6f} μm²/s")
            print(f"Median D (non-overlapping): {np.median(D_nonoverlap):.6f} μm²/s")
            print(f"\nMedian relative difference: {np.median(rel_diff):.1f}%")
            print(f"Mean relative difference: {np.mean(rel_diff):.1f}%")
            print(f"Std relative difference: {np.std(rel_diff):.1f}%")

            # Correlation
            correlation = np.corrcoef(D_overlap, D_nonoverlap)[0, 1]
            print(f"\nCorrelation between methods: {correlation:.3f}")

            # Recommendation
            print("\n" + "-"*70)
            print("RECOMMENDATION:")
            print("-"*70)
            if abs(np.median(rel_diff)) < 10 and correlation > 0.9:
                print("✓ Both methods agree well (< 10% difference)")
                print("  → Use OVERLAPPING for better statistics and smaller error bars")
                print("  → Use NON-OVERLAPPING if independence is critical for your analysis")
            elif abs(np.median(rel_diff)) < 20:
                print("⚠ Moderate difference between methods (10-20%)")
                print("  → Consider trajectory length and noise level")
                print("  → Overlapping may be biased by correlated errors")
            else:
                print("✗ Large difference between methods (> 20%)")
                print("  → Check for systematic errors or short trajectories")
                print("  → May indicate non-diffusive motion")

    print(f"\nResults saved in {output_dir}")

if __name__ == "__main__":
    main()
