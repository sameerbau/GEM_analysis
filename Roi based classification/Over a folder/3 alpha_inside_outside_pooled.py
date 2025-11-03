#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3 alpha_inside_outside_pooled.py

This script analyzes the alpha exponent (anomalous diffusion) properties of
trajectories inside and outside ROIs by pooling data from multiple
roi_trajectory_data pkl files in a folder.

Input:
- Folder containing multiple roi_trajectory_data_*.pkl files
  (Output from 1 IJ ROI loader_within_outside_folderver.py)

Output:
- Summary statistics for pooled alpha values inside and outside ROIs
- Histogram of alpha exponents inside and outside ROIs
- CDF plot of alpha exponents inside and outside ROIs
- Box plot comparing alpha exponents inside and outside ROIs
- Violin plot comparing distributions
- Statistical tests (Mann-Whitney U test, effect size)
- CSV file with detailed statistics

Usage:
python 3 alpha_inside_outside_pooled.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
from scipy import stats

# Global parameters
# =====================================
# Time step in seconds
DT = 0.1
# Minimum trajectory length for alpha calculation
MIN_TRAJECTORY_LENGTH = 10
# Maximum lag time fraction for MSD analysis (use first 30% of trajectory)
MAX_LAG_FRACTION = 0.3
# Minimum R² for alpha fit quality
MIN_R_SQUARED = 0.6
# Alpha bounds for filtering outliers
ALPHA_MIN = 0.0
ALPHA_MAX = 2.5
# Bin size for alpha histogram
ALPHA_BIN_SIZE = 0.05
# Output subfolder name
OUTPUT_SUBFOLDER = 'pooled_alpha_inside_outside_analysis'
# Bootstrap samples for confidence intervals
N_BOOTSTRAP = 1000
# =====================================


def load_roi_data(file_path):
    """
    Load ROI-assigned trajectory data from pickle file.

    Args:
        file_path: Path to the pickle file

    Returns:
        Dictionary containing the ROI-assigned trajectory data
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded: {os.path.basename(file_path)}")
        return data
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None


def calculate_msd(x, y, dt, max_lag=None):
    """
    Calculate mean squared displacement for a trajectory.

    Args:
        x: X coordinates
        y: Y coordinates
        dt: Time step
        max_lag: Maximum lag time to calculate (in frames)

    Returns:
        Tuple of (MSD values, time lags)
    """
    n = len(x)

    # Determine maximum lag time
    if max_lag is None:
        max_lag = n - 1
    else:
        max_lag = min(max_lag, n - 1)

    # Initialize arrays
    msd = np.zeros(max_lag)

    # Calculate displacement for each time lag
    for lag in range(1, max_lag + 1):
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        dr2 = dx**2 + dy**2
        msd[lag - 1] = np.mean(dr2)

    # Create time array
    time_lags = np.arange(1, max_lag + 1) * dt

    return msd, time_lags


def fit_anomalous_diffusion(time_data, msd_data):
    """
    Fit anomalous diffusion model to MSD curve using log-log regression.

    Args:
        time_data: Array of time lag values
        msd_data: Array of MSD values

    Returns:
        Dictionary with fitting parameters and metrics
    """
    # Determine maximum lag time index based on fraction
    max_points = int(len(time_data) * MAX_LAG_FRACTION)
    max_points = min(max(max_points, 4), len(time_data))

    # Extract data for fitting
    t_fit = time_data[:max_points]
    msd_fit = msd_data[:max_points]

    # Filter out any NaN or zero values
    valid_indices = ~np.isnan(msd_fit) & (msd_fit > 0) & (t_fit > 0)
    t_fit = t_fit[valid_indices]
    msd_fit = msd_fit[valid_indices]

    # Check if we have enough points after filtering
    if len(t_fit) < 4:
        return {
            'D': np.nan,
            'alpha': np.nan,
            'r_squared': np.nan
        }

    try:
        # Linear fit on log-log scale (most stable method)
        log_t = np.log(t_fit)
        log_msd = np.log(msd_fit)

        # Linear regression: log(MSD) = log(4D) + alpha*log(t)
        slope, intercept, r_value, _, _ = stats.linregress(log_t, log_msd)

        alpha = slope
        D = np.exp(intercept) / 4  # MSD = 4*D*t^alpha
        r_squared = r_value ** 2

        return {
            'D': D,
            'alpha': alpha,
            'r_squared': r_squared
        }
    except Exception as e:
        return {
            'D': np.nan,
            'alpha': np.nan,
            'r_squared': np.nan
        }


def calculate_alpha_for_trajectories(trajectories):
    """
    Calculate alpha exponent for a list of trajectories.

    Args:
        trajectories: List of trajectory dictionaries

    Returns:
        List of valid alpha values
    """
    alpha_values = []

    for traj in trajectories:
        # Skip trajectories that are too short
        if len(traj['x']) < MIN_TRAJECTORY_LENGTH:
            continue

        # Calculate MSD
        msd, time_lags = calculate_msd(traj['x'], traj['y'], DT)

        # Fit anomalous diffusion model
        fit_results = fit_anomalous_diffusion(time_lags, msd)

        # Store results if fit was successful and meets quality criteria
        if (not np.isnan(fit_results['alpha']) and
            fit_results['r_squared'] >= MIN_R_SQUARED and
            ALPHA_MIN <= fit_results['alpha'] <= ALPHA_MAX):
            alpha_values.append(fit_results['alpha'])

    return alpha_values


def analyze_alpha_single_file(roi_data):
    """
    Extract alpha values from a single ROI data file.

    Args:
        roi_data: Dictionary containing ROI-assigned trajectory data

    Returns:
        Tuple of (inside_alpha, outside_alpha) lists
    """
    inside_alpha = []
    outside_alpha = []

    for roi_id, trajectories in roi_data['roi_trajectories'].items():
        if roi_id == 'unassigned':
            # Trajectories outside ROIs
            outside_alpha.extend(calculate_alpha_for_trajectories(trajectories))
        else:
            # Trajectories inside ROIs
            inside_alpha.extend(calculate_alpha_for_trajectories(trajectories))

    return inside_alpha, outside_alpha


def pool_alpha_data(folder_path):
    """
    Pool alpha data from all roi_trajectory_data_*.pkl files in a folder.

    Args:
        folder_path: Path to folder containing pkl files

    Returns:
        Tuple of (inside_alpha_array, outside_alpha_array) with pooled data
    """
    # Find all roi_trajectory_data pkl files
    pkl_pattern = os.path.join(folder_path, 'roi_trajectory_data_*.pkl')
    pkl_files = glob.glob(pkl_pattern)

    if not pkl_files:
        print(f"No roi_trajectory_data_*.pkl files found in {folder_path}")
        return None, None

    print(f"\nFound {len(pkl_files)} pkl files to process")
    print("="*70)

    # Lists to collect all alpha values
    all_inside_alpha = []
    all_outside_alpha = []

    # Process each file
    for pkl_file in pkl_files:
        roi_data = load_roi_data(pkl_file)

        if roi_data is not None:
            inside_alpha, outside_alpha = analyze_alpha_single_file(roi_data)
            all_inside_alpha.extend(inside_alpha)
            all_outside_alpha.extend(outside_alpha)
            print(f"    Inside: {len(inside_alpha)}, Outside: {len(outside_alpha)}")

    print("="*70)
    print(f"Total pooled - Inside: {len(all_inside_alpha)}, Outside: {len(all_outside_alpha)}")

    return np.array(all_inside_alpha), np.array(all_outside_alpha)


def calculate_bootstrap_ci(data, n_bootstrap=N_BOOTSTRAP):
    """
    Calculate bootstrap confidence interval for mean.

    Args:
        data: Array of values
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary with mean, CI, and SEM
    """
    if len(data) == 0:
        return {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'sem': np.nan}

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'sem': np.std(data) / np.sqrt(len(data)),
        'std': np.std(data)
    }


def calculate_cliffs_delta(group1, group2):
    """
    Calculate Cliff's Delta effect size (non-parametric).

    Interpretation:
        |d| < 0.147: Negligible
        |d| < 0.33:  Small
        |d| < 0.474: Medium
        |d| >= 0.474: Large

    Args:
        group1: Array of values from group 1
        group2: Array of values from group 2

    Returns:
        Cliff's Delta value
    """
    n1 = len(group1)
    n2 = len(group2)

    if n1 == 0 or n2 == 0:
        return np.nan

    # Count pairs where group1 > group2 and group1 < group2
    greater = sum(1 for val1 in group1 for val2 in group2 if val1 > val2)
    less = sum(1 for val1 in group1 for val2 in group2 if val1 < val2)

    delta = (greater - less) / (n1 * n2)
    return delta


def perform_statistical_tests(inside_alpha, outside_alpha):
    """
    Perform statistical tests comparing inside vs outside alpha values.

    Args:
        inside_alpha: Array of alpha values inside ROIs
        outside_alpha: Array of alpha values outside ROIs

    Returns:
        Dictionary with test results
    """
    results = {}

    # Mann-Whitney U test (non-parametric)
    if len(inside_alpha) > 0 and len(outside_alpha) > 0:
        statistic, p_value = stats.mannwhitneyu(inside_alpha, outside_alpha,
                                               alternative='two-sided')
        results['mann_whitney_u'] = statistic
        results['p_value'] = p_value
        results['significant'] = p_value < 0.05

        # Kolmogorov-Smirnov test
        ks_statistic, ks_p_value = stats.ks_2samp(inside_alpha, outside_alpha)
        results['ks_statistic'] = ks_statistic
        results['ks_p_value'] = ks_p_value

        # Effect size (Cliff's Delta)
        results['cliffs_delta'] = calculate_cliffs_delta(inside_alpha, outside_alpha)

        # Bootstrap confidence intervals
        results['inside_stats'] = calculate_bootstrap_ci(inside_alpha)
        results['outside_stats'] = calculate_bootstrap_ci(outside_alpha)
    else:
        results['mann_whitney_u'] = np.nan
        results['p_value'] = np.nan
        results['significant'] = False
        results['ks_statistic'] = np.nan
        results['ks_p_value'] = np.nan
        results['cliffs_delta'] = np.nan

    return results


def plot_alpha_histogram(inside_alpha, outside_alpha, output_path):
    """
    Plot histogram of alpha exponents inside and outside ROIs.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine bin range
    all_alpha = np.concatenate([inside_alpha, outside_alpha])
    bins = np.arange(max(0, np.min(all_alpha) - 0.1),
                     min(2.5, np.max(all_alpha) + 0.1),
                     ALPHA_BIN_SIZE)

    # Plot histograms
    ax.hist(inside_alpha, bins=bins, alpha=0.6, label=f'Inside ROIs (n={len(inside_alpha)})',
            color='steelblue', edgecolor='black')
    ax.hist(outside_alpha, bins=bins, alpha=0.6, label=f'Outside ROIs (n={len(outside_alpha)})',
            color='coral', edgecolor='black')

    # Add reference line at alpha = 1 (normal diffusion)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2,
               label='Normal diffusion (α=1)')

    ax.set_xlabel('Alpha Exponent (α)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Pooled Alpha Exponent Distribution: Inside vs Outside ROIs',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"  Histogram saved to {os.path.basename(output_path)}")


def plot_alpha_cdf(inside_alpha, outside_alpha, output_path):
    """
    Plot CDF of alpha exponents inside and outside ROIs.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute and plot CDF for inside ROIs
    inside_sorted = np.sort(inside_alpha)
    inside_yvals = np.arange(len(inside_sorted)) / float(len(inside_sorted))
    ax.plot(inside_sorted, inside_yvals, label=f'Inside ROIs (n={len(inside_alpha)})',
            linewidth=2.5, color='steelblue')

    # Compute and plot CDF for outside ROIs
    outside_sorted = np.sort(outside_alpha)
    outside_yvals = np.arange(len(outside_sorted)) / float(len(outside_sorted))
    ax.plot(outside_sorted, outside_yvals, label=f'Outside ROIs (n={len(outside_alpha)})',
            linewidth=2.5, color='coral')

    # Add reference line at alpha = 1
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2,
               label='Normal diffusion (α=1)')

    ax.set_xlabel('Alpha Exponent (α)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title('Pooled Alpha Exponent CDF: Inside vs Outside ROIs',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"  CDF plot saved to {os.path.basename(output_path)}")


def plot_alpha_boxplot(inside_alpha, outside_alpha, stats_results, output_path):
    """
    Plot box plot comparing alpha exponents inside and outside ROIs.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Create box plot
    bp = ax.boxplot([inside_alpha, outside_alpha],
                    labels=['Inside ROIs', 'Outside ROIs'],
                    patch_artist=True, showfliers=True, widths=0.6)

    # Color boxes
    colors = ['steelblue', 'coral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add reference line at alpha = 1
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2,
               label='Normal diffusion (α=1)')

    # Add sample sizes and means as text
    inside_stats = stats_results.get('inside_stats', {})
    outside_stats = stats_results.get('outside_stats', {})

    if inside_stats and 'mean' in inside_stats:
        ax.text(1, np.mean(inside_alpha), f"μ={inside_stats['mean']:.3f}",
               ha='right', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if outside_stats and 'mean' in outside_stats:
        ax.text(2, np.mean(outside_alpha), f"μ={outside_stats['mean']:.3f}",
               ha='left', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add statistical significance
    if stats_results.get('significant', False):
        y_max = max(np.max(inside_alpha), np.max(outside_alpha))
        y_pos = y_max + 0.1
        ax.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=2)
        p_val = stats_results['p_value']
        if p_val < 0.001:
            sig_text = '***'
        elif p_val < 0.01:
            sig_text = '**'
        else:
            sig_text = '*'
        ax.text(1.5, y_pos, sig_text, ha='center', va='bottom',
               fontsize=16, fontweight='bold')

    ax.set_ylabel('Alpha Exponent (α)', fontsize=12, fontweight='bold')
    ax.set_title('Pooled Alpha Exponent Comparison: Inside vs Outside ROIs',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"  Box plot saved to {os.path.basename(output_path)}")


def plot_alpha_violin(inside_alpha, outside_alpha, output_path):
    """
    Plot violin plot comparing alpha exponents inside and outside ROIs.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for violin plot
    plot_data = []
    for val in inside_alpha:
        plot_data.append({'Location': 'Inside ROIs', 'Alpha': val})
    for val in outside_alpha:
        plot_data.append({'Location': 'Outside ROIs', 'Alpha': val})
    plot_df = pd.DataFrame(plot_data)

    # Create violin plot
    sns.violinplot(data=plot_df, x='Location', y='Alpha', ax=ax,
                  palette=['steelblue', 'coral'], inner='box')

    # Add reference line at alpha = 1
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2,
               label='Normal diffusion (α=1)')

    ax.set_ylabel('Alpha Exponent (α)', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_title('Pooled Alpha Exponent Distribution: Inside vs Outside ROIs',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"  Violin plot saved to {os.path.basename(output_path)}")


def export_summary_to_csv(inside_alpha, outside_alpha, stats_results, output_path):
    """
    Export summary statistics to CSV file.
    """
    # Prepare summary data
    summary_data = []

    # Inside ROIs
    inside_stats = stats_results.get('inside_stats', {})
    summary_data.append({
        'Location': 'Inside ROIs',
        'N_trajectories': len(inside_alpha),
        'Mean': inside_stats.get('mean', np.mean(inside_alpha) if len(inside_alpha) > 0 else np.nan),
        'Median': inside_stats.get('median', np.median(inside_alpha) if len(inside_alpha) > 0 else np.nan),
        'Std': inside_stats.get('std', np.std(inside_alpha) if len(inside_alpha) > 0 else np.nan),
        'SEM': inside_stats.get('sem', np.nan),
        'CI_lower': inside_stats.get('ci_lower', np.nan),
        'CI_upper': inside_stats.get('ci_upper', np.nan),
        'Min': np.min(inside_alpha) if len(inside_alpha) > 0 else np.nan,
        'Max': np.max(inside_alpha) if len(inside_alpha) > 0 else np.nan,
        'N_subdiffusion': sum(1 for a in inside_alpha if a < 0.9) if len(inside_alpha) > 0 else 0,
        'N_normal': sum(1 for a in inside_alpha if 0.9 <= a <= 1.1) if len(inside_alpha) > 0 else 0,
        'N_superdiffusion': sum(1 for a in inside_alpha if a > 1.1) if len(inside_alpha) > 0 else 0
    })

    # Outside ROIs
    outside_stats = stats_results.get('outside_stats', {})
    summary_data.append({
        'Location': 'Outside ROIs',
        'N_trajectories': len(outside_alpha),
        'Mean': outside_stats.get('mean', np.mean(outside_alpha) if len(outside_alpha) > 0 else np.nan),
        'Median': outside_stats.get('median', np.median(outside_alpha) if len(outside_alpha) > 0 else np.nan),
        'Std': outside_stats.get('std', np.std(outside_alpha) if len(outside_alpha) > 0 else np.nan),
        'SEM': outside_stats.get('sem', np.nan),
        'CI_lower': outside_stats.get('ci_lower', np.nan),
        'CI_upper': outside_stats.get('ci_upper', np.nan),
        'Min': np.min(outside_alpha) if len(outside_alpha) > 0 else np.nan,
        'Max': np.max(outside_alpha) if len(outside_alpha) > 0 else np.nan,
        'N_subdiffusion': sum(1 for a in outside_alpha if a < 0.9) if len(outside_alpha) > 0 else 0,
        'N_normal': sum(1 for a in outside_alpha if 0.9 <= a <= 1.1) if len(outside_alpha) > 0 else 0,
        'N_superdiffusion': sum(1 for a in outside_alpha if a > 1.1) if len(outside_alpha) > 0 else 0
    })

    # Statistical tests
    stats_data = [{
        'Test': 'Mann-Whitney U',
        'Statistic': stats_results.get('mann_whitney_u', np.nan),
        'P_value': stats_results.get('p_value', np.nan),
        'Significant': stats_results.get('significant', False)
    }, {
        'Test': 'Kolmogorov-Smirnov',
        'Statistic': stats_results.get('ks_statistic', np.nan),
        'P_value': stats_results.get('ks_p_value', np.nan),
        'Significant': stats_results.get('ks_p_value', 1.0) < 0.05
    }, {
        'Test': 'Cliffs Delta (effect size)',
        'Statistic': stats_results.get('cliffs_delta', np.nan),
        'P_value': np.nan,
        'Significant': np.nan
    }]

    # Export to CSV
    df_summary = pd.DataFrame(summary_data)
    df_stats = pd.DataFrame(stats_data)

    # Save to single CSV with separate sheets (if using Excel) or two files
    summary_csv = output_path.replace('.csv', '_summary.csv')
    stats_csv = output_path.replace('.csv', '_statistics.csv')

    df_summary.to_csv(summary_csv, index=False)
    df_stats.to_csv(stats_csv, index=False)

    print(f"  Summary exported to {os.path.basename(summary_csv)}")
    print(f"  Statistics exported to {os.path.basename(stats_csv)}")


def print_alpha_summary(inside_alpha, outside_alpha, stats_results):
    """
    Print summary statistics for pooled alpha exponents.
    """
    print("\n" + "="*70)
    print("POOLED ALPHA EXPONENT SUMMARY")
    print("="*70)

    print(f"\nInside ROIs (n={len(inside_alpha)}):")
    if len(inside_alpha) > 0:
        inside_stats = stats_results.get('inside_stats', {})
        mean = inside_stats.get('mean', np.mean(inside_alpha))
        median = inside_stats.get('median', np.median(inside_alpha))
        std = inside_stats.get('std', np.std(inside_alpha))
        sem = inside_stats.get('sem', std/np.sqrt(len(inside_alpha)))
        ci_low = inside_stats.get('ci_lower', np.nan)
        ci_high = inside_stats.get('ci_upper', np.nan)

        print(f"  Mean:   {mean:.3f}")
        print(f"  Median: {median:.3f}")
        print(f"  Std:    {std:.3f}")
        print(f"  SEM:    {sem:.3f}")
        print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"  Min:    {np.min(inside_alpha):.3f}")
        print(f"  Max:    {np.max(inside_alpha):.3f}")

        # Classify diffusion types
        sub_diff = sum(1 for a in inside_alpha if a < 0.9)
        normal = sum(1 for a in inside_alpha if 0.9 <= a <= 1.1)
        super_diff = sum(1 for a in inside_alpha if a > 1.1)
        print(f"\n  Diffusion types:")
        print(f"    Sub-diffusion (α<0.9):    {sub_diff} ({sub_diff/len(inside_alpha)*100:.1f}%)")
        print(f"    Normal (0.9≤α≤1.1):       {normal} ({normal/len(inside_alpha)*100:.1f}%)")
        print(f"    Super-diffusion (α>1.1):  {super_diff} ({super_diff/len(inside_alpha)*100:.1f}%)")

    print(f"\nOutside ROIs (n={len(outside_alpha)}):")
    if len(outside_alpha) > 0:
        outside_stats = stats_results.get('outside_stats', {})
        mean = outside_stats.get('mean', np.mean(outside_alpha))
        median = outside_stats.get('median', np.median(outside_alpha))
        std = outside_stats.get('std', np.std(outside_alpha))
        sem = outside_stats.get('sem', std/np.sqrt(len(outside_alpha)))
        ci_low = outside_stats.get('ci_lower', np.nan)
        ci_high = outside_stats.get('ci_upper', np.nan)

        print(f"  Mean:   {mean:.3f}")
        print(f"  Median: {median:.3f}")
        print(f"  Std:    {std:.3f}")
        print(f"  SEM:    {sem:.3f}")
        print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"  Min:    {np.min(outside_alpha):.3f}")
        print(f"  Max:    {np.max(outside_alpha):.3f}")

        # Classify diffusion types
        sub_diff = sum(1 for a in outside_alpha if a < 0.9)
        normal = sum(1 for a in outside_alpha if 0.9 <= a <= 1.1)
        super_diff = sum(1 for a in outside_alpha if a > 1.1)
        print(f"\n  Diffusion types:")
        print(f"    Sub-diffusion (α<0.9):    {sub_diff} ({sub_diff/len(outside_alpha)*100:.1f}%)")
        print(f"    Normal (0.9≤α≤1.1):       {normal} ({normal/len(outside_alpha)*100:.1f}%)")
        print(f"    Super-diffusion (α>1.1):  {super_diff} ({super_diff/len(outside_alpha)*100:.1f}%)")

    # Statistical comparison
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON")
    print("="*70)

    if not np.isnan(stats_results.get('p_value', np.nan)):
        print(f"\nMann-Whitney U test:")
        print(f"  U statistic: {stats_results['mann_whitney_u']:.2f}")
        print(f"  p-value:     {stats_results['p_value']:.6f}")
        if stats_results['significant']:
            print(f"  Result:      *** SIGNIFICANT (p < 0.05) ***")
        else:
            print(f"  Result:      Not significant (p ≥ 0.05)")

        print(f"\nKolmogorov-Smirnov test:")
        print(f"  KS statistic: {stats_results['ks_statistic']:.4f}")
        print(f"  p-value:      {stats_results['ks_p_value']:.6f}")

        print(f"\nEffect size (Cliff's Delta):")
        delta = stats_results['cliffs_delta']
        print(f"  δ = {delta:.3f}")

        # Interpret effect size
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            interpretation = "NEGLIGIBLE"
        elif abs_delta < 0.33:
            interpretation = "SMALL"
        elif abs_delta < 0.474:
            interpretation = "MEDIUM"
        else:
            interpretation = "LARGE"
        print(f"  Interpretation: {interpretation}")

        if len(inside_alpha) > 0 and len(outside_alpha) > 0:
            inside_mean = stats_results['inside_stats']['mean']
            outside_mean = stats_results['outside_stats']['mean']
            diff_mean = inside_mean - outside_mean
            pct_diff = (diff_mean / outside_mean) * 100 if outside_mean != 0 else 0

            print(f"\n  Difference in means: {diff_mean:+.3f} ({pct_diff:+.1f}%)")

            if diff_mean > 0.05:
                print(f"  → Trajectories inside ROIs show HIGHER alpha values")
                print(f"    (more super-diffusive behavior)")
            elif diff_mean < -0.05:
                print(f"  → Trajectories outside ROIs show HIGHER alpha values")
                print(f"    (more super-diffusive behavior)")
            else:
                print(f"  → No substantial difference in alpha values")

    print("="*70)


def main():
    """
    Main function to analyze pooled alpha properties inside vs outside ROIs.
    """
    print("\n" + "="*70)
    print("POOLED ALPHA EXPONENT ANALYZER: Inside vs Outside ROIs")
    print("="*70)

    # Ask for input folder
    folder_path = input("\nEnter path to folder containing roi_trajectory_data pkl files: ")

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        return

    # Pool alpha data from all files
    print(f"\nCalculating alpha exponents for trajectories...")
    print(f"Parameters:")
    print(f"  Minimum trajectory length: {MIN_TRAJECTORY_LENGTH} frames")
    print(f"  Maximum lag fraction: {MAX_LAG_FRACTION}")
    print(f"  Minimum R²: {MIN_R_SQUARED}")
    print(f"  Alpha range: [{ALPHA_MIN}, {ALPHA_MAX}]")

    inside_alpha, outside_alpha = pool_alpha_data(folder_path)

    if inside_alpha is None or outside_alpha is None:
        print("No data to analyze. Exiting.")
        return

    if len(inside_alpha) == 0 and len(outside_alpha) == 0:
        print("\nNo valid alpha values found. Check:")
        print("  - Trajectory lengths (need at least 10 frames)")
        print("  - Fit quality (R² threshold)")
        return

    # Perform statistical tests
    stats_results = perform_statistical_tests(inside_alpha, outside_alpha)

    # Create output directory
    output_dir = os.path.join(folder_path, OUTPUT_SUBFOLDER)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput will be saved to: {output_dir}")

    # Print summary statistics
    print_alpha_summary(inside_alpha, outside_alpha, stats_results)

    # Generate plots and export data
    print("\n" + "="*70)
    print("GENERATING OUTPUTS")
    print("="*70)

    if len(inside_alpha) > 0 and len(outside_alpha) > 0:
        plot_alpha_histogram(inside_alpha, outside_alpha,
                           os.path.join(output_dir, 'pooled_alpha_histogram.png'))

        plot_alpha_cdf(inside_alpha, outside_alpha,
                      os.path.join(output_dir, 'pooled_alpha_cdf.png'))

        plot_alpha_boxplot(inside_alpha, outside_alpha, stats_results,
                          os.path.join(output_dir, 'pooled_alpha_boxplot.png'))

        plot_alpha_violin(inside_alpha, outside_alpha,
                         os.path.join(output_dir, 'pooled_alpha_violin.png'))

        export_summary_to_csv(inside_alpha, outside_alpha, stats_results,
                            os.path.join(output_dir, 'pooled_alpha_results.csv'))
    else:
        print("Skipping plots - need data in both inside and outside ROIs")

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
