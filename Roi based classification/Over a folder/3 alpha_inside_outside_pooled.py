# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 18:52:42 2025

@author: wanglab-PC-2
"""
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
# Minimum R² for alpha fit quality (stricter for pooled analysis)
MIN_R_SQUARED = 0.8
# Alpha bounds for filtering outliers
ALPHA_MIN = 0.5
ALPHA_MAX = 1.5
# Bin size for alpha histogram
ALPHA_BIN_SIZE = 0.05
# Output subfolder name
OUTPUT_SUBFOLDER = 'pooled_alpha_inside_outside_analysis'
# Bootstrap samples for confidence intervals
N_BOOTSTRAP = 1000
# Minimum fit points required
MIN_FIT_POINTS = 4
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


def calculate_adaptive_max_lag(trajectory_length, min_points=MIN_FIT_POINTS):
    """
    Adaptively calculate max lag based on trajectory length.
    Ensures sufficient averaging while maximizing fitting window.

    Theory: At tau = N/4, ~N/4 non-overlapping pairs available for averaging.

    Args:
        trajectory_length: Number of frames in trajectory
        min_points: Minimum number of points required for fitting

    Returns:
        Maximum lag index to use for fitting
    """
    # Use 25-30% of trajectory as max lag depending on length
    if trajectory_length < 15:
        # For very short trajectories, use 20% to ensure enough pairs
        max_lag_fraction = 0.20
    else:
        max_lag_fraction = MAX_LAG_FRACTION

    max_lag_points = max(int(trajectory_length * max_lag_fraction), min_points)

    return min(max_lag_points, trajectory_length - 1)


def classify_diffusion_type(alpha):
    """
    Classify diffusion based on alpha value.

    Args:
        alpha: Alpha exponent value

    Returns:
        String describing diffusion type
    """
    if alpha < 0.2:
        return 'confined'
    elif alpha < 0.9:
        return 'sub-diffusive'
    elif alpha <= 1.1:
        return 'normal'
    else:
        return 'super-diffusive'


def validate_alpha_fit(fit_results, msd_data, time_data):
    """
    Comprehensive quality control for alpha fitting.

    Args:
        fit_results: Dictionary with fitting results
        msd_data: Array of MSD values
        time_data: Array of time lag values

    Returns:
        Dictionary with validation flags and warnings
    """
    warnings = []
    quality_score = 100  # Start at perfect

    # 1. Check R² threshold
    r_squared = fit_results.get('r_squared', 0)
    if r_squared < 0.7:
        warnings.append('Poor fit quality (R² < 0.7)')
        quality_score -= 30
    elif r_squared < 0.8:
        warnings.append('Moderate fit quality (R² < 0.8)')
        quality_score -= 15

    # 2. Check alpha bounds (physical reasonableness)
    alpha = fit_results.get('alpha', np.nan)
    if np.isnan(alpha) or alpha < 0.0 or alpha > 2.5:
        warnings.append(f'Alpha outside physical range: {alpha:.3f}')
        quality_score -= 40

    # 3. Check trajectory length adequacy
    traj_length = fit_results.get('trajectory_length', 0)
    if traj_length < MIN_TRAJECTORY_LENGTH:
        warnings.append(f'Trajectory too short: {traj_length} frames')
        quality_score -= 20

    # 4. Check number of fit points
    n_fit_points = fit_results.get('n_fit_points', 0)
    if n_fit_points < MIN_FIT_POINTS:
        warnings.append(f'Too few fit points: {n_fit_points}')
        quality_score -= 35

    # 5. Check bootstrap CI width (if available)
    if 'alpha_CI_low' in fit_results and not np.isnan(fit_results['alpha_CI_low']):
        ci_width = fit_results['alpha_CI_high'] - fit_results['alpha_CI_low']
        relative_uncertainty = ci_width / abs(alpha) if alpha != 0 else np.inf

        if relative_uncertainty > 0.5:  # 50% uncertainty
            warnings.append(f'Large alpha uncertainty: ±{relative_uncertainty*100:.1f}%')
            quality_score -= 25

    # 6. Check for residual patterns (systematic curvature)
    if n_fit_points >= 4:
        valid_idx = min(n_fit_points, len(msd_data))
        log_t = np.log(time_data[:valid_idx])
        log_msd = np.log(msd_data[:valid_idx])

        # Remove any NaN/inf values
        valid = np.isfinite(log_t) & np.isfinite(log_msd)
        if np.sum(valid) >= 4:
            log_t = log_t[valid]
            log_msd = log_msd[valid]

            predicted = alpha * log_t + np.log(4 * fit_results.get('D', 1))
            residuals = log_msd - predicted

            if np.std(residuals) > 0.5:
                warnings.append('High residual scatter - possible non-power-law behavior')
                quality_score -= 20

    # Determine quality category
    if quality_score >= 80:
        quality = 'PASS'
    elif quality_score >= 50:
        quality = 'WARNING'
    else:
        quality = 'FAIL'

    return {
        'quality': quality,
        'quality_score': max(quality_score, 0),
        'warnings': warnings,
        'n_warnings': len(warnings)
    }


def fit_anomalous_diffusion(time_data, msd_data, trajectory_length, n_bootstrap=N_BOOTSTRAP):
    """
    Fit anomalous diffusion model to MSD curve using log-log regression with bootstrap CI.

    Args:
        time_data: Array of time lag values
        msd_data: Array of MSD values
        trajectory_length: Total length of trajectory (for adaptive lag selection)
        n_bootstrap: Number of bootstrap iterations for CI estimation

    Returns:
        Dictionary with fitting parameters, metrics, and confidence intervals
    """
    # Use adaptive max lag based on trajectory length
    max_points = calculate_adaptive_max_lag(trajectory_length)

    # Extract data for fitting
    t_fit = time_data[:max_points]
    msd_fit = msd_data[:max_points]

    # Filter out any NaN or zero values
    valid_indices = ~np.isnan(msd_fit) & (msd_fit > 0) & (t_fit > 0)
    t_fit = t_fit[valid_indices]
    msd_fit = msd_fit[valid_indices]

    # Check if we have enough points after filtering
    if len(t_fit) < MIN_FIT_POINTS:
        return {
            'D': np.nan,
            'alpha': np.nan,
            'alpha_CI_low': np.nan,
            'alpha_CI_high': np.nan,
            'alpha_sem': np.nan,
            'r_squared': np.nan,
            'n_fit_points': len(t_fit),
            'trajectory_length': trajectory_length
        }

    try:
        # Linear fit on log-log scale (most stable method)
        log_t = np.log(t_fit)
        log_msd = np.log(msd_fit)

        # Main regression: log(MSD) = log(4D) + alpha*log(t)
        slope, intercept, r_value, _, _ = stats.linregress(log_t, log_msd)

        alpha_main = slope
        D_main = np.exp(intercept) / 4  # MSD = 4*D*t^alpha
        r_squared = r_value ** 2

        # Bootstrap for alpha confidence intervals
        alpha_bootstrap = []
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(t_fit), size=len(t_fit), replace=True)
            try:
                slope_boot, _, _, _, _ = stats.linregress(log_t[indices], log_msd[indices])
                alpha_bootstrap.append(slope_boot)
            except:
                continue

        # Calculate 95% CI
        if len(alpha_bootstrap) > 0:
            alpha_CI_low, alpha_CI_high = np.percentile(alpha_bootstrap, [2.5, 97.5])
            alpha_sem = np.std(alpha_bootstrap)
        else:
            alpha_CI_low = alpha_CI_high = alpha_sem = np.nan

        return {
            'D': D_main,
            'alpha': alpha_main,
            'alpha_CI_low': alpha_CI_low,
            'alpha_CI_high': alpha_CI_high,
            'alpha_sem': alpha_sem,
            'r_squared': r_squared,
            'n_fit_points': len(t_fit),
            'trajectory_length': trajectory_length
        }
    except Exception as e:
        return {
            'D': np.nan,
            'alpha': np.nan,
            'alpha_CI_low': np.nan,
            'alpha_CI_high': np.nan,
            'alpha_sem': np.nan,
            'r_squared': np.nan,
            'n_fit_points': len(t_fit),
            'trajectory_length': trajectory_length
        }


def calculate_alpha_for_trajectories(trajectories):
    """
    Calculate alpha exponent for a list of trajectories with quality control.

    Args:
        trajectories: List of trajectory dictionaries

    Returns:
        Tuple of (alpha_values, detailed_results, filtering_stats)
    """
    alpha_values = []
    detailed_results = []

    # Initialize filtering statistics
    filtering_stats = {
        'total_trajectories': len(trajectories),
        'too_short': 0,
        'insufficient_fit_points': 0,
        'poor_r_squared': 0,
        'alpha_out_of_bounds': 0,
        'nan_alpha': 0,
        'passed_quality_pass': 0,
        'passed_quality_warning': 0,
        'failed_quality': 0
    }

    for traj in trajectories:
        traj_id = traj.get('id', 'unknown')
        traj_length = len(traj['x'])

        # Check 1: Trajectory length
        if traj_length < MIN_TRAJECTORY_LENGTH:
            filtering_stats['too_short'] += 1
            continue

        # Calculate MSD
        msd, time_lags = calculate_msd(traj['x'], traj['y'], DT)

        # Fit anomalous diffusion model with bootstrap CI
        fit_results = fit_anomalous_diffusion(time_lags, msd, traj_length)

        # Check 2: NaN alpha
        if np.isnan(fit_results['alpha']):
            filtering_stats['nan_alpha'] += 1
            continue

        # Check 3: Insufficient fit points
        if fit_results['n_fit_points'] < MIN_FIT_POINTS:
            filtering_stats['insufficient_fit_points'] += 1
            continue

        # Check 4: R²
        if fit_results['r_squared'] < MIN_R_SQUARED:
            filtering_stats['poor_r_squared'] += 1
            continue

        # Check 5: Alpha bounds
        if not (ALPHA_MIN <= fit_results['alpha'] <= ALPHA_MAX):
            filtering_stats['alpha_out_of_bounds'] += 1
            continue

        # Validate fit quality
        validation = validate_alpha_fit(fit_results, msd, time_lags)

        # Store detailed results
        detailed_result = {
            'trajectory_id': traj_id,
            'trajectory_length': traj_length,
            'alpha': fit_results['alpha'],
            'alpha_CI_low': fit_results['alpha_CI_low'],
            'alpha_CI_high': fit_results['alpha_CI_high'],
            'alpha_sem': fit_results['alpha_sem'],
            'D': fit_results['D'],
            'r_squared': fit_results['r_squared'],
            'n_fit_points': fit_results['n_fit_points'],
            'quality': validation['quality'],
            'quality_score': validation['quality_score'],
            'n_warnings': validation['n_warnings'],
            'warnings': '; '.join(validation['warnings']),
            'diffusion_type': classify_diffusion_type(fit_results['alpha'])
        }

        detailed_results.append(detailed_result)

        # Track quality categories
        if validation['quality'] == 'PASS':
            filtering_stats['passed_quality_pass'] += 1
            alpha_values.append(fit_results['alpha'])
        elif validation['quality'] == 'WARNING':
            filtering_stats['passed_quality_warning'] += 1
            alpha_values.append(fit_results['alpha'])  # Still include WARNING
        else:
            filtering_stats['failed_quality'] += 1

    return alpha_values, detailed_results, filtering_stats


def analyze_alpha_single_file(roi_data, filename='unknown'):
    """
    Extract alpha values from a single ROI data file with detailed metrics.

    Args:
        roi_data: Dictionary containing ROI-assigned trajectory data
        filename: Source filename for tracking

    Returns:
        Tuple of (inside_alpha, outside_alpha, inside_detailed, outside_detailed,
                  inside_stats, outside_stats)
    """
    inside_alpha = []
    outside_alpha = []
    inside_detailed = []
    outside_detailed = []
    inside_stats = None
    outside_stats = None

    for roi_id, trajectories in roi_data['roi_trajectories'].items():
        if roi_id == 'unassigned':
            # Trajectories outside ROIs
            alpha_vals, detailed, stats = calculate_alpha_for_trajectories(trajectories)
            outside_alpha.extend(alpha_vals)
            # Add location and source to detailed results
            for d in detailed:
                d['location'] = 'outside'
                d['roi_id'] = 'unassigned'
                d['source_file'] = filename
            outside_detailed.extend(detailed)
            outside_stats = stats
        else:
            # Trajectories inside ROIs
            alpha_vals, detailed, stats = calculate_alpha_for_trajectories(trajectories)
            inside_alpha.extend(alpha_vals)
            # Add location and source to detailed results
            for d in detailed:
                d['location'] = 'inside'
                d['roi_id'] = roi_id
                d['source_file'] = filename
            inside_detailed.extend(detailed)

            # Aggregate stats if processing multiple ROIs
            if inside_stats is None:
                inside_stats = stats
            else:
                # Sum up the counts
                for key in stats:
                    inside_stats[key] += stats[key]

    return (inside_alpha, outside_alpha, inside_detailed, outside_detailed,
            inside_stats, outside_stats)


def pool_alpha_data(folder_path):
    """
    Pool alpha data from all roi_trajectory_data_*.pkl files in a folder with detailed metrics.

    Args:
        folder_path: Path to folder containing pkl files

    Returns:
        Tuple of (inside_alpha_array, outside_alpha_array, all_inside_detailed,
                  all_outside_detailed, pooled_inside_stats, pooled_outside_stats)
    """
    # Find all roi_trajectory_data pkl files
    pkl_pattern = os.path.join(folder_path, 'roi_trajectory_data_*.pkl')
    pkl_files = glob.glob(pkl_pattern)

    if not pkl_files:
        print(f"No roi_trajectory_data_*.pkl files found in {folder_path}")
        return None, None, None, None, None, None

    print(f"\nFound {len(pkl_files)} pkl files to process")
    print("="*70)

    # Lists to collect all alpha values and detailed results
    all_inside_alpha = []
    all_outside_alpha = []
    all_inside_detailed = []
    all_outside_detailed = []

    # Initialize pooled stats
    pooled_inside_stats = None
    pooled_outside_stats = None

    # Process each file
    for pkl_file in pkl_files:
        filename = os.path.basename(pkl_file)
        roi_data = load_roi_data(pkl_file)

        if roi_data is not None:
            (inside_alpha, outside_alpha, inside_detailed, outside_detailed,
             inside_stats, outside_stats) = analyze_alpha_single_file(roi_data, filename)

            all_inside_alpha.extend(inside_alpha)
            all_outside_alpha.extend(outside_alpha)
            all_inside_detailed.extend(inside_detailed)
            all_outside_detailed.extend(outside_detailed)

            print(f"    Inside: {len(inside_alpha)}, Outside: {len(outside_alpha)}")

            # Aggregate statistics
            if inside_stats:
                if pooled_inside_stats is None:
                    pooled_inside_stats = inside_stats
                else:
                    for key in inside_stats:
                        pooled_inside_stats[key] += inside_stats[key]

            if outside_stats:
                if pooled_outside_stats is None:
                    pooled_outside_stats = outside_stats
                else:
                    for key in outside_stats:
                        pooled_outside_stats[key] += outside_stats[key]

    print("="*70)
    print(f"Total pooled - Inside: {len(all_inside_alpha)}, Outside: {len(all_outside_alpha)}")

    return (np.array(all_inside_alpha), np.array(all_outside_alpha),
            all_inside_detailed, all_outside_detailed,
            pooled_inside_stats, pooled_outside_stats)


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


def print_filtering_report(location, stats):
    """
    Print detailed filtering statistics.

    Args:
        location: String describing the location ('Inside ROIs' or 'Outside ROIs')
        stats: Dictionary with filtering statistics
    """
    if stats is None:
        return

    total = stats['total_trajectories']
    if total == 0:
        print(f"\n{location}: No trajectories")
        return

    passed = stats['passed_quality_pass'] + stats['passed_quality_warning']
    filtered = total - passed

    print(f"\n{location}:")
    print(f"  Total trajectories:          {total}")
    print(f"  Passed (PASS):               {stats['passed_quality_pass']} "
          f"({stats['passed_quality_pass']/total*100:.1f}%)")
    print(f"  Passed (WARNING):            {stats['passed_quality_warning']} "
          f"({stats['passed_quality_warning']/total*100:.1f}%)")
    print(f"  Failed quality control:      {stats['failed_quality']} "
          f"({stats['failed_quality']/total*100:.1f}%)")
    print(f"\n  Filtering breakdown:")
    print(f"    Too short (< {MIN_TRAJECTORY_LENGTH} frames):   {stats['too_short']}")
    print(f"    Insufficient fit points:     {stats['insufficient_fit_points']}")
    print(f"    Poor R² (< {MIN_R_SQUARED}):        {stats['poor_r_squared']}")
    print(f"    Alpha out of bounds:         {stats['alpha_out_of_bounds']}")
    print(f"    NaN alpha (fit failed):      {stats['nan_alpha']}")


def export_detailed_results_csv(inside_detailed, outside_detailed, output_dir):
    """
    Export detailed per-trajectory alpha results to CSV for pooled data.

    Args:
        inside_detailed: List of detailed results for inside ROIs
        outside_detailed: List of detailed results for outside ROIs
        output_dir: Directory to save CSV files
    """
    # Combine all results
    all_results = inside_detailed + outside_detailed

    if len(all_results) == 0:
        print("No detailed results to export")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Sort by source file, location, and quality
    df = df.sort_values(['source_file', 'location', 'quality_score'],
                       ascending=[True, True, False])

    # Export to CSV
    csv_path = os.path.join(output_dir, 'pooled_alpha_detailed_results.csv')
    df.to_csv(csv_path, index=False)

    print(f"\nDetailed results exported to: {csv_path}")

    # Also create quality summary by location
    quality_summary = df.groupby(['location', 'quality']).size().unstack(fill_value=0)
    summary_path = os.path.join(output_dir, 'pooled_alpha_quality_summary.csv')
    quality_summary.to_csv(summary_path)

    print(f"Quality summary exported to: {summary_path}")

    # Export diffusion type distribution
    diffusion_summary = df.groupby(['location', 'diffusion_type']).size().unstack(fill_value=0)
    diffusion_path = os.path.join(output_dir, 'pooled_alpha_diffusion_type_summary.csv')
    diffusion_summary.to_csv(diffusion_path)

    print(f"Diffusion type summary exported to: {diffusion_path}")

    # Export per-file summary
    file_summary = df.groupby(['source_file', 'location']).agg({
        'alpha': ['count', 'mean', 'std'],
        'quality_score': 'mean'
    }).round(3)
    file_summary_path = os.path.join(output_dir, 'pooled_alpha_per_file_summary.csv')
    file_summary.to_csv(file_summary_path)

    print(f"Per-file summary exported to: {file_summary_path}")


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
    print(f"  Adaptive lag selection: 20-30% of trajectory length")
    print(f"  Minimum R²: {MIN_R_SQUARED}")
    print(f"  Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"  Alpha range: [{ALPHA_MIN}, {ALPHA_MAX}]")

    (inside_alpha, outside_alpha, inside_detailed, outside_detailed,
     inside_stats, outside_stats) = pool_alpha_data(folder_path)

    if inside_alpha is None or outside_alpha is None:
        print("No data to analyze. Exiting.")
        return

    # Print filtering reports
    print("\n" + "="*70)
    print("POOLED TRAJECTORY FILTERING REPORT")
    print("="*70)

    if inside_stats:
        print_filtering_report("Inside ROIs", inside_stats)

    if outside_stats:
        print_filtering_report("Outside ROIs", outside_stats)

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

        # Export detailed per-trajectory results
        export_detailed_results_csv(inside_detailed, outside_detailed, output_dir)
    else:
        print("Skipping plots - need data in both inside and outside ROIs")
        # Still export detailed results if available
        if inside_detailed or outside_detailed:
            export_detailed_results_csv(inside_detailed, outside_detailed, output_dir)

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()