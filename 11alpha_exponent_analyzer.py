#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11alpha_exponent_analyzer.py

Alpha Exponent Analysis for Single Particle Tracking Data

This script analyzes trajectory data to calculate the alpha exponent from the
generalized Mean Square Displacement (MSD) equation:
    MSD(τ) = 4*D*τ^α

Alpha interpretation:
    α = 1  : Normal (Brownian) diffusion
    α > 1  : Super-diffusion (ballistic, active transport)
    α < 1  : Sub-diffusion (confined, hindered motion)
    α ≈ 0  : Completely confined motion

The script takes analyzed trajectory files (analyzed_*.pkl from step 2) and
calculates alpha values using log-log linear regression.

Input:  analyzed_*.pkl files (from 2diffusion_analyzer.py)
Output:
    - alpha_analyzed_*.pkl (pickle with alpha values)
    - *_alpha_diagnostic_plots.png (visualization)
    - *_alpha_results.csv (all fitting parameters)
    - *_alpha_vs_diffusion.png (correlation plot)

Created: 2025
Author: GEM Analysis Pipeline
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL PARAMETERS - Modify as needed for your data
# ============================================================================

# Time step between frames (seconds)
DT = 0.1

# Minimum trajectory length for analysis (frames)
MIN_TRACK_LENGTH = 10

# Minimum number of time lags needed for alpha fitting
MIN_TIME_LAGS = 4

# Maximum lag to use for fitting (fraction of trajectory length)
# For 10-25 frame trajectories, 0.3 gives 3-7 points for fitting
MAX_LAG_FRACTION = 0.3

# Minimum R² for accepting alpha fit
ALPHA_FIT_MIN_R2 = 0.6

# Alpha range for quality filtering (exclude extreme outliers)
ALPHA_MIN = 0.0
ALPHA_MAX = 2.5

# Fitting method: 'loglog' (recommended) or 'powerlaw'
PRIMARY_FIT_METHOD = 'loglog'

# Whether to also calculate the alternative method for validation
CALCULATE_BOTH_METHODS = True

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_analyzed_data(file_path):
    """
    Load analyzed trajectory data from pickle file.

    Args:
        file_path: Path to analyzed_*.pkl file

    Returns:
        Dictionary with trajectory data
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded data from: {os.path.basename(file_path)}")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def calculate_msd_vs_time_lag(trajectory, dt=DT, max_lag_fraction=MAX_LAG_FRACTION):
    """
    Calculate MSD for multiple time lags.

    This function calculates the Mean Square Displacement for each time lag
    from lag=1 up to a maximum lag determined by max_lag_fraction.

    Args:
        trajectory: Dictionary with 'x' and 'y' coordinate arrays
        dt: Time step between frames (seconds)
        max_lag_fraction: Maximum lag as fraction of trajectory length

    Returns:
        Dictionary with:
            - time_lags: Array of time lag values (seconds)
            - msd_values: Array of MSD values (μm²)
            - n_points: Number of time lags calculated
    """
    x = np.array(trajectory['x'])
    y = np.array(trajectory['y'])

    n_frames = len(x)
    max_lag = max(MIN_TIME_LAGS, int(n_frames * max_lag_fraction))
    max_lag = min(max_lag, n_frames - 1)

    time_lags = []
    msd_values = []

    for lag in range(1, max_lag + 1):
        # Calculate squared displacements for this lag
        squared_displacements = []

        for i in range(n_frames - lag):
            dx = x[i + lag] - x[i]
            dy = y[i + lag] - y[i]
            squared_disp = dx**2 + dy**2
            squared_displacements.append(squared_disp)

        if squared_displacements:
            msd = np.mean(squared_displacements)
            msd_values.append(msd)
            time_lags.append(lag * dt)

    return {
        'time_lags': np.array(time_lags),
        'msd_values': np.array(msd_values),
        'n_points': len(time_lags)
    }


def fit_alpha_loglog(time_lags, msd_values):
    """
    Fit alpha exponent using log-log linear regression.

    Model: log(MSD) = log(4*D) + α*log(τ)

    This is the PRIMARY and RECOMMENDED method as it is more numerically stable
    and less sensitive to noise than direct power-law fitting.

    Args:
        time_lags: Array of time lag values (seconds)
        msd_values: Array of MSD values (μm²)

    Returns:
        Dictionary with:
            - alpha: Alpha exponent (slope of log-log plot)
            - alpha_err: Standard error of alpha
            - D_generalized: Generalized diffusion coefficient
            - intercept: Log-log intercept
            - r_squared: Coefficient of determination
            - method: 'loglog'
    """
    # Remove any zero or negative values
    valid_idx = (time_lags > 0) & (msd_values > 0)

    if np.sum(valid_idx) < MIN_TIME_LAGS:
        return None

    t_valid = time_lags[valid_idx]
    msd_valid = msd_values[valid_idx]

    # Take logarithms
    log_t = np.log10(t_valid)
    log_msd = np.log10(msd_valid)

    try:
        # Linear regression in log-log space
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_t, log_msd)

        # Alpha is the slope
        alpha = slope
        alpha_err = std_err

        # Generalized diffusion coefficient from intercept
        # log(MSD) = log(4*D) + α*log(t)
        # intercept = log(4*D) → D = 10^intercept / 4
        D_generalized = (10**intercept) / 4.0

        r_squared = r_value**2

        return {
            'alpha': alpha,
            'alpha_err': alpha_err,
            'D_generalized': D_generalized,
            'intercept': intercept,
            'r_squared': r_squared,
            'p_value': p_value,
            'method': 'loglog',
            'log_t': log_t,
            'log_msd': log_msd,
            'fit_log_msd': slope * log_t + intercept
        }

    except Exception as e:
        print(f"Error in log-log fitting: {e}")
        return None


def fit_alpha_powerlaw(time_lags, msd_values):
    """
    Fit alpha exponent using direct power-law fitting.

    Model: MSD(τ) = 4*D*τ^α

    This is a VALIDATION method. It can be less stable than log-log regression
    but provides an independent check.

    Args:
        time_lags: Array of time lag values (seconds)
        msd_values: Array of MSD values (μm²)

    Returns:
        Dictionary with:
            - alpha: Alpha exponent
            - alpha_err: Standard error of alpha
            - D_generalized: Generalized diffusion coefficient
            - r_squared: Coefficient of determination
            - method: 'powerlaw'
    """
    # Remove any zero or negative values
    valid_idx = (time_lags > 0) & (msd_values > 0)

    if np.sum(valid_idx) < MIN_TIME_LAGS:
        return None

    t_valid = time_lags[valid_idx]
    msd_valid = msd_values[valid_idx]

    # Power-law function
    def power_law_msd(t, D, alpha):
        return 4 * D * (t ** alpha)

    try:
        # Initial guess: assume normal diffusion (α=1)
        # Estimate D from first few points
        D_init = np.mean(msd_valid[:3] / (4 * t_valid[:3])) if len(msd_valid) >= 3 else 0.1

        # Fit the power-law model
        popt, pcov = curve_fit(
            power_law_msd,
            t_valid,
            msd_valid,
            p0=[D_init, 1.0],
            bounds=([0, 0], [np.inf, 3.0]),
            maxfev=10000
        )

        D_generalized, alpha = popt

        # Get standard errors
        perr = np.sqrt(np.diag(pcov))
        D_err, alpha_err = perr

        # Calculate R²
        msd_fit = power_law_msd(t_valid, D_generalized, alpha)
        residuals = msd_valid - msd_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((msd_valid - np.mean(msd_valid))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            'alpha': alpha,
            'alpha_err': alpha_err,
            'D_generalized': D_generalized,
            'D_err': D_err,
            'r_squared': r_squared,
            'method': 'powerlaw',
            't_fit': t_valid,
            'msd_fit': msd_fit
        }

    except Exception as e:
        print(f"Error in power-law fitting: {e}")
        return None


def classify_diffusion_type(alpha, alpha_err=None):
    """
    Classify diffusion type based on alpha value.

    Args:
        alpha: Alpha exponent value
        alpha_err: Standard error (optional, for confidence assessment)

    Returns:
        String: 'normal', 'sub-diffusion', 'super-diffusion', or 'confined'
    """
    if np.isnan(alpha):
        return 'unknown'

    # Define thresholds with some tolerance
    # If alpha_err is provided, use it for confidence
    if alpha_err is not None and alpha_err > 0:
        # Use 1-sigma confidence interval
        lower_bound = alpha - alpha_err
        upper_bound = alpha + alpha_err

        # If confidence interval spans normal diffusion, be conservative
        if lower_bound < 1.0 < upper_bound:
            return 'normal'

    # Classification based on alpha value
    if alpha < 0.2:
        return 'confined'
    elif alpha < 0.9:
        return 'sub-diffusion'
    elif alpha <= 1.1:
        return 'normal'
    else:
        return 'super-diffusion'


def analyze_trajectory_alpha(trajectory, traj_id, dt=DT):
    """
    Calculate alpha exponent for a single trajectory.

    Args:
        trajectory: Dictionary with trajectory data
        traj_id: Trajectory identifier
        dt: Time step (seconds)

    Returns:
        Dictionary with alpha analysis results
    """
    # Calculate MSD vs time lag
    msd_data = calculate_msd_vs_time_lag(trajectory, dt=dt)

    if msd_data['n_points'] < MIN_TIME_LAGS:
        return None

    # Primary method: log-log fitting
    loglog_result = fit_alpha_loglog(msd_data['time_lags'], msd_data['msd_values'])

    # Secondary method: power-law fitting (if requested)
    powerlaw_result = None
    if CALCULATE_BOTH_METHODS:
        powerlaw_result = fit_alpha_powerlaw(msd_data['time_lags'], msd_data['msd_values'])

    # Use primary method for main results
    if loglog_result is None:
        return None

    alpha = loglog_result['alpha']
    alpha_err = loglog_result['alpha_err']

    # Classify diffusion type
    diffusion_type = classify_diffusion_type(alpha, alpha_err)

    # Compile results
    result = {
        'id': traj_id,
        'x': trajectory['x'],
        'y': trajectory['y'],

        # Alpha from primary method (log-log)
        'alpha': alpha,
        'alpha_err': alpha_err,
        'alpha_r_squared': loglog_result['r_squared'],
        'alpha_p_value': loglog_result['p_value'],
        'D_generalized': loglog_result['D_generalized'],
        'alpha_intercept': loglog_result['intercept'],
        'alpha_method': 'loglog',

        # Diffusion classification
        'diffusion_type': diffusion_type,

        # MSD data
        'msd_time_lags': msd_data['time_lags'],
        'msd_values': msd_data['msd_values'],
        'n_time_lags': msd_data['n_points'],

        # Fitting data for plotting
        'log_t': loglog_result['log_t'],
        'log_msd': loglog_result['log_msd'],
        'fit_log_msd': loglog_result['fit_log_msd'],

        # Original diffusion coefficient (if available)
        'D_original': trajectory.get('D', np.nan),
        'D_err_original': trajectory.get('D_err', np.nan),
        'r_squared_original': trajectory.get('r_squared', np.nan),
        'radius_of_gyration': trajectory.get('radius_of_gyration', np.nan),
        'track_length': len(trajectory['x'])
    }

    # Add power-law results if calculated
    if powerlaw_result is not None:
        result['alpha_powerlaw'] = powerlaw_result['alpha']
        result['alpha_powerlaw_err'] = powerlaw_result['alpha_err']
        result['alpha_powerlaw_r_squared'] = powerlaw_result['r_squared']
        result['D_generalized_powerlaw'] = powerlaw_result['D_generalized']

    return result


def analyze_file(input_file_path, output_dir=None):
    """
    Analyze all trajectories in a single pickle file for alpha exponents.

    Args:
        input_file_path: Path to analyzed_*.pkl file
        output_dir: Directory for output files (default: same as input)

    Returns:
        Dictionary with results for all trajectories
    """
    # Load data
    data = load_analyzed_data(input_file_path)
    if data is None:
        return None

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Get base filename
    base_name = os.path.basename(input_file_path)
    if base_name.startswith('analyzed_'):
        base_name = base_name.replace('analyzed_', '')
    base_name = base_name.replace('.pkl', '')

    print(f"\nAnalyzing alpha exponents for: {base_name}")
    print(f"Total trajectories: {len(data['trajectories'])}")

    # Analyze each trajectory
    alpha_results = []
    n_analyzed = 0
    n_failed = 0

    for traj in data['trajectories']:
        traj_id = traj.get('id', 'unknown')

        # Check minimum length
        if len(traj['x']) < MIN_TRACK_LENGTH:
            n_failed += 1
            continue

        # Analyze trajectory
        result = analyze_trajectory_alpha(traj, traj_id)

        if result is not None:
            # Quality filtering
            if (result['alpha_r_squared'] >= ALPHA_FIT_MIN_R2 and
                ALPHA_MIN <= result['alpha'] <= ALPHA_MAX):
                alpha_results.append(result)
                n_analyzed += 1
            else:
                n_failed += 1
        else:
            n_failed += 1

    print(f"Successfully analyzed: {n_analyzed}")
    print(f"Failed or filtered: {n_failed}")

    if n_analyzed == 0:
        print("No trajectories passed quality filters!")
        return None

    # Compile statistics
    alpha_values = [r['alpha'] for r in alpha_results]
    D_original_values = [r['D_original'] for r in alpha_results if not np.isnan(r['D_original'])]
    D_generalized_values = [r['D_generalized'] for r in alpha_results]

    # Count diffusion types
    diffusion_types = [r['diffusion_type'] for r in alpha_results]
    type_counts = {
        'normal': diffusion_types.count('normal'),
        'sub-diffusion': diffusion_types.count('sub-diffusion'),
        'super-diffusion': diffusion_types.count('super-diffusion'),
        'confined': diffusion_types.count('confined')
    }

    results_summary = {
        'file_name': base_name,
        'n_trajectories_analyzed': n_analyzed,
        'n_trajectories_failed': n_failed,
        'trajectories': alpha_results,

        # Alpha statistics
        'alpha_mean': np.mean(alpha_values),
        'alpha_median': np.median(alpha_values),
        'alpha_std': np.std(alpha_values),
        'alpha_sem': np.std(alpha_values) / np.sqrt(len(alpha_values)),

        # Diffusion coefficient statistics
        'D_original_mean': np.mean(D_original_values) if D_original_values else np.nan,
        'D_generalized_mean': np.mean(D_generalized_values),

        # Diffusion type classification
        'diffusion_type_counts': type_counts,
        'diffusion_type_percentages': {
            k: v/n_analyzed*100 for k, v in type_counts.items()
        },

        # Global parameters used
        'parameters': {
            'DT': DT,
            'MIN_TRACK_LENGTH': MIN_TRACK_LENGTH,
            'MIN_TIME_LAGS': MIN_TIME_LAGS,
            'MAX_LAG_FRACTION': MAX_LAG_FRACTION,
            'ALPHA_FIT_MIN_R2': ALPHA_FIT_MIN_R2,
            'PRIMARY_FIT_METHOD': PRIMARY_FIT_METHOD
        }
    }

    # Print summary
    print(f"\n=== Alpha Analysis Summary ===")
    print(f"Mean alpha: {results_summary['alpha_mean']:.3f} ± {results_summary['alpha_sem']:.3f}")
    print(f"Median alpha: {results_summary['alpha_median']:.3f}")
    print(f"\nDiffusion type distribution:")
    for dtype, pct in results_summary['diffusion_type_percentages'].items():
        print(f"  {dtype}: {pct:.1f}% (n={type_counts[dtype]})")

    return results_summary


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_results_csv(results, output_dir):
    """
    Export alpha analysis results to CSV file.

    Args:
        results: Dictionary with analysis results
        output_dir: Directory for output file
    """
    base_name = results['file_name']

    # Prepare data for CSV
    csv_data = []

    for traj in results['trajectories']:
        row = {
            'trajectory_id': traj['id'],
            'track_length': traj['track_length'],

            # Alpha values (primary method)
            'alpha': traj['alpha'],
            'alpha_err': traj['alpha_err'],
            'alpha_r_squared': traj['alpha_r_squared'],
            'alpha_p_value': traj['alpha_p_value'],
            'alpha_method': traj['alpha_method'],

            # Diffusion coefficients
            'D_generalized': traj['D_generalized'],
            'D_original': traj['D_original'],
            'D_err_original': traj['D_err_original'],

            # Diffusion classification
            'diffusion_type': traj['diffusion_type'],

            # Other metrics
            'radius_of_gyration': traj['radius_of_gyration'],
            'r_squared_original': traj['r_squared_original'],
            'n_time_lags': traj['n_time_lags']
        }

        # Add power-law results if available
        if 'alpha_powerlaw' in traj:
            row['alpha_powerlaw'] = traj['alpha_powerlaw']
            row['alpha_powerlaw_err'] = traj['alpha_powerlaw_err']
            row['alpha_powerlaw_r_squared'] = traj['alpha_powerlaw_r_squared']

        csv_data.append(row)

    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, f'{base_name}_alpha_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nExported CSV: {csv_path}")

    # Also export summary statistics
    summary_data = {
        'Parameter': ['n_trajectories', 'alpha_mean', 'alpha_median', 'alpha_std',
                     'alpha_sem', 'D_original_mean', 'D_generalized_mean',
                     'normal_diffusion_%', 'sub_diffusion_%', 'super_diffusion_%', 'confined_%'],
        'Value': [
            results['n_trajectories_analyzed'],
            results['alpha_mean'],
            results['alpha_median'],
            results['alpha_std'],
            results['alpha_sem'],
            results['D_original_mean'],
            results['D_generalized_mean'],
            results['diffusion_type_percentages']['normal'],
            results['diffusion_type_percentages']['sub-diffusion'],
            results['diffusion_type_percentages']['super-diffusion'],
            results['diffusion_type_percentages']['confined']
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f'{base_name}_alpha_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Exported summary: {summary_path}")

    return csv_path


def export_results_pickle(results, output_dir):
    """
    Export alpha analysis results to pickle file.

    Args:
        results: Dictionary with analysis results
        output_dir: Directory for output file
    """
    base_name = results['file_name']
    pickle_path = os.path.join(output_dir, f'alpha_analyzed_{base_name}.pkl')

    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Exported pickle: {pickle_path}")
    return pickle_path


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_diagnostic_plots(results, output_dir, n_examples=6):
    """
    Create comprehensive diagnostic plots for alpha analysis.

    Args:
        results: Dictionary with analysis results
        output_dir: Directory for output files
        n_examples: Number of example trajectories to plot
    """
    base_name = results['file_name']

    # Create figure with multiple panels
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Panel 1: Alpha distribution histogram
    ax1 = fig.add_subplot(gs[0, 0])
    alpha_values = [t['alpha'] for t in results['trajectories']]
    ax1.hist(alpha_values, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    ax1.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Normal diffusion (α=1)')
    ax1.axvline(results['alpha_mean'], color='green', linestyle='-', linewidth=2,
                label=f'Mean α={results["alpha_mean"]:.2f}')
    ax1.set_xlabel('Alpha exponent', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'Alpha Distribution (n={len(alpha_values)})', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Diffusion type pie chart
    ax2 = fig.add_subplot(gs[0, 1])
    type_counts = results['diffusion_type_counts']
    labels = []
    sizes = []
    colors_pie = []
    color_map = {
        'confined': '#ff6b6b',
        'sub-diffusion': '#ffd93d',
        'normal': '#6bcf7f',
        'super-diffusion': '#4d96ff'
    }

    for dtype in ['confined', 'sub-diffusion', 'normal', 'super-diffusion']:
        if type_counts[dtype] > 0:
            labels.append(f'{dtype}\n({type_counts[dtype]})')
            sizes.append(type_counts[dtype])
            colors_pie.append(color_map[dtype])

    ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Diffusion Type Classification', fontsize=12, fontweight='bold')

    # Panel 3: Alpha vs R²
    ax3 = fig.add_subplot(gs[0, 2])
    alpha_vals = [t['alpha'] for t in results['trajectories']]
    r2_vals = [t['alpha_r_squared'] for t in results['trajectories']]
    ax3.scatter(alpha_vals, r2_vals, alpha=0.6, s=30, c='steelblue', edgecolors='black', linewidth=0.5)
    ax3.axhline(ALPHA_FIT_MIN_R2, color='red', linestyle='--', label=f'Min R²={ALPHA_FIT_MIN_R2}')
    ax3.axvline(1.0, color='orange', linestyle='--', alpha=0.5, label='α=1')
    ax3.set_xlabel('Alpha exponent', fontsize=11)
    ax3.set_ylabel('Fit R²', fontsize=11)
    ax3.set_title('Fit Quality vs Alpha', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Alpha vs Track Length
    ax4 = fig.add_subplot(gs[0, 3])
    track_lengths = [t['track_length'] for t in results['trajectories']]
    ax4.scatter(track_lengths, alpha_vals, alpha=0.6, s=30, c='steelblue', edgecolors='black', linewidth=0.5)
    ax4.axhline(1.0, color='red', linestyle='--', label='α=1')
    ax4.set_xlabel('Track length (frames)', fontsize=11)
    ax4.set_ylabel('Alpha exponent', fontsize=11)
    ax4.set_title('Alpha vs Track Length', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Alpha vs Diffusion Coefficient (MAIN PLOT REQUESTED BY USER)
    ax5 = fig.add_subplot(gs[1, 0])
    D_orig_vals = [t['D_original'] for t in results['trajectories'] if not np.isnan(t['D_original'])]
    alpha_for_D = [t['alpha'] for t in results['trajectories'] if not np.isnan(t['D_original'])]

    if D_orig_vals:
        scatter = ax5.scatter(D_orig_vals, alpha_for_D, alpha=0.6, s=40,
                            c=alpha_for_D, cmap='RdYlGn_r',
                            edgecolors='black', linewidth=0.5, vmin=0.5, vmax=1.5)
        ax5.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Normal diffusion')
        ax5.set_xlabel('Diffusion coefficient D (μm²/s)', fontsize=11)
        ax5.set_ylabel('Alpha exponent', fontsize=11)
        ax5.set_title('Alpha vs Diffusion Coefficient', fontsize=12, fontweight='bold')
        ax5.set_xscale('log')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Alpha', fontsize=9)

    # Panel 6: D_generalized vs D_original
    ax6 = fig.add_subplot(gs[1, 1])
    D_gen_vals = [t['D_generalized'] for t in results['trajectories'] if not np.isnan(t['D_original'])]

    if D_orig_vals and D_gen_vals:
        ax6.scatter(D_orig_vals, D_gen_vals, alpha=0.6, s=40, c='steelblue',
                   edgecolors='black', linewidth=0.5)

        # Add diagonal line
        all_D = D_orig_vals + D_gen_vals
        D_min, D_max = min(all_D), max(all_D)
        ax6.plot([D_min, D_max], [D_min, D_max], 'r--', linewidth=2, label='1:1 line')

        ax6.set_xlabel('D original (μm²/s)', fontsize=11)
        ax6.set_ylabel('D generalized (μm²/s)', fontsize=11)
        ax6.set_title('Generalized vs Original D', fontsize=12, fontweight='bold')
        ax6.set_xscale('log')
        ax6.set_yscale('log')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)

    # Panel 7-12: Example MSD fits (log-log plots)
    # Select diverse examples
    trajectories_sorted = sorted(results['trajectories'], key=lambda x: x['alpha'])
    n_traj = len(trajectories_sorted)

    if n_traj > 0:
        # Select examples spanning the alpha range
        indices = np.linspace(0, n_traj-1, min(n_examples, n_traj), dtype=int)
        examples = [trajectories_sorted[i] for i in indices]

        # Define grid positions for 6 example plots
        # Row 1: cols 2-3, Row 2: cols 0-3
        example_positions = [
            (1, 2), (1, 3),  # Row 1, last 2 columns
            (2, 0), (2, 1), (2, 2), (2, 3)  # Row 2, all 4 columns
        ]

        # Plot examples
        for idx, traj in enumerate(examples[:6]):
            row, col = example_positions[idx]
            ax = fig.add_subplot(gs[row, col])

            # Log-log plot
            ax.plot(traj['log_t'], traj['log_msd'], 'o', markersize=6,
                   alpha=0.7, label='Data', color='steelblue')
            ax.plot(traj['log_t'], traj['fit_log_msd'], '-', linewidth=2,
                   color='red', label=f'Fit: α={traj["alpha"]:.2f}')

            ax.set_xlabel('log₁₀(time lag)', fontsize=9)
            ax.set_ylabel('log₁₀(MSD)', fontsize=9)
            ax.set_title(f'Traj {traj["id"]} | {traj["diffusion_type"]}',
                       fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Add text box with fit info
            textstr = f'R²={traj["alpha_r_squared"]:.3f}\nn={traj["n_time_lags"]}'
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round',
                   facecolor='wheat', alpha=0.5))

    # Add overall title
    fig.suptitle(f'Alpha Exponent Analysis: {base_name}', fontsize=16, fontweight='bold')

    # Save figure
    plot_path = os.path.join(output_dir, f'{base_name}_alpha_diagnostic_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved diagnostic plots: {plot_path}")


def create_alpha_vs_diffusion_plot(results, output_dir):
    """
    Create a dedicated high-quality alpha vs diffusion coefficient plot.

    This is the MAIN CORRELATION PLOT requested by the user.

    Args:
        results: Dictionary with analysis results
        output_dir: Directory for output file
    """
    base_name = results['file_name']

    # Extract data
    data_points = []
    for traj in results['trajectories']:
        if not np.isnan(traj['D_original']):
            data_points.append({
                'D': traj['D_original'],
                'alpha': traj['alpha'],
                'alpha_err': traj['alpha_err'],
                'diffusion_type': traj['diffusion_type'],
                'r_squared': traj['alpha_r_squared']
            })

    if not data_points:
        print("No valid D values for alpha vs D plot")
        return

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Scatter with diffusion type colors
    D_vals = [d['D'] for d in data_points]
    alpha_vals = [d['alpha'] for d in data_points]
    types = [d['diffusion_type'] for d in data_points]

    color_map = {
        'confined': '#ff6b6b',
        'sub-diffusion': '#ffd93d',
        'normal': '#6bcf7f',
        'super-diffusion': '#4d96ff'
    }

    colors = [color_map[t] for t in types]

    ax1.scatter(D_vals, alpha_vals, c=colors, alpha=0.7, s=60,
               edgecolors='black', linewidth=0.8)

    # Add reference line for normal diffusion
    ax1.axhline(1.0, color='black', linestyle='--', linewidth=2.5,
               label='Normal diffusion (α=1)', zorder=1)

    # Shaded regions for diffusion types
    ax1.axhspan(0.9, 1.1, alpha=0.15, color='green', zorder=0)
    ax1.text(ax1.get_xlim()[0] * 1.1, 1.0, 'Normal', fontsize=10,
            verticalalignment='center', color='darkgreen', fontweight='bold')

    ax1.set_xlabel('Diffusion Coefficient D (μm²/s)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Alpha Exponent', fontsize=13, fontweight='bold')
    ax1.set_title(f'Alpha vs Diffusion Coefficient\n{base_name}',
                 fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=11, loc='best')

    # Add custom legend for diffusion types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['confined'], edgecolor='black', label='Confined'),
        Patch(facecolor=color_map['sub-diffusion'], edgecolor='black', label='Sub-diffusion'),
        Patch(facecolor=color_map['normal'], edgecolor='black', label='Normal'),
        Patch(facecolor=color_map['super-diffusion'], edgecolor='black', label='Super-diffusion')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10, title='Diffusion Type')

    # Plot 2: Hexbin density plot
    # Filter for reasonable ranges
    D_log = np.log10(D_vals)

    hexbin = ax2.hexbin(D_log, alpha_vals, gridsize=25, cmap='viridis',
                       mincnt=1, edgecolors='black', linewidths=0.2)
    ax2.axhline(1.0, color='white', linestyle='--', linewidth=2.5,
               label='Normal diffusion (α=1)')

    ax2.set_xlabel('log₁₀(Diffusion Coefficient) (μm²/s)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Alpha Exponent', fontsize=13, fontweight='bold')
    ax2.set_title(f'Density Map: Alpha vs D\nn={len(data_points)} trajectories',
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    # Add colorbar
    cbar = plt.colorbar(hexbin, ax=ax2)
    cbar.set_label('Count', fontsize=11)

    plt.tight_layout()

    # Save
    plot_path = os.path.join(output_dir, f'{base_name}_alpha_vs_diffusion.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved alpha vs D plot: {plot_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run alpha exponent analysis.
    """
    print("="*70)
    print("ALPHA EXPONENT ANALYZER FOR SINGLE PARTICLE TRACKING")
    print("="*70)

    # Get input file or directory
    print("\nInput options:")
    print("1. Single analyzed_*.pkl file")
    print("2. Directory containing multiple analyzed_*.pkl files")

    input_path = input("\nEnter path to file or directory: ").strip()

    if not os.path.exists(input_path):
        print(f"Error: Path does not exist: {input_path}")
        return

    # Output directory
    output_dir_input = input("Enter output directory (press Enter for same as input): ").strip()

    if not output_dir_input:
        if os.path.isfile(input_path):
            output_dir = os.path.dirname(input_path)
        else:
            output_dir = input_path
    else:
        output_dir = output_dir_input
        os.makedirs(output_dir, exist_ok=True)

    # Find files to process
    if os.path.isfile(input_path):
        files_to_process = [input_path]
    else:
        files_to_process = glob.glob(os.path.join(input_path, "analyzed_*.pkl"))

    if not files_to_process:
        print("No analyzed_*.pkl files found!")
        return

    print(f"\nFound {len(files_to_process)} file(s) to process")

    # Process each file
    for file_path in files_to_process:
        print("\n" + "="*70)

        # Analyze file
        results = analyze_file(file_path, output_dir)

        if results is None:
            print(f"Skipping {file_path} - analysis failed")
            continue

        # Export results
        export_results_csv(results, output_dir)
        export_results_pickle(results, output_dir)

        # Create plots
        create_diagnostic_plots(results, output_dir)
        create_alpha_vs_diffusion_plot(results, output_dir)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
