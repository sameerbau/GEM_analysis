#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3 alpha_fitting_method_comparison.py

Compare log-log linear regression vs nonlinear power-law fitting for alpha extraction.

This script analyzes trajectory data using both methods to quantify differences in:
1. Alpha values
2. Diffusion coefficients
3. Fit quality (R²)
4. Computational time
5. Robustness to noise

Theory:
-------
LOG-LOG LINEAR REGRESSION (Standard method):
- Transforms MSD and time to log-space
- Performs linear regression: log(MSD) = log(4D) + α·log(t)
- Slope = α, Intercept = log(4D)
- Pros: Numerically stable, fast, standard approach
- Cons: Can be biased by unequal weighting in log-space

NONLINEAR POWER-LAW FITTING:
- Direct fit to MSD = 4D·t^α
- Uses nonlinear optimization (curve_fit)
- Pros: Direct physical model, can handle non-linearities
- Cons: Slower, requires good initial guesses, less stable

Input:
- Single roi_trajectory_data_*.pkl file with trajectory data

Output:
- Comparison plots showing differences between methods
- CSV with per-trajectory comparison
- Statistical summary of method agreement

Usage:
python 3 alpha_fitting_method_comparison.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from scipy import stats
from scipy.optimize import curve_fit

# Global parameters
# =====================================
DT = 0.1  # Time step (seconds)
MIN_TRAJECTORY_LENGTH = 10
MAX_LAG_FRACTION = 0.3
MIN_R_SQUARED = 0.7
ALPHA_MIN = 0.5
ALPHA_MAX = 1.5
MIN_FIT_POINTS = 4
N_BOOTSTRAP = 1000
OUTPUT_SUBFOLDER = 'alpha_method_comparison'
# =====================================


def load_roi_data(file_path):
    """Load ROI-assigned trajectory data from pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded: {os.path.basename(file_path)}")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def calculate_msd(x, y, dt, max_lag=None):
    """Calculate mean squared displacement for a trajectory."""
    n = len(x)

    if max_lag is None:
        max_lag = n - 1
    else:
        max_lag = min(max_lag, n - 1)

    msd = np.zeros(max_lag)

    for lag in range(1, max_lag + 1):
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        dr2 = dx**2 + dy**2
        msd[lag - 1] = np.mean(dr2)

    time_lags = np.arange(1, max_lag + 1) * dt

    return msd, time_lags


def calculate_adaptive_max_lag(trajectory_length):
    """Adaptively calculate max lag based on trajectory length."""
    if trajectory_length < 15:
        max_lag_fraction = 0.20
    else:
        max_lag_fraction = MAX_LAG_FRACTION

    max_lag_points = max(int(trajectory_length * max_lag_fraction), MIN_FIT_POINTS)

    return min(max_lag_points, trajectory_length - 1)


def power_law_msd(t, D, alpha):
    """Power law MSD model: MSD = 4*D*t^alpha"""
    return 4 * D * np.power(t, alpha)


def fit_alpha_loglog(time_data, msd_data, trajectory_length):
    """
    Fit alpha using log-log linear regression.

    Returns:
        Dictionary with fitting results including timing
    """
    start_time = time.time()

    max_points = calculate_adaptive_max_lag(trajectory_length)
    t_fit = time_data[:max_points]
    msd_fit = msd_data[:max_points]

    # Filter valid data
    valid = ~np.isnan(msd_fit) & (msd_fit > 0) & (t_fit > 0)
    t_fit = t_fit[valid]
    msd_fit = msd_fit[valid]

    if len(t_fit) < MIN_FIT_POINTS:
        return {
            'method': 'log-log',
            'alpha': np.nan,
            'D': np.nan,
            'r_squared': np.nan,
            'n_points': len(t_fit),
            'fit_time': time.time() - start_time,
            'success': False
        }

    try:
        # Log-log regression
        log_t = np.log(t_fit)
        log_msd = np.log(msd_fit)

        slope, intercept, r_value, _, _ = stats.linregress(log_t, log_msd)

        alpha = slope
        D = np.exp(intercept) / 4
        r_squared = r_value ** 2

        fit_time = time.time() - start_time

        return {
            'method': 'log-log',
            'alpha': alpha,
            'D': D,
            'r_squared': r_squared,
            'n_points': len(t_fit),
            'fit_time': fit_time,
            'success': True
        }
    except Exception as e:
        return {
            'method': 'log-log',
            'alpha': np.nan,
            'D': np.nan,
            'r_squared': np.nan,
            'n_points': len(t_fit),
            'fit_time': time.time() - start_time,
            'success': False,
            'error': str(e)
        }


def fit_alpha_powerlaw(time_data, msd_data, trajectory_length):
    """
    Fit alpha using nonlinear power-law fitting.

    Returns:
        Dictionary with fitting results including timing
    """
    start_time = time.time()

    max_points = calculate_adaptive_max_lag(trajectory_length)
    t_fit = time_data[:max_points]
    msd_fit = msd_data[:max_points]

    # Filter valid data
    valid = ~np.isnan(msd_fit) & (msd_fit > 0) & (t_fit > 0)
    t_fit = t_fit[valid]
    msd_fit = msd_fit[valid]

    if len(t_fit) < MIN_FIT_POINTS:
        return {
            'method': 'power-law',
            'alpha': np.nan,
            'D': np.nan,
            'r_squared': np.nan,
            'n_points': len(t_fit),
            'fit_time': time.time() - start_time,
            'success': False
        }

    try:
        # Initial guess from log-log regression
        log_t = np.log(t_fit)
        log_msd = np.log(msd_fit)
        slope, intercept, _, _, _ = stats.linregress(log_t, log_msd)
        D_guess = np.exp(intercept) / 4
        alpha_guess = slope

        # Nonlinear fit
        popt, pcov = curve_fit(
            power_law_msd,
            t_fit,
            msd_fit,
            p0=[D_guess, alpha_guess],
            bounds=([0, 0], [np.inf, 3]),
            maxfev=5000
        )

        D, alpha = popt

        # Calculate R²
        msd_pred = power_law_msd(t_fit, D, alpha)
        ss_res = np.sum((msd_fit - msd_pred)**2)
        ss_tot = np.sum((msd_fit - np.mean(msd_fit))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        fit_time = time.time() - start_time

        return {
            'method': 'power-law',
            'alpha': alpha,
            'D': D,
            'r_squared': r_squared,
            'n_points': len(t_fit),
            'fit_time': fit_time,
            'success': True
        }
    except Exception as e:
        return {
            'method': 'power-law',
            'alpha': np.nan,
            'D': np.nan,
            'r_squared': np.nan,
            'n_points': len(t_fit),
            'fit_time': time.time() - start_time,
            'success': False,
            'error': str(e)
        }


def compare_methods_single_trajectory(traj):
    """
    Compare both fitting methods on a single trajectory.

    Returns:
        Dictionary with comparison results
    """
    traj_length = len(traj['x'])

    # Skip if too short
    if traj_length < MIN_TRAJECTORY_LENGTH:
        return None

    # Calculate MSD
    msd, time_lags = calculate_msd(traj['x'], traj['y'], DT)

    # Fit using both methods
    loglog_result = fit_alpha_loglog(time_lags, msd, traj_length)
    powerlaw_result = fit_alpha_powerlaw(time_lags, msd, traj_length)

    # Combine results
    result = {
        'trajectory_id': traj.get('id', 'unknown'),
        'trajectory_length': traj_length,
        # Log-log results
        'alpha_loglog': loglog_result['alpha'],
        'D_loglog': loglog_result['D'],
        'r2_loglog': loglog_result['r_squared'],
        'time_loglog': loglog_result['fit_time'],
        'success_loglog': loglog_result['success'],
        # Power-law results
        'alpha_powerlaw': powerlaw_result['alpha'],
        'D_powerlaw': powerlaw_result['D'],
        'r2_powerlaw': powerlaw_result['r_squared'],
        'time_powerlaw': powerlaw_result['fit_time'],
        'success_powerlaw': powerlaw_result['success'],
        # Differences
        'alpha_diff': np.nan,
        'alpha_rel_diff_pct': np.nan,
        'D_diff': np.nan,
        'D_rel_diff_pct': np.nan,
        'r2_diff': np.nan
    }

    # Calculate differences if both methods succeeded
    if loglog_result['success'] and powerlaw_result['success']:
        result['alpha_diff'] = powerlaw_result['alpha'] - loglog_result['alpha']
        result['alpha_rel_diff_pct'] = (result['alpha_diff'] / loglog_result['alpha']) * 100

        result['D_diff'] = powerlaw_result['D'] - loglog_result['D']
        result['D_rel_diff_pct'] = (result['D_diff'] / loglog_result['D']) * 100

        result['r2_diff'] = powerlaw_result['r_squared'] - loglog_result['r_squared']

    return result


def analyze_file(roi_data):
    """
    Analyze all trajectories in a file, comparing both methods.

    Returns:
        List of comparison results
    """
    all_results = []

    # Process all trajectories from all ROIs
    for roi_id, trajectories in roi_data['roi_trajectories'].items():
        for traj in trajectories:
            result = compare_methods_single_trajectory(traj)
            if result is not None:
                result['roi_id'] = roi_id
                all_results.append(result)

    return all_results


def create_comparison_plots(results, output_dir):
    """
    Create comprehensive comparison plots.

    Args:
        results: List of comparison dictionaries
        output_dir: Directory to save plots
    """
    # Filter valid results where both methods succeeded
    df = pd.DataFrame(results)
    valid = df[df['success_loglog'] & df['success_powerlaw']].copy()

    if len(valid) == 0:
        print("No valid comparisons to plot")
        return

    print(f"\nCreating comparison plots for {len(valid)} trajectories...")

    # Create comprehensive comparison figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # 1. Alpha comparison scatter
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(valid['alpha_loglog'], valid['alpha_powerlaw'], alpha=0.5, s=20)
    lims = [
        max(valid['alpha_loglog'].min(), valid['alpha_powerlaw'].min()) - 0.1,
        min(valid['alpha_loglog'].max(), valid['alpha_powerlaw'].max()) + 0.1
    ]
    ax1.plot(lims, lims, 'r--', lw=2, label='y=x')
    ax1.set_xlabel('Alpha (log-log)', fontweight='bold')
    ax1.set_ylabel('Alpha (power-law)', fontweight='bold')
    ax1.set_title('Alpha Value Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Alpha relative difference histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(valid['alpha_rel_diff_pct'], bins=30, alpha=0.7, edgecolor='black')
    median_diff = valid['alpha_rel_diff_pct'].median()
    ax2.axvline(median_diff, color='r', linestyle='--', lw=2,
               label=f'Median: {median_diff:.2f}%')
    ax2.set_xlabel('Relative Difference (%)', fontweight='bold')
    ax2.set_ylabel('Count')
    ax2.set_title('(α_powerlaw - α_loglog) / α_loglog × 100')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. D comparison scatter
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(valid['D_loglog'], valid['D_powerlaw'], alpha=0.5, s=20)
    lims = [
        max(valid['D_loglog'].min(), valid['D_powerlaw'].min()),
        min(valid['D_loglog'].max(), valid['D_powerlaw'].max())
    ]
    ax3.plot(lims, lims, 'r--', lw=2, label='y=x')
    ax3.set_xlabel('D (log-log) [μm²/s]', fontweight='bold')
    ax3.set_ylabel('D (power-law) [μm²/s]', fontweight='bold')
    ax3.set_title('Diffusion Coefficient Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')

    # 4. R² comparison
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.scatter(valid['r2_loglog'], valid['r2_powerlaw'], alpha=0.5, s=20)
    ax4.plot([0, 1], [0, 1], 'r--', lw=2, label='y=x')
    ax4.set_xlabel('R² (log-log)', fontweight='bold')
    ax4.set_ylabel('R² (power-law)', fontweight='bold')
    ax4.set_title('Fit Quality Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. R² difference vs alpha_loglog
    ax5 = fig.add_subplot(gs[1, 0])
    scatter = ax5.scatter(valid['alpha_loglog'], valid['r2_diff'],
                         c=valid['trajectory_length'], cmap='viridis',
                         alpha=0.6, s=20)
    ax5.axhline(0, color='r', linestyle='--', lw=1)
    ax5.set_xlabel('Alpha (log-log)', fontweight='bold')
    ax5.set_ylabel('R² Difference (PL - LL)', fontweight='bold')
    ax5.set_title('Fit Quality Difference vs Alpha')
    plt.colorbar(scatter, ax=ax5, label='Trajectory Length')
    ax5.grid(True, alpha=0.3)

    # 6. Computation time comparison
    ax6 = fig.add_subplot(gs[1, 1])
    time_data = [valid['time_loglog']*1000, valid['time_powerlaw']*1000]
    bp = ax6.boxplot(time_data, labels=['Log-Log', 'Power-Law'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    ax6.set_ylabel('Computation Time (ms)', fontweight='bold')
    ax6.set_title('Computational Efficiency')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_yscale('log')

    # 7. Success rate
    ax7 = fig.add_subplot(gs[1, 2])
    total = len(df)
    success_data = {
        'Log-Log': df['success_loglog'].sum() / total * 100,
        'Power-Law': df['success_powerlaw'].sum() / total * 100,
        'Both': len(valid) / total * 100
    }
    bars = ax7.bar(success_data.keys(), success_data.values(),
                   color=['steelblue', 'coral', 'green'], alpha=0.7)
    ax7.set_ylabel('Success Rate (%)', fontweight='bold')
    ax7.set_title('Method Success Rates')
    ax7.set_ylim([0, 105])
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Bland-Altman plot for alpha
    ax8 = fig.add_subplot(gs[1, 3])
    mean_alpha = (valid['alpha_loglog'] + valid['alpha_powerlaw']) / 2
    diff_alpha = valid['alpha_powerlaw'] - valid['alpha_loglog']
    ax8.scatter(mean_alpha, diff_alpha, alpha=0.5, s=20)
    ax8.axhline(diff_alpha.mean(), color='r', linestyle='-', lw=2,
               label=f'Mean: {diff_alpha.mean():.4f}')
    ax8.axhline(diff_alpha.mean() + 1.96*diff_alpha.std(), color='r',
               linestyle='--', lw=1, label='±1.96 SD')
    ax8.axhline(diff_alpha.mean() - 1.96*diff_alpha.std(), color='r',
               linestyle='--', lw=1)
    ax8.set_xlabel('Mean Alpha', fontweight='bold')
    ax8.set_ylabel('Difference (PL - LL)', fontweight='bold')
    ax8.set_title('Bland-Altman Plot: Alpha')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Correlation matrix
    ax9 = fig.add_subplot(gs[2, 0:2])
    corr_vars = ['alpha_loglog', 'alpha_powerlaw', 'D_loglog', 'D_powerlaw',
                 'r2_loglog', 'r2_powerlaw']
    corr_matrix = valid[corr_vars].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
               center=0, ax=ax9, cbar_kws={'label': 'Correlation'})
    ax9.set_title('Parameter Correlation Matrix')

    # 10. Example trajectory fits
    ax10 = fig.add_subplot(gs[2, 2:4])
    # Pick a representative trajectory
    mid_idx = len(valid) // 2
    example = valid.iloc[mid_idx]

    # Reconstruct MSD for this trajectory (simplified - just for visualization)
    ax10.text(0.5, 0.5, f"Comparison Summary\n\n"
             f"Total trajectories analyzed: {total}\n"
             f"Both methods succeeded: {len(valid)} ({len(valid)/total*100:.1f}%)\n\n"
             f"Median alpha difference: {median_diff:.2f}%\n"
             f"Mean alpha difference: {diff_alpha.mean():.4f} ± {diff_alpha.std():.4f}\n"
             f"Correlation (alpha): {valid[['alpha_loglog', 'alpha_powerlaw']].corr().iloc[0,1]:.4f}\n\n"
             f"Median time (log-log): {valid['time_loglog'].median()*1000:.2f} ms\n"
             f"Median time (power-law): {valid['time_powerlaw'].median()*1000:.2f} ms\n"
             f"Speed-up factor: {valid['time_powerlaw'].median()/valid['time_loglog'].median():.1f}x slower",
             ha='center', va='center', fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax10.axis('off')

    plt.suptitle('Alpha Fitting Method Comparison: Log-Log vs Power-Law',
                fontsize=16, fontweight='bold', y=0.995)

    # Save
    output_path = os.path.join(output_dir, 'alpha_method_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison plot saved to: {output_path}")


def print_comparison_summary(results):
    """Print detailed comparison statistics."""
    df = pd.DataFrame(results)
    valid = df[df['success_loglog'] & df['success_powerlaw']].copy()

    print("\n" + "="*70)
    print("METHOD COMPARISON SUMMARY")
    print("="*70)

    print(f"\nTotal trajectories analyzed: {len(df)}")
    print(f"Log-log succeeded:  {df['success_loglog'].sum()} ({df['success_loglog'].sum()/len(df)*100:.1f}%)")
    print(f"Power-law succeeded: {df['success_powerlaw'].sum()} ({df['success_powerlaw'].sum()/len(df)*100:.1f}%)")
    print(f"Both succeeded:     {len(valid)} ({len(valid)/len(df)*100:.1f}%)")

    if len(valid) > 0:
        print("\n" + "-"*70)
        print("ALPHA COMPARISON (both methods succeeded)")
        print("-"*70)
        print(f"Log-log   - Mean: {valid['alpha_loglog'].mean():.4f} ± {valid['alpha_loglog'].std():.4f}")
        print(f"Power-law - Mean: {valid['alpha_powerlaw'].mean():.4f} ± {valid['alpha_powerlaw'].std():.4f}")
        print(f"\nDifference (PL - LL):")
        print(f"  Mean:   {valid['alpha_diff'].mean():.4f} ± {valid['alpha_diff'].std():.4f}")
        print(f"  Median: {valid['alpha_diff'].median():.4f}")
        print(f"  Relative: {valid['alpha_rel_diff_pct'].median():.2f}% (median)")
        print(f"\nCorrelation: {valid[['alpha_loglog', 'alpha_powerlaw']].corr().iloc[0,1]:.4f}")

        print("\n" + "-"*70)
        print("COMPUTATIONAL TIME")
        print("-"*70)
        print(f"Log-log   - Median: {valid['time_loglog'].median()*1000:.2f} ms")
        print(f"Power-law - Median: {valid['time_powerlaw'].median()*1000:.2f} ms")
        print(f"Speed factor: Power-law is {valid['time_powerlaw'].median()/valid['time_loglog'].median():.1f}x slower")

        print("\n" + "-"*70)
        print("FIT QUALITY (R²)")
        print("-"*70)
        print(f"Log-log   - Mean: {valid['r2_loglog'].mean():.4f} ± {valid['r2_loglog'].std():.4f}")
        print(f"Power-law - Mean: {valid['r2_powerlaw'].mean():.4f} ± {valid['r2_powerlaw'].std():.4f}")
        print(f"Difference: {valid['r2_diff'].mean():.4f} ± {valid['r2_diff'].std():.4f}")

        print("\n" + "-"*70)
        print("RECOMMENDATION")
        print("-"*70)

        median_diff_pct = abs(valid['alpha_rel_diff_pct'].median())
        correlation = valid[['alpha_loglog', 'alpha_powerlaw']].corr().iloc[0,1]

        if median_diff_pct < 5 and correlation > 0.95:
            print("✓ Methods show excellent agreement (< 5% difference, r > 0.95)")
            print("  → Use LOG-LOG for routine analysis (faster, more stable)")
            print("  → Use POWER-LAW for validation or special cases")
        elif median_diff_pct < 10 and correlation > 0.90:
            print("⚠ Methods show good agreement (< 10% difference, r > 0.90)")
            print("  → Both methods are acceptable")
            print("  → Consider trajectory-specific quality metrics for selection")
        else:
            print("✗ Methods show significant differences (> 10% or r < 0.90)")
            print("  → Investigate data quality and noise levels")
            print("  → May indicate complex diffusion behavior")
            print("  → Use caution when interpreting alpha values")

    print("="*70)


def export_comparison_csv(results, output_dir):
    """Export detailed comparison results to CSV."""
    df = pd.DataFrame(results)

    # Main results
    csv_path = os.path.join(output_dir, 'alpha_method_comparison_detailed.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed comparison exported to: {csv_path}")

    # Summary statistics
    valid = df[df['success_loglog'] & df['success_powerlaw']].copy()
    if len(valid) > 0:
        summary = {
            'Metric': ['N_total', 'N_both_succeeded', 'Success_rate_pct',
                      'Alpha_LL_mean', 'Alpha_LL_std',
                      'Alpha_PL_mean', 'Alpha_PL_std',
                      'Alpha_diff_mean', 'Alpha_diff_std', 'Alpha_diff_median',
                      'Alpha_rel_diff_pct_median',
                      'Alpha_correlation',
                      'D_LL_mean', 'D_PL_mean',
                      'R2_LL_mean', 'R2_PL_mean',
                      'Time_LL_median_ms', 'Time_PL_median_ms', 'Time_ratio'],
            'Value': [
                len(df),
                len(valid),
                len(valid)/len(df)*100,
                valid['alpha_loglog'].mean(),
                valid['alpha_loglog'].std(),
                valid['alpha_powerlaw'].mean(),
                valid['alpha_powerlaw'].std(),
                valid['alpha_diff'].mean(),
                valid['alpha_diff'].std(),
                valid['alpha_diff'].median(),
                valid['alpha_rel_diff_pct'].median(),
                valid[['alpha_loglog', 'alpha_powerlaw']].corr().iloc[0,1],
                valid['D_loglog'].mean(),
                valid['D_powerlaw'].mean(),
                valid['r2_loglog'].mean(),
                valid['r2_powerlaw'].mean(),
                valid['time_loglog'].median()*1000,
                valid['time_powerlaw'].median()*1000,
                valid['time_powerlaw'].median()/valid['time_loglog'].median()
            ]
        }

        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(output_dir, 'alpha_method_comparison_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary statistics exported to: {summary_path}")


def main():
    """Main function to run method comparison."""
    print("\n" + "="*70)
    print("ALPHA FITTING METHOD COMPARISON")
    print("Log-Log Linear Regression vs Nonlinear Power-Law Fitting")
    print("="*70)

    # Ask for input file
    roi_data_file = input("\nEnter path to roi_trajectory_data pkl file: ")

    if not os.path.isfile(roi_data_file):
        print(f"Error: {roi_data_file} is not a valid file")
        return

    # Load data
    roi_data = load_roi_data(roi_data_file)
    if roi_data is None:
        print("Failed to load data. Exiting.")
        return

    # Analyze all trajectories
    print("\nComparing fitting methods on all trajectories...")
    print(f"Parameters:")
    print(f"  Minimum trajectory length: {MIN_TRAJECTORY_LENGTH} frames")
    print(f"  Adaptive lag selection: 20-30% of trajectory length")
    print(f"  Alpha range: [{ALPHA_MIN}, {ALPHA_MAX}]")

    results = analyze_file(roi_data)

    if not results:
        print("\nNo valid results. Check trajectory lengths.")
        return

    # Create output directory
    output_dir = os.path.join(os.path.dirname(roi_data_file), OUTPUT_SUBFOLDER)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput will be saved to: {output_dir}")

    # Print summary
    print_comparison_summary(results)

    # Export CSV
    export_comparison_csv(results, output_dir)

    # Create plots
    create_comparison_plots(results, output_dir)

    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
