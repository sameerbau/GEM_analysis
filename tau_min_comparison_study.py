#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tau_min_comparison_study.py

This script compares MSD fitting results using different minimum tau values:
- tau_min = 1: Fit from first lag point (current approach)
- tau_min = 3: Fit from third lag point (recommended to avoid motion blur/artifacts)

The comparison helps determine whether excluding the first 2 lag points
significantly affects diffusion coefficient estimates.

Theory:
- Short lag points may be affected by motion blur during camera exposure
- Localization error dominates at very short time lags
- Literature suggests starting from tau=3-5 for robust estimates

Input:
- Processed trajectory data (.pkl files from 1Traj_load_v1.py)

Output:
- Side-by-side comparison of D values
- Statistical analysis of differences
- Diagnostic plots showing effect of tau_min
- CSV report with per-trajectory comparison

Usage:
python tau_min_comparison_study.py
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
from datetime import datetime

# Global parameters (matching v4)
# =====================================
DT = 0.1
CONVERSION = 0.094
TAU_FRACTION = 0.25
MIN_FIT_POINTS = 5
N_BOOTSTRAP = 1000

# Validation thresholds
D_MIN = 0.0001  # μm²/s
D_MAX = 50.0    # μm²/s
R_SQUARED_MIN = 0.7
# =====================================

def linear_msd(t, D, offset):
    """Linear MSD model: MSD = 4*D*t + offset"""
    return 4 * D * t + offset

def calculate_max_tau_index(trajectory_length, tau_fraction=TAU_FRACTION):
    """Calculate maximum tau index for fitting."""
    max_tau_index = int(trajectory_length * tau_fraction)
    warning = None
    if max_tau_index < MIN_FIT_POINTS:
        warning = f"Trajectory too short ({trajectory_length} frames)"
        max_tau_index = min(trajectory_length - 1, MIN_FIT_POINTS)
    return max_tau_index, warning

def fit_msd_single(time_data, msd_data, trajectory_length, tau_min=1, n_bootstrap=N_BOOTSTRAP):
    """
    Fit MSD with specified tau_min value.

    Args:
        time_data: Time lag values
        msd_data: MSD values
        trajectory_length: Number of points in original trajectory
        tau_min: Minimum tau index to start fitting (1-indexed)
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dictionary with fit results
    """
    # Calculate max tau
    max_tau_index, tau_warning = calculate_max_tau_index(trajectory_length)

    # Extract data - START FROM tau_min (convert to 0-indexed)
    start_idx = tau_min - 1  # tau_min=1 -> idx=0, tau_min=3 -> idx=2
    t_fit = time_data[start_idx:max_tau_index]
    msd_fit = msd_data[start_idx:max_tau_index]

    # Filter NaN
    valid_indices = ~np.isnan(msd_fit)
    t_fit = t_fit[valid_indices]
    msd_fit = msd_fit[valid_indices]

    # Check minimum points
    if len(t_fit) < 3:
        return {
            'D': np.nan, 'offset': np.nan, 'sigma_loc': np.nan,
            'D_CI_low': np.nan, 'D_CI_high': np.nan,
            'sigma_loc_CI_low': np.nan, 'sigma_loc_CI_high': np.nan,
            'r_squared': np.nan, 't_fit': t_fit, 'msd_fit': msd_fit,
            'fit_values': np.nan, 'n_fit_points': len(t_fit),
            'max_tau_used': np.nan, 'tau_min_used': tau_min,
            'fit_failed': True, 'failure_reason': 'Insufficient data'
        }

    try:
        # Main fit
        popt, pcov = curve_fit(linear_msd, t_fit, msd_fit)
        D_main, offset_main = popt

        # Calculate sigma_loc
        if offset_main > 0:
            sigma_loc_main = np.sqrt(offset_main / 4.0)
        else:
            sigma_loc_main = 0.0

        # Fit quality
        fit_values = linear_msd(t_fit, D_main, offset_main)
        residuals = msd_fit - fit_values
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((msd_fit - np.mean(msd_fit))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Bootstrap
        D_bootstrap = []
        sigma_loc_bootstrap = []
        n_points = len(t_fit)

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_points, size=n_points, replace=True)
            t_boot = t_fit[indices]
            msd_boot = msd_fit[indices]

            try:
                popt_boot, _ = curve_fit(linear_msd, t_boot, msd_boot)
                D_boot, offset_boot = popt_boot
                D_bootstrap.append(D_boot)
                if offset_boot > 0:
                    sigma_loc_bootstrap.append(np.sqrt(offset_boot / 4.0))
                else:
                    sigma_loc_bootstrap.append(0.0)
            except:
                continue

        # Calculate CI
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
            'n_fit_points': len(t_fit),
            'max_tau_used': t_fit[-1] if len(t_fit) > 0 else np.nan,
            'tau_min_used': tau_min,
            'fit_failed': False
        }
    except Exception as e:
        return {
            'D': np.nan, 'offset': np.nan, 'sigma_loc': np.nan,
            'D_CI_low': np.nan, 'D_CI_high': np.nan,
            'sigma_loc_CI_low': np.nan, 'sigma_loc_CI_high': np.nan,
            'r_squared': np.nan, 't_fit': t_fit, 'msd_fit': msd_fit,
            'fit_values': np.nan, 'n_fit_points': len(t_fit),
            'max_tau_used': np.nan, 'tau_min_used': tau_min,
            'fit_failed': True, 'failure_reason': str(e)
        }

def compare_fits(processed_data, tau_min_values=[1, 3]):
    """
    Fit all trajectories with different tau_min values and compare.

    Args:
        processed_data: Dictionary with trajectory data
        tau_min_values: List of tau_min values to test

    Returns:
        Dictionary with comparison results
    """
    print(f"  Comparing tau_min values: {tau_min_values}")

    comparison_results = {
        'trajectories': [],
        'tau_min_values': tau_min_values
    }

    n_trajectories = len(processed_data['trajectories'])

    for i, trajectory in enumerate(processed_data['trajectories']):
        if i % 50 == 0:
            print(f"    Processing trajectory {i+1}/{n_trajectories}...")

        time_data = processed_data['time_data'][i]
        msd_data = processed_data['msd_data'][i]
        trajectory_length = len(trajectory['x'])

        # Fit with each tau_min value
        traj_comparison = {
            'id': trajectory['id'],
            'trajectory_length': trajectory_length,
            'fits': {}
        }

        for tau_min in tau_min_values:
            fit_result = fit_msd_single(time_data, msd_data, trajectory_length,
                                       tau_min=tau_min)
            traj_comparison['fits'][f'tau_min_{tau_min}'] = fit_result

        # Store full trajectory data for plotting
        traj_comparison['time_data'] = time_data
        traj_comparison['msd_data'] = msd_data
        traj_comparison['x'] = trajectory['x']
        traj_comparison['y'] = trajectory['y']

        comparison_results['trajectories'].append(traj_comparison)

    return comparison_results

def analyze_comparison(comparison_results):
    """
    Statistical analysis of the comparison between tau_min values.

    Returns:
        Dictionary with statistical metrics
    """
    tau_min_values = comparison_results['tau_min_values']

    # Extract D values for each tau_min
    D_values = {f'tau_min_{tm}': [] for tm in tau_min_values}
    sigma_loc_values = {f'tau_min_{tm}': [] for tm in tau_min_values}
    r_squared_values = {f'tau_min_{tm}': [] for tm in tau_min_values}

    # Per-trajectory comparisons
    per_traj_data = []

    for traj in comparison_results['trajectories']:
        traj_data = {'trajectory_id': traj['id'], 'length': traj['trajectory_length']}

        # Check if both fits succeeded
        all_valid = True
        for tm in tau_min_values:
            key = f'tau_min_{tm}'
            fit = traj['fits'][key]

            if not fit['fit_failed'] and not np.isnan(fit['D']):
                D_values[key].append(fit['D'])
                sigma_loc_values[key].append(fit['sigma_loc'] * 1000)  # nm
                r_squared_values[key].append(fit['r_squared'])

                traj_data[f'D_{key}'] = fit['D']
                traj_data[f'sigma_loc_nm_{key}'] = fit['sigma_loc'] * 1000
                traj_data[f'R2_{key}'] = fit['r_squared']
                traj_data[f'n_fit_points_{key}'] = fit['n_fit_points']
            else:
                all_valid = False
                traj_data[f'D_{key}'] = np.nan

        # Calculate differences (only if both valid)
        if all_valid and len(tau_min_values) == 2:
            D1 = traj_data[f'D_tau_min_{tau_min_values[0]}']
            D2 = traj_data[f'D_tau_min_{tau_min_values[1]}']

            traj_data['D_abs_diff'] = D2 - D1
            traj_data['D_rel_diff_percent'] = 100 * (D2 - D1) / D1 if D1 != 0 else np.nan

        per_traj_data.append(traj_data)

    # Statistical summary
    stats_summary = {}

    for tm in tau_min_values:
        key = f'tau_min_{tm}'
        D_arr = np.array(D_values[key])

        if len(D_arr) > 0:
            stats_summary[key] = {
                'n_valid': len(D_arr),
                'D_median': np.median(D_arr),
                'D_mean': np.mean(D_arr),
                'D_std': np.std(D_arr),
                'D_q25': np.percentile(D_arr, 25),
                'D_q75': np.percentile(D_arr, 75),
                'sigma_loc_median_nm': np.median(sigma_loc_values[key]),
                'r_squared_median': np.median(r_squared_values[key])
            }

    # Paired comparison
    if len(tau_min_values) == 2:
        key1 = f'tau_min_{tau_min_values[0]}'
        key2 = f'tau_min_{tau_min_values[1]}'

        # Get paired data (trajectories where both fits succeeded)
        paired_D1 = []
        paired_D2 = []

        for traj_data in per_traj_data:
            if not np.isnan(traj_data.get(f'D_{key1}', np.nan)) and \
               not np.isnan(traj_data.get(f'D_{key2}', np.nan)):
                paired_D1.append(traj_data[f'D_{key1}'])
                paired_D2.append(traj_data[f'D_{key2}'])

        paired_D1 = np.array(paired_D1)
        paired_D2 = np.array(paired_D2)

        if len(paired_D1) > 0:
            # Wilcoxon signed-rank test (paired non-parametric test)
            statistic, p_value = stats.wilcoxon(paired_D1, paired_D2)

            # Calculate differences
            abs_diff = paired_D2 - paired_D1
            rel_diff = 100 * (paired_D2 - paired_D1) / paired_D1

            stats_summary['paired_comparison'] = {
                'n_paired': len(paired_D1),
                'median_abs_diff': np.median(abs_diff),
                'mean_abs_diff': np.mean(abs_diff),
                'median_rel_diff_percent': np.median(rel_diff),
                'mean_rel_diff_percent': np.mean(rel_diff),
                'wilcoxon_statistic': statistic,
                'wilcoxon_p_value': p_value,
                'significant_at_0.05': p_value < 0.05,
                'percent_increased': 100 * np.sum(paired_D2 > paired_D1) / len(paired_D1),
                'percent_decreased': 100 * np.sum(paired_D2 < paired_D1) / len(paired_D1)
            }

    return {
        'stats_summary': stats_summary,
        'per_trajectory_data': per_traj_data,
        'D_values': D_values,
        'sigma_loc_values': sigma_loc_values,
        'r_squared_values': r_squared_values
    }

def create_comparison_plots(comparison_results, analysis_results, output_path, filename):
    """
    Create comprehensive comparison plots.
    """
    tau_min_values = comparison_results['tau_min_values']

    if len(tau_min_values) != 2:
        print("  Plotting currently supports comparing exactly 2 tau_min values")
        return

    tm1, tm2 = tau_min_values
    key1 = f'tau_min_{tm1}'
    key2 = f'tau_min_{tm2}'

    D1 = np.array(analysis_results['D_values'][key1])
    D2 = np.array(analysis_results['D_values'][key2])

    sigma1 = np.array(analysis_results['sigma_loc_values'][key1])
    sigma2 = np.array(analysis_results['sigma_loc_values'][key2])

    per_traj = analysis_results['per_trajectory_data']
    rel_diffs = [t['D_rel_diff_percent'] for t in per_traj if 'D_rel_diff_percent' in t and not np.isnan(t['D_rel_diff_percent'])]

    # Figure 1: Main comparison plots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Scatter D1 vs D2
    ax = axs[0, 0]
    ax.scatter(D1, D2, alpha=0.5, s=30)

    # 1:1 line
    d_min = min(D1.min(), D2.min())
    d_max = max(D1.max(), D2.max())
    ax.plot([d_min, d_max], [d_min, d_max], 'r--', linewidth=2, label='1:1 line')

    # Regression line
    if len(D1) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(D1, D2)
        x_line = np.array([d_min, d_max])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'g-', linewidth=2,
                label=f'Linear fit: slope={slope:.3f}, R²={r_value**2:.3f}')

    ax.set_xlabel(f'D with τ_min={tm1} (μm²/s)')
    ax.set_ylabel(f'D with τ_min={tm2} (μm²/s)')
    ax.set_title(f'Diffusion Coefficient Comparison\n(n={len(D1)} trajectories)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 2: Histogram of D values (overlaid)
    ax = axs[0, 1]
    bins = np.linspace(0, min(np.percentile(D1, 99), np.percentile(D2, 99)), 30)

    ax.hist(D1, bins=bins, alpha=0.5, color='blue', label=f'τ_min={tm1}')
    ax.hist(D2, bins=bins, alpha=0.5, color='red', label=f'τ_min={tm2}')

    ax.axvline(np.median(D1), color='blue', linestyle='--', linewidth=2,
              label=f'Median τ_min={tm1}: {np.median(D1):.6f}')
    ax.axvline(np.median(D2), color='red', linestyle='--', linewidth=2,
              label=f'Median τ_min={tm2}: {np.median(D2):.6f}')

    ax.set_xlabel('Diffusion coefficient (μm²/s)')
    ax.set_ylabel('Count')
    ax.set_title('D Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Relative difference histogram
    ax = axs[0, 2]
    if rel_diffs:
        ax.hist(rel_diffs, bins=30, color='purple', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
        ax.axvline(np.median(rel_diffs), color='green', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(rel_diffs):.1f}%')

        ax.set_xlabel(f'Relative change in D (%)\n[(τ_min={tm2} - τ_min={tm1}) / τ_min={tm1} × 100]')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of D Changes')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 4: Sigma_loc comparison
    ax = axs[1, 0]
    ax.scatter(sigma1, sigma2, alpha=0.5, s=30)

    s_min = min(sigma1.min(), sigma2.min())
    s_max = max(sigma1.max(), sigma2.max())
    ax.plot([s_min, s_max], [s_min, s_max], 'r--', linewidth=2, label='1:1 line')

    ax.set_xlabel(f'σ_loc with τ_min={tm1} (nm)')
    ax.set_ylabel(f'σ_loc with τ_min={tm2} (nm)')
    ax.set_title('Localization Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 5: Bland-Altman plot for D
    ax = axs[1, 1]
    paired_data = [(t[f'D_{key1}'], t[f'D_{key2}']) for t in per_traj
                   if not np.isnan(t.get(f'D_{key1}', np.nan)) and not np.isnan(t.get(f'D_{key2}', np.nan))]

    if paired_data:
        D1_paired = np.array([d[0] for d in paired_data])
        D2_paired = np.array([d[1] for d in paired_data])

        mean_D = (D1_paired + D2_paired) / 2
        diff_D = D2_paired - D1_paired

        ax.scatter(mean_D, diff_D, alpha=0.5, s=30)
        ax.axhline(0, color='red', linestyle='-', linewidth=2)
        ax.axhline(np.mean(diff_D), color='blue', linestyle='--', linewidth=2,
                  label=f'Mean diff: {np.mean(diff_D):.6f}')
        ax.axhline(np.mean(diff_D) + 1.96*np.std(diff_D), color='gray',
                  linestyle='--', linewidth=1, label='±1.96 SD')
        ax.axhline(np.mean(diff_D) - 1.96*np.std(diff_D), color='gray',
                  linestyle='--', linewidth=1)

        ax.set_xlabel('Mean D (μm²/s)')
        ax.set_ylabel(f'Difference (τ_min={tm2} - τ_min={tm1})')
        ax.set_title('Bland-Altman Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 6: Box plots
    ax = axs[1, 2]
    data_to_plot = [D1, D2]
    positions = [1, 2]

    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=False)

    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xticks(positions)
    ax.set_xticklabels([f'τ_min={tm1}', f'τ_min={tm2}'])
    ax.set_ylabel('Diffusion coefficient (μm²/s)')
    ax.set_title('D Distribution Box Plots')
    ax.grid(True, alpha=0.3, axis='y')

    # Add median values as text
    for i, (pos, d_arr) in enumerate(zip(positions, data_to_plot)):
        ax.text(pos, np.median(d_arr), f'{np.median(d_arr):.5f}',
               ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{filename}_tau_comparison_summary.png"), dpi=300)
    plt.close()

    print(f"  Summary plot saved")

    # Figure 2: Example MSD fits showing the difference
    # Select 6 representative trajectories
    trajs = comparison_results['trajectories']

    # Filter valid trajectories with both fits
    valid_trajs = []
    for traj in trajs:
        if not traj['fits'][key1]['fit_failed'] and not traj['fits'][key2]['fit_failed']:
            if not np.isnan(traj['fits'][key1]['D']) and not np.isnan(traj['fits'][key2]['D']):
                valid_trajs.append(traj)

    if len(valid_trajs) >= 6:
        # Select: 2 with largest increase, 2 with largest decrease, 2 middle
        changes = []
        for traj in valid_trajs:
            D_old = traj['fits'][key1]['D']
            D_new = traj['fits'][key2]['D']
            rel_change = 100 * (D_new - D_old) / D_old
            changes.append((traj, rel_change))

        changes.sort(key=lambda x: x[1])

        selected = [
            changes[0][0],  # Largest decrease
            changes[1][0],  # Second largest decrease
            changes[len(changes)//2][0],  # Middle
            changes[len(changes)//2 + 1][0],  # Middle
            changes[-2][0],  # Second largest increase
            changes[-1][0]   # Largest increase
        ]

        fig, axs = plt.subplots(3, 2, figsize=(16, 18))
        axs = axs.flatten()

        for idx, traj in enumerate(selected):
            ax = axs[idx]

            # Plot MSD data
            ax.plot(traj['time_data'], traj['msd_data'], 'o',
                   color='gray', alpha=0.3, markersize=4, label='MSD data')

            # Plot fit with tau_min=1
            fit1 = traj['fits'][key1]
            t_extended = np.linspace(0, traj['time_data'][-1], 100)
            msd_fit1 = linear_msd(t_extended, fit1['D'], fit1['offset'])

            ax.plot(t_extended, msd_fit1, '-', color='blue', linewidth=2,
                   label=f'τ_min={tm1}: D={fit1["D"]:.5f} μm²/s')
            ax.plot(fit1['t_fit'], fit1['msd_fit'], 'o', color='blue',
                   markersize=6, alpha=0.7)

            # Plot fit with tau_min=3
            fit2 = traj['fits'][key2]
            msd_fit2 = linear_msd(t_extended, fit2['D'], fit2['offset'])

            ax.plot(t_extended, msd_fit2, '-', color='red', linewidth=2,
                   label=f'τ_min={tm2}: D={fit2["D"]:.5f} μm²/s')
            ax.plot(fit2['t_fit'], fit2['msd_fit'], 's', color='red',
                   markersize=6, alpha=0.7)

            # Calculate change
            rel_change = 100 * (fit2['D'] - fit1['D']) / fit1['D']

            ax.set_xlabel('Time lag (s)')
            ax.set_ylabel('MSD (μm²)')
            ax.set_title(f"Trajectory {int(traj['id'])} (Length={traj['trajectory_length']})\n"
                        f"Change: {rel_change:+.1f}% | "
                        f"σ_loc: {fit1['sigma_loc']*1000:.1f}→{fit2['sigma_loc']*1000:.1f} nm",
                        fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{filename}_example_msd_fits.png"), dpi=300)
        plt.close()

        print(f"  Example MSD fits saved")

def generate_report(analysis_results, output_path, filename):
    """Generate text report and CSV file."""
    stats = analysis_results['stats_summary']

    # Text report
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("TAU_MIN COMPARISON STUDY REPORT")
    report_lines.append("="*70)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Individual tau_min results
    for key, val in stats.items():
        if key != 'paired_comparison':
            report_lines.append(f"\n{key.upper()}:")
            report_lines.append(f"  Number of valid fits: {val['n_valid']}")
            report_lines.append(f"  D median: {val['D_median']:.6f} μm²/s")
            report_lines.append(f"  D mean ± std: {val['D_mean']:.6f} ± {val['D_std']:.6f} μm²/s")
            report_lines.append(f"  D IQR: [{val['D_q25']:.6f}, {val['D_q75']:.6f}] μm²/s")
            report_lines.append(f"  σ_loc median: {val['sigma_loc_median_nm']:.1f} nm")
            report_lines.append(f"  R² median: {val['r_squared_median']:.3f}")

    # Paired comparison
    if 'paired_comparison' in stats:
        pc = stats['paired_comparison']
        report_lines.append("\n" + "="*70)
        report_lines.append("PAIRED COMPARISON (trajectories with both fits valid):")
        report_lines.append("="*70)
        report_lines.append(f"  Number of paired measurements: {pc['n_paired']}")
        report_lines.append(f"\n  Absolute difference (τ_min=3 - τ_min=1):")
        report_lines.append(f"    Median: {pc['median_abs_diff']:.6f} μm²/s")
        report_lines.append(f"    Mean: {pc['mean_abs_diff']:.6f} μm²/s")
        report_lines.append(f"\n  Relative difference (%):")
        report_lines.append(f"    Median: {pc['median_rel_diff_percent']:.2f}%")
        report_lines.append(f"    Mean: {pc['mean_rel_diff_percent']:.2f}%")
        report_lines.append(f"\n  Direction of change:")
        report_lines.append(f"    Increased: {pc['percent_increased']:.1f}%")
        report_lines.append(f"    Decreased: {pc['percent_decreased']:.1f}%")
        report_lines.append(f"\n  Statistical test (Wilcoxon signed-rank):")
        report_lines.append(f"    Test statistic: {pc['wilcoxon_statistic']:.2f}")
        report_lines.append(f"    p-value: {pc['wilcoxon_p_value']:.4e}")
        report_lines.append(f"    Significant at α=0.05: {'YES' if pc['significant_at_0.05'] else 'NO'}")

        if pc['significant_at_0.05']:
            report_lines.append("\n  ⚠ The difference is statistically significant!")
            report_lines.append("    → Excluding first 2 lag points DOES affect D estimates")
        else:
            report_lines.append("\n  ✓ The difference is NOT statistically significant")
            report_lines.append("    → Excluding first 2 lag points has minimal effect")

    report_lines.append("\n" + "="*70)

    # Save text report
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_file = os.path.join(output_path, f"{filename}_comparison_report.txt")
    with open(report_file, 'w') as f:
        f.write(report_text)

    print(f"\nReport saved to: {report_file}")

    # Save per-trajectory CSV
    df = pd.DataFrame(analysis_results['per_trajectory_data'])
    csv_file = os.path.join(output_path, f"{filename}_per_trajectory_comparison.csv")
    df.to_csv(csv_file, index=False)

    print(f"Per-trajectory data saved to: {csv_file}")

def main():
    """Main comparison function."""
    print("="*70)
    print("TAU_MIN COMPARISON STUDY")
    print("="*70)
    print("\nThis script compares MSD fitting with different tau_min values:")
    print("  - tau_min = 1: Include first lag point (current approach)")
    print("  - tau_min = 3: Exclude first 2 lag points (recommended)")
    print()

    # Get input directory
    input_dir = input("Enter directory with processed trajectories (press Enter for 'processed_trajectories'): ")
    if input_dir == "":
        input_dir = os.path.join(os.getcwd(), "processed_trajectories")

    if not os.path.isdir(input_dir):
        print(f"Error: Directory {input_dir} does not exist")
        return

    # Find files
    file_paths = glob.glob(os.path.join(input_dir, "tracked_*.pkl"))
    if not file_paths:
        print(f"No processed trajectory files found in {input_dir}")
        return

    print(f"\nFound {len(file_paths)} file(s) to process")

    # Create output directory
    output_dir = os.path.join(os.path.dirname(input_dir), "tau_min_comparison_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Process each file
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]

        print(f"\n{'='*70}")
        print(f"Processing: {filename}")
        print(f"{'='*70}")

        # Load data
        try:
            with open(file_path, 'rb') as f:
                processed_data = pickle.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        print(f"Loaded {len(processed_data['trajectories'])} trajectories")

        # Compare fits
        comparison_results = compare_fits(processed_data, tau_min_values=[1, 3])

        # Analyze comparison
        print("\n  Analyzing differences...")
        analysis_results = analyze_comparison(comparison_results)

        # Generate report
        generate_report(analysis_results, output_dir, base_name)

        # Create plots
        print("\n  Creating comparison plots...")
        create_comparison_plots(comparison_results, analysis_results, output_dir, base_name)

        # Save full results
        results_file = os.path.join(output_dir, f"{base_name}_full_comparison.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump({
                'comparison_results': comparison_results,
                'analysis_results': analysis_results
            }, f)
        print(f"\n  Full results saved to: {results_file}")

    print("\n" + "="*70)
    print("COMPARISON STUDY COMPLETE")
    print(f"All results saved in: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
