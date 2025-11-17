#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diffusion_analyzer_v4_validation.py

VERSION 4: Adds comprehensive validation checks

This script analyzes processed trajectory data with automated quality control
and validation checks to flag suspicious measurements.

NEW IN V4 (builds on V3):
- Localization error dominance check (σ²_loc vs MSD)
- Diffusion coefficient range validation (0.0001 - 50 μm²/s)
- Fit quality assessment (R² threshold)
- Bootstrap CI width check (relative uncertainty)
- Comprehensive quality flags and warnings
- Quality report generation

Theory-based validation:
1. If 4σ²_loc > 0.5 * MSD(τ_first), localization error dominates
2. Cytoplasmic D typically 0.001 - 10 μm²/s
3. Good fits should have R² > 0.7
4. Bootstrap CI width / D < 0.5 indicates reasonable uncertainty

Input:
- Processed trajectory data (.pkl files)

Output:
- Analyzed data with quality flags
- Quality validation report
- Diagnostic plots highlighting issues

Usage:
python diffusion_analyzer_v4_validation.py
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

# Global parameters
# =====================================
DT = 0.1
CONVERSION = 0.094
TAU_FRACTION = 0.25
MIN_FIT_POINTS = 5
N_BOOTSTRAP = 1000

# Validation thresholds
D_MIN = 0.0001  # μm²/s - minimum reasonable D
D_MAX = 50.0    # μm²/s - maximum reasonable D
R_SQUARED_MIN = 0.7  # Minimum acceptable R²
LOC_ERROR_THRESHOLD = 0.5  # Fraction: if 4σ²_loc > threshold * MSD, flag
RELATIVE_CI_WIDTH_MAX = 0.5  # Maximum (CI_width / D) for good measurement
# =====================================

def load_processed_data(file_path):
    """Load processed trajectory data."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading processed data from {file_path}: {e}")
        return None

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

def validate_measurement(D, sigma_loc, D_CI_low, D_CI_high, r_squared, msd_first, dt):
    """
    Validate a diffusion measurement and return quality flags.

    Args:
        D: Diffusion coefficient (μm²/s)
        sigma_loc: Localization error (μm)
        D_CI_low, D_CI_high: Bootstrap confidence intervals
        r_squared: Fit quality
        msd_first: MSD value at first tau
        dt: Time step (s)

    Returns:
        Dictionary with validation results and flags
    """
    flags = []
    severity = []  # 'warning' or 'error'

    # Check 1: Localization error dominance
    loc_error_contribution = 4 * sigma_loc**2
    if msd_first > 0 and loc_error_contribution > LOC_ERROR_THRESHOLD * msd_first:
        flags.append(f"Localization error dominates (4σ²={loc_error_contribution:.4f} vs MSD={msd_first:.4f})")
        severity.append('warning')

    # Check 2: Diffusion coefficient range
    if D < D_MIN:
        flags.append(f"D too small ({D:.6f} < {D_MIN} μm²/s) - may be immobile or tracking error")
        severity.append('warning')
    elif D > D_MAX:
        flags.append(f"D too large ({D:.2f} > {D_MAX} μm²/s) - check tracking quality")
        severity.append('error')

    # Check 3: Fit quality
    if r_squared < R_SQUARED_MIN:
        flags.append(f"Poor fit quality (R²={r_squared:.3f} < {R_SQUARED_MIN})")
        severity.append('warning')

    # Check 4: Bootstrap CI width
    if not np.isnan(D_CI_low) and not np.isnan(D_CI_high) and D > 0:
        ci_width = D_CI_high - D_CI_low
        relative_uncertainty = ci_width / D
        if relative_uncertainty > RELATIVE_CI_WIDTH_MAX:
            flags.append(f"Large uncertainty (ΔD/D = {relative_uncertainty:.2f} > {RELATIVE_CI_WIDTH_MAX})")
            severity.append('warning')

    # Check 5: Negative offset (unphysical)
    if sigma_loc == 0:
        flags.append("Negative MSD intercept (unphysical - may indicate systematic error)")
        severity.append('error')

    # Overall quality assessment
    has_error = 'error' in severity
    has_warning = 'warning' in severity

    if has_error:
        quality = 'FAIL'
    elif has_warning:
        quality = 'WARNING'
    else:
        quality = 'PASS'

    return {
        'quality': quality,
        'flags': flags,
        'severity': severity,
        'n_issues': len(flags)
    }

def bootstrap_fit_msd_with_validation(time_data, msd_data, trajectory_length, n_bootstrap=N_BOOTSTRAP):
    """
    Fit MSD with bootstrap and validation.

    Returns results including validation flags.
    """
    # Calculate max tau
    max_tau_index, tau_warning = calculate_max_tau_index(trajectory_length)

    # Extract data
    t_fit = time_data[:max_tau_index]
    msd_fit = msd_data[:max_tau_index]

    # Filter NaN
    valid_indices = ~np.isnan(msd_fit)
    t_fit = t_fit[valid_indices]
    msd_fit = msd_fit[valid_indices]

    if len(t_fit) < 3:
        return {
            'D': np.nan, 'offset': np.nan, 'sigma_loc': np.nan,
            'D_CI_low': np.nan, 'D_CI_high': np.nan,
            'sigma_loc_CI_low': np.nan, 'sigma_loc_CI_high': np.nan,
            'r_squared': np.nan, 't_fit': t_fit, 'msd_fit': msd_fit,
            'fit_values': np.nan, 'n_fit_points': len(t_fit),
            'max_tau_used': np.nan, 'tau_warning': tau_warning,
            'validation': {'quality': 'FAIL', 'flags': ['Insufficient data'],
                          'severity': ['error'], 'n_issues': 1}
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

        # Validate measurement
        msd_first = msd_fit[0]
        validation_results = validate_measurement(
            D_main, sigma_loc_main, D_CI_low, D_CI_high,
            r_squared, msd_first, t_fit[0]
        )

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
            'tau_warning': tau_warning,
            'validation': validation_results  # NEW: Validation results
        }
    except Exception as e:
        return {
            'D': np.nan, 'offset': np.nan, 'sigma_loc': np.nan,
            'D_CI_low': np.nan, 'D_CI_high': np.nan,
            'sigma_loc_CI_low': np.nan, 'sigma_loc_CI_high': np.nan,
            'r_squared': np.nan, 't_fit': t_fit, 'msd_fit': msd_fit,
            'fit_values': np.nan, 'n_fit_points': len(t_fit),
            'max_tau_used': np.nan, 'tau_warning': tau_warning,
            'validation': {'quality': 'FAIL', 'flags': [f'Fitting error: {str(e)}'],
                          'severity': ['error'], 'n_issues': 1}
        }

def calculate_radius_of_gyration(trajectory):
    """Calculate radius of gyration."""
    x = trajectory['x']
    y = trajectory['y']
    x_cm = np.mean(x)
    y_cm = np.mean(y)
    r2 = (x - x_cm)**2 + (y - y_cm)**2
    rg = np.sqrt(np.mean(r2))
    return rg

def analyze_trajectories(processed_data):
    """Analyze trajectories with validation."""
    analyzed_data = {
        'trajectories': [],
        'diffusion_coefficients': [],
        'localization_errors': [],
        'D_CI_low': [],
        'D_CI_high': [],
        'n_fit_points': [],
        'max_tau_used': [],
        'radius_of_gyration': [],
        'track_lengths': [],
        'r_squared_values': [],
        'quality_flags': [],  # NEW: Quality assessments
        'warnings': []
    }

    n_pass = n_warning = n_fail = 0

    for i, trajectory in enumerate(processed_data['trajectories']):
        if i % 10 == 0:
            print(f"  Processing trajectory {i+1}/{len(processed_data['trajectories'])}...")

        time_data = processed_data['time_data'][i]
        msd_data = processed_data['msd_data'][i]
        trajectory_length = len(trajectory['x'])

        # Fit with validation
        fit_results = bootstrap_fit_msd_with_validation(time_data, msd_data, trajectory_length)

        # Count quality
        quality = fit_results['validation']['quality']
        if quality == 'PASS':
            n_pass += 1
        elif quality == 'WARNING':
            n_warning += 1
        else:
            n_fail += 1

        rg = calculate_radius_of_gyration(trajectory)

        # Store results
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
            'validation': fit_results['validation'],  # NEW
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
        analyzed_data['quality_flags'].append(fit_results['validation'])

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

    print(f"\n  Quality Summary:")
    print(f"    ✓ PASS: {n_pass} trajectories")
    print(f"    ⚠ WARNING: {n_warning} trajectories")
    print(f"    ✗ FAIL: {n_fail} trajectories")

    return analyzed_data

def create_quality_report(analyzed_data, output_path, filename):
    """Generate detailed quality validation report."""
    report_data = []

    for traj in analyzed_data['trajectories']:
        val = traj['validation']
        if val['n_issues'] > 0:
            report_data.append({
                'trajectory_id': traj['id'],
                'quality': val['quality'],
                'D': traj['D'],
                'sigma_loc_nm': traj['sigma_loc'] * 1000,
                'R_squared': traj['r_squared'],
                'track_length': traj['track_length'],
                'n_fit_points': traj['n_fit_points'],
                'n_issues': val['n_issues'],
                'flags': ' | '.join(val['flags'])
            })

    if report_data:
        df = pd.DataFrame(report_data)
        report_file = os.path.join(output_path, f"{filename}_quality_report.csv")
        df.to_csv(report_file, index=False)
        print(f"  Quality report saved: {report_file}")
        return df
    else:
        print("  No quality issues detected - all measurements PASS!")
        return None

def main():
    """Main function with validation."""
    print("="*60)
    print("Diffusion Analyzer V4 - Comprehensive Validation")
    print("="*60)
    print(f"\nValidation thresholds:")
    print(f"  D range: {D_MIN} - {D_MAX} μm²/s")
    print(f"  R² minimum: {R_SQUARED_MIN}")
    print(f"  Loc error threshold: {LOC_ERROR_THRESHOLD}")
    print(f"  Max relative CI width: {RELATIVE_CI_WIDTH_MAX}")

    input_dir = input("\nEnter directory (press Enter for processed_trajectories): ")
    if input_dir == "":
        input_dir = os.path.join(os.getcwd(), "processed_trajectories")

    if not os.path.isdir(input_dir):
        print(f"Directory {input_dir} does not exist")
        return

    file_paths = glob.glob(os.path.join(input_dir, "tracked_*.pkl"))
    if not file_paths:
        print(f"No processed trajectory files found")
        return

    print(f"Found {len(file_paths)} files")

    output_dir = os.path.join(os.path.dirname(input_dir), "analyzed_trajectories_v4")
    os.makedirs(output_dir, exist_ok=True)

    summary_data = []

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]

        processed_data = load_processed_data(file_path)
        if processed_data is None:
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing {filename}")
        print(f"{'='*60}")

        analyzed_data = analyze_trajectories(processed_data)

        # Generate quality report
        quality_df = create_quality_report(analyzed_data, output_dir, base_name)

        # Print summary
        if analyzed_data['diffusion_coefficients']:
            median_D = np.median(analyzed_data['diffusion_coefficients'])
            sigma_loc_nm = [s*1000 for s in analyzed_data['localization_errors'] if not np.isnan(s)]
            median_sigma = np.median(sigma_loc_nm) if sigma_loc_nm else np.nan

            n_pass = sum(1 for q in analyzed_data['quality_flags'] if q['quality'] == 'PASS')
            n_warning = sum(1 for q in analyzed_data['quality_flags'] if q['quality'] == 'WARNING')
            n_fail = sum(1 for q in analyzed_data['quality_flags'] if q['quality'] == 'FAIL')

            print(f"\n  Results:")
            print(f"    Median D: {median_D:.6f} μm²/s")
            print(f"    Median σ_loc: {median_sigma:.1f} nm")
            print(f"    Quality: {n_pass} PASS, {n_warning} WARNING, {n_fail} FAIL")

            summary_data.append({
                'filename': base_name,
                'median_D': median_D,
                'median_sigma_loc_nm': median_sigma,
                'n_trajectories': len(analyzed_data['trajectories']),
                'n_pass': n_pass,
                'n_warning': n_warning,
                'n_fail': n_fail
            })

        # Save
        output_file = os.path.join(output_dir, f"analyzed_{base_name}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(analyzed_data, f)

    # Save summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(output_dir, "validation_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n{'='*60}")
        print(f"Summary saved to {summary_csv}")
        print(f"All results in {output_dir}")

if __name__ == "__main__":
    main()
