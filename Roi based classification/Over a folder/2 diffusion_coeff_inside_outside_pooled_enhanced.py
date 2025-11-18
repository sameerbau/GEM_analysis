#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diffusion_coeff_pooled_enhanced.py

Enhanced version of pooled diffusion analysis with:
- Sample sizes displayed on all plots
- Statistical comparison (Mann-Whitney U, Kolmogorov-Smirnov, t-test)
- CSV exports of diffusion values and statistics
- Bootstrapped confidence intervals
- Power analysis

Input:
- Folder containing multiple roi_trajectory_data_*.pkl files

Output:
- Summary statistics with bootstrapped confidence intervals
- Statistical test results
- CSV files with raw data and statistics
- Enhanced plots with sample sizes
- Power analysis report

Usage:
python diffusion_coeff_pooled_enhanced.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Global parameters
# =====================================
# Bin size for diffusion coefficient histogram (in µm^2/s)
D_BIN_SIZE = 0.05
# Output subfolder name
OUTPUT_SUBFOLDER = 'pooled_diffusion_analysis_enhanced'
# Bootstrap parameters
N_BOOTSTRAP = 10000  # Number of bootstrap iterations
BOOTSTRAP_CI = 95  # Confidence interval percentage
# Statistical parameters
ALPHA = 0.05  # Significance level
# Power analysis parameters
POWER_TARGET = 0.8  # Target statistical power (80%)
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
        print(f"Successfully loaded: {os.path.basename(file_path)}")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def analyze_diffusion_single_file(roi_data):
    """
    Extract diffusion coefficients from a single ROI data file.

    Args:
        roi_data: Dictionary containing ROI-assigned trajectory data

    Returns:
        Tuple of (inside_D, outside_D) lists
    """
    inside_D = []
    outside_D = []

    for roi_id, trajectories in roi_data['roi_trajectories'].items():
        if roi_id == 'unassigned':
            # Trajectories outside ROIs
            for traj in trajectories:
                if 'D' in traj and not np.isnan(traj['D']):
                    outside_D.append(traj['D'])
        else:
            # Trajectories inside ROIs
            for traj in trajectories:
                if 'D' in traj and not np.isnan(traj['D']):
                    inside_D.append(traj['D'])

    return inside_D, outside_D


def pool_diffusion_data(folder_path):
    """
    Pool diffusion data from all roi_trajectory_data_*.pkl files in a folder.

    Args:
        folder_path: Path to folder containing pkl files

    Returns:
        Tuple of (inside_D_array, outside_D_array, file_info) with pooled data
    """
    # Find all roi_trajectory_data pkl files
    pkl_pattern = os.path.join(folder_path, 'roi_trajectory_data_*.pkl')
    pkl_files = glob.glob(pkl_pattern)

    if not pkl_files:
        print(f"No roi_trajectory_data_*.pkl files found in {folder_path}")
        return None, None, None

    print(f"\nFound {len(pkl_files)} pkl files to process")
    print("="*50)

    # Lists to collect all diffusion coefficients
    all_inside_D = []
    all_outside_D = []

    # Track file-level statistics
    file_info = []

    # Process each file
    for pkl_file in pkl_files:
        roi_data = load_roi_data(pkl_file)

        if roi_data is not None:
            inside_D, outside_D = analyze_diffusion_single_file(roi_data)
            all_inside_D.extend(inside_D)
            all_outside_D.extend(outside_D)

            file_info.append({
                'filename': os.path.basename(pkl_file),
                'n_inside': len(inside_D),
                'n_outside': len(outside_D)
            })

            print(f"  Inside: {len(inside_D)}, Outside: {len(outside_D)}")

    print("="*50)
    print(f"Total pooled - Inside: {len(all_inside_D)}, Outside: {len(all_outside_D)}")

    return np.array(all_inside_D), np.array(all_outside_D), file_info


def bootstrap_statistic(data, statistic_func, n_bootstrap=N_BOOTSTRAP, ci=BOOTSTRAP_CI):
    """
    Calculate bootstrap confidence intervals for a statistic.

    Args:
        data: Array of data values
        statistic_func: Function to calculate statistic (e.g., np.mean, np.median)
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence interval percentage (e.g., 95 for 95% CI)

    Returns:
        Dictionary with statistic value and confidence intervals
    """
    # Calculate observed statistic
    observed = statistic_func(data)

    # Bootstrap resampling
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_samples.append(statistic_func(sample))

    bootstrap_samples = np.array(bootstrap_samples)

    # Calculate confidence intervals
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    ci_lower = np.percentile(bootstrap_samples, lower_percentile)
    ci_upper = np.percentile(bootstrap_samples, upper_percentile)

    return {
        'value': observed,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_percent': ci,
        'std_error': np.std(bootstrap_samples)
    }


def calculate_statistics_with_bootstrap(data, label):
    """
    Calculate comprehensive statistics with bootstrap confidence intervals.

    Args:
        data: Array of diffusion coefficients
        label: Label for the data (e.g., 'Inside ROIs', 'Outside ROIs')

    Returns:
        Dictionary of statistics with confidence intervals
    """
    stats_dict = {
        'label': label,
        'n': len(data),
        'mean': bootstrap_statistic(data, np.mean),
        'median': bootstrap_statistic(data, np.median),
        'std': bootstrap_statistic(data, np.std),
        'sem': stats.sem(data),
        'min': np.min(data),
        'max': np.max(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25)
    }

    return stats_dict


def perform_statistical_tests(inside_D, outside_D):
    """
    Perform comprehensive statistical comparison between inside and outside ROIs.

    Args:
        inside_D: Array of diffusion coefficients inside ROIs
        outside_D: Array of diffusion coefficients outside ROIs

    Returns:
        Dictionary with test results
    """
    results = {}

    # 1. Mann-Whitney U test (non-parametric)
    u_stat, p_mw = stats.mannwhitneyu(inside_D, outside_D, alternative='two-sided')
    results['mann_whitney'] = {
        'statistic': u_stat,
        'p_value': p_mw,
        'significant': p_mw < ALPHA
    }

    # 2. Kolmogorov-Smirnov test
    ks_stat, p_ks = stats.ks_2samp(inside_D, outside_D)
    results['kolmogorov_smirnov'] = {
        'statistic': ks_stat,
        'p_value': p_ks,
        'significant': p_ks < ALPHA
    }

    # 3. Check normality
    if len(inside_D) >= 3 and len(outside_D) >= 3:
        _, p_norm_inside = stats.shapiro(inside_D[:5000])  # Shapiro test limited to 5000 samples
        _, p_norm_outside = stats.shapiro(outside_D[:5000])

        results['normality'] = {
            'inside_p': p_norm_inside,
            'outside_p': p_norm_outside,
            'inside_normal': p_norm_inside > ALPHA,
            'outside_normal': p_norm_outside > ALPHA
        }

        # 4. If both normal, perform t-test
        if p_norm_inside > ALPHA and p_norm_outside > ALPHA:
            # Check equal variances
            _, p_levene = stats.levene(inside_D, outside_D)
            equal_var = p_levene > ALPHA

            t_stat, p_t = stats.ttest_ind(inside_D, outside_D, equal_var=equal_var)
            results['t_test'] = {
                'statistic': t_stat,
                'p_value': p_t,
                'significant': p_t < ALPHA,
                'equal_variance': equal_var,
                'test_type': "Student's t-test" if equal_var else "Welch's t-test"
            }

    # 5. Effect size - Cohen's d
    mean_inside = np.mean(inside_D)
    mean_outside = np.mean(outside_D)
    std_inside = np.std(inside_D, ddof=1)
    std_outside = np.std(outside_D, ddof=1)
    n_inside = len(inside_D)
    n_outside = len(outside_D)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n_inside - 1) * std_inside**2 + (n_outside - 1) * std_outside**2) /
                         (n_inside + n_outside - 2))
    cohens_d = (mean_inside - mean_outside) / pooled_std

    results['effect_size'] = {
        'cohens_d': cohens_d,
        'interpretation': interpret_cohens_d(cohens_d)
    }

    # 6. Cliff's Delta (non-parametric effect size)
    cliffs_delta = calculate_cliffs_delta(inside_D, outside_D)
    results['effect_size']['cliffs_delta'] = cliffs_delta
    results['effect_size']['cliffs_interpretation'] = interpret_cliffs_delta(cliffs_delta)

    # 7. Mean and median differences with bootstrap CI
    diff_means = bootstrap_statistic(
        np.concatenate([inside_D, -outside_D]),
        lambda x: np.mean(x[x > 0]) - np.mean(-x[x < 0])
    )
    results['difference'] = {
        'mean_diff': mean_inside - mean_outside,
        'median_diff': np.median(inside_D) - np.median(outside_D),
        'mean_diff_ci': diff_means
    }

    return results


def calculate_cliffs_delta(group1, group2):
    """Calculate Cliff's delta effect size."""
    greater = 0
    lesser = 0

    for x in group1:
        for y in group2:
            if x > y:
                greater += 1
            elif x < y:
                lesser += 1

    total_comparisons = len(group1) * len(group2)
    return (greater - lesser) / total_comparisons


def interpret_cliffs_delta(delta):
    """Interpret Cliff's delta effect size."""
    abs_delta = abs(delta)

    if abs_delta < 0.147:
        return "Negligible"
    elif abs_delta < 0.33:
        return "Small"
    elif abs_delta < 0.474:
        return "Medium"
    else:
        return "Large"


def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)

    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


def perform_power_analysis(inside_D, outside_D, target_power=POWER_TARGET, alpha=ALPHA):
    """
    Perform power analysis to determine required sample sizes.

    Args:
        inside_D: Array of diffusion coefficients inside ROIs
        outside_D: Array of diffusion coefficients outside ROIs
        target_power: Target statistical power (default 0.8)
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with power analysis results
    """
    # Calculate observed effect size
    mean_inside = np.mean(inside_D)
    mean_outside = np.mean(outside_D)
    std_inside = np.std(inside_D, ddof=1)
    std_outside = np.std(outside_D, ddof=1)
    pooled_std = np.sqrt((std_inside**2 + std_outside**2) / 2)

    effect_size = abs(mean_inside - mean_outside) / pooled_std

    # Calculate current power using Mann-Whitney U test
    # Approximate power calculation using normal approximation
    n1 = len(inside_D)
    n2 = len(outside_D)

    # Z-score for alpha level (two-tailed)
    z_alpha = stats.norm.ppf(1 - alpha/2)

    # Calculate non-centrality parameter
    n_harmonic = 2 * n1 * n2 / (n1 + n2)
    ncp = effect_size * np.sqrt(n_harmonic / 2)

    # Current power
    current_power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)

    # Calculate required sample size for target power
    z_beta = stats.norm.ppf(target_power)
    n_required = 2 * ((z_alpha + z_beta) / effect_size)**2

    # Calculate sample size for various effect sizes
    effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large
    n_required_by_effect = {}

    for es in effect_sizes:
        n_req = 2 * ((z_alpha + z_beta) / es)**2
        n_required_by_effect[es] = int(np.ceil(n_req))

    power_results = {
        'observed_effect_size': effect_size,
        'current_n1': n1,
        'current_n2': n2,
        'current_power': current_power,
        'target_power': target_power,
        'required_n_per_group': int(np.ceil(n_required)),
        'required_n_by_effect_size': n_required_by_effect,
        'sufficient_power': current_power >= target_power
    }

    return power_results


def export_data_to_csv(inside_D, outside_D, stats_inside, stats_outside,
                       test_results, power_results, file_info, output_dir):
    """
    Export data and statistics to CSV files.

    Args:
        inside_D: Array of diffusion coefficients inside ROIs
        outside_D: Array of diffusion coefficients outside ROIs
        stats_inside: Statistics dictionary for inside ROIs
        stats_outside: Statistics dictionary for outside ROIs
        test_results: Statistical test results
        power_results: Power analysis results
        file_info: File-level information
        output_dir: Output directory path
    """
    # 1. Export raw diffusion values
    max_len = max(len(inside_D), len(outside_D))

    # Pad shorter array with NaN
    inside_padded = np.pad(inside_D, (0, max_len - len(inside_D)),
                          constant_values=np.nan)
    outside_padded = np.pad(outside_D, (0, max_len - len(outside_D)),
                           constant_values=np.nan)

    df_raw = pd.DataFrame({
        'Inside_ROIs_D': inside_padded,
        'Outside_ROIs_D': outside_padded
    })

    raw_csv_path = os.path.join(output_dir, 'raw_diffusion_values.csv')
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"Raw diffusion values saved to: {raw_csv_path}")

    # 2. Export summary statistics
    summary_data = []

    for stats_dict in [stats_inside, stats_outside]:
        summary_data.append({
            'Category': stats_dict['label'],
            'N': stats_dict['n'],
            'Mean': stats_dict['mean']['value'],
            'Mean_CI_Lower': stats_dict['mean']['ci_lower'],
            'Mean_CI_Upper': stats_dict['mean']['ci_upper'],
            'Median': stats_dict['median']['value'],
            'Median_CI_Lower': stats_dict['median']['ci_lower'],
            'Median_CI_Upper': stats_dict['median']['ci_upper'],
            'Std': stats_dict['std']['value'],
            'Std_CI_Lower': stats_dict['std']['ci_lower'],
            'Std_CI_Upper': stats_dict['std']['ci_upper'],
            'SEM': stats_dict['sem'],
            'Min': stats_dict['min'],
            'Max': stats_dict['max'],
            'Q25': stats_dict['q25'],
            'Q75': stats_dict['q75'],
            'IQR': stats_dict['iqr']
        })

    df_summary = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(output_dir, 'summary_statistics.csv')
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Summary statistics saved to: {summary_csv_path}")

    # 3. Export statistical test results
    test_data = []

    # Mann-Whitney U test
    test_data.append({
        'Test': 'Mann-Whitney U',
        'Statistic': test_results['mann_whitney']['statistic'],
        'P_Value': test_results['mann_whitney']['p_value'],
        'Significant': test_results['mann_whitney']['significant'],
        'Alpha': ALPHA
    })

    # Kolmogorov-Smirnov test
    test_data.append({
        'Test': 'Kolmogorov-Smirnov',
        'Statistic': test_results['kolmogorov_smirnov']['statistic'],
        'P_Value': test_results['kolmogorov_smirnov']['p_value'],
        'Significant': test_results['kolmogorov_smirnov']['significant'],
        'Alpha': ALPHA
    })

    # T-test if available
    if 't_test' in test_results:
        test_data.append({
            'Test': test_results['t_test']['test_type'],
            'Statistic': test_results['t_test']['statistic'],
            'P_Value': test_results['t_test']['p_value'],
            'Significant': test_results['t_test']['significant'],
            'Alpha': ALPHA
        })

    df_tests = pd.DataFrame(test_data)
    tests_csv_path = os.path.join(output_dir, 'statistical_tests.csv')
    df_tests.to_csv(tests_csv_path, index=False)
    print(f"Statistical test results saved to: {tests_csv_path}")

    # 4. Export effect sizes
    effect_data = [{
        'Effect_Size_Measure': "Cohen's d",
        'Value': test_results['effect_size']['cohens_d'],
        'Interpretation': test_results['effect_size']['interpretation']
    }, {
        'Effect_Size_Measure': "Cliff's Delta",
        'Value': test_results['effect_size']['cliffs_delta'],
        'Interpretation': test_results['effect_size']['cliffs_interpretation']
    }]

    df_effects = pd.DataFrame(effect_data)
    effects_csv_path = os.path.join(output_dir, 'effect_sizes.csv')
    df_effects.to_csv(effects_csv_path, index=False)
    print(f"Effect sizes saved to: {effects_csv_path}")

    # 5. Export power analysis
    power_data = [{
        'Metric': 'Observed Effect Size',
        'Value': power_results['observed_effect_size']
    }, {
        'Metric': 'Current N (Inside)',
        'Value': power_results['current_n1']
    }, {
        'Metric': 'Current N (Outside)',
        'Value': power_results['current_n2']
    }, {
        'Metric': 'Current Power',
        'Value': power_results['current_power']
    }, {
        'Metric': 'Target Power',
        'Value': power_results['target_power']
    }, {
        'Metric': 'Required N per Group',
        'Value': power_results['required_n_per_group']
    }, {
        'Metric': 'Sufficient Power',
        'Value': 'Yes' if power_results['sufficient_power'] else 'No'
    }]

    df_power = pd.DataFrame(power_data)
    power_csv_path = os.path.join(output_dir, 'power_analysis.csv')
    df_power.to_csv(power_csv_path, index=False)
    print(f"Power analysis saved to: {power_csv_path}")

    # 6. Export file-level information
    df_files = pd.DataFrame(file_info)
    files_csv_path = os.path.join(output_dir, 'file_level_counts.csv')
    df_files.to_csv(files_csv_path, index=False)
    print(f"File-level information saved to: {files_csv_path}")


def plot_diffusion_histogram(inside_D, outside_D, stats_inside, stats_outside, output_path):
    """
    Plot histogram of diffusion coefficients with sample sizes and statistics.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Determine bin range
    max_D = max(np.max(inside_D), np.max(outside_D))
    bins = np.arange(0, max_D + D_BIN_SIZE, D_BIN_SIZE)

    # Plot histograms
    n_inside = stats_inside['n']
    n_outside = stats_outside['n']

    ax.hist(inside_D, bins=bins, alpha=0.6, label=f'Inside ROIs (n={n_inside})', color='blue')
    ax.hist(outside_D, bins=bins, alpha=0.6, label=f'Outside ROIs (n={n_outside})', color='orange')

    # Add mean lines
    mean_inside = stats_inside['mean']['value']
    mean_outside = stats_outside['mean']['value']

    ax.axvline(mean_inside, color='blue', linestyle='--', linewidth=2,
               label=f'Inside Mean: {mean_inside:.3f} µm²/s')
    ax.axvline(mean_outside, color='orange', linestyle='--', linewidth=2,
               label=f'Outside Mean: {mean_outside:.3f} µm²/s')

    ax.set_xlabel('Diffusion Coefficient (µm²/s)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Pooled Diffusion Coefficient Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Histogram saved to {output_path}")


def plot_diffusion_cdf(inside_D, outside_D, stats_inside, stats_outside, output_path):
    """
    Plot CDF with sample sizes and median markers.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    n_inside = stats_inside['n']
    n_outside = stats_outside['n']

    # Compute and plot CDF for inside ROIs
    inside_sorted = np.sort(inside_D)
    inside_cdf = np.arange(1, len(inside_sorted) + 1) / len(inside_sorted)
    ax.plot(inside_sorted, inside_cdf, label=f'Inside ROIs (n={n_inside})',
            linewidth=2, color='blue')

    # Compute and plot CDF for outside ROIs
    outside_sorted = np.sort(outside_D)
    outside_cdf = np.arange(1, len(outside_sorted) + 1) / len(outside_sorted)
    ax.plot(outside_sorted, outside_cdf, label=f'Outside ROIs (n={n_outside})',
            linewidth=2, color='orange')

    # Mark medians
    median_inside = stats_inside['median']['value']
    median_outside = stats_outside['median']['value']

    ax.axvline(median_inside, color='blue', linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'Inside Median: {median_inside:.3f} µm²/s')
    ax.axvline(median_outside, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'Outside Median: {median_outside:.3f} µm²/s')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)

    ax.set_xlabel('Diffusion Coefficient (µm²/s)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Pooled Diffusion Coefficient CDF', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"CDF plot saved to {output_path}")


def plot_diffusion_distributions(inside_D, outside_D, stats_inside, stats_outside, output_path):
    """
    Plot KDE distribution comparison with sample sizes.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    n_inside = stats_inside['n']
    n_outside = stats_outside['n']

    # Create KDE plots
    sns.kdeplot(inside_D, ax=ax, label=f'Inside ROIs (n={n_inside})',
                color='blue', fill=True, alpha=0.4, linewidth=2)
    sns.kdeplot(outside_D, ax=ax, label=f'Outside ROIs (n={n_outside})',
                color='orange', fill=True, alpha=0.4, linewidth=2)

    # Add mean lines
    mean_inside = stats_inside['mean']['value']
    mean_outside = stats_outside['mean']['value']

    ax.axvline(mean_inside, color='blue', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(mean_outside, color='orange', linestyle='--', linewidth=2, alpha=0.8)

    ax.set_xlabel('Diffusion Coefficient (µm²/s)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Pooled Diffusion Coefficient Distribution Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Distribution comparison saved to {output_path}")


def plot_diffusion_boxplot(inside_D, outside_D, stats_inside, stats_outside,
                           test_results, output_path):
    """
    Plot box plot with sample sizes and significance annotation.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    n_inside = stats_inside['n']
    n_outside = stats_outside['n']

    # Create box plot
    bp = ax.boxplot([inside_D, outside_D],
                     labels=[f'Inside ROIs\n(n={n_inside})', f'Outside ROIs\n(n={n_outside})'],
                     patch_artist=True, widths=0.6,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))

    # Color boxes
    colors = ['blue', 'orange']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Add mean markers
    means = [stats_inside['mean']['value'], stats_outside['mean']['value']]
    ax.scatter([1, 2], means, marker='D', s=100, color='green',
               zorder=3, label='Mean', edgecolors='black', linewidths=1.5)

    # Add significance annotation if significant
    p_value = test_results['mann_whitney']['p_value']

    if test_results['mann_whitney']['significant']:
        # Determine significance level
        if p_value < 0.001:
            sig_text = '***'
        elif p_value < 0.01:
            sig_text = '**'
        elif p_value < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'

        # Draw significance bar
        y_max = max(np.max(inside_D), np.max(outside_D))
        y_pos = y_max * 1.05

        ax.plot([1, 1, 2, 2], [y_pos, y_pos*1.02, y_pos*1.02, y_pos], 'k-', linewidth=1.5)
        ax.text(1.5, y_pos*1.03, f'{sig_text}\np={p_value:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        # Not significant
        y_max = max(np.max(inside_D), np.max(outside_D))
        y_pos = y_max * 1.05
        ax.text(1.5, y_pos, f'ns\np={p_value:.4f}',
                ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Diffusion Coefficient (µm²/s)', fontsize=12)
    ax.set_title('Pooled Diffusion Coefficient Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Box plot saved to {output_path}")


def plot_bootstrap_ci(stats_inside, stats_outside, output_path):
    """
    Plot bootstrap confidence intervals for mean and median.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    categories = ['Inside ROIs', 'Outside ROIs']

    # Plot mean with CI
    means = [stats_inside['mean']['value'], stats_outside['mean']['value']]
    ci_lower = [stats_inside['mean']['ci_lower'], stats_outside['mean']['ci_lower']]
    ci_upper = [stats_inside['mean']['ci_upper'], stats_outside['mean']['ci_upper']]
    errors = [[means[i] - ci_lower[i] for i in range(2)],
              [ci_upper[i] - means[i] for i in range(2)]]

    x_pos = np.arange(len(categories))
    ax1.bar(x_pos, means, alpha=0.7, color=['blue', 'orange'], edgecolor='black', linewidth=1.5)
    ax1.errorbar(x_pos, means, yerr=errors, fmt='none', ecolor='black',
                 capsize=5, capthick=2, linewidth=2)

    # Add value labels
    for i, (mean, lower, upper) in enumerate(zip(means, ci_lower, ci_upper)):
        ax1.text(i, mean, f'{mean:.3f}\n[{lower:.3f}, {upper:.3f}]',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{cat}\n(n={stats_inside["n"] if i==0 else stats_outside["n"]})'
                         for i, cat in enumerate(categories)])
    ax1.set_ylabel('Mean Diffusion Coefficient (µm²/s)', fontsize=12)
    ax1.set_title(f'Mean with {BOOTSTRAP_CI}% Bootstrap CI', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')

    # Plot median with CI
    medians = [stats_inside['median']['value'], stats_outside['median']['value']]
    ci_lower = [stats_inside['median']['ci_lower'], stats_outside['median']['ci_lower']]
    ci_upper = [stats_inside['median']['ci_upper'], stats_outside['median']['ci_upper']]
    errors = [[medians[i] - ci_lower[i] for i in range(2)],
              [ci_upper[i] - medians[i] for i in range(2)]]

    ax2.bar(x_pos, medians, alpha=0.7, color=['blue', 'orange'], edgecolor='black', linewidth=1.5)
    ax2.errorbar(x_pos, medians, yerr=errors, fmt='none', ecolor='black',
                 capsize=5, capthick=2, linewidth=2)

    # Add value labels
    for i, (median, lower, upper) in enumerate(zip(medians, ci_lower, ci_upper)):
        ax2.text(i, median, f'{median:.3f}\n[{lower:.3f}, {upper:.3f}]',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{cat}\n(n={stats_inside["n"] if i==0 else stats_outside["n"]})'
                         for i, cat in enumerate(categories)])
    ax2.set_ylabel('Median Diffusion Coefficient (µm²/s)', fontsize=12)
    ax2.set_title(f'Median with {BOOTSTRAP_CI}% Bootstrap CI', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Bootstrap CI plot saved to {output_path}")


def plot_power_analysis(power_results, output_path):
    """
    Create power analysis visualization.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Current vs Required sample size
    categories = ['Current\nSample Size', 'Required\nSample Size\n(80% power)']
    current_avg = (power_results['current_n1'] + power_results['current_n2']) / 2
    values = [current_avg, power_results['required_n_per_group']]
    colors = ['green' if power_results['sufficient_power'] else 'orange', 'blue']

    bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_ylabel('Sample Size per Group', fontsize=12)
    ax1.set_title('Sample Size Analysis', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')

    # Add power status text
    status = 'SUFFICIENT' if power_results['sufficient_power'] else 'INSUFFICIENT'
    status_color = 'green' if power_results['sufficient_power'] else 'red'
    ax1.text(0.5, 0.95, f'Power Status: {status}',
            transform=ax1.transAxes, ha='center', va='top',
            fontsize=11, fontweight='bold', color=status_color,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Required sample sizes for different effect sizes
    effect_sizes = list(power_results['required_n_by_effect_size'].keys())
    sample_sizes = list(power_results['required_n_by_effect_size'].values())
    labels = [f'Small\n(d={es})' if es == 0.2 else f'Medium\n(d={es})' if es == 0.5
              else f'Large\n(d={es})' for es in effect_sizes]

    bars = ax2.bar(labels, sample_sizes, color=['red', 'orange', 'green'],
                   alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, sample_sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Mark observed effect size
    observed_es = power_results['observed_effect_size']
    ax2.axhline(current_avg, color='blue', linestyle='--', linewidth=2,
                label=f'Current n (avg={int(current_avg)})')

    ax2.set_ylabel('Required Sample Size per Group', fontsize=12)
    ax2.set_title(f'Sample Size by Effect Size\n(Observed: d={observed_es:.3f})',
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Power analysis plot saved to {output_path}")


def print_detailed_results(stats_inside, stats_outside, test_results, power_results):
    """
    Print comprehensive results to console.
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("="*70)

    # Summary Statistics
    print("\n" + "-"*70)
    print("SUMMARY STATISTICS WITH BOOTSTRAP CONFIDENCE INTERVALS")
    print("-"*70)

    for stats_dict in [stats_inside, stats_outside]:
        print(f"\n{stats_dict['label']} (n={stats_dict['n']}):")
        print(f"  Mean:   {stats_dict['mean']['value']:.4f} µm²/s "
              f"[{stats_dict['mean']['ci_lower']:.4f}, {stats_dict['mean']['ci_upper']:.4f}]")
        print(f"  Median: {stats_dict['median']['value']:.4f} µm²/s "
              f"[{stats_dict['median']['ci_lower']:.4f}, {stats_dict['median']['ci_upper']:.4f}]")
        print(f"  Std:    {stats_dict['std']['value']:.4f} µm²/s "
              f"[{stats_dict['std']['ci_lower']:.4f}, {stats_dict['std']['ci_upper']:.4f}]")
        print(f"  SEM:    {stats_dict['sem']:.4f} µm²/s")
        print(f"  IQR:    {stats_dict['iqr']:.4f} µm²/s")
        print(f"  Range:  [{stats_dict['min']:.4f}, {stats_dict['max']:.4f}] µm²/s")

    # Statistical Tests
    print("\n" + "-"*70)
    print("STATISTICAL TESTS")
    print("-"*70)

    # Mann-Whitney U
    mw = test_results['mann_whitney']
    print(f"\nMann-Whitney U Test (non-parametric):")
    print(f"  U-statistic: {mw['statistic']:.2f}")
    print(f"  P-value:     {mw['p_value']:.6f}")
    print(f"  Significant: {'YES ***' if mw['significant'] else 'NO'} (α={ALPHA})")

    # Kolmogorov-Smirnov
    ks = test_results['kolmogorov_smirnov']
    print(f"\nKolmogorov-Smirnov Test:")
    print(f"  KS-statistic: {ks['statistic']:.4f}")
    print(f"  P-value:      {ks['p_value']:.6f}")
    print(f"  Significant:  {'YES ***' if ks['significant'] else 'NO'} (α={ALPHA})")

    # T-test if available
    if 't_test' in test_results:
        tt = test_results['t_test']
        print(f"\n{tt['test_type']}:")
        print(f"  T-statistic: {tt['statistic']:.4f}")
        print(f"  P-value:     {tt['p_value']:.6f}")
        print(f"  Significant: {'YES ***' if tt['significant'] else 'NO'} (α={ALPHA})")

    # Effect Sizes
    print("\n" + "-"*70)
    print("EFFECT SIZES")
    print("-"*70)

    es = test_results['effect_size']
    print(f"\nCohen's d:    {es['cohens_d']:.4f} ({es['interpretation']})")
    print(f"Cliff's Delta: {es['cliffs_delta']:.4f} ({es['cliffs_interpretation']})")

    # Differences
    diff = test_results['difference']
    print(f"\nMean Difference:   {diff['mean_diff']:.4f} µm²/s")
    print(f"Median Difference: {diff['median_diff']:.4f} µm²/s")

    # Power Analysis
    print("\n" + "-"*70)
    print("POWER ANALYSIS")
    print("-"*70)

    print(f"\nObserved Effect Size: {power_results['observed_effect_size']:.4f}")
    print(f"Current Sample Sizes: n1={power_results['current_n1']}, n2={power_results['current_n2']}")
    print(f"Current Power:        {power_results['current_power']:.4f} ({power_results['current_power']*100:.1f}%)")
    print(f"Target Power:         {power_results['target_power']:.4f} ({power_results['target_power']*100:.1f}%)")
    print(f"\nRequired Sample Size per Group: {power_results['required_n_per_group']}")
    print(f"Sufficient Power: {'YES ✓' if power_results['sufficient_power'] else 'NO ✗'}")

    print("\nRequired Sample Sizes for Standard Effect Sizes:")
    for es, n_req in power_results['required_n_by_effect_size'].items():
        es_label = "Small" if es == 0.2 else "Medium" if es == 0.5 else "Large"
        print(f"  {es_label} (d={es}): {n_req} per group")

    print("\n" + "="*70)


def main():
    """
    Main function to perform enhanced pooled diffusion analysis.
    """
    print("\n" + "="*70)
    print("ENHANCED POOLED ROI DIFFUSION ANALYZER")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Sample sizes on all plots")
    print("  ✓ Statistical testing (Mann-Whitney U, KS test, t-test)")
    print("  ✓ Bootstrapped confidence intervals")
    print("  ✓ Power analysis")
    print("  ✓ CSV exports of all data and statistics")
    print("="*70)

    # Ask for input folder
    folder_path = input("\nEnter path to folder containing roi_trajectory_data pkl files: ")

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        return

    # Pool diffusion data from all files
    print("\nPooling data from files...")
    inside_D, outside_D, file_info = pool_diffusion_data(folder_path)

    if inside_D is None or outside_D is None:
        print("No data to analyze. Exiting.")
        return

    if len(inside_D) == 0 and len(outside_D) == 0:
        print("No valid diffusion coefficients found. Exiting.")
        return

    if len(inside_D) < 3 or len(outside_D) < 3:
        print("Warning: Very small sample size may lead to unreliable statistical tests.")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(folder_path, f'{OUTPUT_SUBFOLDER}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Calculate statistics with bootstrap CI
    print("\nCalculating statistics with bootstrap confidence intervals...")
    print(f"  Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"  Confidence interval: {BOOTSTRAP_CI}%")

    stats_inside = calculate_statistics_with_bootstrap(inside_D, 'Inside ROIs')
    stats_outside = calculate_statistics_with_bootstrap(outside_D, 'Outside ROIs')

    # Perform statistical tests
    print("\nPerforming statistical tests...")
    test_results = perform_statistical_tests(inside_D, outside_D)

    # Perform power analysis
    print("\nPerforming power analysis...")
    power_results = perform_power_analysis(inside_D, outside_D)

    # Print detailed results
    print_detailed_results(stats_inside, stats_outside, test_results, power_results)

    # Export to CSV
    print("\n" + "-"*70)
    print("EXPORTING DATA TO CSV")
    print("-"*70)
    export_data_to_csv(inside_D, outside_D, stats_inside, stats_outside,
                      test_results, power_results, file_info, output_dir)

    # Generate plots
    print("\n" + "-"*70)
    print("GENERATING PLOTS")
    print("-"*70)

    plot_diffusion_histogram(inside_D, outside_D, stats_inside, stats_outside,
                            os.path.join(output_dir, 'pooled_diffusion_histogram.png'))

    plot_diffusion_cdf(inside_D, outside_D, stats_inside, stats_outside,
                      os.path.join(output_dir, 'pooled_diffusion_cdf.png'))

    plot_diffusion_distributions(inside_D, outside_D, stats_inside, stats_outside,
                                os.path.join(output_dir, 'pooled_diffusion_distributions.png'))

    plot_diffusion_boxplot(inside_D, outside_D, stats_inside, stats_outside, test_results,
                          os.path.join(output_dir, 'pooled_diffusion_boxplot.png'))

    plot_bootstrap_ci(stats_inside, stats_outside,
                     os.path.join(output_dir, 'bootstrap_confidence_intervals.png'))

    plot_power_analysis(power_results,
                       os.path.join(output_dir, 'power_analysis.png'))

    print("\n" + "="*70)
    print(f"ANALYSIS COMPLETE!")
    print(f"\nAll results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
