#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11compare_alpha_across_conditions.py

Multi-Condition Alpha Exponent Comparison for Single Particle Tracking

This script pools alpha exponent data from multiple experimental conditions
and performs comprehensive statistical comparisons.

Use case:
- Compare alpha values across different treatments, time points, or genotypes
- Each condition can have multiple replicate files
- Statistical tests, effect sizes, and publication-quality plots

Input:  Multiple alpha_analyzed_*.pkl files from 11alpha_exponent_analyzer.py
Output:
    - Pooled statistics per condition (CSV)
    - Statistical comparison results (CSV)
    - Box plots, violin plots, CDF comparisons
    - Effect size analysis
    - Diffusion type distribution comparison

Created: 2025
Author: GEM Analysis Pipeline
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================

# Minimum trajectories per condition for valid comparison
MIN_TRAJECTORIES_PER_CONDITION = 10

# Statistical significance level
ALPHA_SIGNIFICANCE = 0.05

# Bootstrap parameters for confidence intervals
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 95  # Confidence interval percentage

# Quality filtering
MIN_R_SQUARED = 0.6
MIN_TRACK_LENGTH = 10

# Plot style
PLOT_STYLE = 'whitegrid'
PLOT_PALETTE = 'Set2'

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_alpha_analyzed_file(file_path):
    """
    Load alpha-analyzed data from pickle file.

    Args:
        file_path: Path to alpha_analyzed_*.pkl file

    Returns:
        Dictionary with alpha analysis results
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded: {os.path.basename(file_path)}")
        return data
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None


def extract_alpha_values(data, min_r2=MIN_R_SQUARED, min_length=MIN_TRACK_LENGTH):
    """
    Extract alpha values from loaded data with quality filtering.

    Args:
        data: Dictionary from alpha_analyzed_*.pkl file
        min_r2: Minimum R² for alpha fit
        min_length: Minimum track length

    Returns:
        Dictionary with extracted values
    """
    if data is None or 'trajectories' not in data:
        return None

    alpha_values = []
    D_values = []
    D_generalized_values = []
    diffusion_types = []
    r_squared_values = []
    track_lengths = []

    for traj in data['trajectories']:
        # Quality filtering
        if (traj['alpha_r_squared'] >= min_r2 and
            traj['track_length'] >= min_length):

            alpha_values.append(traj['alpha'])
            D_values.append(traj['D_original'])
            D_generalized_values.append(traj['D_generalized'])
            diffusion_types.append(traj['diffusion_type'])
            r_squared_values.append(traj['alpha_r_squared'])
            track_lengths.append(traj['track_length'])

    return {
        'alpha': np.array(alpha_values),
        'D_original': np.array(D_values),
        'D_generalized': np.array(D_generalized_values),
        'diffusion_types': diffusion_types,
        'r_squared': np.array(r_squared_values),
        'track_lengths': np.array(track_lengths),
        'n_trajectories': len(alpha_values)
    }


def load_condition_data(file_paths, condition_name):
    """
    Load and pool data from multiple files for a single condition.

    Args:
        file_paths: List of file paths for this condition
        condition_name: Name of the condition

    Returns:
        Dictionary with pooled data
    """
    print(f"\nLoading condition: {condition_name}")
    print(f"  Files: {len(file_paths)}")

    all_alpha = []
    all_D = []
    all_D_gen = []
    all_types = []
    all_r2 = []
    all_lengths = []

    for file_path in file_paths:
        data = load_alpha_analyzed_file(file_path)
        if data is not None:
            extracted = extract_alpha_values(data)
            if extracted is not None and extracted['n_trajectories'] > 0:
                all_alpha.extend(extracted['alpha'])
                all_D.extend(extracted['D_original'])
                all_D_gen.extend(extracted['D_generalized'])
                all_types.extend(extracted['diffusion_types'])
                all_r2.extend(extracted['r_squared'])
                all_lengths.extend(extracted['track_lengths'])

    if len(all_alpha) == 0:
        print(f"  WARNING: No valid trajectories found for {condition_name}")
        return None

    # Count diffusion types
    type_counts = {
        'normal': all_types.count('normal'),
        'sub-diffusion': all_types.count('sub-diffusion'),
        'super-diffusion': all_types.count('super-diffusion'),
        'confined': all_types.count('confined')
    }

    print(f"  Total trajectories: {len(all_alpha)}")
    print(f"  Alpha: {np.mean(all_alpha):.3f} ± {np.std(all_alpha):.3f}")

    return {
        'condition_name': condition_name,
        'alpha': np.array(all_alpha),
        'D_original': np.array(all_D),
        'D_generalized': np.array(all_D_gen),
        'diffusion_types': all_types,
        'r_squared': np.array(all_r2),
        'track_lengths': np.array(all_lengths),
        'n_trajectories': len(all_alpha),
        'n_files': len(file_paths),
        'diffusion_type_counts': type_counts
    }


# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def calculate_bootstrap_ci(data, n_bootstrap=N_BOOTSTRAP, ci=BOOTSTRAP_CI):
    """
    Calculate bootstrap confidence interval for mean.

    Args:
        data: Array of values
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval percentage

    Returns:
        Dictionary with mean, CI, and SEM
    """
    if len(data) == 0:
        return {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'sem': np.nan}

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    ci_lower = np.percentile(bootstrap_means, (100 - ci) / 2)
    ci_upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)

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
    greater = 0
    less = 0

    for val1 in group1:
        for val2 in group2:
            if val1 > val2:
                greater += 1
            elif val1 < val2:
                less += 1

    delta = (greater - less) / (n1 * n2)
    return delta


def interpret_cliffs_delta(delta):
    """
    Interpret Cliff's Delta magnitude.

    Args:
        delta: Cliff's Delta value

    Returns:
        String interpretation
    """
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"


def compare_two_conditions(cond1, cond2):
    """
    Perform statistical comparison between two conditions.

    Args:
        cond1: Data dictionary for condition 1
        cond2: Data dictionary for condition 2

    Returns:
        Dictionary with comparison results
    """
    # Mann-Whitney U test (non-parametric)
    statistic, p_value = stats.mannwhitneyu(cond1['alpha'], cond2['alpha'],
                                           alternative='two-sided')

    # Effect size (Cliff's Delta)
    delta = calculate_cliffs_delta(cond1['alpha'], cond2['alpha'])
    delta_interp = interpret_cliffs_delta(delta)

    # Bootstrap confidence intervals
    cond1_stats = calculate_bootstrap_ci(cond1['alpha'])
    cond2_stats = calculate_bootstrap_ci(cond2['alpha'])

    return {
        'condition_1': cond1['condition_name'],
        'condition_2': cond2['condition_name'],
        'n_traj_1': cond1['n_trajectories'],
        'n_traj_2': cond2['n_trajectories'],
        'mean_alpha_1': cond1_stats['mean'],
        'mean_alpha_2': cond2_stats['mean'],
        'median_alpha_1': cond1_stats['median'],
        'median_alpha_2': cond2_stats['median'],
        'diff_mean': cond1_stats['mean'] - cond2_stats['mean'],
        'diff_median': cond1_stats['median'] - cond2_stats['median'],
        'mann_whitney_u': statistic,
        'p_value': p_value,
        'significant': p_value < ALPHA_SIGNIFICANCE,
        'cliffs_delta': delta,
        'effect_size_interpretation': delta_interp
    }


def compare_multiple_conditions(conditions_data):
    """
    Perform Kruskal-Wallis test for multiple conditions.

    Args:
        conditions_data: List of condition data dictionaries

    Returns:
        Dictionary with test results
    """
    if len(conditions_data) < 2:
        return None

    # Prepare data for Kruskal-Wallis test
    groups = [cond['alpha'] for cond in conditions_data]

    # Kruskal-Wallis H-test
    h_statistic, p_value = stats.kruskal(*groups)

    return {
        'test': 'Kruskal-Wallis',
        'h_statistic': h_statistic,
        'p_value': p_value,
        'significant': p_value < ALPHA_SIGNIFICANCE,
        'n_groups': len(conditions_data)
    }


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_comparison_plots(conditions_data, output_dir):
    """
    Create comprehensive comparison plots.

    Args:
        conditions_data: List of condition data dictionaries
        output_dir: Directory for output files
    """
    # Set style
    sns.set_style(PLOT_STYLE)

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    condition_names = [cond['condition_name'] for cond in conditions_data]
    n_conditions = len(conditions_data)

    # Panel 1: Box plot
    ax1 = fig.add_subplot(gs[0, 0])
    alpha_data = [cond['alpha'] for cond in conditions_data]

    bp = ax1.boxplot(alpha_data, labels=condition_names, patch_artist=True,
                     showfliers=True, widths=0.6)

    # Color boxes
    colors = sns.color_palette(PLOT_PALETTE, n_conditions)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5,
               label='Normal diffusion')
    ax1.set_ylabel('Alpha Exponent', fontsize=12, fontweight='bold')
    ax1.set_title('Alpha Distribution by Condition', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 2: Violin plot
    ax2 = fig.add_subplot(gs[0, 1])

    # Prepare data for violin plot
    plot_data = []
    for cond in conditions_data:
        for alpha_val in cond['alpha']:
            plot_data.append({
                'Condition': cond['condition_name'],
                'Alpha': alpha_val
            })
    plot_df = pd.DataFrame(plot_data)

    sns.violinplot(data=plot_df, x='Condition', y='Alpha', ax=ax2,
                  palette=PLOT_PALETTE, inner='box')
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5,
               label='Normal diffusion')
    ax2.set_ylabel('Alpha Exponent', fontsize=12, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_title('Alpha Distribution (Violin Plot)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 3: CDF comparison
    ax3 = fig.add_subplot(gs[0, 2])

    for cond, color in zip(conditions_data, colors):
        sorted_alpha = np.sort(cond['alpha'])
        cdf = np.arange(1, len(sorted_alpha) + 1) / len(sorted_alpha)
        ax3.plot(sorted_alpha, cdf, linewidth=2.5, label=cond['condition_name'],
                color=color, alpha=0.8)

    ax3.axvline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5,
               label='Normal diffusion')
    ax3.set_xlabel('Alpha Exponent', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax3.set_title('Cumulative Distribution Functions', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Mean ± CI bar plot
    ax4 = fig.add_subplot(gs[1, 0])

    means = []
    cis_lower = []
    cis_upper = []

    for cond in conditions_data:
        stats_result = calculate_bootstrap_ci(cond['alpha'])
        means.append(stats_result['mean'])
        cis_lower.append(stats_result['mean'] - stats_result['ci_lower'])
        cis_upper.append(stats_result['ci_upper'] - stats_result['mean'])

    x_pos = np.arange(n_conditions)
    bars = ax4.bar(x_pos, means, yerr=[cis_lower, cis_upper], capsize=8,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    ax4.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5,
               label='Normal diffusion')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(condition_names, rotation=45, ha='right')
    ax4.set_ylabel('Mean Alpha ± 95% CI', fontsize=12, fontweight='bold')
    ax4.set_title('Mean Alpha with Bootstrap CI', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Panel 5: Diffusion type distribution (stacked bar)
    ax5 = fig.add_subplot(gs[1, 1])

    type_categories = ['confined', 'sub-diffusion', 'normal', 'super-diffusion']
    type_colors = ['#ff6b6b', '#ffd93d', '#6bcf7f', '#4d96ff']

    # Calculate percentages
    type_percentages = {cat: [] for cat in type_categories}

    for cond in conditions_data:
        total = cond['n_trajectories']
        for cat in type_categories:
            count = cond['diffusion_type_counts'][cat]
            pct = (count / total * 100) if total > 0 else 0
            type_percentages[cat].append(pct)

    # Create stacked bar chart
    bottom = np.zeros(n_conditions)
    for cat, color in zip(type_categories, type_colors):
        ax5.bar(condition_names, type_percentages[cat], bottom=bottom,
               label=cat.capitalize(), color=color, alpha=0.8, edgecolor='black')
        bottom += type_percentages[cat]

    ax5.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Diffusion Type Distribution', fontsize=13, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 6: Histogram overlay
    ax6 = fig.add_subplot(gs[1, 2])

    for cond, color in zip(conditions_data, colors):
        ax6.hist(cond['alpha'], bins=30, alpha=0.5, label=cond['condition_name'],
                color=color, edgecolor='black', density=True)

    ax6.axvline(1.0, color='red', linestyle='--', linewidth=2.5,
               label='Normal diffusion')
    ax6.set_xlabel('Alpha Exponent', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax6.set_title('Alpha Distribution Overlay', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # Panel 7: Sample size comparison
    ax7 = fig.add_subplot(gs[2, 0])

    n_traj = [cond['n_trajectories'] for cond in conditions_data]
    bars = ax7.bar(condition_names, n_traj, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.5)

    # Add numbers on bars
    for bar, n in zip(bars, n_traj):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(n)}', ha='center', va='bottom', fontweight='bold')

    ax7.set_ylabel('Number of Trajectories', fontsize=12, fontweight='bold')
    ax7.set_title('Sample Size per Condition', fontsize=13, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 8: Alpha vs D correlation by condition
    ax8 = fig.add_subplot(gs[2, 1])

    for cond, color in zip(conditions_data, colors):
        # Filter valid D values
        valid_idx = np.isfinite(cond['D_original']) & (cond['D_original'] > 0)
        D_valid = cond['D_original'][valid_idx]
        alpha_valid = cond['alpha'][valid_idx]

        if len(D_valid) > 0:
            ax8.scatter(D_valid, alpha_valid, alpha=0.5, s=30,
                       label=cond['condition_name'], color=color)

    ax8.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5,
               label='Normal diffusion')
    ax8.set_xlabel('Diffusion Coefficient D (μm²/s)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Alpha Exponent', fontsize=12, fontweight='bold')
    ax8.set_title('Alpha vs D by Condition', fontsize=13, fontweight='bold')
    ax8.set_xscale('log')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    # Panel 9: Statistical summary text
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    # Perform Kruskal-Wallis test
    kw_result = compare_multiple_conditions(conditions_data)

    summary_text = "STATISTICAL SUMMARY\n" + "="*30 + "\n\n"

    # Overall test
    if kw_result:
        summary_text += f"Kruskal-Wallis Test:\n"
        summary_text += f"  H = {kw_result['h_statistic']:.3f}\n"
        summary_text += f"  p = {kw_result['p_value']:.4f}\n"
        if kw_result['significant']:
            summary_text += "  Result: SIGNIFICANT ✓\n"
        else:
            summary_text += "  Result: Not significant\n"
        summary_text += "\n"

    # Condition summaries
    summary_text += "Condition Means (±SEM):\n"
    for cond in conditions_data:
        stats_result = calculate_bootstrap_ci(cond['alpha'])
        summary_text += f"\n{cond['condition_name']}:\n"
        summary_text += f"  α = {stats_result['mean']:.3f} ± {stats_result['sem']:.3f}\n"
        summary_text += f"  n = {cond['n_trajectories']}\n"

        # Classify overall diffusion type
        mean_alpha = stats_result['mean']
        if mean_alpha < 0.9:
            summary_text += f"  Type: Sub-diffusion\n"
        elif mean_alpha <= 1.1:
            summary_text += f"  Type: Normal\n"
        else:
            summary_text += f"  Type: Super-diffusion\n"

    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
            verticalalignment='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Overall title
    fig.suptitle('Multi-Condition Alpha Exponent Comparison', fontsize=16, fontweight='bold')

    # Save
    plot_path = os.path.join(output_dir, 'alpha_comparison_all_conditions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved comparison plots: {plot_path}")


def create_pairwise_comparison_plot(conditions_data, output_dir):
    """
    Create pairwise comparison matrix plot.

    Args:
        conditions_data: List of condition data dictionaries
        output_dir: Directory for output files
    """
    n_conditions = len(conditions_data)

    if n_conditions < 2:
        return

    # Create comparison matrix
    fig, axes = plt.subplots(n_conditions, n_conditions, figsize=(4*n_conditions, 4*n_conditions))

    if n_conditions == 1:
        axes = np.array([[axes]])
    elif n_conditions == 2:
        axes = axes.reshape(2, 2)

    for i in range(n_conditions):
        for j in range(n_conditions):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram
                cond = conditions_data[i]
                ax.hist(cond['alpha'], bins=30, alpha=0.7, edgecolor='black', color='steelblue')
                ax.axvline(1.0, color='red', linestyle='--', linewidth=2)
                ax.set_title(cond['condition_name'], fontweight='bold')
                ax.set_ylabel('Frequency')
                ax.set_xlabel('Alpha')
            else:
                # Off-diagonal: scatter plot
                cond_x = conditions_data[j]
                cond_y = conditions_data[i]

                # Create paired scatter (random pairing for visualization)
                n_points = min(len(cond_x['alpha']), len(cond_y['alpha']))
                ax.scatter(cond_x['alpha'][:n_points], cond_y['alpha'][:n_points],
                          alpha=0.4, s=20, color='steelblue')

                # Add diagonal line
                ax.plot([0, 2.5], [0, 2.5], 'r--', linewidth=1.5, alpha=0.5)

                # Perform comparison
                comparison = compare_two_conditions(cond_x, cond_y)

                # Add p-value text
                if comparison['significant']:
                    text_color = 'red'
                    sig_text = 'p < 0.05 *'
                else:
                    text_color = 'black'
                    sig_text = 'n.s.'

                ax.text(0.05, 0.95, f"{sig_text}\nδ={comparison['cliffs_delta']:.2f}",
                       transform=ax.transAxes, verticalalignment='top',
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                       color=text_color, fontweight='bold')

                ax.set_xlabel(cond_x['condition_name'])
                ax.set_ylabel(cond_y['condition_name'])

            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'alpha_pairwise_comparison_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved pairwise comparison: {plot_path}")


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_summary_statistics(conditions_data, output_dir):
    """
    Export summary statistics to CSV.

    Args:
        conditions_data: List of condition data dictionaries
        output_dir: Directory for output file
    """
    summary_rows = []

    for cond in conditions_data:
        stats_result = calculate_bootstrap_ci(cond['alpha'])

        row = {
            'condition': cond['condition_name'],
            'n_trajectories': cond['n_trajectories'],
            'n_files': cond['n_files'],

            # Alpha statistics
            'alpha_mean': stats_result['mean'],
            'alpha_median': stats_result['median'],
            'alpha_std': stats_result['std'],
            'alpha_sem': stats_result['sem'],
            'alpha_ci_lower': stats_result['ci_lower'],
            'alpha_ci_upper': stats_result['ci_upper'],

            # Diffusion type counts
            'n_normal': cond['diffusion_type_counts']['normal'],
            'n_sub_diffusion': cond['diffusion_type_counts']['sub-diffusion'],
            'n_super_diffusion': cond['diffusion_type_counts']['super-diffusion'],
            'n_confined': cond['diffusion_type_counts']['confined'],

            # Diffusion type percentages
            'pct_normal': cond['diffusion_type_counts']['normal'] / cond['n_trajectories'] * 100,
            'pct_sub_diffusion': cond['diffusion_type_counts']['sub-diffusion'] / cond['n_trajectories'] * 100,
            'pct_super_diffusion': cond['diffusion_type_counts']['super-diffusion'] / cond['n_trajectories'] * 100,
            'pct_confined': cond['diffusion_type_counts']['confined'] / cond['n_trajectories'] * 100,

            # D statistics (if available)
            'D_original_mean': np.nanmean(cond['D_original']),
            'D_generalized_mean': np.nanmean(cond['D_generalized'])
        }

        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_dir, 'alpha_summary_by_condition.csv')
    df.to_csv(csv_path, index=False)

    print(f"Exported summary statistics: {csv_path}")


def export_pairwise_comparisons(conditions_data, output_dir):
    """
    Export pairwise comparison results to CSV.

    Args:
        conditions_data: List of condition data dictionaries
        output_dir: Directory for output file
    """
    n_conditions = len(conditions_data)

    if n_conditions < 2:
        print("Need at least 2 conditions for pairwise comparisons")
        return

    comparison_rows = []

    for i in range(n_conditions):
        for j in range(i+1, n_conditions):
            comparison = compare_two_conditions(conditions_data[i], conditions_data[j])
            comparison_rows.append(comparison)

    df = pd.DataFrame(comparison_rows)
    csv_path = os.path.join(output_dir, 'alpha_pairwise_comparisons.csv')
    df.to_csv(csv_path, index=False)

    print(f"Exported pairwise comparisons: {csv_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function for multi-condition alpha comparison.
    """
    print("="*70)
    print("ALPHA EXPONENT MULTI-CONDITION COMPARISON")
    print("="*70)

    # Get user input
    print("\nThis script compares alpha values across multiple conditions.")
    print("Each condition can have multiple replicate files.")
    print("\nYou can organize files by:")
    print("  1. Manual grouping (enter file paths for each condition)")
    print("  2. Pattern matching (e.g., control_*.pkl, treatment_*.pkl)")

    mode = input("\nChoose mode (1=Manual, 2=Pattern): ").strip()

    conditions_data = []

    if mode == "1":
        # Manual grouping
        n_conditions = int(input("How many conditions to compare? "))

        for i in range(n_conditions):
            print(f"\n--- Condition {i+1} ---")
            condition_name = input(f"Enter name for condition {i+1}: ").strip()

            file_list_input = input(f"Enter file paths (comma-separated or directory): ").strip()

            # Check if it's a directory
            if os.path.isdir(file_list_input):
                file_paths = glob.glob(os.path.join(file_list_input, "alpha_analyzed_*.pkl"))
            else:
                # Parse comma-separated list
                file_paths = [f.strip() for f in file_list_input.split(',')]

            if file_paths:
                cond_data = load_condition_data(file_paths, condition_name)
                if cond_data is not None:
                    conditions_data.append(cond_data)

    else:
        # Pattern matching
        base_dir = input("\nEnter base directory containing all files: ").strip()

        if not os.path.isdir(base_dir):
            print(f"Error: {base_dir} is not a valid directory")
            return

        print("\nExample patterns:")
        print("  control_*.pkl")
        print("  treatment1_*.pkl")
        print("  *_condition1_*.pkl")

        n_conditions = int(input("\nHow many conditions? "))

        for i in range(n_conditions):
            print(f"\n--- Condition {i+1} ---")
            condition_name = input(f"Enter name for condition {i+1}: ").strip()
            pattern = input(f"Enter file pattern (e.g., control_*.pkl): ").strip()

            # Find matching files
            search_pattern = os.path.join(base_dir, pattern)
            file_paths = glob.glob(search_pattern)

            if not file_paths:
                print(f"  WARNING: No files found matching pattern: {pattern}")
            else:
                cond_data = load_condition_data(file_paths, condition_name)
                if cond_data is not None:
                    conditions_data.append(cond_data)

    # Check if we have enough data
    if len(conditions_data) < 2:
        print("\nError: Need at least 2 conditions with valid data for comparison!")
        return

    # Get output directory
    output_dir = input("\nEnter output directory (press Enter for current): ").strip()
    if not output_dir:
        output_dir = os.getcwd()

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PERFORMING COMPARISONS")
    print(f"{'='*70}")
    print(f"Conditions: {len(conditions_data)}")
    print(f"Output directory: {output_dir}")

    # Perform statistical comparisons
    print("\nCalculating statistics...")
    export_summary_statistics(conditions_data, output_dir)

    if len(conditions_data) >= 2:
        export_pairwise_comparisons(conditions_data, output_dir)

    # Create plots
    print("\nCreating comparison plots...")
    create_comparison_plots(conditions_data, output_dir)

    if len(conditions_data) >= 2 and len(conditions_data) <= 5:
        create_pairwise_comparison_plot(conditions_data, output_dir)

    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*70}")

    # Print summary
    print("\nSUMMARY:")
    for cond in conditions_data:
        stats_result = calculate_bootstrap_ci(cond['alpha'])
        print(f"\n{cond['condition_name']}:")
        print(f"  Mean α: {stats_result['mean']:.3f} ± {stats_result['sem']:.3f}")
        print(f"  95% CI: [{stats_result['ci_lower']:.3f}, {stats_result['ci_upper']:.3f}]")
        print(f"  n = {cond['n_trajectories']} trajectories")

    # Overall test
    kw_result = compare_multiple_conditions(conditions_data)
    if kw_result:
        print(f"\nKruskal-Wallis Test:")
        print(f"  p = {kw_result['p_value']:.4f}")
        if kw_result['significant']:
            print(f"  Result: Conditions are SIGNIFICANTLY different")
        else:
            print(f"  Result: No significant difference detected")


if __name__ == "__main__":
    main()
