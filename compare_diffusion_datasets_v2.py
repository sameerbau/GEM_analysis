#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_diffusion_datasets_v2.py

Enhanced version integrating with diffusion_analyzer_v4_validation.py
Supports hierarchical sample organization, quality-based filtering,
and multi-metric comparisons.

NEW IN V2:
- Hierarchical sample structure support (Sample_A/, Sample_B/)
- Quality-based filtering (PASS/WARNING/FAIL from v4 validation)
- Multi-mode comparison (compare different quality filters)
- Extended metrics: D, σ_loc, R², track length, CI width
- Quality-aware visualizations
- Quality distribution comparison

Input:
- Sample directories containing analyzed trajectory .pkl files from v4_validation
- Each sample can have multiple images/analyzed files

Output:
- Statistical comparisons for multiple metrics
- Quality-aware visualizations
- Multi-mode comparison results (all vs PASS-only vs PASS+WARNING)
- CSV exports with quality flags

Usage:
python compare_diffusion_datasets_v2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
from pathlib import Path
from scipy import stats
from datetime import datetime
from itertools import combinations

# =====================================
# GLOBAL PARAMETERS
# =====================================

# Quality filtering modes
QUALITY_FILTER_MODE = 'all'  # Options: 'all', 'pass_only', 'pass_warning', 'multi_mode'
# 'all': Include all trajectories regardless of quality
# 'pass_only': Only include PASS quality trajectories
# 'pass_warning': Include PASS + WARNING quality trajectories
# 'multi_mode': Compare across all three modes

# Filtering parameters (applied after quality filtering)
MIN_R_SQUARED = 0.7  # Minimum R² value (consistent with v4 validation)
MIN_TRACK_LENGTH = 10  # Minimum track length (in frames)
MAX_DIFFUSION_COEFFICIENT = None  # Maximum D (μm²/s), None = no limit

# Statistical parameters
ALPHA = 0.05  # Significance level
SUBSAMPLE_SIZE = 100  # Number of trajectories per subsample
N_SUBSAMPLES = 50  # Number of subsampling iterations

# Plot parameters
FIGURE_SIZE = (12, 8)
FIGURE_SIZE_LARGE = (18, 12)
FIGURE_DPI = 300

# Quality color scheme
QUALITY_COLORS = {
    'PASS': 'green',
    'WARNING': 'orange',
    'FAIL': 'red',
    'all': 'blue'
}
# =====================================


def load_analyzed_data(file_path):
    """Load analyzed trajectory data from v4 validation pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_quality_flag(trajectory):
    """
    Extract quality flag from trajectory.
    Handles both v4 format and older formats.
    """
    if 'validation' in trajectory and trajectory['validation'] is not None:
        return trajectory['validation'].get('quality', 'UNKNOWN')
    else:
        # Fallback: infer quality from metrics
        if np.isnan(trajectory.get('D', np.nan)):
            return 'FAIL'
        r_squared = trajectory.get('r_squared', 0)
        if r_squared >= 0.8:
            return 'PASS'
        elif r_squared >= 0.7:
            return 'WARNING'
        else:
            return 'FAIL'


def filter_trajectories_by_quality(trajectories, quality_mode='all',
                                   min_r_squared=MIN_R_SQUARED,
                                   min_track_length=MIN_TRACK_LENGTH,
                                   max_diffusion=MAX_DIFFUSION_COEFFICIENT):
    """
    Filter trajectories based on quality flags and additional criteria.

    Args:
        trajectories: List of trajectory dictionaries
        quality_mode: 'all', 'pass_only', 'pass_warning'
        min_r_squared: Minimum R² value
        min_track_length: Minimum track length
        max_diffusion: Maximum diffusion coefficient

    Returns:
        List of filtered trajectories
    """
    filtered = []

    for traj in trajectories:
        # Skip invalid trajectories
        if np.isnan(traj.get('D', np.nan)):
            continue

        # Quality filtering
        quality = extract_quality_flag(traj)

        if quality_mode == 'pass_only' and quality != 'PASS':
            continue
        elif quality_mode == 'pass_warning' and quality not in ['PASS', 'WARNING']:
            continue
        # 'all' mode: no quality filtering

        # Additional filters
        if traj.get('r_squared', 0) < min_r_squared:
            continue

        if traj.get('track_length', 0) < min_track_length:
            continue

        if max_diffusion is not None and traj['D'] > max_diffusion:
            continue

        filtered.append(traj)

    return filtered


def load_sample_hierarchical(sample_dir, sample_name, quality_mode='all'):
    """
    Load all analyzed files from a sample directory.
    Supports hierarchical structure: Sample_A/analyzed_*.pkl

    Args:
        sample_dir: Directory containing analyzed .pkl files
        sample_name: Name for this sample
        quality_mode: Quality filtering mode

    Returns:
        Dictionary with sample data
    """
    print(f"\n  Loading sample: {sample_name}")
    print(f"    Directory: {sample_dir}")
    print(f"    Quality mode: {quality_mode}")

    # Find all analyzed files
    file_pattern = os.path.join(sample_dir, "analyzed_*.pkl")
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        print(f"    Warning: No analyzed_*.pkl files found")
        return None

    print(f"    Found {len(file_paths)} analyzed files")

    # Pool data from all files
    all_trajectories = []
    file_sources = []
    quality_counts = {'PASS': 0, 'WARNING': 0, 'FAIL': 0, 'UNKNOWN': 0}

    for file_path in file_paths:
        analyzed_data = load_analyzed_data(file_path)

        if analyzed_data is None:
            continue

        trajectories = analyzed_data.get('trajectories', [])

        # Count quality before filtering
        for traj in trajectories:
            quality = extract_quality_flag(traj)
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

        # Filter trajectories
        filtered_trajs = filter_trajectories_by_quality(
            trajectories, quality_mode
        )

        all_trajectories.extend(filtered_trajs)
        file_sources.extend([os.path.basename(file_path)] * len(filtered_trajs))

    if not all_trajectories:
        print(f"    Warning: No trajectories passed filtering")
        return None

    # Extract metrics
    data = {
        'name': sample_name,
        'quality_mode': quality_mode,
        'n_files': len(file_paths),
        'quality_counts_original': quality_counts,
        'trajectories': all_trajectories,
        'file_sources': file_sources,
        'D': np.array([t['D'] for t in all_trajectories]),
        'D_err': np.array([t.get('D_CI_high', t['D']) - t['D'] for t in all_trajectories]),
        'sigma_loc': np.array([t.get('sigma_loc', np.nan) * 1000 for t in all_trajectories]),  # nm
        'r_squared': np.array([t.get('r_squared', np.nan) for t in all_trajectories]),
        'track_length': np.array([t.get('track_length', np.nan) for t in all_trajectories]),
        'radius_gyration': np.array([t.get('radius_of_gyration', np.nan) for t in all_trajectories]),
        'quality_flags': np.array([extract_quality_flag(t) for t in all_trajectories])
    }

    # Calculate CI widths
    ci_widths = []
    for t in all_trajectories:
        if 'D_CI_low' in t and 'D_CI_high' in t:
            ci_widths.append(t['D_CI_high'] - t['D_CI_low'])
        else:
            ci_widths.append(np.nan)
    data['D_CI_width'] = np.array(ci_widths)

    print(f"    Loaded {len(all_trajectories)} trajectories after filtering")
    print(f"    Quality (before filtering): PASS={quality_counts['PASS']}, "
          f"WARNING={quality_counts['WARNING']}, FAIL={quality_counts['FAIL']}")

    return data


def compare_quality_distributions(datasets):
    """
    Compare quality flag distributions across datasets.
    Uses Chi-square test.
    """
    sample_names = [d['name'] for d in datasets]

    # Get original quality counts
    quality_data = []
    for dataset in datasets:
        counts = dataset['quality_counts_original']
        quality_data.append([
            counts.get('PASS', 0),
            counts.get('WARNING', 0),
            counts.get('FAIL', 0)
        ])

    quality_array = np.array(quality_data)

    # Chi-square test
    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(quality_array)

        result = {
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof,
            'significant': p_value < ALPHA,
            'quality_distributions': {
                name: dict(zip(['PASS', 'WARNING', 'FAIL'], counts))
                for name, counts in zip(sample_names, quality_data)
            }
        }

        return result
    except Exception as e:
        print(f"Warning: Could not perform chi-square test: {e}")
        return None


def calculate_effect_size(group1, group2):
    """Calculate Cliff's delta effect size."""
    if len(group1) == 0 or len(group2) == 0:
        return np.nan

    greater = lesser = 0
    for x in group1:
        for y in group2:
            if x > y:
                greater += 1
            elif x < y:
                lesser += 1

    total = len(group1) * len(group2)
    return (greater - lesser) / total


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


def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    if len(group1) == 0 or len(group2) == 0:
        return np.nan

    mean1, mean2 = np.mean(group1), np.mean(group2)
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    return (mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan


def compare_metric(data1, data2, metric_name):
    """
    Compare a specific metric between two datasets.

    Returns statistical test results.
    """
    # Remove NaN values
    values1 = data1[~np.isnan(data1)]
    values2 = data2[~np.isnan(data2)]

    if len(values1) < 3 or len(values2) < 3:
        return None

    try:
        # Statistical tests
        u_stat, p_mw = stats.mannwhitneyu(values1, values2, alternative='two-sided')
        ks_stat, p_ks = stats.ks_2samp(values1, values2)

        # Effect sizes
        cliffs_delta = calculate_effect_size(values1, values2)
        cohens_d = calculate_cohens_d(values1, values2)

        result = {
            'metric': metric_name,
            'n1': len(values1),
            'n2': len(values2),
            'mean1': np.mean(values1),
            'mean2': np.mean(values2),
            'median1': np.median(values1),
            'median2': np.median(values2),
            'std1': np.std(values1),
            'std2': np.std(values2),
            'mann_whitney_u': u_stat,
            'mann_whitney_p': p_mw,
            'ks_statistic': ks_stat,
            'ks_p': p_ks,
            'cliffs_delta': cliffs_delta,
            'cliffs_delta_interpretation': interpret_cliffs_delta(cliffs_delta),
            'cohens_d': cohens_d,
            'significant_mw': p_mw < ALPHA,
            'significant_ks': p_ks < ALPHA
        }

        return result
    except Exception as e:
        print(f"Error comparing {metric_name}: {e}")
        return None


def compare_datasets_multi_metric(datasets):
    """
    Compare datasets across multiple metrics.

    Compares: D, σ_loc, R², track_length, CI_width
    """
    comparison_results = {
        'pairwise_comparisons': [],
        'metrics': ['D', 'sigma_loc', 'r_squared', 'track_length', 'D_CI_width']
    }

    # Get all dataset pairs
    dataset_pairs = list(combinations(range(len(datasets)), 2))

    for i, j in dataset_pairs:
        dataset1 = datasets[i]
        dataset2 = datasets[j]

        print(f"\n  Comparing: {dataset1['name']} vs {dataset2['name']}")

        comparison = {
            'dataset1': dataset1['name'],
            'dataset2': dataset2['name'],
            'quality_mode': dataset1['quality_mode'],
            'metrics': {}
        }

        # Compare each metric
        for metric in comparison_results['metrics']:
            result = compare_metric(dataset1[metric], dataset2[metric], metric)
            if result is not None:
                comparison['metrics'][metric] = result

                if result['significant_mw']:
                    print(f"    {metric}: SIGNIFICANT (p={result['mann_whitney_p']:.4e}, "
                          f"effect={result['cliffs_delta_interpretation']})")
                else:
                    print(f"    {metric}: Not significant (p={result['mann_whitney_p']:.4f})")

        comparison_results['pairwise_comparisons'].append(comparison)

    return comparison_results


def subsampling_analysis(data1, data2, name1, name2,
                         subsample_size=SUBSAMPLE_SIZE,
                         n_subsamples=N_SUBSAMPLES):
    """Perform subsampling analysis for robustness assessment."""
    if len(data1) < subsample_size or len(data2) < subsample_size:
        return None

    results = {
        'dataset1': name1,
        'dataset2': name2,
        'p_values': [],
        'effect_sizes': []
    }

    for _ in range(n_subsamples):
        indices1 = np.random.choice(len(data1), subsample_size, replace=False)
        indices2 = np.random.choice(len(data2), subsample_size, replace=False)

        subsample1 = data1[indices1]
        subsample2 = data2[indices2]

        try:
            _, p_value = stats.mannwhitneyu(subsample1, subsample2, alternative='two-sided')
            effect_size = calculate_effect_size(subsample1, subsample2)

            results['p_values'].append(p_value)
            results['effect_sizes'].append(effect_size)
        except:
            continue

    if len(results['p_values']) > 0:
        results['mean_p_value'] = np.mean(results['p_values'])
        results['std_p_value'] = np.std(results['p_values'])
        results['mean_effect_size'] = np.mean(results['effect_sizes'])
        results['std_effect_size'] = np.std(results['effect_sizes'])
        results['prop_significant'] = np.mean(np.array(results['p_values']) < ALPHA)
        return results

    return None


def plot_quality_distribution_comparison(datasets, output_path):
    """Plot quality flag distributions across samples."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

    # Prepare data
    sample_names = [d['name'] for d in datasets]
    pass_counts = [d['quality_counts_original'].get('PASS', 0) for d in datasets]
    warning_counts = [d['quality_counts_original'].get('WARNING', 0) for d in datasets]
    fail_counts = [d['quality_counts_original'].get('FAIL', 0) for d in datasets]

    # Stacked bar chart
    x_pos = np.arange(len(sample_names))
    ax1.bar(x_pos, pass_counts, label='PASS', color=QUALITY_COLORS['PASS'], alpha=0.8)
    ax1.bar(x_pos, warning_counts, bottom=pass_counts,
            label='WARNING', color=QUALITY_COLORS['WARNING'], alpha=0.8)
    ax1.bar(x_pos, fail_counts,
            bottom=np.array(pass_counts) + np.array(warning_counts),
            label='FAIL', color=QUALITY_COLORS['FAIL'], alpha=0.8)

    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Number of trajectories')
    ax1.set_title('Quality Distribution by Sample (Before Filtering)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sample_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Normalized stacked bar
    totals = np.array(pass_counts) + np.array(warning_counts) + np.array(fail_counts)
    pass_pct = np.array(pass_counts) / totals * 100
    warning_pct = np.array(warning_counts) / totals * 100
    fail_pct = np.array(fail_counts) / totals * 100

    ax2.bar(x_pos, pass_pct, label='PASS', color=QUALITY_COLORS['PASS'], alpha=0.8)
    ax2.bar(x_pos, warning_pct, bottom=pass_pct,
            label='WARNING', color=QUALITY_COLORS['WARNING'], alpha=0.8)
    ax2.bar(x_pos, fail_pct, bottom=pass_pct + warning_pct,
            label='FAIL', color=QUALITY_COLORS['FAIL'], alpha=0.8)

    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Quality Distribution by Sample (Normalized)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sample_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "quality_distribution_comparison.png"), dpi=FIGURE_DPI)
    plt.close()


def plot_multi_metric_comparison(datasets, output_path):
    """Create comprehensive multi-metric comparison plots."""
    metrics_info = {
        'D': {'label': 'Diffusion coefficient (μm²/s)', 'log': False},
        'sigma_loc': {'label': 'Localization error σ_loc (nm)', 'log': False},
        'r_squared': {'label': 'Fit quality (R²)', 'log': False},
        'track_length': {'label': 'Track length (frames)', 'log': False},
        'D_CI_width': {'label': 'D confidence interval width (μm²/s)', 'log': False}
    }

    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE_LARGE)
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    for idx, (metric, info) in enumerate(metrics_info.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Violin plot for each dataset
        data_to_plot = []
        labels = []

        for dataset in datasets:
            values = dataset[metric]
            values = values[~np.isnan(values)]
            if len(values) > 0:
                data_to_plot.append(values)
                labels.append(f"{dataset['name']}\n(n={len(values)})")

        if data_to_plot:
            parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)),
                                  showmeans=True, showmedians=True)

            # Color the violins
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel(info['label'])
            ax.set_title(f'{metric} Distribution')
            ax.grid(True, alpha=0.3)

            if info['log'] and all(len(d) > 0 and np.min(d) > 0 for d in data_to_plot):
                ax.set_yscale('log')

    # Remove extra subplot
    if len(metrics_info) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "multi_metric_violin_comparison.png"), dpi=FIGURE_DPI)
    plt.close()


def plot_quality_coded_scatter(datasets, output_path):
    """Create scatter plots with quality color coding."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for dataset in datasets:
        # Plot 1: D vs R² (quality colored)
        ax = axes[0, 0]
        for quality in ['PASS', 'WARNING', 'FAIL']:
            mask = dataset['quality_flags'] == quality
            if np.any(mask):
                ax.scatter(dataset['D'][mask], dataset['r_squared'][mask],
                          c=QUALITY_COLORS[quality], label=f"{dataset['name']}-{quality}",
                          alpha=0.6, s=30)

        # Plot 2: D vs σ_loc (quality colored)
        ax = axes[0, 1]
        for quality in ['PASS', 'WARNING', 'FAIL']:
            mask = dataset['quality_flags'] == quality
            if np.any(mask):
                ax.scatter(dataset['D'][mask], dataset['sigma_loc'][mask],
                          c=QUALITY_COLORS[quality], label=f"{dataset['name']}-{quality}",
                          alpha=0.6, s=30)

        # Plot 3: D vs Track length (quality colored)
        ax = axes[1, 0]
        for quality in ['PASS', 'WARNING', 'FAIL']:
            mask = dataset['quality_flags'] == quality
            if np.any(mask):
                ax.scatter(dataset['D'][mask], dataset['track_length'][mask],
                          c=QUALITY_COLORS[quality], label=f"{dataset['name']}-{quality}",
                          alpha=0.6, s=30)

        # Plot 4: D vs CI width (quality colored)
        ax = axes[1, 1]
        for quality in ['PASS', 'WARNING', 'FAIL']:
            mask = dataset['quality_flags'] == quality
            if np.any(mask):
                ax.scatter(dataset['D'][mask], dataset['D_CI_width'][mask],
                          c=QUALITY_COLORS[quality], label=f"{dataset['name']}-{quality}",
                          alpha=0.6, s=30)

    # Configure axes
    axes[0, 0].set_xlabel('D (μm²/s)')
    axes[0, 0].set_ylabel('R²')
    axes[0, 0].set_title('Diffusion vs Fit Quality')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('D (μm²/s)')
    axes[0, 1].set_ylabel('σ_loc (nm)')
    axes[0, 1].set_title('Diffusion vs Localization Error')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('D (μm²/s)')
    axes[1, 0].set_ylabel('Track length (frames)')
    axes[1, 0].set_title('Diffusion vs Track Length')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('D (μm²/s)')
    axes[1, 1].set_ylabel('CI width (μm²/s)')
    axes[1, 1].set_title('Diffusion vs Uncertainty')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "quality_coded_scatter_plots.png"), dpi=FIGURE_DPI)
    plt.close()


def plot_diffusion_comparison(datasets, output_path):
    """Create detailed diffusion coefficient comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    # Plot 1: KDE comparison
    ax = axes[0, 0]
    for i, dataset in enumerate(datasets):
        D_values = dataset['D'][~np.isnan(dataset['D'])]
        if len(D_values) > 0:
            sns.kdeplot(D_values, ax=ax, label=f"{dataset['name']} (n={len(D_values)})",
                       color=colors[i], alpha=0.7, linewidth=2)
            ax.axvline(np.median(D_values), color=colors[i], linestyle='--', alpha=0.7)

    ax.set_xlabel('Diffusion coefficient (μm²/s)')
    ax.set_ylabel('Density')
    ax.set_title('D Distribution (KDE)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: CDF comparison
    ax = axes[0, 1]
    for i, dataset in enumerate(datasets):
        D_sorted = np.sort(dataset['D'][~np.isnan(dataset['D'])])
        if len(D_sorted) > 0:
            cdf = np.arange(1, len(D_sorted) + 1) / len(D_sorted)
            ax.plot(D_sorted, cdf, label=dataset['name'], color=colors[i], linewidth=2)

    ax.set_xlabel('Diffusion coefficient (μm²/s)')
    ax.set_ylabel('Cumulative probability')
    ax.set_title('Cumulative Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Box plot
    ax = axes[1, 0]
    data_to_plot = [dataset['D'][~np.isnan(dataset['D'])] for dataset in datasets]
    labels = [dataset['name'] for dataset in datasets]

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i])
        box.set_alpha(0.7)

    ax.set_ylabel('Diffusion coefficient (μm²/s)')
    ax.set_title('D Distribution (Box Plot)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    # Plot 4: Histogram comparison
    ax = axes[1, 1]
    all_D = np.concatenate([dataset['D'][~np.isnan(dataset['D'])] for dataset in datasets])
    bins = np.linspace(np.percentile(all_D, 1), np.percentile(all_D, 99), 30)

    for i, dataset in enumerate(datasets):
        D_values = dataset['D'][~np.isnan(dataset['D'])]
        ax.hist(D_values, bins=bins, alpha=0.5, color=colors[i],
               label=f"{dataset['name']} (n={len(D_values)})")

    ax.set_xlabel('Diffusion coefficient (μm²/s)')
    ax.set_ylabel('Count')
    ax.set_title('D Distribution (Histogram)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "diffusion_detailed_comparison.png"), dpi=FIGURE_DPI)
    plt.close()


def plot_effect_size_heatmap(comparison_results, output_path):
    """Create heatmap of effect sizes across metrics."""
    if not comparison_results['pairwise_comparisons']:
        return

    metrics = comparison_results['metrics']
    n_comparisons = len(comparison_results['pairwise_comparisons'])

    # Create matrix for effect sizes
    effect_matrix = np.zeros((n_comparisons, len(metrics)))
    comparison_labels = []

    for i, comp in enumerate(comparison_results['pairwise_comparisons']):
        comparison_labels.append(f"{comp['dataset1']}\nvs\n{comp['dataset2']}")

        for j, metric in enumerate(metrics):
            if metric in comp['metrics']:
                effect_matrix[i, j] = comp['metrics'][metric].get('cliffs_delta', 0)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, n_comparisons * 0.8)))

    im = ax.imshow(effect_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(n_comparisons))
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(comparison_labels, fontsize=9)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cliff's delta effect size", rotation=270, labelpad=20)

    # Add text annotations
    for i in range(n_comparisons):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{effect_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    ax.set_title('Effect Sizes Across Metrics (Cliff\'s Delta)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "effect_size_heatmap.png"), dpi=FIGURE_DPI)
    plt.close()


def export_comparison_results(comparison_results, datasets, quality_comparison, output_path):
    """Export comprehensive comparison results to CSV files."""

    # 1. Dataset summary
    summary_data = []
    for dataset in datasets:
        quality_counts = dataset['quality_counts_original']

        summary = {
            'Sample': dataset['name'],
            'Quality_Mode': dataset['quality_mode'],
            'N_Files': dataset['n_files'],
            'N_Trajectories': len(dataset['trajectories']),
            'N_PASS_Original': quality_counts.get('PASS', 0),
            'N_WARNING_Original': quality_counts.get('WARNING', 0),
            'N_FAIL_Original': quality_counts.get('FAIL', 0),
            'Median_D': np.median(dataset['D'][~np.isnan(dataset['D'])]),
            'Mean_D': np.mean(dataset['D'][~np.isnan(dataset['D'])]),
            'Std_D': np.std(dataset['D'][~np.isnan(dataset['D'])]),
            'Median_sigma_loc_nm': np.median(dataset['sigma_loc'][~np.isnan(dataset['sigma_loc'])]),
            'Median_R_squared': np.median(dataset['r_squared'][~np.isnan(dataset['r_squared'])]),
            'Median_Track_Length': np.median(dataset['track_length'][~np.isnan(dataset['track_length'])])
        }
        summary_data.append(summary)

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_path, "dataset_summary.csv"), index=False)

    # 2. Multi-metric pairwise comparisons
    for comp in comparison_results['pairwise_comparisons']:
        rows = []
        for metric_name, metric_data in comp['metrics'].items():
            row = {
                'Dataset1': comp['dataset1'],
                'Dataset2': comp['dataset2'],
                'Quality_Mode': comp['quality_mode'],
                'Metric': metric_name,
                **metric_data
            }
            rows.append(row)

        if rows:
            df_metrics = pd.DataFrame(rows)
            filename = f"comparison_{comp['dataset1']}_vs_{comp['dataset2']}.csv"
            df_metrics.to_csv(os.path.join(output_path, filename), index=False)

    # 3. Quality comparison
    if quality_comparison is not None:
        quality_df = pd.DataFrame([quality_comparison])
        quality_df.to_csv(os.path.join(output_path, "quality_distribution_comparison.csv"), index=False)

    print(f"\n  Exported comparison results to {output_path}")


def multi_mode_comparison(sample_dirs, sample_names, output_base_dir):
    """
    Compare datasets across multiple quality filtering modes.

    Performs separate analyses for:
    - All trajectories
    - PASS only
    - PASS + WARNING
    """
    modes = ['all', 'pass_only', 'pass_warning']
    mode_labels = {
        'all': 'All Trajectories',
        'pass_only': 'PASS Only',
        'pass_warning': 'PASS + WARNING'
    }

    print("\n" + "="*70)
    print("MULTI-MODE COMPARISON")
    print("="*70)
    print(f"Will compare across {len(modes)} quality filtering modes:")
    for mode in modes:
        print(f"  - {mode_labels[mode]}")

    all_mode_results = {}

    for mode in modes:
        print(f"\n{'='*70}")
        print(f"MODE: {mode_labels[mode]}")
        print(f"{'='*70}")

        # Create output directory for this mode
        output_dir = os.path.join(output_base_dir, f"mode_{mode}")
        os.makedirs(output_dir, exist_ok=True)

        # Load datasets with this quality mode
        datasets = []
        for sample_dir, sample_name in zip(sample_dirs, sample_names):
            dataset = load_sample_hierarchical(sample_dir, sample_name, quality_mode=mode)
            if dataset is not None:
                datasets.append(dataset)

        if len(datasets) < 2:
            print(f"  Warning: Insufficient datasets for mode {mode}, skipping")
            continue

        # Quality distribution comparison (only for 'all' mode)
        quality_comparison = None
        if mode == 'all':
            quality_comparison = compare_quality_distributions(datasets)
            if quality_comparison is not None and quality_comparison['significant']:
                print(f"\n  WARNING: Quality distributions differ significantly (p={quality_comparison['p_value']:.4e})")
                print("  This may affect comparisons. Consider using quality-filtered modes.")

        # Multi-metric comparison
        print(f"\n  Performing multi-metric comparison...")
        comparison_results = compare_datasets_multi_metric(datasets)

        # Subsampling for D
        print(f"\n  Running subsampling analysis on D...")
        subsample_results = []
        dataset_pairs = list(combinations(range(len(datasets)), 2))
        for i, j in dataset_pairs:
            result = subsampling_analysis(
                datasets[i]['D'], datasets[j]['D'],
                datasets[i]['name'], datasets[j]['name']
            )
            if result is not None:
                subsample_results.append(result)

        # Generate plots
        print(f"\n  Generating plots...")
        if mode == 'all':
            plot_quality_distribution_comparison(datasets, output_dir)
        plot_multi_metric_comparison(datasets, output_dir)
        plot_quality_coded_scatter(datasets, output_dir)
        plot_diffusion_comparison(datasets, output_dir)
        plot_effect_size_heatmap(comparison_results, output_dir)

        # Export results
        print(f"\n  Exporting results...")
        export_comparison_results(comparison_results, datasets, quality_comparison, output_dir)

        # Store for summary
        all_mode_results[mode] = {
            'datasets': datasets,
            'comparison_results': comparison_results,
            'subsample_results': subsample_results,
            'quality_comparison': quality_comparison
        }

        print(f"\n  Results for {mode_labels[mode]} saved to {output_dir}")

    # Create cross-mode summary
    create_cross_mode_summary(all_mode_results, output_base_dir)

    return all_mode_results


def create_cross_mode_summary(all_mode_results, output_dir):
    """Create summary comparing results across different quality modes."""
    print(f"\n{'='*70}")
    print("CROSS-MODE SUMMARY")
    print(f"{'='*70}")

    summary_rows = []

    for mode, results in all_mode_results.items():
        datasets = results['datasets']
        comparisons = results['comparison_results']['pairwise_comparisons']

        for comp in comparisons:
            # Get D comparison
            d_comp = comp['metrics'].get('D', {})

            row = {
                'Mode': mode,
                'Comparison': f"{comp['dataset1']} vs {comp['dataset2']}",
                'N1': d_comp.get('n1', 0),
                'N2': d_comp.get('n2', 0),
                'Median1': d_comp.get('median1', np.nan),
                'Median2': d_comp.get('median2', np.nan),
                'Median_Difference': d_comp.get('median1', np.nan) - d_comp.get('median2', np.nan),
                'P_value': d_comp.get('mann_whitney_p', np.nan),
                'Significant': d_comp.get('significant_mw', False),
                'Cliffs_Delta': d_comp.get('cliffs_delta', np.nan),
                'Effect_Size': d_comp.get('cliffs_delta_interpretation', '')
            }
            summary_rows.append(row)

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(os.path.join(output_dir, "cross_mode_summary.csv"), index=False)

        print("\n  Summary of D comparisons across modes:")
        print(df_summary.to_string(index=False))
        print(f"\n  Cross-mode summary saved to {output_dir}/cross_mode_summary.csv")


def main():
    """Main function for enhanced comparison."""
    print("="*70)
    print("Diffusion Dataset Comparison V2")
    print("Enhanced with Quality Filtering and Multi-Metric Analysis")
    print("="*70)

    print(f"\nCurrent settings:")
    print(f"  Quality filter mode: {QUALITY_FILTER_MODE}")
    print(f"  Additional filters: R²≥{MIN_R_SQUARED}, track_length≥{MIN_TRACK_LENGTH}")

    # Get number of samples
    try:
        n_samples = int(input("\nEnter number of samples to compare: "))
        if n_samples < 2:
            print("Need at least 2 samples")
            return
    except ValueError:
        print("Invalid input")
        return

    # Get sample directories
    sample_dirs = []
    sample_names = []

    for i in range(n_samples):
        print(f"\n--- Sample {i+1} ---")
        sample_dir = input(f"Enter directory for sample {i+1}: ")

        if not os.path.isdir(sample_dir):
            print(f"Directory {sample_dir} does not exist")
            return

        sample_name = input(f"Enter name for sample {i+1} (Enter for directory name): ")
        if not sample_name:
            sample_name = os.path.basename(os.path.normpath(sample_dir))
            if not sample_name:
                sample_name = f"Sample_{i+1}"

        sample_dirs.append(sample_dir)
        sample_names.append(sample_name)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), f"comparison_v2_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Check if multi-mode comparison requested
    if QUALITY_FILTER_MODE == 'multi_mode':
        results = multi_mode_comparison(sample_dirs, sample_names, output_dir)
    else:
        # Single mode comparison
        print(f"\nLoading samples with quality mode: {QUALITY_FILTER_MODE}")
        datasets = []
        for sample_dir, sample_name in zip(sample_dirs, sample_names):
            dataset = load_sample_hierarchical(sample_dir, sample_name, quality_mode=QUALITY_FILTER_MODE)
            if dataset is not None:
                datasets.append(dataset)

        if len(datasets) < 2:
            print("Need at least 2 valid datasets")
            return

        # Quality comparison
        quality_comparison = compare_quality_distributions(datasets)
        if quality_comparison is not None:
            if quality_comparison['significant']:
                print(f"\nWARNING: Quality distributions differ significantly!")
                print(f"  Chi-square p-value: {quality_comparison['p_value']:.4e}")

        # Multi-metric comparison
        print("\nPerforming multi-metric comparison...")
        comparison_results = compare_datasets_multi_metric(datasets)

        # Generate plots
        print("\nGenerating plots...")
        plot_quality_distribution_comparison(datasets, output_dir)
        plot_multi_metric_comparison(datasets, output_dir)
        plot_quality_coded_scatter(datasets, output_dir)
        plot_diffusion_comparison(datasets, output_dir)
        plot_effect_size_heatmap(comparison_results, output_dir)

        # Export results
        print("\nExporting results...")
        export_comparison_results(comparison_results, datasets, quality_comparison, output_dir)

        # Save pickle
        output_file = os.path.join(output_dir, "comparison_results.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump({
                'datasets': datasets,
                'comparison_results': comparison_results,
                'quality_comparison': quality_comparison
            }, f)

    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
