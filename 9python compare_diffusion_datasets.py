# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 19:05:49 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_diffusion_datasets.py

This script compares diffusion characteristics between multiple datasets.
It implements robust statistical analysis including effect size calculations,
subsampled comparisons, and comprehensive visualizations.

Input:
- Multiple directories containing analyzed trajectory .pkl files
  or analysis result .pkl files from dataset_diffusion_analyzer.py

Output:
- Statistical comparison with effect sizes
- Visual comparisons between datasets
- Subsampling analysis results
- CSV exports of comparison data

Usage:
python compare_diffusion_datasets.py
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
import time
from datetime import datetime
from itertools import combinations

# Global parameters (modify these as needed)
# =====================================
# Filtering parameters
MIN_R_SQUARED = 0.8  # Minimum R² value for including trajectories
MIN_TRACK_LENGTH = 10  # Minimum track length (in frames)
MAX_DIFFUSION_COEFFICIENT = 5.0  # Maximum diffusion coefficient (μm²/s), set to None to include all

# Statistical parameters
ALPHA = 0.05  # Significance level
SUBSAMPLE_SIZE = 100  # Number of trajectories to use in each subsample
N_SUBSAMPLES = 50  # Number of subsampling iterations

# Plot parameters
FIGURE_SIZE = (12, 8)
FIGURE_DPI = 300
# =====================================

def load_analysis_results(file_path):
    """
    Load analysis results from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary containing the analysis results
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading analysis results from {file_path}: {e}")
        return None

def load_analyzed_data(file_path):
    """
    Load analyzed trajectory data from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary containing the analyzed trajectory data
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading analyzed data from {file_path}: {e}")
        return None

def filter_trajectories(analyzed_data, min_r_squared=MIN_R_SQUARED, 
                        min_track_length=MIN_TRACK_LENGTH,
                        max_diffusion=MAX_DIFFUSION_COEFFICIENT):
    """
    Filter trajectories based on quality criteria.
    
    Args:
        analyzed_data: Dictionary containing analyzed trajectory data
        min_r_squared: Minimum R² value to include
        min_track_length: Minimum track length to include
        max_diffusion: Maximum diffusion coefficient to include
        
    Returns:
        List of filtered trajectory dictionaries
    """
    filtered_data = []
    
    for traj in analyzed_data['trajectories']:
        # Check if diffusion coefficient is valid
        if np.isnan(traj['D']):
            continue
        
        # Apply filters
        if traj['r_squared'] < min_r_squared:
            continue
            
        if traj['track_length'] < min_track_length:
            continue
            
        if max_diffusion is not None and traj['D'] > max_diffusion:
            continue
        
        # Include trajectory if it passes all filters
        filtered_data.append({
            'id': traj['id'],
            'D': traj['D'],
            'D_err': traj['D_err'],
            'r_squared': traj['r_squared'],
            'radius_of_gyration': traj.get('radius_of_gyration', np.nan),
            'track_length': traj['track_length']
        })
    
    return filtered_data

def pool_dataset(file_paths, min_r_squared=MIN_R_SQUARED, 
                min_track_length=MIN_TRACK_LENGTH,
                max_diffusion=MAX_DIFFUSION_COEFFICIENT):
    """
    Pool trajectory data from multiple files in a dataset.
    
    Args:
        file_paths: List of paths to analyzed data files
        min_r_squared: Minimum R² value to include
        min_track_length: Minimum track length to include
        max_diffusion: Maximum diffusion coefficient to include
        
    Returns:
        Dictionary with pooled trajectory data
    """
    pooled_data = {
        'trajectories': [],
        'diffusion_coefficients': [],
        'radius_of_gyration': [],
        'track_lengths': [],
        'r_squared_values': [],
        'file_sources': [],
        'n_files': len(file_paths),
        'original_file_counts': {}
    }
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0].replace('analyzed_', '')
        
        # Load data
        analyzed_data = load_analyzed_data(file_path)
        
        if analyzed_data is None:
            print(f"Skipping {file_name} due to loading errors")
            continue
        
        # Filter trajectories
        filtered_trajectories = filter_trajectories(
            analyzed_data, min_r_squared, min_track_length, max_diffusion
        )
        
        # Store original and filtered counts
        pooled_data['original_file_counts'][base_name] = {
            'original': len(analyzed_data['trajectories']),
            'filtered': len(filtered_trajectories)
        }
        
        # Add to pooled data
        pooled_data['trajectories'].extend(filtered_trajectories)
        
        # Extract metrics for easy access
        for traj in filtered_trajectories:
            pooled_data['diffusion_coefficients'].append(traj['D'])
            pooled_data['radius_of_gyration'].append(traj.get('radius_of_gyration', np.nan))
            pooled_data['track_lengths'].append(traj['track_length'])
            pooled_data['r_squared_values'].append(traj['r_squared'])
            pooled_data['file_sources'].append(base_name)
    
    # Convert lists to numpy arrays for faster processing
    pooled_data['diffusion_coefficients'] = np.array(pooled_data['diffusion_coefficients'])
    pooled_data['radius_of_gyration'] = np.array(pooled_data['radius_of_gyration'])
    pooled_data['track_lengths'] = np.array(pooled_data['track_lengths'])
    pooled_data['r_squared_values'] = np.array(pooled_data['r_squared_values'])
    
    return pooled_data

def load_dataset(dataset_dir, dataset_name):
    """
    Load dataset either from analysis results or from raw files.
    
    Args:
        dataset_dir: Directory containing the dataset
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with dataset information
    """
    # First, check if there's an analysis results file
    analysis_files = glob.glob(os.path.join(dataset_dir, "*_analysis_results.pkl"))
    
    if analysis_files:
        # Load from analysis results
        print(f"Loading analysis results for dataset '{dataset_name}'...")
        analysis_results = load_analysis_results(analysis_files[0])
        
        if analysis_results is not None:
            return {
                'name': dataset_name,
                'analysis_results': analysis_results,
                'diffusion_coefficients': analysis_results['pooled_data']['diffusion_coefficients'],
                'radius_of_gyration': analysis_results['pooled_data']['radius_of_gyration'],
                'track_lengths': analysis_results['pooled_data']['track_lengths'],
                'r_squared_values': analysis_results['pooled_data']['r_squared_values']
            }
    
    # If no analysis results or failed to load, try raw files
    print(f"No analysis results found. Loading raw files for dataset '{dataset_name}'...")
    data_files = glob.glob(os.path.join(dataset_dir, "analyzed_*.pkl"))
    
    if not data_files:
        print(f"No data files found in {dataset_dir}")
        return None
    
    # Pool data from files
    pooled_data = pool_dataset(data_files)
    
    if not pooled_data['trajectories']:
        print(f"No valid trajectories found in dataset '{dataset_name}'")
        return None
    
    return {
        'name': dataset_name,
        'pooled_data': pooled_data,
        'diffusion_coefficients': pooled_data['diffusion_coefficients'],
        'radius_of_gyration': pooled_data['radius_of_gyration'],
        'track_lengths': pooled_data['track_lengths'],
        'r_squared_values': pooled_data['r_squared_values']
    }

def calculate_effect_size(group1, group2):
    """
    Calculate effect size (Cliff's delta) between two groups.
    
    Args:
        group1, group2: Arrays of values to compare
        
    Returns:
        Cliff's delta effect size
    """
    # If either group is empty, return NaN
    if len(group1) == 0 or len(group2) == 0:
        return np.nan
    
    # Calculate dominance matrix
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
    """
    Interpret Cliff's delta effect size.
    
    Args:
        delta: Cliff's delta value
        
    Returns:
        String interpretation of effect size
    """
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
    """
    Calculate Cohen's d effect size.
    
    Args:
        group1, group2: Arrays of values to compare
        
    Returns:
        Cohen's d effect size
    """
    # If either group is empty, return NaN
    if len(group1) == 0 or len(group2) == 0:
        return np.nan
    
    # Calculate means
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    
    # Calculate pooled standard deviation
    n1 = len(group1)
    n2 = len(group2)
    s1 = np.std(group1, ddof=1)
    s2 = np.std(group2, ddof=1)
    
    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    return d

def interpret_cohens_d(d):
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
        
    Returns:
        String interpretation of effect size
    """
    abs_d = abs(d)
    
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"

def compare_datasets(datasets):
    """
    Perform statistical comparison between datasets.
    
    Args:
        datasets: List of dictionaries with dataset information
        
    Returns:
        Dictionary with comparison results
    """
    comparison_results = {
        'pairwise_comparisons': [],
        'subsampling_results': []
    }
    
    # Get all pairwise combinations of datasets
    dataset_pairs = list(combinations(range(len(datasets)), 2))
    
    # Compare each pair
    for i, j in dataset_pairs:
        dataset1 = datasets[i]
        dataset2 = datasets[j]
        
        # Get diffusion coefficients
        d_values1 = dataset1['diffusion_coefficients']
        d_values2 = dataset2['diffusion_coefficients']
        
        # Run statistical tests
        try:
            # Mann-Whitney U test (non-parametric)
            u_stat, p_mw = stats.mannwhitneyu(d_values1, d_values2, alternative='two-sided')
            
            # Kolmogorov-Smirnov test
            ks_stat, p_ks = stats.ks_2samp(d_values1, d_values2)
            
            # Calculate effect sizes
            cliffs_delta = calculate_effect_size(d_values1, d_values2)
            cohens_d = calculate_cohens_d(d_values1, d_values2)
            
            # Store comparison results
            comparison = {
                'dataset1': dataset1['name'],
                'dataset2': dataset2['name'],
                'n1': len(d_values1),
                'n2': len(d_values2),
                'mean1': np.mean(d_values1),
                'mean2': np.mean(d_values2),
                'median1': np.median(d_values1),
                'median2': np.median(d_values2),
                'std1': np.std(d_values1),
                'std2': np.std(d_values2),
                'mann_whitney_u': u_stat,
                'mann_whitney_p': p_mw,
                'ks_statistic': ks_stat,
                'ks_p': p_ks,
                'cliffs_delta': cliffs_delta,
                'cliffs_delta_interpretation': interpret_cliffs_delta(cliffs_delta),
                'cohens_d': cohens_d,
                'cohens_d_interpretation': interpret_cohens_d(cohens_d),
                'significant_mw': p_mw < ALPHA,
                'significant_ks': p_ks < ALPHA
            }
            
            comparison_results['pairwise_comparisons'].append(comparison)
            
            # Perform subsampling analysis
            if len(d_values1) >= SUBSAMPLE_SIZE and len(d_values2) >= SUBSAMPLE_SIZE:
                subsample_results = subsampling_analysis(d_values1, d_values2, dataset1['name'], dataset2['name'])
                comparison_results['subsampling_results'].append(subsample_results)
        
        except Exception as e:
            print(f"Error comparing {dataset1['name']} and {dataset2['name']}: {e}")
    
    return comparison_results

def subsampling_analysis(data1, data2, name1, name2, subsample_size=SUBSAMPLE_SIZE, n_subsamples=N_SUBSAMPLES):
    """
    Perform subsampling analysis to assess robustness of differences.
    
    Args:
        data1, data2: Arrays of diffusion coefficients
        name1, name2: Names of the datasets
        subsample_size: Number of trajectories to use in each subsample
        n_subsamples: Number of subsampling iterations
        
    Returns:
        Dictionary with subsampling results
    """
    results = {
        'dataset1': name1,
        'dataset2': name2,
        'p_values': [],
        'effect_sizes': [],
        'means1': [],
        'means2': [],
        'medians1': [],
        'medians2': []
    }
    
    # Run multiple iterations of subsampling
    for _ in range(n_subsamples):
        # Create subsamples
        indices1 = np.random.choice(len(data1), subsample_size, replace=False)
        indices2 = np.random.choice(len(data2), subsample_size, replace=False)
        
        subsample1 = data1[indices1]
        subsample2 = data2[indices2]
        
        # Calculate statistics
        means = (np.mean(subsample1), np.mean(subsample2))
        medians = (np.median(subsample1), np.median(subsample2))
        
        # Run statistical test
        _, p_value = stats.mannwhitneyu(subsample1, subsample2, alternative='two-sided')
        
        # Calculate effect size
        effect_size = calculate_effect_size(subsample1, subsample2)
        
        # Store results
        results['p_values'].append(p_value)
        results['effect_sizes'].append(effect_size)
        results['means1'].append(means[0])
        results['means2'].append(means[1])
        results['medians1'].append(medians[0])
        results['medians2'].append(medians[1])
    
    # Calculate summary statistics
    results['mean_p_value'] = np.mean(results['p_values'])
    results['std_p_value'] = np.std(results['p_values'])
    results['mean_effect_size'] = np.mean(results['effect_sizes'])
    results['std_effect_size'] = np.std(results['effect_sizes'])
    results['prop_significant'] = np.mean(np.array(results['p_values']) < ALPHA)
    
    return results

def plot_distribution_comparison(datasets, output_path):
    """
    Create plots comparing diffusion coefficient distributions.
    
    Args:
        datasets: List of dictionaries with dataset information
        output_path: Directory to save the plots
    """
    # Set up plot style
    sns.set(style="whitegrid")
    
    # Create histogram comparison
    plt.figure(figsize=FIGURE_SIZE)
    
    # Use different colors for each dataset
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    
    for i, dataset in enumerate(datasets):
        # Plot KDE for each dataset
        sns.kdeplot(dataset['diffusion_coefficients'], 
                   label=f"{dataset['name']} (n={len(dataset['diffusion_coefficients'])})",
                   color=colors[i], alpha=0.7)
        
        # Add vertical lines for means
        plt.axvline(np.mean(dataset['diffusion_coefficients']), 
                   color=colors[i], linestyle='-', alpha=0.5)
        
        # Add vertical lines for medians
        plt.axvline(np.median(dataset['diffusion_coefficients']), 
                   color=colors[i], linestyle='--', alpha=0.5)
    
    plt.xlabel('Diffusion coefficient (μm²/s)')
    plt.ylabel('Density')
    plt.title('Comparison of diffusion coefficient distributions')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, "distribution_comparison.png"), dpi=FIGURE_DPI)
    plt.close()
    
    # Create CDF comparison
    plt.figure(figsize=FIGURE_SIZE)
    
    for i, dataset in enumerate(datasets):
        # Sort data for CDF
        data_sorted = np.sort(dataset['diffusion_coefficients'])
        # Calculate CDF
        cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        
        # Plot CDF
        plt.plot(data_sorted, cdf, label=f"{dataset['name']} (n={len(dataset['diffusion_coefficients'])})",
                color=colors[i], alpha=0.7)
    
    plt.xlabel('Diffusion coefficient (μm²/s)')
    plt.ylabel('Cumulative probability')
    plt.title('Cumulative distribution comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, "cdf_comparison.png"), dpi=FIGURE_DPI)
    plt.close()
    
    # Create box plot comparison
    plt.figure(figsize=FIGURE_SIZE)
    
    # Prepare data for box plot
    data_to_plot = [dataset['diffusion_coefficients'] for dataset in datasets]
    labels = [dataset['name'] for dataset in datasets]
    
    # Create box plot
    plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Customize box colors
    for i, box in enumerate(plt.gca().artists):
        box.set_facecolor(colors[i])
        box.set_alpha(0.7)
    
    plt.ylabel('Diffusion coefficient (μm²/s)')
    plt.title('Box plot comparison of diffusion coefficients')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, "boxplot_comparison.png"), dpi=FIGURE_DPI)
    plt.close()
    
    # Create violin plot comparison
    plt.figure(figsize=FIGURE_SIZE)
    
    # Create dataframe for seaborn
    all_values = []
    all_datasets = []
    
    for dataset in datasets:
        all_values.extend(dataset['diffusion_coefficients'])
        all_datasets.extend([dataset['name']] * len(dataset['diffusion_coefficients']))
    
    df = pd.DataFrame({'Dataset': all_datasets, 'Diffusion': all_values})
    
    # Create violin plot
    sns.violinplot(x='Dataset', y='Diffusion', data=df, palette=colors, alpha=0.7, inner='quartile')
    plt.ylabel('Diffusion coefficient (μm²/s)')
    plt.title('Violin plot comparison of diffusion coefficients')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, "violinplot_comparison.png"), dpi=FIGURE_DPI)
    plt.close()

def plot_statistical_comparison(comparison_results, output_path):
    """
    Create plots visualizing statistical comparisons.
    
    Args:
        comparison_results: Dictionary with comparison results
        output_path: Directory to save the plots
    """
    # Check if we have comparison results
    if not comparison_results['pairwise_comparisons']:
        print("No pairwise comparisons to plot")
        return
    
    # Set up plot style
    sns.set(style="whitegrid")
    
    # Create effect size bar plot
    plt.figure(figsize=FIGURE_SIZE)
    
    # Prepare data
    comparisons = comparison_results['pairwise_comparisons']
    pair_labels = [f"{comp['dataset1']} vs {comp['dataset2']}" for comp in comparisons]
    cliffs_delta = [comp['cliffs_delta'] for comp in comparisons]
    is_significant = [comp['significant_mw'] for comp in comparisons]
    effect_interp = [comp['cliffs_delta_interpretation'] for comp in comparisons]
    
    # Define colors based on significance
    bar_colors = ['green' if sig else 'grey' for sig in is_significant]
    
    # Create bar plot
    bars = plt.bar(pair_labels, cliffs_delta, color=bar_colors, alpha=0.7)
    
    # Add effect size labels
    for i, bar in enumerate(bars):
        if is_significant[i]:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    effect_interp[i], ha='center', va='bottom', fontsize=9)
    
    plt.axhline(0, color='black', linestyle='-', alpha=0.5)
    plt.ylabel("Cliff's delta effect size")
    plt.title("Effect sizes of differences between datasets")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, "effect_size_comparison.png"), dpi=FIGURE_DPI)
    plt.close()
    
    # Create p-value and significance plot
    plt.figure(figsize=FIGURE_SIZE)
    
    # Prepare data
    p_values_mw = [-np.log10(comp['mann_whitney_p']) for comp in comparisons]
    p_values_ks = [-np.log10(comp['ks_p']) for comp in comparisons]
    
    # Create bar plot
    x = np.arange(len(pair_labels))
    width = 0.35
    
    plt.bar(x - width/2, p_values_mw, width, label='Mann-Whitney U', color='blue', alpha=0.7)
    plt.bar(x + width/2, p_values_ks, width, label='Kolmogorov-Smirnov', color='orange', alpha=0.7)
    
    # Add significance threshold line
    plt.axhline(-np.log10(ALPHA), color='red', linestyle='--', 
               label=f'Significance threshold (α={ALPHA})')
    
    plt.xlabel('Dataset comparison')
    plt.ylabel('-log10(p-value)')
    plt.title('Statistical significance of differences')
    plt.xticks(x, pair_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, "pvalue_comparison.png"), dpi=FIGURE_DPI)
    plt.close()
    
    # Create subsampling analysis plot if available
    if comparison_results['subsampling_results']:
        plt.figure(figsize=FIGURE_SIZE)
        
        # Prepare data
        subsample_results = comparison_results['subsampling_results']
        pair_labels = [f"{res['dataset1']} vs {res['dataset2']}" for res in subsample_results]
        prop_significant = [res['prop_significant'] for res in subsample_results]
        mean_effect = [res['mean_effect_size'] for res in subsample_results]
        
        # Create bar plot for proportion of significant subsamples
        plt.bar(pair_labels, prop_significant, color='purple', alpha=0.7)
        
        plt.axhline(0.05, color='red', linestyle='--', label='5% (expected by chance)')
        plt.axhline(0.95, color='green', linestyle='--', label='95% (strong evidence)')
        
        plt.ylabel('Proportion of significant subsamples')
        plt.title(f'Subsampling analysis ({N_SUBSAMPLES} iterations, n={SUBSAMPLE_SIZE} per subsample)')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_path, "subsampling_analysis.png"), dpi=FIGURE_DPI)
        plt.close()
        
        # Create scatterplot of p-values vs effect sizes in subsamples
        plt.figure(figsize=FIGURE_SIZE)
        
        # Use different colors for each comparison
        colors = plt.cm.tab10(np.linspace(0, 1, len(subsample_results)))
        
        for i, res in enumerate(subsample_results):
            plt.scatter(-np.log10(res['p_values']), res['effect_sizes'], 
                       label=f"{res['dataset1']} vs {res['dataset2']}",
                       color=colors[i], alpha=0.5)
            
            # Add center point (mean)
            plt.scatter(-np.log10(res['mean_p_value']), res['mean_effect_size'], 
                       color=colors[i], s=100, edgecolor='black', zorder=10)
        
        plt.axvline(-np.log10(ALPHA), color='red', linestyle='--', 
                   label=f'Significance threshold (α={ALPHA})')
        
        plt.xlabel('-log10(p-value)')
        plt.ylabel("Cliff's delta effect size")
        plt.title('P-values vs effect sizes in subsamples')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_path, "subsample_p_vs_effect.png"), dpi=FIGURE_DPI)
        plt.close()

def export_comparison_results(comparison_results, datasets, output_path):
    """
    Export comparison results to CSV files.
    
    Args:
        comparison_results: Dictionary with comparison results
        datasets: List of dictionaries with dataset information
        output_path: Directory to save the CSV files
    """
    # Export dataset summary
    dataset_summary = []
    
    for dataset in datasets:
        d_values = dataset['diffusion_coefficients']
        
        summary = {
            'Dataset': dataset['name'],
            'Number of trajectories': len(d_values),
            'Mean diffusion coefficient': np.mean(d_values),
            'Median diffusion coefficient': np.median(d_values),
            'Standard deviation': np.std(d_values),
            'Standard error': stats.sem(d_values),
            'Min diffusion coefficient': np.min(d_values),
            'Max diffusion coefficient': np.max(d_values),
            'Coefficient of variation': np.std(d_values) / np.mean(d_values) if np.mean(d_values) != 0 else np.nan
        }
        
        dataset_summary.append(summary)
    
    # Create DataFrame
    df = pd.DataFrame(dataset_summary)
    df.to_csv(os.path.join(output_path, "dataset_summary.csv"), index=False)
    
    # Export pairwise comparisons
    if comparison_results['pairwise_comparisons']:
        df = pd.DataFrame(comparison_results['pairwise_comparisons'])
        df.to_csv(os.path.join(output_path, "pairwise_comparisons.csv"), index=False)
    
    # Export subsampling analysis results
    if comparison_results['subsampling_results']:
        subsample_summary = []
        
        for res in comparison_results['subsampling_results']:
            summary = {
                'Dataset1': res['dataset1'],
                'Dataset2': res['dataset2'],
                'Mean p-value': res['mean_p_value'],
                'Std p-value': res['std_p_value'],
                'Mean effect size': res['mean_effect_size'],
                'Std effect size': res['std_effect_size'],
                'Proportion significant': res['prop_significant']
            }
            
            subsample_summary.append(summary)
        
        df = pd.DataFrame(subsample_summary)
        df.to_csv(os.path.join(output_path, "subsampling_summary.csv"), index=False)
        
        # Export detailed subsampling results
        for i, res in enumerate(comparison_results['subsampling_results']):
            detailed = pd.DataFrame({
                'p_value': res['p_values'],
                'effect_size': res['effect_sizes'],
                'mean1': res['means1'],
                'mean2': res['means2'],
                'median1': res['medians1'],
                'median2': res['medians2']
            })
            
            filename = f"subsampling_detail_{res['dataset1']}_vs_{res['dataset2']}.csv"
            detailed.to_csv(os.path.join(output_path, filename), index=False)

def main():
    """Main function to compare diffusion datasets."""
    print("Diffusion Dataset Comparison")
    print("==========================")
    
    # Get number of datasets to compare
    try:
        n_datasets = int(input("Enter the number of datasets to compare: "))
        if n_datasets < 2:
            print("Need at least 2 datasets for comparison")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        return
    
    # Get dataset directories
    dataset_dirs = []
    dataset_names = []
    
    for i in range(n_datasets):
        dataset_dir = input(f"Enter analyzed directory for dataset {i+1}: ")
        
        if not os.path.isdir(dataset_dir):
            print(f"Directory {dataset_dir} does not exist")
            return
        
        # Get dataset name (default to directory name)
        dataset_name = input(f"Enter name for dataset {i+1} (press Enter to use directory name): ")
        
        if not dataset_name:
            dataset_name = os.path.basename(os.path.normpath(dataset_dir))
            if not dataset_name:
                dataset_name = f"Dataset_{i+1}"
        
        dataset_dirs.append(dataset_dir)
        dataset_names.append(dataset_name)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), f"diffusion_comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = []
    
    for i, (dir_path, name) in enumerate(zip(dataset_dirs, dataset_names)):
        print(f"Loading dataset {i+1}/{n_datasets}: {name}")
        
        dataset = load_dataset(dir_path, name)
        
        if dataset is None:
            print(f"Failed to load dataset {name}")
            continue
        
        datasets.append(dataset)
        print(f"  Loaded {len(dataset['diffusion_coefficients'])} trajectories")
    
    if len(datasets) < 2:
        print("Need at least 2 valid datasets for comparison")
        return
    
    # Compare datasets
    print("\nComparing datasets...")
    comparison_results = compare_datasets(datasets)
    
    # Generate plots
    print("Generating comparison plots...")
    plot_distribution_comparison(datasets, output_dir)
    plot_statistical_comparison(comparison_results, output_dir)
    
    # Export results
    print("Exporting comparison results...")
    export_comparison_results(comparison_results, datasets, output_dir)
    
    # Save comparison results
    output_file = os.path.join(output_dir, "comparison_results.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump({
            'datasets': datasets,
            'comparison_results': comparison_results
        }, f)
    
    print(f"Comparison results saved to {output_file}")
    print(f"All outputs saved to {output_dir}")
    
    # Print summary of results
    print("\nComparison Summary:")
    for comp in comparison_results['pairwise_comparisons']:
        sig_text = "Significant" if comp['significant_mw'] else "Not significant"
        effect_text = comp['cliffs_delta_interpretation']
        
        print(f"{comp['dataset1']} vs {comp['dataset2']}:")
        print(f"  Difference in medians: {comp['median1'] - comp['median2']:.4f} μm²/s")
        print(f"  Statistical significance: {sig_text} (p = {comp['mann_whitney_p']:.4e})")
        print(f"  Effect size: {effect_text} (Cliff's delta = {comp['cliffs_delta']:.4f})")
        
        if comp['significant_mw'] and abs(comp['cliffs_delta']) < 0.33:
            print("  Note: Statistically significant but small effect size")
        elif not comp['significant_mw'] and abs(comp['cliffs_delta']) > 0.33:
            print("  Note: Not statistically significant but moderate to large effect size")
        
        print()

if __name__ == "__main__":
    main()