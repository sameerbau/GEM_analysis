# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 18:57:42 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_diffusion_analyzer.py

This script analyzes diffusion characteristics from a dataset of pickle files
containing analyzed trajectory data. It calculates statistical measures with
confidence intervals using bootstrapping and generates comprehensive visualizations.

Input:
- Directory containing analyzed trajectory .pkl files

Output:
- Statistical summary with confidence intervals
- Distribution plots for diffusion coefficients
- Bootstrap analysis results
- CSV exports of analyzed data

Usage:
python dataset_diffusion_analyzer.py
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

# Global parameters (modify these as needed)
# =====================================
# Bootstrap parameters
N_BOOTSTRAP = 1000  # Number of bootstrap resamples
CONFIDENCE_LEVEL = 0.95  # Confidence interval level (0.95 = 95% CI)

# Filtering parameters
MIN_R_SQUARED = 0.7  # Minimum R² value for including trajectories
MIN_TRACK_LENGTH = 10  # Minimum track length (in frames)
MAX_DIFFUSION_COEFFICIENT = 5.0  # Maximum diffusion coefficient (μm²/s), set to None to include all

# Plot parameters
FIGURE_SIZE = (10, 8)
FIGURE_DPI = 300
# =====================================

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

def perform_bootstrap(data, n_bootstrap=N_BOOTSTRAP, confidence_level=CONFIDENCE_LEVEL):
    """
    Perform bootstrap resampling to estimate statistics and confidence intervals.
    
    Args:
        data: Numpy array of values to bootstrap
        n_bootstrap: Number of bootstrap samples to generate
        confidence_level: Confidence interval level (0.0-1.0)
        
    Returns:
        Dictionary with bootstrap results
    """
    if len(data) == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'sem': np.nan,
            'mean_ci': (np.nan, np.nan),
            'median_ci': (np.nan, np.nan),
            'std_ci': (np.nan, np.nan),
            'cv': np.nan,
            'cv_ci': (np.nan, np.nan),
            'n': 0
        }
    
    # Original statistics
    original_mean = np.mean(data)
    original_median = np.median(data)
    original_std = np.std(data)
    original_sem = stats.sem(data)
    original_cv = original_std / original_mean if original_mean != 0 else np.nan
    
    # Prepare arrays for bootstrap results
    bootstrap_means = np.zeros(n_bootstrap)
    bootstrap_medians = np.zeros(n_bootstrap)
    bootstrap_stds = np.zeros(n_bootstrap)
    bootstrap_cvs = np.zeros(n_bootstrap)
    
    # Perform bootstrap resampling
    n_samples = len(data)
    
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = data[indices]
        
        # Calculate statistics for this sample
        bootstrap_means[i] = np.mean(bootstrap_sample)
        bootstrap_medians[i] = np.median(bootstrap_sample)
        bootstrap_stds[i] = np.std(bootstrap_sample)
        
        # Calculate CV (coefficient of variation)
        mean_value = bootstrap_means[i]
        bootstrap_cvs[i] = bootstrap_stds[i] / mean_value if mean_value != 0 else np.nan
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    mean_ci = np.percentile(bootstrap_means, [lower_percentile, upper_percentile])
    median_ci = np.percentile(bootstrap_medians, [lower_percentile, upper_percentile])
    std_ci = np.percentile(bootstrap_stds, [lower_percentile, upper_percentile])
    cv_ci = np.percentile(bootstrap_cvs[~np.isnan(bootstrap_cvs)], [lower_percentile, upper_percentile])
    
    return {
        'mean': original_mean,
        'median': original_median,
        'std': original_std,
        'sem': original_sem,
        'cv': original_cv,
        'mean_ci': mean_ci,
        'median_ci': median_ci,
        'std_ci': std_ci,
        'cv_ci': cv_ci,
        'n': n_samples,
        'bootstrap_means': bootstrap_means,
        'bootstrap_medians': bootstrap_medians,
        'bootstrap_stds': bootstrap_stds,
        'bootstrap_cvs': bootstrap_cvs
    }

def analyze_dataset(pooled_data, n_bootstrap=N_BOOTSTRAP, confidence_level=CONFIDENCE_LEVEL):
    """
    Analyze pooled dataset and calculate bootstrap statistics.
    
    Args:
        pooled_data: Dictionary with pooled trajectory data
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence interval level
        
    Returns:
        Dictionary with analysis results
    """
    results = {
        'pooled_data': pooled_data,
        'diffusion_bootstrap': None,
        'radius_bootstrap': None,
        'per_file_statistics': {},
        'dataset_name': '',
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Bootstrap statistics for diffusion coefficients
    print(f"Performing bootstrap on {len(pooled_data['diffusion_coefficients'])} diffusion coefficients...")
    results['diffusion_bootstrap'] = perform_bootstrap(
        pooled_data['diffusion_coefficients'], n_bootstrap, confidence_level
    )
    
    # Bootstrap statistics for radius of gyration
    valid_rg = pooled_data['radius_of_gyration'][~np.isnan(pooled_data['radius_of_gyration'])]
    if len(valid_rg) > 0:
        print(f"Performing bootstrap on {len(valid_rg)} radius of gyration values...")
        results['radius_bootstrap'] = perform_bootstrap(
            valid_rg, n_bootstrap, confidence_level
        )
    
    # Calculate per-file statistics
    unique_files = np.unique(pooled_data['file_sources'])
    
    for file_name in unique_files:
        # Get data for this file
        file_indices = np.array(pooled_data['file_sources']) == file_name
        diffusion_values = pooled_data['diffusion_coefficients'][file_indices]
        
        # Calculate statistics
        file_stats = {
            'n': len(diffusion_values),
            'mean': np.mean(diffusion_values),
            'median': np.median(diffusion_values),
            'std': np.std(diffusion_values),
            'sem': stats.sem(diffusion_values),
            'cv': np.std(diffusion_values) / np.mean(diffusion_values)
        }
        
        results['per_file_statistics'][file_name] = file_stats
    
    return results

def plot_diffusion_distribution(analysis_results, output_path):
    """
    Create plots of diffusion coefficient distribution with bootstrap confidence intervals.
    
    Args:
        analysis_results: Dictionary with analysis results
        output_path: Directory to save the plots
    """
    diffusion_data = analysis_results['pooled_data']['diffusion_coefficients']
    bootstrap_results = analysis_results['diffusion_bootstrap']
    
    if len(diffusion_data) == 0:
        print("No diffusion data to plot")
        return
    
    # Set up plot style
    sns.set(style="whitegrid")
    plt.figure(figsize=FIGURE_SIZE)
    
    # Create histogram with KDE
    sns.histplot(diffusion_data, kde=True, color='skyblue')
    
    # Add vertical lines for mean and median with CI
    plt.axvline(bootstrap_results['mean'], color='red', linestyle='-', 
               label=f'Mean: {bootstrap_results["mean"]:.4f} ({bootstrap_results["mean_ci"][0]:.4f}-{bootstrap_results["mean_ci"][1]:.4f})')
    
    plt.axvline(bootstrap_results['median'], color='green', linestyle='-', 
               label=f'Median: {bootstrap_results["median"]:.4f} ({bootstrap_results["median_ci"][0]:.4f}-{bootstrap_results["median_ci"][1]:.4f})')
    
    # Add shaded regions for confidence intervals
    plt.axvspan(bootstrap_results['mean_ci'][0], bootstrap_results['mean_ci'][1], 
               alpha=0.2, color='red')
    
    plt.axvspan(bootstrap_results['median_ci'][0], bootstrap_results['median_ci'][1], 
               alpha=0.2, color='green')
    
    # Customize plot
    plt.xlabel('Diffusion coefficient (μm²/s)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of diffusion coefficients (n={len(diffusion_data)})')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_path, "diffusion_distribution.png"), dpi=FIGURE_DPI)
    plt.close()
    
    # Create bootstrap distribution plots
    plt.figure(figsize=FIGURE_SIZE)
    
    # Plot bootstrap distributions of mean and median
    plt.subplot(2, 1, 1)
    sns.histplot(bootstrap_results['bootstrap_means'], kde=True, color='red')
    plt.axvline(bootstrap_results['mean'], color='black', linestyle='-')
    plt.axvline(bootstrap_results['mean_ci'][0], color='black', linestyle='--')
    plt.axvline(bootstrap_results['mean_ci'][1], color='black', linestyle='--')
    plt.xlabel('Mean diffusion coefficient (μm²/s)')
    plt.ylabel('Frequency')
    plt.title(f'Bootstrap distribution of mean ({N_BOOTSTRAP} resamples)')
    
    plt.subplot(2, 1, 2)
    sns.histplot(bootstrap_results['bootstrap_medians'], kde=True, color='green')
    plt.axvline(bootstrap_results['median'], color='black', linestyle='-')
    plt.axvline(bootstrap_results['median_ci'][0], color='black', linestyle='--')
    plt.axvline(bootstrap_results['median_ci'][1], color='black', linestyle='--')
    plt.xlabel('Median diffusion coefficient (μm²/s)')
    plt.ylabel('Frequency')
    plt.title(f'Bootstrap distribution of median ({N_BOOTSTRAP} resamples)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "bootstrap_distributions.png"), dpi=FIGURE_DPI)
    plt.close()
    
    # Create scatter plot of D vs R²
    plt.figure(figsize=FIGURE_SIZE)
    r_squared = analysis_results['pooled_data']['r_squared_values']
    plt.scatter(r_squared, diffusion_data, alpha=0.5)
    plt.xlabel('R² of MSD fit')
    plt.ylabel('Diffusion coefficient (μm²/s)')
    plt.title('Diffusion coefficient vs fit quality')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, "diffusion_vs_r_squared.png"), dpi=FIGURE_DPI)
    plt.close()
    
    # Create scatter plot of D vs track length
    plt.figure(figsize=FIGURE_SIZE)
    track_lengths = analysis_results['pooled_data']['track_lengths']
    plt.scatter(track_lengths, diffusion_data, alpha=0.5)
    plt.xlabel('Track length (frames)')
    plt.ylabel('Diffusion coefficient (μm²/s)')
    plt.title('Diffusion coefficient vs track length')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, "diffusion_vs_track_length.png"), dpi=FIGURE_DPI)
    plt.close()

def plot_per_file_comparison(analysis_results, output_path):
    """
    Create plots comparing diffusion coefficients across files in the dataset.
    
    Args:
        analysis_results: Dictionary with analysis results
        output_path: Directory to save the plots
    """
    # Extract per-file statistics
    per_file_stats = analysis_results['per_file_statistics']
    
    if not per_file_stats:
        print("No per-file statistics to plot")
        return
    
    # Prepare data for plotting
    file_names = list(per_file_stats.keys())
    mean_values = [stats['mean'] for stats in per_file_stats.values()]
    median_values = [stats['median'] for stats in per_file_stats.values()]
    std_values = [stats['std'] for stats in per_file_stats.values()]
    sem_values = [stats['sem'] for stats in per_file_stats.values()]
    n_values = [stats['n'] for stats in per_file_stats.values()]
    
    # Set up plot style
    sns.set(style="whitegrid")
    
    # Create bar plot of means with error bars
    plt.figure(figsize=FIGURE_SIZE)
    x = np.arange(len(file_names))
    plt.bar(x, mean_values, yerr=sem_values, capsize=5, alpha=0.7)
    plt.xticks(x, file_names, rotation=45, ha='right')
    plt.xlabel('File')
    plt.ylabel('Mean diffusion coefficient (μm²/s)')
    plt.title('Comparison of mean diffusion coefficients across files')
    
    # Add sample sizes
    for i, n in enumerate(n_values):
        plt.text(i, mean_values[i] + sem_values[i] + 0.02, f'n={n}', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "per_file_means.png"), dpi=FIGURE_DPI)
    plt.close()
    
    # Create box plot comparison
    plt.figure(figsize=FIGURE_SIZE)
    
    # Extract diffusion values for each file
    data_to_plot = []
    for file_name in file_names:
        file_indices = np.array(analysis_results['pooled_data']['file_sources']) == file_name
        diffusion_values = analysis_results['pooled_data']['diffusion_coefficients'][file_indices]
        data_to_plot.append(diffusion_values)
    
    # Create box plot
    plt.boxplot(data_to_plot, labels=file_names)
    plt.ylabel('Diffusion coefficient (μm²/s)')
    plt.title('Distribution of diffusion coefficients across files')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, "per_file_boxplot.png"), dpi=FIGURE_DPI)
    plt.close()
    
    # Create violin plot comparison
    plt.figure(figsize=FIGURE_SIZE)
    
    # Create dataframe for seaborn
    all_values = []
    all_files = []
    
    for i, file_name in enumerate(file_names):
        file_indices = np.array(analysis_results['pooled_data']['file_sources']) == file_name
        diffusion_values = analysis_results['pooled_data']['diffusion_coefficients'][file_indices]
        
        all_values.extend(diffusion_values)
        all_files.extend([file_name] * len(diffusion_values))
    
    df = pd.DataFrame({'File': all_files, 'Diffusion': all_values})
    
    # Create violin plot
    sns.violinplot(x='File', y='Diffusion', data=df, inner='quartile')
    plt.ylabel('Diffusion coefficient (μm²/s)')
    plt.title('Distribution of diffusion coefficients across files')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, "per_file_violinplot.png"), dpi=FIGURE_DPI)
    plt.close()

def export_analysis_results(analysis_results, output_path):
    """
    Export analysis results to CSV files.
    
    Args:
        analysis_results: Dictionary with analysis results
        output_path: Directory to save the CSV files
    """
    # Export overall statistics
    diffusion_bootstrap = analysis_results['diffusion_bootstrap']
    
    overall_stats = pd.DataFrame({
        'Metric': [
            'Number of trajectories',
            'Mean diffusion coefficient',
            'Mean D - lower CI',
            'Mean D - upper CI',
            'Median diffusion coefficient',
            'Median D - lower CI',
            'Median D - upper CI',
            'Standard deviation',
            'Standard error of mean',
            'Coefficient of variation'
        ],
        'Value': [
            diffusion_bootstrap['n'],
            diffusion_bootstrap['mean'],
            diffusion_bootstrap['mean_ci'][0],
            diffusion_bootstrap['mean_ci'][1],
            diffusion_bootstrap['median'],
            diffusion_bootstrap['median_ci'][0],
            diffusion_bootstrap['median_ci'][1],
            diffusion_bootstrap['std'],
            diffusion_bootstrap['sem'],
            diffusion_bootstrap['cv']
        ]
    })
    
    overall_stats.to_csv(os.path.join(output_path, "overall_statistics.csv"), index=False)
    
    # Export per-file statistics
    per_file_stats = []
    
    for file_name, stats in analysis_results['per_file_statistics'].items():
        per_file_stats.append({
            'File': file_name,
            'Trajectories': stats['n'],
            'Mean D': stats['mean'],
            'Median D': stats['median'],
            'Std Dev': stats['std'],
            'SEM': stats['sem'],
            'CV': stats['cv'],
            'Original Trajectories': analysis_results['pooled_data']['original_file_counts'][file_name]['original'],
            'Filtered Trajectories': analysis_results['pooled_data']['original_file_counts'][file_name]['filtered']
        })
    
    if per_file_stats:
        per_file_df = pd.DataFrame(per_file_stats)
        per_file_df.to_csv(os.path.join(output_path, "per_file_statistics.csv"), index=False)
    
    # Export all trajectories data
    trajectories_data = []
    
    for i, traj in enumerate(analysis_results['pooled_data']['trajectories']):
        trajectories_data.append({
            'id': traj['id'],
            'D': traj['D'],
            'D_err': traj['D_err'],
            'r_squared': traj['r_squared'],
            'radius_of_gyration': traj.get('radius_of_gyration', np.nan),
            'track_length': traj['track_length'],
            'source_file': analysis_results['pooled_data']['file_sources'][i]
        })
    
    if trajectories_data:
        traj_df = pd.DataFrame(trajectories_data)
        traj_df.to_csv(os.path.join(output_path, "all_trajectories.csv"), index=False)
    
    # Export bootstrap distributions
    bootstrap_df = pd.DataFrame({
        'Bootstrap_Mean': diffusion_bootstrap['bootstrap_means'],
        'Bootstrap_Median': diffusion_bootstrap['bootstrap_medians'],
        'Bootstrap_StdDev': diffusion_bootstrap['bootstrap_stds'],
        'Bootstrap_CV': diffusion_bootstrap['bootstrap_cvs']
    })
    
    bootstrap_df.to_csv(os.path.join(output_path, "bootstrap_distributions.csv"), index=False)

def main():
    """Main function to analyze diffusion data from a dataset."""
    print("Dataset Diffusion Analyzer")
    print("=========================")
    
    # Ask for input directory
    input_dir = input("Enter the directory containing analyzed trajectory files: ")
    
    if not os.path.isdir(input_dir):
        print(f"Directory {input_dir} does not exist")
        return
    
    # Get dataset name
    dataset_name = os.path.basename(os.path.normpath(input_dir))
    if not dataset_name:
        dataset_name = "dataset"
    
    # Get list of analyzed files
    file_paths = glob.glob(os.path.join(input_dir, "analyzed_*.pkl"))
    
    if not file_paths:
        print(f"No analyzed trajectory files found in {input_dir}")
        return
    
    print(f"Found {len(file_paths)} files to analyze in dataset '{dataset_name}'")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(input_dir, f"{dataset_name}_analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Pool data from all files
    print(f"Pooling data from {len(file_paths)} files...")
    start_time = time.time()
    pooled_data = pool_dataset(file_paths, MIN_R_SQUARED, MIN_TRACK_LENGTH, MAX_DIFFUSION_COEFFICIENT)
    
    print(f"Pooled {len(pooled_data['trajectories'])} trajectories after filtering")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    
    # Analyze dataset
    print(f"Analyzing dataset '{dataset_name}'...")
    start_time = time.time()
    analysis_results = analyze_dataset(pooled_data, N_BOOTSTRAP, CONFIDENCE_LEVEL)
    analysis_results['dataset_name'] = dataset_name
    
    print(f"Analysis complete. Time taken: {time.time() - start_time:.2f} seconds")
    
    # Generate plots
    print("Generating plots...")
    plot_diffusion_distribution(analysis_results, output_dir)
    plot_per_file_comparison(analysis_results, output_dir)
    
    # Export results
    print("Exporting results to CSV...")
    export_analysis_results(analysis_results, output_dir)
    
    # Save analysis results for further comparison
    output_file = os.path.join(output_dir, f"{dataset_name}_analysis_results.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(analysis_results, f)
    
    print(f"Analysis results saved to {output_file}")
    print(f"All outputs saved to {output_dir}")
    
    # Print summary statistics
    diff_bootstrap = analysis_results['diffusion_bootstrap']
    print("\nSummary Statistics:")
    print(f"Number of trajectories: {diff_bootstrap['n']}")
    print(f"Mean diffusion coefficient: {diff_bootstrap['mean']:.6f} μm²/s")
    print(f"95% CI for mean: ({diff_bootstrap['mean_ci'][0]:.6f}, {diff_bootstrap['mean_ci'][1]:.6f}) μm²/s")
    print(f"Median diffusion coefficient: {diff_bootstrap['median']:.6f} μm²/s")
    print(f"95% CI for median: ({diff_bootstrap['median_ci'][0]:.6f}, {diff_bootstrap['median_ci'][1]:.6f}) μm²/s")
    print(f"Coefficient of variation: {diff_bootstrap['cv']:.4f}")

if __name__ == "__main__":
    main()