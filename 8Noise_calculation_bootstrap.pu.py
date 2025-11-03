# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 15:56:18 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diffusion_bootstrap.py

This script performs bootstrap resampling on diffusion coefficient data to
estimate inherent error and generate confidence intervals from a single dataset.

Input:
- Analyzed trajectory data (.pkl files) from diffusion_analyzer.py

Output:
- Confidence intervals for diffusion coefficients
- Bootstrap distribution plots
- Error estimation metrics


Understanding the Results
After running the bootstrap analysis, the script generates several outputs in a timestamped directory (e.g., bootstrap_results_20250306_123456/):
1. CSV Files

*_bootstrap_summary.csv: Key statistics including confidence intervals and error percentages
*_bootstrap_distributions.csv: Raw data for all bootstrap resamples
*_analyzed_trajectories.csv: The filtered trajectories used in the analysis

2. Diagnostic Plots
Bootstrap Distributions Plot:

Shows histograms of bootstrapped means and medians
Vertical lines show the original value and confidence intervals
The width of these distributions directly indicates the inherent error/uncertainty
CV (coefficient of variation) quantifies the relative variability
Error margin percentage tells you how much uncertainty to expect

Diffusion Diagnostic Plot:

Top panel: Histogram of all diffusion coefficients with error bars from bootstrap
Shaded regions show confidence intervals for mean and median
Bottom panel: Scatter plot showing relationship between fit quality (R²) and diffusion values

Bootstrap Diagnostics Plot:

Shows relationships between various bootstrap statistics
Helps assess normality and other properties of your error distribution

3. Console Output
The script also outputs key statistics to the console:

Original mean and median diffusion coefficients
Confidence intervals for both
Error margins as percentages
Number of trajectories before and after filtering

The most important number to look at is the error percentage (e.g., ±5.2%). 
This tells you the inherent uncertainty in your diffusion coefficient measurements. 
If you see differences between experimental conditions that are smaller than this percentage,
 they might be due to random sampling rather than true biological or treatment effects.


Usage:
python diffusion_bootstrap.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob
import pickle
import time
from datetime import datetime
from pathlib import Path

# Global parameters that can be modified
# =====================================
# Number of bootstrap resamples to perform
N_BOOTSTRAP = 100
# Confidence interval level (0.95 = 95% CI)
CONFIDENCE_LEVEL = 0.95
# Minimum R² value for including trajectories in bootstrapping
MIN_R_SQUARED = 0.8
# Maximum diffusion coefficient to include (µm²/s)
# Set to None to include all values
MAX_DIFFUSION_COEFFICIENT = None
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
                        max_diffusion=MAX_DIFFUSION_COEFFICIENT):
    """
    Filter trajectories based on quality criteria.
    
    Args:
        analyzed_data: Dictionary containing analyzed trajectory data
        min_r_squared: Minimum R² value to include
        max_diffusion: Maximum diffusion coefficient to include
        
    Returns:
        Dictionary with filtered trajectory data
    """
    filtered_data = []
    
    for traj in analyzed_data['trajectories']:
        # Check if diffusion coefficient is valid
        if np.isnan(traj['D']):
            continue
        
        # Apply filters
        if traj['r_squared'] < min_r_squared:
            continue
            
        if max_diffusion is not None and traj['D'] > max_diffusion:
            continue
        
        # Include trajectory if it passes all filters
        filtered_data.append({
            'id': traj['id'],
            'D': traj['D'],
            'D_err': traj['D_err'],
            'r_squared': traj['r_squared'],
            'track_length': traj['track_length']
        })
    
    return filtered_data

def perform_bootstrap(trajectories, n_bootstrap=N_BOOTSTRAP, 
                     confidence_level=CONFIDENCE_LEVEL):
    """
    Perform bootstrap resampling to estimate error in diffusion coefficients.
    
    Args:
        trajectories: List of dictionaries with trajectory data
        n_bootstrap: Number of bootstrap samples to generate
        confidence_level: Confidence interval level (0.0-1.0)
        
    Returns:
        Dictionary with bootstrap results
    """
    if not trajectories:
        print("No valid trajectories for bootstrapping")
        return None
    
    # Extract diffusion coefficients
    d_values = np.array([traj['D'] for traj in trajectories])
    
    # Original statistics
    original_mean = np.mean(d_values)
    original_median = np.median(d_values)
    original_std = np.std(d_values)
    original_sem = stats.sem(d_values)
    
    # Prepare arrays for bootstrap results
    bootstrap_means = np.zeros(n_bootstrap)
    bootstrap_medians = np.zeros(n_bootstrap)
    bootstrap_stds = np.zeros(n_bootstrap)
    
    # Perform bootstrap resampling
    n_trajectories = len(trajectories)
    
    print(f"Performing {n_bootstrap} bootstrap resamples on {n_trajectories} trajectories...")
    start_time = time.time()
    
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_trajectories, size=n_trajectories, replace=True)
        bootstrap_sample = d_values[indices]
        
        # Calculate statistics for this sample
        bootstrap_means[i] = np.mean(bootstrap_sample)
        bootstrap_medians[i] = np.median(bootstrap_sample)
        bootstrap_stds[i] = np.std(bootstrap_sample)
        
        # Display progress
        if (i + 1) % (n_bootstrap // 10) == 0:
            progress = (i + 1) / n_bootstrap * 100
            elapsed = time.time() - start_time
            remaining = elapsed / (i + 1) * (n_bootstrap - i - 1)
            print(f"  {progress:.1f}% complete - {elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining")
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    mean_ci = np.percentile(bootstrap_means, [lower_percentile, upper_percentile])
    median_ci = np.percentile(bootstrap_medians, [lower_percentile, upper_percentile])
    std_ci = np.percentile(bootstrap_stds, [lower_percentile, upper_percentile])
    
    # Calculate coefficient of variation for bootstrap distribution
    cv_mean = np.std(bootstrap_means) / np.mean(bootstrap_means)
    cv_median = np.std(bootstrap_medians) / np.mean(bootstrap_medians)
    
    # Calculate error margins as percentages of original values
    mean_error_pct = (mean_ci[1] - mean_ci[0]) / (2 * original_mean) * 100
    median_error_pct = (median_ci[1] - median_ci[0]) / (2 * original_median) * 100
    
    # Return results
    results = {
        'original_mean': original_mean,
        'original_median': original_median,
        'original_std': original_std,
        'original_sem': original_sem,
        'bootstrap_means': bootstrap_means,
        'bootstrap_medians': bootstrap_medians,
        'bootstrap_stds': bootstrap_stds,
        'mean_ci': mean_ci,
        'median_ci': median_ci,
        'std_ci': std_ci,
        'cv_mean': cv_mean,
        'cv_median': cv_median,
        'mean_error_pct': mean_error_pct,
        'median_error_pct': median_error_pct,
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level,
        'n_trajectories': n_trajectories
    }
    
    return results

def create_bootstrap_plots(bootstrap_results, output_path, basename):
    """
    Create plots visualizing bootstrap results.
    
    Args:
        bootstrap_results: Dictionary with bootstrap results
        output_path: Directory to save plots
        basename: Base filename for the plots
    """
    if bootstrap_results is None:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Set up plotting style
    sns.set(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (12, 10),
        'savefig.dpi': 300
    })
    
    # Figure 1: Bootstrap distributions with confidence intervals
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Mean distribution
    sns.histplot(bootstrap_results['bootstrap_means'], kde=True, ax=axs[0])
    axs[0].axvline(bootstrap_results['original_mean'], color='r', linestyle='-', 
                   label=f'Original mean: {bootstrap_results["original_mean"]:.4f}')
    axs[0].axvline(bootstrap_results['mean_ci'][0], color='g', linestyle='--',
                  label=f'{bootstrap_results["confidence_level"]*100:.0f}% CI: ({bootstrap_results["mean_ci"][0]:.4f}, {bootstrap_results["mean_ci"][1]:.4f})')
    axs[0].axvline(bootstrap_results['mean_ci'][1], color='g', linestyle='--')
    axs[0].set_xlabel('Mean diffusion coefficient (µm²/s)')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title(f'Bootstrap distribution of mean diffusion coefficient\n'
                    f'CV = {bootstrap_results["cv_mean"]:.4f}, '
                    f'Error margin = ±{bootstrap_results["mean_error_pct"]:.2f}%')
    axs[0].legend()
    
    # Median distribution
    sns.histplot(bootstrap_results['bootstrap_medians'], kde=True, ax=axs[1])
    axs[1].axvline(bootstrap_results['original_median'], color='r', linestyle='-',
                  label=f'Original median: {bootstrap_results["original_median"]:.4f}')
    axs[1].axvline(bootstrap_results['median_ci'][0], color='g', linestyle='--',
                  label=f'{bootstrap_results["confidence_level"]*100:.0f}% CI: ({bootstrap_results["median_ci"][0]:.4f}, {bootstrap_results["median_ci"][1]:.4f})')
    axs[1].axvline(bootstrap_results['median_ci'][1], color='g', linestyle='--')
    axs[1].set_xlabel('Median diffusion coefficient (µm²/s)')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title(f'Bootstrap distribution of median diffusion coefficient\n'
                    f'CV = {bootstrap_results["cv_median"]:.4f}, '
                    f'Error margin = ±{bootstrap_results["median_error_pct"]:.2f}%')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{basename}_bootstrap_distributions.png"))
    plt.close()
    
    # Figure 2: Diagnostic plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Scatterplot of mean vs std bootstrap samples
    axs[0, 0].scatter(bootstrap_results['bootstrap_means'], bootstrap_results['bootstrap_stds'], alpha=0.3)
    axs[0, 0].set_xlabel('Mean diffusion coefficient (µm²/s)')
    axs[0, 0].set_ylabel('Standard deviation (µm²/s)')
    axs[0, 0].set_title('Relationship between bootstrap mean and standard deviation')
    axs[0, 0].grid(True)
    
    # Plot 2: Histogram of standard deviations
    sns.histplot(bootstrap_results['bootstrap_stds'], kde=True, ax=axs[0, 1])
    axs[0, 1].axvline(bootstrap_results['original_std'], color='r', linestyle='-',
                     label=f'Original std: {bootstrap_results["original_std"]:.4f}')
    axs[0, 1].axvline(bootstrap_results['std_ci'][0], color='g', linestyle='--',
                     label=f'{bootstrap_results["confidence_level"]*100:.0f}% CI')
    axs[0, 1].axvline(bootstrap_results['std_ci'][1], color='g', linestyle='--')
    axs[0, 1].set_xlabel('Standard deviation (µm²/s)')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].set_title('Bootstrap distribution of standard deviation')
    axs[0, 1].legend()
    
    # Plot 3: QQ plot for bootstrap means (check normality)
    stats.probplot(bootstrap_results['bootstrap_means'], plot=axs[1, 0])
    axs[1, 0].set_title('Normal Q-Q Plot of Bootstrap Means')
    
    # Plot 4: Scatterplot of mean vs median bootstrap samples
    axs[1, 1].scatter(bootstrap_results['bootstrap_means'], bootstrap_results['bootstrap_medians'], alpha=0.3)
    axs[1, 1].set_xlabel('Mean diffusion coefficient (µm²/s)')
    axs[1, 1].set_ylabel('Median diffusion coefficient (µm²/s)')
    axs[1, 1].set_title('Relationship between bootstrap mean and median')
    # Add identity line
    min_val = min(min(bootstrap_results['bootstrap_means']), min(bootstrap_results['bootstrap_medians']))
    max_val = max(max(bootstrap_results['bootstrap_means']), max(bootstrap_results['bootstrap_medians']))
    axs[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{basename}_bootstrap_diagnostics.png"))
    plt.close()

def create_diagnostic_plot(trajectories, bootstrap_results, output_path, basename):
    """
    Create a diagnostic plot showing the original data distribution and bootstrap errors.
    
    Args:
        trajectories: List of dictionaries with trajectory data
        bootstrap_results: Dictionary with bootstrap results
        output_path: Directory to save the plot
        basename: Base filename for the plot
    """
    if bootstrap_results is None:
        return
    
    # Extract diffusion coefficients
    d_values = np.array([traj['D'] for traj in trajectories])
    
    plt.figure(figsize=(14, 10))
    
    # Main plot: Histogram of diffusion coefficients with bootstrap error bars
    plt.subplot(2, 1, 1)
    
    # Histogram
    counts, bins, _ = plt.hist(d_values, bins=30, alpha=0.7, color='skyblue')
    
    # Add vertical lines for mean and median with error bars
    plt.axvline(bootstrap_results['original_mean'], color='red', linestyle='-', 
               label=f'Mean: {bootstrap_results["original_mean"]:.4f} ± {(bootstrap_results["mean_ci"][1] - bootstrap_results["mean_ci"][0])/2:.4f} µm²/s')
    
    plt.axvline(bootstrap_results['original_median'], color='green', linestyle='-', 
               label=f'Median: {bootstrap_results["original_median"]:.4f} ± {(bootstrap_results["median_ci"][1] - bootstrap_results["median_ci"][0])/2:.4f} µm²/s')
    
    # Add shaded regions for confidence intervals
    plt.axvspan(bootstrap_results['mean_ci'][0], bootstrap_results['mean_ci'][1], 
               alpha=0.2, color='red', label=f'{bootstrap_results["confidence_level"]*100:.0f}% CI for mean')
    
    plt.axvspan(bootstrap_results['median_ci'][0], bootstrap_results['median_ci'][1], 
               alpha=0.2, color='green', label=f'{bootstrap_results["confidence_level"]*100:.0f}% CI for median')
    
    plt.xlabel('Diffusion coefficient (µm²/s)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of diffusion coefficients with bootstrap error estimation\n'
             f'n = {bootstrap_results["n_trajectories"]} trajectories, {bootstrap_results["n_bootstrap"]} bootstrap resamples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot: R² vs D scatter plot
    plt.subplot(2, 1, 2)
    r_squared = np.array([traj['r_squared'] for traj in trajectories])
    track_length = np.array([traj['track_length'] for traj in trajectories])
    
    # Create scatter plot where point size reflects track length
    scatter = plt.scatter(r_squared, d_values, 
                         s=np.sqrt(track_length)*3, # Scale point size by sqrt(track_length)
                         alpha=0.5, c=track_length, cmap='viridis')
    
    plt.colorbar(scatter, label='Track length (frames)')
    
    plt.xlabel('R² of MSD fit')
    plt.ylabel('Diffusion coefficient (µm²/s)')
    plt.title('Relationship between fit quality and diffusion coefficient')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{basename}_diffusion_diagnostic.png"))
    plt.close()

def export_bootstrap_results(bootstrap_results, trajectories, output_path, basename):
    """
    Export bootstrap results to CSV.
    
    Args:
        bootstrap_results: Dictionary with bootstrap results
        trajectories: List of dictionaries with trajectory data
        output_path: Directory to save the CSV file
        basename: Base filename for the CSV file
    """
    if bootstrap_results is None:
        return
    
    # Create data for summary CSV
    summary_data = {
        'Metric': [
            'Number of trajectories',
            'Number of bootstrap resamples',
            'Confidence level',
            'Original mean D',
            'Mean D - lower CI',
            'Mean D - upper CI',
            'Mean D - absolute error',
            'Mean D - error percentage',
            'Original median D',
            'Median D - lower CI',
            'Median D - upper CI',
            'Median D - absolute error',
            'Median D - error percentage',
            'Original standard deviation',
            'Standard deviation - lower CI',
            'Standard deviation - upper CI',
            'Coefficient of variation (mean)',
            'Coefficient of variation (median)'
        ],
        'Value': [
            bootstrap_results['n_trajectories'],
            bootstrap_results['n_bootstrap'],
            bootstrap_results['confidence_level'],
            bootstrap_results['original_mean'],
            bootstrap_results['mean_ci'][0],
            bootstrap_results['mean_ci'][1],
            (bootstrap_results['mean_ci'][1] - bootstrap_results['mean_ci'][0]) / 2,
            bootstrap_results['mean_error_pct'],
            bootstrap_results['original_median'],
            bootstrap_results['median_ci'][0],
            bootstrap_results['median_ci'][1],
            (bootstrap_results['median_ci'][1] - bootstrap_results['median_ci'][0]) / 2,
            bootstrap_results['median_error_pct'],
            bootstrap_results['original_std'],
            bootstrap_results['std_ci'][0],
            bootstrap_results['std_ci'][1],
            bootstrap_results['cv_mean'],
            bootstrap_results['cv_median']
        ]
    }
    
    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_path, f"{basename}_bootstrap_summary.csv"), index=False)
    
    # Create DataFrame for bootstrap distributions
    distributions_df = pd.DataFrame({
        'Bootstrap_Mean': bootstrap_results['bootstrap_means'],
        'Bootstrap_Median': bootstrap_results['bootstrap_medians'],
        'Bootstrap_StdDev': bootstrap_results['bootstrap_stds']
    })
    distributions_df.to_csv(os.path.join(output_path, f"{basename}_bootstrap_distributions.csv"), index=False)
    
    # Create DataFrame for original trajectory data
    trajectories_df = pd.DataFrame(trajectories)
    trajectories_df.to_csv(os.path.join(output_path, f"{basename}_analyzed_trajectories.csv"), index=False)
    
    print(f"Bootstrap results exported to {output_path}")

def main():
    """Main function to perform bootstrap analysis on diffusion data."""
    # Ask for input file
    input_file = input("Enter the path to the analyzed trajectory file (pkl file in the analyzed_trajectory folder): ")
    
    if input_file == "":
        # List all available files for user to choose
        files = glob.glob(os.path.join(os.getcwd(), "analyzed_trajectories", "analyzed_*.pkl"))
        if not files:
            print("No analyzed trajectory files found in the default location.")
            print("Please specify the full path to your file.")
            return
        
        print("\nAvailable files:")
        for i, file in enumerate(files):
            print(f"[{i+1}] {os.path.basename(file)}")
        
        try:
            selection = int(input("\nSelect file number: "))
            if selection < 1 or selection > len(files):
                print("Invalid selection")
                return
            input_file = files[selection-1]
        except ValueError:
            print("Invalid selection")
            return
    
    # Load analyzed data
    analyzed_data = load_analyzed_data(input_file)
    
    if analyzed_data is None:
        print(f"Could not load data from {input_file}")
        return
    
    # Get basename for output files
    basename = os.path.splitext(os.path.basename(input_file))[0].replace("analyzed_", "")
    
    # Filter trajectories
    filtered_trajectories = filter_trajectories(analyzed_data)
    
    print(f"\nLoaded {len(analyzed_data['trajectories'])} trajectories from {input_file}")
    print(f"After filtering, {len(filtered_trajectories)} trajectories remain for bootstrap analysis")
    
    if len(filtered_trajectories) < 10:
        print("Too few trajectories for reliable bootstrap analysis. At least 10 are recommended.")
        return
    
    # Ask for bootstrap parameters
    try:
        n_bootstrap = int(input(f"Enter number of bootstrap resamples [{N_BOOTSTRAP}]: ") or N_BOOTSTRAP)
        confidence_level = float(input(f"Enter confidence level (0-1) [{CONFIDENCE_LEVEL}]: ") or CONFIDENCE_LEVEL)
    except ValueError:
        print("Invalid input. Using default values.")
        n_bootstrap = N_BOOTSTRAP
        confidence_level = CONFIDENCE_LEVEL
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(input_file), f"bootstrap_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform bootstrap analysis
    print(f"\nPerforming bootstrap analysis with {n_bootstrap} resamples and {confidence_level*100:.0f}% confidence level...")
    bootstrap_results = perform_bootstrap(filtered_trajectories, n_bootstrap, confidence_level)
    
    if bootstrap_results is None:
        print("Bootstrap analysis failed")
        return
    
    # Print summary results
    print("\nBootstrap Analysis Results:")
    print("--------------------------")
    print(f"Original mean D: {bootstrap_results['original_mean']:.6f} µm²/s")
    print(f"{confidence_level*100:.0f}% CI for mean: ({bootstrap_results['mean_ci'][0]:.6f}, {bootstrap_results['mean_ci'][1]:.6f}) µm²/s")
    print(f"Error margin for mean: ±{bootstrap_results['mean_error_pct']:.2f}%")
    print()
    print(f"Original median D: {bootstrap_results['original_median']:.6f} µm²/s")
    print(f"{confidence_level*100:.0f}% CI for median: ({bootstrap_results['median_ci'][0]:.6f}, {bootstrap_results['median_ci'][1]:.6f}) µm²/s")
    print(f"Error margin for median: ±{bootstrap_results['median_error_pct']:.2f}%")
    
    # Create plots
    print("\nGenerating diagnostic plots...")
    create_bootstrap_plots(bootstrap_results, output_dir, basename)
    create_diagnostic_plot(filtered_trajectories, bootstrap_results, output_dir, basename)
    
    # Export results
    print("\nExporting results to CSV...")
    export_bootstrap_results(bootstrap_results, filtered_trajectories, output_dir, basename)
    
    print(f"\nBootstrap analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()