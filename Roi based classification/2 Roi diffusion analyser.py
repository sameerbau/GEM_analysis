#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
roi_distribution_analyzer.py

This script adds frequency and cumulative distribution analysis to the ROI diffusion data,
working with the existing roi_loader_improved.py.

Input:
- ROI-assigned trajectory data (.pkl files) from roi_loader_improved.py

Output:
- Frequency and cumulative distribution plots of diffusion coefficients
- Statistical comparisons between ROIs

Usage:
python roi_distribution_analyzer.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import stats
from pathlib import Path
from datetime import datetime
import read_roi  # Required for ROI loading

# Global parameters that can be modified
# =====================================
# Minimum number of trajectories for valid statistics
MIN_TRAJECTORIES_FOR_STATS = 5
# Number of bins for histogram
HISTOGRAM_BINS = 30
# Default diffusion coefficient range for plots
D_PLOT_RANGE = (0, 1.5)
# Output directory name format
OUTPUT_DIR_FORMAT = 'roi_diffusion_analysis_%Y%m%d_%H%M%S'
# =====================================

def load_roi_data(file_path):
    """
    Load ROI-assigned trajectory data from pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary containing the ROI trajectory data
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded ROI data from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading ROI data from {file_path}: {e}")
        return None

def create_frequency_distributions(filtered_data, output_path, bins=HISTOGRAM_BINS, d_range=D_PLOT_RANGE):
    """
    Create frequency distribution plots of diffusion coefficients per ROI.
    
    Args:
        filtered_data: Dictionary with filtered ROI trajectory data
        output_path: Path to save the visualization
        bins: Number of bins for histogram
        d_range: Range of diffusion values to plot (min, max)
        
    Returns:
        None (saves visualization to file)
    """
    # Get valid ROIs (excluding 'unassigned' and with sufficient data)
    valid_roi_ids = [
        roi_id for roi_id, stats in filtered_data['roi_statistics'].items() 
        if stats['n'] >= MIN_TRAJECTORIES_FOR_STATS and roi_id != 'unassigned'
    ]
    
    if not valid_roi_ids:
        print("No valid ROIs with sufficient data for frequency distribution")
        return
    
    # Determine the number of columns and rows for subplots
    n_plots = len(valid_roi_ids)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    
    # Create histograms for each ROI
    for i, roi_id in enumerate(valid_roi_ids):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        # Get diffusion values
        trajectories = filtered_data['roi_trajectories'][roi_id]
        D_values = [traj['D'] for traj in trajectories if 'D' in traj and not np.isnan(traj['D'])]
        
        if not D_values:
            continue
        
        # Create histogram
        counts, bin_edges, _ = ax.hist(D_values, bins=bins, range=d_range, alpha=0.7, density=True)
        
        # Add kernel density estimate
        if len(D_values) > 1:
            x = np.linspace(d_range[0], d_range[1], 1000)
            kde = stats.gaussian_kde(D_values)
            ax.plot(x, kde(x), 'r-', linewidth=2)
        
        # Add mean and median lines
        ax.axvline(np.mean(D_values), color='r', linestyle='--', linewidth=1.5, 
                  label=f'Mean: {np.mean(D_values):.3f} μm²/s')
        ax.axvline(np.median(D_values), color='g', linestyle='--', linewidth=1.5,
                  label=f'Median: {np.median(D_values):.3f} μm²/s')
        
        # Set axis labels and title
        ax.set_xlabel('Diffusion Coefficient (μm²/s)')
        ax.set_ylabel('Frequency (normalized)')
        
        # Shorten ROI ID for clarity
        short_id = roi_id.split('-')[0][:8]
        ax.set_title(f'{short_id} (n={len(D_values)})')
        
        # Add legend
        ax.legend(fontsize=9)
        
        # Add grid
        ax.grid(alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"ROI diffusion frequency distributions saved to {output_path}")

def create_cumulative_distributions(filtered_data, output_path, d_range=D_PLOT_RANGE):
    """
    Create cumulative distribution plots of diffusion coefficients across ROIs.
    
    Args:
        filtered_data: Dictionary with filtered ROI trajectory data
        output_path: Path to save the visualization
        d_range: Range of diffusion values to plot (min, max)
        
    Returns:
        None (saves visualization to file)
    """
    # Get valid ROIs (excluding 'unassigned' and with sufficient data)
    valid_roi_ids = [
        roi_id for roi_id, stats in filtered_data['roi_statistics'].items() 
        if stats['n'] >= MIN_TRAJECTORIES_FOR_STATS and roi_id != 'unassigned'
    ]
    
    if not valid_roi_ids:
        print("No valid ROIs with sufficient data for cumulative distribution")
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Define colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_roi_ids)))
    
    # Plot CDF for each ROI
    for i, roi_id in enumerate(valid_roi_ids):
        # Get diffusion values
        trajectories = filtered_data['roi_trajectories'][roi_id]
        D_values = [traj['D'] for traj in trajectories if 'D' in traj and not np.isnan(traj['D'])]
        
        if not D_values:
            continue
        
        # Calculate ECDF
        sorted_d = np.sort(D_values)
        ecdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        
        # Shorten ROI ID for clarity
        short_id = roi_id.split('-')[0][:8]
        
        # Plot CDF
        plt.plot(sorted_d, ecdf, '-', color=colors[i], linewidth=2,
                label=f"{short_id} (n={len(D_values)})")
        
        # Mark median value
        median_d = np.median(D_values)
        median_idx = np.searchsorted(sorted_d, median_d)
        if median_idx < len(ecdf):
            median_y = ecdf[median_idx]
            plt.scatter(median_d, median_y, s=80, facecolor=colors[i], edgecolor='black', zorder=3)
    
    # Set axis labels and title
    plt.xlabel('Diffusion Coefficient (μm²/s)', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Cumulative Distribution of Diffusion Coefficients Across ROIs', fontsize=14)
    
    # Set x-axis range
    plt.xlim(d_range)
    
    # Add 0.5 probability line
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Median (0.5)')
    
    # Add grid
    plt.grid(alpha=0.3)
    
    # Add legend
    plt.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"ROI diffusion cumulative distributions saved to {output_path}")

def compare_roi_diffusion(filtered_data, min_trajectories=MIN_TRAJECTORIES_FOR_STATS):
    """
    Perform statistical comparison between ROIs.
    
    Args:
        filtered_data: Dictionary with filtered ROI trajectory data
        min_trajectories: Minimum number of trajectories for valid statistics
        
    Returns:
        Dictionary with comparison results
    """
    comparison_results = {
        'roi_pairs': [],
        'p_values': [],
        'effect_sizes': [],
        'significant': [],
        'test_names': []
    }
    
    # Get ROI IDs with sufficient data (excluding 'unassigned')
    valid_roi_ids = [
        roi_id for roi_id, stats in filtered_data['roi_statistics'].items() 
        if stats['n'] >= min_trajectories and roi_id != 'unassigned'
    ]
    
    # Extract D values for each valid ROI
    roi_D_values = {}
    for roi_id in valid_roi_ids:
        trajectories = filtered_data['roi_trajectories'][roi_id]
        roi_D_values[roi_id] = [traj['D'] for traj in trajectories if 'D' in traj and not np.isnan(traj['D'])]
    
    # Compare each pair of ROIs
    for i, roi1 in enumerate(valid_roi_ids):
        for roi2 in valid_roi_ids[i+1:]:
            if not roi_D_values[roi1] or not roi_D_values[roi2]:
                continue
                
            # Check for normality
            if len(roi_D_values[roi1]) >= 3 and len(roi_D_values[roi2]) >= 3:
                _, norm1 = stats.shapiro(roi_D_values[roi1])
                _, norm2 = stats.shapiro(roi_D_values[roi2])
                
                normal_distribution = (norm1 > 0.05) and (norm2 > 0.05)
            else:
                # For small samples, assume non-normal
                normal_distribution = False
            
            # Choose appropriate test
            if normal_distribution:
                # Equal variance check
                _, p_levene = stats.levene(roi_D_values[roi1], roi_D_values[roi2])
                equal_variance = p_levene > 0.05
                
                # Perform t-test
                if equal_variance:
                    t_stat, p_value = stats.ttest_ind(roi_D_values[roi1], roi_D_values[roi2], equal_var=True)
                    test_name = "Student's t-test"
                else:
                    t_stat, p_value = stats.ttest_ind(roi_D_values[roi1], roi_D_values[roi2], equal_var=False)
                    test_name = "Welch's t-test"
                
                # Calculate effect size (Cohen's d)
                mean1, mean2 = np.mean(roi_D_values[roi1]), np.mean(roi_D_values[roi2])
                std1, std2 = np.std(roi_D_values[roi1]), np.std(roi_D_values[roi2])
                n1, n2 = len(roi_D_values[roi1]), len(roi_D_values[roi2])
                
                # Pooled standard deviation
                pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                effect_size = abs(mean1 - mean2) / pooled_std
            else:
                # Non-parametric Mann-Whitney U test
                u_stat, p_value = stats.mannwhitneyu(roi_D_values[roi1], roi_D_values[roi2])
                test_name = "Mann-Whitney U test"
                
                # Effect size for Mann-Whitney (r = Z / sqrt(N))
                n_total = len(roi_D_values[roi1]) + len(roi_D_values[roi2])
                z_score = stats.norm.ppf(1 - p_value/2)  # Two-tailed
                effect_size = abs(z_score) / np.sqrt(n_total)
            
            # Store results
            comparison_results['roi_pairs'].append((roi1, roi2))
            comparison_results['p_values'].append(p_value)
            comparison_results['effect_sizes'].append(effect_size)
            comparison_results['significant'].append(p_value < 0.05)
            comparison_results['test_names'].append(test_name)
    
    return comparison_results

def filter_trajectories(roi_data, r_squared_threshold=0.7):
    """
    Filter trajectories based on quality metrics.
    
    Args:
        roi_data: Dictionary containing ROI trajectory data
        r_squared_threshold: Minimum R² value for quality filter
        
    Returns:
        Dictionary with filtered trajectory data
    """
    filtered_data = {
        'roi_assignments': roi_data.get('roi_assignments', {}),
        'roi_trajectories': {},
        'roi_statistics': {}
    }
    
    for roi_id, trajectories in roi_data['roi_trajectories'].items():
        filtered_trajectories = []
        for traj in trajectories:
            # Apply quality filters
            if 'r_squared' in traj and traj['r_squared'] >= r_squared_threshold and not np.isnan(traj.get('D', np.nan)):
                filtered_trajectories.append(traj)
        
        filtered_data['roi_trajectories'][roi_id] = filtered_trajectories
        
        # Recalculate statistics with filtered trajectories
        if filtered_trajectories:
            D_values = [traj['D'] for traj in filtered_trajectories if 'D' in traj and not np.isnan(traj['D'])]
            
            if D_values:
                filtered_data['roi_statistics'][roi_id] = {
                    'n': len(D_values),
                    'mean_D': np.mean(D_values),
                    'median_D': np.median(D_values),
                    'std_D': np.std(D_values),
                    'sem_D': np.std(D_values) / np.sqrt(len(D_values)),
                    'min_D': np.min(D_values),
                    'max_D': np.max(D_values)
                }
            else:
                filtered_data['roi_statistics'][roi_id] = {
                    'n': 0,
                    'mean_D': np.nan,
                    'median_D': np.nan,
                    'std_D': np.nan,
                    'sem_D': np.nan,
                    'min_D': np.nan,
                    'max_D': np.nan
                }
    
    return filtered_data

def main():
    """
    Main function to analyze diffusion within ROIs and create distribution plots.
    """
    # Ask for input path
    roi_data_file = input("Enter path to ROI trajectory data file (.pkl): ")
    
    # Load ROI data
    roi_data = load_roi_data(roi_data_file)
    if roi_data is None:
        print("Failed to load ROI data. Exiting.")
        return
    
    # Create output directory
    output_dir_name = datetime.now().strftime(OUTPUT_DIR_FORMAT)
    output_base_dir = os.path.dirname(roi_data_file)
    output_dir = os.path.join(output_base_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter trajectories based on quality
    r_squared_threshold = float(input(f"Enter R² threshold for trajectory filtering (default: 0.7): ") or 0.7)
    filtered_data = filter_trajectories(roi_data, r_squared_threshold)
    
    # Create frequency and cumulative distributions
    create_frequency_distributions(filtered_data, os.path.join(output_dir, 'roi_diffusion_frequencies.png'))
    create_cumulative_distributions(filtered_data, os.path.join(output_dir, 'roi_diffusion_cumulative.png'))
    
    # Perform statistical comparison between ROIs
    print("\nCalculating ROI-specific diffusion statistics...")
    comparison_results = compare_roi_diffusion(filtered_data)
    
    # Print comparison results
    if comparison_results['roi_pairs']:
        print("\nStatistical comparisons between ROIs:")
        for i, (roi1, roi2) in enumerate(comparison_results['roi_pairs']):
            p_value = comparison_results['p_values'][i]
            effect_size = comparison_results['effect_sizes'][i]
            test_name = comparison_results['test_names'][i]
            
            sig_str = "significant" if p_value < 0.05 else "not significant"
            print(f"{roi1} vs {roi2}: p = {p_value:.4f} ({sig_str}), effect size = {effect_size:.2f}, test = {test_name}")
    else:
        print("No statistical comparisons performed (insufficient data)")
    
    # Export comparison results to CSV if there are any
    if comparison_results['roi_pairs']:
        comparisons_df = pd.DataFrame({
            'ROI_1': [pair[0] for pair in comparison_results['roi_pairs']],
            'ROI_2': [pair[1] for pair in comparison_results['roi_pairs']],
            'P_Value': comparison_results['p_values'],
            'Effect_Size': comparison_results['effect_sizes'],
            'Significant': comparison_results['significant'],
            'Test': comparison_results['test_names']
        })
        
        comparisons_df.to_csv(os.path.join(output_dir, 'roi_comparisons.csv'), index=False)
        print(f"\nComparison results saved to {os.path.join(output_dir, 'roi_comparisons.csv')}")
    
    # Save the filtered data for future use
    output_file = os.path.join(output_dir, 'roi_diffusion_analysis.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump({
            'filtered_data': filtered_data,
            'comparison_results': comparison_results
        }, f)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()