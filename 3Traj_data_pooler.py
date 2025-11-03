# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 13:04:20 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trajectory_data_pooler.py

This script pools together analyzed trajectory data from multiple files,
calculates overall statistics, and generates summary plots.

Input:
- Analyzed trajectory data (.pkl files) from diffusion_analyzer.py

Output:
- Combined statistics for all analyzed trajectories
- Summary plots comparing different conditions
- CSV file with pooled results for further analysis

Usage:
python trajectory_data_pooler.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob
import pickle
from pathlib import Path

# Global parameters that can be modified
# =====================================
# Minimum R² value for including a trajectory in the pooled analysis
MIN_R_SQUARED = 0.8
# Minimum track length (in frames) for inclusion
MIN_TRACK_LENGTH = 10
# Maximum diffusion coefficient to include (µm²/s)
# Set to None to include all values, or a number to filter out unrealistically high values
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
        Dictionary with filtered trajectory data
    """
    filtered_data = {
        'trajectories': [],
        'diffusion_coefficients': [],
        'radius_of_gyration': [],
        'track_lengths': [],
        'r_squared_values': []
    }
    
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
        filtered_data['trajectories'].append(traj)
        filtered_data['diffusion_coefficients'].append(traj['D'])
        filtered_data['radius_of_gyration'].append(traj['radius_of_gyration'])
        filtered_data['track_lengths'].append(traj['track_length'])
        filtered_data['r_squared_values'].append(traj['r_squared'])
    
    return filtered_data

def pool_data(file_paths):
    """
    Pool data from multiple analyzed files.
    
    Args:
        file_paths: List of paths to analyzed data files
        
    Returns:
        Dictionary containing pooled data for each file and overall
    """
    pooled_data = {
        'per_file': {},
        'overall': {
            'trajectories': [],
            'diffusion_coefficients': [],
            'radius_of_gyration': [],
            'track_lengths': [],
            'r_squared_values': [],
            'file_sources': []
        }
    }
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0].replace('analyzed_tracked_', '')
        
        # Load data
        analyzed_data = load_analyzed_data(file_path)
        
        if analyzed_data is None:
            print(f"Skipping {filename} due to loading errors")
            continue
        
        # Filter trajectories
        filtered_data = filter_trajectories(analyzed_data)
        
        # Store per-file data
        pooled_data['per_file'][base_name] = filtered_data
        
        # Add to overall data
        pooled_data['overall']['trajectories'].extend(filtered_data['trajectories'])
        pooled_data['overall']['diffusion_coefficients'].extend(filtered_data['diffusion_coefficients'])
        pooled_data['overall']['radius_of_gyration'].extend(filtered_data['radius_of_gyration'])
        pooled_data['overall']['track_lengths'].extend(filtered_data['track_lengths'])
        pooled_data['overall']['r_squared_values'].extend(filtered_data['r_squared_values'])
        
        # Record file source for each trajectory
        pooled_data['overall']['file_sources'].extend([base_name] * len(filtered_data['trajectories']))
        
        print(f"Added {len(filtered_data['trajectories'])} trajectories from {base_name}")
    
    return pooled_data

def calculate_statistics(data):
    """
    Calculate statistics for trajectory data.
    
    Args:
        data: Dictionary containing trajectory data
        
    Returns:
        Dictionary with calculated statistics
    """
    stats_data = {}
    
    if not data['diffusion_coefficients']:
        return {
            'mean_D': np.nan,
            'median_D': np.nan,
            'std_D': np.nan,
            'sem_D': np.nan,
            'count': 0,
            'mean_Rg': np.nan,
            'median_Rg': np.nan,
            'std_Rg': np.nan,
            'sem_Rg': np.nan
        }
    
    # Diffusion coefficient statistics
    stats_data['mean_D'] = np.mean(data['diffusion_coefficients'])
    stats_data['median_D'] = np.median(data['diffusion_coefficients'])
    stats_data['std_D'] = np.std(data['diffusion_coefficients'])
    stats_data['sem_D'] = stats.sem(data['diffusion_coefficients'])
    stats_data['count'] = len(data['diffusion_coefficients'])
    
    # Radius of gyration statistics
    stats_data['mean_Rg'] = np.mean(data['radius_of_gyration'])
    stats_data['median_Rg'] = np.median(data['radius_of_gyration'])
    stats_data['std_Rg'] = np.std(data['radius_of_gyration'])
    stats_data['sem_Rg'] = stats.sem(data['radius_of_gyration'])
    
    return stats_data

def create_summary_plots(pooled_data, output_path):
    """
    Create summary plots for the pooled trajectory data.
    
    Args:
        pooled_data: Dictionary containing pooled trajectory data
        output_path: Directory to save the plots
    """
    # First, check if we have enough data
    if not pooled_data['overall']['diffusion_coefficients']:
        print("No valid trajectories to plot")
        return
    
    # Set up a nicer plotting style
    sns.set(style="whitegrid")
    
    # Figure 1: Diffusion coefficient comparison between files
    plt.figure(figsize=(12, 8))
    
    # Prepare data for boxplot
    boxplot_data = []
    boxplot_labels = []
    
    for file_name, file_data in pooled_data['per_file'].items():
        if file_data['diffusion_coefficients']:
            boxplot_data.append(file_data['diffusion_coefficients'])
            boxplot_labels.append(file_name)
    
    if boxplot_data:
        # Create violin plot with individual points
        ax = plt.subplot(111)
        violin_parts = ax.violinplot(boxplot_data, showmeans=True, showmedians=True)
        
        # Color the violin plot parts
        for pc in violin_parts['bodies']:
            pc.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
            if partname in violin_parts:
                violin_parts[partname].set_edgecolor('black')
        
        # Add scatter points
        for i, d in enumerate(boxplot_data):
            # Add jitter to x position
            x = np.random.normal(i+1, 0.05, size=len(d))
            plt.scatter(x, d, alpha=0.6, s=5, color='black')
        
        # Add sample count
        for i, (label, data) in enumerate(zip(boxplot_labels, boxplot_data)):
            plt.text(i+1, plt.ylim()[1] * 0.95, f"n={len(data)}", 
                     horizontalalignment='center', fontsize=10)
        
        plt.xticks(np.arange(1, len(boxplot_labels) + 1), boxplot_labels, rotation=45, ha='right')
        plt.ylabel('Diffusion coefficient (µm²/s)')
        plt.title('Comparison of diffusion coefficients across datasets')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "diffusion_comparison.png"), dpi=300)
        plt.close()
    
    # Figure 2: Correlation between D and Rg
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with linear regression
    sns.regplot(
        x=pooled_data['overall']['radius_of_gyration'],
        y=pooled_data['overall']['diffusion_coefficients'],
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red'}
    )
    
    # Calculate and display correlation
    corr, p_value = stats.pearsonr(
        pooled_data['overall']['radius_of_gyration'],
        pooled_data['overall']['diffusion_coefficients']
    )
    
    plt.title(f'Correlation between diffusion coefficient and radius of gyration\nr = {corr:.4f}, p = {p_value:.4e}')
    plt.xlabel('Radius of gyration (µm)')
    plt.ylabel('Diffusion coefficient (µm²/s)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "D_vs_Rg_correlation.png"), dpi=300)
    plt.close()
    
    # Figure 3: Histograms of combined data
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Diffusion coefficient histogram
    sns.histplot(
        pooled_data['overall']['diffusion_coefficients'],
        kde=True,
        ax=axs[0]
    )
    axs[0].axvline(np.mean(pooled_data['overall']['diffusion_coefficients']), 
                  color='r', linestyle='--', 
                  label=f'Mean: {np.mean(pooled_data["overall"]["diffusion_coefficients"]):.4f} µm²/s')
    axs[0].axvline(np.median(pooled_data['overall']['diffusion_coefficients']), 
                   color='g', linestyle='--', 
                   label=f'Median: {np.median(pooled_data["overall"]["diffusion_coefficients"]):.4f} µm²/s')
    axs[0].set_xlabel('Diffusion coefficient (µm²/s)')
    axs[0].set_ylabel('Count')
    axs[0].set_title(f'Distribution of diffusion coefficients (n={len(pooled_data["overall"]["diffusion_coefficients"])})')
    axs[0].legend()
    
    # Radius of gyration histogram
    sns.histplot(
        pooled_data['overall']['radius_of_gyration'],
        kde=True,
        ax=axs[1]
    )
    axs[1].axvline(np.mean(pooled_data['overall']['radius_of_gyration']), 
                  color='r', linestyle='--', 
                  label=f'Mean: {np.mean(pooled_data["overall"]["radius_of_gyration"]):.4f} µm')
    axs[1].axvline(np.median(pooled_data['overall']['radius_of_gyration']), 
                   color='g', linestyle='--', 
                   label=f'Median: {np.median(pooled_data["overall"]["radius_of_gyration"]):.4f} µm')
    axs[1].set_xlabel('Radius of gyration (µm)')
    axs[1].set_ylabel('Count')
    axs[1].set_title(f'Distribution of radius of gyration (n={len(pooled_data["overall"]["radius_of_gyration"])})')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "pooled_data_histograms.png"), dpi=300)
    plt.close()
    
    # Figure 4: Color-coded scatter plot by file
    if len(pooled_data['per_file']) > 1:
        plt.figure(figsize=(12, 8))
        
        # Create DataFrame for easier plotting with color coding
        df = pd.DataFrame({
            'Radius of gyration (µm)': pooled_data['overall']['radius_of_gyration'],
            'Diffusion coefficient (µm²/s)': pooled_data['overall']['diffusion_coefficients'],
            'Source': pooled_data['overall']['file_sources']
        })
        
        # Create scatter plot with color coding by source file
        sns.scatterplot(
            data=df,
            x='Radius of gyration (µm)',
            y='Diffusion coefficient (µm²/s)',
            hue='Source',
            alpha=0.7
        )
        
        plt.title('Diffusion coefficient vs radius of gyration by dataset')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "D_vs_Rg_by_dataset.png"), dpi=300)
        plt.close()

def export_to_csv(pooled_data, output_path):
    """
    Export pooled data to CSV files.
    
    Args:
        pooled_data: Dictionary containing pooled trajectory data
        output_path: Directory to save the CSV files
    """
    # Per file summary statistics
    file_stats = []
    
    for file_name, file_data in pooled_data['per_file'].items():
        stats = calculate_statistics(file_data)
        stats['file_name'] = file_name
        file_stats.append(stats)
    
    if file_stats:
        stats_df = pd.DataFrame(file_stats)
        stats_df = stats_df[['file_name', 'count', 'mean_D', 'median_D', 'std_D', 'sem_D', 
                              'mean_Rg', 'median_Rg', 'std_Rg', 'sem_Rg']]
        stats_df.to_csv(os.path.join(output_path, "file_statistics.csv"), index=False)
        print(f"File statistics saved to {os.path.join(output_path, 'file_statistics.csv')}")
    
    # Combined trajectory data
    if pooled_data['overall']['trajectories']:
        traj_data = []
        
        for i, traj in enumerate(pooled_data['overall']['trajectories']):
            traj_data.append({
                'source_file': pooled_data['overall']['file_sources'][i],
                'trajectory_id': traj['id'],
                'diffusion_coefficient': traj['D'],
                'D_error': traj['D_err'],
                'r_squared': traj['r_squared'],
                'radius_of_gyration': traj['radius_of_gyration'],
                'track_length': traj['track_length']
            })
        
        traj_df = pd.DataFrame(traj_data)
        traj_df.to_csv(os.path.join(output_path, "all_trajectories.csv"), index=False)
        print(f"Trajectory data saved to {os.path.join(output_path, 'all_trajectories.csv')}")

def main():
    """Main function to pool analyzed trajectory data."""
    # Ask for input directory
    input_dir = input("Enter the directory containing analyzed trajectory files (press Enter for analyzed_trajectories): ")
    
    if input_dir == "":
        # Default to the analyzed_trajectories directory in the current folder
        input_dir = os.path.join(os.getcwd(), "analyzed_trajectories")
    
    # Check if directory exists
    if not os.path.isdir(input_dir):
        print(f"Directory {input_dir} does not exist")
        return
    
    # Get list of analyzed files
    file_paths = glob.glob(os.path.join(input_dir, "analyzed_*.pkl"))
    
    if not file_paths:
        print(f"No analyzed trajectory files found in {input_dir}")
        return
    
    print(f"Found {len(file_paths)} files to pool")
    
    # Create output directory for pooled results
    output_dir = os.path.join(os.path.dirname(input_dir), "pooled_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Pool data from all files
    pooled_data = pool_data(file_paths)
    
    # Print summary statistics for overall data
    overall_stats = calculate_statistics(pooled_data['overall'])
    
    print("\nOverall statistics:")
    print(f"Total number of trajectories: {overall_stats['count']}")
    print(f"Mean diffusion coefficient: {overall_stats['mean_D']:.6f} ± {overall_stats['sem_D']:.6f} µm²/s")
    print(f"Median diffusion coefficient: {overall_stats['median_D']:.6f} µm²/s")
    print(f"Mean radius of gyration: {overall_stats['mean_Rg']:.6f} ± {overall_stats['sem_Rg']:.6f} µm")
    
    # Print per-file statistics
    print("\nPer-file statistics:")
    for file_name, file_data in pooled_data['per_file'].items():
        stats = calculate_statistics(file_data)
        print(f"\n{file_name}:")
        print(f"  Trajectories: {stats['count']}")
        print(f"  Mean D: {stats['mean_D']:.6f} ± {stats['sem_D']:.6f} µm²/s")
        print(f"  Mean Rg: {stats['mean_Rg']:.6f} ± {stats['sem_Rg']:.6f} µm")
    
    # Create summary plots
    create_summary_plots(pooled_data, output_dir)
    
    # Export data to CSV
    export_to_csv(pooled_data, output_dir)
    
    # Save pooled data for further analysis
    output_file = os.path.join(output_dir, "pooled_data.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(pooled_data, f)
    
    print(f"\nPooled data saved to {output_file}")
    print(f"Summary plots and statistics saved in {output_dir}")

if __name__ == "__main__":
    main()