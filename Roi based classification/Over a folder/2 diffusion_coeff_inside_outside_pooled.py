# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 22:16:08 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diffusion_coeff_pooled.py

This script analyzes the diffusion properties of trajectories inside and outside ROIs
by pooling data from multiple ROI-assigned trajectory pkl files in a folder.

Input:
- Folder containing multiple roi_trajectory_data_*.pkl files

Output:
- Summary statistics for pooled diffusion coefficients inside and outside ROIs
- Histogram of diffusion coefficients inside and outside ROIs
- CDF plot of diffusion coefficients inside and outside ROIs
- Distribution comparison plot of diffusion coefficients inside and outside ROIs
- Box plot comparing diffusion coefficients inside and outside ROIs

Usage:
python diffusion_coeff_pooled.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob

# Global parameters
# =====================================
# Bin size for diffusion coefficient histogram (in µm^2/s) 
D_BIN_SIZE = 0.05
# Output subfolder name
OUTPUT_SUBFOLDER = 'pooled_diffusion_analysis'
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
        Tuple of (inside_D_array, outside_D_array) with pooled data
    """
    # Find all roi_trajectory_data pkl files
    pkl_pattern = os.path.join(folder_path, 'roi_trajectory_data_*.pkl')
    pkl_files = glob.glob(pkl_pattern)
    
    if not pkl_files:
        print(f"No roi_trajectory_data_*.pkl files found in {folder_path}")
        return None, None
    
    print(f"\nFound {len(pkl_files)} pkl files to process")
    print("="*50)
    
    # Lists to collect all diffusion coefficients
    all_inside_D = []
    all_outside_D = []
    
    # Process each file
    for pkl_file in pkl_files:
        roi_data = load_roi_data(pkl_file)
        
        if roi_data is not None:
            inside_D, outside_D = analyze_diffusion_single_file(roi_data)
            all_inside_D.extend(inside_D)
            all_outside_D.extend(outside_D)
            print(f"  Inside: {len(inside_D)}, Outside: {len(outside_D)}")
    
    print("="*50)
    print(f"Total pooled - Inside: {len(all_inside_D)}, Outside: {len(all_outside_D)}")
    
    return np.array(all_inside_D), np.array(all_outside_D)


def plot_diffusion_histogram(inside_D, outside_D, output_path):
    """
    Plot histogram of diffusion coefficients inside and outside ROIs.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Determine bin range
    max_D = max(np.max(inside_D), np.max(outside_D))
    bins = np.arange(0, max_D + D_BIN_SIZE, D_BIN_SIZE)
    
    # Plot histograms
    ax.hist(inside_D, bins=bins, alpha=0.7, label='Inside ROIs')
    ax.hist(outside_D, bins=bins, alpha=0.7, label='Outside ROIs')
    
    ax.set_xlabel('Diffusion Coefficient ($\mu m^2/s$)')
    ax.set_ylabel('Count')
    ax.set_title('Pooled Diffusion Coefficient Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Histogram saved to {output_path}")


def plot_diffusion_cdf(inside_D, outside_D, output_path):
    """
    Plot CDF of diffusion coefficients inside and outside ROIs.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute and plot CDF for inside ROIs
    inside_cdf = np.sort(inside_D)
    inside_yvals = np.arange(len(inside_cdf))/float(len(inside_cdf))
    ax.plot(inside_cdf, inside_yvals, label='Inside ROIs', linewidth=2)
    
    # Compute and plot CDF for outside ROIs
    outside_cdf = np.sort(outside_D)
    outside_yvals = np.arange(len(outside_cdf))/float(len(outside_cdf))
    ax.plot(outside_cdf, outside_yvals, label='Outside ROIs', linewidth=2)
    
    ax.set_xlabel('Diffusion Coefficient ($\mu m^2/s$)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Pooled Diffusion Coefficient CDF')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"CDF plot saved to {output_path}")


def plot_diffusion_distributions(inside_D, outside_D, output_path):
    """
    Plot distribution comparison of diffusion coefficients inside and outside ROIs.
    """  
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bins = np.linspace(0, max(np.max(inside_D), np.max(outside_D)), 50)
    
    ax.hist(inside_D, bins=bins, alpha=0.5, density=True, label='Inside ROIs')
    ax.hist(outside_D, bins=bins, alpha=0.5, density=True, label='Outside ROIs')
    
    ax.set_xlabel('Diffusion Coefficient ($\mu m^2/s$)')  
    ax.set_ylabel('Density')
    ax.set_title('Pooled Diffusion Coefficient Distribution Comparison')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300) 
    plt.close()
    
    print(f"Distribution comparison saved to {output_path}")


def plot_diffusion_boxplot(inside_D, outside_D, output_path):
    """
    Plot box plot comparing diffusion coefficients inside and outside ROIs.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.boxplot([inside_D, outside_D], labels=['Inside ROIs', 'Outside ROIs'])
    
    ax.set_ylabel('Diffusion Coefficient ($\mu m^2/s$)')
    ax.set_title('Pooled Diffusion Coefficient Box Plot')
    
    plt.tight_layout() 
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Box plot saved to {output_path}")


def print_diffusion_summary(inside_D, outside_D):
    """
    Print summary statistics for pooled diffusion coefficients.
    """
    print("\n" + "="*50)
    print("Pooled Diffusion Summary")
    print("="*50)
    
    print(f"\nInside ROIs (n={len(inside_D)}):")
    print(f"  Mean:   {np.mean(inside_D):.3f} µm^2/s")
    print(f"  Median: {np.median(inside_D):.3f} µm^2/s") 
    print(f"  Std:    {np.std(inside_D):.3f} µm^2/s")
    print(f"  Min:    {np.min(inside_D):.3f} µm^2/s")
    print(f"  Max:    {np.max(inside_D):.3f} µm^2/s")
    
    print(f"\nOutside ROIs (n={len(outside_D)}):")  
    print(f"  Mean:   {np.mean(outside_D):.3f} µm^2/s")
    print(f"  Median: {np.median(outside_D):.3f} µm^2/s")
    print(f"  Std:    {np.std(outside_D):.3f} µm^2/s") 
    print(f"  Min:    {np.min(outside_D):.3f} µm^2/s")
    print(f"  Max:    {np.max(outside_D):.3f} µm^2/s")
    print("="*50)


def main():
    """
    Main function to analyze pooled diffusion properties.
    """
    print("\nPooled ROI Diffusion Analyzer")
    print("="*50)
    
    # Ask for input folder
    folder_path = input("Enter path to folder containing roi_trajectory_data pkl files: ")
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        return
    
    # Pool diffusion data from all files
    inside_D, outside_D = pool_diffusion_data(folder_path)
    
    if inside_D is None or outside_D is None:
        print("No data to analyze. Exiting.")
        return
    
    if len(inside_D) == 0 and len(outside_D) == 0:
        print("No valid diffusion coefficients found. Exiting.")
        return
    
    # Create output subfolder
    output_dir = os.path.join(folder_path, OUTPUT_SUBFOLDER)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput will be saved to: {output_dir}")
    
    # Print summary statistics 
    print_diffusion_summary(inside_D, outside_D)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_diffusion_histogram(inside_D, outside_D, 
                            os.path.join(output_dir, 'pooled_diffusion_histogram.png'))
    
    plot_diffusion_cdf(inside_D, outside_D, 
                      os.path.join(output_dir, 'pooled_diffusion_cdf.png'))
    
    plot_diffusion_distributions(inside_D, outside_D, 
                                os.path.join(output_dir, 'pooled_diffusion_distributions.png'))
    
    plot_diffusion_boxplot(inside_D, outside_D, 
                          os.path.join(output_dir, 'pooled_diffusion_boxplot.png'))
    
    print(f"\nAll results saved to: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()