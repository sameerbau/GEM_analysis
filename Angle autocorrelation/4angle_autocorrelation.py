# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 13:14:00 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
angle_autocorrelation.py

This script calculates and plots angle autocorrelation functions from trajectory data.
It's designed to work with the output from the trajectory analysis pipeline.

Input:
- Analyzed trajectory data (.pkl files) from diffusion_analyzer.py or trajectory_data_pooler.py

Output:
- Angle autocorrelation plots and data saved in a results folder
- CSV file with crossing times and other metrics

Usage:
python angle_autocorrelation.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
from pathlib import Path

# Global parameters (can be modified)
# =====================================
# Time interval between frames in seconds
DT = 0.1
# Angle correlation figure up to this length
ANGLE_PLOT_CUTOFF = 50
# Cutoff length for trajectory, only consider trajectories with length > L_CUTOFF
L_CUTOFF = 15
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
        print(f"Error loading data from {file_path}: {e}")
        return None

def calculate_angle_autocorrelation(trajectories, dt=DT, angle_plot_cutoff=ANGLE_PLOT_CUTOFF, l_cutoff=L_CUTOFF):
    """
    Calculate angle autocorrelation function for trajectories.
    
    Args:
        trajectories: List of trajectory dictionaries
        dt: Time interval between frames
        angle_plot_cutoff: Maximum lag to calculate
        l_cutoff: Minimum trajectory length to consider
        
    Returns:
        Dictionary with angle autocorrelation results
    """
    # Filter trajectories by length
    filtered_trajectories = [traj for traj in trajectories if len(traj['x']) > l_cutoff]
    
    if not filtered_trajectories:
        print("No trajectories longer than L_CUTOFF found")
        return None
    
    # Initialize results
    cos_angle_t_total = [[] for _ in range(angle_plot_cutoff)]
    
    # Calculate angle correlations for each trajectory
    for traj in filtered_trajectories:
        x_temp = traj['x']
        y_temp = traj['y']
        
        # Calculate displacement vectors
        delta_x = np.diff(x_temp)
        delta_y = np.diff(y_temp)
        
        # Calculate angle correlations for different time lags
        for i in range(1, angle_plot_cutoff + 1):
            if len(delta_x) <= i:
                continue
                
            cos_angle_t_temp_j = []
            for k in range(len(delta_x) - i):
                # Calculate dot product of displacement vectors
                dot_product = delta_x[k] * delta_x[k+i] + delta_y[k] * delta_y[k+i]
                # Calculate magnitudes
                mag1 = np.sqrt(delta_x[k]**2 + delta_y[k]**2)
                mag2 = np.sqrt(delta_x[k+i]**2 + delta_y[k+i]**2)
                
                # Calculate cosine of angle between vectors
                if mag1 > 0 and mag2 > 0:
                    cos_angle = dot_product / (mag1 * mag2)
                    # Ensure value is in valid range for cosine
                    cos_angle = max(min(cos_angle, 1.0), -1.0)
                    cos_angle_t_temp_j.append(cos_angle)
            
            cos_angle_t_total[i-1].extend(cos_angle_t_temp_j)
    
    # Calculate average cos(theta(tau)) and standard error
    mean_cos_angle = np.zeros(angle_plot_cutoff)
    sem_cos_angle = np.zeros(angle_plot_cutoff)
    
    for i in range(angle_plot_cutoff):
        temp = np.array(cos_angle_t_total[i])
        temp = temp[~np.isnan(temp)]
        
        if len(temp) > 0:
            mean_cos_angle[i] = np.mean(temp)
            sem_cos_angle[i] = np.std(temp) / np.sqrt(len(temp))
        else:
            mean_cos_angle[i] = np.nan
            sem_cos_angle[i] = np.nan
    
    # Calculate crossing time (time to first cross zero)
    t_cross = np.nan
    index_temp = np.where(mean_cos_angle < 0)[0]
    
    if len(index_temp) > 0:
        x2 = index_temp[0] + 1  # +1 because Python is 0-indexed
        
        if x2 >= 2:
            x1 = x2 - 1
            y2 = mean_cos_angle[x2-1]
            y1 = mean_cos_angle[x1-1]
            t_cross = (x1*y2 - x2*y1)*dt / (y2-y1)
        else:
            y2 = mean_cos_angle[1]
            y1 = mean_cos_angle[0]
            t_cross = (y2 - 2*y1)*dt / (y2-y1)
    
    return {
        'mean_cos_angle': mean_cos_angle,
        'sem_cos_angle': sem_cos_angle,
        't_cross': t_cross,
        'time_lags': np.arange(1, angle_plot_cutoff + 1) * dt,
        'num_trajectories': len(filtered_trajectories)
    }

def main():
    """Main function to calculate angle autocorrelation for trajectory data."""
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
    
    print(f"Found {len(file_paths)} files to analyze")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(input_dir), "angle_autocorrelation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    all_results = []
    
    # Set up figure
    plt.figure(figsize=(12, 8))
    
    # Use colormap for different files
    colors = plt.cm.jet(np.linspace(0, 1, len(file_paths)))
    
    for i, file_path in enumerate(file_paths):
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0].replace('analyzed_', '')
        
        print(f"Processing {filename}")
        
        # Load analyzed data
        analyzed_data = load_analyzed_data(file_path)
        
        if analyzed_data is None:
            print(f"Skipping {filename} due to loading errors")
            continue
        
        # Calculate angle autocorrelation
        results = calculate_angle_autocorrelation(analyzed_data['trajectories'])
        
        if results is None:
            print(f"Skipping {filename} - no valid trajectories")
            continue
        
        # Store results
        results['filename'] = base_name
        all_results.append(results)
        
        # Plot data
        plt.errorbar(
            results['time_lags'], 
            results['mean_cos_angle'], 
            yerr=results['sem_cos_angle'],
            fmt='.-', 
            color=colors[i],
            label=f"{base_name} (n={results['num_trajectories']})"
        )
    
    # Finalize plot
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel(r'$\langle \cos \theta(t) \rangle_{TE}$', fontsize=12)
    plt.title('Angle Correlation Analysis', fontsize=14)
    plt.grid(True)
    plt.legend(loc='best')
    
    # Add annotation with crossing times
    annotation_text = []
    for result in all_results:
        if np.isnan(result['t_cross']):
            annotation_text.append(f"{result['filename']}: No crossing point")
        else:
            annotation_text.append(f"{result['filename']}: t_cross = {result['t_cross']:.3f} s")
    
    # Add text box with crossing times
    plt.text(
        0.05, 0.95, 
        '\n'.join(['Crossing times:'] + annotation_text),
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'angle_autocorrelation_comparison.png'), dpi=300)
    
    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'Filename': r['filename'],
            'NumTrajectories': r['num_trajectories'],
            'CrossingTime': r['t_cross']
        } for r in all_results
    ])
    
    results_df.to_csv(os.path.join(output_dir, 'angle_autocorrelation_results.csv'), index=False)
    
    # Save individual angle correlation data
    for result in all_results:
        df = pd.DataFrame({
            'TimeLag': result['time_lags'],
            'MeanCosAngle': result['mean_cos_angle'],
            'SEM': result['sem_cos_angle']
        })
        df.to_csv(os.path.join(output_dir, f"{result['filename']}_angle_correlation.csv"), index=False)
    
    print(f"\nAnalysis Summary:")
    for result in all_results:
        if np.isnan(result['t_cross']):
            print(f"{result['filename']}: No crossing point found")
        else:
            print(f"{result['filename']}: t_cross = {result['t_cross']:.3f} s")
    
    print(f"\nResults saved in {output_dir}")

if __name__ == "__main__":
    main()