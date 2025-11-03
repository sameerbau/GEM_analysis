# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 13:15:09 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
velocity_autocorrelation.py

This script calculates and plots velocity autocorrelation functions from trajectory data.
It measures how velocity correlations decay over time.

Input:
- Analyzed trajectory data (.pkl files) from diffusion_analyzer.py or trajectory_data_pooler.py

Output:
- Velocity autocorrelation plots and data saved in a results folder
- CSV file with correlation times and other metrics

Usage:
python velocity_autocorrelation.py
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
# Time step in seconds
DT = 0.1
# Minimum trajectory length
L_CUTOFF = 30
# Maximum time lag to analyze
MAX_TAU = 50
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

def calculate_velocity_autocorrelation(trajectories, dt=DT, max_tau=MAX_TAU, l_cutoff=L_CUTOFF):
    """
    Calculate velocity autocorrelation function for trajectories.
    
    Args:
        trajectories: List of trajectory dictionaries
        dt: Time interval between frames
        max_tau: Maximum lag to calculate
        l_cutoff: Minimum trajectory length to consider
        
    Returns:
        Dictionary with velocity autocorrelation results
    """
    # Filter trajectories by length
    filtered_trajectories = [traj for traj in trajectories if len(traj['x']) > l_cutoff]
    
    if not filtered_trajectories:
        print("No trajectories longer than L_CUTOFF found")
        return None
    
    # Calculate velocities for each trajectory
    vx_all = []
    vy_all = []
    
    for traj in filtered_trajectories:
        x = traj['x']
        y = traj['y']
        
        # Calculate velocities (using central difference for better accuracy)
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        
        vx_all.append(vx)
        vy_all.append(vy)
    
    # Calculate velocity autocorrelation
    Cv = np.zeros(max_tau)
    Cv_sem = np.zeros(max_tau)
    
    for tau in range(1, max_tau + 1):
        corr_values = []
        
        for j in range(len(filtered_trajectories)):
            vx = vx_all[j]
            vy = vy_all[j]
            
            if len(vx) > tau:
                # Calculate normalized velocity correlation
                vx_tau = vx[:-tau]
                vy_tau = vy[:-tau]
                vx_t = vx[tau:]
                vy_t = vy[tau:]
                
                # Normalize velocities
                v_mag_tau = np.sqrt(vx_tau**2 + vy_tau**2)
                v_mag_t = np.sqrt(vx_t**2 + vy_t**2)
                
                # Calculate correlation
                valid_indices = (v_mag_tau > 0) & (v_mag_t > 0)
                if np.any(valid_indices):
                    corr = (vx_tau[valid_indices] * vx_t[valid_indices] + 
                            vy_tau[valid_indices] * vy_t[valid_indices]) / (
                            v_mag_tau[valid_indices] * v_mag_t[valid_indices])
                    corr_values.extend(corr)
        
        # Calculate mean and SEM
        if corr_values:
            Cv[tau-1] = np.nanmean(corr_values)
            Cv_sem[tau-1] = np.nanstd(corr_values) / np.sqrt(np.sum(~np.isnan(corr_values)))
        else:
            Cv[tau-1] = np.nan
            Cv_sem[tau-1] = np.nan
    
    # Calculate correlation time (time to decay to 1/e)
    correlation_time = np.nan
    if not np.isnan(Cv[0]):
        threshold = Cv[0] / np.e
        indices = np.where(Cv < threshold)[0]
        if len(indices) > 0:
            correlation_time = dt * (indices[0] + 1)
    
    return {
        'Cv': Cv,
        'Cv_sem': Cv_sem,
        'correlation_time': correlation_time,
        'time_lags': np.arange(1, max_tau + 1) * dt,
        'num_trajectories': len(filtered_trajectories)
    }

def main():
    """Main function to calculate velocity autocorrelation for trajectory data."""
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
    output_dir = os.path.join(os.path.dirname(input_dir), "velocity_autocorrelation")
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
        
        # Calculate velocity autocorrelation
        results = calculate_velocity_autocorrelation(analyzed_data['trajectories'])
        
        if results is None:
            print(f"Skipping {filename} - no valid trajectories")
            continue
        
        # Store results
        results['filename'] = base_name
        all_results.append(results)
        
        # Plot data
        plt.errorbar(
            results['time_lags'], 
            results['Cv'], 
            yerr=results['Cv_sem'],
            fmt='.-', 
            color=colors[i],
            label=f"{base_name} (n={results['num_trajectories']})"
        )
    
    # Finalize plot
    plt.xlabel('Time lag (s)', fontsize=12)
    plt.ylabel(r'Velocity Autocorrelation $C_v(t)$', fontsize=12)
    plt.title('Velocity Autocorrelation Analysis', fontsize=14)
    plt.grid(True)
    plt.legend(loc='best')
    
    # Add annotation with correlation times
    annotation_text = []
    for result in all_results:
        if np.isnan(result['correlation_time']):
            annotation_text.append(f"{result['filename']}: No decay")
        else:
            annotation_text.append(f"{result['filename']}: τ = {result['correlation_time']:.3f} s")
    
    # Add text box with correlation times
    plt.text(
        0.05, 0.95, 
        '\n'.join(['Correlation times:'] + annotation_text),
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_autocorrelation_comparison.png'), dpi=300)
    
    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'Filename': r['filename'],
            'NumTrajectories': r['num_trajectories'],
            'CorrelationTime': r['correlation_time']
        } for r in all_results
    ])
    
    results_df.to_csv(os.path.join(output_dir, 'velocity_autocorrelation_results.csv'), index=False)
    
    # Save individual velocity correlation data
    for result in all_results:
        df = pd.DataFrame({
            'TimeLag': result['time_lags'],
            'Cv': result['Cv'],
            'Cv_SEM': result['Cv_sem']
        })
        df.to_csv(os.path.join(output_dir, f"{result['filename']}_velocity_correlation.csv"), index=False)
    
    print(f"\nAnalysis Summary:")
    for result in all_results:
        if np.isnan(result['correlation_time']):
            print(f"{result['filename']}: No correlation decay")
        else:
            print(f"{result['filename']}: Correlation time = {result['correlation_time']:.3f} s")
    
    print(f"\nResults saved in {output_dir}")

if __name__ == "__main__":
    main()