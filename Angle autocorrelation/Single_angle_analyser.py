# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 15:05:17 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
single_angle_analyzer.py

Simple script to analyze angle autocorrelation for a single trajectory file.
This provides a foundation that can be extended to multiple files.

Input:
- Single analyzed trajectory file (.pkl)

Output:
- Angle autocorrelation plot
- Crossing time calculation
- Diagnostic plots showing what's being measured
- CSV export of results

Usage:
python single_angle_analyzer.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Global parameters (modify these as needed)
# =====================================
# Time interval between frames in seconds
DT = 0.1
# Maximum lag for angle correlation analysis
ANGLE_PLOT_CUTOFF = 50
# Minimum trajectory length to consider
L_CUTOFF = 15
# Number of example trajectories to show in diagnostics
N_EXAMPLES = 3
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
    
    This measures how the direction of motion persists over time.
    A value of 1 means perfect correlation (same direction),
    0 means no correlation, and -1 means anti-correlation (opposite direction).
    
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
    
    print(f"Using {len(filtered_trajectories)} trajectories (out of {len(trajectories)} total)")
    
    # Initialize results - store all correlation values for each time lag
    cos_angle_t_total = [[] for _ in range(angle_plot_cutoff)]
    
    # Calculate angle correlations for each trajectory
    for traj in filtered_trajectories:
        x_temp = traj['x']
        y_temp = traj['y']
        
        # Calculate displacement vectors between consecutive points
        delta_x = np.diff(x_temp)
        delta_y = np.diff(y_temp)
        
        # Calculate angle correlations for different time lags
        for i in range(1, angle_plot_cutoff + 1):
            if len(delta_x) <= i:
                continue
                
            cos_angle_t_temp_j = []
            for k in range(len(delta_x) - i):
                # Get two displacement vectors separated by time lag i
                # Vector 1: displacement at step k
                # Vector 2: displacement at step k+i
                
                # Calculate dot product of displacement vectors
                dot_product = delta_x[k] * delta_x[k+i] + delta_y[k] * delta_y[k+i]
                
                # Calculate magnitudes of both vectors
                mag1 = np.sqrt(delta_x[k]**2 + delta_y[k]**2)
                mag2 = np.sqrt(delta_x[k+i]**2 + delta_y[k+i]**2)
                
                # Calculate cosine of angle between vectors
                if mag1 > 0 and mag2 > 0:
                    cos_angle = dot_product / (mag1 * mag2)
                    # Ensure value is in valid range for cosine [-1, 1]
                    cos_angle = max(min(cos_angle, 1.0), -1.0)
                    cos_angle_t_temp_j.append(cos_angle)
            
            # Add all correlation values for this time lag
            cos_angle_t_total[i-1].extend(cos_angle_t_temp_j)
    
    # Calculate average cos(theta(tau)) and standard error for each time lag
    mean_cos_angle = np.zeros(angle_plot_cutoff)
    sem_cos_angle = np.zeros(angle_plot_cutoff)
    n_values = np.zeros(angle_plot_cutoff, dtype=int)
    
    for i in range(angle_plot_cutoff):
        temp = np.array(cos_angle_t_total[i])
        temp = temp[~np.isnan(temp)]  # Remove any NaN values
        
        if len(temp) > 0:
            mean_cos_angle[i] = np.mean(temp)
            sem_cos_angle[i] = np.std(temp) / np.sqrt(len(temp))
            n_values[i] = len(temp)
        else:
            mean_cos_angle[i] = np.nan
            sem_cos_angle[i] = np.nan
            n_values[i] = 0
    
    # Calculate crossing time (time when correlation first becomes negative)
    t_cross = calculate_crossing_time(mean_cos_angle, dt)
    
    return {
        'mean_cos_angle': mean_cos_angle,
        'sem_cos_angle': sem_cos_angle,
        't_cross': t_cross,
        'time_lags': np.arange(1, angle_plot_cutoff + 1) * dt,
        'num_trajectories': len(filtered_trajectories),
        'n_values_per_lag': n_values,
        'raw_correlations': cos_angle_t_total
    }

def calculate_crossing_time(mean_cos_angle, dt):
    """
    Calculate the time when angle correlation first crosses zero.
    
    Args:
        mean_cos_angle: Array of mean cosine values
        dt: Time step
        
    Returns:
        Crossing time or NaN if no crossing found
    """
    t_cross = np.nan
    
    # Find first index where correlation becomes negative
    negative_indices = np.where(mean_cos_angle < 0)[0]
    
    if len(negative_indices) > 0:
        cross_index = negative_indices[0]
        
        if cross_index >= 1:
            # Linear interpolation between the last positive and first negative point
            x1 = cross_index - 1
            x2 = cross_index
            y1 = mean_cos_angle[x1]
            y2 = mean_cos_angle[x2]
            
            if (y2 - y1) != 0:  # Avoid division by zero
                # Find where line crosses zero
                t_cross = (x1 + 1 + y1 / (y1 - y2)) * dt
        else:
            # Crossing happens very early, use simple estimate
            if len(mean_cos_angle) > 1:
                y1 = mean_cos_angle[0]
                y2 = mean_cos_angle[1]
                if (y2 - y1) != 0:
                    t_cross = (1 + y1 / (y1 - y2)) * dt
    
    return t_cross

def create_diagnostic_plots(trajectories, angle_results, output_path, filename_base):
    """
    Create diagnostic plots to visualize what's being measured.
    
    Args:
        trajectories: List of trajectory dictionaries
        angle_results: Results from angle autocorrelation analysis
        output_path: Directory to save plots
        filename_base: Base name for output files
    """
    # Filter trajectories by length
    filtered_trajectories = [traj for traj in trajectories if len(traj['x']) > L_CUTOFF]
    
    if not filtered_trajectories:
        print("No trajectories available for diagnostic plots")
        return
    
    # Select example trajectories
    n_examples = min(N_EXAMPLES, len(filtered_trajectories))
    example_trajectories = filtered_trajectories[:n_examples]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Example trajectories
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, n_examples))
    
    for i, traj in enumerate(example_trajectories):
        x, y = traj['x'], traj['y']
        ax.plot(x, y, '-', color=colors[i], alpha=0.7, linewidth=1.5, 
                label=f'Trajectory {i+1} ({len(x)} points)')
        ax.plot(x[0], y[0], 'o', color=colors[i], markersize=8)  # Start point
        ax.plot(x[-1], y[-1], 's', color=colors[i], markersize=8)  # End point
    
    ax.set_xlabel('X position (μm)')
    ax.set_ylabel('Y position (μm)')
    ax.set_title('Example Trajectories\n(Circle=start, Square=end)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 2: Displacement vectors for one trajectory
    ax = axes[0, 1]
    if example_trajectories:
        traj = example_trajectories[0]
        x, y = traj['x'], traj['y']
        dx = np.diff(x)
        dy = np.diff(y)
        
        # Show every few displacement vectors to avoid crowding
        step = max(1, len(dx) // 15)  # Show about 15 arrows
        
        for i in range(0, len(dx), step):
            # Scale arrows for visibility
            scale = 0.8
            ax.arrow(x[i], y[i], dx[i]*scale, dy[i]*scale, 
                    head_width=0.05, head_length=0.03, 
                    fc='red', ec='red', alpha=0.7)
        
        ax.plot(x, y, 'k-', alpha=0.3, linewidth=1)
        ax.set_xlabel('X position (μm)')
        ax.set_ylabel('Y position (μm)')
        ax.set_title('Displacement Vectors\n(Every few steps shown)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Plot 3: Angle changes in one trajectory
    ax = axes[1, 0]
    if example_trajectories:
        traj = example_trajectories[0]
        x, y = traj['x'], traj['y']
        dx = np.diff(x)
        dy = np.diff(y)
        
        # Calculate turning angles (change in direction between consecutive steps)
        turning_angles = []
        for i in range(len(dx) - 1):
            v1 = np.array([dx[i], dy[i]])
            v2 = np.array([dx[i+1], dy[i+1]])
            
            # Calculate angle between consecutive displacement vectors
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = max(min(cos_angle, 1.0), -1.0)
                angle = np.arccos(cos_angle) * 180 / np.pi  # Convert to degrees
                turning_angles.append(angle)
        
        if turning_angles:
            ax.plot(turning_angles, 'o-', alpha=0.7, markersize=4)
            ax.axhline(90, color='red', linestyle='--', alpha=0.5, label='90° (random)')
            ax.set_xlabel('Step number')
            ax.set_ylabel('Turning angle (degrees)')
            ax.set_title('Turning Angles Between Consecutive Steps')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 180)
    
    # Plot 4: The actual angle autocorrelation function
    ax = axes[1, 1]
    
    # Plot the correlation function with error bars
    ax.errorbar(angle_results['time_lags'], angle_results['mean_cos_angle'],
               yerr=angle_results['sem_cos_angle'], 
               fmt='bo-', alpha=0.7, capsize=3, markersize=4)
    
    # Add horizontal reference lines
    ax.axhline(0, color='red', linestyle='--', alpha=0.7, label='Zero correlation')
    ax.axhline(1, color='green', linestyle=':', alpha=0.5, label='Perfect correlation')
    
    # Mark crossing time if available
    if not np.isnan(angle_results['t_cross']):
        ax.axvline(angle_results['t_cross'], color='red', linestyle='-', alpha=0.7)
        ax.text(angle_results['t_cross'], 0.1, 
                f'Crossing time:\n{angle_results["t_cross"]:.2f} s', 
                rotation=0, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Time lag (s)')
    ax.set_ylabel(r'$\langle \cos \theta(\tau) \rangle$')
    ax.set_title(f'Angle Autocorrelation\n({angle_results["num_trajectories"]} trajectories)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{filename_base}_diagnostics.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_main_plot(angle_results, output_path, filename_base):
    """
    Create the main angle autocorrelation plot.
    
    Args:
        angle_results: Results from angle autocorrelation analysis
        output_path: Directory to save plots
        filename_base: Base name for output files
    """
    plt.figure(figsize=(10, 8))
    
    # Plot the correlation function with error bars
    plt.errorbar(angle_results['time_lags'], angle_results['mean_cos_angle'],
                yerr=angle_results['sem_cos_angle'], 
                fmt='bo-', alpha=0.8, capsize=5, markersize=6, linewidth=2,
                label=f'{angle_results["num_trajectories"]} trajectories')
    
    # Add reference lines
    plt.axhline(0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    plt.axhline(1, color='green', linestyle=':', alpha=0.5, linewidth=1)
    
    # Mark crossing time if available
    if not np.isnan(angle_results['t_cross']):
        plt.axvline(angle_results['t_cross'], color='red', linestyle='-', alpha=0.7, linewidth=2)
        plt.text(angle_results['t_cross'], 0.5, 
                f'Crossing time = {angle_results["t_cross"]:.3f} s', 
                rotation=90, ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    plt.xlabel('Time lag τ (s)', fontsize=14)
    plt.ylabel(r'$\langle \cos \theta(\tau) \rangle$', fontsize=14)
    plt.title('Angle Autocorrelation Function', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.3, 1.1)
    
    # Add explanatory text
    explanation = ("Measures directional persistence:\n"
                  "1.0 = perfect persistence\n"
                  "0.0 = no directional memory\n"
                  "-1.0 = reversal tendency")
    plt.text(0.02, 0.98, explanation, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{filename_base}_angle_autocorrelation.png"), 
               dpi=300, bbox_inches='tight')
    plt.close()

def export_results(angle_results, output_path, filename_base):
    """
    Export results to CSV file.
    
    Args:
        angle_results: Results from angle autocorrelation analysis
        output_path: Directory to save CSV
        filename_base: Base name for output files
    """
    # Create detailed results DataFrame
    df = pd.DataFrame({
        'Time_lag_s': angle_results['time_lags'],
        'Mean_cos_angle': angle_results['mean_cos_angle'],
        'SEM_cos_angle': angle_results['sem_cos_angle'],
        'N_values': angle_results['n_values_per_lag']
    })
    
    df.to_csv(os.path.join(output_path, f"{filename_base}_angle_autocorrelation.csv"), index=False)
    
    # Create summary file
    summary = {
        'Filename': [filename_base],
        'Number_of_trajectories': [angle_results['num_trajectories']],
        'Crossing_time_s': [angle_results['t_cross']],
        'Initial_correlation': [angle_results['mean_cos_angle'][0] if len(angle_results['mean_cos_angle']) > 0 else np.nan],
        'Final_correlation': [angle_results['mean_cos_angle'][-1] if len(angle_results['mean_cos_angle']) > 0 else np.nan]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_path, f"{filename_base}_summary.csv"), index=False)
    
    print(f"Results exported to CSV files with prefix '{filename_base}'")

def main():
    """Main function to analyze angle autocorrelation for a single file."""
    print("Single File Angle Autocorrelation Analyzer")
    print("==========================================")
    
    # Get input file
    input_file = input("Enter path to analyzed trajectory file (.pkl): ")
    
    if not os.path.isfile(input_file):
        print(f"File {input_file} does not exist")
        return
    
    # Load data
    print(f"Loading data from {input_file}...")
    analyzed_data = load_analyzed_data(input_file)
    
    if analyzed_data is None:
        print("Failed to load data")
        return
    
    trajectories = analyzed_data.get('trajectories', [])
    if not trajectories:
        print("No trajectories found in the data")
        return
    
    print(f"Found {len(trajectories)} trajectories")
    
    # Calculate angle autocorrelation
    print("Calculating angle autocorrelation...")
    angle_results = calculate_angle_autocorrelation(trajectories)
    
    if angle_results is None:
        print("Failed to calculate angle autocorrelation")
        return
    
    # Create output directory
    input_dir = os.path.dirname(input_file)
    filename_base = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.join(input_dir, f"angle_analysis_{filename_base}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    print("Creating plots...")
    create_main_plot(angle_results, output_dir, filename_base)
    create_diagnostic_plots(trajectories, angle_results, output_dir, filename_base)
    
    # Export results
    print("Exporting results...")
    export_results(angle_results, output_dir, filename_base)
    
    # Save results as pickle for potential reuse
    output_file = os.path.join(output_dir, f"{filename_base}_angle_results.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(angle_results, f)
    
    # Print summary
    print("\nAnalysis Summary:")
    print("================")
    print(f"File: {input_file}")
    print(f"Total trajectories: {len(trajectories)}")
    print(f"Used trajectories: {angle_results['num_trajectories']} (length > {L_CUTOFF})")
    
    if not np.isnan(angle_results['t_cross']):
        print(f"Crossing time: {angle_results['t_cross']:.3f} seconds")
        print(f"This indicates the time scale over which directional memory is lost")
    else:
        print("No crossing time found (correlation never becomes negative)")
        print("This suggests persistent directional motion")
    
    print(f"\nInitial correlation: {angle_results['mean_cos_angle'][0]:.3f}")
    print(f"Final correlation: {angle_results['mean_cos_angle'][-1]:.3f}")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()