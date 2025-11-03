# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 15:26:28 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
velocity_autocorrelation_single.py

Simple script to calculate and plot velocity autocorrelation for a single dataset.
Provides diagnostic plots to understand what's being measured.

Input:
- Single analyzed trajectory .pkl file

Output:
- Velocity autocorrelation plot with diagnostic information
- CSV file with correlation data
- Diagnostic plots showing velocity distributions and sample trajectories

Usage:
python velocity_autocorrelation_single.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Global parameters (modify these as needed)
# =====================================
# Time step in seconds
DT = 0.1
# Minimum trajectory length
MIN_TRACK_LENGTH = 15
# Maximum time lag to analyze
MAX_TAU = 50
# Number of sample trajectories to show in diagnostics
N_DIAGNOSTIC_TRAJECTORIES = 5
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

def calculate_velocity_autocorrelation(trajectories, dt=DT, max_tau=MAX_TAU, min_length=MIN_TRACK_LENGTH):
    """
    Calculate velocity autocorrelation function for trajectories.
    
    Args:
        trajectories: List of trajectory dictionaries
        dt: Time interval between frames
        max_tau: Maximum lag to calculate
        min_length: Minimum trajectory length to consider
        
    Returns:
        Dictionary with velocity autocorrelation results
    """
    # Filter trajectories by length
    filtered_trajectories = [traj for traj in trajectories if len(traj['x']) > min_length]
    
    if not filtered_trajectories:
        print(f"No trajectories longer than {min_length} frames found")
        return None
    
    print(f"Using {len(filtered_trajectories)} trajectories (filtered from {len(trajectories)})")
    
    # Calculate velocities for each trajectory
    velocity_data = []
    
    for traj in filtered_trajectories:
        x = np.array(traj['x'])
        y = np.array(traj['y'])
        
        # Calculate velocities using simple difference
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        
        # Store velocity data
        velocity_data.append({
            'vx': vx,
            'vy': vy,
            'v_mag': np.sqrt(vx**2 + vy**2),
            'traj_id': traj['id'],
            'length': len(vx)
        })
    
    # Calculate velocity autocorrelation
    Cv = np.zeros(max_tau)
    Cv_sem = np.zeros(max_tau)
    n_points = np.zeros(max_tau)
    
    for tau in range(1, max_tau + 1):
        corr_values = []
        
        for vel_data in velocity_data:
            vx = vel_data['vx']
            vy = vel_data['vy']
            
            if len(vx) > tau:
                # Calculate normalized velocity correlation
                vx_0 = vx[:-tau]  # velocity at time t
                vy_0 = vy[:-tau]
                vx_tau = vx[tau:]  # velocity at time t+tau
                vy_tau = vy[tau:]
                
                # Calculate velocity magnitudes
                v_mag_0 = np.sqrt(vx_0**2 + vy_0**2)
                v_mag_tau = np.sqrt(vx_tau**2 + vy_tau**2)
                
                # Only use points where both velocities are non-zero
                valid_indices = (v_mag_0 > 0) & (v_mag_tau > 0)
                
                if np.any(valid_indices):
                    # Calculate dot product of velocity vectors (normalized)
                    dot_product = (vx_0[valid_indices] * vx_tau[valid_indices] + 
                                 vy_0[valid_indices] * vy_tau[valid_indices])
                    
                    # Normalize by velocity magnitudes
                    corr = dot_product / (v_mag_0[valid_indices] * v_mag_tau[valid_indices])
                    corr_values.extend(corr)
        
        # Calculate statistics for this time lag
        if corr_values:
            corr_array = np.array(corr_values)
            corr_array = corr_array[~np.isnan(corr_array)]  # Remove NaN values
            
            if len(corr_array) > 0:
                Cv[tau-1] = np.mean(corr_array)
                Cv_sem[tau-1] = np.std(corr_array) / np.sqrt(len(corr_array))
                n_points[tau-1] = len(corr_array)
            else:
                Cv[tau-1] = np.nan
                Cv_sem[tau-1] = np.nan
                n_points[tau-1] = 0
        else:
            Cv[tau-1] = np.nan
            Cv_sem[tau-1] = np.nan
            n_points[tau-1] = 0
    
    # Calculate correlation time (time to decay to 1/e)
    correlation_time = np.nan
    if not np.isnan(Cv[0]) and Cv[0] > 0:
        threshold = Cv[0] / np.e
        indices = np.where(Cv < threshold)[0]
        if len(indices) > 0:
            correlation_time = dt * (indices[0] + 1)
    
    # Calculate zero-crossing time
    zero_crossing_time = np.nan
    indices = np.where(Cv < 0)[0]
    if len(indices) > 0:
        zero_crossing_time = dt * (indices[0] + 1)
    
    return {
        'Cv': Cv,
        'Cv_sem': Cv_sem,
        'n_points': n_points,
        'correlation_time': correlation_time,
        'zero_crossing_time': zero_crossing_time,
        'time_lags': np.arange(1, max_tau + 1) * dt,
        'num_trajectories': len(filtered_trajectories),
        'velocity_data': velocity_data
    }

def create_diagnostic_plots(results, output_path, base_name):
    """
    Create diagnostic plots to show what's being measured.
    
    Args:
        results: Dictionary with velocity autocorrelation results
        output_path: Directory to save plots
        base_name: Base name for output files
    """
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Velocity autocorrelation function
    ax1 = axes[0, 0]
    ax1.errorbar(results['time_lags'], results['Cv'], 
                yerr=results['Cv_sem'], fmt='o-', capsize=3)
    
    # Add horizontal line at 1/e
    if not np.isnan(results['correlation_time']):
        ax1.axhline(results['Cv'][0] / np.e, color='red', linestyle='--', 
                   label=f'1/e threshold')
        ax1.axvline(results['correlation_time'], color='red', linestyle='--', 
                   label=f'τ_corr = {results["correlation_time"]:.3f} s')
    
    # Add horizontal line at zero
    ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
    if not np.isnan(results['zero_crossing_time']):
        ax1.axvline(results['zero_crossing_time'], color='orange', linestyle='--', 
                   label=f'Zero crossing = {results["zero_crossing_time"]:.3f} s')
    
    ax1.set_xlabel('Time lag (s)')
    ax1.set_ylabel('Velocity Autocorrelation C_v(τ)')
    ax1.set_title('Velocity Autocorrelation Function')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Number of data points vs time lag
    ax2 = axes[0, 1]
    ax2.plot(results['time_lags'], results['n_points'], 'o-')
    ax2.set_xlabel('Time lag (s)')
    ax2.set_ylabel('Number of data points')
    ax2.set_title('Data points available at each time lag')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sample velocity trajectories
    ax3 = axes[1, 0]
    n_show = min(N_DIAGNOSTIC_TRAJECTORIES, len(results['velocity_data']))
    colors = plt.cm.viridis(np.linspace(0, 1, n_show))
    
    for i in range(n_show):
        vel_data = results['velocity_data'][i]
        time_points = np.arange(len(vel_data['v_mag'])) * DT
        ax3.plot(time_points, vel_data['v_mag'], color=colors[i], alpha=0.7,
                label=f"Traj {vel_data['traj_id']}")
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity magnitude (μm/s)')
    ax3.set_title(f'Sample velocity trajectories (showing {n_show})')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Velocity magnitude distribution
    ax4 = axes[1, 1]
    all_velocities = []
    for vel_data in results['velocity_data']:
        all_velocities.extend(vel_data['v_mag'])
    
    ax4.hist(all_velocities, bins=50, alpha=0.7, density=True)
    ax4.axvline(np.mean(all_velocities), color='red', linestyle='--', 
               label=f'Mean = {np.mean(all_velocities):.3f} μm/s')
    ax4.axvline(np.median(all_velocities), color='orange', linestyle='--', 
               label=f'Median = {np.median(all_velocities):.3f} μm/s')
    
    ax4.set_xlabel('Velocity magnitude (μm/s)')
    ax4.set_ylabel('Probability density')
    ax4.set_title('Distribution of velocity magnitudes')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{base_name}_velocity_autocorr_diagnostic.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_main_plot(results, output_path, base_name):
    """
    Create the main velocity autocorrelation plot.
    
    Args:
        results: Dictionary with velocity autocorrelation results
        output_path: Directory to save plots
        base_name: Base name for output files
    """
    plt.figure(figsize=(10, 8))
    
    # Main autocorrelation plot
    plt.errorbar(results['time_lags'], results['Cv'], 
                yerr=results['Cv_sem'], fmt='o-', capsize=3, 
                markersize=4, linewidth=1.5)
    
    # Add reference lines
    if not np.isnan(results['correlation_time']):
        plt.axhline(results['Cv'][0] / np.e, color='red', linestyle='--', alpha=0.7,
                   label=f'1/e threshold')
        plt.axvline(results['correlation_time'], color='red', linestyle='--', alpha=0.7,
                   label=f'τ_corr = {results["correlation_time"]:.3f} s')
    
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    if not np.isnan(results['zero_crossing_time']):
        plt.axvline(results['zero_crossing_time'], color='orange', linestyle='--', alpha=0.7,
                   label=f'Zero crossing = {results["zero_crossing_time"]:.3f} s')
    
    # Add sampling frequency reference
    plt.axvline(DT, color='gray', linestyle=':', alpha=0.7,
               label=f'Sampling period = {DT} s')
    
    plt.xlabel('Time lag τ (s)', fontsize=12)
    plt.ylabel('Velocity Autocorrelation C_v(τ)', fontsize=12)
    plt.title(f'Velocity Autocorrelation Function\n({results["num_trajectories"]} trajectories)', 
             fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text box with key statistics
    stats_text = []
    if not np.isnan(results['correlation_time']):
        stats_text.append(f"Correlation time: {results['correlation_time']:.3f} s")
        stats_text.append(f"Ratio to sampling: {results['correlation_time']/DT:.1f}×")
    else:
        stats_text.append("No exponential decay found")
    
    if not np.isnan(results['zero_crossing_time']):
        stats_text.append(f"Zero crossing: {results['zero_crossing_time']:.3f} s")
    else:
        stats_text.append("No zero crossing found")
    
    plt.text(0.02, 0.98, '\n'.join(stats_text),
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{base_name}_velocity_autocorr.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def export_results(results, output_path, base_name):
    """
    Export results to CSV file.
    
    Args:
        results: Dictionary with velocity autocorrelation results
        output_path: Directory to save files
        base_name: Base name for output files
    """
    # Main autocorrelation data
    df = pd.DataFrame({
        'TimeLag_s': results['time_lags'],
        'Cv': results['Cv'],
        'Cv_SEM': results['Cv_sem'],
        'N_points': results['n_points']
    })
    
    df.to_csv(os.path.join(output_path, f'{base_name}_velocity_autocorr_data.csv'), 
              index=False)
    
    # Summary statistics
    summary = {
        'Parameter': ['Number of trajectories', 'Correlation time (s)', 
                     'Zero crossing time (s)', 'Sampling period (s)',
                     'Correlation time / Sampling period'],
        'Value': [results['num_trajectories'], 
                 results['correlation_time'],
                 results['zero_crossing_time'],
                 DT,
                 results['correlation_time']/DT if not np.isnan(results['correlation_time']) else np.nan]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_path, f'{base_name}_velocity_autocorr_summary.csv'), 
                      index=False)

def main():
    """Main function to analyze velocity autocorrelation for a single file."""
    print("Single File Velocity Autocorrelation Analysis")
    print("=" * 50)
    
    # Get input file
    input_file = input("Enter the path to analyzed trajectory file (.pkl): ")
    
    if not os.path.isfile(input_file):
        print(f"File {input_file} does not exist")
        return
    
    # Load data
    print("Loading trajectory data...")
    analyzed_data = load_analyzed_data(input_file)
    
    if analyzed_data is None:
        print("Failed to load trajectory data")
        return
    
    if 'trajectories' not in analyzed_data:
        print("No trajectories found in the data")
        return
    
    print(f"Found {len(analyzed_data['trajectories'])} trajectories")
    
    # Calculate velocity autocorrelation
    print("Calculating velocity autocorrelation...")
    results = calculate_velocity_autocorrelation(analyzed_data['trajectories'])
    
    if results is None:
        print("Failed to calculate velocity autocorrelation")
        return
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.join(os.path.dirname(input_file), f"velocity_autocorr_{base_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots
    print("Creating diagnostic plots...")
    create_diagnostic_plots(results, output_dir, base_name)
    
    print("Creating main plot...")
    create_main_plot(results, output_dir, base_name)
    
    # Export results
    print("Exporting results...")
    export_results(results, output_dir, base_name)
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Number of trajectories used: {results['num_trajectories']}")
    
    if not np.isnan(results['correlation_time']):
        print(f"Velocity correlation time: {results['correlation_time']:.3f} s")
        print(f"Ratio to sampling period: {results['correlation_time']/DT:.1f}×")
        
        if results['correlation_time'] < 2*DT:
            print("WARNING: Correlation time is close to sampling limit!")
        elif results['correlation_time'] < 5*DT:
            print("CAUTION: Correlation time may be affected by sampling rate")
    else:
        print("No exponential decay found in correlation function")
    
    if not np.isnan(results['zero_crossing_time']):
        print(f"Zero crossing time: {results['zero_crossing_time']:.3f} s")
    else:
        print("No zero crossing found")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()