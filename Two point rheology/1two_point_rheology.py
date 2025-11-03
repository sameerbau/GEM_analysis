# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 16:33:24 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
two_point_rheology.py

This script extends the single-particle trajectory analysis to implement
two-point microrheology (TPM). It calculates the correlated motion of pairs
of particles to extract bulk viscoelastic properties of the medium.

Input:
- Processed trajectory data (.pkl files) from trajectory_processor.py
- Multiple particles must be present in the same field of view

Output:
- Cross-correlation functions between particle pairs
- Longitudinal and transverse correlation components
- Viscoelastic modulus estimations based on generalized Stokes-Einstein relation
- Diagnostic plots of correlation functions and rheological properties

Usage:
python two_point_rheology.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import glob
import pickle
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import UnivariateSpline
import warnings
import seaborn as sns
from matplotlib.colors import LogNorm

# Global parameters that can be modified
# =====================================
# Time step in seconds 
DT = 0.1
# Pixel to micron conversion
CONVERSION = 0.094
# Temperature in Kelvin (for Stokes-Einstein relation)
TEMPERATURE = 298.15
# Particle radius in micrometers
PARTICLE_RADIUS = 0.4
# Minimum number of correlation pairs to compute statistics
MIN_CORR_PAIRS = 5
# Number of distance bins for correlation analysis
N_DISTANCE_BINS = 10
# Minimum and maximum separation distance (μm)
MIN_SEPARATION = 2.0  # Should be larger than particle size
MAX_SEPARATION = 10
# Minimum time lag for analysis (frames)
MIN_TIME_LAG = 1
# Maximum time lag for analysis (frames)
MAX_TIME_LAG = 20
# =====================================

def load_processed_data(file_path):
    """
    Load processed trajectory data from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary containing the processed trajectory data
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading processed data from {file_path}: {e}")
        return None

def prepare_trajectories_for_tpm(processed_data):
    """
    Prepare trajectory data for two-point microrheology analysis.
    
    Args:
        processed_data: Dictionary with processed trajectory data
        
    Returns:
        Dictionary with reorganized trajectory data suitable for TPM
    """
    # Extract relevant data
    trajectories = processed_data.get('trajectories', [])
    
    if not trajectories:
        print("No trajectories found in processed data")
        return None
    
    # Determine the frames covered by the trajectories
    all_frames = []
    
    for traj in trajectories:
        # For each trajectory, get the frames where it exists
        if 'time' in traj:
            # Convert time to frame numbers
            frames = np.round(traj['time'] / DT).astype(int)
            all_frames.extend(frames)
    
    if not all_frames:
        print("No frame information found in trajectories")
        return None
    
    # Determine unique frame numbers and their range
    unique_frames = np.unique(all_frames)
    min_frame = np.min(unique_frames)
    max_frame = np.max(unique_frames)
    
    # Create a dictionary to hold particle positions for each frame
    frame_data = {frame: {'ids': [], 'positions': []} for frame in range(min_frame, max_frame + 1)}
    
    # Populate frame data with trajectories
    for traj in trajectories:
        traj_id = traj['id']
        
        # Get time and convert to frame numbers
        if 'time' in traj:
            frames = np.round(traj['time'] / DT).astype(int)
            
            # For each frame in the trajectory
            for i, frame in enumerate(frames):
                if frame in frame_data:
                    frame_data[frame]['ids'].append(traj_id)
                    frame_data[frame]['positions'].append([traj['x'][i], traj['y'][i]])
    
    # Convert lists to numpy arrays
    for frame in frame_data:
        frame_data[frame]['ids'] = np.array(frame_data[frame]['ids'])
        frame_data[frame]['positions'] = np.array(frame_data[frame]['positions'])
    
    return {
        'frame_data': frame_data,
        'min_frame': min_frame,
        'max_frame': max_frame
    }

def calculate_separation_distances(positions):
    """
    Calculate all pairwise separation distances between particles.
    
    Args:
        positions: Array of particle positions [x, y]
        
    Returns:
        Array of pairwise distances and their vector components
    """
    n_particles = len(positions)
    
    if n_particles < 2:
        return None, None, None
    
    # Calculate all pairwise distances
    distances = squareform(pdist(positions))
    
    # Calculate distance vector components
    dx_matrix = np.zeros((n_particles, n_particles))
    dy_matrix = np.zeros((n_particles, n_particles))
    
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            
            dx_matrix[i, j] = dx
            dx_matrix[j, i] = -dx
            
            dy_matrix[i, j] = dy
            dy_matrix[j, i] = -dy
    
    return distances, dx_matrix, dy_matrix

def track_particle_pairs(tpm_data, time_lag):
    """
    Track pairs of particles over the specified time lag.
    
    Args:
        tpm_data: Dictionary with prepared TPM data
        time_lag: Time lag in frames
        
    Returns:
        Dictionary with particle pair data
    """
    frame_data = tpm_data['frame_data']
    min_frame = tpm_data['min_frame']
    max_frame = tpm_data['max_frame']
    
    # Initialize lists to store pair data
    pair_data = {
        'separation': [],
        'r_parallel': [],
        'r_perpendicular': [],
        'displacement_i_x': [],
        'displacement_i_y': [],
        'displacement_j_x': [],
        'displacement_j_y': [],
        'displacement_dot_product': [],
        'displacement_cross_correlation': []
    }
    
    # Loop through frames
    for frame in range(min_frame, max_frame - time_lag + 1):
        if frame not in frame_data or (frame + time_lag) not in frame_data:
            continue
            
        current_frame = frame_data[frame]
        next_frame = frame_data[frame + time_lag]
        
        # Check if we have particles in both frames
        if len(current_frame['ids']) < 1 or len(next_frame['ids']) < 1:
            continue
            
        # Find particles that appear in both frames
        common_ids = np.intersect1d(current_frame['ids'], next_frame['ids'])
        
        if len(common_ids) < 2:
            continue
            
        # Get positions and indices for the common particles
        current_indices = [np.where(current_frame['ids'] == id_)[0][0] for id_ in common_ids]
        next_indices = [np.where(next_frame['ids'] == id_)[0][0] for id_ in common_ids]
        
        current_positions = current_frame['positions'][current_indices]
        next_positions = next_frame['positions'][next_indices]
        
        # Calculate separations between all pairs at the current frame
        distances, dx_matrix, dy_matrix = calculate_separation_distances(current_positions)
        
        if distances is None:
            continue
            
        # Calculate displacements for each particle
        displacements = next_positions - current_positions
        
        # Process all pairs
        n_particles = len(common_ids)
        
        for i in range(n_particles):
            for j in range(i+1, n_particles):
                # Get separation distance
                separation = distances[i, j]
                
                # Skip if separation is outside our range of interest
                if separation < MIN_SEPARATION or separation > MAX_SEPARATION:
                    continue
                
                # Get displacement vectors
                displacement_i = displacements[i]
                displacement_j = displacements[j]
                
                # Calculate direction vector (unit vector from i to j)
                dx = dx_matrix[i, j]
                dy = dy_matrix[i, j]
                direction = np.array([dx, dy]) / separation
                
                # Project displacements onto parallel and perpendicular directions
                displacement_i_parallel = np.dot(displacement_i, direction) * direction
                displacement_i_perp = displacement_i - displacement_i_parallel
                
                displacement_j_parallel = np.dot(displacement_j, direction) * direction
                displacement_j_perp = displacement_j - displacement_j_parallel
                
                # Calculate dot product and cross correlation of displacements
                dot_product = np.dot(displacement_i, displacement_j)
                
                # Store pair data
                pair_data['separation'].append(separation)
                pair_data['r_parallel'].append(np.dot(direction, displacement_i) * np.dot(direction, displacement_j))
                pair_data['r_perpendicular'].append(np.dot(displacement_i_perp, displacement_j_perp))
                pair_data['displacement_i_x'].append(displacement_i[0])
                pair_data['displacement_i_y'].append(displacement_i[1])
                pair_data['displacement_j_x'].append(displacement_j[0])
                pair_data['displacement_j_y'].append(displacement_j[1])
                pair_data['displacement_dot_product'].append(dot_product)
                pair_data['displacement_cross_correlation'].append(dot_product)
    
    # Convert to numpy arrays
    for key in pair_data:
        pair_data[key] = np.array(pair_data[key])
    
    return pair_data

def bin_correlation_by_distance(pair_data):
    """
    Bin correlation data based on separation distance.
    
    Args:
        pair_data: Dictionary with particle pair data
        
    Returns:
        Dictionary with binned correlation data
    """
    if len(pair_data['separation']) < MIN_CORR_PAIRS:
        return None
    
    # Create distance bins
    distance_bins = np.linspace(MIN_SEPARATION, MAX_SEPARATION, N_DISTANCE_BINS + 1)
    bin_centers = 0.5 * (distance_bins[:-1] + distance_bins[1:])
    
    # Initialize arrays to store binned data
    Dr = np.zeros(N_DISTANCE_BINS)
    Dr_err = np.zeros(N_DISTANCE_BINS)
    Dt = np.zeros(N_DISTANCE_BINS)
    Dt_err = np.zeros(N_DISTANCE_BINS)
    bin_counts = np.zeros(N_DISTANCE_BINS, dtype=int)
    
    # Bin the data
    for i in range(N_DISTANCE_BINS):
        bin_min, bin_max = distance_bins[i], distance_bins[i+1]
        
        # Get indices of pairs in this distance bin
        bin_indices = np.where((pair_data['separation'] >= bin_min) & 
                              (pair_data['separation'] < bin_max))[0]
        
        bin_counts[i] = len(bin_indices)
        
        if bin_counts[i] >= MIN_CORR_PAIRS:
            # Calculate mean and standard error for parallel (Dr) and perpendicular (Dt) components
            Dr[i] = np.mean(pair_data['r_parallel'][bin_indices])
            Dr_err[i] = np.std(pair_data['r_parallel'][bin_indices]) / np.sqrt(bin_counts[i])
            
            Dt[i] = np.mean(pair_data['r_perpendicular'][bin_indices])
            Dt_err[i] = np.std(pair_data['r_perpendicular'][bin_indices]) / np.sqrt(bin_counts[i])
        else:
            Dr[i] = np.nan
            Dr_err[i] = np.nan
            Dt[i] = np.nan
            Dt_err[i] = np.nan
    
    # Return binned data
    binned_data = {
        'bin_centers': bin_centers,
        'bin_counts': bin_counts,
        'Dr': Dr,
        'Dr_err': Dr_err,
        'Dt': Dt,
        'Dt_err': Dt_err
    }
    
    return binned_data

def calculate_viscoelastic_modulus(Dr, r):
    """
    Calculate viscoelastic modulus using generalized Stokes-Einstein relation.
    
    Args:
        Dr: Longitudinal correlation component
        r: Separation distance
        
    Returns:
        Estimated complex modulus G* (in Pa)
    """
    # Boltzmann constant (J/K)
    k_B = 1.38064852e-23
    
    # Convert to SI units
    r_si = r * 1e-6  # m
    Dr_si = Dr * 1e-12  # m^2
    
    # Calculate complex modulus
    # G* = k_B * T / (2πr * Dr)
    G = k_B * TEMPERATURE / (2 * np.pi * r_si * Dr_si)
    
    # Convert to Pa
    return G

def analyze_tpm(processed_data):
    """
    Perform two-point microrheology analysis on trajectory data.
    
    Args:
        processed_data: Dictionary with processed trajectory data
        
    Returns:
        Dictionary with TPM analysis results
    """
    # Prepare data for TPM analysis
    tpm_data = prepare_trajectories_for_tpm(processed_data)
    
    if tpm_data is None:
        print("Failed to prepare data for TPM analysis")
        return None
    
    # Initialize results
    tpm_results = {
        'time_lags': [],
        'binned_correlations': [],
        'moduli': {}
    }
    
    # Analyze for different time lags
    for time_lag in range(MIN_TIME_LAG, MAX_TIME_LAG + 1):
        print(f"Analyzing time lag: {time_lag} frames ({time_lag * DT:.3f} s)")
        
        # Track particle pairs
        pair_data = track_particle_pairs(tpm_data, time_lag)
        
        if len(pair_data['separation']) < MIN_CORR_PAIRS:
            print(f"  Not enough particle pairs for time lag {time_lag}")
            continue
        
        # Bin correlation by distance
        binned_data = bin_correlation_by_distance(pair_data)
        
        if binned_data is None:
            print(f"  Failed to bin correlation data for time lag {time_lag}")
            continue
        
        # Store time lag and binned correlations
        tpm_results['time_lags'].append(time_lag)
        tpm_results['binned_correlations'].append(binned_data)
        
        # Calculate viscoelastic moduli for valid bins
        moduli = []
        
        for i, r in enumerate(binned_data['bin_centers']):
            if not np.isnan(binned_data['Dr'][i]) and binned_data['Dr'][i] > 0:
                G = calculate_viscoelastic_modulus(binned_data['Dr'][i], r)
                moduli.append((r, G))
        
        if moduli:
            tpm_results['moduli'][time_lag] = np.array(moduli)
    
    return tpm_results

def plot_correlation_vs_distance(tpm_results, output_path):
    """
    Plot correlation functions versus distance for different time lags.
    
    Args:
        tpm_results: Dictionary with TPM analysis results
        output_path: Directory to save the plot
    """
    if not tpm_results['time_lags']:
        print("No time lags to plot")
        return
    
    # Set up plot
    plt.figure(figsize=(15, 10))
    
    # Select up to 5 time lags to display
    display_indices = np.linspace(0, len(tpm_results['time_lags'])-1, min(5, len(tpm_results['time_lags']))).astype(int)
    
    # Create subplots for Dr (parallel) and Dt (perpendicular)
    plt.subplot(2, 1, 1)
    
    for idx in display_indices:
        time_lag = tpm_results['time_lags'][idx]
        binned_data = tpm_results['binned_correlations'][idx]
        
        plt.errorbar(binned_data['bin_centers'], binned_data['Dr'],
                     yerr=binned_data['Dr_err'], fmt='o-',
                     label=f'τ = {time_lag * DT:.2f} s')
    
    plt.xlabel('Separation distance r (μm)')
    plt.ylabel('D_r (μm²)')
    plt.title('Longitudinal correlation vs separation distance')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    # Plot 1/r scaling for comparison
    r_range = np.logspace(np.log10(MIN_SEPARATION), np.log10(MAX_SEPARATION), 100)
    reference_scale = r_range[0] * binned_data['Dr'][0] / r_range
    plt.plot(r_range, reference_scale, 'k--', alpha=0.5, label='1/r scaling')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    
    for idx in display_indices:
        time_lag = tpm_results['time_lags'][idx]
        binned_data = tpm_results['binned_correlations'][idx]
        
        plt.errorbar(binned_data['bin_centers'], binned_data['Dt'],
                     yerr=binned_data['Dt_err'], fmt='o-',
                     label=f'τ = {time_lag * DT:.2f} s')
    
    plt.xlabel('Separation distance r (μm)')
    plt.ylabel('D_t (μm²)')
    plt.title('Transverse correlation vs separation distance')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "correlation_vs_distance.png"), dpi=300)
    plt.close()

def plot_correlation_vs_time(tpm_results, output_path):
    """
    Plot correlation functions versus time lag for selected distance bins.
    
    Args:
        tpm_results: Dictionary with TPM analysis results
        output_path: Directory to save the plot
    """
    if not tpm_results['time_lags']:
        print("No time lags to plot")
        return
    
    # Check if we have at least one valid binned correlation
    if not tpm_results['binned_correlations']:
        print("No valid binned correlations")
        return
    
    # Get the distance bins from the first time lag
    bin_centers = tpm_results['binned_correlations'][0]['bin_centers']
    
    # Select up to 5 distance bins to display
    display_indices = np.linspace(0, len(bin_centers)-1, min(5, len(bin_centers))).astype(int)
    
    # Set up plot
    plt.figure(figsize=(15, 10))
    
    # Subplots for Dr and Dt
    plt.subplot(2, 1, 1)
    
    # Extract data for each distance bin
    for bin_idx in display_indices:
        time_lags = np.array(tpm_results['time_lags']) * DT
        Dr_values = []
        Dr_err_values = []
        
        for time_lag_idx in range(len(tpm_results['time_lags'])):
            binned_data = tpm_results['binned_correlations'][time_lag_idx]
            Dr_values.append(binned_data['Dr'][bin_idx])
            Dr_err_values.append(binned_data['Dr_err'][bin_idx])
        
        Dr_values = np.array(Dr_values)
        Dr_err_values = np.array(Dr_err_values)
        
        plt.errorbar(time_lags, Dr_values, yerr=Dr_err_values, fmt='o-',
                     label=f'r = {bin_centers[bin_idx]:.2f} μm')
    
    plt.xlabel('Time lag τ (s)')
    plt.ylabel('D_r (μm²)')
    plt.title('Longitudinal correlation vs time lag')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    # Add reference line for diffusive behavior (linear scaling with time)
    t_range = np.logspace(np.log10(min(time_lags)), np.log10(max(time_lags)), 100)
    reference_scale = t_range / min(time_lags) * Dr_values[0]
    plt.plot(t_range, reference_scale, 'k--', alpha=0.5, label='~ τ¹ (diffusive)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    
    # Extract data for each distance bin
    for bin_idx in display_indices:
        time_lags = np.array(tpm_results['time_lags']) * DT
        Dt_values = []
        Dt_err_values = []
        
        for time_lag_idx in range(len(tpm_results['time_lags'])):
            binned_data = tpm_results['binned_correlations'][time_lag_idx]
            Dt_values.append(binned_data['Dt'][bin_idx])
            Dt_err_values.append(binned_data['Dt_err'][bin_idx])
        
        Dt_values = np.array(Dt_values)
        Dt_err_values = np.array(Dt_err_values)
        
        plt.errorbar(time_lags, Dt_values, yerr=Dt_err_values, fmt='o-',
                     label=f'r = {bin_centers[bin_idx]:.2f} μm')
    
    plt.xlabel('Time lag τ (s)')
    plt.ylabel('D_t (μm²)')
    plt.title('Transverse correlation vs time lag')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "correlation_vs_time.png"), dpi=300)
    plt.close()

def plot_viscoelastic_moduli(tpm_results, output_path):
    """
    Plot viscoelastic moduli derived from two-point correlations.
    
    Args:
        tpm_results: Dictionary with TPM analysis results
        output_path: Directory to save the plot
    """
    if not tpm_results['moduli']:
        print("No moduli data to plot")
        return
    
    # Set up plot
    plt.figure(figsize=(12, 8))
    
    # Select up to 5 time lags to display
    display_time_lags = sorted(list(tpm_results['moduli'].keys()))
    if len(display_time_lags) > 5:
        indices = np.linspace(0, len(display_time_lags)-1, 5).astype(int)
        display_time_lags = [display_time_lags[i] for i in indices]
    
    for time_lag in display_time_lags:
        moduli_data = tpm_results['moduli'][time_lag]
        
        if moduli_data.size > 0:
            distances = moduli_data[:, 0]
            G_values = moduli_data[:, 1]
            
            plt.semilogx(distances, G_values, 'o-',
                         label=f'τ = {time_lag * DT:.2f} s')
    
    plt.xlabel('Separation distance r (μm)')
    plt.ylabel('Viscoelastic modulus G* (Pa)')
    plt.title('Viscoelastic modulus vs separation distance')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "viscoelastic_moduli.png"), dpi=300)
    plt.close()

def plot_correlation_heatmap(tpm_results, output_path):
    """
    Create heatmap of correlation values as function of distance and time lag.
    
    Args:
        tpm_results: Dictionary with TPM analysis results
        output_path: Directory to save the plot
    """
    if not tpm_results['time_lags'] or not tpm_results['binned_correlations']:
        print("No correlation data for heatmap")
        return
    
    # Extract time lags and distance bins
    time_lags = np.array(tpm_results['time_lags']) * DT
    distance_bins = tpm_results['binned_correlations'][0]['bin_centers']
    
    # Create matrices for Dr and Dt values
    Dr_matrix = np.zeros((len(time_lags), len(distance_bins)))
    Dt_matrix = np.zeros((len(time_lags), len(distance_bins)))
    
    # Fill matrices
    for i, time_lag_idx in enumerate(range(len(tpm_results['time_lags']))):
        binned_data = tpm_results['binned_correlations'][time_lag_idx]
        Dr_matrix[i, :] = binned_data['Dr']
        Dt_matrix[i, :] = binned_data['Dt']
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(distance_bins, time_lags)
    
    # Set up plot
    plt.figure(figsize=(15, 10))
    
    # Longitudinal correlation heatmap
    plt.subplot(2, 1, 1)
    plt.pcolormesh(X, Y, Dr_matrix, cmap='viridis', norm=LogNorm())
    plt.colorbar(label='D_r (μm²)')
    plt.xlabel('Separation distance r (μm)')
    plt.ylabel('Time lag τ (s)')
    plt.title('Longitudinal correlation heatmap')
    plt.xscale('log')
    plt.yscale('log')
    
    # Transverse correlation heatmap
    plt.subplot(2, 1, 2)
    plt.pcolormesh(X, Y, Dt_matrix, cmap='viridis', norm=LogNorm())
    plt.colorbar(label='D_t (μm²)')
    plt.xlabel('Separation distance r (μm)')
    plt.ylabel('Time lag τ (s)')
    plt.title('Transverse correlation heatmap')
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "correlation_heatmap.png"), dpi=300)
    plt.close()

def export_tpm_results(tpm_results, output_path):
    """
    Export TPM analysis results to CSV files.
    
    Args:
        tpm_results: Dictionary with TPM analysis results
        output_path: Directory to save the CSV files
    """
    if not tpm_results['time_lags']:
        print("No TPM results to export")
        return
    
    # Export correlation vs distance data
    for i, time_lag in enumerate(tpm_results['time_lags']):
        binned_data = tpm_results['binned_correlations'][i]
        
        # Create DataFrame
        corr_df = pd.DataFrame({
            'Distance_um': binned_data['bin_centers'],
            'Dr_um2': binned_data['Dr'],
            'Dr_error': binned_data['Dr_err'],
            'Dt_um2': binned_data['Dt'],
            'Dt_error': binned_data['Dt_err'],
            'Pair_count': binned_data['bin_counts']
        })
        
        # Save to CSV
        csv_path = os.path.join(output_path, f"correlation_distance_tau{time_lag}.csv")
        corr_df.to_csv(csv_path, index=False)
        print(f"Correlation vs distance data saved to {csv_path}")
    
    # Export moduli data
    for time_lag, moduli_data in tpm_results['moduli'].items():
        if moduli_data.size > 0:
            # Create DataFrame
            moduli_df = pd.DataFrame({
                'Distance_um': moduli_data[:, 0],
                'Modulus_Pa': moduli_data[:, 1]
            })
            
            # Save to CSV
            csv_path = os.path.join(output_path, f"moduli_tau{time_lag}.csv")
            moduli_df.to_csv(csv_path, index=False)
            print(f"Moduli data saved to {csv_path}")
    
    # Export summary of time lags analyzed
    time_lag_df = pd.DataFrame({
        'Time_lag_frames': tpm_results['time_lags'],
        'Time_lag_seconds': np.array(tpm_results['time_lags']) * DT
    })
    
    csv_path = os.path.join(output_path, "time_lags_summary.csv")
    time_lag_df.to_csv(csv_path, index=False)
    print(f"Time lag summary saved to {csv_path}")

def create_diagnostic_plot(processed_data, tpm_data, output_path):
    """
    Create diagnostic plot showing particle tracks and correlation analysis.
    
    Args:
        processed_data: Dictionary with processed trajectory data
        tpm_data: Dictionary with TPM data
        output_path: Directory to save the plot
    """
    plt.figure(figsize=(15, 15))
    
    # Plot 1: Particle trajectories
    ax1 = plt.subplot(2, 2, 1)
    
    # Plot trajectories
    trajectories = processed_data.get('trajectories', [])
    
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, min(10, len(trajectories))))
    color_idx = 0
    
    for traj in trajectories[:10]:  # Limit to 10 trajectories for clarity
        x = traj['x']
        y = traj['y']
        
        # Plot trajectory
        ax1.plot(x, y, '-', color=colors[color_idx], alpha=0.7, linewidth=1)
        
        # Mark start and end points
        ax1.plot(x[0], y[0], 'o', color=colors[color_idx], markersize=5)
        ax1.plot(x[-1], y[-1], 's', color=colors[color_idx], markersize=5)
        
        color_idx = (color_idx + 1) % len(colors)
    
    ax1.set_xlabel('X (μm)')
    ax1.set_ylabel('Y (μm)')
    ax1.set_title('Particle trajectories')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Particles in a selected frame
    ax2 = plt.subplot(2, 2, 2)
    
    # Get a frame with multiple particles
    frame_data = tpm_data['frame_data']
    frame_particles = [(frame, len(data['positions'])) for frame, data in frame_data.items()]
    frame_particles.sort(key=lambda x: x[1], reverse=True)
    
    if frame_particles:
        # Select frame with most particles
        selected_frame = frame_particles[0][0]
        positions = frame_data[selected_frame]['positions']
        
        # Calculate separation distances
        distances, dx_matrix, dy_matrix = calculate_separation_distances(positions)
        
        # Plot particles
        ax2.scatter(positions[:, 0], positions[:, 1], color='blue', s=50, alpha=0.7)
        
        # Plot connecting lines for pairs within distance range
        if distances is not None:
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    if MIN_SEPARATION <= distances[i, j] <= MAX_SEPARATION:
                        # Draw line between particles
                        ax2.plot([positions[i, 0], positions[j, 0]], 
                                [positions[i, 1], positions[j, 1]], 
                                'r-', alpha=0.4)
                        
                        # Mark distance
                        midpoint = 0.5 * (positions[i] + positions[j])
                        ax2.text(midpoint[0], midpoint[1], f'{distances[i, j]:.1f}', 
                                fontsize=8, ha='center', va='center',
                                bbox=dict(facecolor='white', alpha=0.7))
        
        ax2.set_xlabel('X (μm)')
        ax2.set_ylabel('Y (μm)')
        ax2.set_title(f'Particles and separations at frame {selected_frame}')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
    
    # Plot 3: Displacement correlation scatter
    ax3 = plt.subplot(2, 2, 3)
    
    # Track particle pairs for a short time lag
    time_lag = min(MIN_TIME_LAG, 5)
    pair_data = track_particle_pairs(tpm_data, time_lag)
    
    if len(pair_data['displacement_i_x']) > 0:
        # Displacement components for particle i
        dx_i = pair_data['displacement_i_x']
        dy_i = pair_data['displacement_i_y']
        
        # Displacement components for particle j
        dx_j = pair_data['displacement_j_x']
        dy_j = pair_data['displacement_j_y']
        
        # Scatter plot of displacement correlations
        sc = ax3.scatter(dx_i, dx_j, c=pair_data['separation'], 
                        cmap='viridis', alpha=0.7, s=30)
        
        ax3.set_xlabel('Δx_i (μm)')
        ax3.set_ylabel('Δx_j (μm)')
        ax3.set_title(f'Displacement correlation (τ = {time_lag*DT:.2f} s)')
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        
        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax3)
        cbar.set_label('Separation distance (μm)')
        
        # Add reference line
        lim = max(abs(plt.xlim()[0]), abs(plt.xlim()[1]),
                 abs(plt.ylim()[0]), abs(plt.ylim()[1]))
        ax3.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5)
    
    # Plot 4: Correlation vs distance for selected time lag
    ax4 = plt.subplot(2, 2, 4)
    
    if len(pair_data['separation']) >= MIN_CORR_PAIRS:
        # Bin correlation by distance
        binned_data = bin_correlation_by_distance(pair_data)
        
        if binned_data is not None:
            # Plot longitudinal and transverse correlations
            ax4.errorbar(binned_data['bin_centers'], binned_data['Dr'],
                         yerr=binned_data['Dr_err'], fmt='o-', color='blue',
                         label='D_r (longitudinal)')
            
            ax4.errorbar(binned_data['bin_centers'], binned_data['Dt'],
                         yerr=binned_data['Dt_err'], fmt='o-', color='red',
                         label='D_t (transverse)')
            
            ax4.set_xlabel('Separation distance r (μm)')
            ax4.set_ylabel('Correlation (μm²)')
            ax4.set_title(f'Correlation vs distance (τ = {time_lag*DT:.2f} s)')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "tpm_diagnostic.png"), dpi=300)
    plt.close()

def main():
    """Main function to perform two-point microrheology analysis."""
    print("Two-Point Microrheology Analysis")
    print("================================")
    
    # Ask for input directory
    input_dir = input("Enter the directory containing processed trajectory files (press Enter for processed_trajectories): ")
    
    if input_dir == "":
        # Default to the processed_trajectories directory in the current folder
        input_dir = os.path.join(os.getcwd(), "processed_trajectories")
    
    # Check if directory exists
    if not os.path.isdir(input_dir):
        print(f"Directory {input_dir} does not exist")
        return
    
    # Get list of processed files
    file_paths = glob.glob(os.path.join(input_dir, "tracked_*.pkl"))
    
    if not file_paths:
        print(f"No processed trajectory files found in {input_dir}")
        return
    
    print(f"Found {len(file_paths)} files to analyze")
    
    # Create output directory for TPM results
    output_dir = os.path.join(os.path.dirname(input_dir), "tpm_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        
        # Load processed data
        processed_data = load_processed_data(file_path)
        
        if processed_data is None:
            print(f"Skipping {filename} due to loading errors")
            continue
        
        # Check if we have enough trajectories for TPM
        if len(processed_data.get('trajectories', [])) < 5:
            print(f"Skipping {filename} - at least 5 trajectories needed for TPM analysis")
            continue
        
        print(f"Processing {filename} with {len(processed_data['trajectories'])} trajectories")
        
        # Prepare data for TPM analysis
        tpm_data = prepare_trajectories_for_tpm(processed_data)
        
        if tpm_data is None:
            print(f"Failed to prepare {filename} for TPM analysis")
            continue
        
        # Create file-specific output directory
        file_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Create diagnostic plot
        print("Creating diagnostic plot...")
        create_diagnostic_plot(processed_data, tpm_data, file_output_dir)
        
        # Perform TPM analysis
        print("Performing TPM analysis...")
        tpm_results = analyze_tpm(processed_data)
        
        if tpm_results is None or not tpm_results['time_lags']:
            print(f"TPM analysis failed for {filename}")
            continue
        
        # Create plots
        print("Creating correlation vs distance plot...")
        plot_correlation_vs_distance(tpm_results, file_output_dir)
        
        print("Creating correlation vs time plot...")
        plot_correlation_vs_time(tpm_results, file_output_dir)
        
        print("Creating viscoelastic moduli plot...")
        plot_viscoelastic_moduli(tpm_results, file_output_dir)
        
        print("Creating correlation heatmap...")
        plot_correlation_heatmap(tpm_results, file_output_dir)
        
        # Export results to CSV
        print("Exporting results...")
        export_tpm_results(tpm_results, file_output_dir)
        
        # Save TPM results for further analysis
        output_file = os.path.join(file_output_dir, f"tpm_results_{base_name}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(tpm_results, f)
        
        print(f"TPM results saved to {output_file}")
    
    print(f"All files processed. Results saved in {output_dir}")

if __name__ == "__main__":
    main()