# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 13:10:47 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_tpm_analysis.py

Simplified two-point microrheology analysis with noise reduction methods.
This script analyzes particle pair correlations to extract material properties.

WHAT THIS CODE DOES:
1. Takes particle trajectories as input
2. Finds particle pairs at different separations
3. Calculates how their motions are correlated
4. Extracts material properties (stiffness) from correlations

Global parameters - MODIFY THESE
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import warnings

# Global parameters - MODIFY THESE AS NEEDED
# ==========================================
PIXEL_SIZE = 0.094          # micrometers per pixel - CHANGE THIS FOR YOUR SETUP
FRAME_RATE = 10             # frames per second - CHANGE THIS FOR YOUR SETUP  
TEMPERATURE = 298.15        # Kelvin (25°C)
PARTICLE_RADIUS = 0.4       # micrometers

# Analysis parameters
MIN_SEPARATION = 1.0        # minimum distance between particles (μm)
MAX_SEPARATION = 10.0       # maximum distance between particles (μm)
N_DISTANCE_BINS = 8         # number of distance bins
MIN_PAIRS_PER_BIN = 10      # minimum pairs needed for statistics
MAX_TIME_LAG = 20           # maximum time lag to analyze (frames)

# Noise reduction parameters
OUTLIER_THRESHOLD = 3.0     # standard deviations for outlier removal
BOOTSTRAP_SAMPLES = 100     # number of bootstrap samples for error estimation
# ==========================================

def load_trajectory_data(file_path):
    """Load trajectory data from pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {len(data.get('trajectories', []))} trajectories")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def remove_outlier_trajectories(trajectories):
    """
    Remove trajectories with unusual diffusion coefficients.
    This is a noise reduction method from statistical mechanics.
    """
    if not trajectories:
        return trajectories
    
    # Calculate basic diffusion coefficient for each trajectory
    diffusion_coeffs = []
    valid_trajs = []
    
    for traj in trajectories:
        if len(traj['x']) < 10:  # Skip very short trajectories
            continue
            
        # Calculate simple MSD slope
        dx = np.diff(traj['x'])
        dy = np.diff(traj['y'])
        msd = np.cumsum(dx**2 + dy**2)
        
        if len(msd) >= 3:
            # Fit first few points to get diffusion coefficient
            time_points = np.arange(1, min(6, len(msd)+1)) / FRAME_RATE
            try:
                slope, _ = np.polyfit(time_points, msd[:len(time_points)], 1)
                D_apparent = slope / 4  # 2D diffusion
                diffusion_coeffs.append(D_apparent)
                valid_trajs.append(traj)
            except:
                continue
    
    if not diffusion_coeffs:
        return trajectories
    
    # Remove outliers using robust statistics
    median_D = np.median(diffusion_coeffs)
    mad = np.median(np.abs(diffusion_coeffs - median_D))
    
    if mad > 0:
        # Modified Z-score method
        modified_z = 0.6745 * np.abs(diffusion_coeffs - median_D) / mad
        good_indices = modified_z < OUTLIER_THRESHOLD
        
        filtered_trajs = [valid_trajs[i] for i in range(len(valid_trajs)) if good_indices[i]]
        print(f"Removed {len(valid_trajs) - len(filtered_trajs)} outlier trajectories")
        return filtered_trajs
    
    return valid_trajs

def organize_trajectories_by_frame(trajectories):
    """
    Organize trajectory data by frame number.
    This makes pair analysis more efficient.
    """
    frame_data = {}
    
    for traj in trajectories:
        traj_id = traj['id']
        
        # Get frame numbers
        if 'time' in traj:
            frames = np.round(np.array(traj['time']) * FRAME_RATE).astype(int)
        else:
            frames = np.arange(len(traj['x']))
        
        # Store position for each frame
        for i, frame in enumerate(frames):
            if frame not in frame_data:
                frame_data[frame] = {'ids': [], 'positions': []}
            
            frame_data[frame]['ids'].append(traj_id)
            frame_data[frame]['positions'].append([traj['x'][i], traj['y'][i]])
    
    # Convert to numpy arrays
    for frame in frame_data:
        frame_data[frame]['ids'] = np.array(frame_data[frame]['ids'])
        frame_data[frame]['positions'] = np.array(frame_data[frame]['positions'])
    
    return frame_data

def calculate_pair_correlations(frame_data, time_lag):
    """
    Calculate displacement correlations between particle pairs.
    
    This is the core of two-point microrheology:
    - Find particle pairs separated by specific distances
    - Calculate their displacement correlation over time_lag
    """
    pair_correlations = {
        'separations': [],
        'longitudinal_corr': [],  # Motion along the line connecting particles
        'transverse_corr': [],    # Motion perpendicular to connecting line
        'displacements_i': [],
        'displacements_j': []
    }
    
    frames = sorted(frame_data.keys())
    
    for frame in frames:
        next_frame = frame + time_lag
        
        if next_frame not in frame_data:
            continue
        
        current_data = frame_data[frame]
        next_data = frame_data[next_frame]
        
        # Find particles present in both frames
        common_ids = np.intersect1d(current_data['ids'], next_data['ids'])
        
        if len(common_ids) < 2:
            continue
        
        # Get positions for common particles
        current_indices = [np.where(current_data['ids'] == pid)[0][0] for pid in common_ids]
        next_indices = [np.where(next_data['ids'] == pid)[0][0] for pid in common_ids]
        
        current_positions = current_data['positions'][current_indices]
        next_positions = next_data['positions'][next_indices]
        
        # Calculate displacements
        displacements = (next_positions - current_positions) * PIXEL_SIZE
        
        # Calculate all pairwise separations
        separations = pdist(current_positions * PIXEL_SIZE)
        
        # Process each pair
        pair_idx = 0
        for i in range(len(common_ids)):
            for j in range(i+1, len(common_ids)):
                separation = separations[pair_idx]
                pair_idx += 1
                
                # Skip if outside distance range
                if separation < MIN_SEPARATION or separation > MAX_SEPARATION:
                    continue
                
                # Get displacement vectors
                disp_i = displacements[i]
                disp_j = displacements[j]
                
                # Calculate unit vector along separation
                sep_vector = current_positions[j] - current_positions[i]
                sep_vector = sep_vector / np.linalg.norm(sep_vector)
                
                # Project displacements onto parallel and perpendicular directions
                proj_i_parallel = np.dot(disp_i, sep_vector)
                proj_j_parallel = np.dot(disp_j, sep_vector)
                longitudinal_corr = proj_i_parallel * proj_j_parallel
                
                # Perpendicular components
                disp_i_perp = disp_i - proj_i_parallel * sep_vector
                disp_j_perp = disp_j - proj_j_parallel * sep_vector
                transverse_corr = np.dot(disp_i_perp, disp_j_perp)
                
                # Store results
                pair_correlations['separations'].append(separation)
                pair_correlations['longitudinal_corr'].append(longitudinal_corr)
                pair_correlations['transverse_corr'].append(transverse_corr)
                pair_correlations['displacements_i'].append(disp_i)
                pair_correlations['displacements_j'].append(disp_j)
    
    # Convert to numpy arrays
    for key in pair_correlations:
        pair_correlations[key] = np.array(pair_correlations[key])
    
    return pair_correlations

def bin_correlations_by_distance(pair_correlations):
    """
    Bin correlation data by separation distance and apply noise reduction.
    Uses ensemble averaging - a key method in statistical mechanics.
    """
    if len(pair_correlations['separations']) < MIN_PAIRS_PER_BIN:
        return None
    
    # Create distance bins
    bin_edges = np.linspace(MIN_SEPARATION, MAX_SEPARATION, N_DISTANCE_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Initialize arrays
    Dr_mean = np.full(N_DISTANCE_BINS, np.nan)
    Dr_std = np.full(N_DISTANCE_BINS, np.nan)
    Dr_sem = np.full(N_DISTANCE_BINS, np.nan)
    Dt_mean = np.full(N_DISTANCE_BINS, np.nan)
    Dt_std = np.full(N_DISTANCE_BINS, np.nan)
    Dt_sem = np.full(N_DISTANCE_BINS, np.nan)
    bin_counts = np.zeros(N_DISTANCE_BINS, dtype=int)
    
    for i in range(N_DISTANCE_BINS):
        # Find pairs in this distance bin
        in_bin = ((pair_correlations['separations'] >= bin_edges[i]) & 
                  (pair_correlations['separations'] < bin_edges[i+1]))
        
        bin_counts[i] = np.sum(in_bin)
        
        if bin_counts[i] >= MIN_PAIRS_PER_BIN:
            # Extract correlations for this bin
            Dr_bin = pair_correlations['longitudinal_corr'][in_bin]
            Dt_bin = pair_correlations['transverse_corr'][in_bin]
            
            # Remove outliers within the bin (robust statistics)
            Dr_clean = remove_outliers_robust(Dr_bin)
            Dt_clean = remove_outliers_robust(Dt_bin)
            
            if len(Dr_clean) >= 3:
                Dr_mean[i] = np.mean(Dr_clean)
                Dr_std[i] = np.std(Dr_clean)
                Dr_sem[i] = Dr_std[i] / np.sqrt(len(Dr_clean))
            
            if len(Dt_clean) >= 3:
                Dt_mean[i] = np.mean(Dt_clean)
                Dt_std[i] = np.std(Dt_clean)
                Dt_sem[i] = Dt_std[i] / np.sqrt(len(Dt_clean))
    
    return {
        'bin_centers': bin_centers,
        'bin_counts': bin_counts,
        'Dr_mean': Dr_mean,
        'Dr_std': Dr_std,
        'Dr_sem': Dr_sem,
        'Dt_mean': Dt_mean,
        'Dt_std': Dt_std,
        'Dt_sem': Dt_sem
    }

def remove_outliers_robust(data):
    """
    Remove outliers using median absolute deviation (MAD).
    This is more robust than standard deviation for non-Gaussian data.
    """
    if len(data) < 3:
        return data
    
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    if mad == 0:
        return data
    
    # Modified Z-score
    modified_z = 0.6745 * np.abs(data - median) / mad
    return data[modified_z < OUTLIER_THRESHOLD]

def bootstrap_error_estimation(pair_correlations, n_bootstrap=BOOTSTRAP_SAMPLES):
    """
    Estimate errors using bootstrap resampling.
    This is a powerful statistical method for error estimation.
    """
    n_pairs = len(pair_correlations['separations'])
    if n_pairs < 10:
        return None
    
    # Bin the original data
    binned_original = bin_correlations_by_distance(pair_correlations)
    if binned_original is None:
        return None
    
    # Bootstrap resampling
    bootstrap_Dr = []
    bootstrap_Dt = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_pairs, size=n_pairs, replace=True)
        
        # Create resampled data
        resampled_correlations = {}
        for key in pair_correlations:
            resampled_correlations[key] = pair_correlations[key][indices]
        
        # Bin resampled data
        binned_resample = bin_correlations_by_distance(resampled_correlations)
        
        if binned_resample is not None:
            bootstrap_Dr.append(binned_resample['Dr_mean'])
            bootstrap_Dt.append(binned_resample['Dt_mean'])
    
    if not bootstrap_Dr:
        return binned_original
    
    # Calculate bootstrap errors
    bootstrap_Dr = np.array(bootstrap_Dr)
    bootstrap_Dt = np.array(bootstrap_Dt)
    
    # Update error estimates with bootstrap
    binned_original['Dr_bootstrap_std'] = np.nanstd(bootstrap_Dr, axis=0)
    binned_original['Dt_bootstrap_std'] = np.nanstd(bootstrap_Dt, axis=0)
    
    return binned_original

def calculate_viscoelastic_modulus(Dr_values, distances):
    """
    Calculate viscoelastic modulus using generalized Stokes-Einstein relation.
    G* = kT / (2πr * Dr)
    """
    k_B = 1.38064852e-23  # Boltzmann constant (J/K)
    
    moduli = []
    for Dr, r in zip(Dr_values, distances):
        if not np.isnan(Dr) and Dr > 0 and r > 0:
            # Convert to SI units
            r_si = r * 1e-6  # meters
            Dr_si = Dr * 1e-12  # m²
            
            # Calculate modulus
            G = k_B * TEMPERATURE / (2 * np.pi * r_si * Dr_si)
            moduli.append(G)
        else:
            moduli.append(np.nan)
    
    return np.array(moduli)

def analyze_single_file(file_path):
    """
    Analyze a single trajectory file for two-point microrheology.
    """
    print(f"\nAnalyzing: {os.path.basename(file_path)}")
    
    # Load data
    data = load_trajectory_data(file_path)
    if data is None:
        return None
    
    trajectories = data.get('trajectories', [])
    if len(trajectories) < 5:
        print("Need at least 5 trajectories for TPM analysis")
        return None
    
    # Remove outlier trajectories
    trajectories = remove_outlier_trajectories(trajectories)
    print(f"Using {len(trajectories)} trajectories after outlier removal")
    
    # Organize by frame
    frame_data = organize_trajectories_by_frame(trajectories)
    
    # Analyze different time lags
    results = {
        'time_lags': [],
        'binned_data': [],
        'moduli': []
    }
    
    for time_lag in range(1, min(MAX_TIME_LAG + 1, 21)):
        print(f"  Analyzing time lag {time_lag} frames ({time_lag/FRAME_RATE:.2f}s)")
        
        # Calculate pair correlations
        pair_correlations = calculate_pair_correlations(frame_data, time_lag)
        
        if len(pair_correlations['separations']) < MIN_PAIRS_PER_BIN:
            print(f"    Not enough pairs ({len(pair_correlations['separations'])})")
            continue
        
        # Bin correlations with bootstrap error estimation
        binned_data = bootstrap_error_estimation(pair_correlations)
        
        if binned_data is None:
            continue
        
        # Calculate viscoelastic moduli
        moduli = calculate_viscoelastic_modulus(binned_data['Dr_mean'], 
                                               binned_data['bin_centers'])
        
        # Store results
        results['time_lags'].append(time_lag)
        results['binned_data'].append(binned_data)
        results['moduli'].append(moduli)
        
        print(f"    Found {np.sum(binned_data['bin_counts'])} total pairs")
    
    return results

def create_diagnostic_plots(results, output_path, filename):
    """
    Create diagnostic plots showing what the algorithm measures.
    """
    if not results or not results['time_lags']:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Correlation vs distance (shows material properties)
    ax = axes[0, 0]
    
    # Show first few time lags
    for i, time_lag in enumerate(results['time_lags'][:3]):
        binned_data = results['binned_data'][i]
        
        # Plot with error bars
        valid = ~np.isnan(binned_data['Dr_mean'])
        if np.any(valid):
            ax.errorbar(binned_data['bin_centers'][valid], 
                       binned_data['Dr_mean'][valid],
                       yerr=binned_data['Dr_sem'][valid],
                       fmt='o-', label=f'τ = {time_lag/FRAME_RATE:.2f}s')
    
    ax.set_xlabel('Separation distance r (μm)')
    ax.set_ylabel('Longitudinal correlation Dr (μm²)')
    ax.set_title('WHAT WE MEASURE: Particle correlation vs distance')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add reference line
    if results['binned_data']:
        binned_data = results['binned_data'][0]
        valid = ~np.isnan(binned_data['Dr_mean'])
        if np.any(valid):
            r_ref = binned_data['bin_centers'][valid]
            Dr_ref = binned_data['Dr_mean'][valid][0] * r_ref[0] / r_ref
            ax.plot(r_ref, Dr_ref, 'k--', alpha=0.5, label='1/r (viscous fluid)')
            ax.legend()
    
    # Plot 2: Viscoelastic modulus (material stiffness)
    ax = axes[0, 1]
    
    for i, time_lag in enumerate(results['time_lags'][:3]):
        moduli = results['moduli'][i]
        binned_data = results['binned_data'][i]
        
        valid = ~np.isnan(moduli)
        if np.any(valid):
            ax.semilogx(binned_data['bin_centers'][valid], 
                       moduli[valid], 'o-', 
                       label=f'τ = {time_lag/FRAME_RATE:.2f}s')
    
    ax.set_xlabel('Separation distance r (μm)')
    ax.set_ylabel('Viscoelastic modulus G* (Pa)')
    ax.set_title('WHAT WE CALCULATE: Material stiffness')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Data quality metrics
    ax = axes[1, 0]
    
    all_distances = []
    all_counts = []
    all_Dr_values = []
    
    for binned_data in results['binned_data']:
        all_distances.extend(binned_data['bin_centers'])
        all_counts.extend(binned_data['bin_counts'])
        all_Dr_values.extend(binned_data['Dr_mean'])
    
    scatter = ax.scatter(all_distances, all_counts, c=all_Dr_values, 
                        cmap='viridis', s=50, alpha=0.7)
    ax.set_xlabel('Separation distance (μm)')
    ax.set_ylabel('Number of particle pairs')
    ax.set_title('DATA QUALITY: More pairs = better statistics')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Dr correlation (μm²)')
    
    # Plot 4: Summary and interpretation
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate summary statistics
    if results['moduli']:
        all_moduli = []
        for moduli in results['moduli']:
            valid_moduli = moduli[~np.isnan(moduli)]
            all_moduli.extend(valid_moduli)
        
        if all_moduli:
            mean_G = np.mean(all_moduli)
            std_G = np.std(all_moduli)
            
            summary_text = f"ANALYSIS SUMMARY:\n\n"
            summary_text += f"File: {filename}\n"
            summary_text += f"Time lags analyzed: {len(results['time_lags'])}\n"
            summary_text += f"Total particle pairs: {sum(all_counts)}\n\n"
            summary_text += f"MATERIAL PROPERTIES:\n"
            summary_text += f"Viscoelastic modulus: {mean_G:.1e} ± {std_G:.1e} Pa\n\n"
            
            # Material interpretation
            if mean_G < 1e-2:
                material_type = "Very soft (water-like)"
            elif mean_G < 1:
                material_type = "Soft fluid"
            elif mean_G < 100:
                material_type = "Soft gel/cytoplasm"
            elif mean_G < 1000:
                material_type = "Stiff gel"
            else:
                material_type = "Very stiff material"
            
            summary_text += f"Material type: {material_type}\n\n"
            summary_text += f"WHAT THIS MEANS:\n"
            summary_text += f"• Higher modulus = stiffer material\n"
            summary_text += f"• Slope in top-left tells you material type\n"
            summary_text += f"• More data points = more reliable results"
            
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{filename}_tpm_analysis.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_results(results, output_path, filename):
    """Save results to files."""
    if not results or not results['time_lags']:
        return
    
    # Save detailed results as pickle
    pickle_path = os.path.join(output_path, f"{filename}_tpm_results.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary as CSV
    summary_data = []
    for i, time_lag in enumerate(results['time_lags']):
        binned_data = results['binned_data'][i]
        moduli = results['moduli'][i]
        
        for j, distance in enumerate(binned_data['bin_centers']):
            summary_data.append({
                'time_lag_frames': time_lag,
                'time_lag_seconds': time_lag / FRAME_RATE,
                'distance_um': distance,
                'Dr_correlation': binned_data['Dr_mean'][j],
                'Dr_error': binned_data['Dr_sem'][j],
                'Dt_correlation': binned_data['Dt_mean'][j],
                'Dt_error': binned_data['Dt_sem'][j],
                'pair_count': binned_data['bin_counts'][j],
                'modulus_Pa': moduli[j]
            })
    
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_path, f"{filename}_tpm_summary.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"Results saved:")
    print(f"  Detailed: {pickle_path}")
    print(f"  Summary: {csv_path}")

def main():
    """Main function for simple TPM analysis."""
    print("Simple Two-Point Microrheology Analysis")
    print("======================================")
    print("This analyzes particle pair correlations to extract material properties")
    
    # Get input file
    input_file = input("\nEnter path to trajectory file (.pkl): ")
    
    if not os.path.isfile(input_file):
        print(f"File not found: {input_file}")
        return
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(input_file), "tpm_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename for outputs
    filename = os.path.splitext(os.path.basename(input_file))[0]
    
    # Analyze the file
    results = analyze_single_file(input_file)
    
    if results is None:
        print("Analysis failed - check your data")
        return
    
    # Create diagnostic plots
    print("\nCreating diagnostic plots...")
    create_diagnostic_plots(results, output_dir, filename)
    
    # Save results
    print("Saving results...")
    save_results(results, output_dir, filename)
    
    print(f"\nAnalysis complete! Results saved in: {output_dir}")
    print("\nKey outputs:")
    print("- *_tpm_analysis.png: Shows what the algorithm measures")
    print("- *_tpm_summary.csv: Numerical results")
    print("- *_tpm_results.pkl: Detailed results for further analysis")

if __name__ == "__main__":
    main()