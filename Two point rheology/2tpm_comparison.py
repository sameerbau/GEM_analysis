# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 16:37:08 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tpm_comparison.py

This script compares two-point microrheology (TPM) results from multiple samples,
enabling direct comparisons of viscoelastic properties across different conditions.

Input:
- TPM analysis results (.pkl files) from two_point_rheology.py

Output:
- Comparative plots for correlation functions and viscoelastic moduli
- Statistical analysis of differences between samples
- Combined CSV data files for external analysis

Usage:
python tpm_comparison.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import glob
import pickle
import seaborn as sns
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Global parameters that can be modified
# =====================================
# Time step in seconds
DT = 0.1
# Maximum time lag for comparison (frames)
MAX_TIME_LAG = 20
# Statistical significance level
ALPHA = 0.05
# Frequency range for rheological analysis (Hz)
MIN_FREQ = 0.1
MAX_FREQ = 10
# =====================================

def load_tpm_results(file_path):
    """
    Load TPM results from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary containing the TPM results
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading TPM results from {file_path}: {e}")
        return None

def interpolate_correlations(tpm_results, time_lag, r_values):
    """
    Interpolate correlation functions for a specific time lag at specific distances.
    
    Args:
        tpm_results: Dictionary with TPM results
        time_lag: Time lag (frames) to interpolate
        r_values: Array of distance values for interpolation
        
    Returns:
        Tuple of interpolated Dr and Dt values
    """
    # Find time lag in results
    if time_lag not in tpm_results['time_lags']:
        return None, None
    
    # Get index of time lag
    time_lag_idx = tpm_results['time_lags'].index(time_lag)
    
    # Get binned correlation data
    binned_data = tpm_results['binned_correlations'][time_lag_idx]
    
    # Extract distance values and correlation values
    distances = binned_data['bin_centers']
    Dr_values = binned_data['Dr']
    Dt_values = binned_data['Dt']
    
    # Find valid (non-NaN) values
    valid_dr = ~np.isnan(Dr_values)
    valid_dt = ~np.isnan(Dt_values)
    
    # Check if we have enough valid points for interpolation
    if np.sum(valid_dr) < 3 or np.sum(valid_dt) < 3:
        return None, None
    
    # Interpolate Dr and Dt
    try:
        Dr_interp = np.interp(r_values, distances[valid_dr], Dr_values[valid_dr], 
                             left=np.nan, right=np.nan)
        Dt_interp = np.interp(r_values, distances[valid_dt], Dt_values[valid_dt], 
                             left=np.nan, right=np.nan)
        return Dr_interp, Dt_interp
    except Exception as e:
        print(f"Interpolation error: {e}")
        return None, None

def compare_correlations(tpm_results_list, sample_names, time_lag, output_path):
    """
    Compare correlation functions for a specific time lag across samples.
    
    Args:
        tpm_results_list: List of dictionaries with TPM results
        sample_names: List of sample names
        time_lag: Time lag (frames) to compare
        output_path: Directory to save the plot
    """
    # Check if we have enough samples
    if len(tpm_results_list) < 2:
        print("Need at least 2 samples for comparison")
        return
    
    # Create a common set of distance values for interpolation
    min_r = np.inf
    max_r = 0
    
    for tpm_results in tpm_results_list:
        if time_lag not in tpm_results['time_lags']:
            continue
        
        time_lag_idx = tpm_results['time_lags'].index(time_lag)
        binned_data = tpm_results['binned_correlations'][time_lag_idx]
        
        # Update min and max distance
        valid = ~np.isnan(binned_data['Dr'])
        if np.sum(valid) > 0:
            min_r = min(min_r, np.min(binned_data['bin_centers'][valid]))
            max_r = max(max_r, np.max(binned_data['bin_centers'][valid]))
    
    # Check if we have valid range
    if min_r >= max_r:
        print(f"No valid range for time lag {time_lag}")
        return
    
    # Create common distance values
    r_values = np.linspace(min_r, max_r, 100)
    
    # Set up plot
    plt.figure(figsize=(15, 10))
    
    # Plot Dr (longitudinal correlation)
    plt.subplot(2, 1, 1)
    
    for i, (tpm_results, name) in enumerate(zip(tpm_results_list, sample_names)):
        # Interpolate correlations
        Dr_interp, _ = interpolate_correlations(tpm_results, time_lag, r_values)
        
        if Dr_interp is not None:
            plt.plot(r_values, Dr_interp, '-', label=name)
    
    plt.xlabel('Separation distance r (μm)')
    plt.ylabel('D_r (μm²)')
    plt.title(f'Longitudinal correlation comparison (τ = {time_lag * DT:.2f} s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    # Add 1/r scaling reference
    valid_indices = ~np.isnan(Dr_interp)
    if np.sum(valid_indices) > 0:
        ref_idx = np.where(valid_indices)[0][0]
        reference_scale = r_values[ref_idx] * Dr_interp[ref_idx] / r_values
        plt.plot(r_values, reference_scale, 'k--', alpha=0.5, label='1/r scaling')
        plt.legend()
    
    # Plot Dt (transverse correlation)
    plt.subplot(2, 1, 2)
    
    for i, (tpm_results, name) in enumerate(zip(tpm_results_list, sample_names)):
        # Interpolate correlations
        _, Dt_interp = interpolate_correlations(tpm_results, time_lag, r_values)
        
        if Dt_interp is not None:
            plt.plot(r_values, Dt_interp, '-', label=name)
    
    plt.xlabel('Separation distance r (μm)')
    plt.ylabel('D_t (μm²)')
    plt.title(f'Transverse correlation comparison (τ = {time_lag * DT:.2f} s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"correlation_comparison_tau{time_lag}.png"), dpi=300)
    plt.close()

def calculate_scaling_exponent(distances, correlations):
    """
    Calculate scaling exponent for correlation vs distance (D ~ r^α).
    
    Args:
        distances: Array of distance values
        correlations: Array of correlation values
        
    Returns:
        Tuple of scaling exponent and R^2 value
    """
    # Log transform data
    log_distances = np.log(distances)
    log_correlations = np.log(correlations)
    
    # Linear regression on log-transformed data
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_distances, log_correlations)
    
    return slope, r_value**2

def compare_scaling_behavior(tpm_results_list, sample_names, output_path):
    """
    Compare scaling behavior of correlation functions across samples.
    
    Args:
        tpm_results_list: List of dictionaries with TPM results
        sample_names: List of sample names
        output_path: Directory to save the plot
    """
    # Set up plot
    plt.figure(figsize=(12, 8))
    
    # Collect scaling exponents for different time lags
    scaling_data = []
    
    # Find common time lags
    common_time_lags = set()
    for tpm_results in tpm_results_list:
        common_time_lags.update(tpm_results['time_lags'])
    
    common_time_lags = sorted(list(common_time_lags))
    common_time_lags = [tl for tl in common_time_lags if tl <= MAX_TIME_LAG]
    
    # Calculate scaling exponents for each sample and time lag
    for time_lag in common_time_lags:
        for i, (tpm_results, name) in enumerate(zip(tpm_results_list, sample_names)):
            if time_lag not in tpm_results['time_lags']:
                continue
                
            time_lag_idx = tpm_results['time_lags'].index(time_lag)
            binned_data = tpm_results['binned_correlations'][time_lag_idx]
            
            # Get valid data points
            valid_dr = ~np.isnan(binned_data['Dr'])
            valid_dt = ~np.isnan(binned_data['Dt'])
            
            if np.sum(valid_dr) >= 3:
                # Calculate scaling exponent for Dr
                slope_dr, r2_dr = calculate_scaling_exponent(
                    binned_data['bin_centers'][valid_dr],
                    binned_data['Dr'][valid_dr]
                )
                
                scaling_data.append({
                    'Sample': name,
                    'Time_lag': time_lag,
                    'Time_seconds': time_lag * DT,
                    'Component': 'Longitudinal (Dr)',
                    'Scaling_exponent': slope_dr,
                    'R2': r2_dr
                })
            
            if np.sum(valid_dt) >= 3:
                # Calculate scaling exponent for Dt
                slope_dt, r2_dt = calculate_scaling_exponent(
                    binned_data['bin_centers'][valid_dt],
                    binned_data['Dt'][valid_dt]
                )
                
                scaling_data.append({
                    'Sample': name,
                    'Time_lag': time_lag,
                    'Time_seconds': time_lag * DT,
                    'Component': 'Transverse (Dt)',
                    'Scaling_exponent': slope_dt,
                    'R2': r2_dt
                })
    
    # Create DataFrame
    scaling_df = pd.DataFrame(scaling_data)
    
    # Plot scaling exponents vs time lag
    sns.lineplot(data=scaling_df, x='Time_seconds', y='Scaling_exponent', 
                hue='Sample', style='Component', markers=True, dashes=False)
    
    # Add reference line for 1/r scaling (exponent = -1)
    plt.axhline(-1, color='k', linestyle='--', alpha=0.5, label='1/r scaling (α = -1)')
    
    plt.xlabel('Time lag τ (s)')
    plt.ylabel('Scaling exponent α')
    plt.title('Correlation scaling behavior (D ~ r^α)')
    plt.grid(True, alpha=0.3)
    plt.legend(title='')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "scaling_exponents.png"), dpi=300)
    plt.close()
    
    # Save scaling data to CSV
    scaling_df.to_csv(os.path.join(output_path, "scaling_exponents.csv"), index=False)
    
    return scaling_df

def estimate_complex_modulus(frequency, Dr, r):
    """
    Estimate complex modulus from correlation function using generalized Stokes-Einstein relation.
    
    Args:
        frequency: Frequency in Hz
        Dr: Longitudinal correlation value
        r: Separation distance in μm
        
    Returns:
        Complex modulus G* in Pa
    """
    # Constants
    k_B = 1.38064852e-23  # Boltzmann constant (J/K)
    T = 298.15  # Temperature (K)
    
    # Convert to SI units
    r_si = r * 1e-6  # m
    Dr_si = Dr * 1e-12  # m^2
    
    # Calculate complex modulus
    # G* = k_B * T / (2πr * Dr)
    G = k_B * T / (2 * np.pi * r_si * Dr_si)
    
    return G

def compare_viscoelastic_moduli(tpm_results_list, sample_names, output_path):
    """
    Compare viscoelastic moduli across samples.
    
    Args:
        tpm_results_list: List of dictionaries with TPM results
        sample_names: List of sample names
        output_path: Directory to save the plot
    """
    # Check if we have moduli data
    moduli_data = []
    
    for i, (tpm_results, name) in enumerate(zip(tpm_results_list, sample_names)):
        if not tpm_results['moduli']:
            continue
            
        for time_lag, moduli in tpm_results['moduli'].items():
            if moduli.size > 0:
                for r, G in moduli:
                    moduli_data.append({
                        'Sample': name,
                        'Time_lag': time_lag,
                        'Time_seconds': time_lag * DT,
                        'Frequency': 1 / (time_lag * DT),
                        'Distance': r,
                        'Modulus': G
                    })
    
    if not moduli_data:
        print("No moduli data available")
        return
    
    # Create DataFrame
    moduli_df = pd.DataFrame(moduli_data)
    
    # Set up plot
    plt.figure(figsize=(12, 8))
    
    # Group by sample and plot G vs frequency
    for name in sample_names:
        sample_data = moduli_df[moduli_df['Sample'] == name]
        
        if len(sample_data) > 0:
            # Group by frequency and calculate mean
            grouped = sample_data.groupby('Frequency')['Modulus'].mean().reset_index()
            plt.loglog(grouped['Frequency'], grouped['Modulus'], 'o-', label=name)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Complex modulus |G*| (Pa)')
    plt.title('Viscoelastic modulus comparison')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "moduli_comparison.png"), dpi=300)
    plt.close()
    
    # Save moduli data to CSV
    moduli_df.to_csv(os.path.join(output_path, "viscoelastic_moduli.csv"), index=False)
    
    return moduli_df

def plot_3d_correlation_surface(tpm_results_list, sample_names, output_path):
    """
    Create 3D surface plots of correlation functions vs distance and time.
    
    Args:
        tpm_results_list: List of dictionaries with TPM results
        sample_names: List of sample names
        output_path: Directory to save the plots
    """
    for i, (tpm_results, name) in enumerate(zip(tpm_results_list, sample_names)):
        # Collect data for 3D surface
        r_values = []
        t_values = []
        Dr_values = []
        Dt_values = []
        
        for time_lag_idx, time_lag in enumerate(tpm_results['time_lags']):
            binned_data = tpm_results['binned_correlations'][time_lag_idx]
            
            for j, r in enumerate(binned_data['bin_centers']):
                if not np.isnan(binned_data['Dr'][j]):
                    r_values.append(r)
                    t_values.append(time_lag * DT)
                    Dr_values.append(binned_data['Dr'][j])
                
                if not np.isnan(binned_data['Dt'][j]):
                    r_values.append(r)
                    t_values.append(time_lag * DT)
                    Dt_values.append(binned_data['Dt'][j])
        
        # Check if we have enough data points
        if len(r_values) < 10:
            print(f"Not enough data points for 3D surface plot for {name}")
            continue
        
        # Create grid for interpolation
        r_grid = np.linspace(min(r_values), max(r_values), 50)
        t_grid = np.linspace(min(t_values), max(t_values), 50)
        R, T = np.meshgrid(r_grid, t_grid)
        
        # Interpolate Dr and Dt onto grid
        Dr_grid = griddata((r_values, t_values), Dr_values, (R, T), method='cubic')
        
        # Create 3D surface plot for Dr
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(R, T, Dr_grid, cmap='viridis', alpha=0.8,
                              linewidth=0, antialiased=True)
        
        # Add scatter points for actual data
        ax.scatter(r_values, t_values, Dr_values, c='red', s=10, alpha=0.5)
        
        # Customize plot
        ax.set_xlabel('Separation distance r (μm)')
        ax.set_ylabel('Time lag τ (s)')
        ax.set_zlabel('D_r (μm²)')
        ax.set_title(f'Longitudinal correlation surface - {name}')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Save plot
        plt.savefig(os.path.join(output_path, f"Dr_surface_{name.replace(' ', '_')}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def statistical_comparison(scaling_df, moduli_df, output_path):
    """
    Perform statistical comparisons between samples.
    
    Args:
        scaling_df: DataFrame with scaling exponents
        moduli_df: DataFrame with viscoelastic moduli
        output_path: Directory to save the results
    """
    # List to store statistical comparison results
    stat_results = []
    
    # 1. Compare scaling exponents
    if scaling_df is not None and len(scaling_df) > 0:
        # Get unique samples and components
        samples = scaling_df['Sample'].unique()
        components = scaling_df['Component'].unique()
        
        if len(samples) >= 2:
            # Perform pairwise comparisons
            for comp in components:
                for i in range(len(samples)):
                    for j in range(i+1, len(samples)):
                        sample1 = samples[i]
                        sample2 = samples[j]
                        
                        # Get scaling exponents for each sample
                        exponents1 = scaling_df[(scaling_df['Sample'] == sample1) & 
                                              (scaling_df['Component'] == comp)]['Scaling_exponent']
                        exponents2 = scaling_df[(scaling_df['Sample'] == sample2) & 
                                              (scaling_df['Component'] == comp)]['Scaling_exponent']
                        
                        if len(exponents1) == 0 or len(exponents2) == 0:
                            continue
                        
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(exponents1, exponents2, equal_var=False)
                        
                        stat_results.append({
                            'Comparison_type': 'Scaling exponent',
                            'Component': comp,
                            'Sample1': sample1,
                            'Sample2': sample2,
                            'Mean1': exponents1.mean(),
                            'Mean2': exponents2.mean(),
                            'Difference': exponents1.mean() - exponents2.mean(),
                            'T_statistic': t_stat,
                            'P_value': p_value,
                            'Significant': p_value < ALPHA
                        })
    
    # 2. Compare viscoelastic moduli
    if moduli_df is not None and len(moduli_df) > 0:
        # Get unique samples
        samples = moduli_df['Sample'].unique()
        
        if len(samples) >= 2:
            # Group by sample and frequency
            frequency_bins = np.logspace(np.log10(moduli_df['Frequency'].min()), 
                                       np.log10(moduli_df['Frequency'].max()), 5)
            
            moduli_df['Frequency_bin'] = pd.cut(moduli_df['Frequency'], frequency_bins)
            
            # Perform pairwise comparisons for each frequency bin
            for freq_bin in moduli_df['Frequency_bin'].unique():
                for i in range(len(samples)):
                    for j in range(i+1, len(samples)):
                        sample1 = samples[i]
                        sample2 = samples[j]
                        
                        # Get moduli for each sample in this frequency bin
                        moduli1 = moduli_df[(moduli_df['Sample'] == sample1) & 
                                         (moduli_df['Frequency_bin'] == freq_bin)]['Modulus']
                        moduli2 = moduli_df[(moduli_df['Sample'] == sample2) & 
                                         (moduli_df['Frequency_bin'] == freq_bin)]['Modulus']
                        
                        if len(moduli1) == 0 or len(moduli2) == 0:
                            continue
                        
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(moduli1, moduli2, equal_var=False)
                        
                        # Get mean frequency in this bin
                        mean_freq = moduli_df[moduli_df['Frequency_bin'] == freq_bin]['Frequency'].mean()
                        
                        stat_results.append({
                            'Comparison_type': 'Viscoelastic modulus',
                            'Component': f'Frequency ~{mean_freq:.2f} Hz',
                            'Sample1': sample1,
                            'Sample2': sample2,
                            'Mean1': moduli1.mean(),
                            'Mean2': moduli2.mean(),
                            'Difference': moduli1.mean() - moduli2.mean(),
                            'T_statistic': t_stat,
                            'P_value': p_value,
                            'Significant': p_value < ALPHA
                        })
    
    # Create DataFrame and save results
    if stat_results:
        results_df = pd.DataFrame(stat_results)
        results_df.to_csv(os.path.join(output_path, "statistical_comparisons.csv"), index=False)
        
        # Create bar plot of differences
        plt.figure(figsize=(12, 8))
        
        # Filter for significant differences
        sig_results = results_df[results_df['Significant']]
        
        if len(sig_results) > 0:
            sig_results['Comparison'] = sig_results['Sample1'] + ' vs ' + sig_results['Sample2']
            sig_results['Label'] = sig_results['Component'] + ' - ' + sig_results['Comparison']
            
            # Sort by absolute difference
            sig_results = sig_results.iloc[np.argsort(np.abs(sig_results['Difference']))[::-1]]
            
            # Plot
            plt.barh(sig_results['Label'], sig_results['Difference'], 
                    color=sig_results['Difference'].apply(lambda x: 'green' if x > 0 else 'red'))
            
            plt.axvline(0, color='k', linestyle='-', alpha=0.5)
            plt.xlabel('Difference')
            plt.ylabel('Comparison')
            plt.title('Significant differences between samples')
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_path, "significant_differences.png"), dpi=300)
        
        plt.close()
        
        return results_df
    
    return None

def main():
    """Main function to compare TPM results."""
    print("Two-Point Microrheology Comparison")
    print("==================================")
    
    # Ask for input directory
    input_dir = input("Enter the directory containing TPM results folders (press Enter for tpm_results): ")
    
    if input_dir == "":
        # Default to the tpm_results directory in the current folder
        input_dir = os.path.join(os.getcwd(), "tpm_results")
    
    # Check if directory exists
    if not os.path.isdir(input_dir):
        print(f"Directory {input_dir} does not exist")
        return
    
    # Get list of result folders
    result_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    if not result_folders:
        print(f"No TPM result folders found in {input_dir}")
        return
    
    print(f"Found {len(result_folders)} result folders: {result_folders}")
    
    # Ask which folders to compare
    print("\nEnter the folders you want to compare (comma-separated, or 'all' for all folders):")
    selection = input().strip()
    
    if selection.lower() == 'all':
        selected_folders = result_folders
    else:
        selected_folders = [f.strip() for f in selection.split(',')]
        # Verify folders exist
        selected_folders = [f for f in selected_folders if f in result_folders]
    
    if len(selected_folders) < 2:
        print("Need at least 2 valid folders for comparison")
        return
    
    print(f"Selected folders for comparison: {selected_folders}")
    
    # Create output directory for comparison results
    output_dir = os.path.join(input_dir, "comparison_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load TPM results
    tpm_results_list = []
    sample_names = []
    
    for folder in selected_folders:
        # Find TPM results file
        result_files = glob.glob(os.path.join(input_dir, folder, "tpm_results_*.pkl"))
        
        if not result_files:
            print(f"No TPM results file found in {folder}")
            continue
        
        # Load results
        tpm_results = load_tpm_results(result_files[0])
        
        if tpm_results is None:
            print(f"Failed to load TPM results from {folder}")
            continue
        
        tpm_results_list.append(tpm_results)
        sample_names.append(folder)
    
    if len(tpm_results_list) < 2:
        print("Failed to load at least 2 valid TPM results for comparison")
        return
    
    print(f"Loaded TPM results for {len(tpm_results_list)} samples")
    
    # Perform comparisons
    # 1. Compare correlation functions for selected time lags
    for time_lag in [1, 5, 10, 20]:
        print(f"Comparing correlation functions for time lag {time_lag}...")
        compare_correlations(tpm_results_list, sample_names, time_lag, output_dir)
    
    # 2. Compare scaling behavior
    print("Comparing scaling behavior...")
    scaling_df = compare_scaling_behavior(tpm_results_list, sample_names, output_dir)
    
    # 3. Compare viscoelastic moduli
    print("Comparing viscoelastic moduli...")
    moduli_df = compare_viscoelastic_moduli(tpm_results_list, sample_names, output_dir)
    
    # 4. Create 3D correlation surfaces
    print("Creating 3D correlation surfaces...")
    plot_3d_correlation_surface(tpm_results_list, sample_names, output_dir)
    
    # 5. Perform statistical comparison
    print("Performing statistical comparison...")
    stat_results = statistical_comparison(scaling_df, moduli_df, output_dir)
    
    print(f"Comparison completed. Results saved in {output_dir}")

if __name__ == "__main__":
    main()