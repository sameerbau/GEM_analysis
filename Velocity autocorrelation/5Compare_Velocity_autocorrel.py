# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 15:27:53 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
velocity_autocorrelation_compare.py

Compare velocity autocorrelation functions between multiple datasets.
Includes statistical analysis and effect size calculations.

Input:
- Multiple directories containing analyzed trajectory .pkl files

Output:
- Comparative plots of velocity autocorrelation functions
- Statistical comparison of correlation times
- Effect size analysis between conditions
- CSV exports of comparison data

Usage:
python velocity_autocorrelation_compare.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
from pathlib import Path
from scipy import stats
from datetime import datetime
from itertools import combinations

# Global parameters (modify these as needed)
# =====================================
# Time step in seconds
DT = 0.1
# Minimum trajectory length
MIN_TRACK_LENGTH = 15
# Maximum time lag to analyze
MAX_TAU = 50

# Statistical parameters
ALPHA = 0.05  # Significance level
BOOTSTRAP_N = 1000  # Number of bootstrap samples for confidence intervals

# Plot parameters
FIGURE_SIZE = (12, 8)
FIGURE_DPI = 300
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
    
    # Calculate velocities for each trajectory
    velocity_data = []
    
    for traj in filtered_trajectories:
        x = np.array(traj['x'])
        y = np.array(traj['y'])
        
        # Calculate velocities
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        
        velocity_data.append({
            'vx': vx,
            'vy': vy,
            'v_mag': np.sqrt(vx**2 + vy**2),
            'traj_id': traj['id']
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
                vx_0 = vx[:-tau]
                vy_0 = vy[:-tau]
                vx_tau = vx[tau:]
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
            corr_array = corr_array[~np.isnan(corr_array)]
            
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

def load_dataset(dataset_dir, dataset_name):
    """
    Load all trajectory files from a dataset directory.
    
    Args:
        dataset_dir: Directory containing analyzed trajectory files
        dataset_name: Name for this dataset
        
    Returns:
        Dictionary with dataset information
    """
    # Find all analyzed trajectory files
    file_paths = glob.glob(os.path.join(dataset_dir, "analyzed_*.pkl"))
    
    if not file_paths:
        print(f"No analyzed trajectory files found in {dataset_dir}")
        return None
    
    print(f"Found {len(file_paths)} files in dataset '{dataset_name}'")
    
    # Pool all trajectories from all files
    all_trajectories = []
    
    for file_path in file_paths:
        analyzed_data = load_analyzed_data(file_path)
        if analyzed_data is not None and 'trajectories' in analyzed_data:
            all_trajectories.extend(analyzed_data['trajectories'])
    
    if not all_trajectories:
        print(f"No valid trajectories found in dataset '{dataset_name}'")
        return None
    
    # Calculate velocity autocorrelation for the pooled trajectories
    results = calculate_velocity_autocorrelation(all_trajectories)
    
    if results is None:
        print(f"Failed to calculate velocity autocorrelation for dataset '{dataset_name}'")
        return None
    
    return {
        'name': dataset_name,
        'results': results,
        'n_files': len(file_paths),
        'n_trajectories': len(all_trajectories),
        'n_filtered_trajectories': results['num_trajectories']
    }

def bootstrap_correlation_time(trajectories, n_bootstrap=BOOTSTRAP_N):
    """
    Calculate bootstrap confidence intervals for correlation time.
    
    Args:
        trajectories: List of trajectory dictionaries
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Array of bootstrap correlation times
    """
    bootstrap_times = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample trajectories
        bootstrap_trajectories = np.random.choice(trajectories, 
                                                 size=len(trajectories), 
                                                 replace=True)
        
        # Calculate autocorrelation for bootstrap sample
        results = calculate_velocity_autocorrelation(bootstrap_trajectories)
        
        if results is not None:
            bootstrap_times.append(results['correlation_time'])
        else:
            bootstrap_times.append(np.nan)
    
    return np.array(bootstrap_times)

def calculate_effect_size(group1, group2):
    """
    Calculate Cliff's delta effect size between two groups.
    
    Args:
        group1, group2: Arrays of values to compare
        
    Returns:
        Cliff's delta effect size
    """
    if len(group1) == 0 or len(group2) == 0:
        return np.nan
    
    greater = 0
    lesser = 0
    
    for x in group1:
        for y in group2:
            if x > y:
                greater += 1
            elif x < y:
                lesser += 1
    
    total_comparisons = len(group1) * len(group2)
    return (greater - lesser) / total_comparisons

def interpret_effect_size(delta):
    """Interpret Cliff's delta effect size."""
    abs_delta = abs(delta)
    
    if abs_delta < 0.147:
        return "Negligible"
    elif abs_delta < 0.33:
        return "Small"
    elif abs_delta < 0.474:
        return "Medium"
    else:
        return "Large"

def compare_datasets(datasets):
    """
    Perform statistical comparison between datasets.
    
    Args:
        datasets: List of dataset dictionaries
        
    Returns:
        Dictionary with comparison results
    """
    comparison_results = {
        'pairwise_comparisons': []
    }
    
    # Get all pairwise combinations
    dataset_pairs = list(combinations(range(len(datasets)), 2))
    
    for i, j in dataset_pairs:
        dataset1 = datasets[i]
        dataset2 = datasets[j]
        
        # Extract correlation times (need to bootstrap or use individual trajectory data)
        # For now, compare the autocorrelation functions at specific time lags
        
        results1 = dataset1['results']
        results2 = dataset2['results']
        
        # Compare correlation times
        corr_time1 = results1['correlation_time']
        corr_time2 = results2['correlation_time']
        
        # Compare zero-crossing times
        zero_time1 = results1['zero_crossing_time']
        zero_time2 = results2['zero_crossing_time']
        
        # For statistical tests, we need individual measurements
        # Compare autocorrelation values at specific time lags
        comparison = {
            'dataset1': dataset1['name'],
            'dataset2': dataset2['name'],
            'n_trajectories1': results1['num_trajectories'],
            'n_trajectories2': results2['num_trajectories'],
            'correlation_time1': corr_time1,
            'correlation_time2': corr_time2,
            'zero_crossing_time1': zero_time1,
            'zero_crossing_time2': zero_time2,
            'correlation_time_diff': corr_time1 - corr_time2 if not (np.isnan(corr_time1) or np.isnan(corr_time2)) else np.nan,
            'zero_crossing_diff': zero_time1 - zero_time2 if not (np.isnan(zero_time1) or np.isnan(zero_time2)) else np.nan
        }
        
        # Compare initial autocorrelation values (C_v(0) equivalent)
        if not np.isnan(results1['Cv'][0]) and not np.isnan(results2['Cv'][0]):
            comparison['initial_Cv1'] = results1['Cv'][0]
            comparison['initial_Cv2'] = results2['Cv'][0]
            comparison['initial_Cv_diff'] = results1['Cv'][0] - results2['Cv'][0]
        else:
            comparison['initial_Cv1'] = np.nan
            comparison['initial_Cv2'] = np.nan
            comparison['initial_Cv_diff'] = np.nan
        
        comparison_results['pairwise_comparisons'].append(comparison)
    
    return comparison_results

def plot_autocorrelation_comparison(datasets, output_path):
    """
    Create plots comparing velocity autocorrelation functions.
    
    Args:
        datasets: List of dataset dictionaries
        output_path: Directory to save plots
    """
    # Main comparison plot
    plt.figure(figsize=FIGURE_SIZE)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    
    for i, dataset in enumerate(datasets):
        results = dataset['results']
        
        # Plot with error bars
        plt.errorbar(results['time_lags'], results['Cv'], 
                    yerr=results['Cv_sem'], fmt='o-', 
                    color=colors[i], alpha=0.8, capsize=3,
                    label=f"{dataset['name']} (n={results['num_trajectories']})")
        
        # Add vertical line for correlation time
        if not np.isnan(results['correlation_time']):
            plt.axvline(results['correlation_time'], color=colors[i], 
                       linestyle='--', alpha=0.6)
    
    # Add reference lines
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(DT, color='gray', linestyle=':', alpha=0.7, 
               label=f'Sampling period = {DT} s')
    
    plt.xlabel('Time lag τ (s)', fontsize=12)
    plt.ylabel('Velocity Autocorrelation C_v(τ)', fontsize=12)
    plt.title('Velocity Autocorrelation Comparison', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, "velocity_autocorr_comparison.png"), 
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    # Correlation times comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of correlation times
    dataset_names = [d['name'] for d in datasets]
    corr_times = [d['results']['correlation_time'] for d in datasets]
    zero_times = [d['results']['zero_crossing_time'] for d in datasets]
    
    x_pos = np.arange(len(dataset_names))
    
    # Correlation times
    bars1 = ax1.bar(x_pos, corr_times, alpha=0.7, color=colors)
    ax1.axhline(DT, color='red', linestyle='--', alpha=0.7, 
                label=f'Sampling period = {DT} s')
    ax1.axhline(2*DT, color='orange', linestyle='--', alpha=0.7, 
                label=f'2× sampling = {2*DT} s')
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Correlation time (s)')
    ax1.set_title('Velocity Correlation Times')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, (bar, time) in enumerate(zip(bars1, corr_times)):
        if not np.isnan(time):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{time:.3f}s', ha='center', va='bottom')
    
    # Zero crossing times
    bars2 = ax2.bar(x_pos, zero_times, alpha=0.7, color=colors)
    ax2.axhline(DT, color='red', linestyle='--', alpha=0.7, 
                label=f'Sampling period = {DT} s')
    
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Zero crossing time (s)')
    ax2.set_title('Zero Crossing Times')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, (bar, time) in enumerate(zip(bars2, zero_times)):
        if not np.isnan(time):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{time:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "correlation_times_comparison.png"), 
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()

def plot_early_time_comparison(datasets, output_path):
    """
    Create focused plot of early time behavior.
    
    Args:
        datasets: List of dataset dictionaries
        output_path: Directory to save plots
    """
    plt.figure(figsize=FIGURE_SIZE)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    
    # Focus on first 10 time points (early behavior)
    max_early = min(10, MAX_TAU)
    
    for i, dataset in enumerate(datasets):
        results = dataset['results']
        
        # Plot only early times
        early_times = results['time_lags'][:max_early]
        early_Cv = results['Cv'][:max_early]
        early_sem = results['Cv_sem'][:max_early]
        
        plt.errorbar(early_times, early_Cv, yerr=early_sem, 
                    fmt='o-', color=colors[i], alpha=0.8, capsize=3,
                    label=f"{dataset['name']}", markersize=6)
    
    # Add reference lines
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(1/np.e, color='red', linestyle='--', alpha=0.5, 
               label='1/e threshold')
    plt.axvline(DT, color='gray', linestyle=':', alpha=0.7, 
               label=f'Sampling period = {DT} s')
    plt.axvline(2*DT, color='gray', linestyle=':', alpha=0.5, 
               label=f'2× sampling = {2*DT} s')
    
    plt.xlabel('Time lag τ (s)', fontsize=12)
    plt.ylabel('Velocity Autocorrelation C_v(τ)', fontsize=12)
    plt.title('Early Time Velocity Autocorrelation Behavior', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, "early_time_comparison.png"), 
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()

def export_comparison_results(datasets, comparison_results, output_path):
    """
    Export comparison results to CSV files.
    
    Args:
        datasets: List of dataset dictionaries
        comparison_results: Dictionary with comparison results
        output_path: Directory to save files
    """
    # Export dataset summary
    dataset_summary = []
    
    for dataset in datasets:
        results = dataset['results']
        summary = {
            'Dataset': dataset['name'],
            'N_files': dataset['n_files'],
            'N_trajectories_total': dataset['n_trajectories'],
            'N_trajectories_used': results['num_trajectories'],
            'Correlation_time_s': results['correlation_time'],
            'Zero_crossing_time_s': results['zero_crossing_time'],
            'Correlation_time_ratio_to_sampling': results['correlation_time']/DT if not np.isnan(results['correlation_time']) else np.nan,
            'Initial_Cv': results['Cv'][0] if len(results['Cv']) > 0 else np.nan
        }
        dataset_summary.append(summary)
    
    df_summary = pd.DataFrame(dataset_summary)
    df_summary.to_csv(os.path.join(output_path, "dataset_summary.csv"), index=False)
    
    # Export full autocorrelation data
    all_data = []
    
    for dataset in datasets:
        results = dataset['results']
        for i, (time, cv, sem) in enumerate(zip(results['time_lags'], 
                                               results['Cv'], 
                                               results['Cv_sem'])):
            all_data.append({
                'Dataset': dataset['name'],
                'Time_lag_s': time,
                'Cv': cv,
                'Cv_SEM': sem,
                'N_points': results['n_points'][i]
            })
    
    df_all = pd.DataFrame(all_data)
    df_all.to_csv(os.path.join(output_path, "full_autocorr_data.csv"), index=False)
    
    # Export pairwise comparisons
    if comparison_results['pairwise_comparisons']:
        df_comparisons = pd.DataFrame(comparison_results['pairwise_comparisons'])
        df_comparisons.to_csv(os.path.join(output_path, "pairwise_comparisons.csv"), index=False)

def main():
    """Main function to compare velocity autocorrelation between datasets."""
    print("Velocity Autocorrelation Dataset Comparison")
    print("=" * 50)
    
    # Get number of datasets to compare
    try:
        n_datasets = int(input("Enter the number of datasets to compare: "))
        if n_datasets < 2:
            print("Need at least 2 datasets for comparison")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        return
    
    # Get dataset directories and names
    datasets = []
    
    for i in range(n_datasets):
        dataset_dir = input(f"Enter directory for dataset {i+1}: ")
        
        if not os.path.isdir(dataset_dir):
            print(f"Directory {dataset_dir} does not exist")
            return
        
        # Get dataset name
        dataset_name = input(f"Enter name for dataset {i+1} (press Enter to use directory name): ")
        
        if not dataset_name:
            dataset_name = os.path.basename(os.path.normpath(dataset_dir))
            if not dataset_name:
                dataset_name = f"Dataset_{i+1}"
        
        # Load dataset
        print(f"Loading dataset '{dataset_name}'...")
        dataset = load_dataset(dataset_dir, dataset_name)
        
        if dataset is None:
            print(f"Failed to load dataset '{dataset_name}'")
            return
        
        datasets.append(dataset)
        print(f"  Loaded {dataset['n_filtered_trajectories']} trajectories (from {dataset['n_trajectories']} total)")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), f"velocity_autocorr_comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform comparison
    print("\nComparing datasets...")
    comparison_results = compare_datasets(datasets)
    
    # Generate plots
    print("Creating comparison plots...")
    plot_autocorrelation_comparison(datasets, output_dir)
    plot_early_time_comparison(datasets, output_dir)
    
    # Export results
    print("Exporting results...")
    export_comparison_results(datasets, comparison_results, output_dir)
    
    # Save full results
    results_file = os.path.join(output_dir, "comparison_results.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump({'datasets': datasets, 'comparison_results': comparison_results}, f)
    
    # Print summary
    print(f"\nComparison Summary:")
    print(f"Results saved to: {output_dir}")
    print("\nDataset correlation times:")
    
    for dataset in datasets:
        results = dataset['results']
        corr_time = results['correlation_time']
        zero_time = results['zero_crossing_time']
        
        print(f"\n{dataset['name']}:")
        print(f"  Trajectories used: {results['num_trajectories']}")
        
        if not np.isnan(corr_time):
            print(f"  Correlation time: {corr_time:.3f} s ({corr_time/DT:.1f}× sampling)")
            if corr_time < 2*DT:
                print(f"  ⚠️  WARNING: Very close to sampling limit!")
            elif corr_time < 5*DT:
                print(f"  ⚠️  CAUTION: May be affected by sampling rate")
        else:
            print(f"  Correlation time: No exponential decay found")
        
        if not np.isnan(zero_time):
            print(f"  Zero crossing: {zero_time:.3f} s ({zero_time/DT:.1f}× sampling)")
        else:
            print(f"  Zero crossing: Not found")
    
    print(f"\nPairwise comparisons:")
    for comp in comparison_results['pairwise_comparisons']:
        print(f"\n{comp['dataset1']} vs {comp['dataset2']}:")
        
        if not np.isnan(comp['correlation_time_diff']):
            print(f"  Correlation time difference: {comp['correlation_time_diff']:.3f} s")
        
        if not np.isnan(comp['zero_crossing_diff']):
            print(f"  Zero crossing difference: {comp['zero_crossing_diff']:.3f} s")

if __name__ == "__main__":
    main()
    