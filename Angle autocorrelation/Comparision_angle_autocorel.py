# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 13:24:44 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_angle_autocorrelation.py

This script compares angle autocorrelation functions between different experimental conditions.
It provides statistical analysis and visualization of directional persistence differences.

Input:
- Multiple directories containing analyzed trajectory files (.pkl files)

Output:
- Statistical comparison of angle autocorrelations
- Comparative plots showing correlation curves and crossing times
- Effect size analysis and confidence intervals
- CSV exports of comparison data

Usage:
python compare_angle_autocorrelation.py
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
import warnings

# Global parameters (modify these as needed)
# =====================================
# Time interval between frames in seconds
DT = 0.1
# Maximum lag for angle correlation analysis
ANGLE_PLOT_CUTOFF = 50
# Minimum trajectory length to consider
L_CUTOFF = 15

# Statistical parameters
ALPHA = 0.05  # Significance level
N_BOOTSTRAP = 1000  # Number of bootstrap iterations
CONFIDENCE_LEVEL = 0.95  # Confidence interval level

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
    t_cross = calculate_crossing_time(mean_cos_angle, dt)
    
    return {
        'mean_cos_angle': mean_cos_angle,
        'sem_cos_angle': sem_cos_angle,
        't_cross': t_cross,
        'time_lags': np.arange(1, angle_plot_cutoff + 1) * dt,
        'num_trajectories': len(filtered_trajectories),
        'raw_correlations': cos_angle_t_total  # Store for bootstrap analysis
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
    index_temp = np.where(mean_cos_angle < 0)[0]
    
    if len(index_temp) > 0:
        x2 = index_temp[0] + 1  # +1 because Python is 0-indexed
        
        if x2 >= 2:
            x1 = x2 - 1
            y2 = mean_cos_angle[x2-1]
            y1 = mean_cos_angle[x1-1]
            if (y2 - y1) != 0:  # Avoid division by zero
                t_cross = (x1*y2 - x2*y1)*dt / (y2-y1)
        else:
            y2 = mean_cos_angle[1]
            y1 = mean_cos_angle[0]
            if (y2 - y1) != 0:  # Avoid division by zero
                t_cross = (y2 - 2*y1)*dt / (y2-y1)
    
    return t_cross

def bootstrap_crossing_time(raw_correlations, dt=DT, n_bootstrap=N_BOOTSTRAP):
    """
    Calculate bootstrap confidence intervals for crossing time.
    
    Args:
        raw_correlations: List of correlation values for each time lag
        dt: Time step
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        Dictionary with bootstrap results
    """
    crossing_times = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample the correlations
        bootstrap_means = []
        
        for i, corr_values in enumerate(raw_correlations):
            if len(corr_values) > 0:
                # Sample with replacement
                bootstrap_sample = np.random.choice(corr_values, size=len(corr_values), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            else:
                bootstrap_means.append(np.nan)
        
        # Calculate crossing time for this bootstrap sample
        t_cross = calculate_crossing_time(np.array(bootstrap_means), dt)
        if not np.isnan(t_cross):
            crossing_times.append(t_cross)
    
    if crossing_times:
        alpha = 1 - CONFIDENCE_LEVEL
        lower_ci = np.percentile(crossing_times, alpha/2 * 100)
        upper_ci = np.percentile(crossing_times, (1 - alpha/2) * 100)
        
        return {
            'crossing_times': crossing_times,
            'mean': np.mean(crossing_times),
            'std': np.std(crossing_times),
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'n_successful': len(crossing_times)
        }
    else:
        return {
            'crossing_times': [],
            'mean': np.nan,
            'std': np.nan,
            'lower_ci': np.nan,
            'upper_ci': np.nan,
            'n_successful': 0
        }

def load_condition_data(condition_dir, condition_name):
    """
    Load all trajectory files for a given condition.
    
    Args:
        condition_dir: Directory containing analyzed trajectory files
        condition_name: Name of the condition
        
    Returns:
        Dictionary with condition data
    """
    # Get list of analyzed files
    file_paths = glob.glob(os.path.join(condition_dir, "analyzed_*.pkl"))
    
    if not file_paths:
        print(f"No analyzed trajectory files found in {condition_dir}")
        return None
    
    print(f"Loading {len(file_paths)} files for condition '{condition_name}'")
    
    # Pool all trajectories from all files
    all_trajectories = []
    file_info = []
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        analyzed_data = load_analyzed_data(file_path)
        
        if analyzed_data is None:
            print(f"Skipping {filename} due to loading errors")
            continue
        
        trajectories = analyzed_data.get('trajectories', [])
        all_trajectories.extend(trajectories)
        
        file_info.append({
            'filename': filename,
            'n_trajectories': len(trajectories)
        })
    
    if not all_trajectories:
        print(f"No valid trajectories found for condition '{condition_name}'")
        return None
    
    # Calculate angle autocorrelation for the condition
    angle_results = calculate_angle_autocorrelation(all_trajectories)
    
    if angle_results is None:
        print(f"Failed to calculate angle autocorrelation for condition '{condition_name}'")
        return None
    
    # Calculate bootstrap confidence intervals
    bootstrap_results = bootstrap_crossing_time(angle_results['raw_correlations'])
    
    return {
        'name': condition_name,
        'angle_results': angle_results,
        'bootstrap_results': bootstrap_results,
        'file_info': file_info,
        'total_trajectories': len(all_trajectories)
    }

def compare_crossing_times(conditions):
    """
    Statistical comparison of crossing times between conditions.
    
    Args:
        conditions: List of condition dictionaries
        
    Returns:
        Dictionary with comparison results
    """
    comparison_results = []
    
    # Get all pairwise combinations
    condition_pairs = list(combinations(range(len(conditions)), 2))
    
    for i, j in condition_pairs:
        cond1 = conditions[i]
        cond2 = conditions[j]
        
        # Get bootstrap crossing times
        times1 = cond1['bootstrap_results']['crossing_times']
        times2 = cond2['bootstrap_results']['crossing_times']
        
        if len(times1) == 0 or len(times2) == 0:
            print(f"Insufficient data for comparison between {cond1['name']} and {cond2['name']}")
            continue
        
        # Perform statistical tests
        try:
            # Mann-Whitney U test
            u_stat, p_mw = stats.mannwhitneyu(times1, times2, alternative='two-sided')
            
            # Calculate effect size (Cliff's delta)
            cliffs_delta = calculate_cliffs_delta(times1, times2)
            
            # Calculate difference in means
            mean_diff = np.mean(times1) - np.mean(times2)
            
            comparison = {
                'condition1': cond1['name'],
                'condition2': cond2['name'],
                'n1': len(times1),
                'n2': len(times2),
                'mean1': np.mean(times1),
                'mean2': np.mean(times2),
                'median1': np.median(times1),
                'median2': np.median(times2),
                'std1': np.std(times1),
                'std2': np.std(times2),
                'mean_difference': mean_diff,
                'mann_whitney_u': u_stat,
                'mann_whitney_p': p_mw,
                'cliffs_delta': cliffs_delta,
                'effect_size_interpretation': interpret_cliffs_delta(cliffs_delta),
                'significant': p_mw < ALPHA
            }
            
            comparison_results.append(comparison)
            
        except Exception as e:
            print(f"Error comparing {cond1['name']} and {cond2['name']}: {e}")
    
    return comparison_results

def calculate_cliffs_delta(group1, group2):
    """Calculate Cliff's delta effect size."""
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

def interpret_cliffs_delta(delta):
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

def plot_angle_correlations(conditions, output_path):
    """
    Plot angle autocorrelation curves for all conditions.
    
    Args:
        conditions: List of condition dictionaries
        output_path: Directory to save the plot
    """
    plt.figure(figsize=FIGURE_SIZE)
    
    # Use different colors for each condition
    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
    
    for i, condition in enumerate(conditions):
        angle_results = condition['angle_results']
        
        # Plot with error bars
        plt.errorbar(
            angle_results['time_lags'], 
            angle_results['mean_cos_angle'], 
            yerr=angle_results['sem_cos_angle'],
            fmt='.-', 
            color=colors[i],
            label=f"{condition['name']} (n={condition['total_trajectories']})",
            alpha=0.8
        )
        
        # Mark crossing time if available
        if not np.isnan(angle_results['t_cross']):
            plt.axvline(angle_results['t_cross'], color=colors[i], linestyle='--', alpha=0.5)
    
    plt.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    plt.xlabel('Time lag (s)', fontsize=12)
    plt.ylabel(r'$\langle \cos \theta(\tau) \rangle$', fontsize=12)
    plt.title('Angle Autocorrelation Comparison', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, "angle_autocorrelation_comparison.png"), dpi=FIGURE_DPI)
    plt.close()

def plot_crossing_times_comparison(conditions, comparison_results, output_path):
    """
    Plot comparison of crossing times between conditions.
    
    Args:
        conditions: List of condition dictionaries
        comparison_results: List of comparison dictionaries
        output_path: Directory to save the plots
    """
    # Plot 1: Bar plot with confidence intervals
    plt.figure(figsize=FIGURE_SIZE)
    
    condition_names = [cond['name'] for cond in conditions]
    crossing_times = [cond['bootstrap_results']['mean'] for cond in conditions]
    lower_errors = [cond['bootstrap_results']['mean'] - cond['bootstrap_results']['lower_ci'] for cond in conditions]
    upper_errors = [cond['bootstrap_results']['upper_ci'] - cond['bootstrap_results']['mean'] for cond in conditions]
    
    # Handle NaN values
    valid_indices = [i for i, ct in enumerate(crossing_times) if not np.isnan(ct)]
    
    if valid_indices:
        valid_names = [condition_names[i] for i in valid_indices]
        valid_times = [crossing_times[i] for i in valid_indices]
        valid_lower = [lower_errors[i] for i in valid_indices]
        valid_upper = [upper_errors[i] for i in valid_indices]
        
        plt.bar(valid_names, valid_times, 
                yerr=[valid_lower, valid_upper],
                capsize=5, alpha=0.7)
        
        plt.ylabel('Crossing time (s)')
        plt.title('Angle Autocorrelation Crossing Times')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_path, "crossing_times_comparison.png"), dpi=FIGURE_DPI)
        plt.close()
    
    # Plot 2: Distribution of bootstrap crossing times
    plt.figure(figsize=FIGURE_SIZE)
    
    for i, condition in enumerate(conditions):
        bootstrap_times = condition['bootstrap_results']['crossing_times']
        if bootstrap_times:
            plt.hist(bootstrap_times, bins=20, alpha=0.5, 
                    label=f"{condition['name']} (n={len(bootstrap_times)})")
    
    plt.xlabel('Crossing time (s)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Bootstrap Crossing Times')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, "crossing_times_distributions.png"), dpi=FIGURE_DPI)
    plt.close()

def create_diagnostic_plots(conditions, output_path):
    """
    Create diagnostic plots to visualize what's being measured.
    
    Args:
        conditions: List of condition dictionaries
        output_path: Directory to save the plots
    """
    # Select first condition for diagnostic plots
    if not conditions:
        return
    
    condition = conditions[0]
    
    # Load a few example trajectories for visualization
    file_paths = glob.glob(os.path.join("*", "analyzed_*.pkl"))
    if not file_paths:
        return
    
    # Load first file
    analyzed_data = load_analyzed_data(file_paths[0])
    if analyzed_data is None:
        return
    
    trajectories = analyzed_data.get('trajectories', [])
    if not trajectories:
        return
    
    # Filter trajectories by length
    filtered_trajectories = [traj for traj in trajectories if len(traj['x']) > L_CUTOFF]
    
    # Select up to 3 trajectories for visualization
    n_examples = min(3, len(filtered_trajectories))
    example_trajectories = filtered_trajectories[:n_examples]
    
    # Create diagnostic figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Example trajectories
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, n_examples))
    
    for i, traj in enumerate(example_trajectories):
        x, y = traj['x'], traj['y']
        ax.plot(x, y, '-', color=colors[i], alpha=0.7, linewidth=1)
        ax.plot(x[0], y[0], 'o', color=colors[i], markersize=6)  # Start point
        ax.plot(x[-1], y[-1], 's', color=colors[i], markersize=6)  # End point
    
    ax.set_xlabel('X position (μm)')
    ax.set_ylabel('Y position (μm)')
    ax.set_title('Example Trajectories')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 2: Displacement vectors for one trajectory
    ax = axes[0, 1]
    if example_trajectories:
        traj = example_trajectories[0]
        x, y = traj['x'], traj['y']
        dx = np.diff(x)
        dy = np.diff(y)
        
        # Plot displacement vectors as arrows
        for i in range(min(20, len(dx))):  # Show first 20 steps
            ax.arrow(x[i], y[i], dx[i], dy[i], 
                    head_width=0.1, head_length=0.05, 
                    fc='red', ec='red', alpha=0.6)
        
        ax.plot(x, y, 'k-', alpha=0.3, linewidth=1)
        ax.set_xlabel('X position (μm)')
        ax.set_ylabel('Y position (μm)')
        ax.set_title('Displacement Vectors (First 20 Steps)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Plot 3: Angle between consecutive displacements
    ax = axes[1, 0]
    if example_trajectories:
        traj = example_trajectories[0]
        x, y = traj['x'], traj['y']
        dx = np.diff(x)
        dy = np.diff(y)
        
        # Calculate angles between consecutive displacement vectors
        angles = []
        for i in range(len(dx) - 1):
            v1 = np.array([dx[i], dy[i]])
            v2 = np.array([dx[i+1], dy[i+1]])
            
            # Calculate angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = max(min(cos_angle, 1.0), -1.0)  # Ensure valid range
            angle = np.arccos(cos_angle) * 180 / np.pi  # Convert to degrees
            angles.append(angle)
        
        ax.plot(angles, 'o-', alpha=0.7)
        ax.set_xlabel('Step number')
        ax.set_ylabel('Angle between consecutive steps (degrees)')
        ax.set_title('Angular Changes in Trajectory')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Distribution of angles for the condition
    ax = axes[1, 1]
    angle_results = condition['angle_results']
    
    # Show the autocorrelation decay
    ax.plot(angle_results['time_lags'], angle_results['mean_cos_angle'], 'bo-', alpha=0.7)
    ax.fill_between(angle_results['time_lags'], 
                   angle_results['mean_cos_angle'] - angle_results['sem_cos_angle'],
                   angle_results['mean_cos_angle'] + angle_results['sem_cos_angle'],
                   alpha=0.3)
    
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    if not np.isnan(angle_results['t_cross']):
        ax.axvline(angle_results['t_cross'], color='red', linestyle='--', alpha=0.5)
        ax.text(angle_results['t_cross'], 0.5, f't_cross = {angle_results["t_cross"]:.2f} s', 
                rotation=90, ha='right')
    
    ax.set_xlabel('Time lag (s)')
    ax.set_ylabel(r'$\langle \cos \theta(\tau) \rangle$')
    ax.set_title(f'Angle Autocorrelation - {condition["name"]}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "angle_correlation_diagnostics.png"), dpi=FIGURE_DPI)
    plt.close()

def export_results(conditions, comparison_results, output_path):
    """
    Export analysis results to CSV files.
    
    Args:
        conditions: List of condition dictionaries
        comparison_results: List of comparison dictionaries
        output_path: Directory to save the CSV files
    """
    # Export condition summary
    condition_summary = []
    
    for condition in conditions:
        angle_results = condition['angle_results']
        bootstrap_results = condition['bootstrap_results']
        
        summary = {
            'Condition': condition['name'],
            'Total_trajectories': condition['total_trajectories'],
            'Crossing_time': angle_results['t_cross'],
            'Bootstrap_mean_crossing_time': bootstrap_results['mean'],
            'Bootstrap_std_crossing_time': bootstrap_results['std'],
            'Bootstrap_lower_CI': bootstrap_results['lower_ci'],
            'Bootstrap_upper_CI': bootstrap_results['upper_ci'],
            'Bootstrap_n_successful': bootstrap_results['n_successful']
        }
        
        condition_summary.append(summary)
    
    df = pd.DataFrame(condition_summary)
    df.to_csv(os.path.join(output_path, "condition_summary.csv"), index=False)
    
    # Export pairwise comparisons
    if comparison_results:
        df = pd.DataFrame(comparison_results)
        df.to_csv(os.path.join(output_path, "pairwise_comparisons.csv"), index=False)
    
    # Export detailed angle correlation data for each condition
    for condition in conditions:
        angle_results = condition['angle_results']
        
        df = pd.DataFrame({
            'Time_lag_s': angle_results['time_lags'],
            'Mean_cos_angle': angle_results['mean_cos_angle'],
            'SEM_cos_angle': angle_results['sem_cos_angle']
        })
        
        filename = f"angle_correlation_{condition['name'].replace(' ', '_')}.csv"
        df.to_csv(os.path.join(output_path, filename), index=False)

def main():
    """Main function to compare angle autocorrelations between conditions."""
    print("Angle Autocorrelation Comparison")
    print("===============================")
    
    # Get number of conditions to compare
    try:
        n_conditions = int(input("Enter the number of conditions to compare: "))
        if n_conditions < 2:
            print("Need at least 2 conditions for comparison")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        return
    
    # Get condition directories
    condition_dirs = []
    condition_names = []
    
    for i in range(n_conditions):
        condition_dir = input(f"Enter directory for condition {i+1}: ")
        
        if not os.path.isdir(condition_dir):
            print(f"Directory {condition_dir} does not exist")
            return
        
        # Get condition name (default to directory name)
        condition_name = input(f"Enter name for condition {i+1} (press Enter to use directory name): ")
        
        if not condition_name:
            condition_name = os.path.basename(os.path.normpath(condition_dir))
            if not condition_name:
                condition_name = f"Condition_{i+1}"
        
        condition_dirs.append(condition_dir)
        condition_names.append(condition_name)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), f"angle_autocorr_comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load condition data
    print("\nLoading condition data...")
    conditions = []
    
    for i, (dir_path, name) in enumerate(zip(condition_dirs, condition_names)):
        print(f"Loading condition {i+1}/{n_conditions}: {name}")
        
        condition_data = load_condition_data(dir_path, name)
        
        if condition_data is None:
            print(f"Failed to load condition {name}")
            continue
        
        conditions.append(condition_data)
        print(f"  Loaded {condition_data['total_trajectories']} trajectories")
        
        # Print crossing time info
        angle_results = condition_data['angle_results']
        if not np.isnan(angle_results['t_cross']):
            print(f"  Crossing time: {angle_results['t_cross']:.3f} s")
        else:
            print(f"  No crossing time found")
    
    if len(conditions) < 2:
        print("Need at least 2 valid conditions for comparison")
        return
    
    # Compare conditions
    print("\nComparing conditions...")
    comparison_results = compare_crossing_times(conditions)
    
    # Generate plots
    print("Generating comparison plots...")
    plot_angle_correlations(conditions, output_dir)
    plot_crossing_times_comparison(conditions, comparison_results, output_dir)
    create_diagnostic_plots(conditions, output_dir)
    
    # Export results
    print("Exporting comparison results...")
    export_results(conditions, comparison_results, output_dir)
    
    # Save comparison results
    output_file = os.path.join(output_dir, "comparison_results.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump({
            'conditions': conditions,
            'comparison_results': comparison_results
        }, f)
    
    print(f"Comparison results saved to {output_file}")
    print(f"All outputs saved to {output_dir}")
    
    # Print summary of results
    print("\nComparison Summary:")
    print("==================")
    
    # Print condition summaries
    for condition in conditions:
        angle_results = condition['angle_results']
        bootstrap_results = condition['bootstrap_results']
        
        print(f"\n{condition['name']}:")
        print(f"  Total trajectories: {condition['total_trajectories']}")
        
        if not np.isnan(angle_results['t_cross']):
            print(f"  Crossing time: {angle_results['t_cross']:.3f} s")
        else:
            print(f"  No crossing time found")
        
        if not np.isnan(bootstrap_results['mean']):
            print(f"  Bootstrap crossing time: {bootstrap_results['mean']:.3f} ± {bootstrap_results['std']:.3f} s")
            print(f"  95% CI: [{bootstrap_results['lower_ci']:.3f}, {bootstrap_results['upper_ci']:.3f}] s")
        
        print(f"  Bootstrap success rate: {bootstrap_results['n_successful']}/{N_BOOTSTRAP}")
    
    # Print comparison results
    if comparison_results:
        print("\nPairwise Comparisons:")
        print("====================")
        
        for comp in comparison_results:
            sig_text = "Significant" if comp['significant'] else "Not significant"
            effect_text = comp['effect_size_interpretation']
            
            print(f"\n{comp['condition1']} vs {comp['condition2']}:")
            print(f"  Difference in crossing times: {comp['mean_difference']:.3f} s")
            print(f"  Statistical significance: {sig_text} (p = {comp['mann_whitney_p']:.4e})")
            print(f"  Effect size: {effect_text} (Cliff's delta = {comp['cliffs_delta']:.3f})")
            
            if comp['significant'] and abs(comp['cliffs_delta']) < 0.33:
                print("  Note: Statistically significant but small effect size")
            elif not comp['significant'] and abs(comp['cliffs_delta']) > 0.33:
                print("  Note: Not statistically significant but moderate to large effect size")

if __name__ == "__main__":
    main()