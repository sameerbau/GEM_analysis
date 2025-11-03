# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 15:15:12 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
angle_resolution_analyzer.py

Advanced analysis to handle temporal resolution issues in angle autocorrelation.
Provides alternative metrics when crossing times approach the recording frequency.

Input:
- Analyzed trajectory data (.pkl files)

Output:
- Alternative directional persistence metrics
- Persistence length analysis
- Directional change frequency analysis
- Multi-scale temporal analysis

Mean turning angles - How much do particles turn between steps?
Persistence ratios - What fraction of steps maintain direction?
Trajectory straightness - Overall path efficiency
Step size distributions - Are there differences in movement magnitude?

Usage:
python angle_resolution_analyzer.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
from scipy import stats
from datetime import datetime

# Global parameters
# =====================================
DT = 0.1  # Recording frequency (seconds)
ANGLE_PLOT_CUTOFF = 50
L_CUTOFF = 15

# Alternative analysis parameters
MIN_STEP_SIZE = 0.01  # Minimum step size to consider (μm)
PERSISTENCE_THRESHOLDS = [30, 45, 60, 90]  # Angle thresholds for persistence analysis (degrees)
MULTI_SCALE_LAGS = [1, 2, 3, 5, 10, 15, 20]  # Different time lags to analyze
# =====================================

def load_analyzed_data(file_path):
    """Load analyzed trajectory data from pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def calculate_turning_angles(x, y, min_step_size=MIN_STEP_SIZE):
    """
    Calculate turning angles between consecutive displacement vectors.
    
    Args:
        x, y: Trajectory coordinates
        min_step_size: Minimum displacement to consider
        
    Returns:
        Array of turning angles in degrees
    """
    # Calculate displacement vectors
    dx = np.diff(x)
    dy = np.diff(y)
    
    # Filter out very small displacements (noise)
    step_sizes = np.sqrt(dx**2 + dy**2)
    valid_steps = step_sizes >= min_step_size
    
    if np.sum(valid_steps) < 2:
        return np.array([])
    
    # Get valid displacement vectors
    dx_valid = dx[valid_steps]
    dy_valid = dy[valid_steps]
    
    turning_angles = []
    
    for i in range(len(dx_valid) - 1):
        v1 = np.array([dx_valid[i], dy_valid[i]])
        v2 = np.array([dx_valid[i+1], dy_valid[i+1]])
        
        # Calculate angle between consecutive vectors
        dot_product = np.dot(v1, v2)
        magnitudes = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if magnitudes > 0:
            cos_angle = dot_product / magnitudes
            cos_angle = max(min(cos_angle, 1.0), -1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            turning_angles.append(angle)
    
    return np.array(turning_angles)

def calculate_persistence_metrics(trajectories):
    """
    Calculate alternative persistence metrics that don't rely on crossing times.
    
    Args:
        trajectories: List of trajectory dictionaries
        
    Returns:
        Dictionary with various persistence metrics
    """
    results = {
        'mean_turning_angle': [],
        'median_turning_angle': [],
        'persistence_ratios': {threshold: [] for threshold in PERSISTENCE_THRESHOLDS},
        'directional_autocorr_at_lags': {lag: [] for lag in MULTI_SCALE_LAGS},
        'step_size_distribution': [],
        'trajectory_straightness': [],
        'mean_squared_angular_displacement': []
    }
    
    # Filter trajectories by length
    filtered_trajectories = [traj for traj in trajectories if len(traj['x']) > L_CUTOFF]
    
    if not filtered_trajectories:
        return results
    
    for traj in filtered_trajectories:
        x, y = traj['x'], traj['y']
        
        # Calculate turning angles
        turning_angles = calculate_turning_angles(x, y)
        
        if len(turning_angles) > 0:
            results['mean_turning_angle'].append(np.mean(turning_angles))
            results['median_turning_angle'].append(np.median(turning_angles))
            
            # Calculate persistence ratios (fraction of turns below threshold)
            for threshold in PERSISTENCE_THRESHOLDS:
                persistent_fraction = np.sum(turning_angles < threshold) / len(turning_angles)
                results['persistence_ratios'][threshold].append(persistent_fraction)
            
            # Calculate mean squared angular displacement
            if len(turning_angles) > 1:
                msad = np.mean(turning_angles**2)
                results['mean_squared_angular_displacement'].append(msad)
        
        # Calculate directional autocorrelation at specific lags
        dx = np.diff(x)
        dy = np.diff(y)
        
        for lag in MULTI_SCALE_LAGS:
            if len(dx) > lag:
                correlations = []
                for k in range(len(dx) - lag):
                    # Calculate correlation at this lag
                    dot_product = dx[k] * dx[k+lag] + dy[k] * dy[k+lag]
                    mag1 = np.sqrt(dx[k]**2 + dy[k]**2)
                    mag2 = np.sqrt(dx[k+lag]**2 + dy[k+lag]**2)
                    
                    if mag1 > 0 and mag2 > 0:
                        cos_angle = dot_product / (mag1 * mag2)
                        cos_angle = max(min(cos_angle, 1.0), -1.0)
                        correlations.append(cos_angle)
                
                if correlations:
                    results['directional_autocorr_at_lags'][lag].append(np.mean(correlations))
        
        # Calculate step size distribution
        step_sizes = np.sqrt(dx**2 + dy**2)
        results['step_size_distribution'].extend(step_sizes[step_sizes >= MIN_STEP_SIZE])
        
        # Calculate trajectory straightness (end-to-end distance / path length)
        end_to_end = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
        path_length = np.sum(step_sizes)
        if path_length > 0:
            straightness = end_to_end / path_length
            results['trajectory_straightness'].append(straightness)
    
    return results

def analyze_temporal_resolution_effects(trajectories, dt=DT):
    """
    Analyze how temporal resolution affects directional measurements.
    
    Args:
        trajectories: List of trajectory dictionaries
        dt: Time step
        
    Returns:
        Dictionary with resolution analysis results
    """
    results = {
        'frame_to_frame_correlations': [],
        'two_frame_correlations': [],
        'three_frame_correlations': [],
        'step_size_vs_correlation': {'step_sizes': [], 'correlations': []},
        'velocity_persistence': []
    }
    
    filtered_trajectories = [traj for traj in trajectories if len(traj['x']) > L_CUTOFF]
    
    for traj in filtered_trajectories:
        x, y = traj['x'], traj['y']
        dx = np.diff(x)
        dy = np.diff(y)
        
        # Frame-to-frame (lag=1) correlations
        for k in range(len(dx) - 1):
            dot_product = dx[k] * dx[k+1] + dy[k] * dy[k+1]
            mag1 = np.sqrt(dx[k]**2 + dy[k]**2)
            mag2 = np.sqrt(dx[k+1]**2 + dy[k+1]**2)
            
            if mag1 > MIN_STEP_SIZE and mag2 > MIN_STEP_SIZE:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(min(cos_angle, 1.0), -1.0)
                results['frame_to_frame_correlations'].append(cos_angle)
                
                # Store step size vs correlation relationship
                avg_step = (mag1 + mag2) / 2
                results['step_size_vs_correlation']['step_sizes'].append(avg_step)
                results['step_size_vs_correlation']['correlations'].append(cos_angle)
        
        # Two-frame skip (lag=2) correlations
        for k in range(len(dx) - 2):
            dot_product = dx[k] * dx[k+2] + dy[k] * dy[k+2]
            mag1 = np.sqrt(dx[k]**2 + dy[k]**2)
            mag2 = np.sqrt(dx[k+2]**2 + dy[k+2]**2)
            
            if mag1 > MIN_STEP_SIZE and mag2 > MIN_STEP_SIZE:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(min(cos_angle, 1.0), -1.0)
                results['two_frame_correlations'].append(cos_angle)
        
        # Three-frame skip (lag=3) correlations
        for k in range(len(dx) - 3):
            dot_product = dx[k] * dx[k+3] + dy[k] * dy[k+3]
            mag1 = np.sqrt(dx[k]**2 + dy[k]**2)
            mag2 = np.sqrt(dx[k+3]**2 + dy[k+3]**2)
            
            if mag1 > MIN_STEP_SIZE and mag2 > MIN_STEP_SIZE:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(min(cos_angle, 1.0), -1.0)
                results['three_frame_correlations'].append(cos_angle)
        
        # Velocity persistence (how consistent is speed)
        speeds = np.sqrt(dx**2 + dy**2)
        if len(speeds) > 1:
            speed_autocorr = np.corrcoef(speeds[:-1], speeds[1:])[0, 1]
            if not np.isnan(speed_autocorr):
                results['velocity_persistence'].append(speed_autocorr)
    
    return results

def compare_conditions_advanced(condition_data_list, condition_names):
    """
    Advanced comparison between conditions using alternative metrics.
    
    Args:
        condition_data_list: List of condition data dictionaries
        condition_names: List of condition names
        
    Returns:
        Dictionary with comparison results
    """
    comparison_results = {}
    
    # Compare mean turning angles
    turning_angles = [data['mean_turning_angle'] for data in condition_data_list]
    comparison_results['turning_angles'] = compare_metric_lists(
        turning_angles, condition_names, "Mean Turning Angle (degrees)"
    )
    
    # Compare persistence ratios
    for threshold in PERSISTENCE_THRESHOLDS:
        persistence_ratios = [data['persistence_ratios'][threshold] for data in condition_data_list]
        comparison_results[f'persistence_{threshold}deg'] = compare_metric_lists(
            persistence_ratios, condition_names, f"Persistence Ratio (<{threshold}°)"
        )
    
    # Compare directional autocorrelations at specific lags
    for lag in MULTI_SCALE_LAGS:
        autocorrs = [data['directional_autocorr_at_lags'][lag] for data in condition_data_list]
        comparison_results[f'autocorr_lag_{lag}'] = compare_metric_lists(
            autocorrs, condition_names, f"Directional Autocorr (lag={lag})"
        )
    
    # Compare trajectory straightness
    straightness = [data['trajectory_straightness'] for data in condition_data_list]
    comparison_results['straightness'] = compare_metric_lists(
        straightness, condition_names, "Trajectory Straightness"
    )
    
    return comparison_results

def compare_metric_lists(metric_lists, condition_names, metric_name):
    """
    Compare a metric between conditions.
    
    Args:
        metric_lists: List of lists containing metric values for each condition
        condition_names: Names of conditions
        metric_name: Name of the metric being compared
        
    Returns:
        Dictionary with comparison results
    """
    # Filter out empty lists
    valid_data = [(values, name) for values, name in zip(metric_lists, condition_names) if len(values) > 0]
    
    if len(valid_data) < 2:
        return {'error': 'Insufficient data for comparison'}
    
    results = {
        'metric_name': metric_name,
        'condition_summaries': [],
        'pairwise_comparisons': []
    }
    
    # Calculate summaries for each condition
    for values, name in valid_data:
        summary = {
            'condition': name,
            'n': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'sem': stats.sem(values)
        }
        results['condition_summaries'].append(summary)
    
    # Pairwise comparisons
    for i in range(len(valid_data)):
        for j in range(i+1, len(valid_data)):
            values1, name1 = valid_data[i]
            values2, name2 = valid_data[j]
            
            # Statistical test
            try:
                u_stat, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                
                # Effect size
                cliffs_delta = calculate_cliffs_delta(values1, values2)
                
                comparison = {
                    'condition1': name1,
                    'condition2': name2,
                    'mean_diff': np.mean(values1) - np.mean(values2),
                    'median_diff': np.median(values1) - np.median(values2),
                    'p_value': p_value,
                    'cliffs_delta': cliffs_delta,
                    'effect_size': interpret_cliffs_delta(cliffs_delta),
                    'significant': p_value < 0.05
                }
                
                results['pairwise_comparisons'].append(comparison)
                
            except Exception as e:
                print(f"Error in comparison {name1} vs {name2}: {e}")
    
    return results

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

def create_resolution_diagnostic_plots(resolution_data_list, condition_names, output_path):
    """
    Create plots to diagnose temporal resolution issues.
    
    Args:
        resolution_data_list: List of resolution analysis results
        condition_names: List of condition names
        output_path: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Frame-to-frame correlations
    ax = axes[0, 0]
    for i, (data, name) in enumerate(zip(resolution_data_list, condition_names)):
        correlations = data['frame_to_frame_correlations']
        if correlations:
            ax.hist(correlations, bins=30, alpha=0.5, label=f'{name} (n={len(correlations)})', density=True)
    
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Frame-to-frame directional correlation')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Frame-to-Frame Correlations\n(Shows resolution effects)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Correlation vs lag
    ax = axes[0, 1]
    lags = [1, 2, 3]
    
    for i, (data, name) in enumerate(zip(resolution_data_list, condition_names)):
        lag_means = []
        lag_sems = []
        
        for lag in lags:
            if lag == 1:
                correlations = data['frame_to_frame_correlations']
            elif lag == 2:
                correlations = data['two_frame_correlations']
            elif lag == 3:
                correlations = data['three_frame_correlations']
            
            if correlations:
                lag_means.append(np.mean(correlations))
                lag_sems.append(stats.sem(correlations))
            else:
                lag_means.append(np.nan)
                lag_sems.append(np.nan)
        
        ax.errorbar(lags, lag_means, yerr=lag_sems, marker='o', label=name, capsize=5)
    
    ax.set_xlabel('Frame lag')
    ax.set_ylabel('Mean directional correlation')
    ax.set_title('Correlation Decay with Frame Lag')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # Plot 3: Step size vs correlation
    ax = axes[1, 0]
    for i, (data, name) in enumerate(zip(resolution_data_list, condition_names)):
        step_sizes = data['step_size_vs_correlation']['step_sizes']
        correlations = data['step_size_vs_correlation']['correlations']
        
        if step_sizes and correlations:
            # Bin by step size and calculate mean correlation
            step_bins = np.linspace(0, np.percentile(step_sizes, 95), 10)
            bin_centers = []
            bin_correlations = []
            
            for j in range(len(step_bins)-1):
                mask = (np.array(step_sizes) >= step_bins[j]) & (np.array(step_sizes) < step_bins[j+1])
                if np.sum(mask) > 0:
                    bin_centers.append((step_bins[j] + step_bins[j+1]) / 2)
                    bin_correlations.append(np.mean(np.array(correlations)[mask]))
            
            if bin_centers:
                ax.plot(bin_centers, bin_correlations, 'o-', label=name, alpha=0.7)
    
    ax.set_xlabel('Step size (μm)')
    ax.set_ylabel('Mean directional correlation')
    ax.set_title('Correlation vs Step Size\n(Larger steps = more reliable direction)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Velocity persistence
    ax = axes[1, 1]
    for i, (data, name) in enumerate(zip(resolution_data_list, condition_names)):
        velocity_persistence = data['velocity_persistence']
        if velocity_persistence:
            ax.hist(velocity_persistence, bins=20, alpha=0.5, label=f'{name} (n={len(velocity_persistence)})', density=True)
    
    ax.set_xlabel('Speed autocorrelation (frame-to-frame)')
    ax.set_ylabel('Density')
    ax.set_title('Speed Persistence\n(How consistent is particle speed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "temporal_resolution_diagnostics.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_alternative_metrics_plot(comparison_results, output_path):
    """
    Create plots showing alternative persistence metrics.
    
    Args:
        comparison_results: Dictionary with comparison results
        output_path: Directory to save plots
    """
    # Create figure for multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Turning angle comparison
    if 'turning_angles' in comparison_results:
        ax = axes[0, 0]
        data = comparison_results['turning_angles']
        
        condition_names = [summary['condition'] for summary in data['condition_summaries']]
        means = [summary['mean'] for summary in data['condition_summaries']]
        sems = [summary['sem'] for summary in data['condition_summaries']]
        
        bars = ax.bar(condition_names, means, yerr=sems, capsize=5, alpha=0.7)
        ax.set_ylabel('Mean turning angle (degrees)')
        ax.set_title('Mean Turning Angle Comparison\n(Higher = more tortuous paths)')
        ax.grid(True, alpha=0.3)
        
        # Add significance markers
        for comparison in data['pairwise_comparisons']:
            if comparison['significant']:
                ax.text(0.5, max(means) * 1.1, f"p = {comparison['p_value']:.2e}", 
                       ha='center', transform=ax.transData)
    
    # Plot 2: Persistence ratios
    ax = axes[0, 1]
    persistence_data = {}
    
    for threshold in PERSISTENCE_THRESHOLDS:
        key = f'persistence_{threshold}deg'
        if key in comparison_results:
            data = comparison_results[key]
            if 'condition_summaries' in data:
                condition_names = [summary['condition'] for summary in data['condition_summaries']]
                means = [summary['mean'] for summary in data['condition_summaries']]
                persistence_data[threshold] = means
    
    if persistence_data:
        x = np.arange(len(condition_names))
        width = 0.2
        
        for i, threshold in enumerate(PERSISTENCE_THRESHOLDS):
            if threshold in persistence_data:
                offset = (i - len(PERSISTENCE_THRESHOLDS)/2 + 0.5) * width
                ax.bar(x + offset, persistence_data[threshold], width, 
                      label=f'<{threshold}°', alpha=0.7)
        
        ax.set_xlabel('Condition')
        ax.set_ylabel('Persistence ratio')
        ax.set_title('Directional Persistence Ratios\n(Higher = more persistent direction)')
        ax.set_xticks(x)
        ax.set_xticklabels(condition_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Multi-scale autocorrelations
    ax = axes[1, 0]
    
    # Collect autocorrelation data for different lags
    autocorr_data = {}
    for lag in MULTI_SCALE_LAGS[:5]:  # Show first 5 lags
        key = f'autocorr_lag_{lag}'
        if key in comparison_results:
            data = comparison_results[key]
            if 'condition_summaries' in data:
                for summary in data['condition_summaries']:
                    condition = summary['condition']
                    if condition not in autocorr_data:
                        autocorr_data[condition] = {'lags': [], 'values': [], 'errors': []}
                    autocorr_data[condition]['lags'].append(lag * DT)
                    autocorr_data[condition]['values'].append(summary['mean'])
                    autocorr_data[condition]['errors'].append(summary['sem'])
    
    for condition, data in autocorr_data.items():
        if data['lags']:
            ax.errorbar(data['lags'], data['values'], yerr=data['errors'], 
                       marker='o', label=condition, capsize=3)
    
    ax.set_xlabel('Time lag (s)')
    ax.set_ylabel('Directional autocorrelation')
    ax.set_title('Multi-scale Directional Autocorrelation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # Plot 4: Trajectory straightness
    if 'straightness' in comparison_results:
        ax = axes[1, 1]
        data = comparison_results['straightness']
        
        condition_names = [summary['condition'] for summary in data['condition_summaries']]
        means = [summary['mean'] for summary in data['condition_summaries']]
        sems = [summary['sem'] for summary in data['condition_summaries']]
        
        bars = ax.bar(condition_names, means, yerr=sems, capsize=5, alpha=0.7)
        ax.set_ylabel('Trajectory straightness')
        ax.set_title('Trajectory Straightness\n(1.0 = perfectly straight, 0.0 = highly curved)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "alternative_persistence_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function for advanced angle analysis."""
    print("Advanced Angle Autocorrelation Analysis")
    print("=====================================")
    print("Addressing temporal resolution limitations")
    print()
    
    # Get input directories
    condition_dirs = []
    condition_names = []
    
    n_conditions = int(input("Enter number of conditions to compare: "))
    
    for i in range(n_conditions):
        condition_dir = input(f"Enter directory for condition {i+1}: ")
        condition_name = input(f"Enter name for condition {i+1}: ")
        
        condition_dirs.append(condition_dir)
        condition_names.append(condition_name)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), f"advanced_angle_analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and analyze each condition
    print("\nLoading and analyzing conditions...")
    
    condition_persistence_data = []
    condition_resolution_data = []
    
    for i, (condition_dir, condition_name) in enumerate(zip(condition_dirs, condition_names)):
        print(f"Processing {condition_name}...")
        
        # Load all trajectory files for this condition
        file_paths = glob.glob(os.path.join(condition_dir, "analyzed_*.pkl"))
        
        if not file_paths:
            print(f"No files found in {condition_dir}")
            continue
        
        # Pool all trajectories
        all_trajectories = []
        for file_path in file_paths:
            data = load_analyzed_data(file_path)
            if data and 'trajectories' in data:
                all_trajectories.extend(data['trajectories'])
        
        if not all_trajectories:
            print(f"No trajectories found for {condition_name}")
            continue
        
        print(f"  Found {len(all_trajectories)} trajectories")
        
        # Calculate persistence metrics
        persistence_data = calculate_persistence_metrics(all_trajectories)
        condition_persistence_data.append(persistence_data)
        
        # Calculate resolution analysis
        resolution_data = analyze_temporal_resolution_effects(all_trajectories)
        condition_resolution_data.append(resolution_data)
    
    # Compare conditions
    print("Comparing conditions with alternative metrics...")
    comparison_results = compare_conditions_advanced(condition_persistence_data, condition_names)
    
    # Create plots
    print("Creating diagnostic plots...")
    create_resolution_diagnostic_plots(condition_resolution_data, condition_names, output_dir)
    create_alternative_metrics_plot(comparison_results, output_dir)
    
    # Export results
    print("Exporting results...")
    
    # Create summary report
    with open(os.path.join(output_dir, "analysis_summary.txt"), 'w') as f:
        f.write("Advanced Angle Autocorrelation Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("TEMPORAL RESOLUTION ANALYSIS:\n")
        f.write(f"Recording frequency: {DT} s\n")
        f.write("This analysis addresses the limitation that crossing times (~0.1s) are close to recording frequency.\n\n")
        
        f.write("ALTERNATIVE METRICS ANALYZED:\n")
        f.write("1. Mean turning angles between consecutive steps\n")
        f.write("2. Persistence ratios (fraction of small turning angles)\n")
        f.write("3. Multi-scale directional autocorrelations\n")
        f.write("4. Trajectory straightness\n")
        f.write("5. Step size vs correlation relationships\n\n")
        
        # Write detailed comparison results
        for metric_name, results in comparison_results.items():
            if 'pairwise_comparisons' in results:
                f.write(f"\n{results['metric_name']}:\n")
                f.write("-" * 40 + "\n")
                
                for comparison in results['pairwise_comparisons']:
                    f.write(f"{comparison['condition1']} vs {comparison['condition2']}:\n")
                    f.write(f"  Mean difference: {comparison['mean_diff']:.4f}\n")
                    f.write(f"  P-value: {comparison['p_value']:.2e}\n")
                    f.write(f"  Effect size: {comparison['effect_size']} (δ = {comparison['cliffs_delta']:.3f})\n")
                    f.write(f"  Significant: {comparison['significant']}\n\n")
    
    print(f"\nAnalysis complete. Results saved to: {output_dir}")
    
    # Print key findings
    print("\nKEY FINDINGS:")
    print("=============")
    
    # Check turning angles
    if 'turning_angles' in comparison_results:
        ta_data = comparison_results['turning_angles']
        if ta_data['pairwise_comparisons']:
            comp = ta_data['pairwise_comparisons'][0]
            print(f"Mean turning angles: {comp['condition1']} vs {comp['condition2']}")
            print(f"  Difference: {comp['mean_diff']:.2f}° (p = {comp['p_value']:.2e})")
            
            if abs(comp['mean_diff']) > 5:  # More than 5 degrees difference
                print(f"  → Substantial difference in path tortuosity")
            else:
                print(f"  → Similar path tortuosity despite statistical difference")
    
    # Check persistence ratios
    for threshold in PERSISTENCE_THRESHOLDS:
        key = f'persistence_{threshold}deg'
        if key in comparison_results:
            p_data = comparison_results[key]
            if p_data['pairwise_comparisons']:
                comp = p_data['pairwise_comparisons'][0]
                print(f"Persistence (<{threshold}°): {comp['condition1']} vs {comp['condition2']}")
                print(f"  Difference: {comp['mean_diff']:.3f} (p = {comp['p_value']:.2e})")
    
    print("\nINTERPRETATION GUIDE:")
    print("====================")
    print("Since your crossing times (~0.1s) are close to recording frequency:")
    print("1. Focus on TURNING ANGLE differences - more reliable than crossing times")
    print("2. Look at PERSISTENCE RATIOS - what fraction of steps maintain direction")
    print("3. Consider TRAJECTORY STRAIGHTNESS - overall path efficiency")
    print("4. Multi-scale analysis shows if differences persist at longer time scales")

if __name__ == "__main__":
    main()