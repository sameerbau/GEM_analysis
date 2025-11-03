# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 13:37:26 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracking_noise_analyzer.py

This script analyzes the inherent noise in particle tracking data by splitting
trajectories into smaller time chunks and comparing the resulting distributions
of diffusion coefficients.

Input:
- Pickle files containing processed or analyzed trajectory data

Output:
- Diagnostic plots showing consistency in diffusion coefficient measurements
- CSV files with partition analysis results
- Summary plots and statistics

Usage:
python tracking_noise_analyzer.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
from pathlib import Path
from scipy.optimize import curve_fit

# Global parameters (can be modified)
# =====================================
# Default partition sizes (in frames)
PARTITION_SIZES = [5, 10, 15, 20]
# Minimum trajectory length to consider
MIN_TRACK_LENGTH = 5
# Time step in seconds
DT = 0.05
# Significance level for statistical tests
ALPHA = 0.05
# =====================================

def load_data(file_path):
    """
    Load trajectory data from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary containing trajectory data
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def create_partitions(trajectories, partition_size):
    """
    Split trajectories into chunks of specified frame size.
    
    Args:
        trajectories: List of trajectory dictionaries
        partition_size: Size of each partition in frames
        
    Returns:
        Tuple of (partitioned trajectories, partition info)
    """
    partitioned_data = []
    partition_info = []
    partition_counter = 0
    
    for traj in trajectories:
        track_id = traj['id']
        x = traj['x']
        y = traj['y']
        
        # Skip if track is too short
        if len(x) < partition_size:
            continue
        
        # Determine number of partitions
        num_partitions = len(x) // partition_size
        
        # Create partitions
        for p in range(num_partitions):
            start_idx = p * partition_size
            end_idx = (p + 1) * partition_size
            
            # Create a new partition
            partition_data = {
                'id': f"{track_id}_part{p}",
                'original_id': track_id,
                'x': x[start_idx:end_idx],
                'y': y[start_idx:end_idx],
                'partition_idx': p
            }
            
            partitioned_data.append(partition_data)
            
            # Store partition information
            partition_info.append({
                'track_id': track_id,
                'partition_idx': p,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
            
            partition_counter += 1
    
    print(f"Created {partition_counter} partitions from {len(trajectories)} trajectories")
    
    return partitioned_data, partition_info

def linear_msd(t, D, offset):
    """
    Linear MSD model: MSD = 4*D*t + offset
    For 2D diffusion, the factor is 4 (2*dimensions)
    
    Args:
        t: Time lag
        D: Diffusion coefficient
        offset: Y-intercept (related to localization error)
        
    Returns:
        MSD values according to the model
    """
    return 4 * D * t + offset

def calculate_msd(x, y, dt):
    """
    Calculate mean squared displacement for a trajectory.
    
    Args:
        x: X coordinates
        y: Y coordinates
        dt: Time step
        
    Returns:
        Tuple of (MSD values, time lags)
    """
    n = len(x)
    max_lag = n - 1
    msd = np.zeros(max_lag)
    count = np.zeros(max_lag)
    
    # Calculate displacement for each time lag
    for lag in range(1, max_lag + 1):
        for i in range(n - lag):
            dx = x[i + lag] - x[i]
            dy = y[i + lag] - y[i]
            msd[lag - 1] += dx**2 + dy**2
            count[lag - 1] += 1
    
    # Average over all pairs
    msd = msd / count
    time_lags = np.arange(1, max_lag + 1) * dt
    
    return msd, time_lags

def calculate_diffusion_coefficients(partitioned_data, dt):
    """
    Calculate diffusion coefficients for each partition using MSD analysis.
    
    Args:
        partitioned_data: List of partitioned trajectory dictionaries
        dt: Time step in seconds
        
    Returns:
        List of diffusion coefficients
    """
    D_values = []
    
    for partition in partitioned_data:
        # Calculate MSD
        pos_x = np.array(partition['x'])
        pos_y = np.array(partition['y'])
        
        msd, time_lags = calculate_msd(pos_x, pos_y, dt)
        
        # Linear fit to MSD curve (first 4 points or less if not available)
        fit_points = min(4, len(msd))
        if fit_points >= 2:
            try:
                popt, _ = curve_fit(linear_msd, time_lags[:fit_points], msd[:fit_points])
                D = popt[0] / 4  # D = slope/4 for 2D diffusion
                D_values.append(D)
            except:
                # Skip if fitting fails
                pass
    
    return D_values

def compare_partitions(D_partitions):
    """
    Statistical comparison between partitions.
    
    Args:
        D_partitions: List of diffusion coefficients
        
    Returns:
        Dictionary with statistical metrics
    """
    if not D_partitions:
        return {
            'n': 0,
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'sem': np.nan,
            'cv': np.nan,
            'normality_test': {'h': np.nan, 'p': np.nan}
        }
    
    # Basic statistics
    mean_D = np.mean(D_partitions)
    median_D = np.median(D_partitions)
    std_D = np.std(D_partitions)
    sem_D = std_D / np.sqrt(len(D_partitions))
    cv_D = std_D / mean_D if mean_D > 0 else np.nan  # Coefficient of variation
    
    # Test for normality (Shapiro-Wilk)
    if len(D_partitions) >= 3:  # Shapiro-Wilk requires at least 3 samples
        from scipy import stats
        _, p_shapiro = stats.shapiro(D_partitions)
        h_shapiro = p_shapiro < ALPHA
    else:
        h_shapiro = np.nan
        p_shapiro = np.nan
    
    return {
        'n': len(D_partitions),
        'mean': mean_D,
        'median': median_D,
        'std': std_D,
        'sem': sem_D,
        'cv': cv_D,
        'normality_test': {'h': h_shapiro, 'p': p_shapiro}
    }

def assess_consistency(partition_results):
    """
    Assess overall consistency across different partition sizes.
    
    Args:
        partition_results: List of dictionaries with partition analysis results
        
    Returns:
        Dictionary with consistency metrics
    """
    # Extract metrics for each partition size
    partition_sizes = [result['partition_size'] for result in partition_results]
    cv_values = [result['statistics']['cv'] for result in partition_results]
    mean_values = [result['statistics']['mean'] for result in partition_results]
    median_values = [result['statistics']['median'] for result in partition_results]
    std_values = [result['statistics']['std'] for result in partition_results]
    
    # Calculate consistency metrics
    overall_cv = np.std(mean_values) / np.mean(mean_values) if np.mean(mean_values) > 0 else np.nan
    
    # Calculate trend slope
    if len(partition_sizes) > 1:
        trend_poly = np.polyfit(partition_sizes, mean_values, 1)
        trend_slope = trend_poly[0]
        trend_intercept = trend_poly[1]
    else:
        trend_slope = np.nan
        trend_intercept = np.nan
    
    return {
        'partition_sizes': partition_sizes,
        'cv_values': cv_values,
        'mean_values': mean_values,
        'median_values': median_values,
        'std_values': std_values,
        'overall_cv': overall_cv,
        'trend_slope': trend_slope,
        'trend_intercept': trend_intercept
    }

def generate_partition_plots(D_partitions, stats, output_path, partition_size):
    """
    Generate diagnostic plots for a partition analysis.
    
    Args:
        D_partitions: List of diffusion coefficients
        stats: Dictionary with statistical metrics
        output_path: Directory to save the plots
        partition_size: Partition size in frames
    """
    if not D_partitions:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # 1. Histogram of diffusion coefficients
    plt.figure(figsize=(10, 7))
    plt.hist(D_partitions, bins=20, density=True, alpha=0.7, color='dodgerblue')
    
    # Add the mean and median lines
    ylim = plt.ylim()
    plt.plot([stats['mean'], stats['mean']], ylim, 'r-', linewidth=2, 
             label=f'Mean = {stats["mean"]:.4f}')
    plt.plot([stats['median'], stats['median']], ylim, 'g--', linewidth=2, 
             label=f'Median = {stats["median"]:.4f}')
    
    plt.xlabel('Diffusion Coefficient (μm²/s)', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Distribution of Diffusion Coefficients\nPartition Size = {partition_size} frames, CV = {stats["cv"]:.4f}, n = {stats["n"]}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_path, f'partition_{partition_size}_histogram.png'), dpi=300)
    plt.close()
    
    # 2. QQ Plot to assess normality
    from scipy import stats as scipy_stats
    plt.figure(figsize=(8, 8))
    scipy_stats.probplot(D_partitions, dist="norm", plot=plt)
    plt.title(f'QQ Plot for Partition Size {partition_size}\nShapiro-Wilk p-value = {stats["normality_test"]["p"]:.4f}')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_path, f'partition_{partition_size}_qqplot.png'), dpi=300)
    plt.close()
    
    # 3. Box plot with individual points
    plt.figure(figsize=(8, 7))
    plt.boxplot(D_partitions, widths=0.5)
    
    # Add scatter points
    x = np.random.normal(1, 0.1, len(D_partitions))  # Add jitter
    plt.scatter(x, D_partitions, alpha=0.6, s=20, color='dodgerblue')
    
    plt.ylabel('Diffusion Coefficient (μm²/s)', fontsize=12)
    plt.title(f'Diffusion Coefficients for Partition Size {partition_size}\nn = {stats["n"]}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, f'partition_{partition_size}_boxplot.png'), dpi=300)
    plt.close()

def generate_summary_plots(results, output_path):
    """
    Generate summary plots for consistency across partition sizes.
    
    Args:
        results: Dictionary with analysis results
        output_path: Directory to save the plots
    """
    consistency = results['consistency']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # 1. Plot CV vs partition size
    plt.figure(figsize=(10, 7))
    plt.plot(consistency['partition_sizes'], consistency['cv_values'], 'o-', 
             linewidth=2, markersize=10, color='dodgerblue')
    
    plt.xlabel('Partition Size (frames)', fontsize=12)
    plt.ylabel('Coefficient of Variation', fontsize=12)
    plt.title('Consistency vs. Partition Size')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_path, 'cv_vs_partition.png'), dpi=300)
    plt.close()
    
    # 2. Plot Mean and StdDev vs partition size
    plt.figure(figsize=(12, 7))
    
    # Plot mean values
    plt.plot(consistency['partition_sizes'], consistency['mean_values'], 'bo-', 
             linewidth=2, markersize=8, label='Mean')
    
    # Plot standard deviation
    plt.plot(consistency['partition_sizes'], consistency['std_values'], 'ro-', 
             linewidth=2, markersize=8, label='Std Dev')
    
    plt.xlabel('Partition Size (frames)', fontsize=12)
    plt.ylabel('Diffusion Coefficient (μm²/s)', fontsize=12)
    plt.title('Mean and Standard Deviation vs. Partition Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_path, 'mean_std_vs_partition.png'), dpi=300)
    plt.close()
    
    # 3. Plot trend line for mean values
    plt.figure(figsize=(10, 7))
    x = np.array(consistency['partition_sizes'])
    y = np.array(consistency['mean_values'])
    
    plt.plot(x, y, 'bo-', linewidth=2, markersize=8, label='Mean')
    
    # Add trend line if there are at least 2 points
    if len(x) > 1 and not np.isnan(consistency['trend_slope']):
        trend_y = consistency['trend_slope'] * x + consistency['trend_intercept']
        plt.plot(x, trend_y, 'r--', linewidth=2, 
                 label=f'Trend: y = {consistency["trend_slope"]:.4f}x + {consistency["trend_intercept"]:.4f}')
    
    plt.xlabel('Partition Size (frames)', fontsize=12)
    plt.ylabel('Mean Diffusion Coefficient (μm²/s)', fontsize=12)
    plt.title('Mean Diffusion Coefficient vs. Partition Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_path, 'mean_trend_vs_partition.png'), dpi=300)
    plt.close()

def analyze_within_file(file_path, partition_sizes, dt, output_dir):
    """
    Perform within-file partitioning analysis.
    
    Args:
        file_path: Path to the trajectory data file
        partition_sizes: List of partition sizes to analyze
        dt: Time step in seconds
        output_dir: Directory to save results
        
    Returns:
        Dictionary with analysis results
    """
    # Load trajectory data
    data = load_data(file_path)
    
    if data is None:
        print(f"Failed to load data from {file_path}")
        return None
    
    # Extract trajectories
    if 'trajectories' in data:
        trajectories = data['trajectories']
    else:
        print(f"Could not find trajectory data in {file_path}")
        return None
    
    # Initialize results structure
    results = {
        'filename': os.path.basename(file_path),
        'partition_results': [],
        'consistency': None
    }
    
    # Process each partition size
    for partition_size in partition_sizes:
        print(f"Processing partition size {partition_size} frames...")
        
        # Create partitions
        partitioned_data, partition_info = create_partitions(trajectories, partition_size)
        
        # Calculate diffusion coefficients for each partition
        D_partitions = calculate_diffusion_coefficients(partitioned_data, dt)
        
        # Statistical comparison between partitions
        stats = compare_partitions(D_partitions)
        
        # Store results
        partition_result = {
            'partition_size': partition_size,
            'partition_info': partition_info,
            'D_partitions': D_partitions,
            'statistics': stats
        }
        
        results['partition_results'].append(partition_result)
        
        # Generate and save plots
        generate_partition_plots(D_partitions, stats, os.path.join(output_dir, 'partition_plots'), partition_size)
    
    # Overall consistency assessment
    results['consistency'] = assess_consistency(results['partition_results'])
    
    print(f"Overall coefficient of variation: {results['consistency']['overall_cv']:.4f}")
    
    # Generate summary plots
    generate_summary_plots(results, os.path.join(output_dir, 'summary_plots'))
    
    # Save results to CSV
    save_results_to_csv(results, output_dir)
    
    return results

def save_results_to_csv(results, output_dir):
    """
    Save analysis results to CSV files.
    
    Args:
        results: Dictionary with analysis results
        output_dir: Directory to save the CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Partition statistics summary
    partition_stats = []
    
    for result in results['partition_results']:
        partition_stats.append({
            'PartitionSize': result['partition_size'],
            'NumPartitions': result['statistics']['n'],
            'MeanD': result['statistics']['mean'],
            'MedianD': result['statistics']['median'],
            'StdD': result['statistics']['std'],
            'SemD': result['statistics']['sem'],
            'CV': result['statistics']['cv'],
            'NormalityP': result['statistics']['normality_test']['p']
        })
    
    if partition_stats:
        pd.DataFrame(partition_stats).to_csv(
            os.path.join(output_dir, 'partition_statistics.csv'), index=False)
    
    # 2. Diffusion coefficients for each partition
    for result in results['partition_results']:
        if result['D_partitions']:
            d_values = pd.DataFrame({
                'PartitionSize': result['partition_size'],
                'DiffusionCoefficient': result['D_partitions']
            })
            
            d_values.to_csv(
                os.path.join(output_dir, f'diffusion_coefficients_part{result["partition_size"]}.csv'),
                index=False)
    
    # 3. Consistency metrics
    consistency = results['consistency']
    
    consistency_df = pd.DataFrame({
        'PartitionSize': consistency['partition_sizes'],
        'MeanD': consistency['mean_values'],
        'MedianD': consistency['median_values'],
        'StdD': consistency['std_values'],
        'CV': consistency['cv_values']
    })
    
    consistency_df.to_csv(
        os.path.join(output_dir, 'consistency_metrics.csv'), index=False)
    
    # 4. Summary metrics
    summary = {
        'Filename': [results['filename']],
        'OverallCV': [consistency['overall_cv']],
        'TrendSlope': [consistency['trend_slope']],
        'TrendIntercept': [consistency['trend_intercept']]
    }
    
    pd.DataFrame(summary).to_csv(
        os.path.join(output_dir, 'summary_metrics.csv'), index=False)

def main():
    """Main function to run tracking noise analysis."""
    print("Tracking Noise Analysis")
    print("======================")
    
    # Get input file path
    input_file = input("Enter path to trajectory data file in pkl format (e.g tracked_Traj_Cell_em1_z003.nd2_crop.pkl): ")
    
    if not os.path.isfile(input_file):
        print(f"File {input_file} does not exist")
        return
    
    # Get parameters or use defaults
    partition_sizes_input = input(f"Enter partition sizes separated by commas (default: {','.join(map(str, PARTITION_SIZES))}): ")
    if partition_sizes_input.strip():
        try:
            partition_sizes = [int(x.strip()) for x in partition_sizes_input.split(',')]
        except:
            print(f"Invalid input. Using default partition sizes: {PARTITION_SIZES}")
            partition_sizes = PARTITION_SIZES
    else:
        partition_sizes = PARTITION_SIZES
    
    dt_input = input(f"Enter time step in seconds (default: {DT}): ")
    if dt_input.strip():
        try:
            dt = float(dt_input)
        except:
            print(f"Invalid input. Using default time step: {DT}")
            dt = DT
    else:
        dt = DT
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.join(os.path.dirname(input_file), f"noise_analysis_{base_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nAnalyzing file: {input_file}")
    print(f"Partition sizes: {partition_sizes}")
    print(f"Time step: {dt} seconds")
    print(f"Output directory: {output_dir}")
    
    # Perform analysis
    results = analyze_within_file(input_file, partition_sizes, dt, output_dir)
    
    if results:
        # Save full results as pickle
        with open(os.path.join(output_dir, 'full_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        print("\nAnalysis complete!")
        print(f"Results saved in: {output_dir}")
        
        # Print summary
        print("\nSummary:")
        print(f"Overall coefficient of variation: {results['consistency']['overall_cv']:.4f}")
        
        for result in results['partition_results']:
            stats = result['statistics']
            print(f"\nPartition size {result['partition_size']} frames:")
            print(f"  Number of partitions: {stats['n']}")
            print(f"  Mean diffusion coefficient: {stats['mean']:.6f} µm²/s")
            print(f"  Coefficient of variation: {stats['cv']:.4f}")
    else:
        print("Analysis failed.")

if __name__ == "__main__":
    main()