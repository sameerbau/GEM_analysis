#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracking_noise_analyzer.py

This script analyzes the inherent noise in particle tracking data using two approaches:
1. Trajectory partitioning: Splitting individual trajectories into smaller chunks
2. Temporal chunking: Dividing the entire movie into time segments and analyzing each segment

Input:
- Pickle files containing processed trajectory data

Output:
- Diagnostic plots showing consistency in diffusion coefficient measurements
- CSV files with analysis results
- Summary plots and statistics

Usage:
python tracking_noise_analyzer.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy.optimize import curve_fit
from scipy import stats

# Global parameters (can be modified)
# =====================================
# Default partition sizes (in frames)
PARTITION_SIZES = [5, 10, 15, 20]
# Default number of temporal chunks
DEFAULT_NUM_CHUNKS = 3
# Minimum trajectory length to consider
MIN_TRACK_LENGTH = 5
# Time step in seconds
DT = 0.05
# Significance level for statistical tests
ALPHA = 0.05
# Maximum number of lags to use for MSD calculation (as a fraction of track length)
MAX_LAG_FRACTION = 0.5
# Number of points to use for linear MSD fitting
MSD_FIT_POINTS = 4
# =====================================

def load_data(file_path):
    """
    Load trajectory data from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary containing trajectory data or None if loading fails
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Basic validation of the loaded data
        if not isinstance(data, dict):
            print(f"Warning: Loaded data is not a dictionary. Type: {type(data)}")
            return None
            
        if 'trajectories' not in data:
            print("Warning: 'trajectories' key not found in the loaded data")
            return None
            
        print(f"Successfully loaded data from {file_path}")
        print(f"Number of trajectories: {len(data['trajectories'])}")
        
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {str(e)}")
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
        track_id = traj.get('id', f"track_{partition_counter}")
        
        # Check if trajectory has x and y coordinates
        if 'x' not in traj or 'y' not in traj:
            continue
            
        x = traj['x']
        y = traj['y']
        
        # Skip if track is too short for this partition size
        if len(x) < partition_size:
            continue
        
        # Determine number of complete partitions
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

def create_temporal_chunks(trajectories, num_chunks, dt=DT):
    """
    Split trajectories into temporal chunks based on their time span.
    
    Args:
        trajectories: List of trajectory dictionaries
        num_chunks: Number of chunks to create
        dt: Time step in seconds
        
    Returns:
        Tuple of (chunked trajectories, chunk info, chunk boundaries)
    """
    # Determine the total time span
    max_frame = 0
    for traj in trajectories:
        if 'time' in traj and len(traj['time']) > 0:
            max_frame = max(max_frame, np.max(traj['time']) / dt)
        elif 'x' in traj and len(traj['x']) > 0:
            max_frame = max(max_frame, len(traj['x']))
    
    # Calculate chunk boundaries in frames
    chunk_size = max_frame / num_chunks
    chunk_boundaries = [int(i * chunk_size) for i in range(num_chunks + 1)]
    
    # Initialize data structures
    chunked_trajectories = [[] for _ in range(num_chunks)]
    chunk_info = []
    
    # Assign trajectories to chunks based on median frame
    for traj in trajectories:
        # Skip if missing required data
        if 'x' not in traj or 'y' not in traj:
            continue
            
        # Calculate median frame
        if 'time' in traj and len(traj['time']) > 0:
            median_time = np.median(traj['time'])
            median_frame = median_time / dt
        else:
            median_frame = len(traj['x']) / 2
        
        # Determine which chunk this belongs to
        for i in range(num_chunks):
            if chunk_boundaries[i] <= median_frame < chunk_boundaries[i+1]:
                # Add a copy of the trajectory to this chunk
                chunked_trajectories[i].append(traj)
                
                # Store chunk information
                chunk_info.append({
                    'track_id': traj.get('id', 'unknown'),
                    'chunk_index': i,
                    'time_range': (chunk_boundaries[i] * dt, chunk_boundaries[i+1] * dt),
                    'median_frame': median_frame
                })
                break
    
    # Print summary
    for i in range(num_chunks):
        print(f"Chunk {i+1}: {len(chunked_trajectories[i])} trajectories, time range: "
              f"{chunk_boundaries[i]*dt:.2f}s - {chunk_boundaries[i+1]*dt:.2f}s")
    
    return chunked_trajectories, chunk_info, chunk_boundaries

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
    
    # Determine maximum lag to calculate (up to MAX_LAG_FRACTION of track length)
    max_lag = min(n - 1, int(n * MAX_LAG_FRACTION))
    
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

def calculate_diffusion_coefficients(trajectories, dt):
    """
    Calculate diffusion coefficients for a list of trajectories using MSD analysis.
    
    Args:
        trajectories: List of trajectory dictionaries
        dt: Time step in seconds
        
    Returns:
        List of diffusion coefficients
    """
    D_values = []
    failed_fits = 0
    
    for traj in trajectories:
        try:
            # Check if trajectory data exists
            if 'x' not in traj or 'y' not in traj:
                continue
                
            # Extract position data
            pos_x = np.array(traj['x'])
            pos_y = np.array(traj['y'])
            
            # Skip trajectories that are too short
            if len(pos_x) < MSD_FIT_POINTS + 1:  # Need at least n+1 points for n lags
                continue
            
            # Calculate MSD
            msd, time_lags = calculate_msd(pos_x, pos_y, dt)
            
            # Linear fit to MSD curve (first MSD_FIT_POINTS points or less if not available)
            fit_points = min(MSD_FIT_POINTS, len(msd))
            
            if fit_points >= 2:  # Need at least 2 points for fitting
                popt, pcov = curve_fit(linear_msd, time_lags[:fit_points], msd[:fit_points])
                D = popt[0] / 4  # D = slope/4 for 2D diffusion
                
                # Only accept positive diffusion coefficients
                if D > 0:
                    D_values.append(D)
                else:
                    failed_fits += 1
            else:
                failed_fits += 1
        except Exception as e:
            failed_fits += 1
    
    if failed_fits > 0:
        print(f"Warning: {failed_fits} trajectories could not be fitted properly")
    
    print(f"Successfully calculated diffusion coefficients for {len(D_values)} trajectories")
    return D_values

def calculate_statistics(D_values):
    """
    Calculate statistical metrics for a list of diffusion coefficients.
    
    Args:
        D_values: List of diffusion coefficients
        
    Returns:
        Dictionary with statistical metrics
    """
    if not D_values:
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
    mean_D = np.mean(D_values)
    median_D = np.median(D_values)
    std_D = np.std(D_values)
    sem_D = std_D / np.sqrt(len(D_values))
    cv_D = std_D / mean_D if mean_D > 0 else np.nan  # Coefficient of variation
    
    # Test for normality (Shapiro-Wilk)
    if len(D_values) >= 3:  # Shapiro-Wilk requires at least 3 samples
        _, p_shapiro = stats.shapiro(D_values)
        h_shapiro = p_shapiro < ALPHA
    else:
        h_shapiro = np.nan
        p_shapiro = np.nan
    
    return {
        'n': len(D_values),
        'mean': mean_D,
        'median': median_D,
        'std': std_D,
        'sem': sem_D,
        'cv': cv_D,
        'normality_test': {'h': h_shapiro, 'p': p_shapiro}
    }

def assess_consistency(results):
    """
    Assess overall consistency across different analysis units.
    
    Args:
        results: List of dictionaries with analysis results
        
    Returns:
        Dictionary with consistency metrics
    """
    # Extract metrics for each unit (partition size or time chunk)
    units = [result['unit_value'] for result in results]
    cv_values = [result['statistics']['cv'] for result in results]
    mean_values = [result['statistics']['mean'] for result in results]
    median_values = [result['statistics']['median'] for result in results]
    std_values = [result['statistics']['std'] for result in results]
    
    # Filter out NaN values
    valid_indices = ~np.isnan(mean_values)
    if not np.any(valid_indices):
        return {
            'units': units,
            'cv_values': cv_values,
            'mean_values': mean_values,
            'median_values': median_values,
            'std_values': std_values,
            'overall_cv': np.nan,
            'trend_slope': np.nan,
            'trend_intercept': np.nan
        }
    
    units_valid = np.array(units)[valid_indices]
    mean_values_valid = np.array(mean_values)[valid_indices]
    
    # Calculate consistency metrics
    overall_cv = np.std(mean_values_valid) / np.mean(mean_values_valid) if np.mean(mean_values_valid) > 0 else np.nan
    
    # Calculate trend slope
    if len(units_valid) > 1:
        trend_poly = np.polyfit(units_valid, mean_values_valid, 1)
        trend_slope = trend_poly[0]
        trend_intercept = trend_poly[1]
    else:
        trend_slope = np.nan
        trend_intercept = np.nan
    
    return {
        'units': units,
        'cv_values': cv_values,
        'mean_values': mean_values,
        'median_values': median_values,
        'std_values': std_values,
        'overall_cv': overall_cv,
        'trend_slope': trend_slope,
        'trend_intercept': trend_intercept
    }

def generate_diagnostic_plots(D_values, stat_values, output_path, unit_name, unit_value):
    """
    Generate diagnostic plots for an analysis unit.
    
    Args:
        D_values: List of diffusion coefficients
        stat_values: Dictionary with statistical metrics
        output_path: Directory to save the plots
        unit_name: Name of the analysis unit (e.g., 'partition' or 'chunk')
        unit_value: Value of the analysis unit
    """
    if not D_values:
        print(f"No data to plot for {unit_name} {unit_value}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # 1. Histogram of diffusion coefficients
    plt.figure(figsize=(10, 7))
    plt.hist(D_values, bins=20, density=True, alpha=0.7, color='dodgerblue')
    
    # Add the mean and median lines
    ylim = plt.ylim()
    plt.plot([stat_values['mean'], stat_values['mean']], ylim, 'r-', linewidth=2, 
             label=f'Mean = {stat_values["mean"]:.4f}')
    plt.plot([stat_values['median'], stat_values['median']], ylim, 'g--', linewidth=2, 
             label=f'Median = {stat_values["median"]:.4f}')
    
    plt.xlabel('Diffusion Coefficient (μm²/s)', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Distribution of Diffusion Coefficients\n{unit_name.capitalize()} {unit_value}, CV = {stat_values["cv"]:.4f}, n = {stat_values["n"]}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_path, f'{unit_name}_{unit_value}_histogram.png'), dpi=300)
    plt.close()
    
    # 2. QQ Plot to assess normality
    plt.figure(figsize=(8, 8))
    stats.probplot(D_values, dist="norm", plot=plt)
    plt.title(f'QQ Plot for {unit_name.capitalize()} {unit_value}\nShapiro-Wilk p-value = {stat_values["normality_test"]["p"]:.4f}')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_path, f'{unit_name}_{unit_value}_qqplot.png'), dpi=300)
    plt.close()
    
    # 3. Box plot with individual points
    plt.figure(figsize=(8, 7))
    plt.boxplot(D_values, widths=0.5)
    
    # Add scatter points
    x = np.random.normal(1, 0.1, len(D_values))  # Add jitter
    plt.scatter(x, D_values, alpha=0.6, s=20, color='dodgerblue')
    
    plt.ylabel('Diffusion Coefficient (μm²/s)', fontsize=12)
    plt.title(f'Diffusion Coefficients for {unit_name.capitalize()} {unit_value}\nn = {stat_values["n"]}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, f'{unit_name}_{unit_value}_boxplot.png'), dpi=300)
    plt.close()

def generate_summary_plots(consistency, output_path, analysis_type):
    """
    Generate summary plots for consistency across analysis units.
    
    Args:
        consistency: Dictionary with consistency metrics
        output_path: Directory to save the plots
        analysis_type: Type of analysis ('partition' or 'temporal')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Set x-axis label based on analysis type
    if analysis_type == 'partition':
        x_label = 'Partition Size (frames)'
        title_prefix = 'Partition'
    else:  # temporal
        x_label = 'Time Chunk'
        title_prefix = 'Temporal Chunk'
    
    # 1. Plot CV vs analysis unit
    plt.figure(figsize=(10, 7))
    plt.plot(consistency['units'], consistency['cv_values'], 'o-', 
             linewidth=2, markersize=10, color='dodgerblue')
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Coefficient of Variation', fontsize=12)
    plt.title(f'Consistency vs. {title_prefix}')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_path, f'{analysis_type}_cv_vs_unit.png'), dpi=300)
    plt.close()
    
    # 2. Plot Mean and StdDev vs analysis unit
    plt.figure(figsize=(12, 7))
    
    # Plot mean values
    plt.plot(consistency['units'], consistency['mean_values'], 'bo-', 
             linewidth=2, markersize=8, label='Mean')
    
    # Plot standard deviation
    plt.plot(consistency['units'], consistency['std_values'], 'ro-', 
             linewidth=2, markersize=8, label='Std Dev')
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Diffusion Coefficient (μm²/s)', fontsize=12)
    plt.title(f'Mean and Standard Deviation vs. {title_prefix}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_path, f'{analysis_type}_mean_std_vs_unit.png'), dpi=300)
    plt.close()
    
    # 3. Plot trend line for mean values
    plt.figure(figsize=(10, 7))
    x = np.array(consistency['units'])
    y = np.array(consistency['mean_values'])
    
    plt.plot(x, y, 'bo-', linewidth=2, markersize=8, label='Mean')
    
    # Add trend line if there are at least 2 valid points
    valid_indices = ~np.isnan(y)
    if np.sum(valid_indices) > 1 and not np.isnan(consistency['trend_slope']):
        x_valid = x[valid_indices]
        trend_y = consistency['trend_slope'] * x_valid + consistency['trend_intercept']
        plt.plot(x_valid, trend_y, 'r--', linewidth=2, 
                 label=f'Trend: y = {consistency["trend_slope"]:.4f}x + {consistency["trend_intercept"]:.4f}')
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Mean Diffusion Coefficient (μm²/s)', fontsize=12)
    plt.title(f'Mean Diffusion Coefficient vs. {title_prefix}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_path, f'{analysis_type}_mean_trend_vs_unit.png'), dpi=300)
    plt.close()

def save_results_to_csv(results, output_dir, analysis_type):
    """
    Save analysis results to CSV files.
    
    Args:
        results: Dictionary with analysis results
        output_dir: Directory to save the CSV files
        analysis_type: Type of analysis ('partition' or 'temporal')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Statistics summary
    stats_data = []
    
    for result in results[f'{analysis_type}_results']:
        unit_name = 'PartitionSize' if analysis_type == 'partition' else 'TimeChunk'
        stats_data.append({
            unit_name: result['unit_value'],
            'NumTrajectories': result['statistics']['n'],
            'MeanD': result['statistics']['mean'],
            'MedianD': result['statistics']['median'],
            'StdD': result['statistics']['std'],
            'SemD': result['statistics']['sem'],
            'CV': result['statistics']['cv'],
            'NormalityP': result['statistics']['normality_test']['p']
        })
    
    if stats_data:
        pd.DataFrame(stats_data).to_csv(
            os.path.join(output_dir, f'{analysis_type}_statistics.csv'), index=False)
    
    # 2. Diffusion coefficients for each unit
    for result in results[f'{analysis_type}_results']:
        if result['D_values']:
            unit_name = 'Partition' if analysis_type == 'partition' else 'TimeChunk'
            d_values = pd.DataFrame({
                f'{unit_name}': result['unit_value'],
                'DiffusionCoefficient': result['D_values']
            })
            
            d_values.to_csv(
                os.path.join(output_dir, f'diffusion_coefficients_{analysis_type}_{result["unit_value"]}.csv'),
                index=False)
    
    # 3. Consistency metrics
    consistency = results[f'{analysis_type}_consistency']
    
    unit_name = 'PartitionSize' if analysis_type == 'partition' else 'TimeChunk'
    consistency_df = pd.DataFrame({
        unit_name: consistency['units'],
        'MeanD': consistency['mean_values'],
        'MedianD': consistency['median_values'],
        'StdD': consistency['std_values'],
        'CV': consistency['cv_values']
    })
    
    consistency_df.to_csv(
        os.path.join(output_dir, f'{analysis_type}_consistency_metrics.csv'), index=False)
    
    # 4. Summary metrics
    summary = {
        'Filename': [results['filename']],
        'OverallCV': [consistency['overall_cv']],
        'TrendSlope': [consistency['trend_slope']],
        'TrendIntercept': [consistency['trend_intercept']]
    }
    
    pd.DataFrame(summary).to_csv(
        os.path.join(output_dir, f'{analysis_type}_summary_metrics.csv'), index=False)

def perform_partition_analysis(trajectories, partition_sizes, dt, output_dir):
    """
    Perform trajectory partitioning analysis.
    
    Args:
        trajectories: List of trajectory dictionaries
        partition_sizes: List of partition sizes to analyze
        dt: Time step in seconds
        output_dir: Directory to save results
        
    Returns:
        Dictionary with analysis results
    """
    # Initialize results structure
    results = {
        'partition_results': [],
        'consistency': None
    }
    
    # Process each partition size
    for partition_size in partition_sizes:
        print(f"Processing partition size {partition_size} frames...")
        
        # Create partitions
        partitioned_data, partition_info = create_partitions(trajectories, partition_size)
        
        # Calculate diffusion coefficients for each partition
        D_values = calculate_diffusion_coefficients(partitioned_data, dt)
        
        # Statistical comparison between partitions
        stats = calculate_statistics(D_values)
        
        # Store results
        partition_result = {
            'unit_value': partition_size,
            'partition_info': partition_info,
            'D_values': D_values,
            'statistics': stats
        }
        
        results['partition_results'].append(partition_result)
        
        # Generate diagnostic plots
        generate_diagnostic_plots(
            D_values, stats, 
            os.path.join(output_dir, 'partition_plots'), 
            'partition', partition_size
        )
    
    # Overall consistency assessment
    results['consistency'] = assess_consistency(results['partition_results'])
    
    if not np.isnan(results['consistency']['overall_cv']):
        print(f"Partition analysis overall coefficient of variation: {results['consistency']['overall_cv']:.4f}")
    else:
        print("Partition analysis overall coefficient of variation: N/A")
    
    # Generate summary plots
    generate_summary_plots(
        results['consistency'], 
        os.path.join(output_dir, 'summary_plots'),
        'partition'
    )
    
    return results

def perform_temporal_analysis(trajectories, num_chunks, dt, output_dir):
    """
    Perform temporal chunking analysis.
    
    Args:
        trajectories: List of trajectory dictionaries
        num_chunks: Number of time chunks to analyze
        dt: Time step in seconds
        output_dir: Directory to save results
        
    Returns:
        Dictionary with analysis results
    """
    # Initialize results structure
    results = {
        'temporal_results': [],
        'consistency': None
    }
    
    # Create temporal chunks
    chunked_trajectories, chunk_info, chunk_boundaries = create_temporal_chunks(trajectories, num_chunks, dt)
    
    # Process each chunk
    for i, chunk_trajs in enumerate(chunked_trajectories):
        print(f"Processing time chunk {i+1} of {num_chunks}...")
        
        # Calculate diffusion coefficients for trajectories in this chunk
        D_values = calculate_diffusion_coefficients(chunk_trajs, dt)
        
        # Statistical analysis
        stats = calculate_statistics(D_values)
        
        # Store results
        chunk_result = {
            'unit_value': i+1,
            'time_range': (chunk_boundaries[i] * dt, chunk_boundaries[i+1] * dt),
            'num_trajectories': len(chunk_trajs),
            'D_values': D_values,
            'statistics': stats
        }
        
        results['temporal_results'].append(chunk_result)
        
        # Generate diagnostic plots
        generate_diagnostic_plots(
            D_values, stats, 
            os.path.join(output_dir, 'temporal_plots'), 
            'chunk', i+1
        )
    
    # Overall consistency assessment
    results['consistency'] = assess_consistency(results['temporal_results'])
    
    if not np.isnan(results['consistency']['overall_cv']):
        print(f"Temporal analysis overall coefficient of variation: {results['consistency']['overall_cv']:.4f}")
    else:
        print("Temporal analysis overall coefficient of variation: N/A")
    
    # Generate summary plots
    generate_summary_plots(
        results['consistency'], 
        os.path.join(output_dir, 'summary_plots'),
        'temporal'
    )
    
    return results

def analyze_tracking_noise(file_path, analysis_mode, options):
    """
    Perform tracking noise analysis using specified mode and options.
    
    Args:
        file_path: Path to the trajectory data file
        analysis_mode: Analysis mode ('partition', 'temporal', or 'both')
        options: Dictionary with analysis options
        
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
        'options': options,
        'analysis_mode': analysis_mode
    }
    
    # Create output directory if it doesn't exist
    output_dir = options['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform requested analysis
    if analysis_mode in ['partition', 'both']:
        # Get partition sizes
        partition_sizes = options.get('partition_sizes', PARTITION_SIZES)
        
        print("\n" + "="*50)
        print("PERFORMING PARTITION ANALYSIS")
        print("="*50)
        
        # Perform partition analysis
        partition_results = perform_partition_analysis(
            trajectories, partition_sizes, options['dt'], output_dir
        )
        
        # Add to results
        results['partition_results'] = partition_results['partition_results']
        results['partition_consistency'] = partition_results['consistency']
        
        # Save results to CSV
        save_results_to_csv(results, output_dir, 'partition')
    
    if analysis_mode in ['temporal', 'both']:
        # Get number of chunks
        num_chunks = options.get('num_chunks', DEFAULT_NUM_CHUNKS)
        
        print("\n" + "="*50)
        print("PERFORMING TEMPORAL ANALYSIS")
        print("="*50)
        
        # Perform temporal analysis
        temporal_results = perform_temporal_analysis(
            trajectories, num_chunks, options['dt'], output_dir
        )
        
        # Add to results
        results['temporal_results'] = temporal_results['temporal_results']
        results['temporal_consistency'] = temporal_results['consistency']
        
        # Save results to CSV
        save_results_to_csv(results, output_dir, 'temporal')
    
    # Save full results as pickle
    with open(os.path.join(output_dir, 'full_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    return results

def main():
    """Main function to run tracking noise analysis."""
    print("="*50)
    print("TRACKING NOISE ANALYSIS")
    print("="*50)
    
    # Get input file path
    input_file = input("Enter path to trajectory data file: ")
    
    if not os.path.isfile(input_file):
        print(f"Error: File {input_file} does not exist")
        return
    
    # Choose analysis mode
    print("\nAnalysis modes:")
    print("1. Partition trajectories into smaller chunks")
    print("2. Split movie into temporal segments")
    print("3. Both approaches")
    
    mode_choice = input("\nSelect analysis mode (1, 2, or 3): ")
    
    if mode_choice == '1':
        analysis_mode = 'partition'
    elif mode_choice == '2':
        analysis_mode = 'temporal'
    elif mode_choice == '3':
        analysis_mode = 'both'
    else:
        print("Invalid choice. Defaulting to both approaches.")
        analysis_mode = 'both'
    
    # Get parameters or use defaults
    options = {}
    
    # Time step
    dt_input = input(f"\nEnter time step in seconds (default: {DT}): ")
    if dt_input.strip():
        try:
            options['dt'] = float(dt_input)
        except ValueError:
            print(f"Invalid input. Using default time step: {DT}")
            options['dt'] = DT
    else:
        options['dt'] = DT
    
    # Partition sizes (if applicable)
    if analysis_mode in ['partition', 'both']:
        partition_sizes_input = input(f"\nEnter partition sizes separated by commas (default: {','.join(map(str, PARTITION_SIZES))}): ")
        if partition_sizes_input.strip():
            try:
                options['partition_sizes'] = [int(x.strip()) for x in partition_sizes_input.split(',')]
            except ValueError:
                print(f"Invalid input. Using default partition sizes: {PARTITION_SIZES}")
                options['partition_sizes'] = PARTITION_SIZES
        else:
            options['partition_sizes'] = PARTITION_SIZES
    
    # Number of temporal chunks (if applicable)
    if analysis_mode in ['temporal', 'both']:
        num_chunks_input = input(f"\nEnter number of temporal chunks (default: {DEFAULT_NUM_CHUNKS}): ")
        if num_chunks_input.strip():
            try:
                options['num_chunks'] = int(num_chunks_input)
            except ValueError:
                print(f"Invalid input. Using default number of chunks: {DEFAULT_NUM_CHUNKS}")
                options['num_chunks'] = DEFAULT_NUM_CHUNKS
        else:
            options['num_chunks'] = DEFAULT_NUM_CHUNKS
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.join(os.path.dirname(input_file), f"noise_analysis_{base_name}")
    os.makedirs(output_dir, exist_ok=True)
    options['output_dir'] = output_dir
    
    print(f"\nAnalyzing file: {input_file}")
    print(f"Analysis mode: {analysis_mode}")
    print(f"Output directory: {output_dir}")
    
    # Perform analysis
    results = analyze_tracking_noise(input_file, analysis_mode, options)
    
    if results:
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        print(f"Results saved in: {output_dir}")
        
        # Print summary
        if analysis_mode in ['partition', 'both']:
            print("\nPartition Analysis Summary:")
            if np.isnan(results['partition_consistency']['overall_cv']):
                print("Overall coefficient of variation: N/A")
            else:
                print(f"Overall coefficient of variation: {results['partition_consistency']['overall_cv']:.4f}")
            
            if not np.isnan(results['partition_consistency']['trend_slope']):
                print(f"Trend slope: {results['partition_consistency']['trend_slope']:.6f}")
                if abs(results['partition_consistency']['trend_slope']) > 0.01:
                    if results['partition_consistency']['trend_slope'] > 0:
                        print("  • Positive trend suggests superdiffusion at longer time scales")
                    else:
                        print("  • Negative trend suggests subdiffusion at longer time scales")
                else:
                    print("  • Negligible trend suggests normal diffusion")
        
        if analysis_mode in ['temporal', 'both']:
            print("\nTemporal Analysis Summary:")
            if np.isnan(results['temporal_consistency']['overall_cv']):
                print("Overall coefficient of variation: N/A")
            else:
                print(f"Overall coefficient of variation: {results['temporal_consistency']['overall_cv']:.4f}")
            
            if not np.isnan(results['temporal_consistency']['trend_slope']):
                print(f"Trend slope: {results['temporal_consistency']['trend_slope']:.6f}")
                if abs(results['temporal_consistency']['trend_slope']) > 0.01:
                    if results['temporal_consistency']['trend_slope'] > 0:
                        print("  • Positive trend suggests increasing diffusion over experiment time")
                    else:
                        print("  • Negative trend suggests decreasing diffusion over experiment time")
                else:
                    print("  • Negligible trend suggests stable diffusion throughout experiment")
    else:
        print("Analysis failed.")

if __name__ == "__main__":
    main()