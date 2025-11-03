# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 18:47:29 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advanced_diffusion_stats.py

This script implements advanced statistical methods for analyzing diffusion data,
including anomalous diffusion detection, error estimation, and bootstrapping.

Input:
- ROI-assigned trajectory data from roi_loader.py (.pkl files)
- Analyzed diffusion data from roi_diffusion_analyzer.py (.pkl files)

Output:
- Advanced statistical metrics for diffusion analysis
- Visualizations of anomalous diffusion and error estimation
- Results saved as .pkl and .csv files

Usage:
python advanced_diffusion_stats.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import stats
from scipy.optimize import curve_fit
from pathlib import Path
from datetime import datetime

# Global parameters that can be modified
# =====================================
# Time step in seconds
DT = 0.1
# Minimum trajectory length for anomalous diffusion analysis
MIN_TRAJECTORY_LENGTH = 15
# Maximum lag time fraction for MSD analysis (e.g., 0.5 means use first half of MSD curve)
MAX_LAG_FRACTION = 0.5
# Number of bootstrap samples for error estimation
BOOTSTRAP_SAMPLES = 1000
# Output directory name format
OUTPUT_DIR_FORMAT = 'advanced_diffusion_analysis_%Y%m%d_%H%M%S'
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
        print(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def power_law_msd(t, D, alpha):
    """
    Power law MSD model for anomalous diffusion: MSD = 4*D*t^alpha
    
    Args:
        t: Time lag
        D: Generalized diffusion coefficient
        alpha: Anomalous diffusion exponent
        
    Returns:
        MSD values according to the model
    """
    return 4 * D * np.power(t, alpha)

def fit_anomalous_diffusion(time_data, msd_data):
    """
    Fit anomalous diffusion model to MSD curve.
    
    Args:
        time_data: Array of time lag values
        msd_data: Array of MSD values
        
    Returns:
        Dictionary with fitting parameters and metrics
    """
    # Determine maximum lag time index based on fraction
    max_points = int(len(time_data) * MAX_LAG_FRACTION)
    max_points = min(max(max_points, 3), len(time_data))
    
    # Extract data for fitting
    t_fit = time_data[:max_points]
    msd_fit = msd_data[:max_points]
    
    # Filter out any NaN values
    valid_indices = ~np.isnan(msd_fit)
    t_fit = t_fit[valid_indices]
    msd_fit = msd_fit[valid_indices]
    
    # Check if we have enough points after filtering
    if len(t_fit) < 3:
        return {
            'D': np.nan,
            'alpha': np.nan,
            'D_err': np.nan,
            'alpha_err': np.nan,
            'r_squared': np.nan,
            't_fit': t_fit,
            'msd_fit': msd_fit,
            'fit_values': np.nan
        }
    
    try:
        # Initial parameter guesses
        p0 = [0.1, 1.0]  # Initial D and alpha
        
        # Fit the power law model using log-log transformation for stability
        log_t = np.log(t_fit)
        log_msd = np.log(msd_fit)
        
        # Linear fit on log-log scale
        slope, intercept, r_value, _, _ = stats.linregress(log_t, log_msd)
        
        alpha_init = slope
        D_init = np.exp(intercept) / 4  # MSD = 4*D*t^alpha
        
        # Use these as initial guesses for non-linear fit
        p0 = [D_init, alpha_init]
        
        # Non-linear fit on original scale
        popt, pcov = curve_fit(power_law_msd, t_fit, msd_fit, p0=p0)
        D, alpha = popt
        D_err, alpha_err = np.sqrt(np.diag(pcov))
        
        # Calculate fit quality (R²)
        fit_values = power_law_msd(t_fit, D, alpha)
        residuals = msd_fit - fit_values
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((msd_fit - np.mean(msd_fit))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'D': D,
            'alpha': alpha,
            'D_err': D_err,
            'alpha_err': alpha_err,
            'r_squared': r_squared,
            't_fit': t_fit,
            'msd_fit': msd_fit,
            'fit_values': fit_values
        }
    except Exception as e:
        print(f"Error during anomalous diffusion fitting: {e}")
        return {
            'D': np.nan,
            'alpha': np.nan,
            'D_err': np.nan,
            'alpha_err': np.nan,
            'r_squared': np.nan,
            't_fit': t_fit,
            'msd_fit': msd_fit,
            'fit_values': np.nan
        }

def calculate_msd(x, y, dt, max_lag=None):
    """
    Calculate mean squared displacement for a trajectory.
    
    Args:
        x: X coordinates
        y: Y coordinates
        dt: Time step
        max_lag: Maximum lag time to calculate (in frames)
        
    Returns:
        Tuple of (MSD values, time lags)
    """
    n = len(x)
    
    # Determine maximum lag time
    if max_lag is None:
        max_lag = n - 1
    else:
        max_lag = min(max_lag, n - 1)
    
    # Initialize arrays
    msd = np.zeros(max_lag)
    count = np.zeros(max_lag)
    
    # Calculate displacement for each time lag
    for lag in range(1, max_lag + 1):
        # Pre-compute squared displacements for efficiency
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        dr2 = dx**2 + dy**2
        
        # Sum squared displacements
        msd[lag - 1] = np.mean(dr2)
    
    # Create time array
    time_lags = np.arange(1, max_lag + 1) * dt
    
    return msd, time_lags

def calculate_ergodicity_breaking_parameter(trajectories, dt):
    """
    Calculate ergodicity breaking parameter to detect non-ergodic processes.
    
    Args:
        trajectories: List of trajectory dictionaries
        dt: Time step in seconds
        
    Returns:
        Dictionary with ergodicity breaking parameters for different lag times
    """
    # Initialize results
    results = {
        'lag_times': [],
        'eb_parameter': [],
        'eb_error': []
    }
    
    # Determine common lag times to analyze
    # Use lags corresponding to 10%, 25%, 50%, 75% of the shortest trajectory
    min_length = min([len(traj['x']) for traj in trajectories])
    lag_fractions = [0.1, 0.25, 0.5, 0.75]
    lag_times = [int(min_length * fraction) for fraction in lag_fractions]
    lag_times = [lag for lag in lag_times if lag >= 2]  # Ensure lags are at least 2 frames
    
    if not lag_times:
        print("Trajectories too short for ergodicity analysis")
        return results
    
    # Calculate time-averaged MSD for each trajectory at each lag time
    for lag in lag_times:
        tamsd_values = []
        
        for traj in trajectories:
            x = traj['x']
            y = traj['y']
            
            # Skip trajectories shorter than needed
            if len(x) <= lag:
                continue
            
            # Calculate time-averaged MSD for this lag
            dx = x[lag:] - x[:-lag]
            dy = y[lag:] - y[:-lag]
            dr2 = dx**2 + dy**2
            tamsd = np.mean(dr2)
            
            tamsd_values.append(tamsd)
        
        if len(tamsd_values) >= 2:
            # Calculate ergodicity breaking parameter: EB = var(TAMSD)/mean(TAMSD)^2
            mean_tamsd = np.mean(tamsd_values)
            var_tamsd = np.var(tamsd_values)
            eb = var_tamsd / (mean_tamsd**2)
            
            # Calculate error using bootstrap
            eb_bootstrap = []
            for _ in range(100):  # 100 bootstrap samples
                sample = np.random.choice(tamsd_values, size=len(tamsd_values), replace=True)
                mean_sample = np.mean(sample)
                var_sample = np.var(sample)
                eb_sample = var_sample / (mean_sample**2)
                eb_bootstrap.append(eb_sample)
            
            eb_error = np.std(eb_bootstrap)
            
            results['lag_times'].append(lag * dt)
            results['eb_parameter'].append(eb)
            results['eb_error'].append(eb_error)
    
    return results

def bootstrap_diffusion_coefficient(trajectories, n_samples=BOOTSTRAP_SAMPLES):
    """
    Estimate diffusion coefficient uncertainty using bootstrap resampling.
    
    Args:
        trajectories: List of trajectory dictionaries
        n_samples: Number of bootstrap samples
        
    Returns:
        Dictionary with bootstrap statistics
    """
    # Extract diffusion coefficients
    D_values = [traj['D'] for traj in trajectories if 'D' in traj and not np.isnan(traj['D'])]
    
    if len(D_values) < 5:
        print("Not enough valid trajectories for bootstrap analysis")
        return {
            'mean_D': np.mean(D_values) if D_values else np.nan,
            'median_D': np.median(D_values) if D_values else np.nan,
            'std_D': np.std(D_values) if len(D_values) > 1 else np.nan,
            'bootstrap_mean': [] if not D_values else [np.mean(D_values)],
            'bootstrap_median': [] if not D_values else [np.median(D_values)],
            'ci_low': np.nan,
            'ci_high': np.nan
        }
    
    # Perform bootstrap resampling
    bootstrap_means = []
    bootstrap_medians = []
    
    for _ in range(n_samples):
        # Sample with replacement
        sample = np.random.choice(D_values, size=len(D_values), replace=True)
        bootstrap_means.append(np.mean(sample))
        bootstrap_medians.append(np.median(sample))
    
    # Calculate 95% confidence intervals
    ci_low = np.percentile(bootstrap_means, 2.5)
    ci_high = np.percentile(bootstrap_means, 97.5)
    
    return {
        'mean_D': np.mean(D_values),
        'median_D': np.median(D_values),
        'std_D': np.std(D_values),
        'bootstrap_mean': bootstrap_means,
        'bootstrap_median': bootstrap_medians,
        'ci_low': ci_low,
        'ci_high': ci_high
    }

def analyze_anomalous_diffusion(roi_data):
    """
    Analyze anomalous diffusion characteristics for trajectories in each ROI.
    
    Args:
        roi_data: Dictionary containing ROI trajectory data
        
    Returns:
        Dictionary with anomalous diffusion analysis results
    """
    results = {
        'roi_anomalous_diffusion': {}
    }
    
    for roi_id, trajectories in roi_data['roi_trajectories'].items():
        roi_results = {
            'trajectories': [],
            'D_values': [],
            'alpha_values': [],
            'r_squared_values': []
        }
        
        for traj in trajectories:
            # Skip trajectories that are too short
            if len(traj['x']) < MIN_TRAJECTORY_LENGTH:
                continue
            
            # Calculate MSD
            msd, time_lags = calculate_msd(traj['x'], traj['y'], DT)
            
            # Fit anomalous diffusion model
            fit_results = fit_anomalous_diffusion(time_lags, msd)
            
            # Store results if fit was successful
            if not np.isnan(fit_results['D']) and not np.isnan(fit_results['alpha']):
                traj_analysis = {
                    'id': traj['id'],
                    'D': fit_results['D'],
                    'alpha': fit_results['alpha'],
                    'D_err': fit_results['D_err'],
                    'alpha_err': fit_results['alpha_err'],
                    'r_squared': fit_results['r_squared'],
                    'msd': msd,
                    'time_lags': time_lags,
                    't_fit': fit_results['t_fit'],
                    'msd_fit': fit_results['msd_fit'],
                    'fit_values': fit_results['fit_values']
                }
                
                roi_results['trajectories'].append(traj_analysis)
                roi_results['D_values'].append(fit_results['D'])
                roi_results['alpha_values'].append(fit_results['alpha'])
                roi_results['r_squared_values'].append(fit_results['r_squared'])
        
        # Calculate summary statistics
        if roi_results['alpha_values']:
            roi_results['mean_alpha'] = np.mean(roi_results['alpha_values'])
            roi_results['median_alpha'] = np.median(roi_results['alpha_values'])
            roi_results['std_alpha'] = np.std(roi_results['alpha_values'])
            
            # Check diffusion type based on alpha
            roi_results['subdiffusion'] = sum(np.array(roi_results['alpha_values']) < 0.9) / len(roi_results['alpha_values'])
            roi_results['normal_diffusion'] = sum((np.array(roi_results['alpha_values']) >= 0.9) & 
                                                 (np.array(roi_results['alpha_values']) <= 1.1)) / len(roi_results['alpha_values'])
            roi_results['superdiffusion'] = sum(np.array(roi_results['alpha_values']) > 1.1) / len(roi_results['alpha_values'])
        
        results['roi_anomalous_diffusion'][roi_id] = roi_results
    
    return results

def analyze_ergodicity(roi_data):
    """
    Analyze ergodicity properties for trajectories in each ROI.
    
    Args:
        roi_data: Dictionary containing ROI trajectory data
        
    Returns:
        Dictionary with ergodicity analysis results
    """
    results = {
        'roi_ergodicity': {}
    }
    
    for roi_id, trajectories in roi_data['roi_trajectories'].items():
        # Skip ROIs with too few trajectories
        if len(trajectories) < 5:
            continue
        
        # Calculate ergodicity breaking parameter
        eb_results = calculate_ergodicity_breaking_parameter(trajectories, DT)
        
        # Store results if analysis was successful
        if eb_results['lag_times']:
            results['roi_ergodicity'][roi_id] = eb_results
    
    return results

def analyze_bootstrap(roi_data):
    """
    Perform bootstrap analysis for each ROI.
    
    Args:
        roi_data: Dictionary containing ROI trajectory data
        
    Returns:
        Dictionary with bootstrap analysis results
    """
    results = {
        'roi_bootstrap': {}
    }
    
    for roi_id, trajectories in roi_data['roi_trajectories'].items():
        # Skip ROIs with too few trajectories
        if len(trajectories) < 5:
            continue
        
        # Perform bootstrap analysis
        bootstrap_results = bootstrap_diffusion_coefficient(trajectories)
        
        # Store results
        results['roi_bootstrap'][roi_id] = bootstrap_results
    
    return results

def visualize_anomalous_diffusion(analysis_results, output_dir):
    """
    Create visualizations for anomalous diffusion analysis.
    
    Args:
        analysis_results: Dictionary with analysis results
        output_dir: Directory to save visualizations
        
    Returns:
        None (saves visualizations to files)
    """
    # 1. Alpha distribution across ROIs
    plt.figure(figsize=(10, 6))
    
    roi_ids = []
    mean_alphas = []
    std_alphas = []
    counts = []
    
    for roi_id, results in analysis_results['roi_anomalous_diffusion'].items():
        if 'mean_alpha' in results and not np.isnan(results['mean_alpha']):
            roi_ids.append(roi_id.split('-')[0][:8])  # Shortened ROI ID
            mean_alphas.append(results['mean_alpha'])
            std_alphas.append(results['std_alpha'])
            counts.append(len(results['alpha_values']))
    
    if not roi_ids:
        print("No valid anomalous diffusion results for visualization")
        return
    
    # Sort by mean alpha
    sorted_indices = np.argsort(mean_alphas)
    roi_ids = [roi_ids[i] for i in sorted_indices]
    mean_alphas = [mean_alphas[i] for i in sorted_indices]
    std_alphas = [std_alphas[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    # Create bar plot with error bars
    bar_positions = np.arange(len(roi_ids))
    bars = plt.bar(bar_positions, mean_alphas, yerr=std_alphas, 
                  color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add reference lines
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    plt.axhline(y=0.9, color='r', linestyle=':', alpha=0.5)
    plt.axhline(y=1.1, color='r', linestyle=':', alpha=0.5)
    
    # Add annotations
    for i, count in enumerate(counts):
        plt.text(i, mean_alphas[i] + std_alphas[i] + 0.05, f"n={count}", 
                ha='center', va='bottom', fontsize=8)
    
    # Set labels and title
    plt.xlabel('Region of Interest', fontsize=12)
    plt.ylabel('Anomalous Diffusion Exponent (α)', fontsize=12)
    plt.title('Anomalous Diffusion Analysis by ROI', fontsize=14)
    
    # Set x-tick labels
    plt.xticks(bar_positions, roi_ids, rotation=45)
    
    # Add grid and tight layout
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, max(2.0, max(np.array(mean_alphas) + np.array(std_alphas)) + 0.2))
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'anomalous_diffusion_by_roi.png'), dpi=300)
    plt.close()
    
    # 2. Diffusion type distribution
    plt.figure(figsize=(12, 6))
    
    roi_ids = []
    subdiffusion_fracs = []
    normal_diffusion_fracs = []
    superdiffusion_fracs = []
    
    for roi_id, results in analysis_results['roi_anomalous_diffusion'].items():
        if 'subdiffusion' in results:
            roi_ids.append(roi_id.split('-')[0][:8])  # Shortened ROI ID
            subdiffusion_fracs.append(results['subdiffusion'])
            normal_diffusion_fracs.append(results['normal_diffusion'])
            superdiffusion_fracs.append(results['superdiffusion'])
    
    if not roi_ids:
        return
    
    # Sort by normal diffusion fraction
    sorted_indices = np.argsort(normal_diffusion_fracs)
    roi_ids = [roi_ids[i] for i in sorted_indices]
    subdiffusion_fracs = [subdiffusion_fracs[i] for i in sorted_indices]
    normal_diffusion_fracs = [normal_diffusion_fracs[i] for i in sorted_indices]
    superdiffusion_fracs = [superdiffusion_fracs[i] for i in sorted_indices]
    
    # Create stacked bar plot
    bar_width = 0.8
    bar_positions = np.arange(len(roi_ids))
    
    plt.bar(bar_positions, subdiffusion_fracs, bar_width, 
            label='Subdiffusion (α < 0.9)', color='blue', alpha=0.7)
    plt.bar(bar_positions, normal_diffusion_fracs, bar_width, 
            bottom=subdiffusion_fracs, label='Normal (0.9 ≤ α ≤ 1.1)', color='green', alpha=0.7)
    plt.bar(bar_positions, superdiffusion_fracs, bar_width, 
            bottom=np.array(subdiffusion_fracs) + np.array(normal_diffusion_fracs), 
            label='Superdiffusion (α > 1.1)', color='red', alpha=0.7)
    
    # Set labels and title
    plt.xlabel('Region of Interest', fontsize=12)
    plt.ylabel('Fraction of Trajectories', fontsize=12)
    plt.title('Distribution of Diffusion Types by ROI', fontsize=14)
    
    # Set x-tick labels
    plt.xticks(bar_positions, roi_ids, rotation=45)
    
    # Add legend
    plt.legend(loc='upper right')
    
    # Add grid and tight layout
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'diffusion_type_distribution.png'), dpi=300)
    plt.close()
    
    # 3. Example MSD fits for selected trajectories
    # Select one trajectory with each type of diffusion
    example_fits = []
    
    for roi_id, results in analysis_results['roi_anomalous_diffusion'].items():
        if not results['trajectories']:
            continue
            
        # Try to find examples of each diffusion type
        for traj in results['trajectories']:
            if len(example_fits) >= 3:
                break
                
            # Skip if we already have a trajectory of this type
            if len(example_fits) == 0 and traj['alpha'] < 0.9:  # Subdiffusion
                example_fits.append((roi_id, traj, 'Subdiffusion'))
            elif len(example_fits) == 1 and traj['alpha'] >= 0.9 and traj['alpha'] <= 1.1:  # Normal
                example_fits.append((roi_id, traj, 'Normal Diffusion'))
            elif len(example_fits) == 2 and traj['alpha'] > 1.1:  # Superdiffusion
                example_fits.append((roi_id, traj, 'Superdiffusion'))
    
    if example_fits:
        plt.figure(figsize=(15, 5*len(example_fits)))
        
        for i, (roi_id, traj, diff_type) in enumerate(example_fits):
            plt.subplot(len(example_fits), 1, i+1)
            
            # Plot MSD data
            plt.loglog(traj['time_lags'], traj['msd'], 'o', label='MSD data')
            
            # Plot fit
            if not np.isnan(traj['D']) and not np.isnan(traj['alpha']):
                # Extend fit line
                t_extended = np.logspace(np.log10(traj['time_lags'][0]), 
                                         np.log10(traj['time_lags'][-1]), 100)
                msd_extended = power_law_msd(t_extended, traj['D'], traj['alpha'])
                
                plt.loglog(t_extended, msd_extended, '--', 
                          label=f'Fit: D={traj["D"]:.3f}, α={traj["alpha"]:.3f}')
                
                # Highlight fitted region
                plt.loglog(traj['t_fit'], traj['msd_fit'], 'o', color='red', label='Fitted points')
            
            # Set labels and title
            plt.xlabel('Time lag (s)', fontsize=12)
            plt.ylabel('MSD (μm²)', fontsize=12)
            plt.title(f'{diff_type} Example (ROI: {roi_id.split("-")[0]}, Trajectory: {traj["id"]})', 
                     fontsize=14)
            
            # Add grid and legend
            plt.grid(True, alpha=0.3, which='both')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'example_msd_fits.png'), dpi=300)
        plt.close()

def visualize_ergodicity(analysis_results, output_dir):
    """
    Create visualizations for ergodicity analysis.
    
    Args:
        analysis_results: Dictionary with analysis results
        output_dir: Directory to save visualizations
        
    Returns:
        None (saves visualizations to files)
    """
    if 'roi_ergodicity' not in analysis_results or not analysis_results['roi_ergodicity']:
        print("No valid ergodicity results for visualization")
        return
    
    # 1. Ergodicity breaking parameter vs. lag time for each ROI
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(analysis_results['roi_ergodicity'])))
    
    for i, (roi_id, results) in enumerate(analysis_results['roi_ergodicity'].items()):
        color = colors[i % len(colors)]
        roi_label = roi_id.split('-')[0][:8]  # Shortened ROI ID
        
        plt.errorbar(results['lag_times'], results['eb_parameter'], yerr=results['eb_error'], 
                    fmt='o-', color=color, label=roi_label)
    
    # Add reference line for ergodic processes
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Set logarithmic scale for y-axis
    plt.yscale('log')
    
    # Set labels and title
    plt.xlabel('Lag Time (s)', fontsize=12)
    plt.ylabel('Ergodicity Breaking Parameter', fontsize=12)
    plt.title('Ergodicity Analysis by ROI', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ergodicity_breaking.png'), dpi=300)
    plt.close()

def visualize_bootstrap(analysis_results, output_dir):
    """
    Create visualizations for bootstrap analysis.
    
    Args:
        analysis_results: Dictionary with analysis results
        output_dir: Directory to save visualizations
        
    Returns:
        None (saves visualizations to files)
    """
    if 'roi_bootstrap' not in analysis_results or not analysis_results['roi_bootstrap']:
        print("No valid bootstrap results for visualization")
        return
    
    # 1. Bootstrap distribution of mean diffusion coefficient for each ROI
    plt.figure(figsize=(12, 8))
    
    roi_ids = []
    mean_Ds = []
    ci_lows = []
    ci_highs = []
    
    for roi_id, results in analysis_results['roi_bootstrap'].items():
        if 'mean_D' in results and not np.isnan(results['mean_D']):
            roi_ids.append(roi_id.split('-')[0][:8])  # Shortened ROI ID
            mean_Ds.append(results['mean_D'])
            ci_lows.append(results['ci_low'])
            ci_highs.append(results['ci_high'])
    
    if not roi_ids:
        return
    
    # Sort by mean diffusion coefficient
    sorted_indices = np.argsort(mean_Ds)
    roi_ids = [roi_ids[i] for i in sorted_indices]
    mean_Ds = [mean_Ds[i] for i in sorted_indices]
    ci_lows = [ci_lows[i] for i in sorted_indices]
    ci_highs = [ci_highs[i] for i in sorted_indices]
    
    # Calculate error bars
    error_low = [mean - low for mean, low in zip(mean_Ds, ci_lows)]
    error_high = [high - mean for mean, high in zip(mean_Ds, ci_highs)]
    
    # Create bar plot with asymmetric error bars
    bar_positions = np.arange(len(roi_ids))
    plt.bar(bar_positions, mean_Ds, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add error bars
    plt.errorbar(bar_positions, mean_Ds, yerr=[error_low, error_high], fmt='none', 
                ecolor='black', capsize=5)
    
    # Set labels and title
    plt.xlabel('Region of Interest', fontsize=12)
    plt.ylabel('Diffusion Coefficient (μm²/s)', fontsize=12)
    plt.title('Bootstrap Analysis of Diffusion Coefficients by ROI', fontsize=14)
    
    # Set x-tick labels
    plt.xticks(bar_positions, roi_ids, rotation=45)
    
    # Add grid
    plt.grid(axis='y', alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bootstrap_diffusion.png'), dpi=300)
    plt.close()
    
    # 2. Detailed bootstrap distributions for selected ROIs
    # Select up to 4 ROIs with the most diverse mean diffusion coefficients
    if len(roi_ids) > 1:
        indices = [0, len(roi_ids)-1]  # Smallest and largest
        if len(roi_ids) > 3:
            mid_idx = len(roi_ids) // 2
            indices.extend([mid_idx-1, mid_idx+1])
        
        selected_roi_ids = [roi_ids[i] for i in indices if i < len(roi_ids)]
        
        plt.figure(figsize=(10, 8))
        
        for roi_id in selected_roi_ids:
            idx = roi_ids.index(roi_id)
            original_roi_id = list(analysis_results['roi_bootstrap'].keys())[sorted_indices[idx]]
            bootstrap_results = analysis_results['roi_bootstrap'][original_roi_id]
            
            # Plot bootstrap distribution
            sns.kdeplot(bootstrap_results['bootstrap_mean'], label=f'{roi_id} (D={mean_Ds[idx]:.4f})')
        
        # Set labels and title
        plt.xlabel('Diffusion Coefficient (μm²/s)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Bootstrap Distributions of Diffusion Coefficients', fontsize=14)
        
        # Add legend and grid
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bootstrap_distributions.png'), dpi=300)
        plt.close()

def export_results_to_csv(analysis_results, output_dir):
    """
    Export analysis results to CSV files.
    
    Args:
        analysis_results: Dictionary with analysis results
        output_dir: Directory to save CSV files
        
    Returns:
        None (saves CSV files)
    """
    # 1. Anomalous diffusion statistics by ROI
    if 'roi_anomalous_diffusion' in analysis_results:
        anomalous_stats = []
        
        for roi_id, results in analysis_results['roi_anomalous_diffusion'].items():
            if 'mean_alpha' in results:
                anomalous_stats.append({
                    'ROI_ID': roi_id,
                    'NumTrajectories': len(results['alpha_values']),
                    'MeanAlpha': results['mean_alpha'],
                    'MedianAlpha': results['median_alpha'],
                    'StdAlpha': results['std_alpha'],
                    'SubdiffusionFraction': results.get('subdiffusion', np.nan),
                    'NormalDiffusionFraction': results.get('normal_diffusion', np.nan),
                    'SuperdiffusionFraction': results.get('superdiffusion', np.nan)
                })
        
        if anomalous_stats:
            pd.DataFrame(anomalous_stats).to_csv(
                os.path.join(output_dir, 'anomalous_diffusion_statistics.csv'), index=False)
    
    # 2. Individual trajectory anomalous diffusion parameters
    if 'roi_anomalous_diffusion' in analysis_results:
        all_trajectories = []
        
        for roi_id, results in analysis_results['roi_anomalous_diffusion'].items():
            for traj in results['trajectories']:
                all_trajectories.append({
                    'ROI_ID': roi_id,
                    'Trajectory_ID': traj['id'],
                    'D': traj['D'],
                    'Alpha': traj['alpha'],
                    'D_error': traj['D_err'],
                    'Alpha_error': traj['alpha_err'],
                    'R_squared': traj['r_squared']
                })
        
        if all_trajectories:
            pd.DataFrame(all_trajectories).to_csv(
                os.path.join(output_dir, 'all_anomalous_diffusion_parameters.csv'), index=False)
    
    # 3. Ergodicity breaking parameters
    if 'roi_ergodicity' in analysis_results:
        all_eb_data = []
        
        for roi_id, results in analysis_results['roi_ergodicity'].items():
            for i, lag_time in enumerate(results['lag_times']):
                all_eb_data.append({
                    'ROI_ID': roi_id,
                    'LagTime': lag_time,
                    'EB_Parameter': results['eb_parameter'][i],
                    'EB_Error': results['eb_error'][i]
                })
        
        if all_eb_data:
            pd.DataFrame(all_eb_data).to_csv(
                os.path.join(output_dir, 'ergodicity_breaking_parameters.csv'), index=False)
    
    # 4. Bootstrap confidence intervals
    if 'roi_bootstrap' in analysis_results:
        bootstrap_stats = []
        
        for roi_id, results in analysis_results['roi_bootstrap'].items():
            if 'mean_D' in results and not np.isnan(results['mean_D']):
                bootstrap_stats.append({
                    'ROI_ID': roi_id,
                    'MeanD': results['mean_D'],
                    'MedianD': results['median_D'],
                    'StdD': results['std_D'],
                    'CI_Low': results['ci_low'],
                    'CI_High': results['ci_high']
                })
        
        if bootstrap_stats:
            pd.DataFrame(bootstrap_stats).to_csv(
                os.path.join(output_dir, 'bootstrap_confidence_intervals.csv'), index=False)

def main():
    """
    Main function to perform advanced statistical analysis of diffusion data.
    """
    # Ask for input path
    roi_data_file = input("Enter path to ROI trajectory data file (.pkl): ")
    
    # Load ROI data
    roi_data = load_data(roi_data_file)
    if roi_data is None:
        print("Failed to load ROI data. Exiting.")
        return
    
    # Create output directory
    output_dir_name = datetime.now().strftime(OUTPUT_DIR_FORMAT)
    output_base_dir = os.path.dirname(roi_data_file)
    output_dir = os.path.join(output_base_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze anomalous diffusion
    print("\nAnalyzing anomalous diffusion characteristics...")
    anomalous_results = analyze_anomalous_diffusion(roi_data)
    
    # Analyze ergodicity
    print("Analyzing ergodicity...")
    ergodicity_results = analyze_ergodicity(roi_data)
    
    # Perform bootstrap analysis
    print("Performing bootstrap analysis...")
    bootstrap_results = analyze_bootstrap(roi_data)
    
    # Combine results
    analysis_results = {
        'roi_anomalous_diffusion': anomalous_results['roi_anomalous_diffusion'],
        'roi_ergodicity': ergodicity_results['roi_ergodicity'],
        'roi_bootstrap': bootstrap_results['roi_bootstrap']
    }
    
    # Create visualizations
    print("Generating visualizations...")
    visualize_anomalous_diffusion(analysis_results, output_dir)
    visualize_ergodicity(analysis_results, output_dir)
    visualize_bootstrap(analysis_results, output_dir)
    
    # Export results
    export_results_to_csv(analysis_results, output_dir)
    
    # Save full results
    output_file = os.path.join(output_dir, 'advanced_diffusion_analysis.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(analysis_results, f)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()