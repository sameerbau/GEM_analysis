# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 15:53:52 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advanced_diffusion_validation.py

Additional statistical mechanics methods to validate diffusion coefficient calculations.
Complements the existing comparison script with physics-based validation approaches.

Methods included:
1. Mean First Passage Time (MFPT) analysis
2. Hurst exponent calculation
3. Ergodicity breaking parameter
4. Time-averaged vs ensemble-averaged MSD
5. Non-Gaussian parameter analysis
6. Velocity increment distribution analysis
7. Step-length distribution analysis

1. Hurst Exponent (H)

Purpose: Detects anomalous diffusion types
Normal diffusion: H = 0.5
Super-diffusion: H > 0.5 (persistent motion)
Sub-diffusion: H < 0.5 (anti-persistent motion)

2. Non-Gaussian Parameter (α₂)

Purpose: Detects deviations from normal diffusion
Normal: α₂ = 0 (Gaussian displacement distribution)
Anomalous: α₂ > 0 (heavy-tailed distributions)

3. Ergodicity Breaking Parameter (EB)

Purpose: Tests if ensemble average = time average
Ergodic: EB ≈ 0 (classical diffusion)
Non-ergodic: EB > 0 (aging, heterogeneity)

4. Velocity Increment Analysis

Purpose: Tests for jump diffusion or active transport
Normal: Gaussian velocity increments
Anomalous: Non-Gaussian (heavy tails, skewness)

5. Step Length Distribution

Purpose: Detects jump diffusion
Normal: Rayleigh distribution
Jump diffusion: Heavy-tailed distributions

Warning Signs Your Diffusion Calculations May Be Faulty
Based on these methods, be suspicious if you see:

Hurst exponent H significantly ≠ 0.5 → Not simple diffusion
High non-Gaussian parameter α₂ > 0.1 → Complex motion
Ergodicity breaking EB > 0.1 → Time/ensemble averages differ
Non-Gaussian velocity increments → Active transport/jumps
High bootstrap uncertainty >20% → Need longer trajectories
Allan variance α > 0.5 → Undersampling correlations 
Global parameters for validation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import gamma
import warnings

# Global parameters (modify these as needed)
# =====================================
# Time step in seconds
DT = 0.1
# Minimum trajectory length for analysis
MIN_TRACK_LENGTH = 15
# Maximum lag for MSD analysis (fraction of track length)
MAX_LAG_FRACTION = 0.3
# Number of bootstrap samples for error estimation
N_BOOTSTRAP = 100
# Significance level for statistical tests
ALPHA = 0.05
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

def extract_trajectories(data, min_length=MIN_TRACK_LENGTH):
    """Extract trajectory data for advanced analysis."""
    if 'trajectories' not in data:
        return []
    
    filtered_trajectories = []
    for traj in data['trajectories']:
        if len(traj['x']) >= min_length and not np.isnan(traj.get('D', np.nan)):
            filtered_trajectories.append({
                'x': np.array(traj['x']),
                'y': np.array(traj['y']),
                'D': traj.get('D', np.nan),
                'id': traj.get('id', 'unknown')
            })
    
    return filtered_trajectories

def calculate_hurst_exponent(trajectory, dt=DT):
    """
    Calculate Hurst exponent using rescaled range (R/S) analysis.
    
    H = 0.5: Normal diffusion (Brownian motion)
    H > 0.5: Super-diffusion (persistent motion)
    H < 0.5: Sub-diffusion (anti-persistent motion)
    
    Args:
        trajectory: Dictionary with 'x', 'y' coordinates
        dt: Time step
        
    Returns:
        Dictionary with Hurst exponent results
    """
    x, y = trajectory['x'], trajectory['y']
    
    # Calculate displacement series
    dx = np.diff(x)
    dy = np.diff(y)
    displacement_series = np.sqrt(dx**2 + dy**2)
    
    # Remove mean
    displacement_series = displacement_series - np.mean(displacement_series)
    
    # Calculate R/S for different window sizes
    n_values = []
    rs_values = []
    
    max_n = len(displacement_series) // 4
    
    for n in range(10, max_n, max(1, max_n//20)):
        # Number of complete windows
        n_windows = len(displacement_series) // n
        if n_windows < 2:
            continue
        
        rs_list = []
        
        for i in range(n_windows):
            window = displacement_series[i*n:(i+1)*n]
            
            # Calculate cumulative sum
            cumsum = np.cumsum(window)
            
            # Calculate range
            R = np.max(cumsum) - np.min(cumsum)
            
            # Calculate standard deviation
            S = np.std(window)
            
            if S > 0:
                rs_list.append(R / S)
        
        if rs_list:
            n_values.append(n)
            rs_values.append(np.mean(rs_list))
    
    if len(n_values) < 3:
        return None
    
    # Fit log(R/S) vs log(n) to get Hurst exponent
    try:
        log_n = np.log10(n_values)
        log_rs = np.log10(rs_values)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_rs)
        
        return {
            'hurst_exponent': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'n_values': n_values,
            'rs_values': rs_values
        }
    except:
        return None

def calculate_ergodicity_breaking_parameter(trajectories, dt=DT):
    """
    Calculate ergodicity breaking parameter (EB).
    
    EB measures the difference between time-averaged and ensemble-averaged MSD.
    EB ≈ 0: Ergodic system
    EB > 0: Non-ergodic (ergodicity breaking)
    
    Args:
        trajectories: List of trajectory dictionaries
        dt: Time step
        
    Returns:
        Dictionary with EB results
    """
    if len(trajectories) < 2:
        return None
    
    # Calculate time-averaged MSD for each trajectory
    time_averaged_msds = []
    max_lag = min(int(len(traj['x']) * MAX_LAG_FRACTION) for traj in trajectories)
    
    for traj in trajectories:
        x, y = traj['x'], traj['y']
        ta_msd = []
        
        for lag in range(1, max_lag):
            if len(x) <= lag:
                break
            
            # Time-averaged MSD for this trajectory
            squared_displacements = []
            for i in range(len(x) - lag):
                dx = x[i + lag] - x[i]
                dy = y[i + lag] - y[i]
                squared_displacements.append(dx**2 + dy**2)
            
            ta_msd.append(np.mean(squared_displacements))
        
        time_averaged_msds.append(ta_msd)
    
    # Calculate ensemble-averaged MSD
    min_length = min(len(ta_msd) for ta_msd in time_averaged_msds)
    ensemble_msd = []
    
    for lag_idx in range(min_length):
        lag_values = [ta_msd[lag_idx] for ta_msd in time_averaged_msds]
        ensemble_msd.append(np.mean(lag_values))
    
    # Calculate EB parameter
    eb_values = []
    for i, ta_msd in enumerate(time_averaged_msds):
        if len(ta_msd) >= min_length:
            # Calculate relative variance
            numerator = 0
            denominator = 0
            
            for lag_idx in range(min_length):
                if ensemble_msd[lag_idx] > 0:
                    numerator += (ta_msd[lag_idx] - ensemble_msd[lag_idx])**2
                    denominator += ensemble_msd[lag_idx]**2
            
            if denominator > 0:
                eb_values.append(numerator / denominator)
    
    if not eb_values:
        return None
    
    return {
        'eb_parameter': np.mean(eb_values),
        'eb_std': np.std(eb_values),
        'n_trajectories': len(eb_values),
        'time_lags': np.arange(1, min_length + 1) * dt
    }

def calculate_non_gaussian_parameter(trajectories, dt=DT, lag_times=None):
    """
    Calculate non-Gaussian parameter α₂(t).
    
    α₂ = 0: Gaussian distribution (normal diffusion)
    α₂ > 0: Non-Gaussian behavior
    
    Args:
        trajectories: List of trajectory dictionaries
        dt: Time step
        lag_times: Specific lag times to analyze (frames)
        
    Returns:
        Dictionary with non-Gaussian parameter results
    """
    if lag_times is None:
        max_lag = min(int(len(traj['x']) * MAX_LAG_FRACTION) for traj in trajectories)
        lag_times = range(1, max_lag, max(1, max_lag//20))
    
    alpha2_values = []
    time_points = []
    
    for lag in lag_times:
        all_displacements = []
        
        # Collect all displacements for this lag time
        for traj in trajectories:
            x, y = traj['x'], traj['y']
            
            if len(x) <= lag:
                continue
            
            for i in range(len(x) - lag):
                dx = x[i + lag] - x[i]
                dy = y[i + lag] - y[i]
                displacement_squared = dx**2 + dy**2
                all_displacements.append(displacement_squared)
        
        if len(all_displacements) < 10:
            continue
        
        # Calculate moments
        mean_r2 = np.mean(all_displacements)
        mean_r4 = np.mean(np.array(all_displacements)**2)
        
        # Non-Gaussian parameter
        if mean_r2 > 0:
            alpha2 = (mean_r4 / (2 * mean_r2**2)) - 1
            alpha2_values.append(alpha2)
            time_points.append(lag * dt)
    
    return {
        'alpha2_values': alpha2_values,
        'time_points': time_points,
        'mean_alpha2': np.mean(alpha2_values) if alpha2_values else np.nan
    }

def analyze_velocity_increments(trajectories, dt=DT):
    """
    Analyze velocity increment distributions for anomalous diffusion detection.
    
    Normal diffusion: Gaussian velocity increments
    Anomalous diffusion: Non-Gaussian velocity increments
    
    Args:
        trajectories: List of trajectory dictionaries
        dt: Time step
        
    Returns:
        Dictionary with velocity increment analysis
    """
    all_velocity_increments = []
    
    for traj in trajectories:
        x, y = traj['x'], traj['y']
        
        # Calculate velocities
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        
        # Calculate velocity increments
        dvx = np.diff(vx)
        dvy = np.diff(vy)
        
        # Combine into magnitude of velocity increments
        dv_magnitude = np.sqrt(dvx**2 + dvy**2)
        all_velocity_increments.extend(dv_magnitude)
    
    if not all_velocity_increments:
        return None
    
    # Test for normality
    shapiro_stat, shapiro_p = stats.shapiro(all_velocity_increments[:5000])  # Limit for computational efficiency
    
    # Calculate kurtosis (measure of tail heaviness)
    kurtosis = stats.kurtosis(all_velocity_increments)
    
    # Calculate skewness
    skewness = stats.skew(all_velocity_increments)
    
    return {
        'velocity_increments': all_velocity_increments,
        'shapiro_statistic': shapiro_stat,
        'shapiro_p_value': shapiro_p,
        'is_gaussian': shapiro_p > ALPHA,
        'kurtosis': kurtosis,
        'skewness': skewness,
        'mean': np.mean(all_velocity_increments),
        'std': np.std(all_velocity_increments)
    }

def analyze_step_length_distribution(trajectories):
    """
    Analyze step length distributions for jump diffusion detection.
    
    Normal diffusion: Rayleigh distribution
    Jump diffusion: Heavy-tailed distribution
    
    Args:
        trajectories: List of trajectory dictionaries
        
    Returns:
        Dictionary with step length analysis
    """
    all_step_lengths = []
    
    for traj in trajectories:
        x, y = traj['x'], traj['y']
        
        # Calculate step lengths
        dx = np.diff(x)
        dy = np.diff(y)
        step_lengths = np.sqrt(dx**2 + dy**2)
        
        all_step_lengths.extend(step_lengths)
    
    if not all_step_lengths:
        return None
    
    step_lengths = np.array(all_step_lengths)
    
    # Fit Rayleigh distribution
    try:
        rayleigh_param = stats.rayleigh.fit(step_lengths, floc=0)
        rayleigh_ks_stat, rayleigh_ks_p = stats.kstest(step_lengths, 
                                                      lambda x: stats.rayleigh.cdf(x, *rayleigh_param))
    except:
        rayleigh_ks_stat, rayleigh_ks_p = np.nan, np.nan
        rayleigh_param = (np.nan,)
    
    # Fit exponential distribution (for comparison)
    try:
        exp_param = stats.expon.fit(step_lengths)
        exp_ks_stat, exp_ks_p = stats.kstest(step_lengths, 
                                            lambda x: stats.expon.cdf(x, *exp_param))
    except:
        exp_ks_stat, exp_ks_p = np.nan, np.nan
        exp_param = (np.nan, np.nan)
    
    return {
        'step_lengths': step_lengths,
        'rayleigh_param': rayleigh_param,
        'rayleigh_ks_p': rayleigh_ks_p,
        'rayleigh_fits_well': rayleigh_ks_p > ALPHA,
        'exp_param': exp_param,
        'exp_ks_p': exp_ks_p,
        'exp_fits_well': exp_ks_p > ALPHA,
        'mean_step_length': np.mean(step_lengths),
        'std_step_length': np.std(step_lengths)
    }

def bootstrap_diffusion_coefficient(trajectory, n_bootstrap=N_BOOTSTRAP, dt=DT):
    """
    Bootstrap estimate of diffusion coefficient uncertainty.
    
    Args:
        trajectory: Single trajectory dictionary
        n_bootstrap: Number of bootstrap samples
        dt: Time step
        
    Returns:
        Dictionary with bootstrap results
    """
    x, y = trajectory['x'], trajectory['y']
    
    # Calculate squared displacements
    dx = np.diff(x)
    dy = np.diff(y)
    squared_displacements = dx**2 + dy**2
    
    bootstrap_D_values = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled_displacements = np.random.choice(squared_displacements, 
                                                  size=len(squared_displacements), 
                                                  replace=True)
        
        # Calculate MSD and diffusion coefficient
        msd = np.mean(resampled_displacements)
        D_bootstrap = msd / (4 * dt)  # 2D diffusion
        bootstrap_D_values.append(D_bootstrap)
    
    return {
        'bootstrap_D_values': bootstrap_D_values,
        'bootstrap_mean': np.mean(bootstrap_D_values),
        'bootstrap_std': np.std(bootstrap_D_values),
        'bootstrap_ci_lower': np.percentile(bootstrap_D_values, 2.5),
        'bootstrap_ci_upper': np.percentile(bootstrap_D_values, 97.5),
        'original_D': trajectory.get('D', np.nan)
    }

def perform_advanced_validation(dataset_dir, dataset_name):
    """
    Perform comprehensive diffusion validation analysis.
    
    Args:
        dataset_dir: Directory containing analyzed trajectory files
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with validation results
    """
    # Load all trajectory files
    file_paths = glob.glob(os.path.join(dataset_dir, "analyzed_*.pkl"))
    
    if not file_paths:
        print(f"No analyzed files found in {dataset_dir}")
        return None
    
    # Collect all trajectories
    all_trajectories = []
    for file_path in file_paths:
        data = load_analyzed_data(file_path)
        if data:
            trajectories = extract_trajectories(data)
            all_trajectories.extend(trajectories)
    
    if not all_trajectories:
        print(f"No valid trajectories found for {dataset_name}")
        return None
    
    print(f"Analyzing {len(all_trajectories)} trajectories for {dataset_name}")
    
    results = {
        'dataset_name': dataset_name,
        'n_trajectories': len(all_trajectories)
    }
    
    # 1. Hurst exponent analysis
    print("  Calculating Hurst exponents...")
    hurst_results = []
    for traj in all_trajectories[:10]:  # Limit for computational efficiency
        hurst_result = calculate_hurst_exponent(traj)
        if hurst_result:
            hurst_results.append(hurst_result['hurst_exponent'])
    
    if hurst_results:
        results['hurst_analysis'] = {
            'mean_hurst': np.mean(hurst_results),
            'std_hurst': np.std(hurst_results),
            'hurst_values': hurst_results
        }
    
    # 2. Ergodicity breaking parameter
    print("  Calculating ergodicity breaking parameter...")
    eb_result = calculate_ergodicity_breaking_parameter(all_trajectories[:20])
    if eb_result:
        results['ergodicity_breaking'] = eb_result
    
    # 3. Non-Gaussian parameter
    print("  Calculating non-Gaussian parameter...")
    ng_result = calculate_non_gaussian_parameter(all_trajectories[:20])
    if ng_result:
        results['non_gaussian'] = ng_result
    
    # 4. Velocity increment analysis
    print("  Analyzing velocity increments...")
    vi_result = analyze_velocity_increments(all_trajectories[:20])
    if vi_result:
        results['velocity_increments'] = vi_result
    
    # 5. Step length distribution
    print("  Analyzing step length distributions...")
    sl_result = analyze_step_length_distribution(all_trajectories[:20])
    if sl_result:
        results['step_lengths'] = sl_result
    
    # 6. Bootstrap analysis on a few trajectories
    print("  Performing bootstrap analysis...")
    bootstrap_results = []
    for traj in all_trajectories[:5]:
        bootstrap_result = bootstrap_diffusion_coefficient(traj)
        bootstrap_results.append(bootstrap_result)
    
    if bootstrap_results:
        results['bootstrap_analysis'] = bootstrap_results
    
    return results

def create_validation_plots(results, output_path):
    """
    Create comprehensive validation plots.
    
    Args:
        results: Validation results dictionary
        output_path: Directory to save plots
    """
    dataset_name = results['dataset_name']
    
    # Create multi-panel figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Plot 1: Hurst exponent distribution
    ax1 = axes[0, 0]
    if 'hurst_analysis' in results:
        hurst_data = results['hurst_analysis']
        ax1.hist(hurst_data['hurst_values'], bins=15, alpha=0.7, edgecolor='black')
        ax1.axvline(0.5, color='red', linestyle='--', label='Normal diffusion (H=0.5)')
        ax1.axvline(hurst_data['mean_hurst'], color='green', linestyle='-', 
                   label=f'Mean H = {hurst_data["mean_hurst"]:.3f}')
        
        # Interpretation
        if hurst_data['mean_hurst'] > 0.6:
            ax1.text(0.05, 0.95, 'Super-diffusion\n(persistent)', transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
        elif hurst_data['mean_hurst'] < 0.4:
            ax1.text(0.05, 0.95, 'Sub-diffusion\n(anti-persistent)', transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.8))
        else:
            ax1.text(0.05, 0.95, 'Normal diffusion', transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))
        
        ax1.set_xlabel('Hurst Exponent H')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Hurst Exponent Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Non-Gaussian parameter vs time
    ax2 = axes[0, 1]
    if 'non_gaussian' in results:
        ng_data = results['non_gaussian']
        ax2.plot(ng_data['time_points'], ng_data['alpha2_values'], 'o-', alpha=0.7)
        ax2.axhline(0, color='red', linestyle='--', label='Gaussian (α₂=0)')
        ax2.axhline(ng_data['mean_alpha2'], color='green', linestyle='-', 
                   label=f'Mean α₂ = {ng_data["mean_alpha2"]:.3f}')
        
        # Interpretation
        if ng_data['mean_alpha2'] > 0.1:
            ax2.text(0.05, 0.95, 'Non-Gaussian\nbehavior', transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
        else:
            ax2.text(0.05, 0.95, 'Gaussian\nbehavior', transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))
        
        ax2.set_xlabel('Time lag (s)')
        ax2.set_ylabel('Non-Gaussian parameter α₂')
        ax2.set_title('Non-Gaussian Parameter vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Ergodicity breaking parameter
    ax3 = axes[0, 2]
    if 'ergodicity_breaking' in results:
        eb_data = results['ergodicity_breaking']
        eb_value = eb_data['eb_parameter']
        
        # Create a simple bar showing EB parameter
        ax3.bar(['EB Parameter'], [eb_value], color='blue', alpha=0.7)
        ax3.axhline(0, color='red', linestyle='--', label='Ergodic (EB=0)')
        
        # Interpretation
        if eb_value > 0.5:
            interpretation = 'Strong ergodicity\nbreaking'
            color = 'red'
        elif eb_value > 0.1:
            interpretation = 'Weak ergodicity\nbreaking'
            color = 'orange'
        else:
            interpretation = 'Ergodic\nbehavior'
            color = 'green'
        
        ax3.text(0.5, 0.95, interpretation, transform=ax3.transAxes, 
                ha='center', verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        
        ax3.set_ylabel('EB Parameter')
        ax3.set_title(f'Ergodicity Breaking\n(EB = {eb_value:.3f})')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Velocity increment distribution
    ax4 = axes[1, 0]
    if 'velocity_increments' in results:
        vi_data = results['velocity_increments']
        
        # Plot histogram and fitted normal distribution
        n, bins, patches = ax4.hist(vi_data['velocity_increments'][:1000], 
                                   bins=50, density=True, alpha=0.7, edgecolor='black')
        
        # Overlay normal distribution
        x = np.linspace(bins[0], bins[-1], 100)
        normal_pdf = stats.norm.pdf(x, vi_data['mean'], vi_data['std'])
        ax4.plot(x, normal_pdf, 'r-', linewidth=2, label='Normal fit')
        
        # Add statistics
        ax4.text(0.05, 0.95, f'Shapiro p = {vi_data["shapiro_p_value"]:.3e}\nKurtosis = {vi_data["kurtosis"]:.2f}', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Color code based on normality
        if vi_data['is_gaussian']:
            ax4.text(0.05, 0.75, 'Normal diffusion', transform=ax4.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))
        else:
            ax4.text(0.05, 0.75, 'Anomalous diffusion', transform=ax4.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
        
        ax4.set_xlabel('Velocity increment magnitude (μm/s²)')
        ax4.set_ylabel('Probability density')
        ax4.set_title('Velocity Increment Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Step length distribution
    ax5 = axes[1, 1]
    if 'step_lengths' in results:
        sl_data = results['step_lengths']
        
        # Plot histogram
        n, bins, patches = ax5.hist(sl_data['step_lengths'][:1000], 
                                   bins=50, density=True, alpha=0.7, edgecolor='black')
        
        # Overlay Rayleigh distribution if available
        if not np.isnan(sl_data['rayleigh_param'][0]):
            x = np.linspace(bins[0], bins[-1], 100)
            rayleigh_pdf = stats.rayleigh.pdf(x, *sl_data['rayleigh_param'])
            ax5.plot(x, rayleigh_pdf, 'r-', linewidth=2, label='Rayleigh fit')
        
        # Add statistics
        ax5.text(0.05, 0.95, f'Rayleigh p = {sl_data["rayleigh_ks_p"]:.3e}', 
                transform=ax5.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Color code based on fit quality
        if sl_data['rayleigh_fits_well']:
            ax5.text(0.05, 0.85, 'Normal diffusion', transform=ax5.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))
        else:
            ax5.text(0.05, 0.85, 'Jump/anomalous\ndiffusion', transform=ax5.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
        
        ax5.set_xlabel('Step length (μm)')
        ax5.set_ylabel('Probability density')
        ax5.set_title('Step Length Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Bootstrap diffusion coefficient uncertainty
    ax6 = axes[1, 2]
    if 'bootstrap_analysis' in results:
        bootstrap_data = results['bootstrap_analysis']
        
        # Collect all bootstrap results
        original_D = [result['original_D'] for result in bootstrap_data]
        bootstrap_std = [result['bootstrap_std'] for result in bootstrap_data]
        relative_uncertainty = [std/orig if orig > 0 else np.nan 
                              for std, orig in zip(bootstrap_std, original_D)]
        
        # Remove NaN values
        relative_uncertainty = [x for x in relative_uncertainty if not np.isnan(x)]
        
        if relative_uncertainty:
            ax6.hist(relative_uncertainty, bins=15, alpha=0.7, edgecolor='black')
            mean_uncertainty = np.mean(relative_uncertainty)
            ax6.axvline(mean_uncertainty, color='red', linestyle='--', 
                       label=f'Mean = {mean_uncertainty:.1%}')
            
            # Color code based on uncertainty level
            if mean_uncertainty > 0.2:
                ax6.text(0.05, 0.95, 'High uncertainty\n(>20%)', transform=ax6.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
            elif mean_uncertainty > 0.1:
                ax6.text(0.05, 0.95, 'Moderate uncertainty\n(10-20%)', transform=ax6.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
            else:
                ax6.text(0.05, 0.95, 'Low uncertainty\n(<10%)', transform=ax6.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))
            
            ax6.set_xlabel('Relative uncertainty (fraction)')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Bootstrap Uncertainty Analysis')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
    
    # Plot 7: Diffusion type classification summary
    ax7 = axes[2, 0]
    classification_results = []
    classification_labels = []
    
    # Collect all validation metrics
    if 'hurst_analysis' in results:
        mean_hurst = results['hurst_analysis']['mean_hurst']
        if mean_hurst > 0.6:
            classification_results.append(1)  # Super-diffusion
            classification_labels.append('Super-diffusion\n(Hurst)')
        elif mean_hurst < 0.4:
            classification_results.append(-1)  # Sub-diffusion
            classification_labels.append('Sub-diffusion\n(Hurst)')
        else:
            classification_results.append(0)  # Normal
            classification_labels.append('Normal\n(Hurst)')
    
    if 'non_gaussian' in results:
        mean_alpha2 = results['non_gaussian']['mean_alpha2']
        if mean_alpha2 > 0.1:
            classification_results.append(1)  # Non-Gaussian
            classification_labels.append('Non-Gaussian\n(α₂)')
        else:
            classification_results.append(0)  # Gaussian
            classification_labels.append('Gaussian\n(α₂)')
    
    if 'ergodicity_breaking' in results:
        eb_param = results['ergodicity_breaking']['eb_parameter']
        if eb_param > 0.1:
            classification_results.append(1)  # Non-ergodic
            classification_labels.append('Non-ergodic\n(EB)')
        else:
            classification_results.append(0)  # Ergodic
            classification_labels.append('Ergodic\n(EB)')
    
    if 'velocity_increments' in results:
        if not results['velocity_increments']['is_gaussian']:
            classification_results.append(1)  # Anomalous
            classification_labels.append('Anomalous\n(Velocity)')
        else:
            classification_results.append(0)  # Normal
            classification_labels.append('Normal\n(Velocity)')
    
    if 'step_lengths' in results:
        if not results['step_lengths']['rayleigh_fits_well']:
            classification_results.append(1)  # Jump diffusion
            classification_labels.append('Jump diffusion\n(Steps)')
        else:
            classification_results.append(0)  # Normal
            classification_labels.append('Normal\n(Steps)')
    
    if classification_results:
        colors = ['red' if x > 0 else 'green' if x == 0 else 'blue' for x in classification_results]
        bars = ax7.bar(range(len(classification_results)), classification_results, 
                      color=colors, alpha=0.7)
        
        ax7.set_xticks(range(len(classification_labels)))
        ax7.set_xticklabels(classification_labels, rotation=45, ha='right')
        ax7.set_ylabel('Classification score')
        ax7.set_title('Diffusion Type Classification Summary')
        ax7.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax7.grid(True, alpha=0.3)
        
        # Add overall assessment
        anomalous_score = np.sum(np.array(classification_results) > 0)
        total_tests = len(classification_results)
        
        if anomalous_score > total_tests * 0.6:
            overall_text = 'Likely ANOMALOUS\ndiffusion'
            overall_color = 'red'
        elif anomalous_score > total_tests * 0.3:
            overall_text = 'MIXED behavior'
            overall_color = 'orange'
        else:
            overall_text = 'Likely NORMAL\ndiffusion'
            overall_color = 'green'
        
        ax7.text(0.05, 0.95, overall_text, transform=ax7.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=overall_color, alpha=0.8))
    
    # Plot 8: Diffusion coefficient reliability assessment
    ax8 = axes[2, 1]
    reliability_scores = []
    reliability_labels = []
    
    # Assess reliability based on various factors
    if 'bootstrap_analysis' in results:
        bootstrap_data = results['bootstrap_analysis']
        uncertainties = [result['bootstrap_std']/result['original_D'] 
                        for result in bootstrap_data if result['original_D'] > 0]
        if uncertainties:
            mean_uncertainty = np.mean(uncertainties)
            if mean_uncertainty < 0.1:
                reliability_scores.append(3)  # High reliability
            elif mean_uncertainty < 0.2:
                reliability_scores.append(2)  # Medium reliability
            else:
                reliability_scores.append(1)  # Low reliability
            reliability_labels.append(f'Bootstrap\nUnc: {mean_uncertainty:.1%}')
    
    # Sampling adequacy (based on trajectory length)
    mean_track_length = np.mean([len(traj['x']) for traj in all_trajectories[:10]]) if 'all_trajectories' in locals() else 50
    if mean_track_length > 100:
        reliability_scores.append(3)
        reliability_labels.append(f'Track length\n(Good: {mean_track_length:.0f})')
    elif mean_track_length > 50:
        reliability_scores.append(2)
        reliability_labels.append(f'Track length\n(OK: {mean_track_length:.0f})')
    else:
        reliability_scores.append(1)
        reliability_labels.append(f'Track length\n(Poor: {mean_track_length:.0f})')
    
    # Statistical consistency
    if 'hurst_analysis' in results:
        hurst_std = results['hurst_analysis']['std_hurst']
        if hurst_std < 0.1:
            reliability_scores.append(3)
            reliability_labels.append(f'Hurst consistency\n(Good: σ={hurst_std:.2f})')
        elif hurst_std < 0.2:
            reliability_scores.append(2)
            reliability_labels.append(f'Hurst consistency\n(OK: σ={hurst_std:.2f})')
        else:
            reliability_scores.append(1)
            reliability_labels.append(f'Hurst consistency\n(Poor: σ={hurst_std:.2f})')
    
    if reliability_scores:
        colors = ['green' if x == 3 else 'orange' if x == 2 else 'red' for x in reliability_scores]
        bars = ax8.bar(range(len(reliability_scores)), reliability_scores, 
                      color=colors, alpha=0.7)
        
        ax8.set_xticks(range(len(reliability_labels)))
        ax8.set_xticklabels(reliability_labels, rotation=45, ha='right')
        ax8.set_ylabel('Reliability score (1=Poor, 3=Good)')
        ax8.set_title('Diffusion Coefficient Reliability')
        ax8.set_ylim([0, 3.5])
        ax8.grid(True, alpha=0.3)
        
        # Overall reliability assessment
        mean_reliability = np.mean(reliability_scores)
        if mean_reliability >= 2.5:
            reliability_text = 'HIGH reliability'
            reliability_color = 'green'
        elif mean_reliability >= 2.0:
            reliability_text = 'MEDIUM reliability'
            reliability_color = 'orange'
        else:
            reliability_text = 'LOW reliability'
            reliability_color = 'red'
        
        ax8.text(0.05, 0.95, reliability_text, transform=ax8.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=reliability_color, alpha=0.8))
    
    # Plot 9: Recommendations and warnings
    ax9 = axes[2, 2]
    ax9.axis('off')  # Turn off axes for text plot
    
    # Generate recommendations based on analysis
    recommendations = []
    
    if 'hurst_analysis' in results:
        mean_hurst = results['hurst_analysis']['mean_hurst']
        if abs(mean_hurst - 0.5) > 0.2:
            recommendations.append(f"⚠️ Non-normal diffusion detected (H={mean_hurst:.2f})")
    
    if 'non_gaussian' in results:
        if results['non_gaussian']['mean_alpha2'] > 0.1:
            recommendations.append("⚠️ Non-Gaussian behavior detected")
    
    if 'ergodicity_breaking' in results:
        if results['ergodicity_breaking']['eb_parameter'] > 0.1:
            recommendations.append("⚠️ Non-ergodic behavior detected")
    
    if 'velocity_increments' in results:
        if not results['velocity_increments']['is_gaussian']:
            recommendations.append("⚠️ Anomalous velocity increments")
    
    if 'bootstrap_analysis' in results:
        bootstrap_data = results['bootstrap_analysis']
        uncertainties = [result['bootstrap_std']/result['original_D'] 
                        for result in bootstrap_data if result['original_D'] > 0]
        if uncertainties and np.mean(uncertainties) > 0.2:
            recommendations.append("⚠️ High measurement uncertainty")
    
    # Add specific recommendations
    if not recommendations:
        recommendations.append("✓ Normal diffusion behavior")
        recommendations.append("✓ Standard analysis appropriate")
    else:
        recommendations.append("")
        recommendations.append("Recommendations:")
        recommendations.append("• Use longer trajectories")
        recommendations.append("• Consider anomalous diffusion models")
        recommendations.append("• Check experimental conditions")
        recommendations.append("• Verify tracking accuracy")
    
    # Display recommendations
    recommendation_text = "\n".join(recommendations)
    ax9.text(0.05, 0.95, recommendation_text, transform=ax9.transAxes, 
            verticalalignment='top', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax9.set_title('Analysis Summary & Recommendations')
    
    plt.suptitle(f'Advanced Diffusion Validation: {dataset_name}', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{dataset_name}_advanced_validation.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def export_validation_results(results, output_path):
    """Export detailed validation results to CSV files."""
    dataset_name = results['dataset_name']
    
    # Summary statistics
    summary_data = {
        'Parameter': ['Dataset', 'N_trajectories'],
        'Value': [dataset_name, results['n_trajectories']]
    }
    
    # Add Hurst exponent results
    if 'hurst_analysis' in results:
        hurst = results['hurst_analysis']
        summary_data['Parameter'].extend(['Mean_Hurst_exponent', 'Std_Hurst_exponent'])
        summary_data['Value'].extend([hurst['mean_hurst'], hurst['std_hurst']])
    
    # Add non-Gaussian parameter
    if 'non_gaussian' in results:
        ng = results['non_gaussian']
        summary_data['Parameter'].extend(['Mean_non_Gaussian_parameter'])
        summary_data['Value'].extend([ng['mean_alpha2']])
    
    # Add ergodicity breaking
    if 'ergodicity_breaking' in results:
        eb = results['ergodicity_breaking']
        summary_data['Parameter'].extend(['Ergodicity_breaking_parameter'])
        summary_data['Value'].extend([eb['eb_parameter']])
    
    # Add velocity increment analysis
    if 'velocity_increments' in results:
        vi = results['velocity_increments']
        summary_data['Parameter'].extend(['Velocity_increments_Gaussian', 'Velocity_kurtosis', 'Velocity_skewness'])
        summary_data['Value'].extend([vi['is_gaussian'], vi['kurtosis'], vi['skewness']])
    
    # Add step length analysis
    if 'step_lengths' in results:
        sl = results['step_lengths']
        summary_data['Parameter'].extend(['Step_lengths_Rayleigh_fit', 'Mean_step_length'])
        summary_data['Value'].extend([sl['rayleigh_fits_well'], sl['mean_step_length']])
    
    # Add bootstrap analysis
    if 'bootstrap_analysis' in results:
        bootstrap_data = results['bootstrap_analysis']
        original_D = [result['original_D'] for result in bootstrap_data]
        bootstrap_std = [result['bootstrap_std'] for result in bootstrap_data]
        relative_uncertainty = [std/orig if orig > 0 else np.nan 
                              for std, orig in zip(bootstrap_std, original_D)]
        relative_uncertainty = [x for x in relative_uncertainty if not np.isnan(x)]
        
        if relative_uncertainty:
            summary_data['Parameter'].extend(['Mean_bootstrap_uncertainty'])
            summary_data['Value'].extend([np.mean(relative_uncertainty)])
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_path, f'{dataset_name}_validation_summary.csv'), index=False)
    
    # Save detailed results for each analysis
    if 'hurst_analysis' in results:
        hurst_df = pd.DataFrame({'Hurst_exponent': results['hurst_analysis']['hurst_values']})
        hurst_df.to_csv(os.path.join(output_path, f'{dataset_name}_hurst_exponents.csv'), index=False)
    
    if 'non_gaussian' in results:
        ng_data = results['non_gaussian']
        ng_df = pd.DataFrame({
            'Time_lag_s': ng_data['time_points'],
            'Alpha2_parameter': ng_data['alpha2_values']
        })
        ng_df.to_csv(os.path.join(output_path, f'{dataset_name}_non_gaussian.csv'), index=False)
    
    if 'bootstrap_analysis' in results:
        bootstrap_data = results['bootstrap_analysis']
        bootstrap_df = pd.DataFrame([
            {
                'Original_D': result['original_D'],
                'Bootstrap_mean_D': result['bootstrap_mean'],
                'Bootstrap_std_D': result['bootstrap_std'],
                'Bootstrap_CI_lower': result['bootstrap_ci_lower'],
                'Bootstrap_CI_upper': result['bootstrap_ci_upper']
            }
            for result in bootstrap_data
        ])
        bootstrap_df.to_csv(os.path.join(output_path, f'{dataset_name}_bootstrap_analysis.csv'), index=False)

def main():
    """Main function for advanced diffusion validation."""
    print("Advanced Diffusion Validation Analysis")
    print("=" * 50)
    
    # Get dataset directory
    dataset_dir = input("Enter directory containing analyzed trajectory files: ")
    
    if not os.path.isdir(dataset_dir):
        print(f"Directory {dataset_dir} does not exist")
        return
    
    # Get dataset name
    dataset_name = input("Enter dataset name (press Enter to use directory name): ")
    if not dataset_name:
        dataset_name = os.path.basename(os.path.normpath(dataset_dir))
        if not dataset_name:
            dataset_name = "Dataset"
    
    # Create output directory
    output_dir = os.path.join(dataset_dir, f"advanced_validation_{dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform analysis
    print(f"\nPerforming advanced validation on '{dataset_name}'...")
    results = perform_advanced_validation(dataset_dir, dataset_name)
    
    if results is None:
        print("Analysis failed")
        return
    
    # Create plots
    print("Creating validation plots...")
    create_validation_plots(results, output_dir)
    
    # Export results
    print("Exporting results...")
    export_validation_results(results, output_dir)
    
    # Save full results
    with open(os.path.join(output_dir, f'{dataset_name}_full_validation_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Print summary
    print(f"\nAdvanced Validation Summary for {dataset_name}:")
    print(f"Results saved to: {output_dir}")
    print(f"Trajectories analyzed: {results['n_trajectories']}")
    
    # Print key findings
    if 'hurst_analysis' in results:
        mean_hurst = results['hurst_analysis']['mean_hurst']
        print(f"\nHurst Exponent Analysis:")
        print(f"  Mean H = {mean_hurst:.3f}")
        if mean_hurst > 0.6:
            print("  → Super-diffusion (persistent motion)")
        elif mean_hurst < 0.4:
            print("  → Sub-diffusion (anti-persistent motion)")
        else:
            print("  → Normal diffusion")
    
    if 'non_gaussian' in results:
        mean_alpha2 = results['non_gaussian']['mean_alpha2']
        print(f"\nNon-Gaussian Parameter:")
        print(f"  Mean α₂ = {mean_alpha2:.3f}")
        if mean_alpha2 > 0.1:
            print("  → Non-Gaussian behavior detected")
        else:
            print("  → Gaussian behavior (normal)")
    
    if 'ergodicity_breaking' in results:
        eb_param = results['ergodicity_breaking']['eb_parameter']
        print(f"\nErgodicity Analysis:")
        print(f"  EB parameter = {eb_param:.3f}")
        if eb_param > 0.1:
            print("  → Non-ergodic behavior (ensemble ≠ time average)")
        else:
            print("  → Ergodic behavior (normal)")
    
    if 'bootstrap_analysis' in results:
        bootstrap_data = results['bootstrap_analysis']
        uncertainties = [result['bootstrap_std']/result['original_D'] 
                        for result in bootstrap_data if result['original_D'] > 0]
        if uncertainties:
            mean_uncertainty = np.mean(uncertainties)
            print(f"\nMeasurement Uncertainty:")
            print(f"  Mean relative uncertainty = {mean_uncertainty:.1%}")
            if mean_uncertainty > 0.2:
                print("  → High uncertainty - be cautious with conclusions")
            elif mean_uncertainty > 0.1:
                print("  → Moderate uncertainty - acceptable for most purposes")
            else:
                print("  → Low uncertainty - reliable measurements")
    
    print(f"\n{'='*50}")
    print("IMPORTANT INTERPRETATION GUIDELINES:")
    print("• Normal diffusion should show: H≈0.5, α₂≈0, EB≈0, Gaussian increments")
    print("• Anomalous diffusion indicators: H≠0.5, α₂>0, EB>0, non-Gaussian behavior")
    print("• High uncertainty (>20%) suggests longer trajectories needed")
    print("• Always validate assumptions before interpreting diffusion coefficients!")

if __name__ == "__main__":
    main()