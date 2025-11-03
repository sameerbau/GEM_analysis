# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 15:39:28 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alternative_correlation_metrics.py

Alternative metrics for velocity correlation analysis when traditional autocorrelation 
is limited by sampling frequency. Implements multiple approaches that work better 
near the sampling limit.

Input:
- Analyzed trajectory .pkl files

Output:
- Velocity direction persistence analysis
- Step-size correlation analysis  
- Turn angle analysis
- Velocity increment statistics
- Directional change metrics


. Directional Persistence

What it measures: How long velocity directions remain correlated
Why it's better: Less sensitive to sampling frequency than velocity magnitude
Interpretation: Persistence time > 2×DT is well-resolved

2. Step-Size Correlations

What it measures: Correlations in speed (not velocity vectors)
Why it's better: Removes directional complications, focuses on magnitude changes
Good for: Detecting if particles speed up/slow down in correlated ways

3. Turn Angle Analysis

What it measures: How much particles change direction between steps
Key metric: Mean Resultant Length (0 = random, 1 = straight line)
Why it's robust: Turn angles are discrete, less affected by sampling

4. Velocity Increment Correlations

What it measures: Correlations in changes in velocity (acceleration-like)
Why it's useful: Can detect short-term correlations missed by standard autocorr

What to look for:

Directional persistence > 0.2s: Real correlations despite sampling limits
Mean Resultant Length > 0.1: Indicates directional bias (non-random motion)
Step-size correlations: If significant, suggests real velocity correlations
Turn angle patterns: Can reveal underlying motion modes


Usage:
python alternative_correlation_metrics.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
from scipy import stats
from scipy.optimize import curve_fit
import warnings

# Global parameters (modify these as needed)
# =====================================
# Time step in seconds
DT = 0.1
# Minimum trajectory length
MIN_TRACK_LENGTH = 30
# Maximum lag for direction analysis
MAX_LAG_DIRECTION = 20
# Angle binning for circular statistics
ANGLE_BINS = 36  # 10-degree bins
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

def calculate_velocity_directions(trajectories, dt=DT, min_length=MIN_TRACK_LENGTH):
    """
    Calculate velocity directions and related metrics.
    
    Args:
        trajectories: List of trajectory dictionaries
        dt: Time step
        min_length: Minimum trajectory length
        
    Returns:
        Dictionary with direction analysis data
    """
    filtered_trajectories = [traj for traj in trajectories if len(traj['x']) > min_length]
    
    if not filtered_trajectories:
        return None
    
    direction_data = []
    
    for traj in filtered_trajectories:
        x = np.array(traj['x'])
        y = np.array(traj['y'])
        
        # Calculate velocity components
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        
        # Calculate velocity directions (angles)
        directions = np.arctan2(vy, vx)  # Returns angles in [-π, π]
        
        # Calculate speeds
        speeds = np.sqrt(vx**2 + vy**2)
        
        # Calculate turn angles (change in direction between consecutive steps)
        turn_angles = np.diff(directions)
        
        # Handle angle wrapping for turn angles
        turn_angles = np.mod(turn_angles + np.pi, 2*np.pi) - np.pi
        
        direction_data.append({
            'traj_id': traj['id'],
            'directions': directions,
            'speeds': speeds,
            'turn_angles': turn_angles,
            'vx': vx,
            'vy': vy
        })
    
    return {
        'direction_data': direction_data,
        'n_trajectories': len(filtered_trajectories)
    }

def calculate_direction_persistence(direction_data, max_lag=MAX_LAG_DIRECTION):
    """
    Calculate directional persistence - how long directions remain correlated.
    
    This is less sensitive to sampling frequency than velocity autocorrelation.
    
    Args:
        direction_data: Output from calculate_velocity_directions
        max_lag: Maximum lag to analyze
        
    Returns:
        Dictionary with persistence results
    """
    if not direction_data or not direction_data['direction_data']:
        return None
    
    persistence = np.zeros(max_lag)
    persistence_sem = np.zeros(max_lag)
    n_points = np.zeros(max_lag)
    
    for lag in range(max_lag):
        dot_products = []
        
        for traj_data in direction_data['direction_data']:
            directions = traj_data['directions']
            
            if len(directions) > lag:
                # Calculate unit vectors at time t and t+lag
                angles_0 = directions[:-lag] if lag > 0 else directions
                angles_lag = directions[lag:] if lag > 0 else directions
                
                # Convert to unit vectors
                vx_0 = np.cos(angles_0)
                vy_0 = np.sin(angles_0)
                vx_lag = np.cos(angles_lag)
                vy_lag = np.sin(angles_lag)
                
                # Calculate dot product (directional correlation)
                dot_product = vx_0 * vx_lag + vy_0 * vy_lag
                dot_products.extend(dot_product)
        
        if dot_products:
            persistence[lag] = np.mean(dot_products)
            persistence_sem[lag] = np.std(dot_products) / np.sqrt(len(dot_products))
            n_points[lag] = len(dot_products)
        else:
            persistence[lag] = np.nan
            persistence_sem[lag] = np.nan
            n_points[lag] = 0
    
    # Calculate persistence time (time to decay to 1/e)
    persistence_time = np.nan
    if persistence[0] > 0:
        threshold = persistence[0] / np.e
        indices = np.where(persistence < threshold)[0]
        if len(indices) > 0:
            persistence_time = indices[0] * DT
    
    return {
        'persistence': persistence,
        'persistence_sem': persistence_sem,
        'n_points': n_points,
        'time_lags': np.arange(max_lag) * DT,
        'persistence_time': persistence_time
    }

def calculate_step_correlations(direction_data, max_lag=MAX_LAG_DIRECTION):
    """
    Calculate correlations in step sizes (speeds) between time points.
    
    This can reveal correlations not visible in velocity autocorrelation.
    
    Args:
        direction_data: Output from calculate_velocity_directions
        max_lag: Maximum lag to analyze
        
    Returns:
        Dictionary with step correlation results
    """
    if not direction_data or not direction_data['direction_data']:
        return None
    
    # Collect all speeds
    all_speeds = []
    speed_series = []
    
    for traj_data in direction_data['direction_data']:
        speeds = traj_data['speeds']
        all_speeds.extend(speeds)
        speed_series.append(speeds)
    
    all_speeds = np.array(all_speeds)
    mean_speed = np.mean(all_speeds)
    std_speed = np.std(all_speeds)
    
    if std_speed == 0:
        return None
    
    # Calculate step-size autocorrelation
    step_autocorr = np.zeros(max_lag)
    step_sem = np.zeros(max_lag)
    
    for lag in range(max_lag):
        correlations = []
        
        for speeds in speed_series:
            if len(speeds) > lag:
                speeds_0 = speeds[:-lag] if lag > 0 else speeds
                speeds_lag = speeds[lag:] if lag > 0 else speeds
                
                # Normalize speeds
                norm_speeds_0 = (speeds_0 - mean_speed) / std_speed
                norm_speeds_lag = (speeds_lag - mean_speed) / std_speed
                
                # Calculate correlation
                correlation = np.mean(norm_speeds_0 * norm_speeds_lag)
                correlations.append(correlation)
        
        if correlations:
            step_autocorr[lag] = np.mean(correlations)
            step_sem[lag] = np.std(correlations) / np.sqrt(len(correlations))
        else:
            step_autocorr[lag] = np.nan
            step_sem[lag] = np.nan
    
    return {
        'step_autocorr': step_autocorr,
        'step_sem': step_sem,
        'time_lags': np.arange(max_lag) * DT,
        'mean_speed': mean_speed,
        'std_speed': std_speed
    }

def calculate_turn_angle_statistics(direction_data):
    """
    Analyze turn angle distributions and correlations.
    
    Args:
        direction_data: Output from calculate_velocity_directions
        
    Returns:
        Dictionary with turn angle analysis
    """
    if not direction_data or not direction_data['direction_data']:
        return None
    
    # Collect all turn angles
    all_turn_angles = []
    turn_series = []
    
    for traj_data in direction_data['direction_data']:
        turn_angles = traj_data['turn_angles']
        all_turn_angles.extend(turn_angles)
        turn_series.append(turn_angles)
    
    all_turn_angles = np.array(all_turn_angles)
    
    # Calculate turn angle statistics
    mean_turn = np.mean(all_turn_angles)
    std_turn = np.std(all_turn_angles)
    
    # Calculate circular statistics
    # Mean resultant length (measure of directional clustering)
    cos_sum = np.sum(np.cos(all_turn_angles))
    sin_sum = np.sum(np.sin(all_turn_angles))
    n = len(all_turn_angles)
    
    mean_resultant_length = np.sqrt(cos_sum**2 + sin_sum**2) / n
    circular_mean = np.arctan2(sin_sum, cos_sum)
    
    # Calculate turn angle autocorrelation (tendency to continue turning)
    max_lag_turn = min(10, max(len(series) - 1 for series in turn_series))
    turn_autocorr = []
    
    for lag in range(max_lag_turn):
        correlations = []
        
        for turns in turn_series:
            if len(turns) > lag:
                turns_0 = turns[:-lag] if lag > 0 else turns
                turns_lag = turns[lag:] if lag > 0 else turns
                
                # Calculate circular correlation using cosines
                corr = np.mean(np.cos(turns_0 - turns_lag))
                correlations.append(corr)
        
        if correlations:
            turn_autocorr.append(np.mean(correlations))
        else:
            turn_autocorr.append(np.nan)
    
    return {
        'all_turn_angles': all_turn_angles,
        'mean_turn': mean_turn,
        'std_turn': std_turn,
        'mean_resultant_length': mean_resultant_length,
        'circular_mean': circular_mean,
        'turn_autocorr': np.array(turn_autocorr),
        'turn_autocorr_lags': np.arange(max_lag_turn) * DT
    }

def calculate_velocity_increments(direction_data, max_lag=5):
    """
    Analyze velocity increments - changes in velocity between time steps.
    
    This can reveal short-term correlations not visible in standard autocorrelation.
    
    Args:
        direction_data: Output from calculate_velocity_directions
        max_lag: Maximum lag to analyze
        
    Returns:
        Dictionary with velocity increment analysis
    """
    if not direction_data or not direction_data['direction_data']:
        return None
    
    increment_data = []
    
    for traj_data in direction_data['direction_data']:
        vx = traj_data['vx']
        vy = traj_data['vy']
        
        # Calculate velocity increments
        dvx = np.diff(vx)
        dvy = np.diff(vy)
        
        # Calculate increment magnitudes
        dv_mag = np.sqrt(dvx**2 + dvy**2)
        
        increment_data.append({
            'dvx': dvx,
            'dvy': dvy,
            'dv_mag': dv_mag
        })
    
    # Analyze increment correlations
    all_dv_mag = []
    dv_series = []
    
    for data in increment_data:
        all_dv_mag.extend(data['dv_mag'])
        dv_series.append(data['dv_mag'])
    
    all_dv_mag = np.array(all_dv_mag)
    mean_dv = np.mean(all_dv_mag)
    std_dv = np.std(all_dv_mag)
    
    # Calculate increment autocorrelation
    increment_autocorr = np.zeros(max_lag)
    
    for lag in range(max_lag):
        correlations = []
        
        for dv_mag in dv_series:
            if len(dv_mag) > lag:
                dv_0 = dv_mag[:-lag] if lag > 0 else dv_mag
                dv_lag = dv_mag[lag:] if lag > 0 else dv_mag
                
                if std_dv > 0:
                    norm_dv_0 = (dv_0 - mean_dv) / std_dv
                    norm_dv_lag = (dv_lag - mean_dv) / std_dv
                    
                    correlation = np.mean(norm_dv_0 * norm_dv_lag)
                    correlations.append(correlation)
        
        if correlations:
            increment_autocorr[lag] = np.mean(correlations)
        else:
            increment_autocorr[lag] = np.nan
    
    return {
        'increment_autocorr': increment_autocorr,
        'time_lags': np.arange(max_lag) * DT,
        'mean_increment': mean_dv,
        'std_increment': std_dv,
        'all_increments': all_dv_mag
    }

def analyze_dataset_alternative_metrics(dataset_dir, dataset_name):
    """
    Perform alternative correlation analysis on a dataset.
    
    Args:
        dataset_dir: Directory containing trajectory files
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with alternative analysis results
    """
    # Load all trajectory files
    file_paths = glob.glob(os.path.join(dataset_dir, "analyzed_*.pkl"))
    
    if not file_paths:
        print(f"No files found in {dataset_dir}")
        return None
    
    # Pool all trajectories
    all_trajectories = []
    for file_path in file_paths:
        data = load_analyzed_data(file_path)
        if data and 'trajectories' in data:
            all_trajectories.extend(data['trajectories'])
    
    if not all_trajectories:
        print(f"No trajectories found in {dataset_name}")
        return None
    
    print(f"Analyzing {len(all_trajectories)} trajectories for {dataset_name}")
    
    # Calculate velocity directions and related data
    direction_data = calculate_velocity_directions(all_trajectories)
    
    if direction_data is None:
        print(f"Failed to extract direction data for {dataset_name}")
        return None
    
    results = {
        'dataset_name': dataset_name,
        'n_trajectories': direction_data['n_trajectories']
    }
    
    # Perform various alternative analyses
    print("  Calculating directional persistence...")
    persistence_result = calculate_direction_persistence(direction_data)
    results['directional_persistence'] = persistence_result
    
    print("  Calculating step-size correlations...")
    step_result = calculate_step_correlations(direction_data)
    results['step_correlations'] = step_result
    
    print("  Analyzing turn angles...")
    turn_result = calculate_turn_angle_statistics(direction_data)
    results['turn_analysis'] = turn_result
    
    print("  Calculating velocity increments...")
    increment_result = calculate_velocity_increments(direction_data)
    results['velocity_increments'] = increment_result
    
    return results

def create_alternative_metrics_plots(results, output_path):
    """
    Create plots for alternative correlation metrics.
    
    Args:
        results: Analysis results dictionary
        output_path: Directory to save plots
    """
    dataset_name = results['dataset_name']
    
    # Create a comprehensive figure
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Plot 1: Directional persistence
    ax1 = axes[0, 0]
    if 'directional_persistence' in results and results['directional_persistence']:
        pers = results['directional_persistence']
        ax1.errorbar(pers['time_lags'], pers['persistence'], 
                    yerr=pers['persistence_sem'], fmt='o-', capsize=3)
        
        # Add reference lines
        if not np.isnan(pers['persistence_time']):
            ax1.axvline(pers['persistence_time'], color='red', linestyle='--',
                       label=f'Persistence time = {pers["persistence_time"]:.3f} s')
        
        ax1.axhline(1/np.e, color='orange', linestyle='--', alpha=0.7, label='1/e threshold')
        ax1.axvline(DT, color='gray', linestyle=':', alpha=0.7, label=f'Sampling = {DT} s')
        
        ax1.set_xlabel('Time lag (s)')
        ax1.set_ylabel('Directional persistence')
        ax1.set_title('Directional Persistence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Step-size autocorrelation
    ax2 = axes[0, 1]
    if 'step_correlations' in results and results['step_correlations']:
        step = results['step_correlations']
        ax2.plot(step['time_lags'], step['step_autocorr'], 'o-')
        
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.axvline(DT, color='gray', linestyle=':', alpha=0.7, label=f'Sampling = {DT} s')
        
        ax2.set_xlabel('Time lag (s)')
        ax2.set_ylabel('Step-size autocorrelation')
        ax2.set_title('Step-Size Correlations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Turn angle distribution
    ax3 = axes[1, 0]
    if 'turn_analysis' in results and results['turn_analysis']:
        turn = results['turn_analysis']
        
        # Circular histogram of turn angles
        angles_deg = np.degrees(turn['all_turn_angles'])
        ax3.hist(angles_deg, bins=36, density=True, alpha=0.7, range=(-180, 180))
        
        ax3.axvline(np.degrees(turn['circular_mean']), color='red', linestyle='--',
                   label=f'Circular mean = {np.degrees(turn["circular_mean"]):.1f}°')
        ax3.axvline(0, color='black', linestyle='-', alpha=0.3)
        
        ax3.set_xlabel('Turn angle (degrees)')
        ax3.set_ylabel('Probability density')
        ax3.set_title(f'Turn Angle Distribution\nMean Resultant Length = {turn["mean_resultant_length"]:.3f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Turn angle autocorrelation
    ax4 = axes[1, 1]
    if 'turn_analysis' in results and results['turn_analysis']:
        turn = results['turn_analysis']
        
        ax4.plot(turn['turn_autocorr_lags'], turn['turn_autocorr'], 'o-')
        ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax4.axvline(DT, color='gray', linestyle=':', alpha=0.7, label=f'Sampling = {DT} s')
        
        ax4.set_xlabel('Time lag (s)')
        ax4.set_ylabel('Turn angle autocorrelation')
        ax4.set_title('Turn Angle Correlations')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Velocity increment distribution
    ax5 = axes[2, 0]
    if 'velocity_increments' in results and results['velocity_increments']:
        inc = results['velocity_increments']
        
        ax5.hist(inc['all_increments'], bins=50, density=True, alpha=0.7)
        ax5.axvline(inc['mean_increment'], color='red', linestyle='--',
                   label=f'Mean = {inc["mean_increment"]:.3f} μm/s')
        
        ax5.set_xlabel('Velocity increment magnitude (μm/s)')
        ax5.set_ylabel('Probability density')
        ax5.set_title('Velocity Increment Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Velocity increment autocorrelation
    ax6 = axes[2, 1]
    if 'velocity_increments' in results and results['velocity_increments']:
        inc = results['velocity_increments']
        
        ax6.plot(inc['time_lags'], inc['increment_autocorr'], 'o-')
        ax6.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax6.axvline(DT, color='gray', linestyle=':', alpha=0.7, label=f'Sampling = {DT} s')
        
        ax6.set_xlabel('Time lag (s)')
        ax6.set_ylabel('Increment autocorrelation')
        ax6.set_title('Velocity Increment Correlations')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Alternative Correlation Metrics: {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{dataset_name}_alternative_metrics.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def export_alternative_results(results, output_path):
    """Export alternative metrics results to CSV files."""
    dataset_name = results['dataset_name']
    
    # Directional persistence
    if 'directional_persistence' in results and results['directional_persistence']:
        pers = results['directional_persistence']
        pers_df = pd.DataFrame({
            'Time_lag_s': pers['time_lags'],
            'Directional_persistence': pers['persistence'],
            'Persistence_SEM': pers['persistence_sem'],
            'N_points': pers['n_points']
        })
        pers_df.to_csv(os.path.join(output_path, f'{dataset_name}_directional_persistence.csv'), index=False)
    
    # Step correlations
    if 'step_correlations' in results and results['step_correlations']:
        step = results['step_correlations']
        step_df = pd.DataFrame({
            'Time_lag_s': step['time_lags'],
            'Step_autocorrelation': step['step_autocorr'],
            'Step_SEM': step['step_sem']
        })
        step_df.to_csv(os.path.join(output_path, f'{dataset_name}_step_correlations.csv'), index=False)
    
    # Turn analysis
    if 'turn_analysis' in results and results['turn_analysis']:
        turn = results['turn_analysis']
        turn_df = pd.DataFrame({
            'Time_lag_s': turn['turn_autocorr_lags'],
            'Turn_autocorrelation': turn['turn_autocorr']
        })
        turn_df.to_csv(os.path.join(output_path, f'{dataset_name}_turn_analysis.csv'), index=False)
    
    # Velocity increments
    if 'velocity_increments' in results and results['velocity_increments']:
        inc = results['velocity_increments']
        inc_df = pd.DataFrame({
            'Time_lag_s': inc['time_lags'],
            'Increment_autocorrelation': inc['increment_autocorr']
        })
        inc_df.to_csv(os.path.join(output_path, f'{dataset_name}_velocity_increments.csv'), index=False)
    
    # Summary metrics
    summary_data = {
        'Metric': ['Dataset', 'N_trajectories', 'Sampling_period_s'],
        'Value': [dataset_name, results['n_trajectories'], DT]
    }
    
    if 'directional_persistence' in results and results['directional_persistence']:
        pers = results['directional_persistence']
        summary_data['Metric'].append('Directional_persistence_time_s')
        summary_data['Value'].append(pers['persistence_time'])
    
    if 'turn_analysis' in results and results['turn_analysis']:
        turn = results['turn_analysis']
        summary_data['Metric'].extend(['Mean_resultant_length', 'Turn_angle_std_deg'])
        summary_data['Value'].extend([turn['mean_resultant_length'], np.degrees(turn['std_turn'])])
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_path, f'{dataset_name}_alternative_summary.csv'), index=False)

def main():
    """Main function for alternative correlation metrics analysis."""
    print("Alternative Correlation Metrics Analysis")
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
    output_dir = os.path.join(dataset_dir, f"alternative_metrics_{dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform analysis
    print(f"\nPerforming alternative metrics analysis on '{dataset_name}'...")
    results = analyze_dataset_alternative_metrics(dataset_dir, dataset_name)
    
    if results is None:
        print("Analysis failed")
        return
    
    # Create plots
    print("Creating alternative metrics plots...")
    create_alternative_metrics_plots(results, output_dir)
    
    # Export results
    print("Exporting results...")
    export_alternative_results(results, output_dir)
    
    # Save full results
    with open(os.path.join(output_dir, f'{dataset_name}_alternative_metrics_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Print summary
    print(f"\nAlternative Metrics Summary for {dataset_name}:")
    print(f"Results saved to: {output_dir}")
    print(f"Trajectories analyzed: {results['n_trajectories']}")
    
    print(f"\nKey Alternative Metrics:")
    
    if 'directional_persistence' in results and results['directional_persistence']:
        pers = results['directional_persistence']
        if not np.isnan(pers['persistence_time']):
            print(f"  Directional persistence time: {pers['persistence_time']:.3f} s")
            print(f"    → {pers['persistence_time']/DT:.1f}× sampling period")
            
            if pers['persistence_time'] > 2*DT:
                print("    ✓ Well-resolved (>2× sampling)")
            else:
                print("    ⚠️ Near sampling limit")
        else:
            print("  Directional persistence time: No clear decay found")
    
    if 'turn_analysis' in results and results['turn_analysis']:
        turn = results['turn_analysis']
        print(f"  Mean resultant length: {turn['mean_resultant_length']:.3f}")
        
        if turn['mean_resultant_length'] > 0.3:
            print("    → Strong directional bias")
        elif turn['mean_resultant_length'] > 0.1:
            print("    → Moderate directional bias")
        else:
            print("    → Random turning (no bias)")
    
    if 'step_correlations' in results and results['step_correlations']:
        step = results['step_correlations']
        first_step_corr = step['step_autocorr'][1] if len(step['step_autocorr']) > 1 else np.nan
        
        if not np.isnan(first_step_corr):
            print(f"  Step-size correlation (1 lag): {first_step_corr:.3f}")
            
            if abs(first_step_corr) > 0.1:
                print("    → Significant step-size correlations")
            else:
                print("    → Weak step-size correlations")
    
    print(f"\n💡 These metrics are more robust to sampling limitations than velocity autocorrelation!")
    print(f"   They can detect real correlations even when velocity autocorr is at sampling limit.")

if __name__ == "__main__":
    main()