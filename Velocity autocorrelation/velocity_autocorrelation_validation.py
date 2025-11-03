# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 15:35:42 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
velocity_autocorrelation_validation.py

Advanced validation tools for velocity autocorrelation analysis using statistical mechanics methods.
Helps distinguish real correlations from sampling artifacts when correlation times are near sampling frequency.

Input:
- Analyzed trajectory .pkl files

Output:
- Allan variance analysis
- Power spectral density analysis
- Surrogate data testing
- Frequency domain validation
- Enhanced statistical metrics

. Allan Variance Analysis

Purpose: Distinguishes real correlations from different noise types
Interpretation:

α ≈ -1: White noise (good for autocorrelation)
α ≈ 0: Flicker noise
α ≈ +1: Random walk/strong correlations


Your case: If α > 0.5, correlations may be real despite sampling limits

2. Surrogate Data Testing

Purpose: Tests if autocorrelation is statistically significant
Method: Compares real data vs phase-randomized surrogates
Critical: If <5% of time points show significant autocorrelation, it's likely noise

3. Power Spectral Density

Purpose: Checks frequency content vs Nyquist limit
Red flag: If most power is near Nyquist frequency (5 Hz), you're undersampling



Usage:
python velocity_autocorrelation_validation.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
import warnings

# Global parameters (modify these as needed)
# =====================================
# Time step in seconds
DT = 0.1
# Minimum trajectory length
MIN_TRACK_LENGTH = 30
# Maximum time lag to analyze
MAX_TAU = 50

# Allan variance parameters
ALLAN_MAX_TAU = 20  # Maximum Allan time for analysis
N_SURROGATE = 100   # Number of surrogate datasets for testing

# Frequency analysis parameters
WELCH_NPERSEG = 256  # Segment length for Welch's method
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

def extract_velocity_timeseries(trajectories, dt=DT, min_length=MIN_TRACK_LENGTH):
    """
    Extract velocity time series from trajectories.
    
    Args:
        trajectories: List of trajectory dictionaries
        dt: Time step
        min_length: Minimum trajectory length
        
    Returns:
        Dictionary with velocity data
    """
    filtered_trajectories = [traj for traj in trajectories if len(traj['x']) > min_length]
    
    if not filtered_trajectories:
        return None
    
    velocity_series = []
    speed_series = []
    
    for traj in filtered_trajectories:
        x = np.array(traj['x'])
        y = np.array(traj['y'])
        
        # Calculate velocities
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        speed = np.sqrt(vx**2 + vy**2)
        
        velocity_series.append({'vx': vx, 'vy': vy, 'speed': speed, 'id': traj['id']})
        speed_series.extend(speed)
    
    return {
        'velocity_series': velocity_series,
        'all_speeds': np.array(speed_series),
        'n_trajectories': len(filtered_trajectories)
    }

def calculate_allan_variance(data, dt=DT, max_tau=ALLAN_MAX_TAU):
    """
    Calculate Allan variance to distinguish noise from real correlations.
    
    Allan variance σ²(τ) = <[x(t+τ) - x(t)]²>/2 for averaging time τ
    
    Args:
        data: 1D array of measurements
        dt: Sampling time
        max_tau: Maximum averaging time to analyze
        
    Returns:
        Dictionary with Allan variance results
    """
    if len(data) < 10:
        return None
    
    max_m = min(max_tau, len(data) // 3)  # Maximum number of points for averaging
    taus = []
    allan_vars = []
    
    for m in range(1, max_m + 1):
        # Averaging time
        tau = m * dt
        
        # Calculate Allan variance for this averaging time
        n_samples = len(data) - 2*m
        if n_samples <= 0:
            continue
        
        differences = []
        for i in range(n_samples):
            # Average over m points
            avg1 = np.mean(data[i:i+m])
            avg2 = np.mean(data[i+m:i+2*m])
            differences.append((avg2 - avg1)**2)
        
        if differences:
            allan_var = np.mean(differences) / 2
            taus.append(tau)
            allan_vars.append(allan_var)
    
    return {
        'taus': np.array(taus),
        'allan_variance': np.array(allan_vars),
        'n_points': len(data)
    }

def fit_allan_variance_model(taus, allan_vars):
    """
    Fit power-law model to Allan variance: σ²(τ) = A * τ^α
    
    α = -1: white noise
    α = 0: flicker noise
    α = +1: random walk
    
    Args:
        taus: Array of averaging times
        allan_vars: Array of Allan variances
        
    Returns:
        Dictionary with fit parameters
    """
    # Remove zero or negative values
    valid_mask = (allan_vars > 0) & (taus > 0)
    if np.sum(valid_mask) < 3:
        return None
    
    log_taus = np.log10(taus[valid_mask])
    log_vars = np.log10(allan_vars[valid_mask])
    
    try:
        # Linear fit in log space: log(σ²) = log(A) + α*log(τ)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_taus, log_vars)
        
        return {
            'alpha': slope,
            'log_A': intercept,
            'A': 10**intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        }
    except:
        return None

def calculate_power_spectral_density(data, dt=DT, nperseg=WELCH_NPERSEG):
    """
    Calculate power spectral density using Welch's method.
    
    Args:
        data: 1D array of measurements
        dt: Sampling time
        nperseg: Length of each segment for Welch's method
        
    Returns:
        Dictionary with PSD results
    """
    if len(data) < nperseg:
        nperseg = len(data) // 4
    
    try:
        frequencies, psd = signal.welch(data, fs=1/dt, nperseg=nperseg, 
                                       detrend='linear', scaling='density')
        
        return {
            'frequencies': frequencies,
            'psd': psd,
            'nyquist_freq': 1/(2*dt),
            'sampling_freq': 1/dt
        }
    except:
        return None

def generate_surrogate_data(data, method='phase_randomized'):
    """
    Generate surrogate data that preserves certain statistical properties.
    
    Args:
        data: Original data array
        method: Type of surrogate ('phase_randomized', 'shuffled')
        
    Returns:
        Surrogate data array
    """
    if method == 'phase_randomized':
        # Preserve power spectrum but randomize phases
        fft_data = fft(data)
        magnitudes = np.abs(fft_data)
        random_phases = np.random.uniform(0, 2*np.pi, len(fft_data))
        
        # Ensure Hermitian symmetry for real output
        random_phases[0] = 0  # DC component
        if len(data) % 2 == 0:
            random_phases[len(data)//2] = 0  # Nyquist frequency
        
        # Make conjugate symmetric
        for i in range(1, len(data)//2):
            random_phases[-i] = -random_phases[i]
        
        surrogate_fft = magnitudes * np.exp(1j * random_phases)
        surrogate = np.real(np.fft.ifft(surrogate_fft))
        
    elif method == 'shuffled':
        # Randomly shuffle the data
        surrogate = np.random.permutation(data)
    
    else:
        raise ValueError(f"Unknown surrogate method: {method}")
    
    return surrogate

def calculate_velocity_autocorrelation_simple(data, dt=DT, max_tau=MAX_TAU):
    """
    Simple velocity autocorrelation for 1D speed data.
    
    Args:
        data: 1D array of speed values
        dt: Time step
        max_tau: Maximum time lag
        
    Returns:
        Dictionary with autocorrelation results
    """
    if len(data) < max_tau + 1:
        max_tau = len(data) - 1
    
    autocorr = []
    time_lags = []
    
    # Calculate mean and variance
    mean_speed = np.mean(data)
    var_speed = np.var(data)
    
    if var_speed == 0:
        return None
    
    for tau in range(max_tau):
        if len(data) <= tau:
            break
        
        # Calculate autocorrelation
        if tau == 0:
            corr = 1.0
        else:
            covariance = np.mean((data[:-tau] - mean_speed) * (data[tau:] - mean_speed))
            corr = covariance / var_speed
        
        autocorr.append(corr)
        time_lags.append(tau * dt)
    
    return {
        'time_lags': np.array(time_lags),
        'autocorr': np.array(autocorr)
    }

def surrogate_test_autocorrelation(data, n_surrogates=N_SURROGATE, alpha=0.05):
    """
    Test if autocorrelation is significant compared to surrogate data.
    
    Args:
        data: Original speed data
        n_surrogates: Number of surrogate datasets
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    # Calculate original autocorrelation
    original_autocorr = calculate_velocity_autocorrelation_simple(data)
    
    if original_autocorr is None:
        return None
    
    # Generate surrogate autocorrelations
    surrogate_autocorrs = []
    
    for _ in range(n_surrogates):
        surrogate_data = generate_surrogate_data(data, method='phase_randomized')
        surrogate_result = calculate_velocity_autocorrelation_simple(surrogate_data)
        
        if surrogate_result is not None:
            surrogate_autocorrs.append(surrogate_result['autocorr'])
    
    if not surrogate_autocorrs:
        return None
    
    # Calculate confidence intervals from surrogates
    surrogate_autocorrs = np.array(surrogate_autocorrs)
    
    # Calculate percentiles
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    surrogate_lower = np.percentile(surrogate_autocorrs, lower_percentile, axis=0)
    surrogate_upper = np.percentile(surrogate_autocorrs, upper_percentile, axis=0)
    surrogate_mean = np.mean(surrogate_autocorrs, axis=0)
    
    # Test significance
    significant = ((original_autocorr['autocorr'] < surrogate_lower) | 
                  (original_autocorr['autocorr'] > surrogate_upper))
    
    return {
        'original_autocorr': original_autocorr['autocorr'],
        'time_lags': original_autocorr['time_lags'],
        'surrogate_mean': surrogate_mean,
        'surrogate_lower': surrogate_lower,
        'surrogate_upper': surrogate_upper,
        'significant': significant,
        'n_surrogates': len(surrogate_autocorrs)
    }

def analyze_dataset_advanced(dataset_dir, dataset_name):
    """
    Perform advanced analysis on a dataset.
    
    Args:
        dataset_dir: Directory containing trajectory files
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with advanced analysis results
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
    
    # Extract velocity data
    velocity_data = extract_velocity_timeseries(all_trajectories)
    
    if velocity_data is None:
        print(f"Failed to extract velocity data for {dataset_name}")
        return None
    
    print(f"Analyzing {len(velocity_data['velocity_series'])} trajectories for {dataset_name}")
    
    # Perform various analyses
    results = {
        'dataset_name': dataset_name,
        'n_trajectories': velocity_data['n_trajectories'],
        'velocity_data': velocity_data
    }
    
    # Allan variance analysis
    print("  Calculating Allan variance...")
    allan_result = calculate_allan_variance(velocity_data['all_speeds'])
    if allan_result:
        allan_fit = fit_allan_variance_model(allan_result['taus'], allan_result['allan_variance'])
        results['allan_variance'] = allan_result
        results['allan_fit'] = allan_fit
    
    # Power spectral density
    print("  Calculating power spectral density...")
    psd_result = calculate_power_spectral_density(velocity_data['all_speeds'])
    results['psd'] = psd_result
    
    # Surrogate testing
    print("  Performing surrogate data testing...")
    surrogate_result = surrogate_test_autocorrelation(velocity_data['all_speeds'])
    results['surrogate_test'] = surrogate_result
    
    return results

def create_validation_plots(results, output_path):
    """
    Create comprehensive validation plots.
    
    Args:
        results: Analysis results dictionary
        output_path: Directory to save plots
    """
    dataset_name = results['dataset_name']
    
    # Create a multi-panel figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Allan variance
    ax1 = axes[0, 0]
    if 'allan_variance' in results:
        allan = results['allan_variance']
        ax1.loglog(allan['taus'], allan['allan_variance'], 'o-', alpha=0.7)
        
        # Add fit line if available
        if 'allan_fit' in results and results['allan_fit']:
            fit = results['allan_fit']
            fit_line = fit['A'] * allan['taus']**fit['alpha']
            ax1.loglog(allan['taus'], fit_line, 'r--', 
                      label=f'α = {fit["alpha"]:.2f}, R² = {fit["r_squared"]:.3f}')
            ax1.legend()
        
        # Add reference lines for noise types
        ax1.axhline(y=allan['allan_variance'][0], color='gray', linestyle=':', alpha=0.5, 
                   label='White noise (α=-1)')
        
        ax1.set_xlabel('Averaging time τ (s)')
        ax1.set_ylabel('Allan variance σ²(τ)')
        ax1.set_title('Allan Variance Analysis')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Power spectral density
    ax2 = axes[0, 1]
    if 'psd' in results and results['psd']:
        psd = results['psd']
        ax2.loglog(psd['frequencies'], psd['psd'], alpha=0.7)
        ax2.axvline(psd['nyquist_freq'], color='red', linestyle='--', 
                   label=f'Nyquist freq = {psd["nyquist_freq"]:.1f} Hz')
        ax2.axvline(1/DT, color='orange', linestyle='--', 
                   label=f'Sampling freq = {1/DT:.1f} Hz')
        
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power spectral density')
        ax2.set_title('Power Spectral Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Autocorrelation with surrogate confidence intervals
    ax3 = axes[0, 2]
    if 'surrogate_test' in results and results['surrogate_test']:
        surr = results['surrogate_test']
        
        # Plot original autocorrelation
        ax3.plot(surr['time_lags'], surr['original_autocorr'], 'b-', 
                label='Original data', linewidth=2)
        
        # Plot surrogate confidence intervals
        ax3.fill_between(surr['time_lags'], surr['surrogate_lower'], 
                        surr['surrogate_upper'], alpha=0.3, color='gray',
                        label=f'95% surrogate CI (n={surr["n_surrogates"]})')
        
        ax3.plot(surr['time_lags'], surr['surrogate_mean'], 'r--', 
                label='Surrogate mean', alpha=0.7)
        
        # Highlight significant regions
        significant_mask = surr['significant']
        if np.any(significant_mask):
            ax3.scatter(surr['time_lags'][significant_mask], 
                       surr['original_autocorr'][significant_mask],
                       color='red', s=20, label='Significant', zorder=10)
        
        ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax3.axvline(DT, color='gray', linestyle=':', alpha=0.7,
                   label=f'Sampling period = {DT} s')
        
        ax3.set_xlabel('Time lag (s)')
        ax3.set_ylabel('Autocorrelation')
        ax3.set_title('Surrogate Data Test')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Speed distribution
    ax4 = axes[1, 0]
    if 'velocity_data' in results:
        speeds = results['velocity_data']['all_speeds']
        ax4.hist(speeds, bins=50, density=True, alpha=0.7)
        ax4.axvline(np.mean(speeds), color='red', linestyle='--', 
                   label=f'Mean = {np.mean(speeds):.3f} μm/s')
        ax4.axvline(np.median(speeds), color='orange', linestyle='--', 
                   label=f'Median = {np.median(speeds):.3f} μm/s')
        
        ax4.set_xlabel('Speed (μm/s)')
        ax4.set_ylabel('Probability density')
        ax4.set_title('Speed Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Noise analysis summary
    ax5 = axes[1, 1]
    
    # Create a summary of noise characteristics
    noise_metrics = []
    noise_values = []
    noise_colors = []
    
    if 'allan_fit' in results and results['allan_fit']:
        alpha = results['allan_fit']['alpha']
        noise_metrics.append(f'Allan α = {alpha:.2f}')
        
        if alpha < -0.5:
            noise_type = "White noise dominated"
            color = 'green'
        elif alpha > 0.5:
            noise_type = "Correlated motion"
            color = 'blue'
        else:
            noise_type = "Mixed behavior"
            color = 'orange'
        
        noise_values.append(abs(alpha))
        noise_colors.append(color)
    
    if 'surrogate_test' in results and results['surrogate_test']:
        surr = results['surrogate_test']
        n_significant = np.sum(surr['significant'])
        total_points = len(surr['significant'])
        sig_fraction = n_significant / total_points if total_points > 0 else 0
        
        noise_metrics.append(f'Significant: {sig_fraction:.1%}')
        noise_values.append(sig_fraction)
        noise_colors.append('red' if sig_fraction > 0.1 else 'gray')
    
    if noise_metrics:
        bars = ax5.bar(range(len(noise_metrics)), noise_values, color=noise_colors, alpha=0.7)
        ax5.set_xticks(range(len(noise_metrics)))
        ax5.set_xticklabels(noise_metrics, rotation=45, ha='right')
        ax5.set_ylabel('Metric value')
        ax5.set_title('Noise Analysis Summary')
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Frequency content near Nyquist
    ax6 = axes[1, 2]
    if 'psd' in results and results['psd']:
        psd = results['psd']
        
        # Focus on frequencies near Nyquist
        nyquist = psd['nyquist_freq']
        mask = psd['frequencies'] <= nyquist
        
        ax6.semilogx(psd['frequencies'][mask], psd['psd'][mask])
        ax6.axvline(nyquist/10, color='orange', linestyle='--', alpha=0.7,
                   label=f'0.1 × Nyquist = {nyquist/10:.1f} Hz')
        ax6.axvline(nyquist/2, color='red', linestyle='--', alpha=0.7,
                   label=f'0.5 × Nyquist = {nyquist/2:.1f} Hz')
        ax6.axvline(nyquist, color='red', linestyle='-', alpha=0.7,
                   label=f'Nyquist = {nyquist:.1f} Hz')
        
        ax6.set_xlabel('Frequency (Hz)')
        ax6.set_ylabel('Power spectral density')
        ax6.set_title('Frequency Content vs Nyquist Limit')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Advanced Validation Analysis: {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{dataset_name}_validation_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def export_validation_results(results, output_path):
    """Export validation results to CSV files."""
    dataset_name = results['dataset_name']
    
    # Allan variance results
    if 'allan_variance' in results:
        allan = results['allan_variance']
        allan_df = pd.DataFrame({
            'Averaging_time_s': allan['taus'],
            'Allan_variance': allan['allan_variance']
        })
        allan_df.to_csv(os.path.join(output_path, f'{dataset_name}_allan_variance.csv'), index=False)
    
    # PSD results
    if 'psd' in results and results['psd']:
        psd = results['psd']
        psd_df = pd.DataFrame({
            'Frequency_Hz': psd['frequencies'],
            'PSD': psd['psd']
        })
        psd_df.to_csv(os.path.join(output_path, f'{dataset_name}_power_spectrum.csv'), index=False)
    
    # Surrogate test results
    if 'surrogate_test' in results and results['surrogate_test']:
        surr = results['surrogate_test']
        surr_df = pd.DataFrame({
            'Time_lag_s': surr['time_lags'],
            'Original_autocorr': surr['original_autocorr'],
            'Surrogate_mean': surr['surrogate_mean'],
            'Surrogate_lower': surr['surrogate_lower'],
            'Surrogate_upper': surr['surrogate_upper'],
            'Significant': surr['significant']
        })
        surr_df.to_csv(os.path.join(output_path, f'{dataset_name}_surrogate_test.csv'), index=False)
    
    # Summary statistics
    summary_data = {
        'Parameter': ['Dataset', 'N_trajectories', 'Sampling_period_s', 'Nyquist_freq_Hz'],
        'Value': [dataset_name, results['n_trajectories'], DT, 1/(2*DT)]
    }
    
    if 'allan_fit' in results and results['allan_fit']:
        fit = results['allan_fit']
        summary_data['Parameter'].extend(['Allan_alpha', 'Allan_R_squared'])
        summary_data['Value'].extend([fit['alpha'], fit['r_squared']])
    
    if 'surrogate_test' in results and results['surrogate_test']:
        surr = results['surrogate_test']
        n_sig = np.sum(surr['significant'])
        total = len(surr['significant'])
        summary_data['Parameter'].extend(['Surrogate_significant_fraction', 'Surrogate_n_tests'])
        summary_data['Value'].extend([n_sig/total if total > 0 else 0, total])
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_path, f'{dataset_name}_validation_summary.csv'), index=False)

def main():
    """Main function for advanced velocity autocorrelation validation."""
    print("Advanced Velocity Autocorrelation Validation")
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
    output_dir = os.path.join(dataset_dir, f"validation_analysis_{dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform analysis
    print(f"\nPerforming advanced analysis on '{dataset_name}'...")
    results = analyze_dataset_advanced(dataset_dir, dataset_name)
    
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
    print(f"\nValidation Summary for {dataset_name}:")
    print(f"Results saved to: {output_dir}")
    print(f"Trajectories analyzed: {results['n_trajectories']}")
    
    if 'allan_fit' in results and results['allan_fit']:
        fit = results['allan_fit']
        print(f"\nAllan Variance Analysis:")
        print(f"  Power law exponent α = {fit['alpha']:.3f}")
        
        if fit['alpha'] < -0.5:
            print("  → White noise dominated (good for correlation analysis)")
        elif fit['alpha'] > 0.5:
            print("  → Strong correlations present (may affect sampling)")
        else:
            print("  → Mixed noise behavior")
        
        print(f"  Fit quality R² = {fit['r_squared']:.3f}")
    
    if 'surrogate_test' in results and results['surrogate_test']:
        surr = results['surrogate_test']
        n_sig = np.sum(surr['significant'])
        total = len(surr['significant'])
        sig_fraction = n_sig / total if total > 0 else 0
        
        print(f"\nSurrogate Data Test:")
        print(f"  Significant autocorrelation: {sig_fraction:.1%} of time points")
        
        if sig_fraction > 0.2:
            print("  → Strong evidence for real correlations")
        elif sig_fraction > 0.05:
            print("  → Moderate evidence for correlations")
        else:
            print("  → Weak evidence - may be dominated by noise")
    
    print(f"\nSampling Analysis:")
    print(f"  Sampling period: {DT} s")
    print(f"  Nyquist frequency: {1/(2*DT):.1f} Hz")
    print(f"  ⚠️  Correlation times < {2*DT} s are likely sampling artifacts")

if __name__ == "__main__":
    main()