# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 14:40:17 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
noise_characterization.py

Advanced analysis of angle autocorrelation data to characterize and reduce noise.
Applies statistical mechanics methods to extract physical parameters and identify
different motion regimes.

Input:
- Single analyzed trajectory file (.pkl) from diffusion_analyzer.py

Output:
- Detailed noise analysis with multiple statistical mechanics approaches
- Fitted parameters and confidence intervals
- Diagnostic plots showing different analysis methods

Usage:
python noise_characterization.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from scipy.optimize import curve_fit
from scipy import stats
from scipy.signal import savgol_filter
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# Global parameters (can be modified)
# =====================================
# Time interval between frames in seconds
DT = 0.1
# Maximum time lag for analysis
MAX_LAG = 50
# Minimum trajectory length to consider
MIN_TRAJ_LENGTH = 15
# Smoothing parameters
SAVGOL_WINDOW = 7
SAVGOL_ORDER = 2
# Bootstrap iterations for confidence intervals
N_BOOTSTRAP = 100
# =====================================

def load_analyzed_data(file_path):
    """Load analyzed trajectory data from pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {len(data.get('trajectories', []))} trajectories")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_angle_autocorrelation_advanced(trajectories):
    """
    Calculate angle autocorrelation with advanced error analysis.
    """
    # Filter trajectories by length
    filtered_trajectories = [traj for traj in trajectories if len(traj['x']) > MIN_TRAJ_LENGTH]
    print(f"Using {len(filtered_trajectories)} trajectories (length > {MIN_TRAJ_LENGTH})")
    
    if not filtered_trajectories:
        return None
    
    # Store individual trajectory correlations for bootstrap analysis
    individual_correlations = []
    
    for traj in filtered_trajectories:
        x_temp = np.array(traj['x'])
        y_temp = np.array(traj['y'])
        
        # Calculate displacement vectors
        delta_x = np.diff(x_temp)
        delta_y = np.diff(y_temp)
        
        # Calculate correlation for this trajectory
        traj_correlation = np.zeros(MAX_LAG)
        
        for i in range(1, MAX_LAG + 1):
            if len(delta_x) <= i:
                traj_correlation[i-1] = np.nan
                continue
                
            cos_angles = []
            for k in range(len(delta_x) - i):
                dot_product = delta_x[k] * delta_x[k+i] + delta_y[k] * delta_y[k+i]
                mag1 = np.sqrt(delta_x[k]**2 + delta_y[k]**2)
                mag2 = np.sqrt(delta_x[k+i]**2 + delta_y[k+i]**2)
                
                if mag1 > 0 and mag2 > 0:
                    cos_angle = dot_product / (mag1 * mag2)
                    cos_angle = max(min(cos_angle, 1.0), -1.0)
                    cos_angles.append(cos_angle)
            
            if cos_angles:
                traj_correlation[i-1] = np.mean(cos_angles)
            else:
                traj_correlation[i-1] = np.nan
        
        individual_correlations.append(traj_correlation)
    
    # Convert to array and calculate statistics
    individual_correlations = np.array(individual_correlations)
    
    # Calculate mean and standard error
    mean_correlation = np.nanmean(individual_correlations, axis=0)
    std_correlation = np.nanstd(individual_correlations, axis=0)
    n_valid = np.sum(~np.isnan(individual_correlations), axis=0)
    sem_correlation = std_correlation / np.sqrt(np.maximum(n_valid, 1))
    
    return {
        'time_lags': np.arange(1, MAX_LAG + 1) * DT,
        'mean_correlation': mean_correlation,
        'sem_correlation': sem_correlation,
        'std_correlation': std_correlation,
        'individual_correlations': individual_correlations,
        'n_trajectories': len(filtered_trajectories)
    }

def exponential_decay_model(t, A, tau, offset=0):
    """Exponential decay model: A * exp(-t/tau) + offset"""
    return A * np.exp(-t/tau) + offset

def stretched_exponential_model(t, A, tau, beta, offset=0):
    """Stretched exponential: A * exp(-(t/tau)^beta) + offset"""
    return A * np.exp(-np.power(t/tau, beta)) + offset

def ornstein_uhlenbeck_model(t, tau_p):
    """Ornstein-Uhlenbeck correlation: exp(-t/tau_p)"""
    return np.exp(-t/tau_p)

def fit_models(time_lags, correlation, sem):
    """
    Fit different theoretical models to the correlation data.
    """
    # Remove NaN values
    valid_mask = ~np.isnan(correlation) & ~np.isnan(sem) & (sem > 0)
    if np.sum(valid_mask) < 5:
        return {}
    
    t_fit = time_lags[valid_mask]
    c_fit = correlation[valid_mask]
    s_fit = sem[valid_mask]
    
    # Only fit positive correlation region
    positive_mask = c_fit > 0
    if np.sum(positive_mask) < 3:
        return {}
    
    t_pos = t_fit[positive_mask]
    c_pos = c_fit[positive_mask]
    s_pos = s_fit[positive_mask]
    
    results = {}
    
    # 1. Simple exponential decay
    try:
        # Initial guess
        p0 = [c_pos[0], np.mean(t_pos), 0]
        bounds = ([0, 0.01, -1], [2, 10, 1])
        
        popt, pcov = curve_fit(exponential_decay_model, t_pos, c_pos, 
                              sigma=s_pos, p0=p0, bounds=bounds, maxfev=1000)
        
        # Calculate R-squared
        y_pred = exponential_decay_model(t_pos, *popt)
        ss_res = np.sum((c_pos - y_pred)**2)
        ss_tot = np.sum((c_pos - np.mean(c_pos))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        results['exponential'] = {
            'params': popt,
            'errors': np.sqrt(np.diag(pcov)),
            'r_squared': r_squared,
            'persistence_time': popt[1],
            'model_func': exponential_decay_model
        }
    except Exception as e:
        print(f"Exponential fit failed: {e}")
    
    # 2. Stretched exponential
    try:
        p0 = [c_pos[0], np.mean(t_pos), 1.0, 0]
        bounds = ([0, 0.01, 0.1, -1], [2, 10, 2.0, 1])
        
        popt, pcov = curve_fit(stretched_exponential_model, t_pos, c_pos,
                              sigma=s_pos, p0=p0, bounds=bounds, maxfev=1000)
        
        y_pred = stretched_exponential_model(t_pos, *popt)
        ss_res = np.sum((c_pos - y_pred)**2)
        ss_tot = np.sum((c_pos - np.mean(c_pos))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        results['stretched_exponential'] = {
            'params': popt,
            'errors': np.sqrt(np.diag(pcov)),
            'r_squared': r_squared,
            'persistence_time': popt[1],
            'stretch_exponent': popt[2],
            'model_func': stretched_exponential_model
        }
    except Exception as e:
        print(f"Stretched exponential fit failed: {e}")
    
    # 3. Ornstein-Uhlenbeck (normalized)
    try:
        # Normalize correlation
        c_norm = c_pos / c_pos[0] if c_pos[0] > 0 else c_pos
        
        popt, pcov = curve_fit(ornstein_uhlenbeck_model, t_pos, c_norm,
                              sigma=s_pos/c_pos[0] if c_pos[0] > 0 else s_pos,
                              p0=[np.mean(t_pos)], bounds=([0.01], [10]), maxfev=1000)
        
        y_pred = ornstein_uhlenbeck_model(t_pos, *popt) * c_pos[0]
        ss_res = np.sum((c_pos - y_pred)**2)
        ss_tot = np.sum((c_pos - np.mean(c_pos))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        results['ornstein_uhlenbeck'] = {
            'params': popt,
            'errors': np.sqrt(np.diag(pcov)),
            'r_squared': r_squared,
            'persistence_time': popt[0],
            'model_func': lambda t, tau: ornstein_uhlenbeck_model(t, tau) * c_pos[0]
        }
    except Exception as e:
        print(f"Ornstein-Uhlenbeck fit failed: {e}")
    
    return results

def bootstrap_confidence_intervals(individual_correlations, n_bootstrap=N_BOOTSTRAP):
    """
    Calculate confidence intervals using bootstrap resampling.
    """
    n_traj, n_lags = individual_correlations.shape
    
    bootstrap_means = np.zeros((n_bootstrap, n_lags))
    
    for i in range(n_bootstrap):
        # Resample trajectories with replacement
        indices = np.random.choice(n_traj, size=n_traj, replace=True)
        bootstrap_sample = individual_correlations[indices]
        bootstrap_means[i] = np.nanmean(bootstrap_sample, axis=0)
    
    # Calculate confidence intervals
    ci_lower = np.nanpercentile(bootstrap_means, 2.5, axis=0)
    ci_upper = np.nanpercentile(bootstrap_means, 97.5, axis=0)
    
    return ci_lower, ci_upper

def frequency_domain_analysis(correlation, time_lags):
    """
    Analyze correlation in frequency domain to identify characteristic frequencies.
    """
    # Remove NaN values and interpolate if needed
    valid_mask = ~np.isnan(correlation)
    if np.sum(valid_mask) < 10:
        return None
    
    # Interpolate to fill gaps
    correlation_clean = np.interp(time_lags, time_lags[valid_mask], correlation[valid_mask])
    
    # Apply window function to reduce edge effects
    window = np.hanning(len(correlation_clean))
    correlation_windowed = correlation_clean * window
    
    # FFT
    fft_result = fft(correlation_windowed)
    frequencies = fftfreq(len(correlation_windowed), d=DT)
    
    # Power spectral density
    power_spectrum = np.abs(fft_result)**2
    
    # Only positive frequencies
    positive_freq_mask = frequencies > 0
    freq_pos = frequencies[positive_freq_mask]
    power_pos = power_spectrum[positive_freq_mask]
    
    # Find dominant frequency
    peak_idx = np.argmax(power_pos)
    dominant_freq = freq_pos[peak_idx]
    
    return {
        'frequencies': freq_pos,
        'power_spectrum': power_pos,
        'dominant_frequency': dominant_freq,
        'characteristic_time': 1/dominant_freq if dominant_freq > 0 else np.inf
    }

def detect_outlier_trajectories(individual_correlations):
    """
    Identify outlier trajectories using robust statistical methods.
    """
    # Calculate correlation at first few time points for each trajectory
    early_correlation = np.nanmean(individual_correlations[:, :5], axis=1)
    
    # Use interquartile range method
    Q1 = np.nanpercentile(early_correlation, 25)
    Q3 = np.nanpercentile(early_correlation, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (early_correlation < lower_bound) | (early_correlation > upper_bound)
    
    return outlier_mask

def noise_reduction_analysis(correlation_data):
    """
    Apply various noise reduction techniques and compare results.
    """
    time_lags = correlation_data['time_lags']
    correlation = correlation_data['mean_correlation']
    
    # 1. Savitzky-Golay smoothing
    valid_mask = ~np.isnan(correlation)
    if np.sum(valid_mask) >= SAVGOL_WINDOW:
        correlation_smooth = np.full_like(correlation, np.nan)
        correlation_smooth[valid_mask] = savgol_filter(
            correlation[valid_mask], SAVGOL_WINDOW, SAVGOL_ORDER
        )
    else:
        correlation_smooth = correlation.copy()
    
    # 2. Moving average
    window_size = 5
    correlation_ma = np.full_like(correlation, np.nan)
    for i in range(len(correlation)):
        start_idx = max(0, i - window_size//2)
        end_idx = min(len(correlation), i + window_size//2 + 1)
        correlation_ma[i] = np.nanmean(correlation[start_idx:end_idx])
    
    # 3. Outlier removal based on individual trajectories
    individual_corr = correlation_data['individual_correlations']
    outlier_mask = detect_outlier_trajectories(individual_corr)
    
    if np.sum(~outlier_mask) > 0:
        correlation_no_outliers = np.nanmean(individual_corr[~outlier_mask], axis=0)
    else:
        correlation_no_outliers = correlation.copy()
    
    return {
        'original': correlation,
        'savgol_smoothed': correlation_smooth,
        'moving_average': correlation_ma,
        'outliers_removed': correlation_no_outliers,
        'outlier_fraction': np.sum(outlier_mask) / len(outlier_mask)
    }

def create_comprehensive_analysis_plot(correlation_data, fit_results, noise_analysis, freq_analysis, output_dir, filename):
    """
    Create a comprehensive diagnostic plot with multiple analysis panels.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    time_lags = correlation_data['time_lags']
    
    # Panel 1: Raw data with confidence intervals
    ax = axes[0, 0]
    ci_lower, ci_upper = bootstrap_confidence_intervals(correlation_data['individual_correlations'])
    
    ax.fill_between(time_lags, ci_lower, ci_upper, alpha=0.3, color='lightblue', label='95% CI')
    ax.errorbar(time_lags, correlation_data['mean_correlation'], 
                yerr=correlation_data['sem_correlation'], fmt='o-', 
                color='blue', alpha=0.7, label='Raw data')
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time lag (s)')
    ax.set_ylabel('Angle correlation')
    ax.set_title('Raw Data with Confidence Intervals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Model fits
    ax = axes[0, 1]
    
    # Plot raw data
    ax.errorbar(time_lags, correlation_data['mean_correlation'], 
                yerr=correlation_data['sem_correlation'], fmt='o', 
                color='blue', alpha=0.5, label='Data')
    
    # Plot fitted models
    t_fit = np.linspace(time_lags[0], time_lags[-1], 100)
    colors = ['red', 'green', 'orange']
    
    for i, (model_name, results) in enumerate(fit_results.items()):
        if i < len(colors):
            model_func = results['model_func']
            if model_name == 'ornstein_uhlenbeck':
                y_fit = model_func(t_fit, results['params'][0])
            else:
                y_fit = model_func(t_fit, *results['params'])
            
            ax.plot(t_fit, y_fit, '--', color=colors[i], 
                   label=f"{model_name} (R²={results['r_squared']:.3f})")
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time lag (s)')
    ax.set_ylabel('Angle correlation')
    ax.set_title('Model Fits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Noise reduction comparison
    ax = axes[0, 2]
    
    ax.plot(time_lags, noise_analysis['original'], 'o-', alpha=0.5, label='Original')
    ax.plot(time_lags, noise_analysis['savgol_smoothed'], '-', label='Savgol smooth')
    ax.plot(time_lags, noise_analysis['moving_average'], '-', label='Moving average')
    ax.plot(time_lags, noise_analysis['outliers_removed'], '-', 
           label=f'Outliers removed ({noise_analysis["outlier_fraction"]:.1%})')
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time lag (s)')
    ax.set_ylabel('Angle correlation')
    ax.set_title('Noise Reduction Methods')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Individual trajectory correlations (heatmap)
    ax = axes[1, 0]
    
    individual_corr = correlation_data['individual_correlations']
    # Sort trajectories by early correlation value
    early_corr = np.nanmean(individual_corr[:, :5], axis=1)
    sort_indices = np.argsort(early_corr)[::-1]
    
    # Plot first 20 trajectories
    n_show = min(20, len(sort_indices))
    im = ax.imshow(individual_corr[sort_indices[:n_show]], 
                   aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=1.0)
    
    ax.set_xlabel('Time lag index')
    ax.set_ylabel('Trajectory #')
    ax.set_title('Individual Trajectory Correlations')
    plt.colorbar(im, ax=ax, label='Correlation')
    
    # Panel 5: Frequency domain analysis
    ax = axes[1, 1]
    
    if freq_analysis:
        ax.loglog(freq_analysis['frequencies'], freq_analysis['power_spectrum'])
        ax.axvline(freq_analysis['dominant_frequency'], color='r', linestyle='--',
                  label=f'Dominant freq: {freq_analysis["dominant_frequency"]:.3f} Hz')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title('Power Spectral Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Frequency analysis\nnot available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Power Spectral Density')
    
    # Panel 6: Model comparison and residuals
    ax = axes[1, 2]
    
    if fit_results:
        model_names = list(fit_results.keys())
        r_squared_values = [fit_results[name]['r_squared'] for name in model_names]
        persistence_times = [fit_results[name]['persistence_time'] for name in model_names]
        
        bars = ax.bar(range(len(model_names)), r_squared_values, alpha=0.7)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('R² value')
        ax.set_title('Model Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add persistence times as text
        for i, (bar, tau) in enumerate(zip(bars, persistence_times)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'τ={tau:.3f}s', ha='center', va='bottom', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No successful\nmodel fits', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Model Comparison')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}_comprehensive_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def export_analysis_results(correlation_data, fit_results, noise_analysis, output_dir, filename):
    """
    Export all analysis results to CSV files.
    """
    # Main correlation data
    df_main = pd.DataFrame({
        'TimeLag_s': correlation_data['time_lags'],
        'MeanCorrelation': correlation_data['mean_correlation'],
        'SEM': correlation_data['sem_correlation'],
        'STD': correlation_data['std_correlation'],
        'SavgolSmoothed': noise_analysis['savgol_smoothed'],
        'MovingAverage': noise_analysis['moving_average'],
        'OutliersRemoved': noise_analysis['outliers_removed']
    })
    df_main.to_csv(os.path.join(output_dir, f'{filename}_correlation_analysis.csv'), index=False)
    
    # Model fit parameters
    if fit_results:
        fit_data = []
        for model_name, results in fit_results.items():
            fit_data.append({
                'Model': model_name,
                'R_squared': results['r_squared'],
                'PersistenceTime_s': results['persistence_time'],
                'Parameters': str(results['params']),
                'ParameterErrors': str(results['errors'])
            })
        
        df_fits = pd.DataFrame(fit_data)
        df_fits.to_csv(os.path.join(output_dir, f'{filename}_model_fits.csv'), index=False)
    
    # Summary statistics
    summary_data = {
        'Parameter': [
            'Number of trajectories',
            'Mean initial correlation',
            'Correlation at 0.5s',
            'Correlation at 1.0s', 
            'First zero crossing (s)',
            'Outlier fraction',
            'Best model',
            'Best model R²',
            'Best model persistence time (s)'
        ],
        'Value': [
            correlation_data['n_trajectories'],
            correlation_data['mean_correlation'][0] if len(correlation_data['mean_correlation']) > 0 else np.nan,
            np.interp(0.5, correlation_data['time_lags'], correlation_data['mean_correlation']),
            np.interp(1.0, correlation_data['time_lags'], correlation_data['mean_correlation']),
            np.nan,  # Will fill in zero crossing
            noise_analysis['outlier_fraction'],
            '',  # Will fill in best model
            np.nan,
            np.nan
        ]
    }
    
    # Find zero crossing
    zero_crossings = np.where(correlation_data['mean_correlation'] < 0)[0]
    if len(zero_crossings) > 0:
        summary_data['Value'][4] = correlation_data['time_lags'][zero_crossings[0]]
    
    # Find best model
    if fit_results:
        best_model = max(fit_results.keys(), key=lambda k: fit_results[k]['r_squared'])
        summary_data['Value'][6] = best_model
        summary_data['Value'][7] = fit_results[best_model]['r_squared']
        summary_data['Value'][8] = fit_results[best_model]['persistence_time']
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_dir, f'{filename}_summary.csv'), index=False)

def main():
    """
    Main function for advanced noise characterization analysis.
    """
    print("Advanced Angle Autocorrelation Noise Characterization")
    print("=====================================================")
    
    # Get input file
    input_file = input("Enter path to analyzed trajectory file (.pkl): ")
    
    if not os.path.isfile(input_file):
        print(f"File {input_file} not found")
        return
    
    # Load data
    analyzed_data = load_analyzed_data(input_file)
    if analyzed_data is None:
        return
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = f"noise_analysis_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Calculating angle autocorrelations...")
    correlation_data = calculate_angle_autocorrelation_advanced(analyzed_data['trajectories'])
    
    if correlation_data is None:
        print("Failed to calculate correlations")
        return
    
    print("Fitting theoretical models...")
    fit_results = fit_models(
        correlation_data['time_lags'], 
        correlation_data['mean_correlation'], 
        correlation_data['sem_correlation']
    )
    
    print("Applying noise reduction methods...")
    noise_analysis = noise_reduction_analysis(correlation_data)
    
    print("Performing frequency domain analysis...")
    freq_analysis = frequency_domain_analysis(
        correlation_data['mean_correlation'], 
        correlation_data['time_lags']
    )
    
    print("Creating comprehensive analysis plot...")
    create_comprehensive_analysis_plot(
        correlation_data, fit_results, noise_analysis, 
        freq_analysis, output_dir, base_name
    )
    
    print("Exporting results...")
    export_analysis_results(
        correlation_data, fit_results, noise_analysis, 
        output_dir, base_name
    )
    
    # Print summary to console
    print(f"\nAnalysis Summary for {base_name}:")
    print(f"Number of trajectories: {correlation_data['n_trajectories']}")
    print(f"Outlier fraction: {noise_analysis['outlier_fraction']:.1%}")
    
    if fit_results:
        print("\nModel Fit Results:")
        for model_name, results in fit_results.items():
            print(f"  {model_name}:")
            print(f"    R² = {results['r_squared']:.4f}")
            print(f"    Persistence time = {results['persistence_time']:.4f} s")
            if 'stretch_exponent' in results:
                print(f"    Stretch exponent = {results['stretch_exponent']:.4f}")
    
    if freq_analysis:
        print(f"\nFrequency Analysis:")
        print(f"  Dominant frequency: {freq_analysis['dominant_frequency']:.4f} Hz")
        print(f"  Characteristic time: {freq_analysis['characteristic_time']:.4f} s")
    
    print(f"\nDetailed results saved in: {output_dir}")

if __name__ == "__main__":
    main()