# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 12:42:16 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_deff_median.py
This script replaces the MATLAB Get_Deff_Median function, generating publication-quality 
plots for diffusion analysis using pickle files produced by the trajectory analysis pipeline.

Input:
- Analyzed trajectory data (.pkl files) from diffusion_analyzer.py

Output:
- Publication-quality plots for diffusion coefficients
- CSV files with underlying plot data
- Summary statistics for diffusion coefficients
- Statistical tests for group comparisons
- Outlier detection and visualization

Usage:
python get_deff_median.py /path/to/analyzed_trajectories/*.pkl
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
from datetime import datetime
from scipy import stats
from statsmodels.robust import scale
import warnings
from itertools import combinations

# Global parameters (can be modified)
# =====================================
# Default diffusion coefficient range for plotting
MIN_DEFF = 0.05
MAX_DEFF = 0.7
# Default track length range for plotting
MIN_LENGTH = 11
MAX_LENGTH = 30
# Plot settings
FONT_SIZE = 16
LINE_WIDTH = 2.5
FIGURE_SIZE = (10, 8)
# Complex analysis features
COMPLEX_ANALYSIS = True
# Statistical significance level
ALPHA = 0.05
# Outlier detection settings (based on Modified Z-Score)
OUTLIER_THRESHOLD = 3.5
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

def extract_data_from_files(file_paths):
    """
    Extract diffusion coefficients and other data from analyzed files.
    
    Args:
        file_paths: List of paths to analyzed data files
        
    Returns:
        Tuple containing extracted data
    """
    dlin = []  # List of diffusion coefficients for each file
    dmed = []  # Median diffusion coefficient for each file
    dsem = []  # Standard error of the mean for each file
    msd_ensemble = []  # MSD ensemble data for each file
    track_lengths = []  # Track lengths for each file
    names = []  # File names
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0].replace('analyzed_', '')
        names.append(base_name)
        
        # Load data
        data = load_analyzed_data(file_path)
        
        if data is None:
            print(f"Skipping {file_name} due to loading errors")
            # Add placeholder values
            dlin.append([])
            dmed.append(np.nan)
            dsem.append(np.nan)
            msd_ensemble.append([])
            track_lengths.append([])
            continue
        
        # Extract diffusion coefficients
        d_values = data['diffusion_coefficients']
        dlin.append(d_values)
        
        # Calculate median and SEM
        if d_values:
            dmed.append(np.median(d_values))
            dsem.append(np.std(d_values) / np.sqrt(len(d_values)))
        else:
            dmed.append(np.nan)
            dsem.append(np.nan)
        
        # Extract MSD data and track lengths
        msd_all = []
        lengths_all = []
        
        for traj in data['trajectories']:
            if not np.isnan(traj['D']):
                msd_all.append(traj['msd_data'])
                lengths_all.append(traj['track_length'])
        
        # Calculate ensemble average MSD if possible
        if msd_all:
            # Find minimum length for alignment
            min_length = min(len(msd) for msd in msd_all)
            # Truncate all MSDs to the same length
            msd_all_aligned = [msd[:min_length] for msd in msd_all]
            # Calculate ensemble average
            msd_avg = np.nanmean(msd_all_aligned, axis=0)
            msd_ensemble.append(msd_avg)
        else:
            msd_ensemble.append([])
        
        track_lengths.append(lengths_all)
    
    return dlin, dmed, dsem, msd_ensemble, track_lengths, names

def detect_outliers(data, threshold=OUTLIER_THRESHOLD):
    """
    Detect outliers using Modified Z-Score method.
    
    Args:
        data: Array-like with numerical values
        threshold: Threshold for Modified Z-Score (default: 3.5)
        
    Returns:
        Boolean array indicating which values are outliers
    """
    # Convert input to numpy array
    data = np.array(data)
    
    # Calculate median
    median = np.median(data)
    
    # Calculate MAD (Median Absolute Deviation)
    mad = stats.median_abs_deviation(data, scale=1)
    
    # Avoid division by zero
    if mad == 0:
        return np.zeros_like(data, dtype=bool)
    
    # Calculate Modified Z-Score
    modified_z_score = 0.6745 * np.abs(data - median) / mad
    
    # Identify outliers
    outliers = modified_z_score > threshold
    
    return outliers

def shorten_filename(name, max_length=20):
    """
    Shorten a filename for better readability in plots.
    Extracts the embryo identifier from the standard pipeline naming convention
    (e.g. 'tracked_Traj_Em6.nd2_crop' -> 'Em6').
    Falls back to middle-truncation for non-standard names.
    """
    import re
    # Extract identifier between 'Traj_' and '.nd2' (handles Em6, Em002, etc.)
    m = re.search(r'Traj_(.+?)\.nd2', name)
    if m:
        return m.group(1)
    if len(name) <= max_length:
        return name
    half_length = (max_length - 3) // 2
    return name[:half_length] + "..." + name[-half_length:]

def setup_plot_style():
    """Set up a consistent plot style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE + 2,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE - 2,
        'ytick.labelsize': FONT_SIZE - 2,
        'legend.fontsize': FONT_SIZE - 2,
        'lines.linewidth': LINE_WIDTH,
        'figure.figsize': FIGURE_SIZE,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def calculate_effect_size(group1, group2):
    """
    Calculate effect size (Cliff's delta) between two groups.
    
    Args:
        group1, group2: Arrays of values to compare
        
    Returns:
        Cliff's delta effect size
    """
    # If either group is empty, return NaN
    if len(group1) == 0 or len(group2) == 0:
        return np.nan
    
    # Calculate dominance matrix
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
    """
    Interpret Cliff's delta effect size.
    
    Args:
        delta: Cliff's delta value
        
    Returns:
        String interpretation of effect size
    """
    abs_delta = abs(delta)
    
    if abs_delta < 0.147:
        return "Negligible"
    elif abs_delta < 0.33:
        return "Small"
    elif abs_delta < 0.474:
        return "Medium"
    else:
        return "Large"

def perform_statistical_tests(dlin, names, output_dir):
    """
    Perform statistical tests for group comparisons.
    
    Args:
        dlin: List of lists of diffusion coefficients
        names: List of file names
        output_dir: Directory to save the results
        
    Returns:
        DataFrame with statistical test results
    """
    # Create combinations of datasets for pairwise comparisons
    pairs = list(combinations(range(len(names)), 2))
    
    # Initialize results list
    results = []
    
    # Perform pairwise comparisons
    for i, j in pairs:
        # Skip if either dataset is empty
        if not dlin[i] or not dlin[j]:
            continue
            
        # Get group names
        name_i = names[i]
        name_j = names[j]
        
        # Extract data
        data_i = dlin[i]
        data_j = dlin[j]
        
        # Mann-Whitney U test
        try:
            u_stat, p_value = stats.mannwhitneyu(data_i, data_j, alternative='two-sided')
            significant = p_value < ALPHA
        except ValueError as e:
            warnings.warn(f"Mann-Whitney U test failed for {name_i} vs {name_j}: {e}")
            u_stat, p_value, significant = np.nan, np.nan, False
        
        # Kolmogorov-Smirnov test
        try:
            ks_stat, ks_p_value = stats.ks_2samp(data_i, data_j)
            ks_significant = ks_p_value < ALPHA
        except ValueError as e:
            warnings.warn(f"Kolmogorov-Smirnov test failed for {name_i} vs {name_j}: {e}")
            ks_stat, ks_p_value, ks_significant = np.nan, np.nan, False
        
        # Calculate effect size (Cliff's delta)
        cliff_delta = calculate_effect_size(data_i, data_j)
        effect_size_interpretation = interpret_cliffs_delta(cliff_delta)
        
        # Store results
        results.append({
            'Group1': name_i,
            'Group2': name_j,
            'MannWhitney_U': u_stat,
            'MannWhitney_pvalue': p_value,
            'MannWhitney_significant': significant,
            'KS_statistic': ks_stat,
            'KS_pvalue': ks_p_value,
            'KS_significant': ks_significant,
            'CliffDelta': cliff_delta,
            'EffectSize': effect_size_interpretation,
            'Median1': np.median(data_i),
            'Median2': np.median(data_j),
            'MedianDiff': np.median(data_i) - np.median(data_j),
            'Count1': len(data_i),
            'Count2': len(data_j)
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(os.path.join(output_dir, 'statistical_tests.csv'), index=False)
    
    return results_df

def plot_median_bar_graph(dlin, names, output_dir):
    """
    Create a bar graph of median diffusion coefficients.
    
    Args:
        dlin: List of lists of diffusion coefficients
        names: List of file names
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=FIGURE_SIZE)
    
    # Calculate medians and standard errors
    medians = []
    errors = []
    short_names = []
    
    # Print data loading info for debugging
    print(f"Preparing data for {len(dlin)} datasets:")
    
    for i, (d_values, name) in enumerate(zip(dlin, names)):
        print(f"Dataset {i}: {name} - {len(d_values)} points")
        if not d_values:
            print(f"Warning: No data points for {name}")
            continue
        
        medians.append(np.median(d_values))
        # Use bootstrap to estimate standard error of median
        errors.append(np.std(d_values) / np.sqrt(len(d_values)))
        short_names.append(shorten_filename(name))
    
    # Check if we have any data
    if not medians:
        print("No data to plot!")
        return
    
    # Create bar graph with error bars
    x = np.arange(len(medians))
    plt.bar(x, medians, yerr=errors, capsize=5)
    
    # Customize plot
    plt.xticks(x, short_names, rotation=45, ha='right')
    plt.ylabel(r'Median D$_\mathrm{eff}$ (µm$^2$/s)')
    plt.title('Median Effective Diffusion Coefficients')
    y_max = max(medians) + max(errors) if errors else max(medians)
    plt.ylim([0, y_max * 1.2])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'median_diffusion.png'))
    
    # Save data as CSV
    df = pd.DataFrame({
        'Dataset': short_names,
        'Median': medians,
        'StandardError': errors
    })
    df.to_csv(os.path.join(output_dir, 'median_diffusion.csv'), index=False)
    
    plt.close()
def plot_outlier_analysis(dlin, names, output_dir):
    """
    Create plots showing diffusion coefficients with outliers highlighted.
    
    Args:
        dlin: List of lists of diffusion coefficients
        names: List of file names
        output_dir: Directory to save the plots
    """
    # Create output directory for outlier analysis
    outlier_dir = os.path.join(output_dir, 'outlier_analysis')
    os.makedirs(outlier_dir, exist_ok=True)
    
    # Initialize data for outlier summary
    outlier_summary = []
    
    # Process each dataset
    for i, (d_values, name) in enumerate(zip(dlin, names)):
        if not d_values:
            continue
        
        # Detect outliers
        outliers = detect_outliers(d_values)
        outlier_indices = np.where(outliers)[0]
        
        # Calculate percentage of outliers
        outlier_percentage = 100 * np.sum(outliers) / len(d_values)
        
        # Store summary info
        outlier_summary.append({
            'Dataset': name,
            'TotalPoints': len(d_values),
            'Outliers': np.sum(outliers),
            'OutlierPercentage': outlier_percentage,
            'MedianWithOutliers': np.median(d_values),
            'MedianWithoutOutliers': np.median(np.array(d_values)[~outliers]) if np.sum(~outliers) > 0 else np.nan
        })
        
        # Create diagnostic plot for this dataset
        plt.figure(figsize=FIGURE_SIZE)
        
        # Create histogram with outliers highlighted
        plt.hist(np.array(d_values)[~outliers], bins=30, alpha=0.7, label='Normal Points')
        if np.sum(outliers) > 0:
            plt.hist(np.array(d_values)[outliers], bins=30, alpha=0.7, color='red', label='Outliers')
        
        # Add vertical lines for medians
        plt.axvline(x=np.median(d_values), color='black', linestyle='-', 
                   label=f'Median (all): {np.median(d_values):.4f}')
        
        if np.sum(~outliers) > 0:
            plt.axvline(x=np.median(np.array(d_values)[~outliers]), color='blue', linestyle='--',
                       label=f'Median (no outliers): {np.median(np.array(d_values)[~outliers]):.4f}')
        
        # Customize plot
        plt.xlabel(r'D$_\mathrm{eff}$ (µm$^2$/s)')
        plt.ylabel('Frequency')
        plt.title(f'Outlier Analysis: {shorten_filename(name)}\n({np.sum(outliers)} outliers, {outlier_percentage:.1f}%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(outlier_dir, f'outliers_{name}.png'))
        plt.close()
    
    # Create outlier summary DataFrame
    outlier_df = pd.DataFrame(outlier_summary)
    
    # Save summary to CSV
    outlier_df.to_csv(os.path.join(outlier_dir, 'outlier_summary.csv'), index=False)
    
    # Create summary boxplot comparing medians with and without outliers
    if len(outlier_df) > 0:
        plt.figure(figsize=FIGURE_SIZE)
        
        # Prepare data for plotting
        plot_data = []
        for _, row in outlier_df.iterrows():
            plot_data.append({
                'Dataset': shorten_filename(row['Dataset']),
                'Median': row['MedianWithOutliers'],
                'Type': 'With Outliers'
            })
            plot_data.append({
                'Dataset': shorten_filename(row['Dataset']),
                'Median': row['MedianWithoutOutliers'],
                'Type': 'Without Outliers'
            })
        
        # Create DataFrame
        plot_df = pd.DataFrame(plot_data)
        
        # Create grouped bar chart
        sns.barplot(x='Dataset', y='Median', hue='Type', data=plot_df)
        
        # Customize plot
        plt.ylabel(r'Median D$_\mathrm{eff}$ (µm$^2$/s)')
        plt.title('Effect of Outliers on Median Diffusion Coefficients')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(outlier_dir, 'median_comparison.png'))
        plt.close()
    
    return outlier_df

def plot_cdf(dlin, names, output_dir):
    """
    Create a CDF plot of diffusion coefficients.
    
    Args:
        dlin: List of lists of diffusion coefficients
        names: List of file names
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=FIGURE_SIZE)
    
    # Define colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    
    # Store data for CSV
    all_data = []
    
    # Plot CDF for each dataset
    for i, (d_values, name) in enumerate(zip(dlin, names)):
        if not d_values:
            continue
            
        # Calculate ECDF
        sorted_data = np.sort(d_values)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # Plot CDF
        plt.plot(sorted_data, y, '-', color=colors[i], linewidth=LINE_WIDTH, 
                 label=shorten_filename(name))
        
        # Store data for CSV
        for x, f in zip(sorted_data, y):
            all_data.append({
                'FileName': name,
                'DiffusionCoefficient': x,
                'CumulativeProbability': f
            })
    
    # Customize plot
    plt.xlabel(r'D$_\mathrm{eff}$ (µm$^2$/s)')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Effective Diffusion Coefficients')
    all_values = [val for d_list in dlin for val in d_list if d_list]
    x_max = np.percentile(all_values, 99) if all_values else 1
    plt.xlim([0, x_max])
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'cdf_diffusion.png'))
    
    # Save data as CSV
    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(output_dir, 'cdf_diffusion.csv'), index=False)
    
    plt.close()

def plot_histogram(dlin, names, output_dir):
    """
    Create a histogram plot of diffusion coefficients.
    
    Args:
        dlin: List of lists of diffusion coefficients
        names: List of file names
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=FIGURE_SIZE)
    
    # Define colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    
    # Store data for CSV
    all_data = []
    
    # Get max count for normalization
    max_count = 0
    bin_width = 0.05
    bin_range = (0, 6)
    
    for d_values in dlin:
        if not d_values:
            continue
        
        # Calculate histogram
        counts, _ = np.histogram(d_values, bins=np.arange(bin_range[0], bin_range[1] + bin_width, bin_width))
        max_count = max(max_count, np.max(counts))
    
    # Plot normalized histogram for each dataset
    for i, (d_values, name) in enumerate(zip(dlin, names)):
        if not d_values:
            continue
            
        # Calculate histogram
        counts, bin_edges = np.histogram(d_values, bins=np.arange(bin_range[0], bin_range[1] + bin_width, bin_width))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Normalize counts
        normalized_counts = counts / max_count if max_count > 0 else counts
        
        # Plot histogram as line
        plt.plot(bin_centers, normalized_counts, '-', color=colors[i], linewidth=LINE_WIDTH,
                 label=shorten_filename(name))
        
        # Store data for CSV
        for x, y in zip(bin_centers, normalized_counts):
            all_data.append({
                'FileName': name,
                'DiffusionCoefficient': x,
                'NormalizedFrequency': y
            })
    
    # Customize plot
    plt.xlabel(r'D$_\mathrm{eff}$ (µm$^2$/s)')
    plt.ylabel('Normalized Frequency')
    plt.title('Distribution of Effective Diffusion Coefficients')
    all_values = [val for d_list in dlin for val in d_list if d_list]
    x_max = np.percentile(all_values, 99) if all_values else 1
    plt.xlim([0, x_max])
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'histogram_diffusion.png'))
    
    # Save data as CSV
    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(output_dir, 'histogram_diffusion.csv'), index=False)
    
    plt.close()

def plot_track_lengths(track_lengths, names, output_dir):
    """
    Create a plot of track length distributions.
    
    Args:
        track_lengths: List of lists of track lengths
        names: List of file names
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=FIGURE_SIZE)
    
    # Define colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    
    # Store data for CSV
    all_data = []
    
    # Plot track length histogram for each dataset
    for i, (lengths, name) in enumerate(zip(track_lengths, names)):
        if not lengths:
            continue
            
        # Calculate histogram
        bin_width = 5
        counts, bin_edges = np.histogram(lengths, bins=np.arange(0, 100 + bin_width, bin_width))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot histogram as line
        plt.plot(bin_centers, counts, '-', color=colors[i], linewidth=LINE_WIDTH,
                 label=shorten_filename(name))
        
        # Store data for CSV
        for x, y in zip(bin_centers, counts):
            all_data.append({
                'FileName': name,
                'TrackLength': x,
                'Frequency': y
            })
    
    # Customize plot
    plt.xlabel('Track Length (frames)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Track Lengths')
    plt.yscale('log')
    plt.xlim([0, 50])
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'track_lengths.png'))
    
    # Save data as CSV
    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(output_dir, 'track_lengths.csv'), index=False)
    
    plt.close()

def plot_msd_ensemble(msd_ensemble, names, output_dir):
    """
    Create a plot of ensemble-averaged MSD curves.
    
    Args:
        msd_ensemble: List of arrays of MSD values
        names: List of file names
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=FIGURE_SIZE)
    
    # Define colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    
    # Store data for CSV
    all_data = []
    
    # Plot MSD ensemble for each dataset
    for i, (msd_data, name) in enumerate(zip(msd_ensemble, names)):
        if not len(msd_data):
            continue
            
        # Create time points
        timepoints = np.arange(1, len(msd_data) + 1)
        
        # Plot MSD curve
        plt.plot(timepoints, msd_data, '-', color=colors[i], linewidth=LINE_WIDTH,
                 label=shorten_filename(name))
        
        # Store data for CSV
        for t, msd in zip(timepoints, msd_data):
            all_data.append({
                'FileName': name,
                'TimeLag': t,
                'MSD': msd
            })
    
    # Customize plot
    plt.xlabel('Time Lag')
    plt.ylabel(r'MSD (µm$^2$)')
    plt.title('Ensemble-Averaged Mean Squared Displacement')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'msd_ensemble.png'))
    
    # Save data as CSV
    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(output_dir, 'msd_ensemble.csv'), index=False)
    
    plt.close()

def plot_statistical_comparisons(stat_results, output_dir):
    """
    Create a plot of statistical comparison results.
    
    Args:
        stat_results: DataFrame with statistical test results
        output_dir: Directory to save the plot
    """
    # Create output directory for statistical analysis
    stat_dir = os.path.join(output_dir, 'statistical_analysis')
    os.makedirs(stat_dir, exist_ok=True)
    
    # Extract effect sizes
    labels = []
    effect_sizes = []
    
    for _, row in stat_results.iterrows():
        labels.append(f"{row['Group1']} vs {row['Group2']}")
        effect_sizes.append(row['CliffDelta'])
    
    # Create bar plot of effect sizes
    plt.figure(figsize=FIGURE_SIZE)
    x = range(len(labels))
    bars = plt.bar(x, effect_sizes, color=[
        'red' if abs(e) >= 0.474 else
        'orange' if abs(e) >= 0.33 else
        'yellow' if abs(e) >= 0.147 else
        'gray' for e in effect_sizes
    ])
    
    # Annotate bars with effect size interpretation
    for i, row in enumerate(stat_results.itertuples()):
        effect = getattr(row, 'EffectSize')
        plt.text(i, effect_sizes[i] * 1.1 if effect_sizes[i] > 0 else effect_sizes[i] * 0.9,
                effect, ha='center', va='center', rotation=90, fontsize=10)
    
    # Customize plot
    plt.xlabel('Dataset Comparison')
    plt.ylabel('Cliff\'s Delta Effect Size')
    plt.title('Effect Size of Diffusion Coefficient Differences')
    plt.xticks(x, [shorten_filename(label) for label in labels], rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.axhspan(-0.147, 0.147, alpha=0.1, color='gray', label='Negligible')
    plt.axhspan(0.147, 0.33, alpha=0.1, color='yellow', label='Small')
    plt.axhspan(0.33, 0.474, alpha=0.1, color='orange', label='Medium')
    plt.axhspan(0.474, 1, alpha=0.1, color='red', label='Large')
    plt.axhspan(-0.33, -0.147, alpha=0.1, color='yellow')
    plt.axhspan(-0.474, -0.33, alpha=0.1, color='orange')
    plt.axhspan(-1, -0.474, alpha=0.1, color='red')
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(stat_dir, 'effect_sizes.png'))
    plt.close()
def main():
    print("Diffusion Coefficient Analysis and Plotting")
    print("===========================================")
    
    # Set up plot style
    setup_plot_style()
    
    # Ask for input directory
    input_dir = input("Enter the directory containing analyzed trajectory files (press Enter for analyzed_trajectories): ")
    
    if input_dir == "":
        # Default to the analyzed_trajectories directory in the current folder
        input_dir = os.path.join(os.getcwd(), "analyzed_trajectories")
    
    # Check if directory exists
    if not os.path.isdir(input_dir):
        print(f"Directory {input_dir} does not exist")
        return
    
    # Get list of analyzed files
    file_paths = glob.glob(os.path.join(input_dir, "analyzed_*.pkl"))
    
    print(f"Found {len(file_paths)} files to analyze")
    
    # Get input files
    input_files = file_paths
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(file_paths[0]), f"diffusion_plots_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plot style
    setup_plot_style()
    
    # Extract data from files
    dlin, dmed, dsem, msd_ensemble, track_lengths, names = extract_data_from_files(input_files)
    
    # Generate plots and analysis
    plot_median_bar_graph(dlin, names, output_dir)
    plot_cdf(dlin, names, output_dir)
    plot_histogram(dlin, names, output_dir)
    plot_track_lengths(track_lengths, names, output_dir)
    plot_msd_ensemble(msd_ensemble, names, output_dir)
    
    # Perform statistical tests
    stat_results = perform_statistical_tests(dlin, names, output_dir)
    
    # Plot statistical comparisons
    plot_statistical_comparisons(stat_results, output_dir)
    
    # Perform outlier analysis
    outlier_results = plot_outlier_analysis(dlin, names, output_dir)
    
    # Print summary
    print(f"Analysis completed. Results saved in: {output_dir}")

if __name__ == "__main__":
    main()