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

# Global parameters (can be modified)
# =====================================
# Default diffusion coefficient range for plotting
MIN_DEFF = 0.05
MAX_DEFF = 5.0
# Default track length range for plotting
MIN_LENGTH = 11
MAX_LENGTH = 100
# Plot settings
FONT_SIZE = 16
LINE_WIDTH = 2.5
FIGURE_SIZE = (10, 8)
# Complex analysis features
COMPLEX_ANALYSIS = True
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

def shorten_filename(name, max_length=20):
    """
    Shorten a filename for better readability in plots.
    
    Args:
        name: Filename to shorten
        max_length: Maximum length after shortening
        
    Returns:
        Shortened filename
    """
    if len(name) <= max_length:
        return name
    
    # Keep start and end, replace middle with "..."
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

def plot_median_diffusion(dmed, dsem, names, output_dir):
    """
    Create a plot of median diffusion coefficients with error bars.
    
    Args:
        dmed: List of median diffusion coefficients
        dsem: List of standard errors
        names: List of file names
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=FIGURE_SIZE)
    
    # Create x positions and shortened names for x-axis
    x_pos = np.arange(len(names))
    short_names = [shorten_filename(name) for name in names]
    
    # Plot median diffusion coefficients with error bars
    plt.bar(x_pos, dmed, yerr=dsem, capsize=5, alpha=0.7)
    
    # Scatter individual points on top for emphasis
    plt.scatter(x_pos, dmed, s=100, color='red', zorder=3)
    
    # Customize plot
    plt.ylabel(r'Median D$_\mathrm{eff}$ (µm$^2$/s)')
    plt.xlabel('Dataset')
    plt.title('Median Effective Diffusion Coefficients')
    plt.xticks(x_pos, short_names, rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot as PNG
    plt.savefig(os.path.join(output_dir, 'median_diffusion.png'))
    
    # Save data as CSV
    df = pd.DataFrame({
        'FileName': names,
        'MedianDiffusion': dmed,
        'StandardError': dsem
    })
    df.to_csv(os.path.join(output_dir, 'median_diffusion.csv'), index=False)
    
    plt.close()

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
    plt.xlim([0, 6])
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
    plt.xlim([0, 6])
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

def main():
    """Main function to generate diffusion coefficient plots."""
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
    
    # Create output directory for plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(file_paths[0]), f"diffusion_plots_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data from files
    print("\nExtracting data from files...")
    dlin, dmed, dsem, msd_ensemble, track_lengths, names = extract_data_from_files(file_paths)
    
    # Generate plots
    print("Generating plots...")
    
    # Always create these plots
    plot_median_diffusion(dmed, dsem, names, output_dir)
    plot_cdf(dlin, names, output_dir)
    plot_histogram(dlin, names, output_dir)
    
    # Create additional plots if complex analysis is enabled
    if COMPLEX_ANALYSIS:
        plot_track_lengths(track_lengths, names, output_dir)
        plot_msd_ensemble(msd_ensemble, names, output_dir)
    
    # Save summary statistics
    print("Saving summary statistics...")
    summary_data = []
    
    for i, name in enumerate(names):
        summary_data.append({
            'FileName': name,
            'MedianDiffusion': dmed[i],
            'MeanDiffusion': np.mean(dlin[i]) if dlin[i] else np.nan,
            'StdDiffusion': np.std(dlin[i]) if dlin[i] else np.nan,
            'SEMDiffusion': dsem[i],
            'NumTrajectories': len(dlin[i]),
            'MinDiffusion': np.min(dlin[i]) if dlin[i] else np.nan,
            'MaxDiffusion': np.max(dlin[i]) if dlin[i] else np.nan,
            'NumTracks': len(track_lengths[i]) if track_lengths[i] else 0,
            'MeanTrackLength': np.mean(track_lengths[i]) if track_lengths[i] else np.nan
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'diffusion_summary.csv'), index=False)
    
    print(f"\nAnalysis complete! Results saved in: {output_dir}")
    print("\nSummary of results:")
    for i, name in enumerate(names):
        print(f"\n{name}:")
        print(f"  Median Diffusion: {dmed[i]:.6f} µm²/s")
        print(f"  Standard Error: {dsem[i]:.6f} µm²/s")
        print(f"  Number of Trajectories: {len(dlin[i])}")

if __name__ == "__main__":
    main()