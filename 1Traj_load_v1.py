# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 12:53:48 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trajectory_processor.py

This script loads trajectory data from CSV files and processes it for diffusion analysis.
It's the first step in a multi-step process to analyze particle trajectories.

Input:
- CSV files containing trajectory data (similar to '_Tracks.csv' from TrackMate)

Output:
- Processed trajectory data saved as .pkl files for further analysis

Usage:
python trajectory_processor.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import glob
import pickle
from pathlib import Path

# Global parameters that can be modified
# =====================================
# Time step in seconds (default: 0.05s = 50ms)
DT = 0.1
# Conversion factor from pixels to μm (default: 1.0 for TrackMate which outputs in μm)
CONVERSION = 0.094
# Minimum track length (in frames) to consider for analysis
MIN_TRACK_LENGTH = 10
# =====================================

def load_trajectory_file(file_path):
    """
    Load a trajectory file with robust error handling.
    Tries multiple delimiters and handles various file formats.
    
    Args:
        file_path: Path to the trajectory file
        
    Returns:
        DataFrame containing the trajectory data or None if loading failed
    """
    print(f"Loading file: {file_path}")
    
    # List of delimiters to try
    delimiters = [',', '\t', ';', ' ']
    
    # Try each delimiter
    for delimiter in delimiters:
        try:
            # Try to read with the current delimiter
            df = pd.read_csv(file_path, delimiter=delimiter)
            
            # Check if we have required columns
            required_columns = ['Trajectory', 'Frame', 'x', 'y']
            if all(col in df.columns for col in required_columns):
                print(f"  Successfully loaded with delimiter: '{delimiter}'")
                return df
            else:
                # If the header names don't match expected, try without header
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter, header=None)
                    # Try to infer column meaning based on content
                    if df.shape[1] >= 4:  # At least trajectory, frame, x, y
                        # Rename columns based on typical position
                        df.columns = ['Trajectory', 'Frame', 'x', 'y'] + [f'col_{i+5}' for i in range(df.shape[1]-4)]
                        print(f"  Loaded without header using delimiter: '{delimiter}'")
                        print(f"  Assumed column order: Trajectory, Frame, x, y, ...")
                        return df
                except Exception as e:
                    continue
                    
        except Exception as e:
            continue
    
    # If we've tried all delimiters and none worked, try a more flexible approach
    try:
        # Try pandas' csv sniffer
        df = pd.read_csv(file_path, sep=None, engine='python')
        print("  Loaded using pandas' automatic delimiter detection")
        
        # Try to identify key columns
        col_names = df.columns.tolist()
        
        # Look for trajectory column
        traj_candidates = [col for col in col_names if 'traj' in col.lower() or 'track' in col.lower()]
        
        # Look for frame column
        frame_candidates = [col for col in col_names if 'frame' in col.lower() or 'time' in col.lower()]
        
        # Look for position columns
        x_candidates = [col for col in col_names if col.lower() == 'x' or 'pos_x' in col.lower()]
        y_candidates = [col for col in col_names if col.lower() == 'y' or 'pos_y' in col.lower()]
        
        # If we found good candidates, rename columns
        if traj_candidates and frame_candidates and x_candidates and y_candidates:
            df = df.rename(columns={
                traj_candidates[0]: 'Trajectory',
                frame_candidates[0]: 'Frame',
                x_candidates[0]: 'x',
                y_candidates[0]: 'y'
            })
            print("  Identified and renamed key columns")
            return df
        
        return df
    except Exception as e:
        # If all else fails, print debugging info and return None
        print(f"  ERROR: Could not load file: {e}")
        try:
            # Print first few lines to help diagnose the issue
            with open(file_path, 'r') as f:
                print("  First few lines of the file:")
                for i, line in enumerate(f):
                    if i < 5:  # Print only first 5 lines
                        print(f"    {line.strip()}")
                    else:
                        break
        except:
            pass
        
        return None

def process_trajectories(df, conversion_factor=CONVERSION):
    """
    Process trajectory data:
    1. Convert coordinates to μm if needed
    2. Calculate displacements and squared displacements
    3. Group by trajectory ID
    
    Args:
        df: DataFrame with trajectory data
        conversion_factor: Factor to convert pixel units to μm
        
    Returns:
        Dictionary with processed trajectory data
    """
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Convert coordinates to μm if needed
    data['x_um'] = data['x'] * conversion_factor
    data['y_um'] = data['y'] * conversion_factor
    
    # Initialize results dictionary
    processed_data = {
        'trajectories': [],
        'trajectory_lengths': [],
        'msd_data': [],
        'time_data': []
    }
    
    # Group by trajectory ID
    grouped = data.groupby('Trajectory')
    
    for trajectory_id, trajectory in grouped:
        # Sort by frame to ensure correct order
        trajectory = trajectory.sort_values('Frame')
        
        # Skip trajectories that are too short
        if len(trajectory) < MIN_TRACK_LENGTH:
            continue
            
        # Extract coordinates
        x = trajectory['x_um'].values
        y = trajectory['y_um'].values
        frames = trajectory['Frame'].values
        
        # Calculate time in seconds
        time = frames * DT
        
        # Calculate displacements
        dx = np.diff(x)
        dy = np.diff(y)
        dr2 = dx**2 + dy**2  # squared displacement
        
        # Calculate MSD for different time lags
        max_dt_index = len(trajectory) // 2
        msd = np.zeros(max_dt_index)
        
        for dt_index in range(1, max_dt_index + 1):
            # Calculate all possible displacements for this time lag
            total_displacement = 0
            count = 0
            
            for i in range(len(trajectory) - dt_index):
                dx = x[i + dt_index] - x[i]
                dy = y[i + dt_index] - y[i]
                total_displacement += dx**2 + dy**2
                count += 1
                
            if count > 0:
                msd[dt_index - 1] = total_displacement / count
            else:
                msd[dt_index - 1] = np.nan
        
        # Store trajectory data
        processed_data['trajectories'].append({
            'id': trajectory_id,
            'x': x,
            'y': y,
            'time': time,
            'dx': dx,
            'dy': dy,
            'dr2': dr2
        })
        
        processed_data['trajectory_lengths'].append(len(trajectory))
        processed_data['msd_data'].append(msd)
        processed_data['time_data'].append(np.arange(1, max_dt_index + 1) * DT)
    
    return processed_data

def create_diagnostic_plot(processed_data, output_path, filename):
    """
    Create diagnostic plots for the processed trajectory data.
    
    Args:
        processed_data: Dictionary containing the processed trajectory data
        output_path: Directory to save the plot
        filename: Base filename for the plot
    """
    if not processed_data['trajectories']:
        print("No trajectories to plot")
        return
        
    # Select a few trajectories to visualize (up to 5)
    num_trajectories = min(5, len(processed_data['trajectories']))
    selected_indices = np.linspace(0, len(processed_data['trajectories'])-1, num_trajectories, dtype=int)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Trajectory paths
    ax = axs[0, 0]
    for idx in selected_indices:
        traj = processed_data['trajectories'][idx]
        ax.plot(traj['x'], traj['y'], '-o', label=f"Traj {int(traj['id'])}")
    
    ax.set_xlabel('X position (μm)')
    ax.set_ylabel('Y position (μm)')
    ax.set_title('Sample Trajectories')
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Squared displacements vs time for selected trajectories
    ax = axs[0, 1]
    for idx in selected_indices:
        traj = processed_data['trajectories'][idx]
        ax.plot(traj['time'][1:], traj['dr2'], '-o', label=f"Traj {int(traj['id'])}")
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Squared displacement (μm²)')
    ax.set_title('Squared displacements vs time')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Histogram of trajectory lengths
    ax = axs[1, 0]
    ax.hist(processed_data['trajectory_lengths'], bins=20)
    ax.set_xlabel('Trajectory length (frames)')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of trajectory lengths')
    ax.grid(True)
    
    # Plot 4: MSD for selected trajectories
    ax = axs[1, 1]
    for idx in selected_indices:
        time_data = processed_data['time_data'][idx]
        msd_data = processed_data['msd_data'][idx]
        ax.plot(time_data, msd_data, '-o', label=f"Traj {int(processed_data['trajectories'][idx]['id'])}")
    
    ax.set_xlabel('Time lag (s)')
    ax.set_ylabel('Mean squared displacement (μm²)')
    ax.set_title('MSD vs time lag')
    ax.legend()
    ax.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{filename}_diagnostic.png"), dpi=300)
    plt.close()

def main():
    """Main function to process all trajectory files in a directory."""
    # Ask for input directory
    input_dir = input("Enter the directory containing trajectory files (press Enter for current directory): ")
    
    if input_dir == "":
        input_dir = os.getcwd()
    
    # Ask for file pattern
    file_pattern = input("Enter file pattern for trajectory files (e.g., '*.csv', press Enter for all CSV files): ")
    
    if file_pattern == "":
        file_pattern = "*.csv"
    
    # Get list of files
    file_paths = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not file_paths:
        print(f"No files matching '{file_pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(file_paths)} files to process")
    
    # Create output directory for intermediate files
    output_dir = os.path.join(input_dir, "processed_trajectories")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        
        # Load data
        df = load_trajectory_file(file_path)
        
        if df is None:
            print(f"Skipping {filename} due to loading errors")
            continue
        
        print(f"Processing {filename} with {len(df)} data points")
        
        # Process trajectories
        processed_data = process_trajectories(df)
        
        # Save processed data
        output_file = os.path.join(output_dir, f"tracked_{base_name}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"Processed data saved to {output_file}")
        
        # Create diagnostic plot
        create_diagnostic_plot(processed_data, output_dir, f"tracked_{base_name}")
        
    print(f"All files processed. Results saved in {output_dir}")

if __name__ == "__main__":
    main()