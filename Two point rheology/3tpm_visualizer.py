# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 17:12:50 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tpm_visualizer.py

This script provides an interactive visualization tool for exploring two-point
microrheology (TPM) results. It allows users to interactively select time lags
and distance ranges, and visualize correlation functions and viscoelastic properties.

Input:
- TPM analysis results (.pkl files) from two_point_rheology.py

Output:
- Interactive visualization of TPM results
- Selected views can be saved as high-quality plots

Dependencies:
- matplotlib with interactive backend
- numpy, pandas, pickle

Usage:
python tpm_visualizer.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import glob
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots

# Global parameters that can be modified
# =====================================
# Time step in seconds
DT = 0.1
# Temperature (K)
TEMPERATURE = 298.15
# =====================================

def load_tpm_results(file_path):
    """
    Load TPM results from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary containing the TPM results
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading TPM results from {file_path}: {e}")
        return None

class TPMVisualizer:
    def __init__(self, tpm_results):
        """
        Initialize the visualizer with TPM results.
        
        Args:
            tpm_results: Dictionary with TPM results from two_point_rheology.py
        """
        self.tpm_results = tpm_results
        self.current_time_lag_idx = 0
        self.current_display = 'Dr'  # Default to longitudinal correlation
        self.log_scale = True
        self.show_errorbars = True
        
        # Initialize the plot
        self.setup_plot()
    
    def setup_plot(self):
        """
        Set up the interactive plot.
        """
        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(left=0.25, bottom=0.25)
        
        # Get time lags
        self.time_lags = self.tpm_results['time_lags']
        
        if not self.time_lags:
            print("No time lags available in the data")
            return
        
        # Get initial data
        initial_time_lag = self.time_lags[self.current_time_lag_idx]
        initial_data = self.tpm_results['binned_correlations'][self.current_time_lag_idx]
        
        # Initial plot
        self.line, = self.ax.plot(initial_data['bin_centers'], 
                                 initial_data['Dr'], 'o-', label='Dr')
        
        if self.show_errorbars:
            self.errorbar = self.ax.errorbar(initial_data['bin_centers'], 
                                           initial_data['Dr'], 
                                           yerr=initial_data['Dr_err'],
                                           fmt='none', ecolor='gray', alpha=0.5)
        else:
            self.errorbar = None
        
        # Set up axis labels and title
        self.ax.set_xlabel('Separation distance r (μm)')
        self.ax.set_ylabel('Correlation (μm²)')
        self.title = self.ax.set_title(f'Correlation function (τ = {initial_time_lag * DT:.2f} s)')
        
        # Set log scale if needed
        if self.log_scale:
            self.ax.set_xscale('log')
            self.ax.set_yscale('log')
        
        # Add grid
        self.ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Add time lag slider
        axcolor = 'lightgoldenrodyellow'
        self.ax_time_lag = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        
        self.slider_time_lag = Slider(
            self.ax_time_lag, 'Time Lag', 
            0, len(self.time_lags) - 1,
            valinit=self.current_time_lag_idx,
            valstep=1
        )
        
        # Add buttons for display options
        self.ax_display = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
        self.radio_display = RadioButtons(
            self.ax_display, ('Dr', 'Dt', 'Modulus G*'),
            active=0
        )
        
        # Add buttons for scale options
        self.ax_scale = plt.axes([0.025, 0.3, 0.15, 0.15], facecolor=axcolor)
        self.check_scale = CheckButtons(
            self.ax_scale, ['Log scale', 'Error bars'],
            [self.log_scale, self.show_errorbars]
        )
        
        # Add save button
        self.ax_save = plt.axes([0.025, 0.05, 0.15, 0.04])
        self.button_save = Button(self.ax_save, 'Save Plot', color=axcolor, hovercolor='0.975')
        
        # Connect callbacks
        self.slider_time_lag.on_changed(self.update_time_lag)
        self.radio_display.on_clicked(self.update_display)
        self.check_scale.on_clicked(self.update_scale)
        self.button_save.on_clicked(self.save_plot)
        
        # Show the plot
        plt.show()
    
    def update_time_lag(self, val):
        """
        Update the plot when time lag slider is changed.
        
        Args:
            val: New slider value
        """
        self.current_time_lag_idx = int(val)
        self.update_plot()
    
    def update_display(self, label):
        """
        Update the plot when display option is changed.
        
        Args:
            label: Selected display option
        """
        self.current_display = label
        self.update_plot()
    
    def update_scale(self, label):
        """
        Update the plot when scale options are changed.
        
        Args:
            label: Scale option that was toggled
        """
        if label == 'Log scale':
            self.log_scale = not self.log_scale
        elif label == 'Error bars':
            self.show_errorbars = not self.show_errorbars
        
        self.update_plot()
    
    def update_plot(self):
        """
        Update the plot with current settings.
        """
        # Get current time lag
        time_lag = self.time_lags[self.current_time_lag_idx]
        
        # Get binned correlation data
        binned_data = self.tpm_results['binned_correlations'][self.current_time_lag_idx]
        
        # Update data based on display option
        if self.current_display == 'Dr':
            y_data = binned_data['Dr']
            y_err = binned_data['Dr_err']
            y_label = 'Longitudinal correlation Dr (μm²)'
        elif self.current_display == 'Dt':
            y_data = binned_data['Dt']
            y_err = binned_data['Dt_err']
            y_label = 'Transverse correlation Dt (μm²)'
        else:  # Modulus G*
            # Calculate modulus from Dr
            y_data = []
            x_data = []
            
            for i, r in enumerate(binned_data['bin_centers']):
                if not np.isnan(binned_data['Dr'][i]) and binned_data['Dr'][i] > 0:
                    # Boltzmann constant (J/K)
                    k_B = 1.38064852e-23
                    
                    # Convert to SI units
                    r_si = r * 1e-6  # m
                    Dr_si = binned_data['Dr'][i] * 1e-12  # m^2
                    
                    # Calculate complex modulus
                    G = k_B * TEMPERATURE / (2 * np.pi * r_si * Dr_si)
                    
                    y_data.append(G)
                    x_data.append(r)
            
            if not y_data:
                print("No valid modulus data available")
                return
                
            y_err = None
            y_label = 'Viscoelastic modulus G* (Pa)'
        
        # Update line data
        if self.current_display == 'Modulus G*':
            self.line.set_data(x_data, y_data)
        else:
            self.line.set_data(binned_data['bin_centers'], y_data)
        
        # Update error bars
        if self.errorbar:
            self.ax.collections.clear()
            if self.show_errorbars and y_err is not None:
                if self.current_display == 'Modulus G*':
                    # No error bars for modulus
                    pass
                else:
                    self.errorbar = self.ax.errorbar(binned_data['bin_centers'], 
                                                  y_data, yerr=y_err,
                                                  fmt='none', ecolor='gray', alpha=0.5)
        
        # Update axis labels and title
        self.ax.set_ylabel(y_label)
        self.title.set_text(f'Correlation function (τ = {time_lag * DT:.2f} s)')
        
        # Update scale
        if self.log_scale:
            self.ax.set_xscale('log')
            self.ax.set_yscale('log')
        else:
            self.ax.set_xscale('linear')
            self.ax.set_yscale('linear')
        
        # Update limits
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
    
    def save_plot(self, event):
        """
        Save the current plot to a file.
        
        Args:
            event: Button click event
        """
        # Create new figure for saving (higher quality)
        save_fig, save_ax = plt.subplots(figsize=(10, 8), dpi=300)
        
        # Get current data
        time_lag = self.time_lags[self.current_time_lag_idx]
        binned_data = self.tpm_results['binned_correlations'][self.current_time_lag_idx]
        
        # Prepare data based on display option
        if self.current_display == 'Dr':
            y_data = binned_data['Dr']
            y_err = binned_data['Dr_err']
            y_label = 'Longitudinal correlation Dr (μm²)'
        elif self.current_display == 'Dt':
            y_data = binned_data['Dt']
            y_err = binned_data['Dt_err']
            y_label = 'Transverse correlation Dt (μm²)'
        else:  # Modulus G*
            # Calculate modulus from Dr
            y_data = []
            x_data = []
            
            for i, r in enumerate(binned_data['bin_centers']):
                if not np.isnan(binned_data['Dr'][i]) and binned_data['Dr'][i] > 0:
                    # Boltzmann constant (J/K)
                    k_B = 1.38064852e-23
                    
                    # Convert to SI units
                    r_si = r * 1e-6  # m
                    Dr_si = binned_data['Dr'][i] * 1e-12  # m^2
                    
                    # Calculate complex modulus
                    G = k_B * TEMPERATURE / (2 * np.pi * r_si * Dr_si)
                    
                    y_data.append(G)
                    x_data.append(r)
            
            if not y_data:
                print("No valid modulus data to save")
                return
                
            y_err = None
            y_label = 'Viscoelastic modulus G* (Pa)'
        
        # Create plot
        if self.current_display == 'Modulus G*':
            save_ax.plot(x_data, y_data, 'o-', label=self.current_display)
        else:
            save_ax.plot(binned_data['bin_centers'], y_data, 'o-', label=self.current_display)
            
            if self.show_errorbars and y_err is not None:
                save_ax.errorbar(binned_data['bin_centers'], y_data, yerr=y_err,
                              fmt='none', ecolor='gray', alpha=0.5)
        
        # Set labels and title
        save_ax.set_xlabel('Separation distance r (μm)')
        save_ax.set_ylabel(y_label)
        save_ax.set_title(f'Correlation function (τ = {time_lag * DT:.2f} s)')
        
        # Set scale
        if self.log_scale:
            save_ax.set_xscale('log')
            save_ax.set_yscale('log')
        
        # Add grid
        save_ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Add reference line for 1/r scaling if appropriate
        if self.current_display in ['Dr', 'Dt'] and self.log_scale:
            # Find first valid point for reference
            valid_idx = np.where(~np.isnan(y_data))[0]
            if len(valid_idx) > 0:
                idx = valid_idx[0]
                if self.current_display == 'Modulus G*':
                    r_ref = x_data[idx]
                    y_ref = y_data[idx]
                else:
                    r_ref = binned_data['bin_centers'][idx]
                    y_ref = y_data[idx]
                
                # Create reference line
                r_range = np.logspace(np.log10(min(binned_data['bin_centers'])), 
                                     np.log10(max(binned_data['bin_centers'])), 100)
                
                # For Dr, expect ~1/r scaling
                if self.current_display == 'Dr':
                    ref_line = r_ref * y_ref / r_range
                    save_ax.plot(r_range, ref_line, 'k--', alpha=0.5, label='1/r scaling')
                    save_ax.legend()
        
        # Tight layout
        save_fig.tight_layout()
        
        # Get save path
        save_path = f"tpm_plot_{self.current_display}_tau{time_lag}.png"
        
        # Save figure
        save_fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(save_fig)
        
        print(f"Plot saved to {save_path}")

def main():
    """Main function to visualize TPM results."""
    print("Two-Point Microrheology Visualizer")
    print("==================================")
    
    # Ask for input file
    input_file = input("Enter the path to a TPM results file (.pkl): ")
    
    if not os.path.isfile(input_file):
        print(f"File {input_file} does not exist")
        
        # Try to find TPM result files in current directory
        tpm_files = glob.glob("*/tpm_results_*.pkl")
        
        if not tpm_files:
            print("No TPM result files found in current directory or subdirectories")
            return
        
        print(f"Found the following TPM result files:")
        for i, file in enumerate(tpm_files):
            print(f"{i+1}. {file}")
        
        selection = input("Enter the number of the file to visualize: ")
        try:
            idx = int(selection) - 1
            if idx < 0 or idx >= len(tpm_files):
                print("Invalid selection")
                return
            input_file = tpm_files[idx]
        except ValueError:
            print("Invalid input")
            return
    
    # Load TPM results
    tpm_results = load_tpm_results(input_file)
    
    if tpm_results is None:
        print(f"Failed to load TPM results from {input_file}")
        return
    
    print(f"Loaded TPM results with {len(tpm_results['time_lags'])} time lags")
    
    # Create visualizer
    visualizer = TPMVisualizer(tpm_results)

if __name__ == "__main__":
    main()