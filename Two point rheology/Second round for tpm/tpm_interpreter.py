#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tpm_results_interpreter_corrected.py

This script interprets two-point microrheology results by creating
simplified explanatory plots and analysis.

What TPM measures:
1. How particles move together (correlated motion)
2. Material properties from particle interactions
3. Viscoelastic behavior of the medium

Global parameters (you can modify these)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# Global parameters - MODIFY THESE AS NEEDED
# ==========================================
PIXEL_SIZE = 0.094  # micrometers per pixel
FRAME_RATE = 10     # frames per second (1/DT)
TEMPERATURE = 298.15  # Kelvin
PARTICLE_RADIUS = 0.4  # micrometers
# ==========================================

def load_tpm_results(file_path):
    """Load TPM results from pickle file."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def explain_correlation_plots():
    """Create explanatory figure for correlation concepts."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Particle pair concept
    ax = axes[0, 0]
    
    # Draw two particles and separation vector
    particle1 = [2, 3]
    particle2 = [5, 4]
    
    # Draw particles
    circle1 = plt.Circle(particle1, 0.3, color='blue', alpha=0.7)
    circle2 = plt.Circle(particle2, 0.3, color='red', alpha=0.7)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    # Draw separation vector
    ax.arrow(particle1[0], particle1[1], 
             particle2[0]-particle1[0], particle2[1]-particle1[1],
             head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Draw displacement vectors (example)
    displacement1 = [0.5, 0.2]
    displacement2 = [0.3, 0.4]
    
    ax.arrow(particle1[0], particle1[1], displacement1[0], displacement1[1],
             head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.7)
    ax.arrow(particle2[0], particle2[1], displacement2[0], displacement2[1],
             head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
    
    ax.set_xlim(1, 6)
    ax.set_ylim(2, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Particle Pair Concept\nBlue and red arrows = displacements\nBlack arrow = separation vector')
    
    # Plot 2: Parallel vs Perpendicular components
    ax = axes[0, 1]
    
    # Show how displacements are decomposed
    separation_angle = np.arctan2(particle2[1]-particle1[1], particle2[0]-particle1[0])
    
    # Parallel component (along separation vector)
    parallel_comp = np.dot(displacement1, [np.cos(separation_angle), np.sin(separation_angle)])
    parallel_vector = parallel_comp * np.array([np.cos(separation_angle), np.sin(separation_angle)])
    
    # Perpendicular component
    perp_vector = np.array(displacement1) - parallel_vector
    
    ax.arrow(0, 0, displacement1[0], displacement1[1],
             head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='Total displacement')
    ax.arrow(0, 0, parallel_vector[0], parallel_vector[1],
             head_width=0.1, head_length=0.1, fc='green', ec='green', label='Parallel component (Dr)')
    ax.arrow(parallel_vector[0], parallel_vector[1], perp_vector[0], perp_vector[1],
             head_width=0.1, head_length=0.1, fc='orange', ec='orange', label='Perpendicular component (Dt)')
    
    ax.set_xlim(-0.2, 0.8)
    ax.set_ylim(-0.2, 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Displacement Decomposition')
    ax.legend()
    
    # Plot 3: Expected scaling behaviors
    ax = axes[1, 0]
    
    # Show theoretical scaling for different materials
    r_range = np.logspace(0, 1, 50)  # 1 to 10 micrometers
    
    # Viscous fluid: Dr ~ 1/r
    Dr_viscous = 1.0 / r_range
    
    # Elastic solid: Dr ~ 1/r²
    Dr_elastic = 1.0 / (r_range**2)
    
    # Viscoelastic: intermediate behavior
    Dr_viscoelastic = 1.0 / (r_range**1.5)
    
    ax.loglog(r_range, Dr_viscous, 'b-', label='Viscous fluid (1/r)', linewidth=2)
    ax.loglog(r_range, Dr_elastic, 'r-', label='Elastic solid (1/r²)', linewidth=2)
    ax.loglog(r_range, Dr_viscoelastic, 'g-', label='Viscoelastic (1/r^1.5)', linewidth=2)
    
    ax.set_xlabel('Separation distance r (μm)')
    ax.set_ylabel('Longitudinal correlation Dr')
    ax.set_title('Expected Scaling Behaviors\n(Different material types)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    
    # Plot 4: Time dependence interpretation
    ax = axes[1, 1]
    
    time_range = np.logspace(-2, 1, 50)  # 0.01 to 10 seconds
    
    # Diffusive: Dr ~ t
    Dr_diffusive = time_range
    
    # Subdiffusive: Dr ~ t^α (α < 1)
    Dr_subdiffusive = time_range**0.5
    
    # Superdiffusive: Dr ~ t^α (α > 1)
    Dr_superdiffusive = time_range**1.5
    
    ax.loglog(time_range, Dr_diffusive, 'b-', label='Diffusive (t¹)', linewidth=2)
    ax.loglog(time_range, Dr_subdiffusive, 'r-', label='Subdiffusive (t^0.5)', linewidth=2)
    ax.loglog(time_range, Dr_superdiffusive, 'g-', label='Superdiffusive (t^1.5)', linewidth=2)
    
    ax.set_xlabel('Time lag τ (s)')
    ax.set_ylabel('Correlation Dr')
    ax.set_title('Time Dependence Interpretation\n(Different transport mechanisms)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def interpret_tpm_results(tpm_results, output_path):
    """
    Create interpretation plots for TPM results with explanations.
    """
    if not tpm_results or not tpm_results['time_lags']:
        print("No TPM results to interpret")
        return
    
    # Create explanatory plots
    fig = explain_correlation_plots()
    plt.savefig(os.path.join(output_path, "tpm_explanation.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analysis of actual results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Correlation vs distance with interpretation
    ax = axes[0, 0]
    
    if tpm_results['binned_data']:
        # Take first time lag for analysis
        binned_data = tpm_results['binned_data'][0]
        time_lag = tpm_results['time_lags'][0]
        
        # Check what keys are in binned_data
        print("Keys in binned_data:", binned_data.keys())
        
        # Try to find the right keys for distance and correlation data
        distance_key = None
        dr_key = None
        dr_err_key = None
        
        # Look for distance-related keys
        for key in binned_data.keys():
            if 'distance' in key.lower() or 'center' in key.lower() or 'bin' in key.lower():
                distance_key = key
                break
        
        # Look for Dr-related keys
        for key in binned_data.keys():
            if 'dr' in key.lower() and 'err' not in key.lower():
                dr_key = key
                break
        
        # Look for Dr error keys
        for key in binned_data.keys():
            if 'dr' in key.lower() and 'err' in key.lower():
                dr_err_key = key
                break
        
        print(f"Found keys - Distance: {distance_key}, Dr: {dr_key}, Dr_err: {dr_err_key}")
        
        if distance_key and dr_key:
            distances = binned_data[distance_key]
            dr_values = binned_data[dr_key]
            dr_errors = binned_data[dr_err_key] if dr_err_key else None
            
            # Plot data
            if dr_errors is not None:
                ax.errorbar(distances, dr_values, yerr=dr_errors, 
                           fmt='o-', color='blue',
                           label=f'Your data (τ = {time_lag * (1/FRAME_RATE):.2f} s)')
            else:
                ax.plot(distances, dr_values, 'o-', color='blue',
                       label=f'Your data (τ = {time_lag * (1/FRAME_RATE):.2f} s)')
            
            # Add reference lines for comparison
            valid_mask = ~np.isnan(dr_values) & (dr_values > 0)
            if np.any(valid_mask):
                distances_valid = distances[valid_mask]
                dr_values_valid = dr_values[valid_mask]
                
                # Scale reference lines to match your data
                scale_factor = dr_values_valid[0] * distances_valid[0]
                
                ref_1_over_r = scale_factor / distances_valid
                ref_1_over_r2 = scale_factor / (distances_valid**2)
                
                ax.plot(distances_valid, ref_1_over_r, 'k--', alpha=0.5, label='1/r (viscous)')
                ax.plot(distances_valid, ref_1_over_r2, 'r--', alpha=0.5, label='1/r² (elastic)')
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Separation distance r (μm)')
            ax.set_ylabel('Longitudinal correlation Dr (μm²)')
            ax.set_title('Your Data vs Theoretical Predictions')
            ax.grid(True, which='both', alpha=0.3)
            ax.legend()
            
            # Add interpretation text
            interpretation = "Interpretation:\n"
            interpretation += "• If your data follows 1/r → viscous fluid\n"
            interpretation += "• If your data follows 1/r² → elastic solid\n"
            interpretation += "• In between → viscoelastic material"
            
            ax.text(0.05, 0.95, interpretation, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Modulus estimation
    ax = axes[0, 1]
    
    if tpm_results['moduli'] and len(tpm_results['moduli']) > 0:
        moduli_data = tpm_results['moduli'][0]  # First time lag
        if moduli_data.size > 0:
            if moduli_data.ndim == 2 and moduli_data.shape[1] >= 2:
                distances = moduli_data[:, 0]
                G_values = moduli_data[:, 1]
                
                ax.semilogx(distances, G_values, 'o-', color='red')
                ax.set_xlabel('Separation distance r (μm)')
                ax.set_ylabel('Viscoelastic modulus G* (Pa)')
                ax.set_title('Estimated Material Stiffness')
                ax.grid(True, alpha=0.3)
                
                # Add interpretation
                if len(G_values) > 0:
                    mean_G = np.mean(G_values)
                    interpretation = f"Material Stiffness:\n"
                    interpretation += f"G* ≈ {mean_G:.1e} Pa\n\n"
                    interpretation += "Reference values:\n"
                    interpretation += "• Water: ~1e-3 Pa\n"
                    interpretation += "• Cytoplasm: ~1-100 Pa\n"
                    interpretation += "• Gel: ~100-1000 Pa"
                    
                    ax.text(0.05, 0.95, interpretation, transform=ax.transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'Moduli data format not recognized', 
                       transform=ax.transAxes, ha='center', va='center')
    else:
        ax.text(0.5, 0.5, 'No moduli data available', 
               transform=ax.transAxes, ha='center', va='center')
    
    # Plot 3: Quality metrics
    ax = axes[1, 0]
    
    # Show data quality metrics
    if tpm_results['binned_data']:
        distances = []
        pair_counts = []
        dr_values = []
        
        for binned_data in tpm_results['binned_data']:
            # Find the right keys again
            distance_key = None
            dr_key = None
            count_key = None
            
            for key in binned_data.keys():
                if 'distance' in key.lower() or 'center' in key.lower() or 'bin' in key.lower():
                    distance_key = key
                elif 'dr' in key.lower() and 'err' not in key.lower():
                    dr_key = key
                elif 'count' in key.lower() or 'n_' in key.lower() or 'num' in key.lower():
                    count_key = key
            
            if distance_key and dr_key:
                distances.extend(binned_data[distance_key])
                dr_values.extend(binned_data[dr_key])
                
                if count_key:
                    pair_counts.extend(binned_data[count_key])
                else:
                    # If no count key, use ones
                    pair_counts.extend(np.ones(len(binned_data[distance_key])))
        
        if distances and dr_values:
            # Plot pair count vs distance
            scatter = ax.scatter(distances, pair_counts, c=dr_values, 
                               cmap='viridis', alpha=0.7, s=50)
            
            ax.set_xlabel('Separation distance (μm)')
            ax.set_ylabel('Number of particle pairs')
            ax.set_title('Data Quality: Pair Count vs Distance')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Dr value (μm²)')
            
            # Add quality interpretation
            total_pairs = sum(pair_counts)
            interpretation = f"Data Quality:\n"
            interpretation += f"Total pairs: {total_pairs:.0f}\n"
            interpretation += f"Distance range: {min(distances):.1f}-{max(distances):.1f} μm\n\n"
            interpretation += "Good data has:\n"
            interpretation += "• Many pairs (>100 total)\n"
            interpretation += "• Even distribution\n"
            interpretation += "• Consistent Dr values"
            
            ax.text(0.05, 0.95, interpretation, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Plot 4: Summary recommendations
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary text
    summary_text = "HOW TO INTERPRET YOUR RESULTS:\n\n"
    summary_text += "1. CORRELATION vs DISTANCE (top left):\n"
    summary_text += "   • Slope tells you material type\n"
    summary_text += "   • Steeper slope = more solid-like\n"
    summary_text += "   • Shallower slope = more fluid-like\n\n"
    
    summary_text += "2. VISCOELASTIC MODULUS (top right):\n"
    summary_text += "   • Higher values = stiffer material\n"
    summary_text += "   • Compare to reference values\n"
    summary_text += "   • Should be roughly constant vs distance\n\n"
    
    summary_text += "3. DATA QUALITY (bottom left):\n"
    summary_text += "   • More pairs = better statistics\n"
    summary_text += "   • Even distribution is good\n"
    summary_text += "   • Consistent colors indicate reliability\n\n"
    
    summary_text += "4. WHAT TO LOOK FOR:\n"
    summary_text += "   • Smooth curves (not noisy)\n"
    summary_text += "   • Consistent scaling behavior\n"
    summary_text += "   • Reasonable modulus values\n"
    summary_text += "   • Sufficient data points"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "tpm_interpretation.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_interpretation_report(tpm_results, output_path):
    """Create a text report explaining the results."""
    
    report_path = os.path.join(output_path, "tpm_interpretation_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("TWO-POINT MICRORHEOLOGY RESULTS INTERPRETATION\n")
        f.write("=" * 50 + "\n\n")
        
        if not tpm_results or not tpm_results['time_lags']:
            f.write("No valid TPM results found.\n")
            return
        
        f.write(f"Analysis covered {len(tpm_results['time_lags'])} time lags\n")
        f.write(f"Time range: {tpm_results['time_lags'][0] * (1/FRAME_RATE):.3f} to {tpm_results['time_lags'][-1] * (1/FRAME_RATE):.3f} seconds\n\n")
        
        # Analyze first time lag in detail
        if tpm_results['binned_data']:
            binned_data = tpm_results['binned_data'][0]
            
            f.write("CORRELATION ANALYSIS (first time lag):\n")
            f.write("-" * 30 + "\n")
            f.write(f"Available keys in binned data: {list(binned_data.keys())}\n\n")
            
            # Try to find distance and Dr data
            distance_key = None
            dr_key = None
            
            for key in binned_data.keys():
                if 'distance' in key.lower() or 'center' in key.lower() or 'bin' in key.lower():
                    distance_key = key
                elif 'dr' in key.lower() and 'err' not in key.lower():
                    dr_key = key
            
            if distance_key and dr_key:
                distances = binned_data[distance_key]
                dr_values = binned_data[dr_key]
                
                valid_mask = ~np.isnan(dr_values) & (dr_values > 0)
                if np.any(valid_mask):
                    distances_valid = distances[valid_mask]
                    dr_values_valid = dr_values[valid_mask]
                    
                    f.write(f"Distance range: {distances_valid[0]:.2f} to {distances_valid[-1]:.2f} μm\n")
                    f.write(f"Dr range: {np.min(dr_values_valid):.2e} to {np.max(dr_values_valid):.2e} μm²\n")
                    
                    # Try to fit power law
                    if len(distances_valid) >= 3:
                        log_r = np.log(distances_valid)
                        log_Dr = np.log(dr_values_valid)
                        slope, intercept = np.polyfit(log_r, log_Dr, 1)
                        
                        f.write(f"\nPower law fit: Dr ~ r^{slope:.2f}\n")
                        
                        if slope > -0.5:
                            material_type = "Very fluid-like (unusual)"
                        elif slope > -1.2:
                            material_type = "Fluid-like (viscous)"
                        elif slope > -1.8:
                            material_type = "Viscoelastic"
                        else:
                            material_type = "Solid-like (elastic)"
                        
                        f.write(f"Material interpretation: {material_type}\n")
            else:
                f.write("Could not identify distance and Dr data keys.\n")
        
        # Modulus analysis
        if tpm_results['moduli']:
            f.write("\nVISCOELASTIC MODULUS:\n")
            f.write("-" * 20 + "\n")
            
            all_moduli = []
            for moduli_data in tpm_results['moduli']:
                if moduli_data.size > 0 and moduli_data.ndim == 2 and moduli_data.shape[1] >= 2:
                    all_moduli.extend(moduli_data[:, 1])
            
            if all_moduli:
                mean_G = np.mean(all_moduli)
                std_G = np.std(all_moduli)
                
                f.write(f"Mean modulus: {mean_G:.2e} ± {std_G:.2e} Pa\n")
                
                # Compare to known materials
                if mean_G < 1e-2:
                    comparison = "Very soft (like water)"
                elif mean_G < 1:
                    comparison = "Soft fluid"
                elif mean_G < 100:
                    comparison = "Typical cytoplasm/soft gel"
                elif mean_G < 1000:
                    comparison = "Stiff gel"
                else:
                    comparison = "Very stiff material"
                
                f.write(f"Material comparison: {comparison}\n")
        
        f.write("\nINTERPRETATION GUIDE:\n")
        f.write("-" * 20 + "\n")
        f.write("1. Look at the slope in correlation vs distance plots\n")
        f.write("2. Check if modulus values are reasonable for your system\n")
        f.write("3. Ensure you have enough data points for statistics\n")
        f.write("4. Compare different time lags to see time-dependent behavior\n")
        f.write("5. Check for consistency across different measurements\n")
    
    print(f"Interpretation report saved to {report_path}")

def main():
    """Main function to interpret TPM results."""
    print("Two-Point Microrheology Results Interpreter")
    print("==========================================")
    
    # Ask for TPM results file
    input_file = input("Enter path to TPM results file (.pkl): ")
    
    if not os.path.isfile(input_file):
        print(f"File {input_file} not found")
        return
    
    # Load results
    tpm_results = load_tpm_results(input_file)
    
    if tpm_results is None:
        print("Failed to load TPM results")
        return
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(input_file), "interpretation")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating interpretation plots...")
    interpret_tpm_results(tpm_results, output_dir)
    
    print("Creating interpretation report...")
    create_interpretation_report(tpm_results, output_dir)
    
    print(f"Interpretation complete! Results saved in {output_dir}")
    print("\nKey files created:")
    print("- tpm_explanation.png: Basic concepts")
    print("- tpm_interpretation.png: Analysis of your data")
    print("- tpm_interpretation_report.txt: Detailed explanation")

if __name__ == "__main__":
    main()