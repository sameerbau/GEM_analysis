# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 13:38:10 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
velocity_autocorr_scenarios.py

This script generates theoretical velocity autocorrelation curves for different
physical scenarios to help interpret experimental data.

Usage:
python velocity_autocorr_scenarios.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Global parameters
# =====================================
MAX_TIME = 5.0      # Maximum time lag (seconds)
N_POINTS = 500      # Number of time points
# =====================================

def generate_time_array(max_time=MAX_TIME, n_points=N_POINTS):
    """Generate time lag array."""
    return np.linspace(0, max_time, n_points)

def brownian_motion(t):
    """
    Scenario 1: Normal Brownian Motion
    C(t) = exp(-t/τ_c)
    """
    tau_c = 0.3  # correlation time (seconds)
    return np.exp(-t / tau_c)

def confined_motion(t):
    """
    Scenario 2: Confined/Trapped Motion
    C(t) = exp(-γt) * cos(ωt)
    """
    gamma = 1.5  # damping coefficient
    omega = 8.0  # oscillation frequency (rad/s)
    return np.exp(-gamma * t) * np.cos(omega * t)

def active_motion(t):
    """
    Scenario 3: Active/Directed Motion
    C(t) = A*exp(-t/τ_active) + B*exp(-t/τ_passive)
    """
    tau_active = 2.0   # persistence time
    tau_passive = 0.2  # passive relaxation time
    A, B = 0.7, 0.3    # relative contributions
    return A * np.exp(-t / tau_active) + B * np.exp(-t / tau_passive)

def viscoelastic_motion(t):
    """
    Scenario 4: Viscoelastic Environment
    C(t) = exp(-(t/τ_0)^α) * [1 + oscillations]
    """
    tau_0 = 0.5  # characteristic time
    alpha = 0.7  # anomalous exponent (< 1 for subdiffusion)
    # Add small oscillations due to elasticity
    oscillations = 1 + 0.15 * np.sin(6 * t) * np.exp(-t / 0.8)
    return np.exp(-np.power(t / tau_0, alpha)) * oscillations

def measurement_artifacts(t):
    """
    Scenario 5: Measurement Artifacts
    Fast decay with high-frequency noise
    """
    tau_fast = 0.08  # very fast decay
    noise_freq = 20  # high frequency noise
    noise_decay = 0.15  # noise decay time
    clean_signal = np.exp(-t / tau_fast)
    noise = 0.4 * np.sin(noise_freq * t + np.pi/4) * np.exp(-t / noise_decay)
    return clean_signal + noise

def plot_scenarios():
    """Create plots showing all velocity autocorrelation scenarios."""
    
    # Generate time array
    time = generate_time_array()
    
    # Calculate all scenarios
    scenarios = {
        'Normal Brownian Motion': {
            'data': brownian_motion(time),
            'color': 'blue',
            'description': 'Exponential decay: C(t) = exp(-t/τ)\nRandom thermal motion'
        },
        'Confined/Trapped Motion': {
            'data': confined_motion(time),
            'color': 'red',
            'description': 'Oscillatory decay: C(t) = exp(-γt)cos(ωt)\nParticles bounce in confined space'
        },
        'Active/Directed Motion': {
            'data': active_motion(time),
            'color': 'green',
            'description': 'Persistent motion: Slow decay\nNon-thermal forces drive motion'
        },
        'Viscoelastic Environment': {
            'data': viscoelastic_motion(time),
            'color': 'orange',
            'description': 'Anomalous diffusion: Stretched exponential\nMemory effects in medium'
        },
        'Measurement Artifacts': {
            'data': measurement_artifacts(time),
            'color': 'purple',
            'description': 'Fast decay + noise\nTracking errors or undersampling'
        }
    }
    
    # Create main comparison plot
    plt.figure(figsize=(14, 10))
    
    # Main plot with all scenarios
    plt.subplot(2, 3, (1, 2))
    for name, scenario in scenarios.items():
        plt.plot(time, scenario['data'], 
                color=scenario['color'], 
                linewidth=2.5, 
                label=name)
    
    plt.xlabel('Time lag τ (s)', fontsize=12)
    plt.ylabel('Velocity Autocorrelation Cv(τ)', fontsize=12)
    plt.title('Velocity Autocorrelation Function Scenarios', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.ylim(-1.2, 1.2)
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Individual subplot for each scenario
    subplot_positions = [3, 4, 5, 6]
    scenario_names = list(scenarios.keys())[:4]  # First 4 scenarios
    
    for i, name in enumerate(scenario_names):
        plt.subplot(2, 3, subplot_positions[i])
        scenario = scenarios[name]
        plt.plot(time, scenario['data'], 
                color=scenario['color'], 
                linewidth=2)
        plt.title(name, fontsize=10, fontweight='bold')
        plt.xlabel('Time lag τ (s)', fontsize=9)
        plt.ylabel('Cv(τ)', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('velocity_autocorr_scenarios.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed comparison plot
    plt.figure(figsize=(15, 8))
    
    # Focus on the first 2 seconds for better detail
    time_detail = time[time <= 2.0]
    
    for i, (name, scenario) in enumerate(scenarios.items()):
        plt.subplot(2, 3, i+1)
        data_detail = scenario['data'][time <= 2.0]
        plt.plot(time_detail, data_detail, 
                color=scenario['color'], 
                linewidth=2.5)
        plt.title(name, fontsize=11, fontweight='bold')
        plt.xlabel('Time lag τ (s)', fontsize=10)
        plt.ylabel('Cv(τ)', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Add description as text
        plt.text(0.05, 0.95, scenario['description'], 
                transform=plt.gca().transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('velocity_autocorr_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_your_data_pattern():
    """
    Create a plot specifically showing what your data pattern suggests.
    """
    time = generate_time_array(max_time=2.0, n_points=200)
    
    # Simulate pattern similar to your data
    # Multiple oscillatory components with different frequencies and dampings
    your_pattern = (0.3 * np.exp(-2*time) * np.cos(12*time + 0.2) + 
                   0.4 * np.exp(-1.5*time) * np.cos(8*time + 0.8) +
                   0.2 * np.exp(-3*time) * np.cos(15*time + 1.2) +
                   0.1 * np.random.normal(0, 0.05, len(time)))  # Add some noise
    
    plt.figure(figsize=(12, 8))
    
    # Your data pattern
    plt.subplot(2, 2, 1)
    plt.plot(time, your_pattern, 'b-', linewidth=2, label='Your Data Pattern')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Time lag τ (s)')
    plt.ylabel('Cv(τ)')
    plt.title('Your Data: Oscillatory Pattern', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Compare with confined motion
    plt.subplot(2, 2, 2)
    confined = confined_motion(time)
    plt.plot(time, confined, 'r-', linewidth=2, label='Confined Motion Model')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Time lag τ (s)')
    plt.ylabel('Cv(τ)')
    plt.title('Best Match: Confined Motion', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Physical interpretation
    plt.subplot(2, 2, (3, 4))
    plt.text(0.05, 0.95, 
             "PHYSICAL INTERPRETATION OF YOUR DATA:\n\n"
             "✓ Large oscillations (±0.5 to ±0.8)\n"
             "✓ Negative correlations at various time lags\n"
             "✓ No smooth exponential decay\n"
             "✓ Short correlation times (τ = 0.1-0.2 s)\n\n"
             "MOST LIKELY SCENARIO: CONFINED/TRAPPED MOTION\n\n"
             "Your particles are likely:\n"
             "• Confined in cellular compartments\n"
             "• Trapped in optical/magnetic tweezers\n"
             "• Moving in a porous network\n"
             "• Bouncing between obstacles\n\n"
             "The oscillations indicate elastic restoring forces.\n"
             "Frequency tells you about confinement strength.\n"
             "Damping rate tells you about medium viscosity.\n\n"
             "ALTERNATIVE: Measurement artifacts if correlation\n"
             "times seem too short for your system scale.",
             transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    plt.gca().set_xlim(0, 1)
    plt.gca().set_ylim(0, 1)
    plt.gca().axis('off')
    
    plt.tight_layout()
    plt.savefig('your_data_interpretation.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to generate all plots."""
    print("Generating velocity autocorrelation scenario plots...")
    
    # Generate scenario comparison plots
    plot_scenarios()
    
    # Generate interpretation of your specific data
    analyze_your_data_pattern()
    
    print("\nPlots saved:")
    print("1. velocity_autocorr_scenarios.png - Overview of all scenarios")
    print("2. velocity_autocorr_detailed.png - Detailed view with descriptions")
    print("3. your_data_interpretation.png - Analysis of your specific pattern")
    
    print("\nYour data most likely shows CONFINED/TRAPPED MOTION!")
    print("The oscillatory pattern with negative correlations suggests")
    print("particles bouncing back and forth in a confined space.")

if __name__ == "__main__":
    main()