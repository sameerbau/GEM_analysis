# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 14:34:15 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
angle_autocorrelation_scenarios.py

This script generates different scenarios of angle autocorrelation functions
to help interpret experimental data. It simulates various types of particle
motion and shows their characteristic angle correlation signatures.

Output:
- Comparison plot showing different motion scenarios
- Individual diagnostic plots for each scenario
- CSV files with simulated data

Usage:
python angle_autocorrelation_scenarios.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

# Global parameters (can be modified)
# =====================================
# Time step in seconds
DT = 0.1
# Number of time points for simulation
N_TIMESTEPS = 500
# Number of particles per scenario
N_PARTICLES = 50
# Maximum time lag for correlation analysis
MAX_LAG = 50
# Random seed for reproducibility
RANDOM_SEED = 42
# =====================================

def generate_brownian_motion(n_timesteps, n_particles, diffusion_coeff=1.0):
    """
    Generate pure Brownian motion trajectories.
    
    Returns:
        trajectories: List of dictionaries with x, y coordinates
    """
    np.random.seed(RANDOM_SEED)
    trajectories = []
    
    for i in range(n_particles):
        # Random walk
        dx = np.random.normal(0, np.sqrt(2 * diffusion_coeff * DT), n_timesteps-1)
        dy = np.random.normal(0, np.sqrt(2 * diffusion_coeff * DT), n_timesteps-1)
        
        # Cumulative sum to get positions
        x = np.cumsum(np.concatenate([[0], dx]))
        y = np.cumsum(np.concatenate([[0], dy]))
        
        trajectories.append({'x': x, 'y': y, 'id': i})
    
    return trajectories

def generate_persistent_motion(n_timesteps, n_particles, persistence_time=10):
    """
    Generate trajectories with directional persistence.
    
    Args:
        persistence_time: Average time before direction change (in frames)
    """
    np.random.seed(RANDOM_SEED + 1)
    trajectories = []
    
    for i in range(n_particles):
        x = np.zeros(n_timesteps)
        y = np.zeros(n_timesteps)
        
        # Initial direction
        theta = np.random.uniform(0, 2*np.pi)
        speed = 0.5
        
        for t in range(1, n_timesteps):
            # Occasionally change direction
            if np.random.random() < 1/persistence_time:
                theta += np.random.normal(0, np.pi/4)
            
            # Add small random perturbations
            theta += np.random.normal(0, 0.1)
            
            # Move in current direction
            x[t] = x[t-1] + speed * np.cos(theta) * DT
            y[t] = y[t-1] + speed * np.sin(theta) * DT
        
        trajectories.append({'x': x, 'y': y, 'id': i})
    
    return trajectories

def generate_confined_motion(n_timesteps, n_particles, confinement_radius=5.0):
    """
    Generate trajectories confined within a circular region.
    """
    np.random.seed(RANDOM_SEED + 2)
    trajectories = []
    
    for i in range(n_particles):
        x = np.zeros(n_timesteps)
        y = np.zeros(n_timesteps)
        
        # Start at center
        x[0] = np.random.uniform(-1, 1)
        y[0] = np.random.uniform(-1, 1)
        
        for t in range(1, n_timesteps):
            # Brownian step
            dx = np.random.normal(0, 0.3)
            dy = np.random.normal(0, 0.3)
            
            new_x = x[t-1] + dx
            new_y = y[t-1] + dy
            
            # Check if outside confinement
            r = np.sqrt(new_x**2 + new_y**2)
            if r > confinement_radius:
                # Reflect back
                new_x = x[t-1] - dx
                new_y = y[t-1] - dy
            
            x[t] = new_x
            y[t] = new_y
        
        trajectories.append({'x': x, 'y': y, 'id': i})
    
    return trajectories

def generate_oscillatory_motion(n_timesteps, n_particles, frequency=0.1):
    """
    Generate trajectories with oscillatory components.
    """
    np.random.seed(RANDOM_SEED + 3)
    trajectories = []
    
    for i in range(n_particles):
        t_array = np.arange(n_timesteps) * DT
        
        # Oscillatory motion + noise
        amplitude = 2.0
        phase = np.random.uniform(0, 2*np.pi)
        
        x = amplitude * np.cos(2*np.pi*frequency*t_array + phase) + \
            np.cumsum(np.random.normal(0, 0.1, n_timesteps))
        y = amplitude * np.sin(2*np.pi*frequency*t_array + phase) + \
            np.cumsum(np.random.normal(0, 0.1, n_timesteps))
        
        trajectories.append({'x': x, 'y': y, 'id': i})
    
    return trajectories

def generate_correlated_motion(n_timesteps, n_particles, correlation_time=20):
    """
    Generate trajectories with velocity correlations (Ornstein-Uhlenbeck process).
    """
    np.random.seed(RANDOM_SEED + 4)
    trajectories = []
    
    for i in range(n_particles):
        x = np.zeros(n_timesteps)
        y = np.zeros(n_timesteps)
        
        # Ornstein-Uhlenbeck process for velocity
        vx = 0
        vy = 0
        gamma = 1/correlation_time  # Friction coefficient
        
        for t in range(1, n_timesteps):
            # Update velocities with friction and noise
            vx = vx * (1 - gamma*DT) + np.random.normal(0, np.sqrt(2*gamma*DT))
            vy = vy * (1 - gamma*DT) + np.random.normal(0, np.sqrt(2*gamma*DT))
            
            # Update positions
            x[t] = x[t-1] + vx * DT
            y[t] = y[t-1] + vy * DT
        
        trajectories.append({'x': x, 'y': y, 'id': i})
    
    return trajectories

def calculate_angle_autocorrelation(trajectories, max_lag=MAX_LAG):
    """
    Calculate angle autocorrelation function for trajectories.
    """
    cos_angle_t_total = [[] for _ in range(max_lag)]
    
    for traj in trajectories:
        x_temp = traj['x']
        y_temp = traj['y']
        
        # Calculate displacement vectors
        delta_x = np.diff(x_temp)
        delta_y = np.diff(y_temp)
        
        # Calculate angle correlations for different time lags
        for i in range(1, max_lag + 1):
            if len(delta_x) <= i:
                continue
                
            for k in range(len(delta_x) - i):
                # Calculate dot product of displacement vectors
                dot_product = delta_x[k] * delta_x[k+i] + delta_y[k] * delta_y[k+i]
                # Calculate magnitudes
                mag1 = np.sqrt(delta_x[k]**2 + delta_y[k]**2)
                mag2 = np.sqrt(delta_x[k+i]**2 + delta_y[k+i]**2)
                
                # Calculate cosine of angle between vectors
                if mag1 > 0 and mag2 > 0:
                    cos_angle = dot_product / (mag1 * mag2)
                    cos_angle = max(min(cos_angle, 1.0), -1.0)
                    cos_angle_t_total[i-1].append(cos_angle)
    
    # Calculate average and standard error
    mean_cos_angle = np.zeros(max_lag)
    sem_cos_angle = np.zeros(max_lag)
    
    for i in range(max_lag):
        temp = np.array(cos_angle_t_total[i])
        temp = temp[~np.isnan(temp)]
        
        if len(temp) > 0:
            mean_cos_angle[i] = np.mean(temp)
            sem_cos_angle[i] = np.std(temp) / np.sqrt(len(temp))
        else:
            mean_cos_angle[i] = np.nan
            sem_cos_angle[i] = np.nan
    
    # Calculate crossing time
    t_cross = np.nan
    index_temp = np.where(mean_cos_angle < 0)[0]
    
    if len(index_temp) > 0:
        x2 = index_temp[0] + 1
        if x2 >= 2:
            x1 = x2 - 1
            y2 = mean_cos_angle[x2-1]
            y1 = mean_cos_angle[x1-1]
            if y2 != y1:
                t_cross = (x1*y2 - x2*y1)*DT / (y2-y1)
    
    return {
        'mean_cos_angle': mean_cos_angle,
        'sem_cos_angle': sem_cos_angle,
        't_cross': t_cross,
        'time_lags': np.arange(1, max_lag + 1) * DT
    }

def plot_trajectories(scenarios_data, output_dir):
    """
    Plot sample trajectories for each scenario.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    scenario_names = list(scenarios_data.keys())
    
    for i, (name, data) in enumerate(scenarios_data.items()):
        if i >= 6:  # Only plot first 6 scenarios
            break
            
        ax = axes[i]
        trajectories = data['trajectories']
        
        # Plot first 5 trajectories
        for j, traj in enumerate(trajectories[:5]):
            ax.plot(traj['x'], traj['y'], '-', alpha=0.7, linewidth=1)
            ax.plot(traj['x'][0], traj['y'][0], 'o', markersize=4)  # Start point
        
        ax.set_title(name)
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for i in range(len(scenario_names), 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_scenarios.png'), dpi=300)
    plt.close()

def plot_angle_correlations(scenarios_data, output_dir):
    """
    Plot angle autocorrelation functions for all scenarios.
    """
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios_data)))
    
    for i, (name, data) in enumerate(scenarios_data.items()):
        results = data['correlation_results']
        
        plt.errorbar(
            results['time_lags'],
            results['mean_cos_angle'],
            yerr=results['sem_cos_angle'],
            fmt='.-',
            color=colors[i],
            label=f"{name} (t_cross = {results['t_cross']:.3f} s)" if not np.isnan(results['t_cross']) else f"{name} (no crossing)",
            alpha=0.8
        )
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Time lag τ (s)', fontsize=12)
    plt.ylabel(r'$\langle \cos \theta(\tau) \rangle$', fontsize=12)
    plt.title('Angle Autocorrelation Functions - Different Motion Scenarios', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'angle_correlation_scenarios.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_interpretation_guide(output_dir):
    """
    Create a text file explaining how to interpret different patterns.
    """
    guide_text = """
ANGLE AUTOCORRELATION INTERPRETATION GUIDE
==========================================

1. PURE BROWNIAN MOTION
   - Rapid exponential decay to zero
   - Crossing time: ~0.1-0.5 seconds
   - No oscillations
   - Interpretation: Completely random motion, no directional memory

2. PERSISTENT MOTION  
   - Slower decay, higher crossing time
   - Crossing time: 1-5 seconds
   - Smooth decay curve
   - Interpretation: Particles maintain direction (active motion, swimming)

3. CONFINED MOTION
   - May show oscillations around zero
   - Complex crossing pattern
   - Interpretation: Particles bounce off boundaries, creating correlations

4. OSCILLATORY MOTION
   - Clear oscillations in correlation function
   - Multiple zero crossings
   - Interpretation: Periodic or quasi-periodic motion

5. CORRELATED MOTION (Ornstein-Uhlenbeck)
   - Exponential decay with specific time constant
   - Single crossing time
   - Interpretation: Velocity correlations, viscoelastic medium

EXPERIMENTAL DATA INTERPRETATION:
- Fast decay (< 0.1s): Thermal motion dominates
- Slow decay (> 1s): Active or guided motion
- Oscillations: Confinement or external forces
- No crossing: Highly persistent or ballistic motion
- Multiple crossings: Complex environment or multiple populations

NOISE CHARACTERIZATION METHODS:
- Compare experimental crossing times with theoretical values
- Fit exponential decay: C(τ) = exp(-τ/τ_p) where τ_p is persistence time
- Use complementary techniques: MSD analysis, velocity autocorrelation
- Apply Kramers-Kronig analysis for frequency domain insights
"""
    
    with open(os.path.join(output_dir, 'interpretation_guide.txt'), 'w') as f:
        f.write(guide_text)

def main():
    """
    Generate different angle autocorrelation scenarios.
    """
    print("Generating angle autocorrelation scenarios...")
    
    # Create output directory
    output_dir = "angle_correlation_scenarios"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate different motion scenarios
    scenarios = {
        'Pure Brownian': generate_brownian_motion(N_TIMESTEPS, N_PARTICLES),
        'Persistent Motion': generate_persistent_motion(N_TIMESTEPS, N_PARTICLES, persistence_time=15),
        'Confined Motion': generate_confined_motion(N_TIMESTEPS, N_PARTICLES),
        'Oscillatory Motion': generate_oscillatory_motion(N_TIMESTEPS, N_PARTICLES),
        'Correlated Motion': generate_correlated_motion(N_TIMESTEPS, N_PARTICLES),
    }
    
    # Calculate angle autocorrelations
    scenarios_data = {}
    
    for name, trajectories in scenarios.items():
        print(f"Calculating correlations for {name}...")
        correlation_results = calculate_angle_autocorrelation(trajectories)
        
        scenarios_data[name] = {
            'trajectories': trajectories,
            'correlation_results': correlation_results
        }
    
    # Create plots
    print("Creating trajectory plots...")
    plot_trajectories(scenarios_data, output_dir)
    
    print("Creating correlation plots...")
    plot_angle_correlations(scenarios_data, output_dir)
    
    # Export data to CSV
    print("Exporting data...")
    for name, data in scenarios_data.items():
        results = data['correlation_results']
        df = pd.DataFrame({
            'TimeLag_s': results['time_lags'],
            'MeanCosAngle': results['mean_cos_angle'],
            'SEM': results['sem_cos_angle']
        })
        
        filename = name.lower().replace(' ', '_')
        df.to_csv(os.path.join(output_dir, f'{filename}_correlation.csv'), index=False)
    
    # Create summary
    summary_data = []
    for name, data in scenarios_data.items():
        results = data['correlation_results']
        summary_data.append({
            'Scenario': name,
            'CrossingTime_s': results['t_cross'],
            'InitialCorrelation': results['mean_cos_angle'][0],
            'DecayRate': -np.log(results['mean_cos_angle'][4]/results['mean_cos_angle'][0])/results['time_lags'][4] if results['mean_cos_angle'][0] > 0 else np.nan
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'scenario_summary.csv'), index=False)
    
    # Create interpretation guide
    create_interpretation_guide(output_dir)
    
    print(f"\nScenario generation complete!")
    print(f"Results saved in: {output_dir}")
    print("\nScenario Summary:")
    print(summary_df.to_string(index=False))
    
    print(f"\nCheck the interpretation_guide.txt for detailed explanations.")

if __name__ == "__main__":
    main()