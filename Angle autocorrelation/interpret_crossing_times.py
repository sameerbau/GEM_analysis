# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 15:14:04 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interpret_crossing_times.py

Simple tool to help interpret angle autocorrelation crossing times,
especially when they're close to the recording frequency.

Usage:
python interpret_crossing_times.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Your experimental parameters
DT = 0.1  # Recording frequency (seconds)

def interpret_crossing_times():
    """
    Provide interpretation of crossing time results.
    """
    print("Angle Autocorrelation Crossing Time Interpretation")
    print("=" * 50)
    print()
    
    # Get your results
    print("Enter your results:")
    crossing_time_1 = float(input("Crossing time for condition 1 (e.g., Preblastoderm): "))
    crossing_time_2 = float(input("Crossing time for condition 2 (e.g., Rab1): "))
    condition_1 = input("Name of condition 1: ")
    condition_2 = input("Name of condition 2: ")
    
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS")
    print("=" * 50)
    
    # Calculate relative difference
    difference = abs(crossing_time_1 - crossing_time_2)
    relative_diff = difference / DT
    
    print(f"\nYour Results:")
    print(f"  {condition_1}: {crossing_time_1:.3f} s")
    print(f"  {condition_2}: {crossing_time_2:.3f} s")
    print(f"  Difference: {difference:.3f} s")
    print(f"  Recording frequency: {DT} s")
    print(f"  Difference as fraction of recording interval: {relative_diff:.2f}")
    
    print("\n" + "-" * 30)
    print("INTERPRETATION")
    print("-" * 30)
    
    # Interpretation based on relationship to recording frequency
    if crossing_time_1 <= DT and crossing_time_2 <= DT:
        print("\n🔍 TEMPORAL RESOLUTION LIMITATION DETECTED")
        print("Both crossing times are ≤ recording frequency")
        print("This means particles lose directional memory within 1-2 frames")
        print()
        print("What this tells us:")
        print("• Both conditions show very rapid directional decorrelation")
        print("• Particles change direction almost immediately (highly random walk)")
        print("• The difference, while statistically significant, may be at the")
        print("  limit of what your temporal resolution can reliably measure")
        print()
        print("Biological interpretation:")
        print("• Both systems show highly stochastic motion")
        print("• Very little directional persistence")
        print("• The difference suggests one condition is slightly more random")
        
        if crossing_time_1 < crossing_time_2:
            faster_condition = condition_1
            slower_condition = condition_2
        else:
            faster_condition = condition_2
            slower_condition = condition_1
        
        print(f"• {faster_condition} loses directionality faster than {slower_condition}")
        
    elif crossing_time_1 > 2*DT or crossing_time_2 > 2*DT:
        print("\n✅ RELIABLE MEASUREMENT")
        print("At least one crossing time is > 2× recording frequency")
        print("This suggests reliable measurement of directional persistence")
        
    else:
        print("\n⚠️  BORDERLINE MEASUREMENT")
        print("Crossing times are 1-2× recording frequency")
        print("Results should be interpreted with caution")
    
    print("\n" + "-" * 30)
    print("RECOMMENDATIONS")
    print("-" * 30)
    
    print("\n1. ADDITIONAL ANALYSES TO PERFORM:")
    print("   • Mean turning angles between steps")
    print("   • Fraction of steps with small turning angles (<30°)")
    print("   • Trajectory straightness (end-to-end/path length)")
    print("   • Step size distributions")
    
    print("\n2. EXPERIMENTAL CONSIDERATIONS:")
    if difference < 0.05:
        print("   • Consider higher recording frequency if possible")
        print("   • Focus on ensemble properties rather than individual crossing times")
    
    print("   • Validate with longer trajectories if available")
    print("   • Consider analyzing at multiple time scales")
    
    print("\n3. STATISTICAL CONSIDERATIONS:")
    print("   • Large effect size (Cliff's delta ≈ 1.0) suggests real difference")
    print("   • But magnitude may be limited by temporal resolution")
    print("   • Bootstrap confidence intervals help assess reliability")
    
    print("\n" + "=" * 50)
    print("WHAT YOUR RESULTS LIKELY MEAN")
    print("=" * 50)
    
    print(f"\nBoth {condition_1} and {condition_2} show:")
    print("• Highly stochastic motion with little directional persistence")
    print("• Rapid loss of directional memory (within 1-2 frames)")
    print("• Behavior consistent with random walk or near-random walk")
    
    print(f"\nThe difference between conditions suggests:")
    if crossing_time_1 < crossing_time_2:
        print(f"• {condition_1} has slightly MORE stochastic motion")
        print(f"• {condition_2} retains direction slightly longer")
    else:
        print(f"• {condition_2} has slightly MORE stochastic motion")
        print(f"• {condition_1} retains direction slightly longer")
    
    print("\nHowever, the biological significance of this difference")
    print("should be confirmed with additional analyses due to")
    print("temporal resolution limitations.")
    
    # Create a simple visualization
    create_interpretation_plot(crossing_time_1, crossing_time_2, condition_1, condition_2)

def create_interpretation_plot(t1, t2, name1, name2):
    """
    Create a simple plot to visualize the crossing times relative to recording frequency.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Crossing times relative to recording frequency
    conditions = [name1, name2]
    times = [t1, t2]
    colors = ['blue', 'red']
    
    bars = ax1.bar(conditions, times, color=colors, alpha=0.7)
    ax1.axhline(DT, color='black', linestyle='--', linewidth=2, 
                label=f'Recording frequency ({DT} s)')
    ax1.axhline(2*DT, color='gray', linestyle=':', linewidth=1, 
                label=f'2× Recording frequency ({2*DT} s)')
    
    ax1.set_ylabel('Crossing time (s)')
    ax1.set_title('Crossing Times vs Recording Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{time:.3f}s', ha='center', va='bottom')
    
    # Plot 2: Conceptual illustration
    time_points = np.linspace(0, 0.5, 100)
    
    # Simulate what the autocorrelation might look like
    autocorr1 = np.exp(-time_points / t1)
    autocorr2 = np.exp(-time_points / t2)
    
    ax2.plot(time_points, autocorr1, 'b-', linewidth=2, label=f'{name1} (τ = {t1:.3f}s)')
    ax2.plot(time_points, autocorr2, 'r-', linewidth=2, label=f'{name2} (τ = {t2:.3f}s)')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(DT, color='gray', linestyle=':', alpha=0.5, label=f'Recording interval')
    
    ax2.set_xlabel('Time lag (s)')
    ax2.set_ylabel('Directional correlation')
    ax2.set_title('Conceptual Autocorrelation Decay')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('crossing_time_interpretation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'crossing_time_interpretation.png'")

if __name__ == "__main__":
    interpret_crossing_times()