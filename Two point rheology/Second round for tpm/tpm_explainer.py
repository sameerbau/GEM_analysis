# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 13:14:24 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tpm_code_explainer.py

This script explains how the original two-point microrheology code works
by breaking it down into understandable components.

Run this to understand the structure and flow of the TPM analysis.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def explain_tpm_workflow():
    """
    Create a visual explanation of the TPM workflow.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Step 1: Load trajectories
    ax = axes[0, 0]
    ax.set_title("STEP 1: Load Trajectory Data", fontweight='bold', fontsize=12)
    
    # Simulate some trajectory data
    t = np.linspace(0, 10, 100)
    x1 = 2 + 0.5 * np.cumsum(np.random.randn(100) * 0.1)
    y1 = 3 + 0.5 * np.cumsum(np.random.randn(100) * 0.1)
    x2 = 5 + 0.5 * np.cumsum(np.random.randn(100) * 0.1)
    y2 = 4 + 0.5 * np.cumsum(np.random.randn(100) * 0.1)
    
    ax.plot(x1, y1, 'b-', alpha=0.7, label='Particle 1')
    ax.plot(x2, y2, 'r-', alpha=0.7, label='Particle 2')
    ax.plot(x1[0], y1[0], 'bo', markersize=8)
    ax.plot(x2[0], y2[0], 'ro', markersize=8)
    ax.set_xlabel('X position (μm)')
    ax.set_ylabel('Y position (μm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    code_text = """
Key function: load_processed_data()
What it does:
• Loads .pkl files with trajectories
• Each trajectory has x, y positions
• Multiple particles tracked over time
"""
    ax.text(0.02, 0.02, code_text, transform=ax.transAxes, 
           fontsize=8, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Step 2: Organize by frames
    ax = axes[0, 1]
    ax.set_title("STEP 2: Organize Data by Frame", fontweight='bold', fontsize=12)
    
    # Show frame-by-frame organization
    frame_example = np.array([[2, 3], [5, 4], [3, 6], [7, 2]])
    ax.scatter(frame_example[:, 0], frame_example[:, 1], s=100, c=['blue', 'red', 'green', 'orange'])
    
    for i, (x, y) in enumerate(frame_example):
        ax.annotate(f'ID: {i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('X position (μm)')
    ax.set_ylabel('Y position (μm)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    code_text = """
Key function: prepare_trajectories_for_tpm()
What it does:
• Groups particles by frame number
• Creates frame_data dictionary
• Makes pair analysis efficient
"""
    ax.text(0.02, 0.02, code_text, transform=ax.transAxes, 
           fontsize=8, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Step 3: Find particle pairs
    ax = axes[0, 2]
    ax.set_title("STEP 3: Find Particle Pairs", fontweight='bold', fontsize=12)
    
    # Show particle pairs with separation distances
    particles = np.array([[2, 3], [5, 4], [3, 6], [7, 2]])
    ax.scatter(particles[:, 0], particles[:, 1], s=100, c=['blue', 'red', 'green', 'orange'])
    
    # Draw lines between particles within distance range
    for i in range(len(particles)):
        for j in range(i+1, len(particles)):
            dist = np.linalg.norm(particles[i] - particles[j])
            if 2 <= dist <= 6:  # Example distance range
                ax.plot([particles[i, 0], particles[j, 0]], 
                       [particles[i, 1], particles[j, 1]], 'k-', alpha=0.5)
                mid_point = (particles[i] + particles[j]) / 2
                ax.text(mid_point[0], mid_point[1], f'{dist:.1f}μm', 
                       ha='center', va='center', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X position (μm)')
    ax.set_ylabel('Y position (μm)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    code_text = """
Key function: calculate_separation_distances()
What it does:
• Finds all particle pairs
• Calculates separation distances
• Filters by distance range
"""
    ax.text(0.02, 0.02, code_text, transform=ax.transAxes, 
           fontsize=8, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # Step 4: Calculate correlations
    ax = axes[1, 0]
    ax.set_title("STEP 4: Calculate Displacement Correlations", fontweight='bold', fontsize=12)
    
    # Show displacement vectors and their correlations
    p1 = np.array([2, 3])
    p2 = np.array([5, 4])
    
    # Original positions
    ax.scatter(*p1, s=100, c='blue', label='Particle 1 (t=0)')
    ax.scatter(*p2, s=100, c='red', label='Particle 2 (t=0)')
    
    # Displaced positions
    disp1 = np.array([0.3, 0.2])
    disp2 = np.array([0.1, 0.4])
    p1_new = p1 + disp1
    p2_new = p2 + disp2
    
    ax.scatter(*p1_new, s=100, c='blue', marker='s', label='Particle 1 (t=τ)')
    ax.scatter(*p2_new, s=100, c='red', marker='s', label='Particle 2 (t=τ)')
    
    # Draw displacement vectors
    ax.arrow(p1[0], p1[1], disp1[0], disp1[1], head_width=0.1, head_length=0.1, 
             fc='blue', ec='blue', alpha=0.7)
    ax.arrow(p2[0], p2[1], disp2[0], disp2[1], head_width=0.1, head_length=0.1, 
             fc='red', ec='red', alpha=0.7)
    
    # Draw separation vector
    sep_vec = p2 - p1
    ax.arrow(p1[0], p1[1], sep_vec[0], sep_vec[1], head_width=0.1, head_length=0.1, 
             fc='black', ec='black', alpha=0.5, linestyle='--')
    
    ax.set_xlabel('X position (μm)')
    ax.set_ylabel('Y position (μm)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    code_text = """
Key function: track_particle_pairs()
What it does:
• Tracks same particles across time
• Calculates displacement vectors
• Computes parallel/perpendicular correlations
"""
    ax.text(0.02, 0.02, code_text, transform=ax.transAxes, 
           fontsize=8, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Step 5: Bin by distance
    ax = axes[1, 1]
    ax.set_title("STEP 5: Bin Correlations by Distance", fontweight='bold', fontsize=12)
    
    # Show binning concept
    distances = np.array([2.1, 2.8, 3.2, 4.1, 4.7, 5.3, 5.9])
    correlations = np.array([0.8, 0.6, 0.7, 0.4, 0.3, 0.2, 0.15])
    
    # Create bins
    bin_edges = np.array([2, 3, 4, 5, 6])
    bin_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    
    for i in range(len(bin_edges)-1):
        in_bin = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
        if np.any(in_bin):
            ax.scatter(distances[in_bin], correlations[in_bin], 
                      c=bin_colors[i], s=100, alpha=0.7, 
                      label=f'Bin {i+1}: {bin_edges[i]}-{bin_edges[i+1]}μm')
            
            # Show bin average
            bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
            bin_avg = np.mean(correlations[in_bin])
            ax.plot(bin_center, bin_avg, 'ko', markersize=10)
    
    ax.set_xlabel('Separation distance (μm)')
    ax.set_ylabel('Correlation value')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    code_text = """
Key function: bin_correlation_by_distance()
What it does:
• Groups pairs by distance
• Averages correlations in each bin
• Calculates error bars
"""
    ax.text(0.02, 0.02, code_text, transform=ax.transAxes, 
           fontsize=8, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    # Step 6: Extract material properties
    ax = axes[1, 2]
    ax.set_title("STEP 6: Extract Material Properties", fontweight='bold', fontsize=12)
    
    # Show correlation vs distance and resulting modulus
    r_range = np.linspace(2, 6, 20)
    Dr_example = 0.5 / r_range  # 1/r scaling example
    
    ax.loglog(r_range, Dr_example, 'bo-', label='Dr correlation data')
    
    # Show different scaling behaviors
    Dr_viscous = 0.5 / r_range
    Dr_elastic = 0.5 / (r_range**2)
    
    ax.loglog(r_range, Dr_viscous, 'g--', alpha=0.7, label='Viscous (1/r)')
    ax.loglog(r_range, Dr_elastic, 'r--', alpha=0.7, label='Elastic (1/r²)')
    
    ax.set_xlabel('Separation distance r (μm)')
    ax.set_ylabel('Longitudinal correlation Dr')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    code_text = """
Key function: calculate_viscoelastic_modulus()
What it does:
• Uses Stokes-Einstein relation
• G* = kT / (2πr * Dr)
• Extracts material stiffness
"""
    ax.text(0.02, 0.02, code_text, transform=ax.transAxes, 
           fontsize=8, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7))
    
    plt.tight_layout()
    return fig

def explain_key_concepts():
    """
    Explain the key physics concepts behind TPM.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Concept 1: Why particle pairs?
    ax = axes[0, 0]
    ax.set_title("WHY ANALYZE PARTICLE PAIRS?", fontweight='bold', fontsize=12)
    
    # Show individual vs correlated motion
    t = np.linspace(0, 5, 50)
    
    # Independent motion (viscous fluid)
    x1_indep = np.cumsum(np.random.randn(50) * 0.1)
    x2_indep = np.cumsum(np.random.randn(50) * 0.1)
    
    # Correlated motion (elastic medium)
    noise = np.random.randn(50) * 0.1
    x1_corr = np.cumsum(noise + np.random.randn(50) * 0.05)
    x2_corr = np.cumsum(noise + np.random.randn(50) * 0.05)
    
    ax.plot(t, x1_indep, 'b-', label='Particle 1 (independent)', alpha=0.7)
    ax.plot(t, x2_indep, 'r-', label='Particle 2 (independent)', alpha=0.7)
    ax.plot(t, x1_corr + 3, 'b-', linewidth=2, label='Particle 1 (correlated)')
    ax.plot(t, x2_corr + 3, 'r-', linewidth=2, label='Particle 2 (correlated)')
    
    ax.axhline(y=1.5, color='k', linestyle='--', alpha=0.5)
    ax.text(2.5, 1.5, 'Viscous fluid\n(independent motion)', ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(2.5, 4.5, 'Elastic medium\n(correlated motion)', ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Concept 2: Longitudinal vs Transverse
    ax = axes[0, 1]
    ax.set_title("LONGITUDINAL vs TRANSVERSE", fontweight='bold', fontsize=12)
    
    # Draw two particles and their displacement components
    p1 = np.array([1, 2])
    p2 = np.array([4, 3])
    
    # Displacement vectors
    disp1 = np.array([0.5, 0.3])
    disp2 = np.array([0.2, 0.4])
    
    # Separation vector
    sep_vec = p2 - p1
    sep_unit = sep_vec / np.linalg.norm(sep_vec)
    
    # Project displacements
    proj1_parallel = np.dot(disp1, sep_unit) * sep_unit
    proj2_parallel = np.dot(disp2, sep_unit) * sep_unit
    
    proj1_perp = disp1 - proj1_parallel
    proj2_perp = disp2 - proj2_parallel
    
    # Draw particles
    ax.scatter(*p1, s=150, c='blue', alpha=0.7)
    ax.scatter(*p2, s=150, c='red', alpha=0.7)
    
    # Draw separation vector
    ax.arrow(p1[0], p1[1], sep_vec[0], sep_vec[1], head_width=0.1, head_length=0.1,
             fc='black', ec='black', alpha=0.5)
    ax.text((p1[0]+p2[0])/2, (p1[1]+p2[1])/2 + 0.2, 'separation', ha='center')
    
    # Draw parallel components
    ax.arrow(p1[0], p1[1], proj1_parallel[0], proj1_parallel[1], 
             head_width=0.08, head_length=0.08, fc='green', ec='green', linewidth=2)
    ax.arrow(p2[0], p2[1], proj2_parallel[0], proj2_parallel[1], 
             head_width=0.08, head_length=0.08, fc='green', ec='green', linewidth=2)
    
    # Draw perpendicular components
    ax.arrow(p1[0] + proj1_parallel[0], p1[1] + proj1_parallel[1], 
             proj1_perp[0], proj1_perp[1], 
             head_width=0.08, head_length=0.08, fc='orange', ec='orange', linewidth=2)
    ax.arrow(p2[0] + proj2_parallel[0], p2[1] + proj2_parallel[1], 
             proj2_perp[0], proj2_perp[1], 
             head_width=0.08, head_length=0.08, fc='orange', ec='orange', linewidth=2)
    
    ax.text(0.5, 4.5, 'Longitudinal (Dr):\nMotion along separation\n→ Compression/extension', 
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(0.5, 0.5, 'Transverse (Dt):\nMotion perpendicular\n→ Shear deformation', 
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Concept 3: Material scaling laws
    ax = axes[1, 0]
    ax.set_title("MATERIAL SCALING LAWS", fontweight='bold', fontsize=12)
    
    r_range = np.logspace(0, 1, 50)
    
    # Different material behaviors
    Dr_viscous = 1.0 / r_range
    Dr_elastic = 1.0 / (r_range**2)
    Dr_viscoelastic = 1.0 / (r_range**1.5)
    
    ax.loglog(r_range, Dr_viscous, 'b-', linewidth=3, label='Viscous fluid (1/r)')
    ax.loglog(r_range, Dr_elastic, 'r-', linewidth=3, label='Elastic solid (1/r²)')
    ax.loglog(r_range, Dr_viscoelastic, 'g-', linewidth=3, label='Viscoelastic (1/r^1.5)')
    
    ax.set_xlabel('Separation distance r (μm)')
    ax.set_ylabel('Longitudinal correlation Dr')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.annotate('Steeper slope\n= More solid-like', xy=(3, 0.1), xytext=(5, 0.3),
               arrowprops=dict(arrowstyle='->', color='red'),
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Concept 4: What the modulus tells us
    ax = axes[1, 1]
    ax.set_title("VISCOELASTIC MODULUS INTERPRETATION", fontweight='bold', fontsize=12)
    
    # Show modulus ranges for different materials
    materials = ['Water', 'Honey', 'Cytoplasm', 'Soft gel', 'Stiff gel', 'Rubber']
    moduli = [1e-3, 1e-1, 10, 100, 1000, 1e6]
    colors = ['lightblue', 'gold', 'lightgreen', 'orange', 'red', 'purple']
    
    bars = ax.barh(range(len(materials)), moduli, color=colors, alpha=0.7)
    ax.set_yticks(range(len(materials)))
    ax.set_yticklabels(materials)
    ax.set_xlabel('Viscoelastic modulus G* (Pa)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, modulus) in enumerate(zip(bars, moduli)):
        ax.text(modulus * 2, i, f'{modulus:.0e} Pa', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_code_flow_diagram():
    """
    Create a flowchart showing the code structure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_title("TPM CODE FLOW DIAGRAM", fontweight='bold', fontsize=16)
    
    # Define boxes and their positions
    boxes = [
        ("load_processed_data()", (2, 9), "Load trajectory\n.pkl files"),
        ("prepare_trajectories_for_tpm()", (2, 7.5), "Organize data\nby frames"),
        ("track_particle_pairs()", (2, 6), "Find pairs &\ncalculate correlations"),
        ("bin_correlation_by_distance()", (2, 4.5), "Group by distance\n& average"),
        ("calculate_viscoelastic_modulus()", (2, 3), "Extract material\nproperties"),
        ("create_diagnostic_plot()", (6, 6), "Visualization"),
        ("plot_correlation_vs_distance()", (6, 4.5), "Main results"),
        ("export_tpm_results()", (6, 3), "Save to CSV"),
    ]
    
    # Draw boxes
    for func_name, (x, y), description in boxes:
        # Function box
        rect = plt.Rectangle((x-0.8, y-0.3), 1.6, 0.6, 
                           facecolor='lightblue', edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y, func_name, ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Description box
        rect_desc = plt.Rectangle((x-0.8, y-0.8), 1.6, 0.4, 
                                facecolor='lightyellow', edgecolor='gray', linewidth=1)
        ax.add_patch(rect_desc)
        ax.text(x, y-0.6, description, ha='center', va='center', fontsize=8)
    
    # Draw arrows showing flow
    arrows = [
        ((2, 8.7), (2, 8.1)),  # load -> prepare
        ((2, 7.2), (2, 6.6)),  # prepare -> track
        ((2, 5.7), (2, 5.1)),  # track -> bin
        ((2, 4.2), (2, 3.6)),  # bin -> modulus
        ((2.8, 6), (5.2, 6)),  # track -> diagnostic
        ((2.8, 4.5), (5.2, 4.5)),  # bin -> plot
        ((2.8, 3), (5.2, 3)),  # modulus -> export
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # Add main loop annotation
    loop_rect = plt.Rectangle((0.5, 2.5), 3, 4.5, 
                            facecolor='none', edgecolor='blue', 
                            linewidth=2, linestyle='--')
    ax.add_patch(loop_rect)
    ax.text(0.2, 5, 'MAIN\nANALYSIS\nLOOP', rotation=90, ha='center', va='center',
           fontweight='bold', color='blue', fontsize=10)
    
    # Add output section annotation
    output_rect = plt.Rectangle((4.5, 2.5), 3, 4, 
                              facecolor='none', edgecolor='green', 
                              linewidth=2, linestyle='--')
    ax.add_patch(output_rect)
    ax.text(7.8, 4.5, 'OUTPUT\n&\nVISUALIZATION', rotation=90, ha='center', va='center',
           fontweight='bold', color='green', fontsize=10)
    
    ax.set_xlim(0, 8)
    ax.set_ylim(2, 10)
    ax.axis('off')
    
    return fig

def main():
    """Main function to create all explanatory materials."""
    print("Creating TPM Code Explanation Materials...")
    
    # Create output directory
    output_dir = "tpm_explanation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create workflow explanation
    print("1. Creating workflow explanation...")
    fig1 = explain_tpm_workflow()
    plt.savefig(os.path.join(output_dir, "tpm_workflow_explanation.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create physics concepts explanation
    print("2. Creating physics concepts explanation...")
    fig2 = explain_key_concepts()
    plt.savefig(os.path.join(output_dir, "tpm_physics_concepts.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create code flow diagram
    print("3. Creating code flow diagram...")
    fig3 = create_code_flow_diagram()
    plt.savefig(os.path.join(output_dir, "tpm_code_flow.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create text explanation
    print("4. Creating text explanation...")
    with open(os.path.join(output_dir, "tpm_explanation.txt"), 'w') as f:
        f.write("TWO-POINT MICRORHEOLOGY (TPM) CODE EXPLANATION\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("WHAT TPM DOES:\n")
        f.write("-" * 15 + "\n")
        f.write("TPM analyzes how pairs of particles move together to extract\n")
        f.write("properties of the material they're embedded in (like cytoplasm).\n\n")
        
        f.write("KEY PHYSICS CONCEPTS:\n")
        f.write("-" * 20 + "\n")
        f.write("1. PARTICLE PAIRS: Instead of looking at single particles,\n")
        f.write("   TPM looks at how particle pairs move relative to each other.\n\n")
        
        f.write("2. CORRELATIONS: If particles move independently → fluid\n")
        f.write("   If particles move together → more solid-like material\n\n")
        
        f.write("3. DISTANCE DEPENDENCE: How correlation changes with distance\n")
        f.write("   tells you about material properties:\n")
        f.write("   • 1/r scaling → viscous fluid\n")
        f.write("   • 1/r² scaling → elastic solid\n")
        f.write("   • In between → viscoelastic material\n\n")
        
        f.write("MAIN CODE FUNCTIONS:\n")
        f.write("-" * 19 + "\n")
        f.write("1. load_processed_data(): Loads trajectory files\n")
        f.write("2. prepare_trajectories_for_tpm(): Organizes data by frame\n")
        f.write("3. track_particle_pairs(): Finds pairs and calculates correlations\n")
        f.write("4. bin_correlation_by_distance(): Groups results by distance\n")
        f.write("5. calculate_viscoelastic_modulus(): Extracts material stiffness\n\n")
        
        f.write("HOW TO INTERPRET RESULTS:\n")
        f.write("-" * 25 + "\n")
        f.write("1. CORRELATION vs DISTANCE plot:\n")
        f.write("   • Steeper slope = more solid-like material\n")
        f.write("   • Shallower slope = more fluid-like material\n\n")
        
        f.write("2. VISCOELASTIC MODULUS:\n")
        f.write("   • Higher values = stiffer material\n")
        f.write("   • Compare to known materials:\n")
        f.write("     - Water: ~0.001 Pa\n")
        f.write("     - Cytoplasm: ~1-100 Pa\n")
        f.write("     - Gels: ~100-1000 Pa\n\n")
        
        f.write("3. DATA QUALITY:\n")
        f.write("   • Need many particle pairs for good statistics\n")
        f.write("   • Smooth curves indicate reliable results\n")
        f.write("   • Noisy data suggests insufficient statistics\n\n")
        
        f.write("COMMON ISSUES:\n")
        f.write("-" * 13 + "\n")
        f.write("• Not enough particles → poor statistics\n")
        f.write("• Particles too close → measurement artifacts\n")
        f.write("• Particles too far → weak correlations\n")
        f.write("• Short trajectories → limited time information\n")
    
    print(f"\nAll explanation materials saved in: {output_dir}/")
    print("\nFiles created:")
    print("- tpm_workflow_explanation.png: Step-by-step workflow")
    print("- tpm_physics_concepts.png: Physics behind TPM")
    print("- tpm_code_flow.png: Code structure diagram")
    print("- tpm_explanation.txt: Detailed text explanation")

if __name__ == "__main__":
    main()