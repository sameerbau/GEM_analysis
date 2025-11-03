# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 18:49:58 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diffusion_analysis_launcher.py

This script provides a menu-based interface to run the ROI-based diffusion analysis workflow.
Users can choose to run individual modules or the complete workflow in sequence.

Usage:
python diffusion_analysis_launcher.py
"""

import os
import sys
import subprocess
from pathlib import Path
import time
from datetime import datetime

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the program header."""
    clear_screen()
    print("=" * 80)
    print("ROI-BASED DIFFUSION ANALYSIS TOOLKIT".center(80))
    print("=" * 80)
    print()

def print_module_info(module_num, module_name, module_desc):
    """Print formatted module information."""
    print(f"  {module_num}. {module_name}")
    print(f"     {module_desc}")
    print()

def display_menu():
    """Display the main menu."""
    print_header()
    print("AVAILABLE MODULES:")
    print()
    
    print_module_info(1, "ROI Loader", 
                    "Load ImageJ ROIs and assign trajectories to regions.")
    
    print_module_info(2, "ROI Diffusion Analyzer", 
                    "Analyze diffusion within ROIs and perform statistical comparisons.")
    
    print_module_info(3, "Advanced Diffusion Statistics", 
                    "Implement advanced statistical methods for analyzing anomalous diffusion.")
    
    print_module_info(4, "Diffusion Heatmap Generator", 
                    "Create heatmaps and spatial visualizations of diffusion properties.")
    
    print_module_info(5, "Run Complete Workflow", 
                    "Run all modules in sequence (1 → 2 → 3 → 4).")
    
    print("  0. Exit")
    print()
    print("=" * 80)

def run_module(module_path):
    """
    Run a specific module using the Python interpreter.
    
    Args:
        module_path: Path to the module script
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get the current Python interpreter
        python_exe = sys.executable
        
        # Run the module with the current Python interpreter
        process = subprocess.run([python_exe, module_path], 
                               stdout=sys.stdout, stderr=sys.stderr)
        
        # Check if the process was successful
        return process.returncode == 0
    
    except Exception as e:
        print(f"Error running module: {e}")
        return False

def run_complete_workflow():
    """
    Run the complete workflow by executing all modules in sequence.
    
    Returns:
        True if all modules ran successfully, False otherwise
    """
    modules = [
        'roi_loader.py',
        'roi_diffusion_analyzer.py',
        'advanced_diffusion_stats.py',
        'diffusion_heatmap_generator.py'
    ]
    
    print_header()
    print("RUNNING COMPLETE WORKFLOW")
    print("=" * 80)
    print()
    
    success = True
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for i, module in enumerate(modules):
        module_path = os.path.join(current_dir, module)
        
        print(f"Step {i+1}/{len(modules)}: Running {module}...")
        print("-" * 40)
        
        # Run the module
        step_success = run_module(module_path)
        
        if step_success:
            print(f"\n✓ {module} completed successfully.")
        else:
            print(f"\n✗ {module} failed to complete.")
            success = False
            break
        
        # Pause between modules unless it's the last one
        if i < len(modules) - 1:
            print("\nProceeding to next step in 3 seconds...")
            time.sleep(3)
        
        print("\n" + "=" * 40 + "\n")
    
    if success:
        print("✓ Complete workflow executed successfully!")
    else:
        print("✗ Workflow execution stopped due to an error.")
    
    return success

def main():
    """Main function to run the launcher."""
    while True:
        display_menu()
        
        # Get user choice
        try:
            choice = int(input("Enter your choice (0-5): ").strip())
        except ValueError:
            print("Invalid input. Please enter a number.")
            time.sleep(2)
            continue
        
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        if choice == 0:
            print("\nExiting program. Goodbye!")
            break
            
        elif choice == 1:
            module_path = os.path.join(current_dir, 'roi_loader.py')
            run_module(module_path)
            
        elif choice == 2:
            module_path = os.path.join(current_dir, 'roi_diffusion_analyzer.py')
            run_module(module_path)
            
        elif choice == 3:
            module_path = os.path.join(current_dir, 'advanced_diffusion_stats.py')
            run_module(module_path)
            
        elif choice == 4:
            module_path = os.path.join(current_dir, 'diffusion_heatmap_generator.py')
            run_module(module_path)
            
        elif choice == 5:
            run_complete_workflow()
            
        else:
            print("Invalid choice. Please try again.")
            time.sleep(2)
            continue
        
        # Pause before showing the menu again
        input("\nPress Enter to return to the main menu...")

if __name__ == "__main__":
    main()