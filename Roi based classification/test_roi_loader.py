#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_roi_loader.py

This script tests loading ROIs from ImageJ and Cellpose to understand the differences.
Can be run directly in Spyder or via command line.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pickle
import zipfile
import read_roi
import roifile
import sys
import pprint

# Global variables that can be changed
# Set your default ROI file path here for Spyder execution
DEFAULT_ROI_FILE = r"C:\Users\wanglab-PC-2\Desktop\GEM-ER\KDEL GEM\New folder\ND2 files\Nd2 files\Preblastoderm\Nd2\Membrnae one\PB_em1.nd2_membrane__rois.zip"  # Change this to your ROI file path
VISUALIZE_RESULTS = True  # Set to False to disable visualization

def print_roi_info(roi_data):
    """
    Print information about the loaded ROI for debugging
    """
    print(f"ROI Data Type: {type(roi_data)}")
    
    if isinstance(roi_data, dict):
        print(f"Number of ROIs: {len(roi_data)}")
        
        # Print keys of first ROI
        if roi_data:
            first_roi_name = next(iter(roi_data))
            first_roi = roi_data[first_roi_name]
            print(f"First ROI: {first_roi_name}")
            print(f"  Keys: {list(first_roi.keys())}")
            
            # Check if 'x' and 'y' exist
            if 'x' in first_roi and 'y' in first_roi:
                print(f"  Has x and y coordinates: Yes")
                print(f"  Number of points: {len(first_roi['x'])}")
            else:
                print(f"  Has x and y coordinates: No")
                
            # Print sample of keys and values
            print("  Sample data:")
            for k, v in list(first_roi.items())[:5]:
                print(f"    {k}: {v}")
    else:
        print("ROI data is not a dictionary")

def load_rois_read_roi(roi_file):
    """
    Load ROIs using the read_roi library
    """
    try:
        print(f"Loading ROIs from {roi_file} using read_roi...")
        if roi_file.lower().endswith('.zip'):
            rois = read_roi.read_roi_zip(roi_file)
        else:
            rois = read_roi.read_roi_file(roi_file)
        
        print("Loading successful")
        return rois
    except Exception as e:
        print(f"Error loading ROIs with read_roi: {e}")
        return None

def load_rois_roifile(roi_file):
    """
    Load ROIs using the roifile library
    """
    try:
        print(f"Loading ROIs from {roi_file} using roifile...")
        
        if roi_file.lower().endswith('.zip'):
            rois = roifile.roiread(roi_file)
            
            # Convert to similar format as read_roi
            roi_dict = {}
            for i, roi in enumerate(rois):
                name = roi.name if roi.name else f"ROI_{i+1}"
                coords = roi.coordinates()
                
                roi_dict[name] = {
                    'type': roi.roitype.name.lower(),
                    'x': coords[:, 0].tolist() if coords is not None else [],
                    'y': coords[:, 1].tolist() if coords is not None else [],
                    'width': roi.width,
                    'height': roi.height,
                    'left': roi.left,
                    'top': roi.top
                }
            
            print("Loading successful")
            return roi_dict
        else:
            roi = roifile.ImagejRoi.fromfile(roi_file)
            name = roi.name if roi.name else "ROI_1"
            coords = roi.coordinates()
            
            roi_dict = {
                name: {
                    'type': roi.roitype.name.lower(),
                    'x': coords[:, 0].tolist() if coords is not None else [],
                    'y': coords[:, 1].tolist() if coords is not None else [],
                    'width': roi.width,
                    'height': roi.height,
                    'left': roi.left,
                    'top': roi.top
                }
            }
            
            print("Loading successful")
            return roi_dict
    except Exception as e:
        print(f"Error loading ROIs with roifile: {e}")
        return None

def visualize_rois(rois, title="ROI Visualization", save_path=None):
    """
    Create visualization of ROIs to see what's loaded
    
    Parameters:
    -----------
    rois : dict
        Dictionary containing ROI data
    title : str
        Plot title
    save_path : str, optional
        If provided, save the figure to this path instead of showing
    """
    if not rois:
        print("No ROIs to visualize")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Use different colors for ROIs
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(rois))))
    
    # Plot each ROI
    for i, (roi_id, roi) in enumerate(rois.items()):
        if 'x' in roi and 'y' in roi and len(roi['x']) > 0:
            # Create polygon patch
            color_idx = i % len(colors)
            poly = Polygon(list(zip(roi['x'], roi['y'])), 
                         closed=True, 
                         fill=False, 
                         edgecolor=colors[color_idx], 
                         linewidth=2,
                         label=f"{roi_id}")
            plt.gca().add_patch(poly)
    
    # Set plot parameters
    plt.title(title)
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.grid(alpha=0.3)
    
    # Set origin at top-left corner to match ImageJ coordinates
    plt.gca().invert_yaxis()
    
    # Set equal aspect to prevent distortion
    plt.axis('equal')
    
    # Add legend for ROIs
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

def process_roi_file(roi_file, visualize=VISUALIZE_RESULTS):
    """
    Process an ROI file and optionally visualize the results
    """
    if not os.path.exists(roi_file):
        print(f"File not found: {roi_file}")
        return
    
    output_dir = os.path.dirname(roi_file)
    
    # Try loading with read_roi
    rois_read_roi = load_rois_read_roi(roi_file)
    if rois_read_roi:
        print("\nROI Information (read_roi):")
        print_roi_info(rois_read_roi)
        
        # Visualize if requested
        if visualize:
            visualize_rois(rois_read_roi, "ROIs Loaded with read_roi")
            
            # Save visualization
            base_name = os.path.splitext(os.path.basename(roi_file))[0]
            save_path = os.path.join(output_dir, f"{base_name}_read_roi_viz.png")
            visualize_rois(rois_read_roi, "ROIs Loaded with read_roi", save_path)
    
    # Try loading with roifile
    rois_roifile = load_rois_roifile(roi_file)
    if rois_roifile:
        print("\nROI Information (roifile):")
        print_roi_info(rois_roifile)
        
        # Visualize if requested
        if visualize:
            visualize_rois(rois_roifile, "ROIs Loaded with roifile")
            
            # Save visualization
            base_name = os.path.splitext(os.path.basename(roi_file))[0]
            save_path = os.path.join(output_dir, f"{base_name}_roifile_viz.png")
            visualize_rois(rois_roifile, "ROIs Loaded with roifile", save_path)
    
    return {
        'read_roi': rois_read_roi,
        'roifile': rois_roifile
    }

def main():
    """
    Main function - can be run in Spyder or via command line
    """
    # Check if file path was provided as command line argument
    if len(sys.argv) > 1:
        roi_file = sys.argv[1]
        print(f"Using command line argument for ROI file: {roi_file}")
        process_roi_file(roi_file)
    else:
        # Use default path for Spyder execution
        print(f"Using default ROI file: {DEFAULT_ROI_FILE}")
        process_roi_file(DEFAULT_ROI_FILE)

# For Spyder execution
if __name__ == "__main__":
    main()
    
    # Uncomment the following line if you want to interactively select a file
    # For interactive file selection, you could use:
    # from tkinte