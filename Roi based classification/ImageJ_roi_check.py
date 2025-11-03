#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
roi_orientation_check_exact.py

This script loads and displays ROIs in both original orientation and with
flipped y-coordinates to verify the correct coordinate system.

Key features:
- Maintains EXACT pixel coordinates without any rescaling
- Forces consistent axes limits between plots
- Shows coordinate axes and numerical pixel values
- Uses the same scale for both original and flipped plots
- Adds grid lines at 100-pixel intervals for reference

Usage:
python roi_orientation_check_exact.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.patches import Rectangle

try:
    # Try to import read_roi for ImageJ ZIP files
    from read_roi import read_roi_zip
except ImportError:
    print("Warning: read_roi package not found. Install with: pip install read-roi")
    print("Install with: pip install read-roi")

def load_roi_from_zip(file_path):
    """
    Load ROIs from an ImageJ ZIP file, keeping exact pixel coordinates.
    
    Args:
        file_path: Path to the ROI zip file
        
    Returns:
        Dictionary mapping ROI names to coordinates
    """
    try:
        rois = read_roi_zip(file_path)
        
        # Convert to a more convenient format
        processed_rois = {}
        for name, roi in rois.items():
            if 'x' in roi and 'y' in roi:
                # Convert to numpy arrays, keeping exact pixel coordinates
                x = np.array(roi['x'])
                y = np.array(roi['y'])
                
                processed_rois[name] = {
                    'x': x,
                    'y': y,
                    'type': roi.get('type', 'unknown'),
                    'position': roi.get('position', 0)  # Slice number for stacks
                }
        
        return processed_rois
    except Exception as e:
        print(f"Error loading ROIs from {file_path}: {e}")
        return None

def get_roi_bounds(rois):
    """
    Get the bounding box of all ROIs to ensure consistent axis limits.
    
    Args:
        rois: Dictionary mapping ROI names to coordinates
        
    Returns:
        Tuple of (min_x, max_x, min_y, max_y)
    """
    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
    
    for roi in rois.values():
        x, y = roi['x'], roi['y']
        min_x = min(min_x, np.min(x))
        max_x = max(max_x, np.max(x))
        min_y = min(min_y, np.min(y))
        max_y = max(max_y, np.max(y))
    
    return min_x, max_x, min_y, max_y

def create_visualization_figure(image_size):
    """
    Create a figure with two subplots for the visualization.
    
    Args:
        image_size: Tuple of (width, height) for the image
        
    Returns:
        Figure and axes objects
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Set titles
    ax1.set_title("Original ROIs (ImageJ coordinate system, top-left origin)", fontsize=14)
    ax2.set_title("Flipped ROIs (bottom-left origin)", fontsize=14)
    
    # Set labels
    for ax in [ax1, ax2]:
        ax.set_xlabel('X position (pixels)', fontsize=12)
        ax.set_ylabel('Y position (pixels)', fontsize=12)
    
    # Set aspect ratio to be equal (1:1 pixel aspect ratio)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    
    return fig, (ax1, ax2)

def visualize_rois_with_exact_pixels(rois, image_size, output_path=None, image_path=None):
    """
    Visualize ROIs in both original and flipped coordinate systems with exact pixel dimensions.
    
    Args:
        rois: Dictionary mapping ROI names to coordinates
        image_size: Tuple of (width, height) for the image in pixels
        output_path: Path to save the visualization (optional)
        image_path: Path to background image (optional)
    """
    if not rois:
        print("No ROIs to visualize")
        return
    
    width, height = image_size
    
    # Get bounds to ensure consistent axes
    min_x, max_x, min_y, max_y = get_roi_bounds(rois)
    
    # Add margin to bounds
    margin = max(width, height) * 0.05
    min_x = max(0, min_x - margin)
    max_x = min(width, max_x + margin)
    min_y = max(0, min_y - margin)
    max_y = min(height, max_y + margin)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = create_visualization_figure(image_size)
    
    # If an image path is provided, load and display the image
    if image_path and os.path.exists(image_path):
        try:
            img = plt.imread(image_path)
            # Display image with correct orientation in both subplots
            ax1.imshow(img, origin='upper', extent=[0, width, height, 0])  # Original (top-left origin)
            ax2.imshow(img, origin='lower', extent=[0, width, 0, height])  # Flipped (bottom-left origin)
        except Exception as e:
            print(f"Error loading image: {e}")
    
    # Generate different colors for different ROIs
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(rois))))
    
    # Plot only a subset of ROIs if there are too many
    roi_items = list(rois.items())
    if len(roi_items) > 10:
        print(f"Showing only 10 of {len(roi_items)} ROIs for clarity")
        roi_items = roi_items[:10]
    
    # Plot ROIs on both subplots
    for i, (name, roi) in enumerate(roi_items):
        color_idx = i % len(colors)  # Cycle through colors if more than 10 ROIs
        
        # Get coordinates
        x, y = roi['x'], roi['y']
        
        # Plot original ROI
        ax1.plot(x, y, '-', color=colors[color_idx], linewidth=2)
        ax1.plot(x, y, 'o', color=colors[color_idx], markersize=4)
        
        # Plot flipped ROI
        ax2.plot(x, height - y, '-', color=colors[color_idx], linewidth=2, label=name)
        ax2.plot(x, height - y, 'o', color=colors[color_idx], markersize=4)
        
        # Add labels
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
        ax1.text(centroid_x, centroid_y, name, fontsize=9, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7))
        ax2.text(centroid_x, height - centroid_y, name, fontsize=9, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Set axes limits precisely
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_y, max_y)
    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(height - max_y, height - min_y)
    
    # Add grid lines at 100-pixel intervals
    for ax in [ax1, ax2]:
        ax.grid(True, which='both', color='gray', linestyle='--', alpha=0.5)
        
        # Add more detailed grid at 100-pixel intervals
        x_ticks = np.arange(0, width + 1, 100)
        y_ticks = np.arange(0, height + 1, 100)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        
        # Add minor ticks at 50-pixel intervals
        minor_x_ticks = np.arange(0, width + 1, 50)
        minor_y_ticks = np.arange(0, height + 1, 50)
        ax.set_xticks(minor_x_ticks, minor=True)
        ax.set_yticks(minor_y_ticks, minor=True)
        ax.grid(which='minor', alpha=0.2)
    
    # Add coordinate system references
    # Origin point
    ax1.plot(0, 0, 'ro', markersize=10)
    ax1.text(10, 20, "Origin (0,0)", color='red', fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7))
    
    ax2.plot(0, height, 'ro', markersize=10)
    ax2.text(10, height - 20, "Origin (0,0) after flip", color='red', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7))
    
    # Add reference boxes at the corners
    for ax, y_pos in [(ax1, 0), (ax2, height)]:
        corner_size = 50
        rect = Rectangle((0, y_pos), corner_size, corner_size * (-1 if y_pos > 0 else 1),
                       linewidth=2, edgecolor='red', facecolor='none', alpha=0.7)
        ax.add_patch(rect)
        ax.text(corner_size/2, y_pos + corner_size/2 * (-1 if y_pos > 0 else 1), 
               "Corner\nReference", color='red', ha='center', va='center', fontsize=8,
               bbox=dict(facecolor='white', alpha=0.7))
    
    # Add legend to second subplot
    ax2.legend(loc='best')
    
    # Ensure layout is tight
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROI visualization saved to {output_path}")
    
    plt.show()

def main():
    """Main function to check ROI orientation with exact pixel coordinates."""
    print("\nROI Orientation Check with Exact Pixel Coordinates")
    print("===============================================")
    
    # Ask for ROI file
    roi_file = input("Enter path to ImageJ ROI ZIP file: ")
    if not os.path.exists(roi_file):
        print(f"File {roi_file} does not exist")
        return
    
    # Load ROIs in exact pixel coordinates
    rois = load_roi_from_zip(roi_file)
    if not rois:
        print("Failed to load ROIs")
        return
    
    print(f"Loaded {len(rois)} ROIs")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(roi_file), "roi_orientation_check")
    os.makedirs(output_dir, exist_ok=True)
    
    # Ask for image dimensions
    try:
        image_width = int(input("Enter original image width in pixels: "))
        image_height = int(input("Enter original image height in pixels: "))
        image_size = (image_width, image_height)
        print(f"Using image dimensions: {image_width} x {image_height} pixels")
    except ValueError:
        print("Invalid input. Using default size of 512x512 pixels.")
        image_size = (512, 512)
    
    # Ask if user wants to overlay an image
    image_path = input("Enter path to background image (optional, press Enter to skip): ")
    if image_path and not os.path.exists(image_path):
        print(f"Image file {image_path} does not exist. Proceeding without image.")
        image_path = None
    
    # Visualize ROIs with exact pixel coordinates
    visualize_rois_with_exact_pixels(
        rois, 
        image_size=image_size,
        output_path=os.path.join(output_dir, "roi_orientation_exact_pixels.png"),
        image_path=image_path
    )
    
    print("\nCheck the visualizations to determine which orientation is correct.")
    print("- Left plot: Original ROIs (ImageJ coordinate system, top-left origin)")
    print("- Right plot: Flipped ROIs (bottom-left origin)")
    print("\nBoth plots show EXACT pixel coordinates without rescaling.")
    print(f"\nVisualization saved to {output_dir}")

if __name__ == "__main__":
    main()