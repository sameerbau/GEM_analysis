# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 18:48:51 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diffusion_heatmap_generator.py

This script creates heatmaps and spatial visualizations of diffusion properties,
allowing for visual analysis of spatial patterns in diffusion behavior.

Input:
- ROI-assigned trajectory data (.pkl files)
- Analyzed diffusion data (.pkl files)
- Original image file for background reference (optional)

Output:
- Heatmaps showing diffusion coefficient distribution
- Spatial visualizations of anomalous diffusion exponents
- Combined visualizations with statistical overlay

Usage:
python diffusion_heatmap_generator.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from scipy.interpolate import griddata
import pickle
from pathlib import Path
from datetime import datetime
from skimage import io, transform
import warnings

# Global parameters that can be modified
# =====================================
# Resolution for interpolation grid
GRID_RESOLUTION = 100
# Alpha transparency for heatmap overlay
HEATMAP_ALPHA = 0.7
# Color map for diffusion coefficient
DIFFUSION_CMAP = 'viridis'
# Color map for anomalous diffusion exponent
ALPHA_CMAP = 'coolwarm'
# Marker size for trajectory points
MARKER_SIZE = 5
# Output directory name format
OUTPUT_DIR_FORMAT = 'diffusion_heatmap_%Y%m%d_%H%M%S'
# =====================================

def load_data(file_path):
    """
    Load trajectory or ROI data from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary containing the data
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def load_image(image_path):
    """
    Load an image file for background reference.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        NumPy array containing the image data
    """
    try:
        # Suppress warnings about low contrast
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = io.imread(image_path)
        
        # If image has multiple channels, convert to grayscale
        if image.ndim > 2:
            # Simple grayscale conversion by averaging channels
            image = np.mean(image, axis=2).astype(np.uint8)
        
        print(f"Successfully loaded image from {image_path}")
        return image
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        return None

def create_diffusion_heatmap(trajectories, output_path, image=None, 
                           property_name='D', cmap=DIFFUSION_CMAP, 
                           title='Diffusion Coefficient Heatmap'):
    """
    Create a heatmap visualization of diffusion properties.
    
    Args:
        trajectories: List of trajectory dictionaries
        output_path: Path to save the visualization
        image: Optional background image
        property_name: Name of property to visualize ('D' or 'alpha')
        cmap: Colormap to use
        title: Title for the plot
        
    Returns:
        None (saves visualization to file)
    """
    # Extract trajectory positions and property values
    positions_x = []
    positions_y = []
    property_values = []
    
    for traj in trajectories:
        # Skip trajectories without the requested property
        if property_name not in traj or np.isnan(traj[property_name]):
            continue
        
        # Use mean position of trajectory
        positions_x.append(np.mean(traj['x']))
        positions_y.append(np.mean(traj['y']))
        property_values.append(traj[property_name])
    
    if not property_values:
        print(f"No valid {property_name} values found for heatmap")
        return
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Determine plot limits
    x_min, x_max = min(positions_x), max(positions_x)
    y_min, y_max = min(positions_y), max(positions_y)
    
    # Add padding to limits
    padding = 0.1  # 10% padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding
    
    # If image is provided, display it as background
    if image is not None:
        # Determine image extent to match trajectory coordinates
        img_extent = [x_min, x_max, y_max, y_min]  # Note: y is flipped for image coordinates
        plt.imshow(image, cmap='gray', extent=img_extent, alpha=0.5)
    
    # Create interpolation grid
    grid_x = np.linspace(x_min, x_max, GRID_RESOLUTION)
    grid_y = np.linspace(y_min, y_max, GRID_RESOLUTION)
    xi, yi = np.meshgrid(grid_x, grid_y)
    
    # Interpolate property values on the grid
    # Use 'linear' interpolation with a fill value for points outside the convex hull
    fill_value = np.mean(property_values)
    zi = griddata((positions_x, positions_y), property_values, (xi, yi), 
                 method='linear', fill_value=fill_value)
    
    # Create contour plot
    contour = plt.contourf(xi, yi, zi, 20, cmap=cmap, alpha=HEATMAP_ALPHA)
    
    # Add colorbar
    cbar = plt.colorbar(contour)
    if property_name == 'D':
        cbar.set_label('Diffusion Coefficient (μm²/s)', fontsize=12)
    elif property_name == 'alpha':
        cbar.set_label('Anomalous Diffusion Exponent (α)', fontsize=12)
    
    # Add trajectory points
    plt.scatter(positions_x, positions_y, c=property_values, cmap=cmap, 
               edgecolor='black', s=MARKER_SIZE, alpha=0.8, zorder=3)
    
    # Set plot parameters
    plt.title(title, fontsize=14)
    plt.xlabel('X Position (pixels)', fontsize=12)
    plt.ylabel('Y Position (pixels)', fontsize=12)
    
    # Use origin at top-left corner to match image coordinates
    plt.gca().invert_yaxis()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Heatmap saved to {output_path}")

def create_roi_heatmap(roi_data, rois_info, output_path, image=None,
                     property_name='mean_D', cmap=DIFFUSION_CMAP,
                     title='ROI-based Diffusion Heatmap'):
    """
    Create a heatmap visualization based on ROI statistics.
    
    Args:
        roi_data: Dictionary containing ROI trajectory data and statistics
        rois_info: Dictionary with ROI geometry information
        output_path: Path to save the visualization
        image: Optional background image
        property_name: Name of property to visualize ('mean_D' or 'mean_alpha')
        cmap: Colormap to use
        title: Title for the plot
        
    Returns:
        None (saves visualization to file)
    """
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Determine property statistics across all ROIs
    property_values = []
    
    # Find appropriate property values
    if property_name == 'mean_D':
        for roi_id, stats in roi_data['roi_statistics'].items():
            if roi_id != 'unassigned' and stats['n'] > 0:
                property_values.append(stats['mean_D'])
    elif property_name == 'mean_alpha':
        # Check if advanced analysis data is available
        if 'roi_anomalous_diffusion' in roi_data:
            for roi_id, results in roi_data['roi_anomalous_diffusion'].items():
                if roi_id != 'unassigned' and 'mean_alpha' in results:
                    property_values.append(results['mean_alpha'])
    
    if not property_values:
        print(f"No valid {property_name} values found for ROI heatmap")
        return
    
    # Determine property range for colormap
    property_min = min(property_values)
    property_max = max(property_values)
    
    # If image is provided, display it as background
    if image is not None:
        # Determine image extent
        x_coords = []
        y_coords = []
        for roi_id in roi_data['roi_assignments']:
            if roi_id != 'unassigned' and roi_id in rois_info:
                roi = rois_info[roi_id]
                if 'x' in roi and 'y' in roi:
                    x_coords.extend(roi['x'])
                    y_coords.extend(roi['y'])
        
        if x_coords and y_coords:
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add padding
            padding = 0.1
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_min -= x_range * padding
            x_max += x_range * padding
            y_min -= y_range * padding
            y_max += y_range * padding
            
            # Display image
            img_extent = [x_min, x_max, y_max, y_min]  # Note: y is flipped for image coordinates
            plt.imshow(image, cmap='gray', extent=img_extent, alpha=0.5)
    
    # Create polygon patches for ROIs
    patches_list = []
    colors_list = []
    texts = []
    
    for roi_id in roi_data['roi_assignments']:
        if roi_id != 'unassigned' and roi_id in rois_info:
            roi = rois_info[roi_id]
            
            if 'x' in roi and 'y' in roi:
                # Create polygon for this ROI
                polygon = patches.Polygon(list(zip(roi['x'], roi['y'])), closed=True)
                patches_list.append(polygon)
                
                # Determine color based on property value
                if property_name == 'mean_D' and roi_id in roi_data['roi_statistics']:
                    stats = roi_data['roi_statistics'][roi_id]
                    if stats['n'] > 0 and not np.isnan(stats['mean_D']):
                        # Normalize property value to [0, 1] range
                        norm_value = (stats['mean_D'] - property_min) / (property_max - property_min)
                        colors_list.append(norm_value)
                        
                        # Add text annotation with property value
                        centroid_x = np.mean(roi['x'])
                        centroid_y = np.mean(roi['y'])
                        texts.append({
                            'x': centroid_x,
                            'y': centroid_y,
                            'text': f"{stats['mean_D']:.3f}\nn={stats['n']}"
                        })
                    else:
                        # Use default color for ROIs with no data
                        colors_list.append(0)
                elif property_name == 'mean_alpha' and 'roi_anomalous_diffusion' in roi_data:
                    # Check if ROI has anomalous diffusion data
                    if roi_id in roi_data['roi_anomalous_diffusion']:
                        results = roi_data['roi_anomalous_diffusion'][roi_id]
                        if 'mean_alpha' in results:
                            # Normalize property value to [0, 1] range
                            norm_value = (results['mean_alpha'] - property_min) / (property_max - property_min)
                            colors_list.append(norm_value)
                            
                            # Add text annotation
                            centroid_x = np.mean(roi['x'])
                            centroid_y = np.mean(roi['y'])
                            texts.append({
                                'x': centroid_x,
                                'y': centroid_y,
                                'text': f"α={results['mean_alpha']:.2f}\nn={len(results['alpha_values'])}"
                            })
                        else:
                            colors_list.append(0)
                    else:
                        colors_list.append(0)
                else:
                    colors_list.append(0)
    
    # Create patch collection with specified colormap
    if patches_list:
        # Convert normalized values to colormap colors
        if cmap == DIFFUSION_CMAP:
            cmap_obj = plt.cm.get_cmap(cmap)
        elif cmap == ALPHA_CMAP:
            # For alpha values, use a diverging colormap centered at 1.0
            # Create a custom colormap
            cmap_obj = LinearSegmentedColormap.from_list(
                'alpha_cmap', 
                [(0, 'blue'), (0.45, 'lightblue'), 
                 (0.5, 'white'), 
                 (0.55, 'lightsalmon'), (1.0, 'red')]
            )
        else:
            cmap_obj = plt.cm.get_cmap(cmap)
        
        colors = [cmap_obj(c) for c in colors_list]
        
        # Create patch collection
        collection = PatchCollection(patches_list, facecolors=colors, 
                                    edgecolors='black', linewidths=1, alpha=HEATMAP_ALPHA)
        plt.gca().add_collection(collection)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_obj)
        sm.set_array([property_min, property_max])
        cbar = plt.colorbar(sm)
        
        if property_name == 'mean_D':
            cbar.set_label('Mean Diffusion Coefficient (μm²/s)', fontsize=12)
        elif property_name == 'mean_alpha':
            cbar.set_label('Mean Anomalous Diffusion Exponent (α)', fontsize=12)
        
        # Add text annotations
        for text_item in texts:
            plt.text(text_item['x'], text_item['y'], text_item['text'], 
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, pad=2))
    
    # Set plot parameters
    plt.title(title, fontsize=14)
    plt.xlabel('X Position (pixels)', fontsize=12)
    plt.ylabel('Y Position (pixels)', fontsize=12)
    
    # Use origin at top-left corner to match image coordinates
    plt.gca().invert_yaxis()
    
    # Set equal aspect to prevent distortion
    plt.axis('equal')
    
    # Try to set axis limits if we have ROI coordinates
    if x_coords and y_coords:
        plt.xlim(x_min, x_max)
        plt.ylim(y_max, y_min)  # Note: y is flipped
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"ROI heatmap saved to {output_path}")

def create_trajectory_visualization(trajectories, output_path, image=None,
                                  color_by='D', cmap=DIFFUSION_CMAP,
                                  title='Trajectory Visualization'):
    """
    Create a visualization showing individual trajectories colored by property.
    
    Args:
        trajectories: List of trajectory dictionaries
        output_path: Path to save the visualization
        image: Optional background image
        color_by: Property to use for coloring ('D', 'alpha', or 'length')
        cmap: Colormap to use
        title: Title for the plot
        
    Returns:
        None (saves visualization to file)
    """
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Extract property values for coloring
    property_values = []
    valid_trajectories = []
    
    for traj in trajectories:
        # Skip trajectories without the requested property
        if color_by == 'length':
            # Use trajectory length
            if 'x' in traj:
                property_values.append(len(traj['x']))
                valid_trajectories.append(traj)
        elif color_by in traj and not np.isnan(traj[color_by]):
            property_values.append(traj[color_by])
            valid_trajectories.append(traj)
    
    if not valid_trajectories:
        print(f"No valid trajectories found for visualization")
        return
    
    # Determine property range for colormap
    property_min = min(property_values)
    property_max = max(property_values)
    
    # Determine plot limits
    x_coords = []
    y_coords = []
    
    for traj in valid_trajectories:
        x_coords.extend(traj['x'])
        y_coords.extend(traj['y'])
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add padding
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding
    
    # If image is provided, display it as background
    if image is not None:
        # Display image
        img_extent = [x_min, x_max, y_max, y_min]  # Note: y is flipped for image coordinates
        plt.imshow(image, cmap='gray', extent=img_extent, alpha=0.5)
    
    # Create colormap for trajectories
    cmap_obj = plt.cm.get_cmap(cmap)
    norm = plt.Normalize(property_min, property_max)
    
    # Plot each trajectory with color based on property
    for i, traj in enumerate(valid_trajectories):
        # Get property value and normalized color
        prop_val = property_values[i]
        color = cmap_obj(norm(prop_val))
        
        # Plot trajectory
        plt.plot(traj['x'], traj['y'], '-', color=color, linewidth=1.5, alpha=0.7)
        
        # Add marker for trajectory start
        plt.plot(traj['x'][0], traj['y'][0], 'o', color=color, 
                markersize=5, markeredgecolor='black')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    
    if color_by == 'D':
        cbar.set_label('Diffusion Coefficient (μm²/s)', fontsize=12)
    elif color_by == 'alpha':
        cbar.set_label('Anomalous Diffusion Exponent (α)', fontsize=12)
    elif color_by == 'length':
        cbar.set_label('Trajectory Length (frames)', fontsize=12)
    
    # Set plot parameters
    plt.title(title, fontsize=14)
    plt.xlabel('X Position (pixels)', fontsize=12)
    plt.ylabel('Y Position (pixels)', fontsize=12)
    
    # Use origin at top-left corner to match image coordinates
    plt.gca().invert_yaxis()
    
    # Set axis limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_max, y_min)  # Note: y is flipped
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Trajectory visualization saved to {output_path}")

def create_combined_visualization(roi_data, advanced_data, rois_info, output_path, image=None):
    """
    Create a comprehensive visualization combining diffusion coefficient and anomalous exponent.
    
    Args:
        roi_data: Dictionary containing ROI trajectory data
        advanced_data: Dictionary containing advanced analysis results
        rois_info: Dictionary with ROI geometry information
        output_path: Path to save the visualization
        image: Optional background image
        
    Returns:
        None (saves visualization to file)
    """
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    
    # Flatten axes array for easier indexing
    axs = axs.flatten()
    
    # 1. ROI-based diffusion coefficient heatmap (upper left)
    ax = axs[0]
    plt.sca(ax)
    
    # Determine diffusion coefficient values across all ROIs
    d_values = []
    for roi_id, stats in roi_data['roi_statistics'].items():
        if roi_id != 'unassigned' and stats['n'] > 0:
            d_values.append(stats['mean_D'])
    
    if not d_values:
        print("No valid diffusion coefficient values found")
        return
    
    # Determine range for colormap
    d_min = min(d_values)
    d_max = max(d_values)
    
    # If image is provided, display it as background
    if image is not None:
        # Determine image extent based on ROI coordinates
        x_coords = []
        y_coords = []
        for roi_id in roi_data['roi_assignments']:
            if roi_id != 'unassigned' and roi_id in rois_info:
                roi = rois_info[roi_id]
                if 'x' in roi and 'y' in roi:
                    x_coords.extend(roi['x'])
                    y_coords.extend(roi['y'])
        
        if x_coords and y_coords:
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add padding
            padding = 0.1
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_min -= x_range * padding
            x_max += x_range * padding
            y_min -= y_range * padding
            y_max += y_range * padding
            
            # Display image
            img_extent = [x_min, x_max, y_max, y_min]
            ax.imshow(image, cmap='gray', extent=img_extent, alpha=0.5)
    
    # Create polygon patches for ROIs
    patches_list = []
    d_colors = []
    d_texts = []
    
    for roi_id in roi_data['roi_assignments']:
        if roi_id != 'unassigned' and roi_id in rois_info:
            roi = rois_info[roi_id]
            
            if 'x' in roi and 'y' in roi:
                # Create polygon for this ROI
                polygon = patches.Polygon(list(zip(roi['x'], roi['y'])), closed=True)
                patches_list.append(polygon)
                
                # Determine color based on diffusion coefficient
                if roi_id in roi_data['roi_statistics']:
                    stats = roi_data['roi_statistics'][roi_id]
                    if stats['n'] > 0 and not np.isnan(stats['mean_D']):
                        # Normalize to [0, 1] range
                        norm_value = (stats['mean_D'] - d_min) / (d_max - d_min)
                        d_colors.append(norm_value)
                        
                        # Add text annotation
                        centroid_x = np.mean(roi['x'])
                        centroid_y = np.mean(roi['y'])
                        d_texts.append({
                            'x': centroid_x,
                            'y': centroid_y,
                            'text': f"D={stats['mean_D']:.3f}\nn={stats['n']}"
                        })
                    else:
                        d_colors.append(0)
                else:
                    d_colors.append(0)
    
    # Create patch collection with diffusion coefficient colormap
    if patches_list:
        # Convert normalized values to colormap colors
        d_cmap = plt.cm.get_cmap(DIFFUSION_CMAP)
        d_colors_rgb = [d_cmap(c) for c in d_colors]
        
        # Create patch collection
        collection = patches.PatchCollection(patches_list, facecolors=d_colors_rgb, 
                                          edgecolors='black', linewidths=1, alpha=HEATMAP_ALPHA)
        ax.add_collection(collection)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=d_cmap)
        sm.set_array([d_min, d_max])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Mean Diffusion Coefficient (μm²/s)', fontsize=10)
        
        # Add text annotations
        for text_item in d_texts:
            ax.text(text_item['x'], text_item['y'], text_item['text'], 
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, pad=2))
    
    # Set subplot title and labels
    ax.set_title('Diffusion Coefficient by ROI', fontsize=12)
    ax.set_xlabel('X Position (pixels)', fontsize=10)
    ax.set_ylabel('Y Position (pixels)', fontsize=10)
    
    # Use origin at top-left corner to match image coordinates
    ax.invert_yaxis()
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # 2. ROI-based anomalous exponent heatmap (upper right)
    ax = axs[1]
    plt.sca(ax)
    
    # Check if we have anomalous diffusion data
    if 'roi_anomalous_diffusion' in advanced_data:
        # Determine anomalous exponent values across all ROIs
        alpha_values = []
        for roi_id, results in advanced_data['roi_anomalous_diffusion'].items():
            if roi_id != 'unassigned' and 'mean_alpha' in results:
                alpha_values.append(results['mean_alpha'])
        
        if alpha_values:
            # Determine range for colormap
            alpha_min = min(alpha_values)
            alpha_max = max(alpha_values)
            
            # If image is provided, display it as background
            if image is not None and x_coords and y_coords:
                ax.imshow(image, cmap='gray', extent=img_extent, alpha=0.5)
            
            # Create polygon patches for ROIs
            alpha_colors = []
            alpha_texts = []
            
            for roi_id in roi_data['roi_assignments']:
                if roi_id != 'unassigned' and roi_id in rois_info:
                    roi = rois_info[roi_id]
                    
                    if 'x' in roi and 'y' in roi:
                        # Determine color based on anomalous exponent
                        if roi_id in advanced_data['roi_anomalous_diffusion']:
                            results = advanced_data['roi_anomalous_diffusion'][roi_id]
                            if 'mean_alpha' in results:
                                # Normalize to [0, 1] range
                                norm_value = (results['mean_alpha'] - alpha_min) / (alpha_max - alpha_min)
                                alpha_colors.append(norm_value)
                                
                                # Add text annotation
                                centroid_x = np.mean(roi['x'])
                                centroid_y = np.mean(roi['y'])
                                alpha_texts.append({
                                    'x': centroid_x,
                                    'y': centroid_y,
                                    'text': f"α={results['mean_alpha']:.2f}\nn={len(results['alpha_values'])}"
                                })
                            else:
                                alpha_colors.append(0.5)  # Neutral value for missing data
                        else:
                            alpha_colors.append(0.5)  # Neutral value for missing data
            
            # Create patch collection with anomalous exponent colormap
            if patches_list:
                # Create a diverging colormap centered at 1.0
                alpha_cmap = LinearSegmentedColormap.from_list(
                    'alpha_cmap', 
                    [(0, 'blue'), (0.45, 'lightblue'), 
                     (0.5, 'white'), 
                     (0.55, 'lightsalmon'), (1.0, 'red')]
                )
                
                alpha_colors_rgb = [alpha_cmap(c) for c in alpha_colors]
                
                # Create patch collection
                collection = patches.PatchCollection(patches_list, facecolors=alpha_colors_rgb, 
                                                  edgecolors='black', linewidths=1, alpha=HEATMAP_ALPHA)
                ax.add_collection(collection)
                
                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=alpha_cmap)
                sm.set_array([alpha_min, alpha_max])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('Mean Anomalous Diffusion Exponent (α)', fontsize=10)
                
                # Add reference line for α=1 (normal diffusion)
                cbar.ax.axhline(y=(1.0 - alpha_min) / (alpha_max - alpha_min), color='k', linestyle='--', linewidth=1)
                
                # Add text annotations
                for text_item in alpha_texts:
                    ax.text(text_item['x'], text_item['y'], text_item['text'], 
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.7, pad=2))
    
    # Set subplot title and labels
    ax.set_title('Anomalous Diffusion Exponent by ROI', fontsize=12)
    ax.set_xlabel('X Position (pixels)', fontsize=10)
    ax.set_ylabel('Y Position (pixels)', fontsize=10)
    
    # Use origin at top-left corner to match image coordinates
    ax.invert_yaxis()
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # 3. Diffusion coefficient vs. anomalous exponent scatter plot (lower left)
    ax = axs[2]
    plt.sca(ax)
    
    # Check if we have both diffusion and anomalous diffusion data
    if 'roi_anomalous_diffusion' in advanced_data:
        # Collect pairs of diffusion coefficient and anomalous exponent
        d_alpha_pairs = []
        roi_labels = []
        
        for roi_id, results in advanced_data['roi_anomalous_diffusion'].items():
            if roi_id != 'unassigned' and 'mean_alpha' in results:
                # Get diffusion coefficient from ROI statistics
                if roi_id in roi_data['roi_statistics']:
                    stats = roi_data['roi_statistics'][roi_id]
                    if stats['n'] > 0 and not np.isnan(stats['mean_D']):
                        d_alpha_pairs.append((stats['mean_D'], results['mean_alpha']))
                        roi_labels.append(roi_id.split('-')[0][:8])  # Shortened ROI ID
        
        if d_alpha_pairs:
            # Separate diffusion coefficient and anomalous exponent values
            d_values = [pair[0] for pair in d_alpha_pairs]
            alpha_values = [pair[1] for pair in d_alpha_pairs]
            
            # Create scatter plot
            scatter = ax.scatter(d_values, alpha_values, c='blue', s=100, alpha=0.7, 
                               edgecolor='black', zorder=3)
            
            # Add ROI labels
            for i, label in enumerate(roi_labels):
                ax.annotate(label, (d_values[i], alpha_values[i]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold')
            
            # Add reference line for α=1 (normal diffusion)
            ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.7)
            
            # Add shaded regions for different diffusion types
            ax.axhspan(0, 0.9, color='blue', alpha=0.1, label='Subdiffusion')
            ax.axhspan(0.9, 1.1, color='green', alpha=0.1, label='Normal Diffusion')
            ax.axhspan(1.1, 2.0, color='red', alpha=0.1, label='Superdiffusion')
            
            # Add trendline
            if len(d_values) > 1:
                z = np.polyfit(d_values, alpha_values, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(d_values), max(d_values), 100)
                ax.plot(x_trend, p(x_trend), "r--", linewidth=2,
                       label=f'Trend: α = {z[0]:.3f}*D + {z[1]:.3f}')
                
                # Calculate correlation
                correlation, p_value = stats.pearsonr(d_values, alpha_values)
                ax.text(0.05, 0.05, f'Correlation: {correlation:.3f} (p={p_value:.3f})',
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.7, pad=5))
            
            # Set axis limits with some padding
            ax.set_xlim(min(d_values) * 0.9, max(d_values) * 1.1)
            ax.set_ylim(min(min(alpha_values) * 0.9, 0), max(max(alpha_values) * 1.1, 2.0))
    
    # Set subplot title and labels
    ax.set_title('Diffusion Coefficient vs. Anomalous Exponent', fontsize=12)
    ax.set_xlabel('Diffusion Coefficient (μm²/s)', fontsize=10)
    ax.set_ylabel('Anomalous Diffusion Exponent (α)', fontsize=10)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Diffusion coefficient distribution with bootstrapping (lower right)
    ax = axs[3]
    plt.sca(ax)
    
    # Check if we have bootstrap data
    if 'roi_bootstrap' in advanced_data:
        # Collect mean diffusion coefficients and confidence intervals
        d_means = []
        d_errors_low = []
        d_errors_high = []
        roi_labels = []
        
        for roi_id, results in advanced_data['roi_bootstrap'].items():
            if roi_id != 'unassigned' and 'mean_D' in results and not np.isnan(results['mean_D']):
                d_means.append(results['mean_D'])
                d_errors_low.append(results['mean_D'] - results['ci_low'])
                d_errors_high.append(results['ci_high'] - results['mean_D'])
                roi_labels.append(roi_id.split('-')[0][:8])  # Shortened ROI ID
        
        if d_means:
            # Sort by mean diffusion coefficient
            sorted_indices = np.argsort(d_means)
            d_means = [d_means[i] for i in sorted_indices]
            d_errors_low = [d_errors_low[i] for i in sorted_indices]
            d_errors_high = [d_errors_high[i] for i in sorted_indices]
            roi_labels = [roi_labels[i] for i in sorted_indices]
            
            # Create bar plot with error bars
            x_pos = np.arange(len(d_means))
            ax.bar(x_pos, d_means, yerr=[d_errors_low, d_errors_high], 
                  capsize=5, color='skyblue', edgecolor='black', alpha=0.7)
            
            # Set x-tick labels
            ax.set_xticks(x_pos)
            ax.set_xticklabels(roi_labels, rotation=45, ha='right')
    
    # Set subplot title and labels
    ax.set_title('Diffusion Coefficient with 95% Confidence Intervals', fontsize=12)
    ax.set_xlabel('Region of Interest', fontsize=10)
    ax.set_ylabel('Diffusion Coefficient (μm²/s)', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Combined visualization saved to {output_path}")

def main():
    """
    Main function to generate heatmaps and spatial visualizations.
    """
    # Ask for input paths
    roi_data_file = input("Enter path to ROI trajectory data file (.pkl): ")
    
    # Load ROI data
    roi_data = load_data(roi_data_file)
    if roi_data is None:
        print("Failed to load ROI data. Exiting.")
        return
    
    # Check for advanced analysis data
    advanced_data_file = input("Enter path to advanced diffusion analysis file (.pkl, optional): ")
    advanced_data = None
    if advanced_data_file:
        advanced_data = load_data(advanced_data_file)
    
    # Ask for ROI geometry information
    roi_geometry_file = input("Enter path to ImageJ ROI ZIP file: ")
    img_height = int(input("Enter original image height in pixels: "))
    
    # Load ROI geometry
    from read_roi import read_roi_zip
    rois_info = None
    try:
        rois_info = read_roi_zip(roi_geometry_file)
        
        # Transform coordinates if image height is provided
        if img_height is not None:
            for roi_key in rois_info:
                roi = rois_info[roi_key]
                if 'y' in roi:
                    # Flip Y coordinates
                    roi['y'] = img_height - np.array(roi['y'])
        
        print(f"Loaded {len(rois_info)} ROIs")
    except Exception as e:
        print(f"Error loading ROIs from {roi_geometry_file}: {e}")
        return
    
    # Ask for background image (optional)
    image_file = input("Enter path to background image file (optional): ")
    background_image = None
    if image_file:
        background_image = load_image(image_file)
    
    # Create output directory
    output_dir_name = datetime.now().strftime(OUTPUT_DIR_FORMAT)
    output_base_dir = os.path.dirname(roi_data_file)
    output_dir = os.path.join(output_base_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare trajectory data for heatmaps
    all_trajectories = []
    for roi_id, trajectories in roi_data['roi_trajectories'].items():
        all_trajectories.extend(trajectories)
    
    # Create diffusion coefficient heatmap
    create_diffusion_heatmap(all_trajectories, 
                           os.path.join(output_dir, 'diffusion_heatmap.png'),
                           image=background_image,
                           property_name='D',
                           cmap=DIFFUSION_CMAP,
                           title='Diffusion Coefficient Heatmap')
    
    # Create ROI-based diffusion coefficient heatmap
    create_roi_heatmap(roi_data, rois_info,
                     os.path.join(output_dir, 'roi_diffusion_heatmap.png'),
                     image=background_image,
                     property_name='mean_D',
                     cmap=DIFFUSION_CMAP,
                     title='ROI-based Diffusion Coefficient Heatmap')
    
    # Create trajectory visualization
    create_trajectory_visualization(all_trajectories,
                                  os.path.join(output_dir, 'trajectory_visualization.png'),
                                  image=background_image,
                                  color_by='D',
                                  cmap=DIFFUSION_CMAP,
                                  title='Trajectories Colored by Diffusion Coefficient')
    
    # If advanced analysis data is available, create additional visualizations
    if advanced_data:
        # Check if anomalous diffusion data is available
        if 'roi_anomalous_diffusion' in advanced_data:
            # Prepare trajectories with anomalous diffusion data
            anomalous_trajectories = []
            for roi_id, results in advanced_data['roi_anomalous_diffusion'].items():
                anomalous_trajectories.extend(results['trajectories'])
            
            # Create anomalous diffusion exponent heatmap
            create_diffusion_heatmap(anomalous_trajectories, 
                                   os.path.join(output_dir, 'anomalous_diffusion_heatmap.png'),
                                   image=background_image,
                                   property_name='alpha',
                                   cmap=ALPHA_CMAP,
                                   title='Anomalous Diffusion Exponent Heatmap')
            
            # Create ROI-based anomalous diffusion exponent heatmap
            create_roi_heatmap(advanced_data, rois_info,
                             os.path.join(output_dir, 'roi_anomalous_diffusion_heatmap.png'),
                             image=background_image,
                             property_name='mean_alpha',
                             cmap=ALPHA_CMAP,
                             title='ROI-based Anomalous Diffusion Exponent Heatmap')
            
            # Create trajectory visualization colored by anomalous exponent
            create_trajectory_visualization(anomalous_trajectories,
                                          os.path.join(output_dir, 'trajectory_alpha_visualization.png'),
                                          image=background_image,
                                          color_by='alpha',
                                          cmap=ALPHA_CMAP,
                                          title='Trajectories Colored by Anomalous Diffusion Exponent')
        
        # Create combined visualization
        create_combined_visualization(roi_data, advanced_data, rois_info,
                                   os.path.join(output_dir, 'combined_visualization.png'),
                                   image=background_image)
    
    print(f"\nAll visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()