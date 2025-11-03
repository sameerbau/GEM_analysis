import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import read_roi

def load_rois(roi_file):
    """
    Load ROIs from ImageJ ROI file using read_roi library with fallback options.
    """
    try:
        # Attempt to load ROIs using read_roi library
        if roi_file.lower().endswith('.zip'):
            rois = read_roi.read_roi_zip(roi_file)
        else:
            rois = read_roi.read_roi_file(roi_file)
        print(f"Loaded {len(rois)} ROIs using read_roi library.")
        return rois
    except Exception as e:
        print(f"Error loading ROIs using read_roi library: {e}")
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

def main():
    # Ask the user for the path to the ROI file
    roi_file = input("Enter the path to the ImageJ ROI file (ZIP format): ")

    # Load the ROIs
    rois = load_rois(roi_file)

    if rois:
        # Set the output path to save the graph in the same folder as the ROI file
        output_path = os.path.join(os.path.dirname(roi_file), 'rois_visualization.png')
        
        # Visualize the ROIs and save the figure
        visualize_rois(rois, title="ROI Visualization", save_path=output_path)
    else:
        print("Failed to load ROIs. Please check the ROI file.")

if __name__ == '__main__':
    main()