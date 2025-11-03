# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 11:52:30 2025

@author: wanglab-PC-2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_
roi_attributes_inspector.py

This script examines ROI files to determine available attributes and structure
from different ROI loading libraries (read_roi and roifile).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import inspect
import read_roi
import roifile

# Global variables - can be edited
DEFAULT_ROI_FILE = r"C:\Users\wanglab-PC-2\Desktop\GEM-ER\KDEL GEM\New folder\ND2 files\Nd2 files\Preblastoderm\Nd2\Membrnae one\PB_em1.nd2_membrane__rois.zip"
OUTPUT_DIR = os.path.dirname(DEFAULT_ROI_FILE) if os.path.exists(DEFAULT_ROI_FILE) else "."
SAVE_INSPECTION = True

def inspect_roi_object(roi_obj, name="Unknown"):
    """
    Thoroughly inspect an ROI object and return its attributes and methods
    """
    print(f"\n{'='*50}")
    print(f"Inspecting ROI: {name}")
    print(f"{'='*50}")
    
    print(f"Type: {type(roi_obj)}")
    
    # Try different approaches to get attributes
    attributes = {}
    
    # Method 1: dir()
    print("\nAll attributes and methods:")
    attr_list = dir(roi_obj)
    for attr in attr_list:
        if not attr.startswith('__'):
            try:
                value = getattr(roi_obj, attr)
                if not callable(value):
                    attributes[attr] = value
                    print(f"  {attr}: {value}")
                else:
                    print(f"  {attr}: <method>")
            except Exception as e:
                print(f"  {attr}: <error accessing: {e}>")
    
    # Method 2: For dictionaries
    if isinstance(roi_obj, dict):
        print("\nDictionary keys:")
        for key, value in roi_obj.items():
            print(f"  {key}: {type(value)}")
            if isinstance(value, (list, np.ndarray)) and len(value) > 5:
                print(f"    {type(value)} with {len(value)} items")
                print(f"    First 5 items: {value[:5]}")
            else:
                print(f"    Value: {value}")
    
    return attributes

def inspect_read_roi_file(roi_file):
    """
    Inspect ROIs loaded with read_roi library
    """
    print("\n\nINSPECTING READ_ROI LIBRARY")
    print("-" * 50)
    
    try:
        # Load ROIs
        if roi_file.lower().endswith('.zip'):
            rois = read_roi.read_roi_zip(roi_file)
        else:
            rois = read_roi.read_roi_file(roi_file)
        
        print(f"Successfully loaded {len(rois)} ROIs with read_roi")
        
        # Inspect the structure
        print("\nOverall structure:")
        print(f"Type: {type(rois)}")
        
        # Inspect first ROI
        if rois:
            first_roi_name = next(iter(rois))
            first_roi = rois[first_roi_name]
            inspect_roi_object(first_roi, first_roi_name)
            
            # Get a list of common attributes across all ROIs
            common_attrs = set(first_roi.keys())
            for roi_name, roi in list(rois.items())[1:]:
                common_attrs &= set(roi.keys())
            
            print("\nCommon attributes across all ROIs:")
            for attr in common_attrs:
                print(f"  {attr}")
        
        return rois
    except Exception as e:
        print(f"Error inspecting with read_roi: {e}")
        return None

def inspect_roifile_object(roi_file):
    """
    Inspect ROIs loaded with roifile library
    """
    print("\n\nINSPECTING ROIFILE LIBRARY")
    print("-" * 50)
    
    try:
        # Check roifile version
        print(f"Roifile version: {roifile.__version__ if hasattr(roifile, '__version__') else 'Unknown'}")
        
        # Load ROIs
        if roi_file.lower().endswith('.zip'):
            rois = roifile.roiread(roi_file)
            print(f"Successfully loaded {len(rois)} ROIs with roifile")
            
            # Inspect the structure
            print("\nOverall structure:")
            print(f"Type: {type(rois)}")
            
            # Inspect first ROI
            if rois and len(rois) > 0:
                first_roi = rois[0]
                inspect_roi_object(first_roi, f"ROI_{0}")
                
                # Get coordinates
                try:
                    coords = first_roi.coordinates()
                    print("\nCoordinates:")
                    print(f"Type: {type(coords)}")
                    print(f"Shape: {coords.shape if hasattr(coords, 'shape') else 'No shape'}")
                    if hasattr(coords, 'shape') and coords.shape[0] > 0:
                        print(f"First 5 points: {coords[:5]}")
                except Exception as e:
                    print(f"Error getting coordinates: {e}")
                
                # Try to access specific attributes
                attrs_to_check = ['top', 'left', 'bottom', 'right', 'width', 'height', 'position', 'name', 'roitype']
                print("\nChecking specific attributes:")
                for attr in attrs_to_check:
                    try:
                        if hasattr(first_roi, attr):
                            value = getattr(first_roi, attr)
                            print(f"  {attr}: {value}")
                        else:
                            print(f"  {attr}: <not present>")
                    except Exception as e:
                        print(f"  {attr}: <error: {e}>")
                
                # Check properties and methods
                print("\nMethods and properties:")
                methods_to_check = ['coordinates', 'contains', 'frompoints', 'plot']
                for method in methods_to_check:
                    try:
                        if hasattr(first_roi, method):
                            print(f"  {method}: <available>")
                        else:
                            print(f"  {method}: <not present>")
                    except Exception as e:
                        print(f"  {method}: <error: {e}>")
            
            return rois
        else:
            roi = roifile.ImagejRoi.fromfile(roi_file)
            print(f"Successfully loaded 1 ROI with roifile")
            inspect_roi_object(roi, "Single ROI")
            return [roi]
    except Exception as e:
        print(f"Error inspecting with roifile: {e}")
        return None

def save_inspection_report(output_file, roi_file):
    """
    Redirect stdout to a file to save the inspection report
    """
    # Save original stdout
    original_stdout = sys.stdout
    
    try:
        # Redirect stdout to file
        with open(output_file, 'w') as f:
            sys.stdout = f
            
            # Run the inspections
            print(f"ROI Attributes Inspection Report")
            print(f"File: {roi_file}")
            print(f"Date: {__import__('datetime').datetime.now()}")
            print("\n" + "="*80 + "\n")
            
            inspect_read_roi_file(roi_file)
            inspect_roifile_object(roi_file)
    finally:
        # Restore stdout
        sys.stdout = original_stdout
        print(f"Inspection report saved to: {output_file}")

def main():
    # Check if file path was provided as command line argument
    if len(sys.argv) > 1:
        roi_file = sys.argv[1]
        print(f"Using command line argument for ROI file: {roi_file}")
    else:
        # Use default path
        roi_file = DEFAULT_ROI_FILE
        print(f"Using default ROI file: {roi_file}")
    
    if not os.path.exists(roi_file):
        print(f"Error: ROI file not found: {roi_file}")
        return
    
    # Setup output file
    base_name = os.path.splitext(os.path.basename(roi_file))[0]
    output_file = os.path.join(OUTPUT_DIR, f"{base_name}_roi_inspection.txt")
    
    # Save inspection to file if requested
    if SAVE_INSPECTION:
        save_inspection_report(output_file, roi_file)
        print(f"Inspection report saved to: {output_file}")
    else:
        # Run inspections directly to console
        inspect_read_roi_file(roi_file)
        inspect_roifile_object(roi_file)

if __name__ == "__main__":
    main()