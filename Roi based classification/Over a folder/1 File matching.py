# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 13:43:01 2025

@author: wanglab-PC-2
"""
import os

def match_files(roi_zip_folder, process_traj_folder, analysed_traj_folder):
    matched_files = []
    
    for file in os.listdir(roi_zip_folder):
        if file.endswith("_rois.zip"):
            base_name = file.rsplit("_", 3)[0]
            roi_zip = os.path.join(roi_zip_folder, file)
            process_traj_pkl = os.path.join(process_traj_folder, f"tracked_Traj_{base_name}_crop.pkl")
            analysed_traj_pkl = os.path.join(analysed_traj_folder, f"analyzed_tracked_Traj_{base_name}_crop.pkl")
            
            if os.path.exists(process_traj_pkl) and os.path.exists(analysed_traj_pkl):
                matched_files.append((roi_zip, process_traj_pkl, analysed_traj_pkl))
            else:
                print(f"Warning: Corresponding pkl files not found for {file}")
    
    return matched_files

def main():
    roi_zip_folder = input("Enter the path to the ROI zip folder: ")
    process_traj_folder = input("Enter the path to the processed trajectory pkl folder: ")
    analysed_traj_folder = input("Enter the path to the analyzed trajectory pkl folder: ")
    
    matched_files = match_files(roi_zip_folder, process_traj_folder, analysed_traj_folder)
    
    for roi_zip, process_traj_pkl, analysed_traj_pkl in matched_files:
        print("Matched files:")
        print(roi_zip)
        print(process_traj_pkl)
        print(analysed_traj_pkl)
        print()

if __name__ == "__main__":
    main()