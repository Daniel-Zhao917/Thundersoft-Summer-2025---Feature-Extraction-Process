#!/usr/bin/env python3
"""
Process OpenFace output CSVs to add Eye Aspect Ratio (EAR) and Pupil Scale metrics
Usage: python3 Batch_Processing.py /path/to/openface_csvs/ /output/path/
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path

input_path = "/Users/zhaoda/Desktop/ThundersoftSummer2025/Thundersoft-Summer-2025---Feature-Extraction-Process/Processed_features"
output_path = "/Users/zhaoda/Desktop/ThundersoftSummer2025/Thundersoft-Summer-2025---Feature-Extraction-Process/Step2Deliverables"


cols = [
    "frame",
    "timestamp",
    'eye_lmk_x_36', 'eye_lmk_y_36', 'eye_lmk_x_42', 'eye_lmk_y_42', 'eye_lmk_x_38', 'eye_lmk_y_38', 'eye_lmk_x_40', 'eye_lmk_y_40', 'eye_lmk_x_46', 'eye_lmk_y_46', 'eye_lmk_x_44', 'eye_lmk_y_44',
    'eye_lmk_x_8', 'eye_lmk_y_8', 'eye_lmk_x_14', 'eye_lmk_y_14', 'eye_lmk_x_10', 'eye_lmk_y_10', 'eye_lmk_x_12', 'eye_lmk_y_12', 'eye_lmk_x_18', 'eye_lmk_y_18', 'eye_lmk_x_16', 'eye_lmk_y_16',
    'eye_lmk_x_51', 'eye_lmk_y_51', 'eye_lmk_x_55', 'eye_lmk_y_55', 
    'eye_lmk_x_23', 'eye_lmk_y_23', 'eye_lmk_x_27', 'eye_lmk_y_27', 
    "gaze_angle_x", "gaze_angle_y",
    "pose_Rx", "pose_Ry", "pose_Rz"

]

'''
"AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", 
"AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", 
"AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
"AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c",
"AU09_c", "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c",
"AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c",  
''' 


undesired_cols = [
    'eye_lmk_x_36', 'eye_lmk_y_36', 'eye_lmk_x_42', 'eye_lmk_y_42', 'eye_lmk_x_38', 'eye_lmk_y_38', 'eye_lmk_x_40', 'eye_lmk_y_40', 'eye_lmk_x_46', 'eye_lmk_y_46', 'eye_lmk_x_44', 'eye_lmk_y_44',
    'eye_lmk_x_8', 'eye_lmk_y_8', 'eye_lmk_x_14', 'eye_lmk_y_14', 'eye_lmk_x_10', 'eye_lmk_y_10', 'eye_lmk_x_12', 'eye_lmk_y_12', 'eye_lmk_x_18', 'eye_lmk_y_18', 'eye_lmk_x_16', 'eye_lmk_y_16',
    'eye_lmk_x_51', 'eye_lmk_y_51', 'eye_lmk_x_55', 'eye_lmk_y_55', 
    'eye_lmk_x_23', 'eye_lmk_y_23', 'eye_lmk_x_27', 'eye_lmk_y_27'

]

def calculate_ear(row):
    """Calculate Eye Aspect Ratio from facial landmarks"""
    def eye_aspect_ratio(eye_points):
        # Vertical distances
        A = np.linalg.norm(eye_points[3] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        # Horizontal distance
        C = np.linalg.norm(eye_points[0] - eye_points[1])
        return (A + B) / (2.0 * C)
    
    # Left eye landmarks
    left_eye = np.array([
        [row['eye_lmk_x_36'], row['eye_lmk_y_36']], # p1 
        [row['eye_lmk_x_42'], row['eye_lmk_y_42']], # p4
        [row['eye_lmk_x_38'], row['eye_lmk_y_38']], # p2
        [row['eye_lmk_x_40'], row['eye_lmk_y_40']], # p3
        [row['eye_lmk_x_46'], row['eye_lmk_y_46']], # p6
        [row['eye_lmk_x_44'], row['eye_lmk_y_44']]  # p5
    ])
    
    # Right eye landmarks
    right_eye = np.array([
        [row['eye_lmk_x_8'],  row['eye_lmk_y_8']],  # p1
        [row['eye_lmk_x_14'], row['eye_lmk_y_14']], # p4
        [row['eye_lmk_x_10'], row['eye_lmk_y_10']], # p2
        [row['eye_lmk_x_12'], row['eye_lmk_y_12']], # p3
        [row['eye_lmk_x_18'], row['eye_lmk_y_18']], # p6
        [row['eye_lmk_x_16'], row['eye_lmk_y_16']]  # p5
    ])
    
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

def calculate_pupil_scale(row, side):
    try:
        if side == "left":
            dx = row['eye_lmk_x_51'] - row['eye_lmk_x_55']
            dy = row['eye_lmk_y_51'] - row['eye_lmk_y_55']
        elif side == "right":
            dx = row['eye_lmk_x_23'] - row['eye_lmk_x_27']
            dy = row['eye_lmk_y_23'] - row['eye_lmk_y_27']
        return np.sqrt(dx**2 + dy**2)  # Euclidean distance
    except:
        return 0.0  # Return 0 if landmarks missing

def process_csv(input_csv, output_dir):
    """Process a single CSV file"""
    df = pd.read_csv(input_csv, usecols=cols)
    
    # Add custom metrics
    df['EAR'] = df.apply(calculate_ear, axis=1)
    df['Pupil_Scale (Left)'] = df.apply(calculate_pupil_scale, axis=1, args= ("left",))
    df['Pupil_Scale (Right)'] = df.apply(calculate_pupil_scale, axis=1, args=("right",))

    df = df.drop(columns = undesired_cols)
    
    # Save enhanced CSV
    out_path = Path(output_dir) / Path(input_csv).name
    df.to_csv(out_path, index=False)
    print(f"Processed: {input_csv} -> {out_path}")

def main(input_dir, output_dir):
    """Batch process all CSVs in directory"""
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all CSV files
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        sys.exit(1)
    
    print(f"Processing {len(csv_files)} files...")
    for csv_file in csv_files:
        process_csv(csv_file, output_dir)
    
    print("\nProcessing complete! Enhanced CSVs saved to:", output_dir)

if __name__ == "__main__":
    #if len(sys.argv) != 3:
    #    print("Usage: python3 Batch_Processing.py <input_dir> <output_dir>")
    #    sys.exit(1)
    
    main(input_path, output_path)