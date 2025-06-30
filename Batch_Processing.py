#!/usr/bin/env python3
"""
Process OpenFace output CSVs to add Eye Aspect Ratio (EAR) and Pupil Scale metrics
Usage: python3 enhance_openface.py /path/to/openface_csvs/ /output/path/
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def calculate_ear(row):
    """Calculate Eye Aspect Ratio from facial landmarks"""
    def eye_aspect_ratio(eye_points):
        # Vertical distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        # Horizontal distance
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        return (A + B) / (2.0 * C)
    
    # Left eye landmarks (indices 36-41 in OpenFace)
    left_eye = np.array([
        [row[' x_36'], row[' y_36']],
        [row[' x_37'], row[' y_37']],
        [row[' x_38'], row[' y_38']],
        [row[' x_39'], row[' y_39']],
        [row[' x_40'], row[' y_40']],
        [row[' x_41'], row[' y_41']]
    ])
    
    # Right eye landmarks (indices 42-47)
    right_eye = np.array([
        [row[' x_42'], row[' y_42']],
        [row[' x_43'], row[' y_43']],
        [row[' x_44'], row[' y_44']],
        [row[' x_45'], row[' y_45']],
        [row[' x_46'], row[' y_46']],
        [row[' x_47'], row[' y_47']]
    ])
    
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

def calculate_pupil_scale(row):
    """Estimate pupil size from landmark distances"""
    try:
        # Pupil landmarks (left=68, right=81 in OpenFace)
        left_pupil = np.array([row[' x_68'], row[' y_68']])
        right_pupil = np.array([row[' x_81'], row[' y_81']])
        return np.linalg.norm(left_pupil - right_pupil)
    except:
        return 0.0  # Return 0 if landmarks missing

def process_csv(input_csv, output_dir):
    """Process a single CSV file"""
    df = pd.read_csv(input_csv)
    
    # Add custom metrics
    df['EAR'] = df.apply(calculate_ear, axis=1)
    df['Pupil_Scale'] = df.apply(calculate_pupil_scale, axis=1)
    
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
    if len(sys.argv) != 3:
        print("Usage: python3 enhance_openface.py <input_dir> <output_dir>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])