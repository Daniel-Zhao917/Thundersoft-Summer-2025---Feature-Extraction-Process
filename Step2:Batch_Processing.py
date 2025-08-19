#!/usr/bin/env python3
"""
Step-2  Frame-level feature extraction
OpenFace CSV  →  frame-level metrics  →  cleaned CSV
Usage:  python Step2_BatchProcessing.py <in_dir> <out_dir>
"""

import os, sys, glob, numpy as np, pandas as pd, math
from pathlib import Path

IN_DIR  = '/Users/zhaoda/Desktop/ThundersoftSummer2025/Data/RawCSVs'
OUT_DIR = '/Users/zhaoda/Desktop/8:19 step 2 CSVs (w:AUs 1-9)'

# --- OpenFace columns we actually need (plugin outputs) ---
BASE_COLS = ["frame", "timestamp", "confidence",
             "pose_Rx", "pose_Ry",
             "gaze_angle_x", "gaze_angle_y",
             "p_scale"] + \
            [f"{xy}_{i}" for xy in ["x", "y"] for i in range(0, 55)]

# Add Action Units (AUs) columns
AU_COLS = [f"AU{au:02d}_r" for au in [1,2,4,5,6,7,9]]
NEED_COLS = BASE_COLS + AU_COLS

# ---------- EAR helper ----------
def eye_aspect_ratio(pts):
    """Compute EAR from 6 eye landmarks (OpenFace order)."""
    A = np.linalg.norm(pts[1] - pts[5])  # vertical 1
    B = np.linalg.norm(pts[2] - pts[4])  # vertical 2
    C = np.linalg.norm(pts[0] - pts[3])  # horizontal
    return (A + B) / (2.0 * C)

def compute_ear(row):
    """Manual EAR using OpenFace facial landmarks (68-point format)"""
    # Left eye landmarks (indices 36-41)
    left = np.array([
        [row['x_36'], row['y_36']],  # p1
        [row['x_37'], row['y_37']],  # p2
        [row['x_38'], row['y_38']],  # p3
        [row['x_39'], row['y_39']],  # p4
        [row['x_40'], row['y_40']],  # p5
        [row['x_41'], row['y_41']]   # p6
    ])

    # Right eye landmarks (indices 42-47)
    right = np.array([
        [row['x_42'], row['y_42']],  # p1
        [row['x_43'], row['y_43']],  # p2
        [row['x_44'], row['y_44']],  # p3
        [row['x_45'], row['y_45']],  # p4
        [row['x_46'], row['y_46']],  # p5
        [row['x_47'], row['y_47']]   # p6
    ])

    return (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0

# ---------- 2.  frame_metrics ----------
def degree(int):
    return (int / math.pi * 180)

def absolute_value(int):
    result = int
    if (result < 0):
        result = -result
    return(result)

def frame_metrics(row):
    # radian to degree and absolute value
    for attribute in ['pose_Rx', 'pose_Ry', 'gaze_angle_x', 'gaze_angle_y']:
        row[attribute] = absolute_value(degree(row[attribute]))
    
    # square every attribute except ear 
    for attribute in ['pose_Rx', 'pose_Ry', 'gaze_angle_x', 'gaze_angle_y', 'p_scale']:
        row[attribute] = pow(row[attribute], 2)
  
    head_r = np.sqrt(pow(row['pose_Rx'], 2) + pow(row['pose_Ry'], 2))
    head_theta = np.arctan2(row['pose_Ry'], row['pose_Rx'])

    gaze_r = np.sqrt(pow(row['gaze_angle_x'], 2) + pow(row['gaze_angle_y'], 2))
    gaze_theta = np.arctan2(row['gaze_angle_y'], row['gaze_angle_x'])

    # prefer plugin, fall back to manual EAR
    if "eye_lmk_EAR_avg" in row.index:
        ear = row["eye_lmk_EAR_avg"]
    else:
        ear = compute_ear(row)

    # because ear could < 1, but always greater than 0.05, thus first time ear by 20, then square
    ear = pow(ear * 20, 2)
  
    # Extract Action Units
    au_features = {au: row[au] for au in AU_COLS if au in row.index}

    return {
        "frame": row["frame"], 
        "timestamp": row["timestamp"], 
        "confidence": row["confidence"], 
        "head_r": head_r, 
        "head_theta": head_theta, 
        "pose_Rx": row["pose_Rx"], 
        "pose_Ry": row["pose_Ry"], 
        "gaze_angle_x": row["gaze_angle_x"], 
        "gaze_angle_y": row["gaze_angle_y"], 
        "gaze_r": gaze_r, 
        "gaze_theta": gaze_theta, 
        "EAR": ear, 
        "P_scale": row["p_scale"],
        **au_features  # Include all AU features
    }

def process_one(in_csv, out_dir):
    df = pd.read_csv(in_csv, usecols=lambda c: c in NEED_COLS)
    feat = pd.DataFrame([frame_metrics(row) for _, row in df.iterrows()])
    # filters the rows where confidence is lower than 0.3
    feat = feat[feat["confidence"] >= 0.3]
    out_file = Path(out_dir) / Path(in_csv).name
    feat.to_csv(out_file, index=False)
    print("Saved", out_file)

# ----------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
for csv_path in glob.glob(os.path.join(IN_DIR, "*.csv")):
    process_one(csv_path, OUT_DIR)
