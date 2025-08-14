#!/usr/bin/env python3
"""
Step-2  Frame-level feature extraction
OpenFace CSV  →  frame-level metrics  →  cleaned CSV
Usage:  python Step2_BatchProcessing.py <in_dir> <out_dir>
"""

import os, sys, glob, numpy as np, pandas as pd
from pathlib import Path

IN_DIR  = '/Users/zhaoda/Desktop/8:6 RawCSVs'
OUT_DIR = '/Users/zhaoda/Desktop/step 2 csvs'

# --- OpenFace columns we actually need (plugin outputs) ---
NEED_COLS = ["frame", "timestamp",
             "pose_Rx", "pose_Ry",
             "gaze_angle_x", "gaze_angle_y",
             "p_scale"] + \
            [f"{xy}_{i}" for xy in ["x", "y"] for i in range(0, 55)]

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
def frame_metrics(row):
    psc  = row["p_scale"]

    # prefer plugin, fall back to manual EAR
    if "eye_lmk_EAR_avg" in row.index:
        ear = row["eye_lmk_EAR_avg"]
    else:
        ear = compute_ear(row)

    return {"frame": row["frame"], "timestamp": row["timestamp"], "pose_Rx": row["pose_Rx"], "pose_Ry": row["pose_Ry"], "gaze_angle_x": row["gaze_angle_x"], "gaze_angle_y": row["gaze_angle_y"], "EAR": ear, "P_scale": psc}

def process_one(in_csv, out_dir):
    df = pd.read_csv(in_csv, usecols=lambda c: c in NEED_COLS)
    # No filtering beyond basic sanity
    feat = pd.DataFrame([frame_metrics(row) for _, row in df.iterrows()])
    out_file = Path(out_dir) / Path(in_csv).name
    feat.to_csv(out_file, index=False)
    print("Saved", out_file)

# ----------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
for csv_path in glob.glob(os.path.join(IN_DIR, "*.csv")):
    process_one(csv_path, OUT_DIR)
