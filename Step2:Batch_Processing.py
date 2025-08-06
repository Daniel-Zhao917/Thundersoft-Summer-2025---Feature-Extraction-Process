#!/usr/bin/env python3
"""
Step-2  Frame-level feature extraction
OpenFace CSV  →  frame-level metrics  →  cleaned CSV
Usage:  python Step2_BatchProcessing.py <in_dir> <out_dir>
"""

import os, sys, glob, numpy as np, pandas as pd
from pathlib import Path

IN_DIR  = '/Users/zhaoda/Desktop/8:5 test out new feature extraction/output'
OUT_DIR = '/Users/zhaoda/Desktop/8:5 test out new feature extraction/Step 2 csvs'

# --- OpenFace columns we actually need (plugin outputs) ---
NEED_COLS = ["frame", "timestamp",
             "pose_Rx", "pose_Ry", "pose_Rz",
             "gaze_angle_x", "gaze_angle_y",
             "p_scale"] + \
            [f"eye_lmk_{xy}_{i}" for xy in ["x", "y"] for i in range(0, 55)]

# ---------- EAR helper ----------
def eye_aspect_ratio(pts):
    """Compute EAR from 6 eye landmarks (OpenFace order)."""
    A = np.linalg.norm(pts[1] - pts[5])  # vertical 1
    B = np.linalg.norm(pts[2] - pts[4])  # vertical 2
    C = np.linalg.norm(pts[0] - pts[3])  # horizontal
    return (A + B) / (2.0 * C)

def compute_ear(row):
    """Manual EAR using corrected OpenFace indices."""
    left = np.array([
        [row['eye_lmk_x_8'], row['eye_lmk_y_8']],  # p1
        [row['eye_lmk_x_10'], row['eye_lmk_y_10']],  # p2
        [row['eye_lmk_x_12'], row['eye_lmk_y_12']],  # p3
        [row['eye_lmk_x_14'], row['eye_lmk_y_14']],  # p4
        [row['eye_lmk_x_16'], row['eye_lmk_y_16']],  # p5
        [row['eye_lmk_x_18'], row['eye_lmk_y_18']]   # p6
    ])

    right = np.array([
        [row['eye_lmk_x_36'], row['eye_lmk_y_36']],  # p1
        [row['eye_lmk_x_38'], row['eye_lmk_y_38']],  # p2
        [row['eye_lmk_x_40'], row['eye_lmk_y_40']],  # p3
        [row['eye_lmk_x_42'], row['eye_lmk_y_42']],  # p4
        [row['eye_lmk_x_44'], row['eye_lmk_y_44']],  # p5
        [row['eye_lmk_x_46'], row['eye_lmk_y_46']]   # p6
    ])

    return (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
# ---------- 2.  frame_metrics ----------
def frame_metrics(row):
    head = np.sqrt(row["pose_Rx"]**2 + row["pose_Ry"]**2 + row["pose_Rz"]**2)
    gaze = np.sqrt(row["gaze_angle_x"]**2 + row["gaze_angle_y"]**2)
    psc  = row["p_scale"]

    # prefer plugin, fall back to manual EAR
    if "eye_lmk_EAR_avg" in row.index:
        ear = row["eye_lmk_EAR_avg"]
    else:
        ear = compute_ear(row)

    return {"head": head, "gaze": gaze, "EAR": ear, "P_scale": psc}

def process_one(in_csv, out_dir):
    df = pd.read_csv(in_csv, usecols=lambda c: c in NEED_COLS)
    # No filtering beyond basic sanity
    feat = pd.DataFrame([frame_metrics(row) for _, row in df.iterrows()])
    feat.insert(0, "timestamp", df["timestamp"])
    out_file = Path(out_dir) / Path(in_csv).name
    feat.to_csv(out_file, index=False)
    print("Saved", out_file)

# ----------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
for csv_path in glob.glob(os.path.join(IN_DIR, "*.csv")):
    process_one(csv_path, OUT_DIR)