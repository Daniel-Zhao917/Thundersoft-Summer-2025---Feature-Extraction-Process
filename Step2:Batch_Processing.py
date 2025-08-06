#!/usr/bin/env python3
"""
Step-2  Frame-level feature extraction
OpenFace CSV  →  frame-level metrics  →  cleaned CSV
Usage:  python Step2_BatchProcessing.py <in_dir> <out_dir>
"""

import os, sys, glob, numpy as np, pandas as pd
from pathlib import Path

# ------------------------ CONFIG ------------------------
IN_DIR  = sys.argv[1] if len(sys.argv) > 1 else "OpenFace_csvs"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "FrameFeatures"

# We only need the columns that end up becoming the 4 signals
NEED_COLS = ["frame","timestamp","confidence",
             # --- head pose (degrees) ---
             "pose_Rx","pose_Ry","pose_Rz",
             # --- gaze (radians) ---
             "gaze_angle_x","gaze_angle_y",
             # --- eye landmarks for EAR ---
             "eye_lmk_x_36","eye_lmk_y_36","eye_lmk_x_37","eye_lmk_y_37",
             "eye_lmk_x_38","eye_lmk_y_38","eye_lmk_x_39","eye_lmk_y_39",
             "eye_lmk_x_40","eye_lmk_y_40","eye_lmk_x_41","eye_lmk_y_41",
             "eye_lmk_x_42","eye_lmk_y_42","eye_lmk_x_43","eye_lmk_y_43",
             "eye_lmk_x_44","eye_lmk_y_44","eye_lmk_x_45","eye_lmk_y_45",
             "eye_lmk_x_46","eye_lmk_y_46","eye_lmk_x_47","eye_lmk_y_47",
             # --- PDM scale (OpenFace shape parameter 0) ---
             "p_scale"]

# ------------------------ HELPERS ------------------------
def eye_aspect_ratio(pts):
    A = np.linalg.norm(pts[1]-pts[5])  # vertical
    B = np.linalg.norm(pts[2]-pts[4])  # vertical
    C = np.linalg.norm(pts[0]-pts[3])  # horizontal
    return (A+B)/(2.0*C)

def frame_metrics(row):
    """Return dict with 4 frame-level signals"""
    # 1. Head movement magnitude (degrees)
    head = np.sqrt(row["pose_Rx"]**2 + row["pose_Ry"]**2 + row["pose_Rz"]**2)
    # 2. Gaze movement magnitude (radians)
    gaze = np.sqrt(row["gaze_angle_x"]**2 + row["gaze_angle_y"]**2)
    # 3. Average EAR across both eyes
    left  = row.loc[[f"eye_lmk_x_{i}" for i in range(36,42)] +
                    [f"eye_lmk_y_{i}" for i in range(36,42)]].values.reshape(6,2)
    right = row.loc[[f"eye_lmk_x_{i}" for i in range(42,48)] +
                    [f"eye_lmk_y_{i}" for i in range(42,48)]].values.reshape(6,2)
    ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
    # 4. PDM scale (OpenFace already gives it)
    p_scale = row["p_scale"]
    return {"head": head, "gaze": gaze, "EAR": ear, "P_scale": p_scale}

def process_one(in_csv, out_dir):
    df = pd.read_csv(in_csv, usecols=lambda c: c in NEED_COLS)
    # Basic sanity filter
    df = df[df["confidence"] > 0.75].reset_index(drop=True)
    # Compute 4 signals
    feat = pd.DataFrame([frame_metrics(row) for _, row in df.iterrows()])
    # Keep timestamp for alignment later
    feat.insert(0, "timestamp", df["timestamp"])
    out_file = Path(out_dir) / Path(in_csv).name
    feat.to_csv(out_file, index=False)
    print("Saved", out_file)

# ------------------------ MAIN ------------------------
os.makedirs(OUT_DIR, exist_ok=True)
for csv_path in glob.glob(os.path.join(IN_DIR, "*.csv")):
    process_one(csv_path, OUT_DIR)