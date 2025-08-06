#!/usr/bin/env python3
"""
Step-2  Frame-level feature extraction
OpenFace CSV  →  frame-level metrics  →  cleaned CSV
Usage:  python Step2_BatchProcessing.py <in_dir> <out_dir>
"""

import os, sys, glob, numpy as np, pandas as pd
from pathlib import Path

IN_DIR  = sys.argv[1] if len(sys.argv) > 1 else "OpenFace_csvs"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "FrameFeatures"

# --- OpenFace columns we actually need (plugin outputs) ---
NEED_COLS = ["frame", "timestamp",
             "pose_Rx", "pose_Ry", "pose_Rz",
             "gaze_angle_x", "gaze_angle_y",
             "eye_lmk_EAR_avg",  # OpenFace plugin
             "p_scale"]          # OpenFace plugin

# ----------------------------------------------------------
def frame_metrics(row):
    """Compute 4 paper features for one frame"""
    head = np.sqrt(row["pose_Rx"]**2 + row["pose_Ry"]**2 + row["pose_Rz"]**2)
    gaze = np.sqrt(row["gaze_angle_x"]**2 + row["gaze_angle_y"]**2)
    ear  = row["eye_lmk_EAR_avg"]
    psc  = row["p_scale"]
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