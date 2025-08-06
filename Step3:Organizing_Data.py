#!/usr/bin/env python3
"""
Step-3  150-frame sliding windows + per-driver Z-score
FrameFeatures/  →  final 3-D tensor (windows × 150 × 4)
"""

import os, numpy as np, pandas as pd
from pathlib import Path

DATA_DIR   = "FrameFeatures"          # output of Step2
OUT_DIR    = "Windows"                # will create BAC folders here
WIN        = 150
STRIDE     = 1                        # 150-frame sliding window
BAC_MAP    = {"0":0, "5":1, "10":2}  # label for LSTM

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------ HELPERS ------------------------
def make_windows(df):
    """Return (N,150,4) np array"""
    vals = df[["head","gaze","EAR","P_scale"]].values
    N = len(vals) - WIN + 1
    if N <= 0: return None
    windows = np.stack([vals[i:i+WIN] for i in range(N)])
    return windows.astype(np.float32)

def z_score_driver(driver_frames):
    """Use first 1/3 of sober frames to compute μ & σ, then normalize whole driver"""
    sober = driver_frames[driver_frames["BAC"]==0]
    ref = sober.iloc[:len(sober)//3]      # first 1/3 sober
    mu  = ref[["head","gaze","EAR","P_scale"]].mean()
    std = ref[["head","gaze","EAR","P_scale"]].std().replace(0,1)
    driver_frames[["head","gaze","EAR","P_scale"]] = (
        driver_frames[["head","gaze","EAR","P_scale"]] - mu) / std
    return driver_frames

# ------------------------ MAIN PIPELINE ------------------------
# Build one big table: cols=[head,gaze,EAR,P_scale,timestamp,driver,BAC]
all_frames = []
for csv in Path(DATA_DIR).glob("*.csv"):
    # Expected naming: <driver>_0.csv, <driver>_5.csv, <driver>_10.csv
    driver, bac_str = csv.stem.rsplit("_",1)
    bac = int(bac_str)
    df = pd.read_csv(csv)
    df["driver"] = driver
    df["BAC"]    = bac
    all_frames.append(df)
df_all = pd.concat(all_frames, ignore_index=True)

# ---- Per-driver Z-score ----
df_all = (df_all.groupby("driver", group_keys=False)
                .apply(z_score_driver)
                .reset_index(drop=True))

# ---- Sliding windows per (driver,BAC) ----
for (driver, bac), sub in df_all.groupby(["driver","BAC"]):
    wins = make_windows(sub)
    if wins is None: continue
    label = BAC_MAP[str(bac)]
    save_dir = Path(OUT_DIR) / str(bac)
    save_dir.mkdir(exist_ok=True)
    np.save(save_dir / f"{driver}_{bac}.npy", wins)
    # also save a tiny label file
    np.save(save_dir / f"{driver}_{bac}_label.npy",
            np.full(len(wins), label, dtype=np.int8))

print("Window extraction & normalization done.")