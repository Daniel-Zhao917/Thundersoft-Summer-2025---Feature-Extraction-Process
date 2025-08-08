#!/usr/bin/env python3
"""
Step-3  150-frame sliding windows + per-driver Z-score
FrameFeatures/  →  (N_windows, 4) tensor per driver-BAC
"""

import os, numpy as np, pandas as pd
from pathlib import Path

DATA_DIR = "/Users/zhaoda/Desktop/8:5 test out new feature extraction/Step 2 csvs"
OUT_DIR  = "/Users/zhaoda/Desktop/8:5 test out new feature extraction/windows"
WIN      = 150
STRIDE   = 1
BAC_MAP  = {"0":0, "5":1, "10":2}

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------
def make_windows(df):
    """Return (N,4) array where each row = mean over 150-frame window"""
    vals = df[["head","gaze","EAR","P_scale"]]
    N = len(vals) - WIN + 1
    if N <= 0:
        return None
    # collapse each window to its mean
    means = np.array([vals[i:i+WIN].mean(axis=0) for i in range(0, N, STRIDE)])
    return means.astype(np.float32)

def z_score_driver(driver_frames):
    """Use first 1/3 of sober frames to compute μ & σ, then normalize whole driver"""
    sober = driver_frames[driver_frames["BAC"]==0]
    sober_first_third_number = len(sober)//3
    ref = sober.iloc[:sober_first_third_number]          # first 1/3 sober
    mu  = ref[["head","gaze","EAR","P_scale"]].mean()
    std = ref[["head","gaze","EAR","P_scale"]].std().replace(0, 1)
    driver_frames[["head","gaze","EAR","P_scale"]] = \
        (driver_frames[["head","gaze","EAR","P_scale"]] - mu) / std
    driver_frames = driver_frames[sober_first_third_number + 1:]
    return driver_frames

# ----------------------------------------------------------
# 1. Gather all frames into one table
all_frames = []
for csv in Sorted(Path(DATA_DIR).glob("*.csv")):
    driver, bac_str = csv.stem.rsplit("_", 1)
    bac = int(bac_str)
    df = pd.read_csv(csv)
    df["driver"] = driver
    df["BAC"]    = bac
    all_frames.append(df)
df_all = pd.concat(all_frames, ignore_index=True)

# 2. Per-driver Z-score
df_all = (df_all.groupby("driver", group_keys=False)
                .apply(z_score_driver)
                .reset_index(drop=True))

# 3. Sliding-window summary per (driver, BAC)
for (driver, bac), sub in df_all.groupby(["driver","BAC"]):
    wins = make_windows(sub)
    if wins is None:
        continue
    label = BAC_MAP[str(bac)]
    save_dir = Path(OUT_DIR) / str(bac)
    save_dir.mkdir(exist_ok=True)
    np.save(save_dir / f"{driver}_{bac}.npy", wins)
    np.save(save_dir / f"{driver}_{bac}_label.npy",
            np.full(len(wins), label, dtype=np.int8))


print("Window extraction & normalization complete.")
