#!/usr/bin/env python3
"""
Step-3: 150-frame sliding windows + per-driver Z-score (as described in Keshtkaran et al. WACV 2024)
- Uses first 1/3 of sober frames as reference for per-driver Z-score
- Excludes reference frames from final dataset
- Projects head pose (pose_Rx, pose_Ry) → polar coordinates (head_r, head_theta)
- Computes windowed mean AND std (captures variation over time)
- No global normalization (avoids data leakage)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
DATA_DIR = "/Users/zhaoda/Desktop/8:14 step 2 CSVs"
OUT_DIR  = "/Users/zhaoda/Desktop/8:15 all windows (stride 75)"
WIN      = 150
STRIDE   = 75  # 50% overlap for smoother LSTM input
BAC_MAP  = {"0": 0, "5": 1, "10": 2}  # Sober, Low AII, Severe AII

os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------
def to_polar(df):
    """Convert Cartesian head pose (pose_Rx, pose_Ry) to polar coordinates"""
    df = df.copy()
    x = df['pose_Rx']
    y = df['pose_Ry']
    df['head_r'] = np.sqrt(x**2 + y**2)                    # Radial distance (magnitude of movement)
    df['head_theta'] = np.arctan2(y, x)                    # Angular direction
    return df


def make_windows(df):
    """
    Return (N, 10) array where each row = [mean + std] over 150-frame window
    Features: gaze_angle_x, gaze_angle_y, EAR, P_scale, head_r, head_theta
    → 6 features × 2 (mean & std) = 12 → but we drop raw pose_Rx/Ry
    """
    # Use only key behavioral features including polar head pose
    feature_cols = [
        "gaze_angle_x", "gaze_angle_y",
        "EAR", "P_scale",
        "head_r", "head_theta"
    ]
    vals = df[feature_cols].copy()

    N = len(vals) - WIN + 1
    if N <= 0:
        return None

    windows = []
    for i in range(0, N, STRIDE):
        window_data = vals[i:i+WIN]
        window_mean = window_data.mean(axis=0).values
        window_std = window_data.std(axis=0).values
        window_stats = np.concatenate([window_mean, window_std])  # (12,)
        windows.append(window_stats)

    return np.array(windows).astype(np.float32)


def z_score_driver(driver_frames):
    """
    Per-driver Z-score using first 1/3 of BAC=0 frames as reference.
    Normalize ALL frames, then EXCLUDE reference frames from final dataset.
    """
    # Work on a copy
    df = driver_frames.copy()

    # Convert to polar coordinates FIRST (before normalization)
    df = to_polar(df)

    # Get sober frames (BAC = 0)
    sober = df[df["BAC"] == 0]
    if len(sober) < 3:
        print(f"Warning: Driver has insufficient sober frames ({len(sober)}). Skipping.")
        return None

    # Use first 1/3 of sober frames as reference
    sober_first_third_number = max(1, len(sober) // 3)
    ref_indices = sober.index[:sober_first_third_number]
    ref = df.loc[ref_indices]

    # Select features to normalize
    feature_cols = [
        "gaze_angle_x", "gaze_angle_y",
        "EAR", "P_scale",
        "head_r", "head_theta"
    ]

    mu = ref[feature_cols].mean()
    std = ref[feature_cols].std().replace(0, 1)  # Avoid division by zero

    # Apply Z-score normalization to ALL frames
    df[feature_cols] = (df[feature_cols] - mu) / std

    # CRITICAL: Exclude reference frames from final dataset
    df = df[~df.index.isin(ref_indices)].reset_index(drop=True)

    return df


# ----------------------------------------------------------
# 1. Gather all frames into one table
all_frames = []
for csv in sorted(Path(DATA_DIR).glob("*.csv")):
    driver, bac_str = csv.stem.rsplit("_", 1)
    bac = int(bac_str)
    df = pd.read_csv(csv)
    df["driver"] = driver
    df["BAC"] = bac
    all_frames.append(df)

df_all = pd.concat(all_frames, ignore_index=True)

# 2. Apply per-driver Z-score with polar projection
df_all = (df_all.groupby("driver", group_keys=False)
                .apply(z_score_driver)
                .reset_index(drop=True))

# Drop any failed drivers
df_all = df_all.dropna()
if len(df_all) == 0:
    raise ValueError("No valid data remaining after normalization. Check input data quality.")

# 3. Sliding-window summary per (driver, BAC)
for (driver, bac), sub in df_all.groupby(["driver", "BAC"]):
    if len(sub) < WIN:
        print(f"Skipping {driver}_{bac}: Insufficient frames ({len(sub)} < {WIN})")
        continue

    wins = make_windows(sub)
    if wins is None or len(wins) == 0:
        continue

    label = BAC_MAP[str(bac)]
    save_dir = Path(OUT_DIR) / str(bac)
    save_dir.mkdir(exist_ok=True)

    # Save windowed features and labels
    np.save(save_dir / f"{driver}_{bac}.npy", wins)
    np.save(save_dir / f"{driver}_{bac}_label.npy", 
            np.full(len(wins), label, dtype=np.int8))

print(f"Window extraction complete. Generated features with polar head pose. Final frame count: {len(df_all)}")