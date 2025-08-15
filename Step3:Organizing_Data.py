#!/usr/bin/env python3
"""
Step-3: 150-frame sliding windows + per-driver Z-score (as described in Keshtkaran et al. WACV 2024)
"""

import os, numpy as np, pandas as pd
from pathlib import Path

DATA_DIR = "/Users/zhaoda/Desktop/8:5 test out new feature extraction/Step 2 csvs"
OUT_DIR  = "/Users/zhaoda/Desktop/8:5 test out new feature extraction/windows"
WIN      = 150
STRIDE   = 75  # Using stride of half window size as per best practices for LSTM inputs
BAC_MAP  = {"0":0, "5":1, "10":2}  # Sober, Low AII, Severe AII

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------
def make_windows(df):
    """
    Return (N,12) array where each row = mean AND std over 150-frame window
    Per paper's Figure 3, we need to capture variations (std), not just means
    """
    # Only use the 4 key features identified in the paper as significant
    # Table 4 shows gaze angle, p-scale, and EAR have pronounced effects
    feature_cols = ["gaze_angle_x", "gaze_angle_y", "EAR", "P_scale", "pose_Rx", "pose_Ry"]
    vals = df[feature_cols].copy()
    
    N = len(vals) - WIN + 1
    if N <= 0:
        return None
    
    windows = []
    for i in range(0, N, STRIDE):
        window_data = vals[i:i+WIN]
        # Compute BOTH mean and std to capture variations as shown in paper's Figure 3
        window_mean = window_data.mean(axis=0).values
        window_std = window_data.std(axis=0).values
        # Combine mean and std into single feature vector
        window_stats = np.concatenate([window_mean, window_std])
        windows.append(window_stats)
    
    return np.array(windows).astype(np.float32)

def z_score_driver(driver_frames):
    """
    Per paper Section 3.1(c): Use first 1/3 of sober frames to compute μ & σ
    Then normalize whole driver, EXCLUDING the reference frames from final dataset
    """
    # Get only sober frames (BAC=0)
    sober = driver_frames[driver_frames["BAC"] == 0].copy()
    
    # Check if we have enough sober frames
    if len(sober) < 3:
        print(f"Warning: Driver has insufficient sober frames ({len(sober)}). Skipping.")
        return None
    
    # Get first 1/3 of sober frames as reference (as per paper)
    sober_first_third_number = max(1, len(sober) // 3)
    ref = sober.iloc[:sober_first_third_number]
    
    # Calculate normalization parameters from reference frames
    feature_cols = ["pose_Rx", "pose_Ry", "gaze_angle_x", "gaze_angle_y", "EAR", "P_scale"]
    mu = ref[feature_cols].mean()
    # Replace 0 std with 1 to avoid division by zero
    std = ref[feature_cols].std().replace(0, 1)
    
    # Create a copy to avoid modifying the original data
    result = driver_frames.copy()
    
    # Apply normalization to ALL frames (including reference frames)
    result[feature_cols] = (result[feature_cols] - mu) / std
    
    # CRITICAL: Exclude reference frames from final dataset (as per paper)
    # Keep all frames EXCEPT the first 1/3 of sober frames
    sober_indices = driver_frames[driver_frames["BAC"] == 0].index[:sober_first_third_number]
    result = result[~result.index.isin(sober_indices)].reset_index(drop=True)
    
    return result

# ----------------------------------------------------------
# 1. Gather all frames into one table
all_frames = []
for csv in sorted(Path(DATA_DIR).glob("*.csv")):
    driver, bac_str = csv.stem.rsplit("_", 1)
    bac = int(bac_str)
    df = pd.read_csv(csv)
    df["driver"] = driver
    df["BAC"]    = bac
    all_frames.append(df)
df_all = pd.concat(all_frames, ignore_index=True)

# 2. Per-driver Z-score (NO GLOBAL NORMALIZATION)
df_all = (df_all.groupby("driver", group_keys=False)
                .apply(z_score_driver)
                .reset_index(drop=True))

# Filter out any drivers with insufficient data after normalization
df_all = df_all.dropna()
if len(df_all) == 0:
    raise ValueError("No valid data remaining after normalization. Check input data quality.")

# 3. Sliding-window summary per (driver, BAC)
for (driver, bac), sub in df_all.groupby(["driver", "BAC"]):
    # Ensure we have enough frames for at least one window
    if len(sub) < WIN:
        print(f"Skipping {driver}_{bac}: Insufficient frames ({len(sub)} < {WIN})")
        continue
        
    wins = make_windows(sub)
    if wins is None or len(wins) == 0:
        continue
        
    label = BAC_MAP[str(bac)]
    save_dir = Path(OUT_DIR) / str(bac)
    save_dir.mkdir(exist_ok=True)
    
    # Save window features
    np.save(save_dir / f"{driver}_{bac}.npy", wins)
    # Save corresponding labels
    np.save(save_dir / f"{driver}_{bac}_label.npy", 
            np.full(len(wins), label, dtype=np.int8))

print(f"Window extraction & normalization complete. Generated {len(df_all)} valid frames.")