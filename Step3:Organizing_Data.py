#!/usr/bin/env python3
"""
Step-3: Fixed version with polar head pose + t-test (resolves KeyError and missing BAC=0)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind
import warnings

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = "/Users/zhaoda/Desktop/8:14 step 2 CSVs"
OUT_DIR = "/Users/zhaoda/Desktop/8:15 windows (75 stride) v2"
WIN = 150
STRIDE = 75
BAC_MAP = {"0": 0, "5": 1, "10": 2}

os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------
# Helper Functions
# -------------------------------
def to_polar(df):
    df = df.copy()
    x = df['pose_Rx']
    y = df['pose_Ry']
    df['head_r'] = np.sqrt(x**2 + y**2)
    df['head_theta'] = np.arctan2(y, x)
    return df

def make_windows(df):
    feature_cols = ['gaze_angle_x', 'gaze_angle_y', 'EAR', 'P_scale', 'head_r', 'head_theta']
    vals = df[feature_cols].copy().values
    N = len(vals) - WIN + 1
    if N <= 0:
        return None
    windows = []
    for i in range(0, N, STRIDE):
        window_data = vals[i:i+WIN]
        window_mean = np.mean(window_data, axis=0)
        window_std = np.std(window_data, axis=0)
        window_stats = np.concatenate([window_mean, window_std])
        windows.append(window_stats)
    return np.array(windows, dtype=np.float32)

def z_score_driver(driver_frames):
    """
    Normalize using first 1/3 of BAC=0 frames as reference.
    Exclude those reference frames from output.
    Returns None if no BAC=0 data.
    """
    # Add polar coordinates
    driver_frames = to_polar(driver_frames)

    # Extract sober (BAC=0) frames
    sober = driver_frames[driver_frames["BAC"] == 0]
    if len(sober) == 0:
        print(f"Skipping driver: No BAC=0 (sober) data found for normalization.")
        return None

    if len(sober) < 3:
        print(f"Skipping driver: Not enough BAC=0 frames ({len(sober)}) for reference.")
        return None

    # Use first 1/3 of BAC=0 frames as reference
    ref_size = max(1, len(sober) // 3)
    ref_indices = sober.index[:ref_size]
    ref = driver_frames.loc[ref_indices]

    feature_cols = ['gaze_angle_x', 'gaze_angle_y', 'EAR', 'P_scale', 'head_r', 'head_theta']
    mu = ref[feature_cols].mean()
    std = ref[feature_cols].std().replace(0, 1.0)

    # Normalize all frames
    result = driver_frames.copy()
    result[feature_cols] = (result[feature_cols] - mu) / std

    # Exclude reference frames
    result = result[~result.index.isin(ref_indices)].reset_index(drop=True)
    return result

def select_features_by_ttest(windows_dict, p_threshold=0.05):
    """
    Perform t-test between Sober (0) and Impaired (5,10)
    Uses .iloc for correct column indexing
    """
    all_data = []
    for (driver, bac), feats in windows_dict.items():
        label = 0 if bac == 0 else 1  # binary: sober vs impaired
        for f in feats:
            row = list(f) + [label, driver]
            all_data.append(row)

    if len(all_data) == 0:
        print("No windowed data for t-test. Using all 12 features.")
        return list(range(12))

    data_df = pd.DataFrame(all_data)
    data_df.columns = [f'f{i}' for i in range(12)] + ['impaired', 'driver']

    p_values = []
    for i in range(12):  # for each feature
        p = 1.0
        group_comparisons = []
        for drv in data_df['driver'].unique():
            sub = data_df[data_df['driver'] == drv]
            sober_vals = sub[sub['impaired'] == 0][f'f{i}']
            imp_vals = sub[sub['impaired'] == 1][f'f{i}']
            if len(sober_vals) > 1 and len(imp_vals) > 1:
                _, p_ind = ttest_ind(sober_vals, imp_vals, nan_policy='omit')
                group_comparisons.append(p_ind)
        # Use median p across drivers to avoid outlier influence
        if group_comparisons:
            p = np.median(group_comparisons)
        p_values.append(p)

    significant_indices = [i for i, p in enumerate(p_values) if p < p_threshold]
    if len(significant_indices) == 0:
        print("⚠️ No features passed t-test. Falling back to all features.")
        significant_indices = list(range(12))

    print(f"✅ Selected {len(significant_indices)} features: {significant_indices}")
    return significant_indices

# -------------------------------
# Main Pipeline
# -------------------------------
print("Loading and preprocessing data...")

all_frames = []
for csv in sorted(Path(DATA_DIR).glob("*.csv")):
    stem = csv.stem
    try:
        driver, bac_str = stem.rsplit("_", 1)
        bac = int(bac_str)
    except Exception as e:
        print(f"Skipping {csv.name}: unexpected filename format.")
        continue

    df = pd.read_csv(csv)
    df["driver"] = driver
    df["BAC"] = bac
    all_frames.append(df)

df_all = pd.concat(all_frames, ignore_index=True)
print(f"Loaded {len(df_all)} total frames from {len(all_frames)} files.")

# Group by driver and apply normalization
print("Applying per-driver Z-score normalization (using BAC=0 as reference)...")
df_normalized = df_all.groupby("driver", group_keys=False).apply(z_score_driver)

# Drop None groups (drivers without BAC=0)
df_normalized = df_normalized.dropna()
if isinstance(df_normalized, pd.Series):
    # In case groupby returns a Series due to internal issue
    print("⚠️ Groupby returned Series. Something went wrong.")
    exit(1)

df_normalized = df_normalized.reset_index(drop=True)
print(f"Data after normalization: {len(df_normalized)} frames.")

# Create windows
print("Creating sliding windows...")
windows_dict = {}
for (driver, bac), sub in df_normalized.groupby(["driver", "BAC"]):
    if len(sub) < WIN:
        print(f"Skipping {driver}_{bac}: insufficient frames ({len(sub)} < {WIN})")
        continue
    wins = make_windows(sub)
    if wins is not None and len(wins) > 0:
        windows_dict[(driver, bac)] = wins

if not windows_dict:
    raise ValueError("No windows generated. Check data length and normalization.")

# Feature selection
print("Running t-test for feature selection...")
selected_feature_indices = select_features_by_ttest(windows_dict, p_threshold=0.05)

# Save results
print("Saving windowed features and labels...")
for (driver, bac), wins in windows_dict.items():
    wins_filtered = wins[:, selected_feature_indices].astype(np.float32)
    label = BAC_MAP[str(bac)]
    save_dir = Path(OUT_DIR) / str(bac)
    save_dir.mkdir(exist_ok=True)
    
    np.save(save_dir / f"{driver}_{bac}.npy", wins_filtered)
    np.save(save_dir / f"{driver}_{bac}_label.npy", 
            np.full(len(wins_filtered), label, dtype=np.int8))

print(f"✅ Processing complete. Final feature dim: {len(selected_feature_indices)}")