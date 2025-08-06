#!/usr/bin/env python3
"""
Step-1  Extract OpenFace CSVs from driver videos
Creates CSVs that contain *only* the 4 columns required by the paper.
"""

import subprocess
from pathlib import Path

input_dir  = Path("/Users/zhaoda/Desktop/8:5 test out new feature extraction").expanduser()
output_dir = Path("/Users/zhaoda/Desktop/8:5 test out new feature extraction/output").expanduser()
output_dir.mkdir(exist_ok=True)

for video in input_dir.glob("*.avi"):
    cmd = [
        "/Users/zhaoda/Desktop/OpenFace/build/bin/FeatureExtraction",
        "-f", str(video),
        "-aus", "-gaze", "-pose", "-2Dfp",         # existing
        "-pdmparams",                              # ➕ adds p_scale
        "-eye",                                    # ➕ adds eye_lmk_EAR_avg
        "-out_dir", str(output_dir),
        "-of", f"{video.stem}.csv"
    ]
    subprocess.run(cmd, check=True)