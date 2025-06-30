import subprocess
from pathlib import Path

input_dir = Path("/Users/zhaoda/Desktop/ThundersoftSummer2025/Data").expanduser()
output_dir = Path("/Users/zhaoda/Desktop/ThundersoftSummer2025/Data/Processed_features").expanduser()
output_dir.mkdir(exist_ok=True)

for video in input_dir.glob("*.avi"):
    cmd = [
        "/Users/zhaoda/Desktop/OpenFace/build/bin/FeatureExtraction",
        "-f", str(video),
        "-aus", "-gaze", "-pose", "-2Dfp",
        "-out_dir", str(output_dir),
        "-of", f"{video.stem}.csv"
    ]
    subprocess.run(cmd, check=True)