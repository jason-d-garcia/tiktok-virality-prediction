import cv2
import numpy as np
import pandas as pd
import os
import librosa
from tqdm import tqdm

VIDEO_DIR = "../data/raw/videos"
OUTPUT_CSV = "../data/video_qualitative_features.csv"

def optical_flow_features(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    if not ret:
        return None

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    flows = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.linalg.norm(flow, axis=2)
        flows.append(mag.mean())
        prev_gray = gray

    cap.release()
    flows = np.array(flows)

    return {
        "motion_mean": flows.mean(),
        "motion_std": flows.std(),
        "motion_entropy": np.histogram(flows, bins=20, density=True)[0].var()
    }

def audio_features(video_path):
    y, sr = librosa.load(video_path, sr=22050)
    rms = librosa.feature.rms(y=y)[0]

    return {
        "audio_energy_mean": rms.mean(),
        "audio_energy_std": rms.std()
    }

def process_all():
    rows = []

    for fname in tqdm(os.listdir(VIDEO_DIR)):
        if not fname.endswith(".mp4"):
            continue

        vid_path = os.path.join(VIDEO_DIR, fname)

        motion = optical_flow_features(vid_path)
        audio = audio_features(vid_path)

        if motion is None:
            continue

        row = {"video_id": fname}
        row.update(motion)
        row.update(audio)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print("✅ Saved", OUTPUT_CSV)

if __name__ == "__main__":
    process_all()