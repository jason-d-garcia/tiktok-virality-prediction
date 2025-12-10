import cv2
import numpy as np
import pandas as pd
import os
import librosa
from tqdm import tqdm

VIDEO_DIR = "../data/raw/videos"
OUTPUT_CSV = "../data/video_qualitative_features_3s.csv"
N_SECONDS = 3.0  # match CLIP / Whisper window


def optical_flow_features_3s(video_path, n_seconds=N_SECONDS):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None

    max_frames = int(fps * n_seconds)

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return None

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    flows = []
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.linalg.norm(flow, axis=2)
        flows.append(mag.mean())

        prev_gray = gray
        frame_idx += 1

    cap.release()

    flows = np.array(flows)
    if flows.size == 0:
        return None

    # simple entropy-ish proxy: variance of histogram of flow magnitudes
    hist, _ = np.histogram(flows, bins=20, density=True)
    motion_entropy = hist.var()

    return {
        "motion_mean_3s": float(flows.mean()),
        "motion_std_3s": float(flows.std()),
        "motion_entropy_3s": float(motion_entropy),
        "frames_used_3s": int(len(flows)),
    }


def audio_features_3s(video_path, n_seconds=N_SECONDS):
    # librosa will load full audio, we just crop to first n_seconds
    y, sr = librosa.load(video_path, sr=22050)
    max_samples = int(sr * n_seconds)
    y = y[:max_samples]

    if y.size == 0:
        return None

    rms = librosa.feature.rms(y=y)[0]

    return {
        "audio_energy_mean_3s": float(rms.mean()),
        "audio_energy_std_3s": float(rms.std()),
    }


def process_all_3s():
    rows = []

    for fname in tqdm(os.listdir(VIDEO_DIR)):
        if not fname.lower().endswith((".mp4", ".mov", ".mkv")):
            continue

        vid_path = os.path.join(VIDEO_DIR, fname)

        motion = optical_flow_features_3s(vid_path)
        audio = audio_features_3s(vid_path)

        if motion is None or audio is None:
            # If one fails, skip so you don't poison the merge later
            print(f"Skipping {fname}: could not extract 3s features.")
            continue

        row = {"video_id": fname}
        row.update(motion)
        row.update(audio)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved 3-second qualitative features to {OUTPUT_CSV}")


if __name__ == "__main__":
    process_all_3s()
