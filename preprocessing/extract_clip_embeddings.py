import os
import cv2
import torch
import clip
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "videos"
OUT_DIR = PROJECT_ROOT / "data" / "embeddings"
OUT_FILE = OUT_DIR / "X_clip.npy"

SECONDS_TO_SAMPLE = 3        # first 3 seconds
TARGET_FPS = 5               # frames per second to sample

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ----------------------------
# SETUP
# ----------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using device: {DEVICE}")

model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

video_files = sorted([f for f in VIDEO_DIR.glob("*.mp4")])

print(f"Found {len(video_files)} videos.")



# ----------------------------
# FRAME SAMPLING FUNCTION
# ----------------------------

def sample_frames_first_n_seconds(video_path, seconds=3, target_fps=5):
    cap = cv2.VideoCapture(str(video_path))

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps == 0:
        native_fps = 30  # fallback

    max_frames = int(seconds * target_fps)
    frame_interval = max(int(native_fps // target_fps), 1)

    frames = []
    count = 0
    grabbed = 0

    while grabbed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            grabbed += 1

        count += 1

    cap.release()
    return frames

# ----------------------------
# EMBEDDING LOOP
# ----------------------------

all_embeddings = []

with torch.no_grad():
    for video_path in tqdm(video_files):
        frames = sample_frames_first_n_seconds(
            video_path,
            seconds=SECONDS_TO_SAMPLE,
            target_fps=TARGET_FPS
        )

        if len(frames) == 0:
            print(f"No frames found in {video_path.name}")
            all_embeddings.append(np.zeros(512))
            continue

        clip_inputs = torch.stack([
            preprocess(Image.fromarray(frame))
            for frame in frames
        ]).to(DEVICE)

        image_features = model.encode_image(clip_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        video_embedding = image_features.mean(dim=0).cpu().numpy()
        all_embeddings.append(video_embedding)

# ----------------------------
# SAVE OUTPUT
# ----------------------------

X_clip = np.stack(all_embeddings)
np.save(OUT_FILE, X_clip)

print("CLIP embeddings saved to:", OUT_FILE)
print("Shape:", X_clip.shape)