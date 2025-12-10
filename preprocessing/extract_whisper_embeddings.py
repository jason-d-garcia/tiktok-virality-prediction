import numpy as np
import torch
import whisper
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "videos"
OUT_FILE = PROJECT_ROOT / "data" / "embeddings" / "X_whisper.npy"

DEVICE = "cpu"   # keep CPU on Mac (MPS causes this exact error)

# ----------------------------
# LOAD MODEL
# ----------------------------

print("Using device:", DEVICE)
model = whisper.load_model("base").to(DEVICE)

# ----------------------------
# LOAD VIDEOS
# ----------------------------

video_files = sorted(VIDEO_DIR.glob("*.mp4"))
print(f"Found {len(video_files)} videos.")

all_embeddings = []

# ----------------------------
# EXTRACTION LOOP
# ----------------------------

for video_path in tqdm(video_files, desc="Extracting Whisper"):

    try:
        # Whisper SAFE loader (no permute, no torchaudio)
        audio = whisper.load_audio(str(video_path))

        # ---- HARD TRIM TO FIRST 3 SECONDS ----
        SAMPLE_RATE = 16000
        N_SAMPLES = 3 * SAMPLE_RATE  # 3 seconds
        audio = audio[:N_SAMPLES]

        # ---- THEN PAD TO 30s FOR WHISPER ----
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        with torch.no_grad():
            embedding = model.encoder(mel.unsqueeze(0)).mean(dim=1)

        embedding = embedding.cpu().numpy().squeeze()
        all_embeddings.append(embedding)

    except Exception as e:
        print(f"Audio failed for {video_path.name}: {e}")
        all_embeddings.append(np.zeros(512))

# ----------------------------
# SAVE OUTPUT
# ----------------------------

X_whisper = np.vstack(all_embeddings)
np.save(OUT_FILE, X_whisper)

print("Whisper embeddings saved to:", OUT_FILE)
print("Shape:", X_whisper.shape)
