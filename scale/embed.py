"""
Batch CLIP / Whisper / caption embeddings for the post table, written as resumable shards.

    python -m scale.embed --posts data/scale/posts.parquet --clips-dir data/scale/clips \
        --out-dir data/scale/emb --modalities video,audio,text --seconds 5

Per post:
  video  CLIP ViT-B/32 image embeddings of frames sampled at 2 fps over the first
         `seconds`, L2-normalised and mean-pooled (512-d)
  audio  Whisper-base encoder states over the first `seconds` only, mean-pooled
         (512-d). preprocessing/extract_whisper_embeddings.py pools all 1,500
         encoder frames of the 30 s padded window, so ~90% of that average is
         padding. Emily's videos should be re-embedded here for consistency.
  text   CLIP text embedding of the caption, truncated to 77 tokens (512-d)

Each shard is an .npz with `ids` (source_postid) and one array per modality;
rows a modality couldn't produce are NaN. Already-embedded ids are skipped, so use
one --out-dir per modality set (e.g. emb/text for every caption, emb/av for the
posts that have clips).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

WHISPER_FRAMES_PER_SECOND = 50   # encoder stride: 20 ms per frame
SHARD_SIZE = 2000


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def existing_ids(out_dir):
    ids = set()
    for shard in out_dir.glob("shard_*.npz"):
        ids.update(np.load(shard, allow_pickle=True)["ids"].tolist())
    return ids


def sample_frames(video_path, seconds, fps):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(native / fps)), 1)
    frames, idx = [], 0
    while idx < seconds * native:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


class Embedder:
    def __init__(self, modalities, device, seconds):
        import clip
        self.modalities = modalities
        self.device = device
        self.seconds = seconds
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        self.tokenize = clip.tokenize
        if "audio" in modalities:
            import whisper
            self.whisper = whisper
            # Whisper's encoder fails on MPS for some ops; CPU is fast enough for base.
            whisper_device = "cpu" if device == "mps" else device
            self.whisper_model = whisper.load_model("base", device=whisper_device)

    @torch.no_grad()
    def video(self, path):
        from PIL import Image
        frames = sample_frames(path, self.seconds, fps=2)
        if not frames:
            return None
        batch = torch.stack([self.clip_preprocess(Image.fromarray(f)) for f in frames]).to(self.device)
        feats = self.clip_model.encode_image(batch).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.mean(0).cpu().numpy()

    @torch.no_grad()
    def audio(self, path):
        w = self.whisper
        audio = w.load_audio(str(path))[: self.seconds * w.audio.SAMPLE_RATE]
        if len(audio) == 0:
            return None
        mel = w.log_mel_spectrogram(w.pad_or_trim(audio)).to(self.whisper_model.device)
        states = self.whisper_model.encoder(mel.unsqueeze(0))[0]
        n_real = max(1, int(len(audio) / w.audio.SAMPLE_RATE * WHISPER_FRAMES_PER_SECOND))
        return states[:n_real].mean(0).float().cpu().numpy()

    @torch.no_grad()
    def text(self, captions):
        tokens = self.tokenize([c or " " for c in captions], truncate=True).to(self.device)
        feats = self.clip_model.encode_text(tokens).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()


def write_shard(out_dir, ids, arrays):
    n = len(list(out_dir.glob("shard_*.npz")))
    path = out_dir / f"shard_{n:05d}.npz"
    np.savez(path, ids=np.array(ids, dtype=object), **{k: np.stack(v) for k, v in arrays.items()})
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts", required=True)
    parser.add_argument("--clips-dir", default=None, help="needed for video/audio")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--modalities", default="video,audio,text")
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--delete-clips", action="store_true", help="remove each clip once embedded")
    args = parser.parse_args()

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    needs_clip = bool({"video", "audio"} & set(modalities))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = Path(args.clips_dir) if args.clips_dir else None

    posts = pd.read_parquet(args.posts, columns=["post_id", "source", "caption"])
    posts["key"] = posts["source"] + "_" + posts["post_id"].astype(str)
    posts = posts[~posts["key"].isin(existing_ids(out_dir))]
    if needs_clip:
        if clips_dir is None:
            raise SystemExit("--clips-dir is required for video/audio")
        posts["clip"] = posts["key"].map(lambda k: clips_dir / f"{k}.mp4")
        posts = posts[posts["clip"].map(Path.exists)]
    print(f"{len(posts)} posts to embed: {modalities}")
    if posts.empty:
        return

    device = pick_device(args.device)
    print("Device:", device)
    emb = Embedder(modalities, device, args.seconds)
    nan = np.full(512, np.nan, dtype=np.float32)

    for start in tqdm(range(0, len(posts), SHARD_SIZE), desc="shards"):
        chunk = posts.iloc[start:start + SHARD_SIZE]
        arrays = {m: [] for m in modalities}
        if "text" in modalities:
            for i in range(0, len(chunk), 256):
                arrays["text"].extend(emb.text(chunk["caption"].iloc[i:i + 256].tolist()))
        for row in chunk.itertuples(index=False):
            for m in ("video", "audio"):
                if m not in modalities:
                    continue
                try:
                    vec = getattr(emb, m)(row.clip)
                except Exception as e:  # corrupt or silent clip: record NaN, keep going
                    print(f"{m} failed for {row.key}: {e}")
                    vec = None
                arrays[m].append(nan if vec is None else vec.astype(np.float32))
            if args.delete_clips and needs_clip:
                Path(row.clip).unlink(missing_ok=True)
        write_shard(out_dir, chunk["key"].tolist(), arrays)


def load_embeddings(emb_dir, keys, modalities=("video", "audio", "text")):
    """Embeddings aligned to `keys`; rows with no shard entry are NaN. Returns {modality: (N, 512)}."""
    index = {k: i for i, k in enumerate(keys)}
    out = {m: np.full((len(keys), 512), np.nan, dtype=np.float32) for m in modalities}
    for shard in sorted(Path(emb_dir).glob("shard_*.npz")):
        data = np.load(shard, allow_pickle=True)
        rows = [index.get(k, -1) for k in data["ids"]]
        mask = np.array([r >= 0 for r in rows])
        target = np.array([r for r in rows if r >= 0], dtype=int)
        for m in modalities:
            if m in data.files and mask.any():
                out[m][target] = data[m][mask]
    return out


if __name__ == "__main__":
    main()
