# Scaling up: pretrain on many creators, adapt to Emily

Pretrain a multimodal model (metadata + CLIP video + Whisper audio + CLIP caption
text) on public TikTok posts. It predicts the **distribution** of views, not just
a point. Then adapt it to Emily and calibrate its intervals with conformal prediction.

## Pipeline

| Step | Command | Output |
| --- | --- | --- |
| 1. Build the post table | `python -m scale.build_posts --emily data/raw/tiktok_stats.csv --tiktok10m <dir> --einzz <dir> --max-posts-per-source 200000 --out data/scale/posts.parquet` | one row per post, prepublication features |
| 2. Caption embeddings (all posts) | `python -m scale.embed --posts data/scale/posts.parquet --out-dir data/scale/emb/text --modalities text` | `emb/text/shard_*.npz` |
| 3. Fetch clips (subset) | `python -m scale.fetch_clips --posts data/scale/posts.parquet --out-dir data/scale/clips --seconds 5 --limit 20000` | first 5 s of each video + `manifest.csv` |
| 3b. Emily's clips | `python -m scale.link_emily_clips --clips-dir data/scale/clips` | links her existing downloads |
| 4. Video + audio embeddings | `python -m scale.embed --posts data/scale/posts.parquet --clips-dir data/scale/clips --out-dir data/scale/emb/av --modalities video,audio --delete-clips` | `emb/av/shard_*.npz` |
| 5. Pretrain | `python -m scale.train --posts data/scale/posts.parquet --emb-dirs data/scale/emb/text data/scale/emb/av --modalities video,audio,text --head mdn --out models/scale/mdn_all.pt` | checkpoint + a row in `reports/scale_results.csv` |
| 6. Adapt to Emily | `python -m scale.finetune --posts data/scale/posts.parquet --checkpoint models/scale/mdn_all.pt --emb-dirs data/scale/emb/text data/scale/emb/av` | rows per mode and calibration |

Every step is resumable. Re-run the same command and it skips finished work.

Smoke test on synthetic data (no downloads, CPU only): `python -m tests.smoke_scale`.

## Experiment grid

Run `scale.train` once per cell. Each run appends a row to `reports/scale_results.csv`.

- **Heads** (`--head`): `gaussian` (lognormal), `mdn` (lognormal mixture, `--n-components 2|3`, check `mdn_weights` for collapse), `quantile` (pinball), `negbin` (counts).
- **Modalities** (`--modalities`): `""` (metadata only), `text`, `video`, `audio`, `video,audio,text`. Modality dropout lets one model handle posts with or without clips.
- **Followers** (`--use-followers`): off by default. Follower counts are read at collection time, so they leak post-publication growth.

Scores are on held-out creators, never seen in training: median MAE, Spearman,
pinball loss, 80/90% coverage, interval width, interval score and count-scale
NLL. Every score sits next to the creator baseline (the median of that creator's
previous 10 posts) on the same rows.

## Known limits

- Views are one cumulative snapshot. TikTok-10M has `collected_time`, so post age is a control. EinzzCookie and Emily don't record it.
- Creator-history features use the *final* view counts of earlier posts, which overstates what a creator knew on publish day.
- None of the public sets ship video files. Clips come from yt-dlp, which TikTok's terms don't permit. Keep the clip count modest and delete clips after embedding.
- Whisper runs on CPU on Apple Silicon (MPS fails on some encoder ops). CLIP uses MPS.
