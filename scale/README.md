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

## TikTok-10M, shared splits and the leaderboard

Run these once the shards are on the external drive (`<drive>` below). Steps 0–2
need no clips or embeddings, so the creator baseline leaderboard exists before any model does.

| Step | Command | Output |
| --- | --- | --- |
| 0. Dry run (no download) | `python -m tests.synthetic_tiktok10m --out data/synthetic/tiktok10m`, then steps A–D on that folder | proves the chain end to end |
| A. Ingest TikTok-10M | `python -m scale.ingest_tiktok10m --src /Volumes/<drive>/hf/tiktok10m --out /Volumes/<drive>/tiktok/tiktok10m_posts.parquet` | deduped post table + `.card.json` (duplicates, creator sizes, truncation histogram, ages). Streams one row group at a time; re-runs skip finished shards. `--max-shards 1` for a first look |
| B. Build the post table | step 1 above with `--tiktok10m /Volumes/<drive>/tiktok/tiktok10m_posts.parquet` (plus `--emily`, `--smp`, `--einzz`) | `posts.parquet` |
| C. Splits | `python -m scale.splits --posts data/scale/posts.parquet --out data/scale/splits.parquet` | one row per post with `split_creator`, `split_time`, `split_strict`; `.card.json` with leakage checks. Exits non-zero if a hard check fails |
| D. Leaderboard | `python -m scale.leaderboard --posts data/scale/posts.parquet --splits data/scale/splits.parquet --scheme strict --preds runs/*.parquet` | `reports/leaderboard_strict.csv` and `.md` |

**Split schemes.** `creator`: held-out creators by hash, any date. `time`: global
date cutoffs (last 10% of dates test, 10% before that val) with a 7-day embargo
before each cutoff. `strict` (headline): held-out creators *and* later dates, so a
test post's creator and period are both unseen; earlier posts of held-out creators
are `history` (feed creator-history features only), training creators' later posts
are `excluded`. SMP-Video and EinzzCookie are `external` in every scheme; Emily is
`emily_adapt/cal/test` (oldest 60/20/20%, as in `scale.finetune`). Strict test is
roughly val_frac × test_frac of the posts (~1%), so on a ~20K-post sample raise
`--test-frac`.

**Leakage checks.** Hard: unique keys, every post assigned, no external or Emily
post in train/val, creator-disjoint train/val/test, train ends at least the
embargo before val and val before test. Soft (reported): test posts that repeat a
training post (same creator, caption and length), and the share of training
labels read after the test window opened. With one snapshot per post that share
is ~100%, which means training labels carry platform state from after the test
period starts: a known limit of cumulative-snapshot data, not a split bug.

**Prediction files** for the leaderboard: `source`, `post_id`, `q0.5` and any other
`qX` quantile columns on the log1p(views) scale, optional `model`. Write them with
`scale.leaderboard.write_predictions`, or pass `--splits <table> --scheme strict
--pred-out runs/<name>.parquet` to `scale.train`. Built-in baselines
(`global_train`, `creator_median10`, `creator_median10+q`) are always scored, and
every row shows MAE, Spearman, quantile CRPS, coverage, interval score, MAE on the
SMP label scale, and the MAE gap to `creator_median10` with a creator-clustered
bootstrap 95% CI. Checks: `python -m tests.test_pipeline`.

**Unattended run** (steps 2–4 in one command, for a weekend on an external drive): `caffeinate -i python -m scale.weekend_run --posts data/scale/posts.parquet --root /Volumes/<drive>/tiktok --target 100000`. It runs a 200-post pilot per source (drops sources below `--min-success`), then fetches whole-creator batches while the previous batch embeds, deletes clips after embedding, backs off once on a low-success batch and stops on a second one, or when the disk drops below `--min-free-gb`. Progress: `<root>/run_log.jsonl`.

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

- Views are one cumulative snapshot. Post age is a control: TikTok-10M age runs to `stats_time` (`--tiktok10m-views-at` switches it back), SMP-Video to its crawl date. EinzzCookie and Emily don't record it. Posts younger than 7 days at collection are dropped (`--min-age-days`).
- Creator-history features (and the creator baseline) use only posts published at least 7 days earlier. Their counts are still final counts, so history remains a little optimistic.
- EinzzCookie (6,258 unique posts, ~1 per creator) and SMP-Video (4,000 labeled posts, ~1.1 per creator) are test sets only: `build_posts` keeps them whole (no creator sampling or `--min-posts-per-creator`), and `scale.train` never pretrains on them. SMP is the untruncated calibration target; creator-history features are nearly empty for it.
- SMP-Video does not release views. Its label is s = log2(views / days) + 1, with days counted to an unpublished crawl date, so views are rebuilt as days × 2^(label − 1) using the date passed as `--smp-collected`. The crawl falls between the last test post (2023-12-31) and the Hugging Face upload (2025-03-16); we assume **2024-01-15** until the organisers confirm it. These rows carry `views_estimated = True`. Shifting the date moves an SMP post's views by the factor (assumed age / true age); with a 2025-03 crawl, the newest train posts would be under-counted ~4×. The label divides by age, which assumes views grow linearly, and in the train split the median label rises ~1.6 log2 units from 2022 Q1 to 2023 Q3. Column names were checked against the release files. Unit checks: `python -m tests.test_posts`.
- None of the public sets ship video files. Clips come from yt-dlp, which TikTok's terms don't permit. Keep the clip count modest and delete clips after embedding.
- Whisper runs on CPU on Apple Silicon (MPS fails on some encoder ops). CLIP uses MPS.
