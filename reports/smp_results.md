# SMP-Video: virality model and conformal intervals (first pass)

Data: SMP-Video labeled train split, 4,000 posts from 3,488 creators. Target is the challenge label
log2(views/day) + 1, modeled directly, so the missing crawl date does not matter.
Protocol: 5-fold creator-disjoint CV; 25% of each fold's training creators calibrate the intervals.
Full tables: `smp_benchmark.md`, `smp_benchmark.csv`, `smp_conformal.csv`. Code: `scale/smp_benchmark.py`, `scale/conformal.py`.

## Point prediction

| features (gradient boosting) | MAE | Spearman | MAPE |
|---|---|---|---|
| predict the mean | 2.39 | 0.00 | 0.444 |
| metadata (time, caption stats, duration) | 2.37 | 0.11 | 0.440 |
| + TF-IDF caption text | 2.22 | 0.33 | 0.409 |
| + CLIP caption embeddings | 2.12 | 0.42 | 0.391 |
| followers only (ridge) | 1.93 | 0.54 | 0.327 |
| + creator stats (followers, likes, video count...) | **1.58** | **0.70** | **0.259** |

- Creator stats carry most of the signal. Followers alone beat every content-only model.
  These stats are read at crawl time, so they include growth after posting; challenge entries use them anyway.
- Among signals known before posting, caption semantics matter: CLIP adds about +0.10 Spearman over TF-IDF.
  Posting time adds almost nothing, matching the winning team's ablation.
- Reference: the 2025 video-track winner (MVP, arXiv 2507.00950) reports MAPE 0.1754 with CatBoost on creator
  stats, X-CLIP video frames, Word2Vec text, tag popularity and label-outlier filtering. That score is on the
  challenge's own evaluation split and is not directly comparable to our creator-disjoint CV, but the gap
  (0.259 vs 0.175) says video embeddings, tag popularity and outlier handling are worth trying next.

## Conformal intervals (target 90%)

Coverage overall and by creator-size quartile, content + CLIP features (no creator stats):

| method | all | smallest creators | Q2 | Q3 | largest | width |
|---|---|---|---|---|---|---|
| raw quantile model | 0.86 | 0.77 | 0.92 | 0.91 | 0.84 | 8.3 |
| split conformal | 0.91 | 0.80 | 0.95 | 0.94 | 0.94 | 8.7 |
| CQR | 0.90 | 0.82 | 0.94 | 0.94 | 0.90 | 9.1 |
| **Mondrian split (by creator size)** | 0.91 | 0.91 | 0.90 | 0.91 | 0.91 | 8.6 |
| Mondrian CQR | 0.90 | 0.91 | 0.90 | 0.90 | 0.90 | 9.0 |

- The raw quantile model under-covers (86% for a 90% target). Every conformal method fixes the average.
- Only Mondrian fixes it for small creators: plain split/CQR cover them 80–82% of the time, Mondrian 91%,
  with no wider intervals on average. Mondrian split also has the best interval score.
- With creator stats, intervals shrink from ~8.6 to ~6.4 label units (a factor of 2^6.4 ≈ 85x in views/day
  between ends). Interval width is the honest headline: even the best model's 90% range spans about two
  orders of magnitude of daily views.
- CQR gives the most even coverage across predicted-level terciles; split conformal over-covers posts
  predicted to do well and under-covers posts predicted to do poorly (with creator stats, 0.95 vs 0.85).

## Next options (not started)

1. Video-frame embeddings for the 4,000 SMP posts (needs yt-dlp clip fetching and CLIP on the Mac; the most
   expensive step).
2. Cheap gains toward the MVP recipe: tag-popularity features, Huber loss/CatBoost, label-outlier filtering.
3. Combine Mondrian and CQR across both creator size and predicted level; add adaptive conformal for
   time-ordered evaluation once TikTok-10M splits land.
