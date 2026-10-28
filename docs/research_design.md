# Research design: distributional TikTok performance prediction under selection and shift

Prepared 2026-10-19 on the `pretraining` branch, after a review of the code in `scale/`, `src/` and `reports/`, and a direct audit of the public datasets. Every number below marked **[audited]** was computed in this session from the raw files. Anything marked **[assumption]** has not been verified and is a planning estimate.

---

## 0. The short version

1. **The data is not what the plan assumes, and that changes the project.**
   - The "~1 million record" Hugging Face set (EinzzCookie) has **6,258 unique posts**. The other ~944K rows repeat those posts **[audited]**. It also has no follower counts and roughly one post per creator, so it supports no creator-history modeling.
   - TikTok-10M is real and large, but it was **selected on the outcome**: views are sharply cut off below about 30K (10^4.5) **[audited]**. Any marginal or conditional distribution fit to it naively is the distribution *of trending posts*, not of TikTok posts.
   - SMP-Video is about 6K posts from 4.5K users (about 1.3 posts per user), and its label is views per day, not views.
   - Emily is 140 posts.

2. **Recommended primary direction: C, conditional density estimation.** The precise question is:
   > *How much of the conditional distribution of a video's creator-relative performance can pretrained multimodal content representations predict at upload time? Do those predictive distributions stay calibrated when the population, the time period or the data-collection mechanism changes?*

   Directions D (shift) and E (conformal) are the **evaluation** layer, not separate projects. Direction A falls out as an ablation measured with proper scoring rules. Direction B (latent regimes) is demoted to a **falsification analysis**. I expect it to show that the "regimes" are a mixture approximation to a skewed, heavy-tailed continuous distribution, and the design lets that be shown rigorously.

3. **The most distinctive methodological piece is something the data forces on us: truncation-aware likelihoods.** TikTok-10M only contains posts above a view threshold, so the likelihood should be $p(y\mid x)/\{1-F(L\mid x)\}$ on $y>L$. That yields a testable claim that a plain regression project cannot make: *a truncation-aware model transfers to untruncated targets (SMP-Video, Emily) with better calibration than a naive model fit to the same data.* I found no published TikTok popularity work that models this selection. That is a statement about what I searched, not a novelty guarantee (see section 3).

4. **Do not expect content embeddings to move point accuracy much.** The SMP 2025 video-track winner reports MAPE 0.1754. Removing video embeddings raises it to 0.1818, and removing user-profile features raises it to 0.3010 [Ye et al. 2025]. Emily's own leakage-safe evaluation in this repo has the creator baseline (MAE 0.91 on log views) beating every model tried (MAE 1.27–1.40) (`reports/evaluation_results.csv`). The project should be designed to produce a defensible result if that happens again, and it does.

---

## 1. Executive summary

Ordinary view-count regression on TikTok is a weak project on its own. The irreducible noise is large [Salganik et al. 2006; Martin et al. 2016], creator popularity dominates the explained variance, and a strong creator baseline is hard to beat. Predicting $p(Y\mid X)$ instead of $\mathbb E[Y\mid X]$ is a *better-posed* target here, for three concrete reasons, none of which is that it sounds more sophisticated:

- **The noise is the object of interest.** When most variance is irreducible, whether the model is right about *how uncertain* it is matters more than a 2% change in MAE. Proper scoring rules (CRPS, log score) reward that; MAE does not.
- **Decision-relevant quantities are tail functionals**, such as $P(\text{views} > 5\times \text{creator median}\mid x)$. These come directly from a conditional density and only indirectly from a point estimate.
- **The data's selection mechanism is a statement about distributions.** Truncation at a view threshold cannot be handled by a point regressor without bias, but it is a one-line change to a likelihood.

The intuition that we can identify *which latent distribution a video belongs to* should be **rejected as stated**. Each video yields one draw $y_i$. Without strong assumptions, the data identify the conditional law $P(Y\mid X=x)$ as a population object (the distribution across videos sharing covariates $x$). They do not identify a per-video distribution, and they do not identify a discrete "regime" variable as anything more than a parameterization of that conditional law. A $K$-component mixture fit to $y$ produces components whether or not regimes exist. With $n$ in the hundreds of thousands, BIC will happily choose $K>1$ for any skewed, heavy-tailed unimodal density. Section 4 shows how to test this rather than assume it.

**Recommendation:** C as the unifying framework, with D and E as its evaluation, A as an ablation inside it, and B as a pre-registered falsification test.

---

## 2. Problem formulation

### 2.1 Verified data (what each source actually supports)

| Source | Unique posts | Creators | Creator history | Outcome | Time info | Selection | Video files |
|---|---|---|---|---|---|---|---|
| TikTok-10M (`The-data-company/TikTok-10M`) | 10M (1M per shard; ~0.14% exact-duplicate rows) **[audited]** | ~423K in one shard; 14K creators with ≥10 posts in that shard, covering 342K posts **[audited]** | Partial: only that creator's *trending* posts | `play_count` (cumulative) | `create_time`; `collected_time` and `stats_time` both present. `stats_time` is up to ~98 days after `collected_time` **[audited]**, and the stats were most likely read then **[assumption: verify]** | **Left-truncated at ~10^4.5 views**; card says "trending content", "US-based POI" | No; page URLs only |
| EinzzCookie (`EinzzCookie/tiktok-videos-users-info`) | **6,258** (950,656 rows, each post repeated ~150×, same `play_count`) **[audited]** | ~5.6K; median 1 post per creator **[audited]** | None | `statistics.play_count` | `create_time` only (Jun–Sep 2026); no collection time | Unknown; 56% DE region in sampled rows **[audited]** | No |
| SMP-Video (`smpchallenge/SMP-Video`, gated) | ~6K train+test; ~4.5K users [Ye et al. 2025] | ~1.3 posts per user | Weak | Label $s=\log_2(r/d)+1$, $r$ = views, $d$ = days since posting [Ye et al. 2025, §4.1] | 24-month span | Unknown | No; `SMP-Video_video_source.txt` only |
| Emily (`data/raw/tiktok_stats.csv`) | 140 **[audited]** | 1 | Full (if the 140 are all her posts in the window **[verify]**) | Views, plus watch time, traffic sources and more for some rows | 2019-07 to 2023-12; collection date not recorded | Probably none by outcome **[verify]** | 140 local mp4s on Emily's machine, not this Mac |

**Consequences**
- *Only TikTok-10M is large enough for deep learning.* It is truncated, its creator histories are themselves truncated, and post age at `stats_time` has median 204 days and a 1st–99th percentile range of 1–928 days **[audited]**.
- EinzzCookie should be **deduplicated and treated as a small extra test set**, not a pretraining source. `posts._finish` already drops duplicates on `(source, post_id)`, so this is mostly a framing fix.
- SMP-Video is the best **untruncated, multi-creator** external test set, but its target is a rate, not a count. `load_smp` must invert $r = d\cdot 2^{s-1}$, not $d(2^s-1)$.
- Emily is a **single-creator, small-n transfer target**. Any coverage estimate on her 28-post test fold has a 95% binomial CI of roughly ±0.12 at 90% nominal coverage.

### 2.2 Outcome and horizon

Let $V_i(a)$ be cumulative views of post $i$ at age $a$, and let $Y_i = \log(1+V_i(a_i))$ at the observed age $a_i$. We never see trajectories, only one $(a_i, V_i(a_i))$ per post.

**Prediction time is upload time** (no early-view features). The early-trajectory setting [Szabo & Huberman 2008; Cheng et al. 2014] is a different and much easier problem. It cannot be built from these datasets, because none of them has multiple snapshots per post. EinzzCookie's repeated rows carry identical counts **[audited]**.

**Primary target:** creator-relative log views,
$$R_i \;=\; Y_i - b_{c(i)}, \qquad b_c = \operatorname{median}\{Y_j : j \text{ an earlier post of creator } c,\ \text{last } k=10\},$$
modeled as $p(R\mid X, \log a)$. Age enters as a *nuisance covariate*. At prediction time it is set to a reference horizon $a^\star$ (proposed: 90 days), with sensitivity at 30 and 180.

Why creator-relative:
- Between-creator variance dominates. The current code already uses $b_c$ as a baseline, and on Emily it beats every model.
- Predicting $Y$ lets a model score well by recognising big creators. Predicting $R$ asks the scientifically interesting question: *does content explain deviations from this creator's norm?*
- It partially controls for exposure confounding by follower base, which is otherwise unobserved, since follower counts are post-publication.

**Secondary targets:**
1. $Y$ itself with creator-history features as inputs. This is the "practical" forecast and lets us compare against the literature.
2. Tail event $E_i = \{R_i > \log 5\}$, meaning views above 5× the creator's recent median. This is scored with the Brier score and reliability diagrams.
3. SMP's rate target, $\log_2(r/d)+1$, for comparability with SMP results only.

**Rejected or deferred targets:**
- Raw counts as a regression target: the scale is dominated by a few posts.
- Engagement rate: likes/views exists in TikTok-10M, but it is a different construct and the observation window problem is the same.
- Discrete popularity classes: arbitrary thresholds, and they are a lossy functional of $p(R\mid x)$.
- Ranking (Spearman) is reported as a secondary diagnostic.

### 2.3 Inputs, and what is known at upload time

| Feature | Upload-time? | Note |
|---|---|---|
| Caption text, hashtags, duration, posting hour/day | Yes | |
| CLIP frames (first 5 s), Whisper encoder states (first 5 s) | Yes | Content as uploaded |
| Music id, `music_original` | Yes | Sound *popularity* counts must be computed from strictly earlier posts only |
| Creator history $b_c$, prior std, post count, gap since last post | Approximately | Uses *final* views of earlier posts, read at collection time. That is optimistic, because the creator knew less on upload day. In TikTok-10M, history is also restricted to trending posts. Report as a known bias |
| Follower count | **No** | Collection-time value; leaks growth. Off by default (`--use-followers`) |
| `vq_score` (TikTok-10M) | Unknown | Platform-computed quality score; timing unknown. Ablate, never in the primary model |
| Likes, shares, comments | **No** | Post-publication |
| Post age $a$ | Nuisance | Control, fixed to $a^\star$ when forecasting |

### 2.4 Selection: the truncation model

In TikTok-10M, the share of posts per 0.25-wide bin of $\log_{10} V$ jumps from 0.13% in [4.25, 4.5) to 1.6% in [4.5, 4.75) and 6.7% in [4.75, 5.0). Only 0.044% of posts are below 10^4.25 **[audited]**. The working model is left truncation at a known threshold $\ell = \log(1+L)$:
$$p_{\text{obs}}(y\mid x) = \frac{p_\theta(y\mid x)\,\mathbf 1\{y>\ell\}}{1-F_\theta(\ell\mid x)}.$$

For a Gaussian or mixture head, $F_\theta$ is closed-form, so the loss changes by one $\log(1-F)$ term. Caveats:
1. The real inclusion rule is unknown. "Trending" may depend on velocity rather than level, so $\ell$ is estimated, with sensitivity at $\ell\in\{\log 10^{4.25}, \log 10^{4.5}\}$.
2. Truncation is on $Y$, so for the creator-relative target the truncation point becomes $\ell - b_c$, which varies by creator.
3. Creator baselines $b_c$ computed from truncated histories are biased upward. Small creators are hit hardest.

**This is the main identifying assumption of the project and must be stated as one.**

### 2.5 Identifiability

- With one draw per post, $P(R\mid X=x)$ is identified only through smoothness of the model across $x$. Calibration can be tested *marginally* and on *groups* (PIT histograms by bin of $x$ or of predicted scale), never per video.
- Aleatoric and epistemic uncertainty separate only under model assumptions. Ensemble disagreement [Lakshminarayanan et al. 2017] is a heuristic proxy for epistemic uncertainty and is used only as a shift *detector* (H5), not as a decomposition claim.
- Nothing here is causal. Content features correlate with exposure decisions made by the recommender, which are unobserved.

---

## 3. Literature review

**Reading status.** "Checked" means I confirmed authors, venue and abstract from the arXiv or proceedings page in this session. Only Ye et al. 2025 had its full text read, and only via a fetched summary of the HTML version. "From memory" means I'm confident the reference exists but did not re-open it this session. Verify those before citing.

### 3.1 Popularity prediction and its limits
- **Salganik, Dodds & Watts (2006).** Experimental study of inequality and unpredictability in an artificial cultural market. *Science* 311:854–856. doi:10.1126/science.1121066. *Checked.* Social influence increases both inequality and unpredictability of success. This is the foundational argument for irreducible noise.
- **Martin, Hofman, Sharma, Anderson & Watts (2016).** Exploring limits to prediction in complex social systems. WWW 2016, 683–694. arXiv:1602.01013. *Checked.* Best models explain under half the variance in Twitter cascade size, and simulations suggest the ceiling is low even with perfect data. This motivates distributional targets.
- **Cheng, Adamic, Dow, Kleinberg & Leskovec (2014).** Can cascades be predicted? WWW 2014. arXiv:1403.4608. *Checked.* Growth is predictable *given early observations*. That defines the post-publication setting this project explicitly excludes.
- **Szabo & Huberman (2008).** Predicting the popularity of online content. arXiv:0811.0405 (published in CACM 2010 per my memory; not confirmed). *Checked (arXiv).* Log-linear early-to-late view relation.

### 3.2 Short-video and multimodal popularity
- **Ye, Zhang, Wu, Chen, Yu, Yang, Song et al. (2025).** MVP: Winning Solution to SMP Challenge 2025 Video Track. arXiv:2507.00950. *Checked (full-text summary).* SMP-Video has 6K posts and 4.5K users, label $\log_2(r/d)+1$. They use X-CLIP frame features plus PCA, user and metadata features, and CatBoost with Huber loss. In the ablation, removing video embeddings moves MAPE from 0.1754 to 0.1818, while removing the user profile moves it to 0.3010. **This is the strongest available prior for this project: content adds little beyond creator features.**
- **Ye, Zeng, Zhang, Chen, Yu & Song (2026).** OmniTrend: content–context modeling for scalable social popularity prediction. arXiv:2604.26252. *Checked (abstract).* Separates intrinsic appeal from exposure context.
- **Li, Li, Lu, Li, Ma, Krishnan & Wang (2024).** Delving deep into engagement prediction of short videos. ECCV 2024. arXiv:2410.00289. *Checked.* 90K Snapchat UGC videos; argues for watch-percentage engagement targets over view counts.
- **Li et al. (2025).** VQualA 2025 Challenge on Engagement Prediction for Short Videos. ICCV 2025 Workshops. arXiv:2509.02969. *Checked (listing).*
- **Wu, Cheng, Liu, Liu, Zeng & Luo (2019).** SMP Challenge: an overview of the Social Media Prediction Challenge 2019. ACM MM. *Checked (listing).* There is also an overview-and-analysis paper, arXiv:2405.10497 (*listing seen*).
- **3M micro-video popularity (2025).** arXiv:2502.17038 (INFORMS 2024 challenge). *Listing seen.* Retrieval memory bank plus missing-modality handling.

### 3.3 Representations
- **Radford et al. (2021).** CLIP: learning transferable visual models from natural language supervision. ICML. arXiv:2103.00020. *From memory (ID consistent with search).*
- **Radford et al. (2023).** Whisper: robust speech recognition via large-scale weak supervision. ICML 2023. arXiv:2212.04356. *From memory.*
- **Ni et al. (2022).** X-CLIP, the video-adapted CLIP used by MVP. *From memory; verify the exact variant before citing.*

### 3.4 Distributional regression and scoring
- **Bishop (1994).** Mixture density networks. Tech. report NCRG/94/004, Aston University. *Checked (repository record).*
- **Gneiting & Raftery (2007).** Strictly proper scoring rules, prediction, and estimation. *JASA* 102:359–378. *Checked.* CRPS, interval score, log score.
- **Kendall & Gal (2017).** What uncertainties do we need in Bayesian deep learning for computer vision? NeurIPS. arXiv:1703.04977. *Checked.* Heteroscedastic aleatoric losses.
- **Lakshminarayanan, Pritzel & Blundell (2017).** Deep ensembles. arXiv:1612.01474. *Checked (listing).*
- **Ovadia et al. (2019).** Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. NeurIPS. arXiv:1906.02530. *Checked.* Calibration degrades under shift; ensembles hold up best.
- **Rügamer, Kolb & Klein.** Semi-structured deep distributional regression. arXiv:2002.05777. *Checked (listing)*; also the `deepregression` package in *JSS* 105(2), 2023. **Baumann, Hothorn & Rügamer.** Deep conditional transformation models. arXiv:2010.07860. *Checked (listing).*
- **Kuleshov, Fenner & Ermon (2018).** Accurate uncertainties for deep learning using calibrated regression. ICML. arXiv:1807.00263. *From memory.*
- **Grimit, Gneiting, Berrocal & Johnson (2006).** Closed-form CRPS for Gaussian mixtures. *QJRMS*. *From memory.*
- **Clauset, Shalizi & Newman (2009).** Power-law distributions in empirical data. *SIAM Review*. arXiv:0706.1062. *From memory.* Relevant to tail claims.
- **Grinsztajn, Oyallon & Varoquaux (2022).** Why do tree-based models still outperform deep learning on tabular data? NeurIPS D&B. *From memory.* Justifies GBDT baselines.

### 3.5 Mixtures, regimes and identifiability
- **McLachlan & Peel (2000).** *Finite Mixture Models.* Wiley. *From memory.*
- **Hennig (2010).** Methods for merging Gaussian mixture components. *ADAC* 4:3–34. *From memory.* Mixture components are not clusters.

### 3.6 Conformal prediction
- **Romano, Patterson & Candès (2019).** Conformalized quantile regression. NeurIPS. arXiv:1905.03222. *Checked.*
- **Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018).** Distribution-free predictive inference for regression. *JASA*. arXiv:1604.04173. *From memory.*
- **Tibshirani, Barber, Candès & Ramdas (2019).** Conformal prediction under covariate shift. NeurIPS. arXiv:1904.06019. *Checked.*
- **Barber, Candès, Ramdas & Tibshirani (2023).** Conformal prediction beyond exchangeability. *Ann. Statist.* 51(2):816–845. arXiv:2202.13415. *Checked.*
- **Barber, Candès, Ramdas & Tibshirani (2021).** The limits of distribution-free conditional predictive inference. *Information and Inference* 10:455–482. arXiv:1903.04684. *Checked.*
- **Barber, Candès, Ramdas & Tibshirani (2021).** Predictive inference with the jackknife+. *Ann. Statist.* 49(1). *From memory.* Useful for Emily's small n.
- **Gibbs & Candès (2021).** Adaptive conformal inference under distribution shift. NeurIPS. arXiv:2106.00170. *Checked.*
- **Chernozhukov, Wüthrich & Zhu (2021).** Distributional conformal prediction. *PNAS*. *From memory.* PIT-based scores for density models.
- **Angelopoulos & Bates (2023).** Conformal prediction: a gentle introduction. *FnTML* 16(4). arXiv:2107.07511. *Checked.*

### 3.7 Gaps this project can address
1. Published short-video popularity work, as far as I found, is **point regression** (MAPE, SRC) on SMP-style targets. I found no paper reporting proper scoring rules, PIT calibration or conformal coverage for TikTok view distributions. This is an absence in my search, not a proven gap.
2. No work I found models the **outcome-dependent selection** of scraped "trending" TikTok datasets. Truncated likelihoods are textbook (survival analysis, econometrics), so the contribution is the *application and evaluation of transfer*, not the method.
3. Evidence on content embeddings' *incremental* value over creator features exists only as challenge ablations (MAPE). A clustered-bootstrap estimate with proper scores is a modest but real contribution.

---

## 4. Competing directions

| | Novelty | Rigor attainable | DL relevance | Data feasibility | Compute | Likely to yield a finding | Risk |
|---|---|---|---|---|---|---|---|
| **A** multimodal value | Low (challenge ablations exist) | High | High | Good (TikTok-10M + clips) | Medium (clip download is the bottleneck) | High (likely a *small* effect, measurable) | Low |
| **B** latent regimes | Looks novel; scientifically weak | Low unless framed as a test | Medium | Good | Low | Mostly a negative result | **High** (non-identifiable) |
| **C** conditional densities | Medium | High (proper scores) | High (MDN, heteroscedastic nets, truncated likelihoods) | Good | Low on top of A | High | Medium |
| **D** shift/generalization | Medium | Medium–high (real shifts: creator, time, dataset, selection) | Medium | Good: four sources, different mechanisms | Low | High | Medium (Emily is small) |
| **E** conformal | Low as a method; application new | High (theory is crisp) | Low–medium | Good | Very low | High (coverage under shift will break somewhere) | Low |

**Recommendation:** C as the primary direction, D and E as the evaluation, A as the ablation, B as the falsification test. B is not dropped: a careful, negative regime result is worth a section, and it answers the question you care about honestly.

---

## 5. Research questions and falsifiable hypotheses

**Primary question.** On held-out creators, does a multimodal conditional density model $p_\theta(R\mid x)$ give better proper scores than (i) the creator baseline with a fitted residual distribution and (ii) a metadata-plus-history density model? Do its predictive distributions remain calibrated under creator, temporal and cross-dataset shift?

Pre-registered primary metric: **mean CRPS of $R$ on the creator-disjoint TikTok-10M test set**, paired differences, with a 95% creator-clustered bootstrap CI.

- **H1 (content value).** Adding text+video+audio embeddings to the metadata+history model lowers CRPS. *Falsified if* the 95% CI of $\Delta\text{CRPS}$ contains 0, or the effect is below 1% of baseline CRPS (set as the materiality threshold in advance). Prior expectation: small and positive, with text > video > audio.
- **H2 (distributional form).** A heteroscedastic Gaussian beats a homoscedastic one in log score. A $K\ge2$ MDN beats a heteroscedastic Gaussian *only in the upper tail* (tail Brier score, upper-quantile pinball), not in median MAE. *Falsified if* the MDN adds nothing anywhere (then the simpler head is the result), or if it wins in the median only.
- **H3 (regimes are not regimes).** The marginal of $R$ is fit as well by a single skewed, heavy-tailed family (skew-$t$ or sinh-arcsinh) as by a Gaussian mixture, in held-out log-likelihood. A parametric bootstrap from the fitted skew-$t$ reproduces BIC's choice of $K>1$. *Falsified (regimes supported) if* a mixture beats skew-$t$ out of sample by a margin outside bootstrap variation, components are stable across bootstrap resamples (ARI > 0.8) and across time halves, and $P(Z\mid x)$ beats the same-coarsened conditional density at predicting membership.
- **H4 (selection).** On SMP-Video and Emily, truncation-aware training gives better calibration (PIT KS distance, 80/90% coverage) than naive training on the same TikTok-10M data. *Falsified if* no difference or worse. Note this tests the truncation model jointly with transfer.
- **H5 (shift detection).** Raw interval coverage drops under temporal and cross-dataset shift. Split-conformal/CQR calibrated in-domain does not restore it. Calibration on a small target-domain set restores marginal coverage at a measurable width cost. Ensemble disagreement and a domain classifier's score each predict where coverage fails (AUROC > 0.6 for "miss" vs "cover").

---

## 6. Data verification checklist (before any training run)

Partly done here (✔); the rest is required.

- ✔ TikTok-10M schema, size, truncation, creator counts, duplicates (one shard). Still to do: all 10 shards for `user_id`, `create_time`, `stats_time`, `play_count`. That's a column-subset read of about 40 MB per shard, so no full download is needed. Then confirm whether shards are random or creator-partitioned.
- ✔ EinzzCookie duplication and missing follower counts.
- ☐ **`stats_time` vs `collected_time`**: confirm which timestamp `play_count` belongs to. The current loader uses `collected_time`, which understates age by up to ~98 days.
- ☐ SMP-Video, after Jason accepts the gate and logs in: fields per file, whether raw views exist, how $d$ is defined, whether `post_time` exists, user fields (followers are post-publication), and the train/test split mechanism.
- ☐ Emily: are the 140 posts *all* her posts in 2019–2023, or a selection? When were stats read? (Ask her.)
- ☐ Clip availability: fetch a random sample of 300 TikTok-10M URLs and measure the success rate and latency before committing to 10K+.
- ☐ Near-duplicates: CLIP cosine > 0.95 across splits; same `music_id` + caption reposts.
- ☐ Licensing: SMP license (research use; also bans using it to build or improve foundation generative AI models, which doesn't apply here). TikTok-10M is "other" with no stated terms. EinzzCookie is MIT. Clip downloads conflict with TikTok's terms (already noted in `scale/README.md`); keep clips transient.

---

## 7. Proposed architecture

### 7.1 Model ladder (each rung must beat the one below on the primary metric to justify the next)

| ID | Model | Distribution | Purpose |
|---|---|---|---|
| M0 | Global median of $R$ (= 0 by construction) + empirical residual quantiles | Unconditional | Floor |
| M1 | Creator baseline $b_c$ + creator-pooled residual distribution (empirical, or $t$ fit) | Location-only | **The baseline to beat**; currently wins on Emily |
| M2 | Ridge / linear quantile regression on tabular features | Gaussian / quantile | Linear reference |
| M3 | GBDT on tabular features (+ PCA-64 of each embedding): LightGBM quantile objective at 19 levels, or NGBoost-Normal | Quantiles / Gaussian | Strong tabular baseline [Grinsztajn et al.] |
| M4 | MLP, tabular only, heteroscedastic Gaussian (current `FusionNet`, modalities="") | Gaussian | Neural tabular reference |
| M5 | M4 + caption text (CLIP text) | Gaussian | Text increment |
| M6 | M5 + video (CLIP frames) + audio (Whisper encoder pool), late fusion with modality dropout (current `FusionNet`) | Gaussian | Full content increment |
| M7 | M6 with heads: Student-$t$ / sinh-arcsinh; MDN $K\in\{2,3\}$; dense quantile (19 levels) | Flexible | Distributional form (H2) |
| M8 | M6/M7 with **truncated likelihood** | As above | Selection (H4) |
| M9 | Deep ensemble ($M=5$) of the best of M6–M8 | Mixture of members | Epistemic proxy, shift detection (H5) |
| (ext) | Creator context encoder, gated fusion, Mixture-of-Experts | — | Only if M6 > M4 materially |

### 7.2 Mathematical forms

Encoders: $h_m = \phi_m(e_m)\in\mathbb R^{256}$ for $m\in\{\text{tab},\text{text},\text{video},\text{audio}\}$, with learned missing-vectors and modality dropout $p=0.2$ (as implemented). Then $h=\psi([h_{\text{tab}};h_{\text{text}};h_{\text{video}};h_{\text{audio}}])$.

- Heteroscedastic Gaussian: $R\mid x\sim\mathcal N(\mu(h),\sigma^2(h))$.
- MDN: $p(r\mid x)=\sum_k\pi_k(h)\,\mathcal N(r;\mu_k(h),\sigma_k^2(h))$. Use $\sigma_k\ge\sigma_{\min}=0.05$, entropy monitoring, and report $\bar\pi$ (collapse check; already logged as `mdn_weights`).
- Sinh-arcsinh (Jones & Pewsey 2009; *from memory*): a 4-parameter skew/tail family. It is the right *unimodal* competitor to MDN for H2/H3.
- Truncated: $\mathcal L_i = -\log p_\theta(y_i\mid x_i) + \log\big(1-F_\theta(\ell\mid x_i)\big)$ for TikTok-10M rows. Rows from untruncated sources use the plain likelihood.
- Quantile head: pinball loss over $\tau\in\{0.05,0.10,\dots,0.95\}$ with monotone increments (as implemented, but at 19 levels so CRPS can be approximated as $\text{CRPS}\approx 2\,\overline{\text{pinball}}$).

**Upload-time prediction:** evaluate at $\log a = \log a^\star$. For the *evaluation* of a model trained with $\log a$ as an input, score at each post's actual $a_i$ (the observable), and separately report the $a^\star$ forecast's sensitivity.

### 7.3 What not to build, and why
- **Cross-modal attention over frame tokens, temporal transformers, end-to-end or adapter fine-tuning of CLIP/Whisper.** The expected gain is bounded by MVP's +0.006 MAPE for video embeddings. Cost: storing per-frame tokens (10×512 per clip) and GPU fine-tuning. Revisit only if M6−M5 is material *and* the learning curve (E7) is still rising.
- **A unified "regime + density + hierarchy" model.** Regime probabilities are a functional of the MDN; a hierarchy over creators with ~1 trending post each (most TikTok-10M creators) has nothing to pool.
- **Whisper transcription in the 5 s window.** Five seconds carries too little speech for transcripts to matter. Whisper *encoder* states are a mixed acoustic-linguistic representation, not a dedicated acoustic one; label them as such. A dedicated audio model (CLAP, BEATs) is an extension.

---

## 8. Experimental protocol

### 8.1 Splits (all on TikTok-10M after creator-sampled loading)
- **S1 creator-disjoint** (primary; implemented in `scale.data.creator_split`, hash-based 80/10/10).
- **S2 temporal + creator-disjoint**: train on `create_time` < 2024-07-01, validate on 2024-07 to 2024-12, test on ≥ 2025-01, with creators disjoint.
- **S3 cross-dataset**: train on TikTok-10M, test on SMP-Video (rate target converted), EinzzCookie-dedup and Emily, with no target data used for fitting.
- **S4 small-target adaptation**: Emily chronological 60/20/20 (implemented in `scale.finetune`). Also a jackknife+/CV+ variant, since the 28-post test fold is too small for a 3-way split to be informative.
- **S0 random split**: reported only as an optimistic reference, to show the leakage gap.

### 8.2 Leakage controls
- Creator-history features use strictly earlier posts (implemented).
- `global_median` imputation in `features.add_features` uses all rows, including test creators. Compute it on train creators only (small fix).
- Music-popularity features (if added) are counted from earlier posts only.
- Near-duplicate removal across splits (CLIP cosine > 0.95).
- Follower count and `vq_score` excluded from the primary model.

### 8.3 Metrics

| Family | Primary | Secondary diagnostics |
|---|---|---|
| Point | — | MAE of predictive median on $R$; Spearman; MAE on $Y$ for comparability |
| Distribution | **CRPS** | Log score (NLL, same scale for all density heads); 19-level pinball; PIT histogram + KS distance; central 50/80/90% coverage and width; interval score |
| Tail | Brier score for $E=\{R>\log 5\}$ | Reliability diagram; upper-quantile (0.9, 0.95) pinball |
| Regimes (H3) | Held-out log-likelihood of mixtures vs skew-$t$ | BIC curve vs parametric-bootstrap null; bootstrap ARI; component drift across time halves |
| Shift | ΔCRPS and Δcoverage from S1 to S2 to S3 | Coverage by creator-popularity decile, duration bin and source (group-conditional) |

For the quantile head, NLL is undefined, so compare heads on CRPS and pinball, which are well defined for all of them.

### 8.4 Statistical testing
- All comparisons are **paired** on the same test rows.
- **Creator-clustered bootstrap** (resample creators, 2,000 replicates) for every Δmetric CI. Posts within a creator are dependent.
- Holm correction across the five pre-registered hypotheses. Everything else is labelled exploratory.
- Seeds: 3 seeds per configuration for neural models; report the mean, with seed spread alongside the bootstrap CI.
- Coverage CIs: exact binomial (Clopper–Pearson) per group. On Emily, report them; don't interpret point coverage.

### 8.5 Conformal procedures
- **Split CQR** [Romano et al.] on the quantile and MDN heads. **Distributional conformal** [Chernozhukov et al.] for density heads, with score $|F_\theta(y\mid x)-\tfrac12|$.
- **Exchangeability unit = creator.** Posts within a creator are not exchangeable with each other's calibration residuals. Calibrate on *one random post per calibration creator* (exchangeable if creators are iid draws), or use all posts and report the guarantee as approximate.
- **Mondrian** (group-conditional) by creator-popularity tercile, to approximate conditional coverage. Exact conditional coverage is impossible distribution-free [Barber et al. 2021, *Inf. Inference*].
- **Weighted conformal** for S3 [Tibshirani et al. 2019], with density ratios from a domain classifier on embeddings. Report it as valid *only under covariate shift*. Truncation is label-dependent selection, which violates that.
- **ACI** [Gibbs & Candès 2021] on the chronological stream in S2, as the online alternative.
- In the write-up, state: marginal $1-\alpha$ coverage holds under exchangeability of calibration and test scores; none of it is guaranteed under S2/S3; empirical coverage under shift is a *measurement*, not a guarantee.

---

## 9. Experiment matrix (priority order)

| # | Experiment | Establishes | Does *not* establish | Cost |
|---|---|---|---|---|
| E0 | Data audit tables + truncation plot + age curve | Feasibility; selection | — | Done in part |
| E1 | M0–M4 on S1, Gaussian head, text-free | Baseline ladder; whether a neural net beats GBDT on tabular | Content value | CPU minutes |
| E2 | M5, M6 vs M4 (H1), Gaussian head, S1 | Incremental value of text, video, audio | Causality; value for unseen *content types* | Embedding cost |
| E3 | Head comparison on M6: homoscedastic, Gaussian, sinh-arcsinh, MDN-2, MDN-3, quantile-19 (H2) | Whether flexible densities pay | That the components mean anything | Cheap |
| E4 | Regime analysis on $R$ (H3) | Whether "regimes" survive a skew-$t$ null | Mechanism, even if they survive | CPU |
| E5 | Truncated vs naive likelihood, evaluated on S3 (H4) | Selection correction transfers | Exact inclusion rule | Cheap |
| E6 | Conformal: split, CQR, DCP, Mondrian, weighted, ACI on S1–S4 (H5) | Coverage behaviour under each shift | Conditional guarantees | Cheap |
| E7 | Learning curve: n_clips ∈ {2.5K, 5K, 10K, 20K} for M6−M5 | Whether scaling video is worth it | Asymptotic value | Reuses embeddings |
| E8 | Modality leave-one-out on M6 (text, video, audio each removed) | Which modality carries the increment | Interaction effects beyond pairwise | Cheap |
| E9 | Deep ensemble + domain classifier as shift detectors (H5) | Can we flag where intervals fail | Calibrated epistemic uncertainty | 5× training |
| E10 | Emily adaptation: zeroshot, shift, head, full (+CQR, jackknife+) | Practical transfer for one creator | Generalization to other creators | Cheap |

Highest information per compute: **E1, E3, E4, E5, E6** (no new embeddings), then **E2+E7** (shared embeddings).

---

## 10. Compute and feasibility

**This Mac:** M1, 8 GB RAM, **7.7 GB free disk** **[audited]**. Disk, not compute, is the binding local constraint. Store embeddings as float16 (1 KB per modality per post) and delete clips after embedding (already supported).

The estimates below are **[assumption]**s, labelled by basis:

| Step | Unit cost (assumed) | 10K clips / 200K captions | 100K clips / 1M captions |
|---|---|---|---|
| TikTok-10M metadata, column subset | ~40 MB per shard | 0.4 GB, minutes | Same |
| Caption embeddings, CLIP text, MPS | ~1–2K captions/s | 2–4 min; 200 MB fp16 | 10–20 min; 1 GB |
| Clip fetch, yt-dlp, 5 s, 2 workers, 1.5 s sleep | ~1–2 s/clip effective; failure rate unknown (measure first) | 3–6 h | 30–60 h, likely rate-limited |
| CLIP frames (10 frames/clip), MPS | ~100–200 img/s | 8–17 min | 1.5–3 h (≤10 min on an A10G-class GPU) |
| Whisper-base encoder, CPU (pads to 30 s) | ~0.3–0.6 s/clip | 1–2 h | 8–17 h (≪1 h on GPU) |
| Train FusionNet, 200K rows | CPU/MPS | Minutes per run | Tens of minutes per run |

**Scenarios**
- **Low (Mac only):** about 200K TikTok-10M posts by creator sampling (all with text), 10K with clips, plus SMP (6K, clips if available), EinzzCookie-dedup (6K) and Emily (140). Enough for E1–E10. **This is the recommended course-project scope.**
- **Medium (one cloud GPU, about $0.5–1/h):** 1M captions, 50K clips. Only if E7 shows the video increment still rising at 20K.
- **High (500K+ clips):** not recommended. Terms-of-service exposure, mostly download-bound, and the expected gain is small.

**Staged gates**
- **Stage 1** (≤10K clips): E0–E6, E8. Gate: proceed to 20K only if M6−M5 ΔCRPS > 0 with the bootstrap CI excluding 0, *or* the E7 slope between 5K and 10K is positive.
- **Stage 2** (20K): E7, E9, E10. Gate to medium: projected gain at 50K exceeds the 1% materiality threshold.

---

## 11. Minimum viable project (what a strong final submission needs)

1. Verified data audit, including the truncation and duplication findings.
2. Baselines M0–M4 on S1.
3. Frozen-embedding multimodal model M6 (text + 10K-clip video/audio) vs M4, H1 with clustered CIs.
4. One distributional extension: heteroscedastic Gaussian vs MDN-2 vs quantile-19, evaluated with CRPS, PIT and tail Brier (H2).
5. One rigorous shift test: S2 temporal + creator-disjoint, *and* S3 to SMP-Video (if access arrives; Emily otherwise).
6. Split-CQR with coverage reported in-domain vs under shift, with group-conditional coverage (H5, partial).
7. A report that states the identification assumptions (2.4, 2.5) and treats a null H1 as a result.

## 12. Ambitious extension (only after the MVP works)

- Truncated-likelihood training and the transfer test (H4). This is the strongest candidate for the "interesting" result; promote it into the MVP if time allows.
- Regime falsification analysis (H3) in full.
- Deep ensembles and shift detection (H5 complete).
- **Creator context encoder:** a DeepSets/attention encoder over a creator's earlier posts (their embeddings and $R$ values) gives a learned random effect $u_c$, with cold-start falling back to the prior. It replaces the hand-built $b_c$ statistics and is a genuine hierarchical model. It is only meaningful for creators with ≥5 posts, about 35K creators per TikTok-10M shard **[audited]**.
- Weighted conformal for S3 and ACI for S2.
- Dedicated audio embeddings (CLAP/BEATs) vs the Whisper encoder.
- Medium-scale clips, if gated in.

---

## 13. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Truncation model misspecified (selection on velocity, not level) | High | Biased densities; H4 confounded | Sensitivity over $\ell$; report H4 as a joint test; age-stratified truncation check (the 1st percentile of $\log_{10} V$ is 4.49 for posts aged 0–7 days vs 4.74 for posts older than 180 days **[audited]**, so the edge moves with age, which suggests selection on views at some earlier time or on velocity) |
| Wrong timestamp for views (`collected_time` vs `stats_time`) | Medium | Age control wrong by up to ~98 days | Verify; switch the loader to `stats_time` if confirmed |
| Clip fetch failure or rate limits | High | Smaller AV set | Measure on 300 URLs first; MVP works with text-only content if needed; modality dropout already handles missing AV |
| SMP access delayed | Medium | Lose the cleanest S3 target | EinzzCookie-dedup (6K) and Emily are fallbacks |
| Creator baseline unbeatable | Medium–high | H1 null | Pre-registered; frame as "content adds ≤ X% CRPS with CI", which is still a result |
| MDN collapse / degenerate components | Medium | Spurious H2 | σ floor, weight monitoring, multiple seeds, compare against sinh-arcsinh |
| Arbitrary "regimes" | High | Over-claiming | H3 null model; never call components mechanisms |
| Emily too small for coverage claims | Certain | Wide CIs | Clopper–Pearson CIs; jackknife+; present as a case study |
| Disk space (7.7 GB) | High | Pipeline stalls | fp16 embeddings; column-subset parquet reads; delete clips |
| Time | Medium | Ambitious items cut | Section 11 is self-contained |

---

## 14. Critical self-review (as a skeptical reviewer)

1. **"Your training data is a censored sample of hits; nothing you learn is about TikTok."** Partly fair. The answer is H4. If truncation-aware training transfers better to untruncated SMP/Emily, the selection model is doing real work. If not, the paper states that conclusions are about trending posts only. *Negative result value:* it documents a pitfall in a dataset others will use.
2. **"Creator-relative targets in TikTok-10M are computed from truncated histories, so $b_c$ is biased."** Correct. Report it, compare against the alternative of using $Y$ with history features as inputs (secondary target), and check on SMP/Emily where history is untruncated.
3. **"Your 'distribution' is just a Gaussian with a learned variance."** If E3 shows MDN ≈ Gaussian, that *is* the finding: TikTok residual uncertainty is well described by a heteroscedastic unimodal law, and regime language is unwarranted.
4. **"CRPS gains of 1% are not meaningful."** The materiality threshold is set in advance, and tail Brier/reliability is reported where differences matter for decisions.
5. **"Conformal under shift is invalid, so why report it?"** Because measuring *how* it fails, and whether a cheap detector anticipates the failure, is the point of H5. No guarantee is claimed beyond exchangeability.
6. **"The age covariate conflates growth with cohort and algorithm change."** Yes. S2 isolates time, and age is a nuisance with a stated reference horizon. Trajectory data that would separate them does not exist in any source.
7. **"Novelty?"** Claimed only as: first (that I found) proper-score, selection-aware, shift-evaluated distributional study of TikTok views. Not a new method.

**If complex models underperform simple baselines,** the project still answers: (a) how much distributional information is available at upload time; (b) whether flexible heads are warranted; (c) how coverage behaves under four real shifts. A rigorous "the creator baseline plus a well-calibrated residual distribution is as good as it gets" is a publishable-quality negative result for a course project, and consistent with [Martin et al. 2016].

---

## 15. Implementation handoff for Claude Code

Builds on the existing `scale/` package. Each milestone ends with tests that pass on synthetic data (`tests/`) before it is run on real data.

### Repository changes

```
scale/
  posts.py         FIX  tiktok10m: views timestamp = stats_time (after verification); keep latest row per id
                   ADD  load_smp (r = d * 2**(s-1)), einzz dedup assertion, --smp in build_posts
  features.py      FIX  global_median from train creators only (pass split or compute after split)
                   ADD  R target (y_rel = y_log - creator_baseline), truncation point per row
  audit.py         NEW  per-source tables: n, creators, posts/creator, view quantiles, truncation plot, age curve
  model.py         ADD  StudentT / SinhArcsinh heads; truncated wrapper (log(1-F(l|x)) term); 19-level quantiles
  metrics.py       ADD  crps (Gaussian closed form; mixture closed form; quantile approx), pit, pit_ks,
                        tail_brier, group_coverage
  conformal.py     NEW  split, cqr (move from finetune.py), dcp, mondrian, weighted (domain-classifier ratios),
                        aci, jackknife_plus; one-post-per-creator calibration sampler
  baselines.py     NEW  M0, M1 (creator baseline + residual dist), M2 ridge/linear-quantile, M3 LightGBM-quantile/NGBoost
  regimes.py       NEW  GMM K=1..6 vs skew-t/sinh-arcsinh, held-out LL, parametric bootstrap BIC null, bootstrap ARI
  bootstrap.py     NEW  creator-clustered paired bootstrap for any metric difference; Holm correction
  splits.py        NEW  S0-S4 definitions (creator hash, temporal cutoffs, cross-source)
  run_grid.py      NEW  reads experiments.yaml, runs E1-E10, appends to reports/scale_results.csv with config hash
experiments.yaml   NEW  the matrix in section 9
```

### Milestones and acceptance criteria

| M | Deliverable | Acceptance |
|---|---|---|
| 1 | Loader fixes + `audit.py` | Audit table reproduces this memo's numbers on shard 0; EinzzCookie loads to 6,258 rows; unit test: `load_smp` inverts a synthetic $s$ to $r$ exactly |
| 2 | Metrics + bootstrap | `crps` matches `properscoring`/numeric integration within 1e-4 on Gaussian and 3-component mixtures; bootstrap CI covers the true Δ in ≥93/100 synthetic replicates at 95% |
| 3 | Baselines + splits | E1 runs end to end; S1 creator sets are disjoint (asserted); S2 dates respected (asserted) |
| 4 | Heads incl. truncated | On synthetic truncated lognormal data ($n=50$K, $\ell$ known), the truncated Gaussian head recovers $\mu,\sigma$ within 2%, and the naive head is visibly biased (test asserts both) |
| 5 | Conformal | On exchangeable synthetic data, split/CQR/DCP coverage is within ±1.5% of nominal over 200 repeats; Mondrian per-group coverage is within ±3% |
| 6 | Regimes | On synthetic skew-$t$ data, the bootstrap null reproduces BIC's $K>1$ choice; on a true 3-component mixture, held-out LL prefers the mixture |
| 7 | Real-data stage 1 | E1–E6, E8 rows in `reports/scale_results.csv` with CIs; figures: PIT, reliability, coverage by group, truncation |
| 8 | Stage 2 + Emily | E7, E9, E10; final tables |

### Immediate changes to the data run already in progress
- Treat EinzzCookie as **6K posts**, not a pretraining source.
- Check `stats_time` before computing `log_age_days` for TikTok-10M.
- Use $r = d\cdot 2^{s-1}$ for SMP if raw views are absent (MVP's definition). Confirm against the dataset files once access is granted.
- Before fetching 20K clips, measure the fetch success rate on 300 URLs.
- Keep `--max-posts-per-source` for TikTok-10M, but read only the needed columns from all 10 shards rather than downloading full shards. Disk is tight.
