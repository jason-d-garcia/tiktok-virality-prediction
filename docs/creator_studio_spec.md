# Creator Studio: draft comparison and feedback (spec v0.1)

A small app where a creator enters several versions of the same post (captions,
hooks, thumbnails, lengths, posting times), sees a predicted view range for
each, a ranking that says when versions are actually distinguishable, and
concrete feedback on how to improve each one.

## 1. Who and why

- **User:** one creator (first target: Emily, ~1M followers) choosing between
  drafts before posting.
- **Question it answers:** "Which of these versions should I post, how
  confident is that call, and what should I change?"
- **What it must not do:** pretend to precision it doesn't have. Our honest
  baseline (median of the creator's last 10 posts) has MAE 0.91 log-units, and
  the best model so far only helps ranking. The app leads with ranges, not
  point scores.

## 2. Inputs

A **post plan** = shared context + 2 to 6 **variants**.

| Field | Level | Required | Notes |
|---|---|---|---|
| Creator | context | yes | selects history used for the baseline (Emily's CSV for now) |
| Caption | variant | yes | free text incl. hashtags; hashtags parsed out |
| Hook | variant | yes | the first line said or shown on screen (first 1 to 3 s) |
| Length (s) | variant | yes | |
| Posting time | variant | defaults to context | lets the creator compare time slots |
| Thumbnail / cover | variant | optional | image upload |
| Clip | variant | optional, later | first 5 s; only used once the real model has video embeddings |

A variant only needs to differ in one field. The UI starts from "Variant A",
and "Duplicate" copies it so the creator changes one thing at a time.

## 3. Outputs per variant

1. **Predicted views as a range:** median and an 80% conformal interval,
   shown in views (e.g. "~120K, likely 40K to 900K") and as a bar against the
   creator's recent-posts band.
2. **Relative to your usual:** predicted median vs the creator baseline, as
   "about 1.4x your typical post".
3. **Drivers:** the 3 to 5 features moving this variant up or down vs the
   baseline (SHAP from the real model; signed heuristic contributions from the
   stub).
4. **Feedback:** short critique of hook, caption, hashtags, length and
   thumbnail, plus 2 to 3 concrete edits (section 6).
5. **Suggested rewrites (optional button):** 3 alternative hooks or captions,
   which are added as new variants and scored like the rest.

## 4. Ranking

- Sort by predicted median log views.
- Group variants into **tiers**: two variants are in the same tier when their
  predicted medians differ by less than the **resolution** `delta` (default
  0.15 log10 units, about 1.4x). The UI says "A and C are effectively tied"
  rather than "A wins by 3%".
- Show win probability only once the real model gives a predictive
  distribution; with the stub we show tiers only.
- Open question for the modeling thread: variants share creator and time
  noise, so the uncertainty of a *difference* is much smaller than each
  interval. Calibrating paired differences needs paired data (A/B reposts),
  which we don't have, so v1 uses the fixed `delta` and says so.

## 5. Uncertainty

- **v1 (prototype):** split conformal on the stub's residuals over Emily's
  140 posts in time order (calibrate on older posts, report empirical coverage
  on newer ones). Interval = median plus/minus the conformal quantile of
  |residual|.
- **v2:** plug in the CQR / Mondrian intervals from the "Train virality model
  with conformal intervals" thread; the predictor interface already returns
  `lower`, `upper`, `coverage`.
- Always show the nominal and empirical coverage in a footnote ("80% target,
  78% on held-out posts").

## 6. LLM feedback

Claude reviews each variant with a fixed rubric, grounded in the model's
drivers so the text and the numbers agree:

- **Hook (first 1 to 3 s):** is there a reason to keep watching (question,
  tension, payoff promise)? Under ~10 words?
- **Caption:** clear, matches the hook, has a light call to action.
- **Hashtags:** niche + broad mix, 3 to 6 tags, relevant to the video.
- **Length:** vs what has worked for this creator.
- **Thumbnail (if uploaded, via vision):** readable text, face, contrast.
- **Comparison note:** one paragraph on which variant to post and why, citing
  the tiers, not overclaiming.

Rules: never invent statistics; say "the model thinks" for model-derived
claims; no edits that change what the video is. Without an API key the app
falls back to rule-based tips from the same rubric, so it always runs.

## 7. Architecture

```
app/creator_studio.py      Streamlit UI (variants, upload, results)
creator_studio/schema.py   Variant, PostPlan, Prediction dataclasses
creator_studio/predictor.py
    Predictor protocol: predict(plan) -> list[Prediction]
    StubPredictor: creator baseline + heuristic adjustments + split conformal
creator_studio/ranking.py  tiers by resolution delta
creator_studio/feedback.py Claude reviewer + rule-based fallback
tests/test_creator_studio.py
```

The real model replaces `StubPredictor` by implementing `predict` (and
optionally `explain`); nothing else changes.

## 8. Out of scope for the prototype

Video upload and embedding, TikTok account login, multi-creator support,
storing past plans, deployment. All are straightforward later additions.

## 9. Decisions needed from Jason

1. **UI framework:** Streamlit (fastest, runs locally with `streamlit run`)
   vs a static web page. Recommendation: Streamlit.
2. **LLM in v1:** call Claude for feedback when `ANTHROPIC_API_KEY` is set
   (costs a few cents per plan) vs rule-based only. Recommendation: both, with
   Claude optional.
