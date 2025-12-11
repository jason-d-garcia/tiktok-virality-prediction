# 📈 Multimodal TikTok Virality Prediction

This project builds and analyzes a **multimodal machine learning system** to predict **log-transformed TikTok views** using a combination of:

- Posting metadata  
- Account momentum features  
- Text semantics (CLIP + Whisper)  
- Limited early audiovisual energy signals  

The goal is to quantify **how much TikTok performance is predictable** and to analyze **which modalities drive virality under different content regimes**.

---

## 🔍 Project Overview

TikTok analytics typically provide only coarse feedback (views, likes) without explaining *why* a post succeeds. This project explores whether post metadata and multimodal signals can meaningfully predict video performance and how those signals vary across different types of content.

Key challenges addressed:

- Extreme class imbalance & heavy-tailed outcomes
- Small-n, high-p regime (140 videos × 2,000+ features)
- Multimodal fusion of vision, audio, and text
- Nonlinear interpretability via SHAP

---

## 📊 Dataset

- 140 TikTok videos from a creator with ~1M followers
- 36+ manually curated features per post
- All metadata and engagement statistics collected by hand directly from TikTok Creator Studio
- Modalities include:
  - Posting time & length
  - Hashtag & caption text
  - Account momentum features
  - CLIP visual embeddings (first 3 seconds)
  - Whisper audio embeddings (first 3 seconds)
  - Low-level audio & motion energy statistics

**Target variable:**
- `log(Views)`

---

## 🧠 Modeling Pipeline

### Feature Engineering
- Cyclical time encoding (sin/cos)
- Rolling momentum windows (7, 14, 28 days)
- Block-wise PCA for dimensionality control

### Models Evaluated
- Ridge Regression  
- Elastic Net  
- XGBoost Regressor  

### Validation
- Time-series cross-validation  
- Expanding-window cross-validation  
- Comparison against metadata-only baseline

---

## 📈 Key Results

- **Best performance achieved by XGBoost with block PCA**
- Linear models captured weak global trends but failed on nonlinear regimes
- Semantics + momentum dominated predictive power
- Early audiovisual energy alone was not sufficient for virality prediction
- Calibration curves revealed:
  - A predictable normal-performance regime
  - A highly unpredictable viral regime

---

## 🧭 Representation Learning & Interpretability

This project goes beyond prediction by analyzing how different content regimes behave.

### SHAP Analysis
- Global SHAP explanations for XGBoost
- Block-level modality importance
- Cluster-specific SHAP fingerprints

### UMAP + HDBSCAN Clustering
- Learned a low-dimensional representation of videos
- Automatically discovered distinct content regimes
- Each cluster exhibited different feature dependencies

---

## ⚠️ Limitations

- Small sample size (n = 140) limits deep learning feasibility  
- High dimensionality relative to n favors frozen encoders + tree models  
- Viral outliers remain fundamentally difficult to predict  
- Correlational modeling only (no causal guarantees)

---

## 🚀 Future Work

- Mixture or regime-switching models
- Explicit viral-probability modeling
- Qualitative annotation of clusters
- Multi-target prediction (saves, shares, watch-time)
- Larger-scale dataset expansion

---

## 📂 Repository Structure

```text
├── clustering/        # Clustering analysis & UMAP/HDBSCAN results
├── data/              # Raw and processed data
├── models/            # Trained model checkpoints
├── preprocessing/     # Feature engineering & Clip/Whisper Embeddings
├── reports/           # Figures, SHAP plots, calibration curves
├── src/               # Core training & analysis scripts
├── util/              # Helper utilities
├── README.md
└── requirements.txt
```