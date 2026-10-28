"""Assemble model inputs: tabular features + aligned embeddings, with train-fitted scaling."""

import numpy as np
import torch

from scale.embed import load_embeddings
from scale.features import FOLLOWER_FEATURES, TABULAR_FEATURES
from scale.splits import creator_split  # noqa: F401  (re-exported for scale.train)


def tabular_columns(use_followers):
    return TABULAR_FEATURES + (FOLLOWER_FEATURES if use_followers else [])


def load_modalities(df, emb_dirs, modalities):
    keys = (df["source"] + "_" + df["post_id"].astype(str)).tolist()
    out = {m: np.full((len(df), 512), np.nan, dtype=np.float32) for m in modalities}
    for emb_dir in emb_dirs:
        got = load_embeddings(emb_dir, keys, modalities)
        for m in modalities:
            fill = np.isnan(out[m]).any(1) & ~np.isnan(got[m]).any(1)
            out[m][fill] = got[m][fill]
    return out


class Scaler:
    """Per-column mean/std from training rows, NaN-aware."""

    def fit(self, X):
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0) + 1e-6
        self.mean = np.nan_to_num(self.mean)
        self.std = np.nan_to_num(self.std, nan=1.0)
        return self

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)

    def state(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, s):
        obj = cls()
        obj.mean, obj.std = np.array(s["mean"]), np.array(s["std"])
        return obj


def to_tensors(tab, emb, y_log, y_count, device):
    return (
        torch.tensor(tab, device=device),
        {m: torch.tensor(v, device=device) for m, v in emb.items()},
        torch.tensor(y_log, dtype=torch.float32, device=device),
        torch.tensor(y_count, dtype=torch.float32, device=device),
    )


def batches(n, batch_size, shuffle, rng=None):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]
