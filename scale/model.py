"""
Multimodal fusion network with interchangeable distributional heads.

Each modality (tabular, video, audio, text) gets its own encoder. A modality that's
missing for a post (no clip downloaded, silent audio) is replaced by a learned
"missing" vector, and whole modalities are dropped at random during training so
the model also works on the millions of posts with metadata and captions only.

Heads, all predicting the distribution of views:
  gaussian  log1p(views) ~ Normal(mu, sigma)            (lognormal on views)
  mdn       log1p(views) ~ mixture of K Normals         (lognormal mixture)
  quantile  monotone quantiles of log1p(views), pinball loss
  negbin    views ~ NegativeBinomial(mean, dispersion)

Every head exposes quantiles on the log1p scale, so they're scored the same way,
and all but `quantile` give a count-scale NLL.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
EMBED_MODALITIES = ("video", "audio", "text")

# =========================
# ===== ENCODERS =========
# =========================

class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, d, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.missing = nn.Parameter(torch.zeros(d))

    def forward(self, x, present):
        h = self.net(torch.nan_to_num(x))
        return torch.where(present[:, None], h, self.missing.expand_as(h))


class FusionNet(nn.Module):
    def __init__(self, tab_dim, modalities, head, d=256, dropout=0.1, modality_dropout=0.2, **head_kw):
        super().__init__()
        self.modalities = tuple(modalities)
        self.modality_dropout = modality_dropout
        self.tab = nn.Sequential(nn.Linear(tab_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.encoders = nn.ModuleDict({m: ModalityEncoder(512, d, dropout) for m in self.modalities})
        fused = d * (1 + len(self.modalities))
        self.trunk = nn.Sequential(
            nn.Linear(fused, 2 * d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d, d), nn.GELU(),
        )
        self.head = HEADS[head](d, **head_kw)

    def embed(self, tab, emb):
        parts = [self.tab(tab)]
        for m in self.modalities:
            x = emb[m]
            present = ~torch.isnan(x).any(dim=1)
            if self.training and self.modality_dropout > 0:
                present = present & (torch.rand(len(x), device=x.device) > self.modality_dropout)
            parts.append(self.encoders[m](x, present))
        return self.trunk(torch.cat(parts, dim=1))

    def forward(self, tab, emb):
        return self.head(self.embed(tab, emb))

# =========================
# ===== HEADS ============
# =========================

LOG_2PI = math.log(2 * math.pi)


def _log_to_count_nll(log_density, y_count):
    # density of n from the density of log1p(n): f(n) ~= f_log(log1p n) / (1 + n)
    return -(log_density - torch.log1p(y_count))


def _mixture_cdf(x, logits, mu, sigma):
    w = torch.softmax(logits, dim=-1)
    return (w * 0.5 * (1 + torch.erf((x[..., None] - mu) / (sigma * math.sqrt(2))))).sum(-1)


class GaussianHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.out = nn.Linear(d, 2)

    def forward(self, h):
        mu, s = self.out(h).unbind(-1)
        return {"mu": mu, "sigma": F.softplus(s) + 1e-3}

    def log_density(self, p, y_log):
        z = (y_log - p["mu"]) / p["sigma"]
        return -0.5 * (z ** 2 + LOG_2PI) - torch.log(p["sigma"])

    def loss(self, p, y_log, y_count):
        return -self.log_density(p, y_log).mean()

    def nll_count(self, p, y_log, y_count):
        return _log_to_count_nll(self.log_density(p, y_log), y_count)

    def quantiles(self, p, qs=QUANTILES):
        z = torch.tensor([math.sqrt(2) * torch.erfinv(torch.tensor(2 * q - 1)).item() for q in qs],
                         device=p["mu"].device)
        return p["mu"][:, None] + p["sigma"][:, None] * z


class MDNHead(nn.Module):
    def __init__(self, d, n_components=3):
        super().__init__()
        self.k = n_components
        self.out = nn.Linear(d, 3 * n_components)

    def forward(self, h):
        logits, mu, s = self.out(h).chunk(3, dim=-1)
        return {"logits": logits, "mu": mu, "sigma": F.softplus(s) + 1e-3}

    def log_density(self, p, y_log):
        z = (y_log[:, None] - p["mu"]) / p["sigma"]
        comp = -0.5 * (z ** 2 + LOG_2PI) - torch.log(p["sigma"])
        return torch.logsumexp(torch.log_softmax(p["logits"], -1) + comp, dim=-1)

    def loss(self, p, y_log, y_count):
        return -self.log_density(p, y_log).mean()

    def nll_count(self, p, y_log, y_count):
        return _log_to_count_nll(self.log_density(p, y_log), y_count)

    @torch.no_grad()
    def quantiles(self, p, qs=QUANTILES, iters=50):
        lo = (p["mu"] - 6 * p["sigma"]).min(-1).values
        hi = (p["mu"] + 6 * p["sigma"]).max(-1).values
        out = []
        for q in qs:
            a, b = lo.clone(), hi.clone()
            for _ in range(iters):
                mid = (a + b) / 2
                below = _mixture_cdf(mid, p["logits"], p["mu"], p["sigma"]) < q
                a = torch.where(below, mid, a)
                b = torch.where(below, b, mid)
            out.append((a + b) / 2)
        return torch.stack(out, dim=1)

    def mixture_weights(self, p):
        """Mean component weights; one near 1.0 means the mixture collapsed."""
        return torch.softmax(p["logits"], -1).mean(0)


class QuantileHead(nn.Module):
    def __init__(self, d, qs=QUANTILES):
        super().__init__()
        self.q_levels = tuple(qs)
        self.qs = torch.tensor(qs)
        self.out = nn.Linear(d, len(qs))

    def forward(self, h):
        raw = self.out(h)
        # first quantile free, the rest as positive increments: no crossing
        return {"q": torch.cat([raw[:, :1], raw[:, :1] + torch.cumsum(F.softplus(raw[:, 1:]), 1)], 1)}

    def loss(self, p, y_log, y_count):
        return pinball(p["q"], y_log, self.qs.to(y_log.device)).mean()

    def nll_count(self, p, y_log, y_count):
        return torch.full_like(y_log, float("nan"))

    def quantiles(self, p, qs=QUANTILES):
        if tuple(qs) != self.q_levels:
            raise ValueError("QuantileHead only predicts the quantiles it was trained on")
        return p["q"]


class NegBinHead(nn.Module):
    """Mean on the log scale; dispersion r (variance = m + m^2 / r)."""

    def __init__(self, d):
        super().__init__()
        self.out = nn.Linear(d, 2)

    def forward(self, h):
        log_m, r = self.out(h).unbind(-1)
        return {"log_m": log_m, "r": F.softplus(r) + 1e-4}

    def log_pmf(self, p, y_count):
        y = y_count.double()
        r = p["r"].double()
        log_m = p["log_m"].double()
        log_r_m = torch.logaddexp(torch.log(r), log_m)
        return (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
                + r * (torch.log(r) - log_r_m) + y * (log_m - log_r_m)).float()

    def loss(self, p, y_log, y_count):
        # per-count NLL scales with log(views); divide by a constant to keep step sizes sane
        return -self.log_pmf(p, y_count).mean() / 10.0

    def nll_count(self, p, y_log, y_count):
        return -self.log_pmf(p, y_count)

    @torch.no_grad()
    def quantiles(self, p, qs=QUANTILES):
        from scipy.stats import nbinom
        m = torch.exp(p["log_m"]).double().cpu().numpy()
        r = p["r"].double().cpu().numpy()
        prob = r / (r + m)
        out = np.stack([np.log1p(nbinom.ppf(q, r, prob)) for q in qs], axis=1)
        return torch.tensor(out, dtype=torch.float32, device=p["log_m"].device)


HEADS = {"gaussian": GaussianHead, "mdn": MDNHead, "quantile": QuantileHead, "negbin": NegBinHead}


def pinball(q_pred, y, qs):
    diff = y[:, None] - q_pred
    return torch.maximum(qs * diff, (qs - 1) * diff).mean(-1)
