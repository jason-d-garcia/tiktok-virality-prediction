"""
Pretrain the general model on every source except Emily, with creator-disjoint
validation and test sets.

    python -m scale.train --posts data/scale/posts.parquet \
        --emb-dirs data/scale/emb/text data/scale/emb/av \
        --modalities video,audio,text --head mdn --out models/scale/mdn_all.pt

Run once per head and modality set to fill the comparison table; every run
appends one row to reports/scale_results.csv.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scale.data import Scaler, batches, creator_split, load_modalities, tabular_columns, to_tensors
from scale.features import creator_baseline
from scale.metrics import score_forecast
from scale.model import FusionNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = Path(os.environ.get("SCALE_RESULTS", PROJECT_ROOT / "reports" / "scale_results.csv"))


def init_location(model, mean_log):
    """Start every head near the training mean so early steps aren't spent finding it."""
    head = model.head
    with torch.no_grad():
        bias = head.out.bias
        name = type(head).__name__
        if name in ("GaussianHead", "NegBinHead"):
            bias[0] = mean_log
        elif name == "MDNHead":
            bias[head.k:2 * head.k] = mean_log + torch.linspace(-1.5, 1.5, head.k)
        elif name == "QuantileHead":
            bias[0] = mean_log - 2.0


@torch.no_grad()
def predict(model, tab, emb, y_log, y_count, batch_size=4096):
    model.eval()
    qs, nll = [], []
    for idx in batches(len(tab), batch_size, shuffle=False):
        p = model(tab[idx], {m: v[idx] for m, v in emb.items()})
        qs.append(model.head.quantiles(p).cpu())
        nll.append(model.head.nll_count(p, y_log[idx], y_count[idx]).cpu())
    return torch.cat(qs).numpy(), torch.cat(nll).numpy()


def fit(model, train, val, epochs, lr, batch_size, patience, device, params=None, log_every=1):
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    best, best_state, bad = float("inf"), None, 0
    tab, emb, y_log, y_count = train
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for idx in batches(len(tab), batch_size, shuffle=True, rng=rng):
            p = model(tab[idx], {m: v[idx] for m, v in emb.items()})
            loss = model.head.loss(p, y_log[idx], y_count[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(idx)
        val_loss = evaluate_loss(model, val, batch_size)
        if epoch % log_every == 0:
            print(f"epoch {epoch:3d} train {total / len(tab):.4f} val {val_loss:.4f}")
        if val_loss < best - 1e-4:
            best, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


@torch.no_grad()
def evaluate_loss(model, data, batch_size):
    model.eval()
    tab, emb, y_log, y_count = data
    total = 0.0
    for idx in batches(len(tab), batch_size, shuffle=False):
        p = model(tab[idx], {m: v[idx] for m, v in emb.items()})
        total += model.head.loss(p, y_log[idx], y_count[idx]).item() * len(idx)
    return total / len(tab)


def prepare(df, emb_dirs, modalities, use_followers):
    cols = tabular_columns(use_followers)
    tab = df[cols].to_numpy(dtype=np.float64)
    emb = load_modalities(df, emb_dirs, modalities) if modalities else {}
    return tab, emb, cols


def append_result(row):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(RESULTS_PATH) if RESULTS_PATH.exists() else pd.DataFrame()
    pd.concat([rows, pd.DataFrame([row])], ignore_index=True).to_csv(RESULTS_PATH, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts", required=True)
    parser.add_argument("--emb-dirs", nargs="*", default=[])
    parser.add_argument("--modalities", default="", help="comma list from video,audio,text; empty = metadata only")
    parser.add_argument("--head", default="gaussian", choices=["gaussian", "mdn", "quantile", "negbin"])
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--use-followers", action="store_true",
                        help="followers are read at collection time, so this is post-publication info")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    modalities = [m for m in args.modalities.split(",") if m]
    df = pd.read_parquet(args.posts)
    df = df[df["source"] != "emily"].reset_index(drop=True)
    df["baseline"] = creator_baseline(df)
    split = creator_split(df["creator_id"])
    print({s: int((split == s).sum()) for s in ("train", "val", "test")}, "posts by split")

    tab, emb, cols = prepare(df, args.emb_dirs, modalities, args.use_followers)
    tr = split == "train"
    tab_scaler = Scaler().fit(tab[tr])
    emb_scalers = {m: Scaler().fit(v[tr]) for m, v in emb.items()}
    tab_s = tab_scaler.transform(tab)
    emb_s = {m: emb_scalers[m].transform(v) for m, v in emb.items()}
    for m, v in emb.items():
        print(f"{m}: {np.mean(~np.isnan(v).any(1)):.1%} of posts have it")

    y_log = df["y_log"].to_numpy(np.float32)
    y_count = df["views"].to_numpy(np.float32)
    sets = {
        s: to_tensors(tab_s[split == s], {m: v[split == s] for m, v in emb_s.items()},
                      y_log[split == s], y_count[split == s], args.device)
        for s in ("train", "val", "test")
    }

    head_kw = {"n_components": args.n_components} if args.head == "mdn" else {}
    model = FusionNet(len(cols), modalities, args.head, **head_kw).to(args.device)
    init_location(model, float(y_log[tr].mean()))

    start = time.time()
    fit(model, sets["train"], sets["val"], args.epochs, args.lr, args.batch_size, args.patience, args.device)

    q_pred, nll = predict(model, *sets["test"])
    te = split == "test"
    result = score_forecast(y_log[te], q_pred, nll, df.loc[te, "baseline"].to_numpy())
    result.update({
        "stage": "pretrain", "head": args.head, "modalities": "+".join(["tab"] + modalities),
        "followers": args.use_followers, "n_train": int(tr.sum()), "seconds": round(time.time() - start),
    })
    if args.head == "mdn":
        with torch.no_grad():
            model.eval()
            p = model(sets["test"][0][:4096], {m: v[:4096] for m, v in sets["test"][1].items()})
            result["mdn_weights"] = " ".join(f"{w:.2f}" for w in model.head.mixture_weights(p).tolist())
    print(json.dumps(result, indent=2))
    append_result(result)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"tab_dim": len(cols), "modalities": modalities, "head": args.head, "head_kw": head_kw,
                   "use_followers": args.use_followers},
        "tab_scaler": tab_scaler.state(),
        "emb_scalers": {m: s.state() for m, s in emb_scalers.items()},
    }, args.out)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
