"""
Adapt the pretrained model to Emily and calibrate its intervals.

Emily's posts are split chronologically: oldest 60% to adapt, next 20% to
calibrate, newest 20% held out for the final score. Adaptation modes:

  zeroshot   pretrained model as-is
  shift      zeroshot plus a constant shift = mean residual on the adapt set
  head       freeze the encoders and trunk, retrain the head
  full       fine-tune everything at a small learning rate

Every mode is scored raw and after conformalized quantile regression (CQR) on the
calibration set, at 80% and 90%. Coverage guarantees need the calibration and
test posts to be exchangeable; over time they may not be, which is the point of
reporting both.

    python -m scale.finetune --posts data/scale/posts.parquet --checkpoint models/scale/mdn_all.pt \
        --emb-dirs data/scale/emb/text data/scale/emb/av
"""

import argparse
import copy
import json

import numpy as np
import pandas as pd
import torch

from scale.data import Scaler, load_modalities, tabular_columns, to_tensors
from scale.features import creator_baseline, impute_history
from scale.metrics import score_forecast
from scale.model import QUANTILES, FusionNet
from scale.train import append_result, fit, predict

MODES = ("zeroshot", "shift", "head", "full")


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = FusionNet(cfg["tab_dim"], cfg["modalities"], cfg["head"], **cfg["head_kw"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    scalers = (Scaler.from_state(ckpt["tab_scaler"]),
               {m: Scaler.from_state(s) for m, s in ckpt["emb_scalers"].items()})
    return model, cfg, scalers


def cqr(q_cal, y_cal, q_test, lo_q, hi_q, alpha, qs=QUANTILES):
    """Conformalized quantile regression: widen [lo, hi] by the calibrated non-conformity quantile."""
    lo_i, hi_i = qs.index(lo_q), qs.index(hi_q)
    scores = np.maximum(q_cal[:, lo_i] - y_cal, y_cal - q_cal[:, hi_i])
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    e = np.quantile(scores, level, method="higher")
    out = q_test.copy()
    out[:, lo_i] -= e
    out[:, hi_i] += e
    return out, float(e)


def chrono_split(n, adapt=0.6, cal=0.2):
    a, c = int(n * adapt), int(n * (adapt + cal))
    return np.arange(a), np.arange(a, c), np.arange(c, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--emb-dirs", nargs="*", default=[])
    parser.add_argument("--creator", default="emily", help="creator_id to adapt to")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(0)
    model0, cfg, (tab_scaler, emb_scalers) = load_checkpoint(args.checkpoint, args.device)
    df = pd.read_parquet(args.posts)
    df = df[df["creator_id"] == args.creator].sort_values("create_time").reset_index(drop=True)
    if len(df) < 30:
        raise SystemExit(f"Only {len(df)} posts for {args.creator}")
    df["baseline"] = creator_baseline(df)
    df = impute_history(df, cfg["history_fill"])

    cols = tabular_columns(cfg["use_followers"])
    tab = tab_scaler.transform(df[cols].to_numpy(np.float64))
    emb = load_modalities(df, args.emb_dirs, cfg["modalities"]) if cfg["modalities"] else {}
    emb = {m: emb_scalers[m].transform(v) for m, v in emb.items()}
    y_log = df["y_log"].to_numpy(np.float32)
    y_count = df["views"].to_numpy(np.float32)

    adapt, cal, test = chrono_split(len(df))
    # inner split of the adapt set for early stopping
    inner = int(len(adapt) * 0.8)
    part = lambda idx: to_tensors(tab[idx], {m: v[idx] for m, v in emb.items()}, y_log[idx], y_count[idx], args.device)
    sets = {"fit": part(adapt[:inner]), "stop": part(adapt[inner:]), "adapt": part(adapt),
            "cal": part(cal), "test": part(test)}
    print(f"{args.creator}: adapt {len(adapt)}, calibrate {len(cal)}, test {len(test)}")

    for mode in MODES:
        model = copy.deepcopy(model0)
        if mode == "head":
            fit(model, sets["fit"], sets["stop"], args.epochs, 1e-3, 64, 20, args.device,
                params=model.head.parameters(), log_every=50)
        elif mode == "full":
            fit(model, sets["fit"], sets["stop"], args.epochs, 1e-4, 64, 20, args.device, log_every=50)

        q_adapt, _ = predict(model, *sets["adapt"])
        q_cal, _ = predict(model, *sets["cal"])
        q_test, nll = predict(model, *sets["test"])
        if mode == "shift":
            shift = float(np.mean(y_log[adapt] - q_adapt[:, QUANTILES.index(0.5)]))
            q_cal, q_test = q_cal + shift, q_test + shift
            nll = None  # a shifted density's NLL isn't recomputed here

        rows = [("raw", q_test, {})]
        for lo_q, hi_q, alpha in ((0.10, 0.90, 0.2), (0.05, 0.95, 0.1)):
            q_conf, e = cqr(q_cal, y_log[cal], q_test, lo_q, hi_q, alpha)
            rows.append((f"cqr{int((1 - alpha) * 100)}", q_conf, {"cqr_margin": e}))
        for calib, q, extra in rows:
            res = score_forecast(y_log[test], q, nll if calib == "raw" else None, df.loc[test, "baseline"].to_numpy())
            # a cqr row only adjusts one interval; report the one it calibrated
            res.update({"stage": f"adapt:{mode}", "calibration": calib, "creator": args.creator,
                        "head": cfg["head"], "modalities": "+".join(["tab"] + cfg["modalities"]), **extra})
            print(json.dumps({k: res[k] for k in res if k in (
                "stage", "calibration", "mae_median", "baseline_mae", "coverage_80", "coverage_90", "width_80", "width_90")}))
            append_result(res)


if __name__ == "__main__":
    main()
