#!/usr/bin/env python3
"""
Refit-cadence experiment: how often must the model be updated?
Re-estimation every 7, 28 or 91 days, or once at R = 2186, for ARIMA, SARIMA, ETS, AR, NaiveDOW
(on CPU) and MLP (on GPU, with --nn), at h = 1 and 7. Evaluated on the evaluation-span targets.
Writes output/v2/results/cadence.csv.
"""
import argparse
import time

import numpy as np
import pandas as pd

from common import OUT, EVAL_START, EVAL_END, SEED, BASE_NN, load_series, load_json, ensure_dir
from run_stat import forecast_origins, dow_fn
import stat_models as sm

_, Y, _ = load_series()
CAD = {"once@2186": None, "91d": 91, "28d": 28, "7d": 7}
HS = [1, 7]
FIRST = 2186
RES = ensure_dir(OUT / "results")


def schedule(c):
    last = EVAL_END - 1
    return [FIRST] if c is None else list(range(FIRST, last + 1, c))


def run_stat(models):
    params = load_json(OUT / "tuning" / "stat_params.json")
    rows = []
    for name in models:
        for cname, c in CAD.items():
            refits = np.array(schedule(c))
            fit_s, sq = 0.0, {h: [] for h in HS}
            origins = np.arange(EVAL_START - max(HS), EVAL_END)
            which = refits[np.searchsorted(refits, origins, side="right") - 1]
            F = np.full((len(origins), 28), np.nan)
            for R in refits:
                sel = which == R
                if not sel.any():
                    continue
                t0 = time.perf_counter(); st = sm.MODELS[name][0](Y, dow_fn, int(R), params[name]); fit_s += time.perf_counter() - t0
                F[sel] = forecast_origins(name, st, origins[sel], Y)
            for h in HS:
                tg = np.arange(EVAL_START, EVAL_END + 1)
                f = F[tg - h - origins[0], h - 1]
                rows.append({"model": name, "cadence": cname, "h": h, "n_refits": len(refits),
                             "RMSE": float(np.sqrt(np.mean((Y[tg] - f) ** 2))),
                             "RMSE_naive": float(np.sqrt(np.mean((Y[tg] - Y[tg - h]) ** 2))),
                             "total_fit_seconds": fit_s})
            print(name, cname, rows[-1], flush=True)
    return rows


def run_mlp():
    import nn_models as nn
    tuned = {str(h): load_json(OUT / "tuning" / f"nn_MLP_h{h}.json") for h in (1, 7)}
    rows = []
    for cname, c in CAD.items():
        refits = np.array(schedule(c))
        for h in HS:
            origins = np.arange(EVAL_START - h, EVAL_END - h + 1)
            which = refits[np.searchsorted(refits, origins, side="right") - 1]
            f = np.full(len(origins), np.nan); fit_s = 0.0
            for R in refits:
                sel = which == R
                if not sel.any():
                    continue
                t0 = time.perf_counter()
                f[sel], _ = nn.fit_predict("MLP", tuned[str(h)]["best"], Y, dow_fn, int(R), h, origins[sel],
                                           seed=SEED + int(R) + 7 * h)
                fit_s += time.perf_counter() - t0
            tg = origins + h
            rows.append({"model": "MLP", "cadence": cname, "h": h, "n_refits": len(refits),
                         "RMSE": float(np.sqrt(np.mean((Y[tg] - f) ** 2))),
                         "RMSE_naive": float(np.sqrt(np.mean((Y[tg] - Y[tg - h]) ** 2))),
                         "total_fit_seconds": fit_s})
            print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn", action="store_true")
    a = ap.parse_args()
    path = RES / "cadence.csv"
    rows = run_mlp() if a.nn else run_stat(["NaiveDOW", "AR", "ETS", "ARIMA", "SARIMA"])
    df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old.model.isin(df.model.unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False, float_format="%.4f")
