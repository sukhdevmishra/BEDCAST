#!/usr/bin/env python3
"""
Statistical learners and benchmarks: tuning on development data, then rolling-origin
out-of-sample forecasts from every origin 1368..3376.

    python3 run_stat.py tune            -> output/v2/tuning/stat_params.json
    python3 run_stat.py roll [--start YYYY-MM-DD]   -> output/v2/forecasts/<Model>.npz
"""
import argparse
import sys
import warnings

import numpy as np
from joblib import Parallel, delayed
from skopt import gp_minimize
from skopt.space import Integer

from common import (OUT, HORIZONS, H_MAX, REFITS, TUNE_FOLDS, N_TUNE, SEED, LAST_INDEX,
                    load_series, all_origins, refit_for_origin, save_json, load_json, ensure_dir, Timer)
import stat_models as sm

warnings.filterwarnings("ignore")
N_JOBS = 16

_, Y, DOW = load_series()
DOW0 = int(DOW[0])


def dow_fn(t):
    return (DOW0 + t) % 7


def _forecast_many(name, st, origins, y):
    fc = sm.MODELS[name][1]
    return np.vstack([fc(st, y, dow_fn, o) for o in origins])


def forecast_origins(name, st, origins, y, n_jobs=N_JOBS):
    chunks = [c for c in np.array_split(np.asarray(origins), max(1, min(n_jobs, len(origins)))) if len(c)]
    parts = Parallel(n_jobs=n_jobs)(delayed(_forecast_many)(name, st, c, y) for c in chunks)
    return np.vstack(parts)


# ── tuning ────────────────────────────────────────────────────────────────────
def validation_score(name, params, y=Y):
    """Mean over folds and horizons of MAE / naive MAE on validation targets."""
    ratios = []
    for train_end, v0, v1 in TUNE_FOLDS:
        try:
            st = sm.MODELS[name][0](y, dow_fn, train_end, params)
        except Exception:
            return 10.0
        origins = np.arange(train_end, v1)
        F = forecast_origins(name, st, origins, y, n_jobs=8)
        for h in HORIZONS:
            tgt = origins + h
            keep = (tgt >= max(v0, train_end + h)) & (tgt <= v1)
            if not keep.any():
                continue
            err = np.abs(y[tgt[keep]] - F[keep, h - 1])
            nerr = np.abs(y[tgt[keep]] - y[origins[keep]])
            ratios.append(err.mean() / nerr.mean())
    return float(np.mean(ratios)) if ratios and np.all(np.isfinite(ratios)) else 10.0


SPACES = {
    "ARIMA": [Integer(0, 5, name="p"), Integer(0, 1, name="d"), Integer(0, 3, name="q")],
    "SARIMA": [Integer(0, 3, name="p"), Integer(0, 1, name="d"), Integer(0, 3, name="q"),
               Integer(0, 2, name="P"), Integer(0, 1, name="D"), Integer(0, 2, name="Q")],
    "EMA": [Integer(2, 60, name="span")],
}


def tune():
    out, trace = {}, {}
    for name, space in SPACES.items():
        names = [d.name for d in space]
        hist = []

        def obj(x):
            pr = {k: int(v) for k, v in zip(names, x)}
            with Timer.log(name, 0, -1, "tune_eval"):
                s = validation_score(name, pr)
            hist.append({**pr, "score": s})
            print(f"  {name} {pr} -> {s:.4f}", flush=True)
            return s

        res = gp_minimize(obj, space, n_calls=N_TUNE, n_initial_points=8, random_state=SEED)
        out[name] = {k: int(v) for k, v in zip(names, res.x)}
        trace[name] = hist
    # ETS: all six configurations (fewer than the 15-evaluation budget)
    hist = []
    for c in range(len(sm.ETS_CONFIGS)):
        s = validation_score("ETS", {"config": c})
        hist.append({"config": c, **{k: str(v) for k, v in sm.ETS_CONFIGS[c].items()}, "score": s})
        print(f"  ETS config {c} {sm.ETS_CONFIGS[c]} -> {s:.4f}", flush=True)
    out["ETS"] = {"config": int(min(hist, key=lambda r: r["score"])["config"])}
    trace["ETS"] = hist
    for name in ["Naive", "NaiveDOW", "SeasonalNaive", "AR"]:
        out[name] = {}
    save_json(out, OUT / "tuning" / "stat_params.json")
    save_json(trace, OUT / "tuning" / "stat_tuning_trace.json")
    print(out)


# ── rolling-origin forecasts ──────────────────────────────────────────────────
def roll(start_index=0, tag=""):
    params = load_json(OUT / "tuning" / "stat_params.json")
    origins = all_origins()
    y = Y.copy()
    ensure_dir(OUT / "forecasts")
    for name in ["Naive", "NaiveDOW", "SeasonalNaive", "AR", "EMA", "ETS", "ARIMA", "SARIMA"]:
        F = np.full((len(origins), H_MAX), np.nan)
        refit_of = np.array([refit_for_origin(o) for o in origins])
        states = {}
        for R in REFITS:
            sel = refit_of == R
            if not sel.any():
                continue
            # S1 sensitivity: drop observations before start_index from estimation only
            with Timer.log(name + tag, 0, R, "fit"):
                st = fit_from(name, params[name], y, R, start_index)
            states[R] = st
            with Timer.log(name + tag, 0, R, "predict"):
                F[sel] = forecast_origins(name, st, origins[sel], y)
            print(f"{name}{tag} refit {R}: {sel.sum()} origins", flush=True)
        np.savez_compressed(OUT / "forecasts" / f"{name}{tag}.npz", origins=origins, F=F)
    # Comb: equal-weight mean of ARIMA, SARIMA and ETS
    Fs = [np.load(OUT / "forecasts" / f"{m}{tag}.npz")["F"] for m in ["ARIMA", "SARIMA", "ETS"]]
    np.savez_compressed(OUT / "forecasts" / f"Comb{tag}.npz", origins=origins, F=np.mean(Fs, axis=0))


def fit_from(name, pr, y, R, start_index):
    if start_index == 0:
        return sm.MODELS[name][0](y, dow_fn, R, pr)
    # Estimate on y[start_index..R]; weekday indexing is shifted accordingly.
    sub = y[start_index:]
    shift = start_index

    def dfn(t, s=shift):
        return dow_fn(t + s)
    st = sm.MODELS[name][0](sub, dfn, R - shift, pr)
    if name == "ARIMA" or name == "SARIMA" or name == "ETS":
        return st          # parameters only; forecasting re-filters the full history (sensitivity analysis)
    if name in ("EMA",):
        return st          # weekday offsets are position-invariant because dfn realigns weekdays
    return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "roll"])
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tune() if a.mode == "tune" else roll(a.start_index, a.tag)
