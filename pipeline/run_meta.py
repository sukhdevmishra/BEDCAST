#!/usr/bin/env python3
"""
Stacking. Meta-features are the nine base forecasts FOR THE TARGET DATE, expressed as
increments over the origin value. The meta-target is y[o+h] - y[o].
Variant A: base forecasts only (primary). Variant B: adds the target-date weekday one-hot and the last
seven daily changes. Hyperparameters are tuned once per horizon on warm-up targets 1368..2282; the
meta-learners are then refitted at every R_k >= 1731 on all out-of-sample base forecasts with target <= R_k.

    python3 run_meta.py      -> output/v2/forecasts/meta_<Learner>-<Variant>.npz
"""
import warnings

import numpy as np
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from skopt import gp_minimize
from skopt.space import Integer, Real

from common import (OUT, HORIZONS, H_MAX, REFITS, META_FIRST_REFIT, EVAL_START, N_TUNE, SEED, LAST_INDEX,
                    BASE_ALL, META, load_series, all_origins, refit_for_origin, dow_onehot,
                    save_json, ensure_dir, Timer)

warnings.filterwarnings("ignore")
_, Y, DOW = load_series()
DOW0 = int(DOW[0])
ORIG = all_origins()
META_TUNE_FOLDS = [(1917, 2099), (2100, EVAL_START - 1)]   # validation target windows (last 2 x 183 d of warm-up)


def base_matrix(h):
    return np.column_stack([np.load(OUT / "forecasts" / f"{m}.npz")["F"][:, h - 1] for m in BASE_ALL])


def features(h, variant):
    B = base_matrix(h)
    o = ORIG
    X = B - Y[o][:, None]
    if variant == "B":
        cal = dow_onehot((DOW0 + o + h) % 7)
        lags = np.column_stack([Y[o] - Y[o - j] for j in range(1, 8)])
        X = np.column_stack([X, cal, lags])
    tgt_idx = o + h
    valid_t = tgt_idx <= LAST_INDEX
    t = np.full(len(o), np.nan)
    t[valid_t] = Y[tgt_idx[valid_t]] - Y[o[valid_t]]
    return X, t, tgt_idx


SPACES = {
    "ElasticNet": [Real(1e-4, 1e2, prior="log-uniform", name="alpha"), Real(0.0, 1.0, name="l1_ratio")],
    "DecisionTree": [Integer(1, 8, name="max_depth"), Integer(2, 10, name="min_samples_split")],
    "RandomForest": [Integer(10, 100, prior="log-uniform", name="n_estimators"), Integer(1, 8, name="max_depth")],
    "LightGBM": [Integer(10, 100, prior="log-uniform", name="n_estimators"),
                 Real(1e-3, 1e-1, prior="log-uniform", name="learning_rate"), Integer(1, 8, name="max_depth")],
}


def make(name, p):
    if name == "ElasticNet":
        return make_pipeline(StandardScaler(), ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], max_iter=20000))
    if name == "DecisionTree":
        return DecisionTreeRegressor(max_depth=p["max_depth"], min_samples_split=p["min_samples_split"], random_state=SEED)
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=p["n_estimators"], max_depth=p["max_depth"], random_state=SEED, n_jobs=4)
    return LGBMRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                         random_state=SEED, n_jobs=4, verbose=-1)


def tune_one(name, X, t, tgt):
    space = SPACES[name]
    names = [d.name for d in space]

    def obj(x):
        p = {k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in zip(names, x)}
        maes = []
        for v0, v1 in META_TUNE_FOLDS:
            tr = tgt < v0
            va = (tgt >= v0) & (tgt <= v1)
            m = make(name, p).fit(X[tr], t[tr])
            maes.append(np.mean(np.abs(t[va] - m.predict(X[va]))))
        return float(np.mean(maes))

    r = gp_minimize(obj, space, n_calls=N_TUNE, n_initial_points=8, random_state=SEED)
    return {k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in zip(names, r.x)}, float(r.fun)


def run_one(name, variant, h):
    X, t, tgt = features(h, variant)
    ok = np.isfinite(X).all(1)
    tune_rows = ok & (tgt <= EVAL_START - 1) & np.isfinite(t)
    p, score = tune_one(name, X[tune_rows], t[tune_rows], tgt[tune_rows])
    pred = np.full(len(ORIG), np.nan)
    refit_of = np.array([refit_for_origin(o) for o in ORIG])
    coefs = []
    for R in REFITS:
        if R < META_FIRST_REFIT:
            continue
        sel = refit_of == R
        tr = ok & np.isfinite(t) & (tgt <= R)
        with Timer.log(f"{name}-{variant}", h, R, "fit"):
            m = make(name, p).fit(X[tr], t[tr])
        pred[sel] = Y[ORIG[sel]] + m.predict(X[sel])
        if name == "ElasticNet":
            coefs.append({"R": int(R), "coef": m[-1].coef_.tolist(), "intercept": float(m[-1].intercept_),
                          "x_scale": m[0].scale_.tolist()})
    return name, variant, h, p, score, pred, coefs


def main():
    jobs = [(n, v, h) for n in META for v in ("A", "B") for h in HORIZONS]
    res = Parallel(n_jobs=8)(delayed(run_one)(*j) for j in jobs)
    params, coefs = {}, {}
    out = {}
    for name, variant, h, p, score, pred, cf in res:
        params.setdefault(f"{name}-{variant}", {})[str(h)] = {"params": p, "val_mae_increment": score}
        out.setdefault(f"{name}-{variant}", np.full((len(ORIG), H_MAX), np.nan))[:, h - 1] = pred
        if cf:
            coefs.setdefault(f"{name}-{variant}", {})[str(h)] = cf
    for k, F in out.items():
        np.savez_compressed(OUT / "forecasts" / f"meta_{k}.npz", origins=ORIG, F=F)
    save_json(params, OUT / "tuning" / "meta_params.json")
    save_json(coefs, OUT / "models" / "elasticnet_coefficients.json")
    print(params)


if __name__ == "__main__":
    main()
