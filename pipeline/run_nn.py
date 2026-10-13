#!/usr/bin/env python3
"""
Neural base learners: per-architecture, per-horizon tuning on development data,
then rolling-origin forecasts. Each architecture is run as a separate process.

    CUDA_VISIBLE_DEVICES=0 python3 run_nn.py tune MLP LSTM
    CUDA_VISIBLE_DEVICES=0 python3 run_nn.py roll MLP LSTM
"""
import argparse
import sys

import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

from common import (OUT, HORIZONS, H_MAX, REFITS, TUNE_FOLDS, N_TUNE, SEED, BASE_NN,
                    load_series, all_origins, refit_for_origin, save_json, load_json, ensure_dir, Timer)
import nn_models as nn

_, Y, DOW = load_series()
DOW0 = int(DOW[0])


def dow_fn(t):
    return (DOW0 + t) % 7


def seed_for(arch, h, R):
    return SEED + 100000 * BASE_NN.index(arch) + 1000 * h + (R % 1000)


SPACE = [Integer(16, 128, name="units"), Integer(1, 3, name="layers"),
         Real(1e-4, 1e-2, prior="log-uniform", name="lr"), Real(0.0, 0.3, name="dropout"),
         Categorical([16, 32, 64], name="batch")]


def tune(archs, horizons=HORIZONS):
    for arch in archs:
        for h in horizons:
            hist = []

            def obj(x):
                hp = {"units": int(x[0]), "layers": int(x[1]), "lr": float(x[2]),
                      "dropout": float(x[3]), "batch": int(x[4])}
                maes = []
                for train_end, v0, v1 in TUNE_FOLDS:
                    origins = np.arange(train_end, v1 - h + 1)
                    origins = origins[origins + h >= max(v0, train_end + h)]
                    with Timer.log(arch, h, train_end, "tune_fit"):
                        pred, ep = nn.fit_predict(arch, hp, Y, dow_fn, train_end, h, origins,
                                                  seed=seed_for(arch, h, train_end))
                    maes.append(np.mean(np.abs(Y[origins + h] - pred)))
                s = float(np.mean(maes))
                hist.append({**hp, "val_mae": s})
                print(f"  {arch} h={h} {hp} -> {s:.4f}", flush=True)
                return s

            r = gp_minimize(obj, SPACE, n_calls=N_TUNE, n_initial_points=8, random_state=SEED)
            best = {"units": int(r.x[0]), "layers": int(r.x[1]), "lr": float(r.x[2]),
                    "dropout": float(r.x[3]), "batch": int(r.x[4])}
            save_json({"best": best, "trace": hist}, OUT / "tuning" / f"nn_{arch}_h{h}.json")


def roll(archs, horizons=HORIZONS):
    origins = all_origins()
    refit_of = np.array([refit_for_origin(o) for o in origins])
    ensure_dir(OUT / "forecasts")
    for arch in archs:
        for h in horizons:
            F = np.full((len(origins), H_MAX), np.nan)
            epochs = []
            hp = load_json(OUT / "tuning" / f"nn_{arch}_h{h}.json")["best"]
            mdir = ensure_dir(OUT / "models" / arch / f"h{h}")
            for R in REFITS:
                sel = refit_of == R
                if not sel.any():
                    continue
                with Timer.log(arch, h, R, "fit+predict"):
                    pred, ep = nn.fit_predict(arch, hp, Y, dow_fn, R, h, origins[sel],
                                              seed=seed_for(arch, h, R),
                                              save_path=mdir / f"R{R}.keras")
                F[sel, h - 1] = pred
                epochs.append({"h": h, "R": R, "epochs": ep})
                print(f"{arch} h={h} refit {R}: {sel.sum()} origins, {ep} epochs", flush=True)
            np.savez_compressed(OUT / "forecasts" / f"{arch}_h{h}.npz", origins=origins, F=F)
            save_json(epochs, OUT / "logs" / f"nn_epochs_{arch}_h{h}.json")


def merge(archs):
    """Combine per-horizon forecast files into output/v2/forecasts/<arch>.npz."""
    origins = all_origins()
    for arch in archs:
        F = np.full((len(origins), H_MAX), np.nan)
        for h in HORIZONS:
            F[:, h - 1] = np.load(OUT / "forecasts" / f"{arch}_h{h}.npz")["F"][:, h - 1]
        np.savez_compressed(OUT / "forecasts" / f"{arch}.npz", origins=origins, F=F)
        print("merged", arch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "roll", "both", "merge"])
    ap.add_argument("archs", nargs="+", choices=BASE_NN)
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    a = ap.parse_args()
    if a.mode == "merge":
        merge(a.archs); sys.exit()
    if a.mode in ("tune", "both"):
        tune(a.archs, a.horizons)
    if a.mode in ("roll", "both"):
        roll(a.archs, a.horizons)
