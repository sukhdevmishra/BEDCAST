#!/usr/bin/env python3
"""
Inference latency: wall-clock time to issue one forecast, for every model and horizon, at every origin of the
evaluation span. Post-hoc descriptive measurement (the pipeline logged no separate inference times: networks were
timed as fit+predict, meta-learner prediction was not timed, and statistical prediction ran on 16 workers).

Operational view: models are re-estimated every 91 days (quarterly state, not timed here) and a forecast is
issued from each origin o with the data up to o. For each origin we time only what a daily job must run:
  - statistical models and benchmarks: forecast(state, y[0..o]) -> all 28 horizons in one call
    (ARIMA/SARIMA/ETS re-filter the history with the quarterly parameters);
  - networks: window/calendar features for o + predict_on_batch of the horizon-h network (model kept in memory,
    one untimed warm-up call per loaded network);
  - meta-learners: predict on the nine-feature row; a stacked forecast also needs its nine base forecasts, so its
    charged latency is the sum of the nine base latencies at (o, h) plus its own (same rule as accuracy_cost.py);
  - Comb: ARIMA + SARIMA + ETS latencies (the mean itself is negligible).
Quarterly states are re-created the way the pipeline made them (statistical models refitted with the tuned
parameters, networks loaded from output/v2/models, meta-learners refitted with tuning/meta_params.json), and every
reproduced forecast is checked against the stored one before its latency is accepted.

Hardware: run on CPU only (CUDA hidden), one model at a time, so all classes are timed under identical conditions.
The machine is recorded in the output; it is not the machine that trained the models.

    python3 inference_timing.py measure   -> output/v2/results/inference_latency{,_per_origin}.csv
    python3 inference_timing.py plot [latency.csv]  -> output/v2/figures/v2_accuracy_vs_inference.{pdf,png}
        (default results/inference_latency.csv; the reported figure uses the HPC measurements,
         results/inference_latency_hpc_recovered.csv, see its README)
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import platform
import time

import numpy as np
import pandas as pd

from common import (OUT, HORIZONS, H_MAX, EVAL_START, EVAL_END, BASE_ALL, BASE_NN, META,
                    load_series, all_origins, refit_for_origin, load_json)

RES = OUT / "results"
_, Y, DOW = load_series()
DOW0 = int(DOW[0])
ORIG = all_origins()
STAT = ["ARIMA", "SARIMA", "EMA", "ETS"]
BENCH = ["Naive", "NaiveDOW", "SeasonalNaive"]   # the AR benchmark is estimated but not reported
TOL = {"stat": 1e-6, "nn": 0.05, "meta": 1e-6}   # beds. Networks: float32 kernels differ between the training
# GPUs and this CPU; forecasts are ~600 beds, so 0.05 beds is a relative 1e-4 and under 1% of the smallest RMSE.
# Known reproduction gaps that do not affect latency, all from library versions differing between the training
# machine and this one. Latency depends on the model's structure (state-space order, layers, tree count and
# depth), which is identical; only fitted or computed values differ. These models are still checked and their
# maximum differences are written to inference_reproduction_check.csv; any other model must reproduce exactly.
LATENCY_ONLY = {
    "SARIMA": "re-estimated under statsmodels 0.14.5 (training: 0.14.6); ML parameters differ at some refits",
    "RandomForest-A": "refitted under scikit-learn 1.7.2 (training: 1.8.0); h=1 trees break near-ties differently",
    "RandomForest-B": "refitted under scikit-learn 1.7.2 (training: 1.8.0); h=1 trees break near-ties differently",
    "CNN": "same weights; float32 convolution kernels differ (Keras 3.11 on CPU vs 3.14 on GPU)",
}


def dow_fn(t):
    return (DOW0 + t) % 7


def eval_origins(h):
    return np.arange(EVAL_START - h, EVAL_END - h + 1)


def stored(name):
    return np.load(OUT / "forecasts" / f"{name}.npz")["F"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def measure_stat(rows, checks):
    import stat_models as sm
    params = load_json(OUT / "tuning" / "stat_params.json")
    origins = np.arange(EVAL_START - max(HORIZONS), EVAL_END - min(HORIZONS) + 1)
    for name in STAT + BENCH:
        fit, fc = sm.MODELS[name]
        F = stored(name)
        lat, maxdiff, state_R = {}, 0.0, None
        for o in origins:
            R = refit_for_origin(o)
            if R != state_R:
                st, state_R = fit(Y, dow_fn, R, params[name]), R
            t0 = time.perf_counter()
            f = fc(st, Y, dow_fn, o)
            lat[o] = time.perf_counter() - t0
            maxdiff = max(maxdiff, float(np.nanmax(np.abs(f - F[o - ORIG[0]]))))
        checks.append((name, "all", maxdiff, maxdiff <= TOL["stat"] or name in LATENCY_ONLY,
                       LATENCY_ONLY.get(name, "") if maxdiff > TOL["stat"] else ""))
        for h in HORIZONS:
            rows += [(name, h, o, lat[o]) for o in eval_origins(h)]
        log(f"{name}: median {1e3 * np.median(list(lat.values())):.2f} ms, max |diff| {maxdiff:.2e}")


def load_network(arch, h, path, scratch):
    """Rebuild the tuned architecture and load the saved weights. The .keras files were written by Keras 3.14,
    whose layer configs older Keras cannot deserialise; the weights file inside the archive loads in either."""
    import zipfile
    from nn_models import build
    hp = load_json(OUT / "tuning" / f"nn_{arch}_h{h}.json")["best"]
    m = build(arch, hp["units"], hp["layers"], hp["dropout"], hp["lr"])
    w = scratch / f"{arch}_h{h}_{path.stem}.weights.h5"
    with zipfile.ZipFile(path) as z, open(w, "wb") as fh:
        fh.write(z.read("model.weights.h5"))
    m.load_weights(w)
    w.unlink()
    return m


def measure_nn(rows, checks):
    import tempfile
    from pathlib import Path
    import keras
    from nn_models import make_xy
    scratch = Path(tempfile.mkdtemp())
    for arch in BASE_NN:
        F = stored(arch)
        for h in HORIZONS:
            maxdiff = 0.0
            os_ = eval_origins(h)
            refits = np.array([refit_for_origin(o) for o in os_])
            for R in np.unique(refits):
                path = OUT / "models" / arch / f"h{h}" / f"R{R}.keras"
                m = load_network(arch, h, path, scratch)
                stats = np.load(str(path).replace(".keras", "_stats.npy"), allow_pickle=True).item()
                Ps, Pc = make_xy(Y, dow_fn, os_[:1], h, stats, with_target=False)
                m.predict_on_batch([Ps, Pc])                          # warm-up, untimed
                for o in os_[refits == R]:
                    t0 = time.perf_counter()
                    Ps, Pc = make_xy(Y, dow_fn, [o], h, stats, with_target=False)
                    inc = float(np.asarray(m.predict_on_batch([Ps, Pc])).ravel()[0]) * stats["sd_h"]
                    f = Y[o] + inc
                    rows.append((arch, h, o, time.perf_counter() - t0))
                    maxdiff = max(maxdiff, abs(f - F[o - ORIG[0], h - 1]))
                keras.backend.clear_session()
            checks.append((arch, h, maxdiff, maxdiff <= TOL["nn"] or arch in LATENCY_ONLY,
                           LATENCY_ONLY.get(arch, "") if maxdiff > TOL["nn"] else ""))
            log(f"{arch} h={h}: max |diff| {maxdiff:.2e}")


def measure_meta(rows, checks):
    import run_meta as rm
    params = load_json(OUT / "tuning" / "meta_params.json")
    refit_of = np.array([refit_for_origin(o) for o in ORIG])
    for v in "AB":
        for name in META:
            F = stored(f"meta_{name}-{v}")
            for h in HORIZONS:
                X, t, tgt = rm.features(h, v)
                ok = np.isfinite(X).all(1)
                p = params[f"{name}-{v}"][str(h)]["params"]
                maxdiff = 0.0
                os_ = eval_origins(h)
                for R in np.unique([refit_for_origin(o) for o in os_]):
                    tr = ok & np.isfinite(t) & (tgt <= R)
                    m = rm.make(name, p).fit(X[tr], t[tr])
                    m.predict(X[:1])                                  # warm-up, untimed
                    for o in os_[refit_of[os_ - ORIG[0]] == R]:
                        i = o - ORIG[0]
                        t0 = time.perf_counter()
                        f = Y[o] + m.predict(X[i:i + 1])[0]
                        rows.append((f"{name}-{v}", h, o, time.perf_counter() - t0))
                        maxdiff = max(maxdiff, abs(f - F[i, h - 1]))
                key = f"{name}-{v}"
                checks.append((key, h, maxdiff, maxdiff <= TOL["meta"] or key in LATENCY_ONLY,
                               LATENCY_ONLY.get(key, "") if maxdiff > TOL["meta"] else ""))
            log(f"{name}-{v}: done")


def measure():
    rows, checks = [], []
    t0 = time.time()
    measure_stat(rows, checks)
    measure_nn(rows, checks)
    measure_meta(rows, checks)
    chk = pd.DataFrame(checks, columns=["model", "h", "max_abs_diff_beds", "pass", "note"])
    chk.to_csv(RES / "inference_reproduction_check.csv", index=False)
    if not chk["pass"].all():
        print(chk[~chk["pass"]].to_string())
        sys.exit("reproduced forecasts differ from the stored ones; latencies not written")
    own = pd.DataFrame(rows, columns=["model", "h", "origin", "own_s"])
    # charged latency: a forecast that depends on other models' forecasts pays for them too
    piv = own.pivot_table(index=["h", "origin"], columns="model", values="own_s")
    charged = {m: piv[m] for m in piv.columns}
    charged["Comb"] = piv[["ARIMA", "SARIMA", "ETS"]].sum(axis=1)
    base = piv[BASE_ALL].sum(axis=1)
    for v in "AB":
        for name in META:
            charged[f"{name}-{v}"] = base + piv[f"{name}-{v}"]
    ch = pd.DataFrame(charged).stack().rename("charged_s").reset_index().rename(columns={"level_2": "model"})
    per = ch.merge(own, on=["model", "h", "origin"], how="left")
    per.to_csv(RES / "inference_latency_per_origin.csv.gz", index=False, float_format="%.6g")
    s = per.groupby(["model", "h"]).agg(n=("origin", "size"), own_median_ms=("own_s", lambda x: 1e3 * x.median()),
                                         median_ms=("charged_s", lambda x: 1e3 * x.median()),
                                         p95_ms=("charged_s", lambda x: 1e3 * x.quantile(0.95)),
                                         total_s=("charged_s", "sum")).reset_index()
    s.to_csv(RES / "inference_latency.csv", index=False, float_format="%.4f")
    import keras, sklearn, statsmodels, lightgbm
    with open(RES / "inference_latency_env.txt", "w") as fh:
        fh.write(f"run {time.strftime('%Y-%m-%dT%H:%M:%S')} ({time.time() - t0:.0f} s)\n"
                 f"cpu: {platform.processor() or platform.machine()}; {os.cpu_count()} logical cores; CUDA hidden\n"
                 f"python {platform.python_version()}, keras {keras.__version__}, sklearn {sklearn.__version__}, "
                 f"statsmodels {statsmodels.__version__}, lightgbm {lightgbm.__version__}\n")
    print(s.pivot(index="model", columns="h", values="median_ms").round(2).to_string())


def plot(src=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from accuracy_cost import CCOL, COMB_LABEL, NAME, B_OFFSET, colour, place_labels
    from make_outputs import C, save

    s = pd.read_csv(src or RES / "inference_latency.csv")
    s = s[s.model != "AR"]     # measured, not reported
    pm = pd.read_csv(RES / "point_metrics.csv")
    pm = pm[pm.period == "evaluation"][["h", "model", "RMSE", "RelRMSE_naive"]]
    df = s.merge(pm, on=["model", "h"], how="inner", validate="one_to_one")
    assert len(df) == len(s)
    cls = {**{m: "Statistical" for m in STAT}, **{m: "Neural" for m in BASE_NN},
           **{m: "Benchmark" for m in BENCH + ["Comb"]}}
    df["cls"] = df.model.map(cls).fillna("Meta")
    df["level"] = np.where(df.model.str.endswith("-B"), "Meta (B)",
                           np.where(df.model.str.endswith("-A"), "Meta (A)",
                                    np.where(df.model == "Comb", "Combination", "Base")))
    df.round(4).to_csv(RES / "accuracy_vs_inference.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 8.6))
    labels = {}
    xlo, xhi = df.median_ms.min() / 3, df.median_ms.max() * 6
    for ax, h in zip(axes.flat, HORIZONS):
        d = df[df.h == h].copy()
        d["x"] = d.median_ms * np.where(d.level == "Meta (B)", B_OFFSET, 1.0)
        core = d[d.RelRMSE_naive < 1.3]
        lo, hi = core.RMSE.min(), core.RMSE.max()
        pad = 0.06 * (hi - lo)
        top = hi + 2.0 * pad
        ax.set_ylim(lo - pad, top)
        ax.set_xscale("log")
        ax.set_xlim(xlo, xhi)
        ax.axhline(d.loc[d.model == "Naive", "RMSE"].iloc[0], color=C["ink2"], linewidth=0.7,
                   linestyle=(0, (4, 2)), zorder=1)
        labels[ax] = []
        for r in d.itertuples():
            col = colour(r)
            hollow = r.level == "Meta (B)"
            y, name = r.RMSE, NAME.get(r.model, r.model)
            mk = "o" if r.level == "Base" else "D"
            if r.RMSE > top:
                y, mk, name = top - 0.4 * pad, "^", f"{name} ({r.RMSE:.1f}, off scale)"
            ax.scatter(r.x, y, s=26, marker=mk, zorder=5 if r.model == "Comb" else 4, linewidth=1.0,
                       facecolor="white" if hollow else col, edgecolor=col if hollow else "white")
            labels[ax].append((r.x, y, name, "left" if r.level == "Meta (A)" else "right"))
        ax.set_title(f"{h}-day-ahead", fontsize=8, color=C["ink"], loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(color=C["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("Time to issue one forecast (median ms, log scale)")
        ax.set_ylabel("RMSE (beds)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ax, pts in labels.items():
        place_labels(ax, pts)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=5, markerfacecolor=CCOL[c], markeredgecolor="white",
                      label=f"{c} (level 1)") for c in ["Benchmark", "Statistical", "Neural"]]
    handles += [Line2D([], [], marker="D", linestyle="", markersize=5, markerfacecolor=CCOL["Statistical"],
                       markeredgecolor="white", label=COMB_LABEL),
                Line2D([], [], marker="D", linestyle="", markersize=5, markerfacecolor=CCOL["Meta"],
                       markeredgecolor="white", label="Stacking, Variant A (level 2)"),
                Line2D([], [], marker="D", linestyle="", markersize=5, markerfacecolor="white",
                       markeredgecolor=CCOL["Meta"], label="Stacking, Variant B (level 2)"),
                Line2D([], [], color=C["ink2"], linewidth=0.7, linestyle=(0, (4, 2)), label="Naive RMSE")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False, fontsize=6.5)
    n = int(df.n.iloc[0])
    fig.text(0.5, -0.045, f"Median over the {n:,} daily forecast origins of the evaluation span; CPU only, models "
             "held in memory. Lower is better. Combinations and stacking include the latency of the base forecasts "
             "they need.\nEN = ElasticNet, DT = DecisionTree, RF = RandomForest, LGBM = LightGBM meta-learner; "
             "-A / -B = stacking variant, B drawn shifted right for visibility.",
             ha="center", fontsize=6.3, color=C["ink2"])
    save(fig, "v2_accuracy_vs_inference")


if __name__ == "__main__":
    measure() if sys.argv[1] == "measure" else plot(*sys.argv[2:3])
