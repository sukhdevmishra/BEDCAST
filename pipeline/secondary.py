#!/usr/bin/env python3
"""Secondary outputs: S1 sensitivity (training from 2012-03-01) and the computational-cost summary.
Writes output/v2/results/s1_sensitivity.csv, timing_summary.csv and the matching tex fragments."""
import numpy as np
import pandas as pd

from common import OUT, HORIZONS, EVAL_START, EVAL_END, BASE_ALL, load_series, all_origins
from make_outputs import tex_table, lab

_, Y, _ = load_series()
O = all_origins()
RES = OUT / "results"
MODELS = ["ARIMA", "SARIMA", "ETS", "EMA", "NaiveDOW", "Comb"]   # the AR benchmark is estimated but not reported


def s1():
    rows = []
    tg = np.arange(EVAL_START, EVAL_END + 1)
    for h in HORIZONS:
        r = tg - h - O[0]
        naive = np.sqrt(np.mean((Y[tg] - Y[tg - h]) ** 2))
        for m in MODELS:
            a = np.load(OUT / "forecasts" / f"{m}.npz")["F"][r, h - 1]
            b = np.load(OUT / "forecasts" / f"{m}_S1.npz")["F"][r, h - 1]
            rows.append({"h": h, "model": m, "RelRMSE_primary": np.sqrt(np.mean((Y[tg] - a) ** 2)) / naive,
                         "RelRMSE_S1": np.sqrt(np.mean((Y[tg] - b) ** 2)) / naive})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "s1_sensitivity.csv", index=False, float_format="%.4f")
    trs = []
    for m in MODELS:
        s = df[df.model == m].set_index("h")
        trs.append(f"{lab(m)} & " + " & ".join(f"{s.loc[h].RelRMSE_primary:.3f} / {s.loc[h].RelRMSE_S1:.3f}" for h in HORIZONS) + " \\\\")
    tex_table("v2_s1_sensitivity", "Sensitivity S1: estimation from 1 March 2012 (excluding two anomalous low-occupancy episodes). Entries: RMSE relative to naive, primary / S1.",
              "tab:v2_s1", "l cccc", "Model & $h=1$ & $h=7$ & $h=14$ & $h=28$", trs, wide=True)
    return df


def timing():
    from common import REFITS
    t = pd.read_csv(OUT / "timing.csv")
    # rolling re-estimations only: excludes tuning and the tuning processes restarted on 2026-09-25
    t = t[t.refit.isin(REFITS) & t.stage.isin(["fit", "predict", "fit+predict"]) & ~t.model.str.contains("_S1")]
    base = t[t.model.isin(BASE_ALL + ["Naive", "NaiveDOW", "SeasonalNaive"])
             | t.model.str.endswith("-A") | t.model.str.endswith("-B")]
    g = base.groupby("model").agg(total_s=("seconds", "sum"), n=("seconds", "size"))
    per_refit = base.groupby(["model", "refit"]).seconds.sum().groupby("model").mean()
    g["mean_s_per_refit_all_horizons"] = per_refit
    g = g.sort_values("total_s", ascending=False)
    g.to_csv(RES / "timing_summary.csv", float_format="%.2f")
    rows = [f"{lab(m)} & {r.mean_s_per_refit_all_horizons:.1f} & {r.total_s / 3600:.2f} \\\\" for m, r in g.iterrows()
            if not m.endswith("-B") and "_S1" not in m]
    tex_table("v2_timing", "Computational cost of the rolling evaluation (wall-clock; two RTX 4090 GPUs for networks, CPU otherwise).",
              "tab:v2_timing", "l rr", "Model & s / re-est. & Total (h)", rows,
              note="Times include fitting and forecasting from every origin in the 91-day block. Networks ran as 20 concurrent "
                   "processes sharing two GPUs, and statistical models used 16 CPU workers, so absolute times are indicative; "
                   "the ratios between model classes are the robust quantity. The stacking ensemble's cost is the sum of its nine "
                   "base learners plus the meta-learner.")
    return g


if __name__ == "__main__":
    print(s1().pivot(index="model", columns="h", values=["RelRMSE_primary", "RelRMSE_S1"]).round(3).to_string())
    print(timing().round(1).to_string())
