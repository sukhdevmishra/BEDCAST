#!/usr/bin/env python3
"""
Gantt chart of the full v2 training process, from t = 0 to the last meta-learner fit.
Descriptive only (no analysis choice): every bar length is a logged wall-clock duration from
output/v2/timing.csv. Two stages were not timed by the pipeline and are marked as such rather than estimated:
ETS tuning (run_stat.tune) and meta-learner tuning (run_meta.tune_one). For the meta stage as a whole, an upper
bound on wall-clock is taken from file modification times (last merged base forecast -> meta outputs).

Schedule rules (they follow the code, not the calendar):
  - CPU track (run_stat.py): tuning ARIMA -> SARIMA -> EMA -> ETS, then rolling fits in the order
    Naive, NaiveDOW, SeasonalNaive, EMA, ETS, ARIMA, SARIMA, back to back. (The AR benchmark also ran, after
    SeasonalNaive, for 0.45 s in all; it is not reported and is left out.)
  - GPU track (launch_nn.sh): 20 concurrent processes (5 networks x 4 horizons), each tuning then rolling.
    The relaunched processes started within 13 s of each other (checked against tuning-file mtimes); drawn at t = 0.
  - Barrier: run_meta.py reads all nine base learners' forecasts, so every meta-learner starts after the last
    base learner finishes.
  The CPU and GPU tracks are independent and are drawn concurrently from t = 0; on 2026-09-25 the CPU track
  actually ran first (about 11:32-11:39 UTC) and the networks from 11:43 UTC. The CPU track is not on the critical path.

    python3 gantt.py   -> output/v2/figures/v2_training_gantt.{pdf,png}, output/v2/results/training_schedule.csv
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from common import OUT, HORIZONS, REFITS, BASE_NN, META
from make_outputs import C, RES, save

TUNE_CALLS = 30          # 15 GP evaluations x 2 development folds per network and horizon
STAT_TUNE = ["ARIMA", "SARIMA", "EMA", "ETS"]
STAT_ROLL = ["Naive", "NaiveDOW", "SeasonalNaive", "EMA", "ETS", "ARIMA", "SARIMA"]
BENCH = ["Naive", "NaiveDOW", "SeasonalNaive"]
COL = {"tune": "#86b6ef", "roll": "#1c5cab", "meta": "#eb6834", "bench": C["grey"]}


def build_schedule():
    t = pd.read_csv(OUT / "timing.csv")
    rows = []

    def add(lane, group, stage, start, dur, note=""):
        rows.append({"lane": lane, "group": group, "stage": stage, "start_s": start, "end_s": start + dur,
                     "seconds": dur, "note": note})
        return start + dur

    # CPU track
    clock = 0.0
    for m in STAT_TUNE:
        s = t[(t.model == m) & (t.stage == "tune_eval")].seconds
        if len(s):
            clock = add(m, "stat", "tune", clock, s.sum())
        else:
            add(m, "stat", "tune_untimed", clock, 0.0, "tuning not logged by run_stat.py")
    for m in STAT_ROLL:
        s = t[(t.model == m) & t.stage.isin(["fit", "predict"]) & t.refit.isin(REFITS)].seconds.sum()
        clock = add("Benchmarks" if m in BENCH else m, "bench" if m in BENCH else "stat", "roll", clock, s,
                    m if m in BENCH else "")

    # GPU track: keep only the last 30 tuning calls per (network, horizon); earlier h=1 rows are from the
    # first launch, which was stopped and relaunched per horizon.
    for a in BASE_NN:
        for h in HORIZONS:
            tu = t[(t.model == a) & (t.horizon == h) & (t.stage == "tune_fit")].tail(TUNE_CALLS)
            assert len(tu) == TUNE_CALLS
            ro = t[(t.model == a) & (t.horizon == h) & (t.stage == "fit+predict")]
            assert len(ro) == len(REFITS)
            end = add(f"{a} h={h}", "nn", "tune", 0.0, tu.seconds.sum())
            add(f"{a} h={h}", "nn", "roll", end, ro.seconds.sum())

    df = pd.DataFrame(rows)
    barrier = df[df.group.isin(["stat", "nn"])].end_s.max()

    for v in "AB":
        for m in META:
            s = t[(t.model == f"{m}-{v}") & (t.stage == "fit")].seconds
            add(f"{m} ({v})", f"meta{v}", "tune_untimed", barrier, 0.0, "tuning not logged by run_meta.py")
            add(f"{m} ({v})", f"meta{v}", "meta", barrier, s.sum(), f"{len(s)} fits (19 refits x 4 horizons)")

    # Upper bound on the whole meta stage (tuning + fitting, 8 joblib workers), from file mtimes.
    last_base = max(os.path.getmtime(OUT / "forecasts" / f"{a}.npz") for a in BASE_NN)
    meta_out = max(os.path.getmtime(OUT / "forecasts" / f"meta_{m}-{v}.npz") for m in META for v in "AB")
    add("Meta stage, wall-clock", "metaW", "meta_wall", barrier, meta_out - last_base,
        "upper bound: last merged base forecast to last meta output file (includes tuning and process start)")
    return pd.DataFrame(rows), barrier


def lanes():
    order = ["ARIMA", "SARIMA", "EMA", "ETS", "Benchmarks"]
    order += [f"{a} h={h}" for a in BASE_NN for h in HORIZONS]
    order += [f"{m} ({v})" for v in "AB" for m in META] + ["Meta stage, wall-clock"]
    return {lane: i for i, lane in enumerate(order)}


def fmt(s):
    return f"{s:,.0f} s" if s >= 10 else f"{s:.1f} s"


def draw(df, barrier):
    y = lanes()
    meta_end = df[df.group == "metaW"].end_s.iloc[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 7.4), sharey=True,
                                 gridspec_kw={"width_ratios": [4.3, 1.0], "wspace": 0.06})
    for ax in (a1, a2):
        for r in df.itertuples():
            i = y[r.lane]
            if r.stage == "tune_untimed":
                continue
            col = COL["bench"] if r.group == "bench" else COL["meta"] if r.stage in ("meta", "meta_wall") else COL[r.stage]
            hatch = "////" if r.stage == "meta_wall" else None
            ax.barh(i, r.seconds, left=r.start_s, height=0.62, color="white" if hatch else col,
                    edgecolor=col if hatch else "white", linewidth=0.8 if hatch else 0.6, hatch=hatch)
        ax.axvline(barrier, color=C["ink"], linewidth=0.9, linestyle=(0, (4, 2)), zorder=3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", color=C["grid"], linewidth=0.5)
        ax.set_axisbelow(True)

    # left panel: whole process
    a1.set_xlim(0, barrier * 1.13)
    a1.set_xticks(range(0, int(barrier) + 1, 1000))
    for a in BASE_NN:
        for h in HORIZONS:
            seg = df[df.lane == f"{a} h={h}"].set_index("stage")
            i = y[f"{a} h={h}"]
            for st in ("tune", "roll"):
                r = seg.loc[st]
                a1.text(r.start_s + r.seconds / 2, i, f"{r.seconds:,.0f}", ha="center", va="center", fontsize=5.8,
                        color=C["ink"] if st == "tune" else "white")
            a1.text(seg.loc["roll"].end_s + 40, i, f"= {seg.seconds.sum():,.0f} s", va="center", fontsize=5.8,
                    color=C["ink2"], zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    for lane in ["ARIMA", "SARIMA", "EMA", "ETS", "Benchmarks"]:
        seg = df[df.lane == lane]
        parts = [f"Naive, Naive + weekday, seasonal naive: roll {fmt(seg.seconds.sum())} in total"] if lane == "Benchmarks" else []
        for r in ([] if lane == "Benchmarks" else seg.itertuples()):
            parts.append("tuning not logged" if r.stage == "tune_untimed" else f"{'tune' if r.stage == 'tune' else 'roll'} {fmt(r.seconds)}")
        a1.text(seg.end_s.max() + 40, y[lane], ";  ".join(parts), va="center", fontsize=5.8, color=C["ink2"])
    a1.text(barrier - 30, -0.95, f"all nine base learners done\nt = {barrier:,.0f} s ({barrier / 60:.1f} min)",
            ha="right", va="bottom", fontsize=6.5, color=C["ink"])
    a1.set_xlabel("Elapsed wall-clock time from start of training, t (s)")

    # right panel: zoom on the meta stage
    a2.set_xlim(barrier - 4, barrier + 30)
    a2.set_xticks([barrier + d for d in (0, 10, 20, 30)])
    a2.set_xticklabels([f"{barrier:,.0f}"] + [f"+{d}" for d in (10, 20, 30)])
    for r in df[df.group.str.startswith("meta")].itertuples():
        if r.stage == "tune_untimed":
            continue
        txt = f"≤ {r.seconds:.1f} s" if r.stage == "meta_wall" else f"{r.seconds:.2f} s fit"
        a2.text(r.end_s + 0.5, y[r.lane], txt, va="center", fontsize=5.8, color=C["ink2"])
    a2.text(barrier + 0.5, -0.95, "meta-learners start\nonly after the barrier", va="bottom", fontsize=6.5, color=C["ink"])
    a2.set_xlabel("t (s), meta stage", labelpad=4)
    a2.tick_params(axis="y", length=0)

    # y axis and group separators
    a1.set_yticks(list(y.values()))
    a1.set_yticklabels(list(y.keys()), fontsize=6.5)
    a1.set_ylim(len(y) - 0.4, -1.1)
    for b in (4.5, 24.5, 28.5, 32.5):
        for ax in (a1, a2):
            ax.axhline(b, color=C["ink2"], linewidth=0.4)
    for yy, name in [(1.5, "Statistical\n(CPU)"), (14.5, "Neural\n(2 GPUs,\n20 processes)"),
                     (26.5, "Meta (A)\nprimary"), (30.5, "Meta (B)")]:
        a1.text(-0.235, yy, name, transform=a1.get_yaxis_transform(), ha="center", va="center", fontsize=6.5,
                color=C["ink"], rotation=90)

    handles = [Patch(color=COL["tune"], label="Base-learner tuning (15 GP evaluations x 2 folds)"),
               Patch(color=COL["roll"], label="Base-learner rolling re-estimation (23 refits, incl. forecasting)"),
               Patch(color=COL["bench"], label="Benchmarks (not ensemble inputs)"),
               Patch(color=COL["meta"], label="Meta-learner fitting (19 refits x 4 horizons; tuning not logged)"),
               Patch(facecolor="white", edgecolor=COL["meta"], hatch="////",
                     label="Whole meta stage incl. tuning, upper bound from file timestamps")]
    a1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.62, -0.075), ncol=2, frameon=False, fontsize=6.2)
    a1.text(0.62, -0.185, f"Stacking ensemble available at t ≤ {meta_end:,.0f} s ({meta_end / 60:.1f} min); "
             f"the meta stage is {100 * (meta_end - barrier) / meta_end:.1f}% of the elapsed time.",
             ha="center", fontsize=7, color=C["ink"], transform=a1.transAxes)
    return fig


def main():
    df, barrier = build_schedule()
    df.round(4).to_csv(RES / "training_schedule.csv", index=False)
    save(draw(df, barrier), "v2_training_gantt")
    nn = df[df.group == "nn"]
    print(f"barrier (all base learners done): {barrier:.1f} s")
    print(f"GPU busy time summed over 20 processes: {nn.seconds.sum():.0f} s "
          f"(tuning {nn[nn.stage == 'tune'].seconds.sum():.0f}, rolling {nn[nn.stage == 'roll'].seconds.sum():.0f})")
    print(f"CPU track (logged): {df[df.group.isin(['stat', 'bench'])].seconds.sum():.1f} s")
    print(df[df.group.str.startswith("meta")][["lane", "stage", "seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
