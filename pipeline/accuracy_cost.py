#!/usr/bin/env python3
"""
RMSE against training time: one point per model and horizon, coloured by learner class, shaped by level
(circle = base/level 1, diamond = level 2), each labelled with its model name. Comb, a benchmark that averages three
statistical learners, is drawn as a filled diamond (a combination) in the statistical-learner colour. Post-hoc descriptive figure;
no analysis choice is made here.

Accuracy: RMSE (beds) over the evaluation span (output/v2/results/point_metrics.csv).
Training time: logged compute seconds needed before the model can issue its evaluation forecasts at horizon h
(output/v2/timing.csv via gantt.build_schedule), on the rolling 91-day re-estimation schedule:
  - statistical models and benchmarks: tuning + all 23 rolling re-estimations incl. forecasting (one fit serves
    every horizon); the naive forecast has no fit, so its time is forecasting only;
  - networks: tuning + 23 re-estimations of the horizon-h network (one process per horizon);
  - Comb: its three members (ARIMA, SARIMA, ETS);
  - meta-learners: all nine base learners at horizon h + 19 meta re-estimations at h. A stacked model cannot
    exist without its base learners, so it is charged for them. Summed compute, not elapsed time, so the
    figure does not depend on how many GPUs or workers ran in parallel.
ETS tuning and meta-learner tuning were not logged, so those training times are lower bounds.

    python3 accuracy_cost.py   -> output/v2/figures/v2_accuracy_vs_time.{pdf,png},
                                  output/v2/results/accuracy_vs_time_learners.csv
"""
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from common import OUT, HORIZONS, BASE_ALL, BASE_NN, META
from gantt import build_schedule
from make_outputs import C, RES, save

STAT = ["ARIMA", "SARIMA", "EMA", "ETS"]
BENCH = ["Naive", "NaiveDOW", "SeasonalNaive"]   # the AR benchmark is estimated but not reported


def training_seconds():
    sch, _ = build_schedule()
    rows = []
    stat = {m: sch[(sch.lane == m) & sch.stage.isin(["tune", "roll"])].seconds.sum() for m in STAT}
    bench = sch[sch.group == "bench"].set_index("note").seconds
    for h in HORIZONS:
        nn = {a: sch[sch.lane == f"{a} h={h}"].seconds.sum() for a in BASE_NN}
        base = {**stat, **nn}
        for m in BASE_ALL:
            rows.append((h, m, "Statistical" if m in STAT else "Neural", "Base", base[m]))
        for m in BENCH:
            rows.append((h, m, "Benchmark", "Base", bench[m]))
        rows.append((h, "Comb", "Benchmark", "Combination", stat["ARIMA"] + stat["SARIMA"] + stat["ETS"]))
        base_total = sum(base.values())
        for v in "AB":
            for m in META:
                fit = meta_fit(m, v, h)
                rows.append((h, f"{m}-{v}", "Meta", f"Meta ({v})", base_total + fit))
    return pd.DataFrame(rows, columns=["h", "model", "cls", "level", "train_s"])


def meta_fit(m, v, h):
    t = pd.read_csv(OUT / "timing.csv")
    return t[(t.model == f"{m}-{v}") & (t.horizon == h) & (t.stage == "fit")].seconds.sum()


CCOL = {"Benchmark": C["grey"], "Statistical": "#1baf7a", "Neural": C["primary"], "Meta": C["dow"]}
COMB_LABEL = "Comb benchmark: mean of ARIMA, SARIMA, ETS (level 2)"


def colour(r):
    """Comb is a benchmark but averages statistical learners, so it takes their colour."""
    return CCOL["Statistical"] if r.model == "Comb" else CCOL[r.cls]
NAME = {"Naive": "Naive", "NaiveDOW": "Naive+weekday", "SeasonalNaive": "Seasonal naive", "Comb": "Comb",
        **{f"{m}-{v}": f"{a}-{v}" for m, a in zip(META, ["EN", "DT", "RF", "LGBM"]) for v in "AB"}}
B_OFFSET = 1.8   # display-only horizontal nudge for Variant B (values in the CSV are unshifted)
LABEL_PT = 5.4


def place_labels(ax, pts, obstacles=()):
    """Labels beside their points (right, or left when side == "left"), pushed apart vertically (in pixels)
    wherever two labels would share horizontal space; a thin leader joins a displaced label to its point.
    obstacles: further (x, y) data points, unlabelled, that labels must also avoid."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px = fig.dpi / 72
    gap = 4 * px
    items = []
    for x, y, text, side in pts:
        X, Y = ax.transData.transform((x, y))
        t = ax.text(0, 0, text, fontsize=LABEL_PT)
        w, hgt = t.get_window_extent(rend).width, t.get_window_extent(rend).height
        t.remove()
        x0 = X - gap - w if side == "left" else X + gap
        items.append({"x0": x0, "x1": x0 + w, "y": Y, "Y": Y, "h": hgt * 1.05, "px": X, "text": text, "side": side})
    marks = [(d["px"], d["y"]) for d in items]   # every marker is a fixed obstacle for every other label
    marks += [tuple(ax.transData.transform(o)) for o in obstacles]
    rad = 4 * px
    for _ in range(2000):
        moved = False
        for d in items:
            for mx, my in marks:
                if mx == d["px"] and my == d["y"]:
                    continue
                if d["x1"] < mx - rad or d["x0"] > mx + rad:
                    continue
                need = d["h"] / 2 + rad - abs(d["Y"] - my)
                if need > 0.01:
                    d["Y"] += need if d["Y"] >= my else -need
                    moved = True
        items.sort(key=lambda d: d["Y"])
        for i, a_ in enumerate(items):
            for b_ in items[i + 1:]:
                if a_["x1"] < b_["x0"] or b_["x1"] < a_["x0"]:
                    continue
                need = (a_["h"] + b_["h"]) / 2 - (b_["Y"] - a_["Y"])
                if need > 0.01:
                    a_["Y"] -= need / 2
                    b_["Y"] += need / 2
                    moved = True
        if not moved:
            break
    inv = ax.transData.inverted()
    halo = [pe.withStroke(linewidth=1.6, foreground="white")]
    for d in items:
        left = d["side"] == "left"
        tx, ty = inv.transform((d["x1"] if left else d["x0"], d["Y"]))
        ax.text(tx, ty, d["text"], fontsize=LABEL_PT, color=C["ink2"], va="center", ha="right" if left else "left",
                zorder=3, path_effects=halo)
        if abs(d["Y"] - d["y"]) > 2 * px:
            sgn = -1 if left else 1
            p0 = inv.transform((d["px"] + sgn * 2.5 * px, d["y"]))
            p1 = inv.transform(((d["x1"] + 0.5 * px) if left else (d["x0"] - 0.5 * px), d["Y"]))
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=C["grey"], linewidth=0.4, zorder=2)


def draw(df):
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 8.6))
    labels = {}
    for ax, h in zip(axes.flat, HORIZONS):
        d = df[df.h == h].copy()
        d["x"] = d.train_s * d.level.eq("Meta (B)").map({True: B_OFFSET, False: 1.0})
        naive = d.loc[d.model == "Naive", "RMSE"].iloc[0]
        # y range from the non-outlying points; anything far above is pinned to the top edge
        core = d[d.RelRMSE_naive < 1.3]
        lo, hi = core.RMSE.min(), core.RMSE.max()
        pad = 0.06 * (hi - lo)
        top = hi + 2.0 * pad
        ax.set_ylim(lo - pad, top)
        ax.set_xscale("log")
        ax.set_xlim(0.15, 2e5)
        ax.axhline(naive, color=C["ink2"], linewidth=0.7, linestyle=(0, (4, 2)), zorder=1)
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
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.grid(color=C["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("Training time (compute s, log scale)")
        ax.set_ylabel("RMSE (beds)")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
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
    fig.text(0.5, -0.045, "Lower is better. EN = ElasticNet, DT = DecisionTree, RF = RandomForest, LGBM = LightGBM meta-learner; "
             "-A / -B = stacking variant.\nStacking time includes its nine base learners. Variant B is drawn shifted "
             "right for visibility (its cost equals Variant A's).", ha="center", fontsize=6.3, color=C["ink2"])
    return fig


COL_LABELS = {"Naive", "NaiveDOW", "SARIMA", "Comb", f"{META[0]}-A"}   # the models the text discusses


def draw_col(df):
    """One-column version of draw(): same points and encoding, labels only on the models the text discusses."""
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.35), sharex=True)
    labels, others = {}, {}
    for i, (ax, h) in enumerate(zip(axes.flat, HORIZONS)):
        d = df[df.h == h].copy()
        d["x"] = d.train_s * d.level.eq("Meta (B)").map({True: B_OFFSET, False: 1.0})
        naive = d.loc[d.model == "Naive", "RMSE"].iloc[0]
        core = d[d.RelRMSE_naive < 1.3]
        lo, hi = core.RMSE.min(), core.RMSE.max()
        pad = 0.06 * (hi - lo)
        top = hi + 2.0 * pad
        ax.set_ylim(lo - 3 * pad, top)   # room below the lowest point for a displaced label
        ax.set_xscale("log")
        ax.set_xlim(0.15, 2e5)
        ax.axhline(naive, color=C["ink2"], linewidth=0.6, linestyle=(0, (4, 2)), zorder=1)
        labels[ax], others[ax] = [], []
        for r in d.itertuples():
            col = colour(r)
            hollow = r.level == "Meta (B)"
            y, mk, name = r.RMSE, ("o" if r.level == "Base" else "D"), NAME.get(r.model, r.model)
            if r.RMSE > top:
                y, mk, name = top - 0.4 * pad, "^", f"{name} ({r.RMSE:.1f}, off scale)"
            ax.scatter(r.x, y, s=11, marker=mk, zorder=5 if r.model == "Comb" else 4, linewidth=0.7,
                       facecolor="white" if hollow else col, edgecolor=col if hollow else "white")
            if r.model in COL_LABELS or mk == "^":
                labels[ax].append((r.x, y, name, "left" if r.level == "Meta (A)" or r.model == "Comb" else "right"))
            else:
                others[ax].append((r.x, y))
        ax.set_title(f"{h}-day-ahead", fontsize=6.5, color=C["ink"], loc="left", pad=2)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.grid(color=C["grid"], linewidth=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=5.5, length=2, pad=1.5)
        if i >= 2:
            ax.set_xlabel("Training compute (s, log scale)", fontsize=6)
        if i % 2 == 0:
            ax.set_ylabel("RMSE (beds)", fontsize=6)
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=0.6, w_pad=0.6)
    for ax, pts in labels.items():
        place_labels(ax, pts, others[ax])
    handles = [Line2D([], [], marker="o", linestyle="", markersize=3.5, markerfacecolor=CCOL[c], markeredgecolor="white",
                      label=c) for c in ["Benchmark", "Statistical", "Neural"]]
    handles += [Line2D([], [], marker="D", linestyle="", markersize=3.5, markerfacecolor=CCOL["Statistical"],
                       markeredgecolor="white", label="Comb (benchmark)"),
                Line2D([], [], marker="D", linestyle="", markersize=3.5, markerfacecolor=CCOL["Meta"],
                       markeredgecolor="white", label="Stacking A"),
                Line2D([], [], marker="D", linestyle="", markersize=3.5, markerfacecolor="white",
                       markeredgecolor=CCOL["Meta"], label="Stacking B"),
                Line2D([], [], color=C["ink2"], linewidth=0.6, linestyle=(0, (4, 2)), label="Naive RMSE")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=5.5,
               handletextpad=0.3, columnspacing=0.8)
    return fig


def main():
    df = training_seconds()
    pm = pd.read_csv(RES / "point_metrics.csv")
    pm = pm[pm.period == "evaluation"][["h", "model", "RelRMSE_naive", "RMSE"]]
    df = df.merge(pm, on=["h", "model"], how="left", validate="one_to_one")
    assert df.RelRMSE_naive.notna().all()
    df.round(4).to_csv(RES / "accuracy_vs_time_learners.csv", index=False)
    save(draw(df), "v2_accuracy_vs_time")
    save(draw_col(df), "v2_accuracy_vs_time_col")
    print(df.pivot(index="model", columns="h", values="train_s").round(1).to_string())


if __name__ == "__main__":
    main()
