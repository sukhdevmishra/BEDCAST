#!/usr/bin/env python3
"""
Supplementary figure for the capacity-based exceedance event (adopted after the results were examined): occupancy above
EXCEED_RATE x BED_CAPACITY (85% of 643 sanctioned beds = 546.55 beds). Reads only evaluate.py outputs; no
analysis choice is made here.

  (a) daily occupancy as a percentage of sanctioned capacity, 2011-2020, with the capacity-based threshold,
      the originally specified 90th-percentile threshold, and the evaluation span;
  (b) Brier skill score against climatology by horizon (evaluation span), capacity-based threshold;
  (c, d) ensemble (ElasticNet, Variant A) exceedance probability over the evaluation span at h = 7 and h = 28,
      with the days on which the event occurred.

    python3 exceedance_capacity.py   -> output/v2/figures/v2_exceedance_capacity.{pdf,png}
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import BED_CAPACITY, EXCEED_RATE, EVAL_START, EVAL_END, CONF_START, HORIZONS, load_series
from make_outputs import C, RES, lab, save, style

PRIMARY = "ElasticNet-A"
HIGHLIGHT = {"SARIMA": C["comb"], "NaiveDOW": C["dow"], PRIMARY: C["primary"]}
NAMES = {"SARIMA": "SARIMA", "NaiveDOW": "Naive + weekday", PRIMARY: "Stacking (EN-A)"}


def main():
    dates, Y, _ = load_series()
    pb = pd.read_csv(RES / "probabilistic.csv")
    p90 = pd.read_csv(RES / "probabilistic_p90.csv")
    daily = pd.read_csv(RES / "probabilistic_exceedance_daily.csv.gz")
    thr = EXCEED_RATE * BED_CAPACITY
    thr90 = p90.threshold_beds.iloc[0]
    assert np.isclose(pb.threshold_beds.iloc[0], thr, atol=1e-3)
    pct = 100 * Y / BED_CAPACITY

    fig = plt.figure(figsize=(7.16, 6.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1], width_ratios=[1, 1.9], hspace=0.55, wspace=0.28)
    a = fig.add_subplot(gs[0, :])
    b = fig.add_subplot(gs[1:, 0])
    c1 = fig.add_subplot(gs[1, 1])
    c2 = fig.add_subplot(gs[2, 1], sharex=c1)

    # (a) occupancy relative to capacity
    a.axvspan(dates[EVAL_START], dates[EVAL_END], color="#e3eefc", linewidth=0, zorder=0)
    a.plot(dates, pct, color=C["grey"], linewidth=0.4, zorder=2)
    a.plot(dates, pd.Series(pct).rolling(30, center=True).mean(), color=C["ink"], linewidth=1.0, zorder=3)
    a.axhline(100, color=C["ink"], linewidth=1.0, zorder=4)
    a.axhline(100 * thr / BED_CAPACITY, color=C["dow"], linewidth=1.2, linestyle=(0, (4, 2)), zorder=4)
    # a.axhline(100 * thr90 / BED_CAPACITY, color=C["ink2"], linewidth=0.8, linestyle=(0, (1, 1.5)), zorder=4)
    a.legend(handles=[Line2D([], [], color=C["ink"], linewidth=1.0, label=f"Sanctioned capacity, {BED_CAPACITY} beds (100%)"),
                      Line2D([], [], color=C["dow"], linewidth=1.2, linestyle=(0, (4, 2)),
                             label=f"Event threshold, {100 * EXCEED_RATE:.0f}% of capacity = {thr:.2f} beds"),
                    #   Line2D([], [], color=C["ink2"], linewidth=0.8, linestyle=(0, (1, 1.5)),
                            #  label=f"Original threshold, {thr90:.1f} beds ({100 * thr90 / BED_CAPACITY:.1f}%)"),
                      Line2D([], [], color=C["ink"], linewidth=1.0, label="Occupancy, 30-day centred mean")],
             loc="lower left", bbox_to_anchor=(0.2, 0.02), frameon=False, fontsize=6.5)
    i_max = int(np.argmax(Y))
    a.scatter(dates[i_max], pct[i_max], s=14, color=C["ink"], zorder=6)
    a.annotate(f"Maximum {Y[i_max]:.0f} beds ({pct[i_max]:.1f}%), {dates[i_max].date()}", (dates[i_max], pct[i_max]),
               xytext=(-7, 0), textcoords="offset points", ha="right", va="center", fontsize=6.5, color=C["ink"])
    a.text(dates[(EVAL_START + EVAL_END) // 2], 52, "Evaluation span", ha="center", fontsize=7, color=C["ink2"])
    a.set_ylim(30, 106)
    a.set_xlim(dates[0], dates[3377])
    a.set_ylabel("Occupancy (% of capacity)")
    a.set_title("(a) Daily occupancy relative to sanctioned capacity", fontsize=8, loc="left", color=C["ink"])
    style(a)

    # (b) BSS by horizon
    ev = pb[pb.period == "evaluation"]
    x = np.arange(len(HORIZONS))
    for m in ev.model.unique():
        v = [ev[(ev.model == m) & (ev.h == h)].brier_skill_vs_clim.iloc[0] for h in HORIZONS]
        if m in HIGHLIGHT:
            b.plot(x, v, color=HIGHLIGHT[m], linewidth=2, marker="o", markersize=5, markeredgecolor="white", zorder=3)
        else:
            b.plot(x, v, color=C["grey"], linewidth=0.8, alpha=0.7, zorder=2)
    b.axhline(0, color=C["ink2"], linewidth=0.9, linestyle=(0, (4, 2)))
    b.text(0, 0.015, "climatology", ha="left", va="bottom", fontsize=6.5, color=C["ink2"])
    b.set_xticks(x, [str(h) for h in HORIZONS])
    b.set_xlabel("Forecast horizon (days)")
    b.set_ylabel("Brier skill score")
    b.set_ylim(-0.12, 0.85)
    b.set_title(f"(b) Skill for occupancy > {thr:.2f} beds", fontsize=8, loc="left", color=C["ink"])
    style(b)
    handles = [Line2D([], [], color=col, linewidth=2, marker="o", markersize=4, markeredgecolor="white", label=NAMES[m])
               for m, col in HIGHLIGHT.items()]
    handles.append(Line2D([], [], color=C["grey"], linewidth=0.8, label="Naive, ETS, Comb"))
    b.legend(handles=handles, frameon=False, fontsize=6.3, loc="upper right")

    # (c) daily exceedance probability of the ensemble
    for ax, h, tag in ((c1, 7, "c"), (c2, 28, "d")):
        d = daily[(daily.model == PRIMARY) & (daily.h == h)].sort_values("t")
        dt = dates[d.t.values]
        ev_days = d.y.values > thr
        ax.vlines(dt[ev_days], 0, 1, color="#fbe0d4", linewidth=0.9, zorder=0)
        ax.plot(dt, d.clim.values, color=C["ink2"], linewidth=0.9, linestyle=(0, (4, 2)), zorder=2)
        ax.plot(dt, d.p_exceed.values, color=C["primary"], linewidth=0.9, zorder=3)
        ax.axvline(dates[CONF_START], color=C["ink2"], linewidth=0.6, zorder=1)
        bss = ev[(ev.model == PRIMARY) & (ev.h == h)].brier_skill_vs_clim.iloc[0]
        ax.set_title(f"({tag}) Stacking (EN-A), {h}-day-ahead: P(occupancy > {thr:.2f}); BSS = {bss:+.2f}",
                     fontsize=7.5, loc="left", color=C["ink"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        style(ax)
    c1.text(dates[CONF_START], 1.0, " confirmatory", fontsize=6.3, color=C["ink2"], va="top")
    c1.set_xlim(dates[EVAL_START], dates[EVAL_END])
    c2.xaxis.set_major_locator(mdates.YearLocator())
    c2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(c1.get_xticklabels(), visible=False)
    handles = [Line2D([], [], color=C["primary"], linewidth=1.2, label="Forecast probability"),
               Line2D([], [], color=C["ink2"], linewidth=0.9, linestyle=(0, (4, 2)), label="Climatological rate"),
               Line2D([], [], color="#f4b89c", linewidth=4, label="Event occurred")]
    c2.legend(handles=handles, frameon=False, fontsize=6.3, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    fig.text(0.5, -0.005, f"Event: occupancy above {100 * EXCEED_RATE:.0f}% of the {BED_CAPACITY} sanctioned beds. "
             f"It occurred on {100 * ev.event_rate.iloc[0]:.1f}% of the 1,088 evaluation days.\nProbabilities: share of the "
             f"preceding 365 days' out-of-sample errors that, added to the point forecast, exceed the threshold.",
             ha="center", va="top", fontsize=6.3, color=C["ink2"])
    save(fig, "v2_exceedance_capacity")
    save(draw_col(dates, Y, pct, thr, ev, daily), "v2_exceedance_capacity_col")
    print(ev.pivot(index="model", columns="h", values="brier_skill_vs_clim").round(3).to_string())


def draw_col(dates, Y, pct, thr, ev, daily):
    """The same four panels sized for one journal column, with shorter labels (v2_exceedance_capacity_col)."""
    fs, fs_s = 6, 5.3
    fig = plt.figure(figsize=(3.5, 3.9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], width_ratios=[1, 1.55], hspace=0.62, wspace=0.42)
    a = fig.add_subplot(gs[0, :])
    b = fig.add_subplot(gs[1:, 0])
    c1 = fig.add_subplot(gs[1, 1])
    c2 = fig.add_subplot(gs[2, 1], sharex=c1)

    a.axvspan(dates[EVAL_START], dates[EVAL_END], color="#e3eefc", linewidth=0, zorder=0)
    a.plot(dates, pct, color=C["grey"], linewidth=0.3, zorder=2)
    a.plot(dates, pd.Series(pct).rolling(30, center=True).mean(), color=C["ink"], linewidth=0.8, zorder=3)
    a.axhline(100, color=C["ink"], linewidth=0.8, zorder=4)
    a.axhline(100 * thr / BED_CAPACITY, color=C["dow"], linewidth=1.0, linestyle=(0, (4, 2)), zorder=4)
    a.legend(handles=[Line2D([], [], color=C["ink"], linewidth=0.8, label=f"Capacity, {BED_CAPACITY} beds"),
                      Line2D([], [], color=C["dow"], linewidth=1.0, linestyle=(0, (4, 2)),
                             label=f"Threshold, {100 * EXCEED_RATE:.0f}% ({thr:.2f} beds)"),
                      Line2D([], [], color=C["ink"], linewidth=0.8, label="30-day mean")],
             loc="lower left", bbox_to_anchor=(0.16, 0.0), frameon=False, fontsize=fs_s, ncol=1, handlelength=1.6)
    a.text(dates[(EVAL_START + EVAL_END) // 2], 50, "Evaluation span", ha="center", fontsize=fs_s, color=C["ink2"])
    a.set_ylim(30, 106)
    a.set_xlim(dates[0], dates[3377])
    a.set_ylabel("% of capacity", fontsize=fs)
    a.set_title("(a) Occupancy relative to sanctioned capacity", fontsize=fs, loc="left", color=C["ink"], pad=2)
    style(a)

    x = np.arange(len(HORIZONS))
    for m in ev.model.unique():
        v = [ev[(ev.model == m) & (ev.h == h)].brier_skill_vs_clim.iloc[0] for h in HORIZONS]
        if m in HIGHLIGHT:
            b.plot(x, v, color=HIGHLIGHT[m], linewidth=1.3, marker="o", markersize=3, markeredgecolor="white", zorder=3)
        else:
            b.plot(x, v, color=C["grey"], linewidth=0.6, alpha=0.7, zorder=2)
    b.axhline(0, color=C["ink2"], linewidth=0.7, linestyle=(0, (4, 2)))
    b.text(0, 0.015, "climatology", ha="left", va="bottom", fontsize=fs_s, color=C["ink2"])
    b.set_xticks(x, [str(h) for h in HORIZONS])
    b.set_xlabel("Horizon (days)", fontsize=fs)
    b.set_ylabel("Brier skill score", fontsize=fs)
    b.set_ylim(-0.12, 0.85)
    b.set_title("(b) Skill by horizon", fontsize=fs, loc="left", color=C["ink"], pad=2)
    style(b)
    handles = [Line2D([], [], color=col, linewidth=1.3, marker="o", markersize=3, markeredgecolor="white", label=NAMES[m])
               for m, col in HIGHLIGHT.items()]
    handles.append(Line2D([], [], color=C["grey"], linewidth=0.6, label="Naive, ETS, Comb"))
    b.legend(handles=handles, frameon=False, fontsize=fs_s - 0.3, loc="upper right", handlelength=1.4)

    for ax, h, tag in ((c1, 7, "c"), (c2, 28, "d")):
        d = daily[(daily.model == PRIMARY) & (daily.h == h)].sort_values("t")
        dt = dates[d.t.values]
        ax.vlines(dt[d.y.values > thr], 0, 1, color="#fbe0d4", linewidth=0.6, zorder=0)
        ax.plot(dt, d.clim.values, color=C["ink2"], linewidth=0.7, linestyle=(0, (4, 2)), zorder=2)
        ax.plot(dt, d.p_exceed.values, color=C["primary"], linewidth=0.6, zorder=3)
        ax.axvline(dates[CONF_START], color=C["ink2"], linewidth=0.5, zorder=1)
        bss = ev[(ev.model == PRIMARY) & (ev.h == h)].brier_skill_vs_clim.iloc[0]
        ax.set_title(f"({tag}) Stacking, {h}-day-ahead; BSS {bss:+.2f}", fontsize=fs, loc="left", color=C["ink"], pad=2)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability", fontsize=fs)
        style(ax)
    c1.set_xlim(dates[EVAL_START], dates[EVAL_END])
    c2.xaxis.set_major_locator(mdates.YearLocator())
    c2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(c1.get_xticklabels(), visible=False)
    for ax in (a, b, c1, c2):
        ax.tick_params(labelsize=fs_s, length=2, pad=1.5)
    handles = [Line2D([], [], color=C["primary"], linewidth=1.0, label="Probability"),
               Line2D([], [], color=C["ink2"], linewidth=0.7, linestyle=(0, (4, 2)), label="Climatology"),
               Line2D([], [], color="#f4b89c", linewidth=3, label="Event")]
    c2.legend(handles=handles, frameon=False, fontsize=fs_s, ncol=3, loc="upper center", bbox_to_anchor=(0.42, -0.24),
              handlelength=1.4, columnspacing=0.8)
    return fig


if __name__ == "__main__":
    main()
