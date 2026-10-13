#!/usr/bin/env python3
"""
Lag-level correlation structure of the series (Supplementary Fig. S1): ACF and PACF of the raw and of the
first-differenced series, lags 1-90, with 95% intervals. Descriptive; results-independent.

    python3 lag_correlation.py   -> output/v2/figures/v2_lag_correlation_col.{pdf,png},
                                    output/v2/results/lag_correlation.csv
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

from common import LOOKBACK, load_series
from make_outputs import RES, save

ALPHA = 0.05
COLOR_RAW, COLOR_DIFF = "#1f739e", "#d90000"


def acf_pacf(x):
    r, r_ci = acf(x, nlags=LOOKBACK, alpha=ALPHA, fft=True)
    p, p_ci = pacf(x, nlags=LOOKBACK, alpha=ALPHA)
    return r[1:], r_ci[1:], p[1:], p_ci[1:]          # lag 0 dropped


def main():
    _, y, _ = load_series()
    raw, dif = acf_pacf(y), acf_pacf(np.diff(y))
    lags = np.arange(1, LOOKBACK + 1)
    cols = {}
    for tag, (r, r_ci, p, p_ci), n in (("raw", raw, len(y)), ("diff", dif, len(y) - 1)):
        cols.update({f"N_{tag}": n, f"ACF_{tag}": r, f"ACF_{tag}_lo": r_ci[:, 0], f"ACF_{tag}_hi": r_ci[:, 1],
                     f"PACF_{tag}": p, f"PACF_{tag}_lo": p_ci[:, 0], f"PACF_{tag}_hi": p_ci[:, 1]})
    pd.DataFrame({"Lag": lags, **cols}).to_csv(RES / "lag_correlation.csv", index=False, float_format="%.6f")

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.7), sharex=True)
    for ax, k, ylabel in ((axes[0], 0, "ACF"), (axes[1], 2, "PACF")):
        ax.fill_between(lags, raw[k + 1][:, 0], raw[k + 1][:, 1], color=COLOR_RAW, alpha=0.15, linewidth=0, zorder=1)
        ax.plot(lags, raw[k], color=COLOR_RAW, linewidth=0.8, marker="o", markersize=1.2, label="Raw series $y_t$", zorder=3)
        ax.plot(lags, dif[k], color=COLOR_DIFF, linewidth=0.6, marker="o", markersize=1.0, alpha=0.85,
                label="First-differenced $\\Delta y_t$", zorder=2)
        ax.axhline(0, color="black", linewidth=0.4, zorder=1)
        for wl in range(7, LOOKBACK + 1, 7):
            ax.axvline(wl, color="0.75", linewidth=0.35, linestyle=":", zorder=0)
        ax.set_ylabel(ylabel, fontsize=6.5)
        ax.set_xlim(0.5, LOOKBACK + 0.5)
        ax.tick_params(labelsize=6, length=2, pad=1.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].set_xlabel("Lag (days)", fontsize=6.5)
    axes[0].legend(loc="upper right", frameon=False, handlelength=1.5, fontsize=5.8)
    fig.tight_layout(h_pad=0.4)
    save(fig, "v2_lag_correlation_col")


if __name__ == "__main__":
    main()
