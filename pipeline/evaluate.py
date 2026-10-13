#!/usr/bin/env python3
"""
Evaluation and inference. Reads output/v2/forecasts/*.npz; writes output/v2/results/.

Point accuracy, the Model Confidence Set, the three pre-specified primary comparisons (Clark-West /
HAC-DM with HLN correction, Holm within horizon, confirmatory sign check), probabilistic scores,
Giacomini-Rossi fluctuation tests, HAC sup-Wald bias-shift tests, and the base-learner error-correlation matrix.
"""
import numpy as np
import pandas as pd
from arch.bootstrap import MCS
from scipy import stats

from common import (OUT, HORIZONS, EVAL_START, EVAL_END, CONF_START, LAST_INDEX, DEV_END, SEED, REFITS,
                    BASE_ALL, META, BED_CAPACITY, EXCEED_RATE, load_series, all_origins, refit_for_origin,
                    ensure_dir, save_json)

_, Y, DOW = load_series()
ORIG = all_origins()
RES = ensure_dir(OUT / "results")
BENCH = ["Naive", "NaiveDOW", "SeasonalNaive", "Comb"]   # the AR benchmark is estimated but not evaluated
METAS = [f"{m}-{v}" for v in ("A", "B") for m in META]
PRIMARY = "ElasticNet-A"
PROB_MODELS = ["Naive", "NaiveDOW", "Comb", "SARIMA", "ETS", PRIMARY]
ALL = BASE_ALL + BENCH + METAS


def load_all():
    F = {}
    for m in BASE_ALL + BENCH:
        F[m] = np.load(OUT / "forecasts" / f"{m}.npz")["F"]
    for m in METAS:
        F[m] = np.load(OUT / "forecasts" / f"meta_{m}.npz")["F"]
    return F


def frame(F, h, t0, t1):
    """Forecasts for targets t0..t1 at horizon h (origin = target - h)."""
    tgt = np.arange(t0, t1 + 1)
    rows = tgt - h - ORIG[0]
    df = pd.DataFrame({m: F[m][rows, h - 1] for m in F}, index=tgt)
    df.insert(0, "y", Y[tgt])
    df.insert(1, "origin", tgt - h)
    return df


# ── HAC helpers ───────────────────────────────────────────────────────────────
def bw(n, h):
    return max(h - 1, int(np.floor(1.5 * n ** (1 / 3))))


def lrv(x, L):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    v = x @ x / n
    for k in range(1, L + 1):
        v += 2 * (1 - k / (L + 1)) * (x[k:] @ x[:-k]) / n
    return v


def dm_hln(e_a, e_b, h):
    """Two-sided DM on squared loss with HAC variance and HLN correction. Positive => A worse."""
    d = e_a ** 2 - e_b ** 2
    n = len(d)
    s = d.mean() / np.sqrt(lrv(d, bw(n, h)) / n)
    s *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(s), float(2 * stats.t.sf(abs(s), n - 1)), float(d.mean())


def clark_west(e_small, e_big, f_small, f_big, h):
    """One-sided Clark-West test of H0: the nesting model does not improve on the nested one."""
    f = e_small ** 2 - (e_big ** 2 - (f_small - f_big) ** 2)
    n = len(f)
    s = f.mean() / np.sqrt(lrv(f, bw(n, h)) / n)
    return float(s), float(stats.norm.sf(s)), float(np.mean(e_small ** 2 - e_big ** 2))


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ── simulated critical values (seeded) ────────────────────────────────────────
def _bm(n_paths=20000, n=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((n_paths, n)) / np.sqrt(n), axis=1)


_BM = None


def gr_critical(mu=0.3, alpha=0.05):
    global _BM
    _BM = _bm() if _BM is None else _BM
    n = _BM.shape[1]
    m = int(round(mu * n))
    B = np.concatenate([np.zeros((_BM.shape[0], 1)), _BM], axis=1)
    stat = np.max(np.abs(B[:, m:] - B[:, :-m]), axis=1) / np.sqrt(mu)
    return float(np.quantile(stat, 1 - alpha))


def supwald_null(trim=0.15):
    global _BM
    _BM = _bm() if _BM is None else _BM
    n = _BM.shape[1]
    tau = np.arange(1, n + 1) / n
    keep = (tau >= trim) & (tau <= 1 - trim)
    bb = _BM - tau[None, :] * _BM[:, -1:]
    return np.max((bb[:, keep] ** 2) / (tau[keep] * (1 - tau[keep])), axis=1)


def gr_fluctuation(d, h, mu=0.3):
    n = len(d)
    m = int(round(mu * n))
    sig = np.sqrt(lrv(d, bw(n, h)))
    c = np.concatenate([[0.0], np.cumsum(d)])
    Ft = (c[m:] - c[:-m]) / (sig * np.sqrt(m))
    return float(np.max(np.abs(Ft))), Ft


def supwald_mean_shift(e, h, null_dist, trim=0.15):
    n = len(e)
    om = lrv(e, bw(n, h))
    ks = np.arange(int(np.ceil(trim * n)), int(np.floor((1 - trim) * n)) + 1)
    c = np.cumsum(e)
    W = [((c[k - 1] / k - (c[-1] - c[k - 1]) / (n - k)) ** 2) / (om * (1 / k + 1 / (n - k))) for k in ks]
    s = float(max(W))
    return s, float(np.mean(null_dist >= s))


# ── main ──────────────────────────────────────────────────────────────────────
def point_metrics(F):
    rows = []
    for h in HORIZONS:
        mase_scale = np.mean(np.abs(Y[h:DEV_END + 1] - Y[:DEV_END + 1 - h]))
        for period, (a, b) in {"evaluation": (EVAL_START, EVAL_END), "confirmatory": (CONF_START, EVAL_END),
                               "lockdown_tail": (EVAL_END + 1, LAST_INDEX)}.items():
            df = frame(F, h, a, b)
            e_naive = df.y - df.Naive
            e_comb = df.y - df.Comb
            for m in ALL:
                e = df.y - df[m]
                rows.append({"h": h, "period": period, "model": m, "n": len(e),
                             "RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e)),
                             "MASE": np.mean(np.abs(e)) / mase_scale,
                             "RelRMSE_naive": np.sqrt(np.mean(e ** 2) / np.mean(e_naive ** 2)),
                             "RelRMSE_comb": np.sqrt(np.mean(e ** 2) / np.mean(e_comb ** 2))})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "point_metrics.csv", index=False, float_format="%.4f")
    return out


def mcs(F):
    rows = []
    for h in HORIZONS:
        for period, (a, b) in {"evaluation": (EVAL_START, EVAL_END), "confirmatory": (CONF_START, EVAL_END)}.items():
            df = frame(F, h, a, b)
            L = pd.DataFrame({m: (df.y - df[m]) ** 2 for m in ALL})
            # Forecasts identical to an earlier model (e.g. seasonal naive == naive when h is a multiple of 7)
            # make the loss differential identically zero; the MCS is run on distinct models and a
            # duplicate inherits its twin's result.
            twin = {}
            for i, m in enumerate(ALL):
                for m0 in ALL[:i]:
                    if m0 not in twin and np.allclose(df[m].values, df[m0].values):
                        twin[m] = m0
                        break
            distinct = [m for m in ALL if m not in twin]
            mc = MCS(L[distinct], size=0.10, reps=10000, block_size=max(7, h), method="R", bootstrap="stationary", seed=SEED)
            mc.compute()
            pv = mc.pvalues.iloc[:, 0]
            for m in ALL:
                ref = twin.get(m, m)
                rows.append({"h": h, "period": period, "model": m, "MCS_pvalue": float(pv.get(ref, np.nan)),
                             "in_MCS90": ref in mc.included, "identical_to": twin.get(m, "")})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "mcs.csv", index=False, float_format="%.4f")
    return out


def primary(F):
    rows = []
    for h in HORIZONS:
        full = frame(F, h, EVAL_START, EVAL_END)
        conf = frame(F, h, CONF_START, EVAL_END)
        res = []
        for bench in ["Naive", "NaiveDOW", "Comb"]:
            out = {}
            for tag, df in (("full", full), ("conf", conf)):
                e_m, e_b = (df.y - df[PRIMARY]).values, (df.y - df[bench]).values
                if bench == "Naive":
                    s, p, md = clark_west(e_b, e_m, df[bench].values, df[PRIMARY].values, h)
                    out[tag] = (s, p, -md)      # report mean loss difference as (primary - bench)
                else:
                    s, p, md = dm_hln(e_m, e_b, h)
                    out[tag] = (s, p, md)
            res.append({"h": h, "comparison": f"{PRIMARY} vs {bench}",
                        "test": "Clark-West (one-sided)" if bench == "Naive" else "HAC-DM, HLN (two-sided)",
                        "stat": out["full"][0], "p": out["full"][1],
                        "mean_sq_loss_diff_primary_minus_bench": out["full"][2],
                        "conf_mean_sq_loss_diff": out["conf"][2], "conf_stat": out["conf"][0],
                        "conf_p_unadjusted": out["conf"][1]})
        adj = holm([r["p"] for r in res])
        for r, pa in zip(res, adj):
            r["p_holm"] = pa
            better = r["mean_sq_loss_diff_primary_minus_bench"] < 0
            r["decision"] = ("primary better" if better else "primary worse") if pa < 0.05 else "no significant difference"
            r["confirmatory_same_sign"] = bool(np.sign(r["conf_mean_sq_loss_diff"]) == np.sign(r["mean_sq_loss_diff_primary_minus_bench"]))
            r["claim_supported"] = bool(pa < 0.05 and r["confirmatory_same_sign"])
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "primary_comparisons.csv", index=False, float_format="%.5f")
    return out


def probabilistic(F, thr=None, name="probabilistic"):
    """Interval and exceedance scores. Primary event (adopted after the results were examined): occupancy above
    EXCEED_RATE x BED_CAPACITY. Called a second time with the originally specified threshold (90th percentile of
    targets < EVAL_START) for the record; interval metrics do not depend on the threshold."""
    taus = [0.025, 0.1, 0.9, 0.975]
    if thr is None:
        thr = EXCEED_RATE * BED_CAPACITY
    rows, daily = [], []
    for h in HORIZONS:
        for m in PROB_MODELS:
            f_all = F[m][:, h - 1]
            tgt_all = ORIG + h
            ok = np.isfinite(f_all) & (tgt_all <= LAST_INDEX)
            err_all = np.full(len(ORIG), np.nan)
            err_all[ok] = Y[tgt_all[ok]] - f_all[ok]
            recs = []
            for t in range(EVAL_START, EVAL_END + 1):
                o = t - h
                R = refit_for_origin(o)
                past = ok & (tgt_all <= R) & (tgt_all > R - 365)
                ep = err_all[past]
                if len(ep) < 100:
                    continue
                f = f_all[o - ORIG[0]]
                q = np.quantile(ep, taus)
                clim = np.mean(Y[:R + 1] > thr)
                p_exc = np.mean(f + ep > thr)
                yv = Y[t]
                pin = np.mean([max(tau * (yv - (f + qq)), (tau - 1) * (yv - (f + qq))) for tau, qq in zip(taus, q)])
                recs.append({"t": t, "cov80": f + q[1] <= yv <= f + q[2], "cov95": f + q[0] <= yv <= f + q[3],
                             "w80": q[2] - q[1], "w95": q[3] - q[0], "pinball": pin,
                             "brier": (p_exc - (yv > thr)) ** 2, "brier_clim": (clim - (yv > thr)) ** 2})
                daily.append({"h": h, "model": m, "t": t, "p_exceed": p_exc, "clim": clim, "y": yv})
            r = pd.DataFrame(recs)
            for period, a in (("evaluation", EVAL_START), ("confirmatory", CONF_START)):
                s = r[r.t >= a]
                rows.append({"h": h, "model": m, "period": period, "n": len(s),
                             "coverage80": s.cov80.mean(), "coverage95": s.cov95.mean(),
                             "width80": s.w80.mean(), "width95": s.w95.mean(), "pinball": s.pinball.mean(),
                             "brier": s.brier.mean(), "brier_skill_vs_clim": 1 - s.brier.mean() / s.brier_clim.mean(),
                             "threshold_beds": thr, "event_rate": np.mean(Y[s.t.values] > thr)})
    out = pd.DataFrame(rows)
    out.to_csv(RES / f"{name}.csv", index=False, float_format="%.4f")
    pd.DataFrame(daily).to_csv(RES / f"{name}_exceedance_daily.csv.gz", index=False, float_format="%.4f")
    return out


def stability(F):
    crit = gr_critical()
    null = supwald_null()
    gr_rows, sw_rows, paths = [], [], {}
    for h in HORIZONS:
        df = frame(F, h, EVAL_START, EVAL_END)
        for bench in ["Naive", "NaiveDOW", "Comb"]:
            d = ((df.y - df[PRIMARY]) ** 2 - (df.y - df[bench]) ** 2).values
            s, Ft = gr_fluctuation(d, h)
            paths[f"h{h}_{bench}"] = Ft.tolist()
            gr_rows.append({"h": h, "comparison": f"{PRIMARY} vs {bench}", "max_abs_fluctuation": s,
                            "critical_5pct_mu0.3": crit, "reject_stability": s > crit})
        ps = []
        for m in PROB_MODELS:
            s, p = supwald_mean_shift((df.y - df[m]).values, h, null)
            ps.append(p)
            sw_rows.append({"h": h, "model": m, "supWald_HAC": s, "p": p})
        for r, pa in zip(sw_rows[-len(PROB_MODELS):], holm(ps)):
            r["p_holm"] = pa
    pd.DataFrame(gr_rows).to_csv(RES / "stability_fluctuation.csv", index=False, float_format="%.4f")
    pd.DataFrame(sw_rows).to_csv(RES / "stability_bias_shift.csv", index=False, float_format="%.4f")
    save_json(paths, RES / "fluctuation_paths.json")


def error_correlation(F):
    df = frame(F, 1, EVAL_START, EVAL_END)
    E = pd.DataFrame({m: df.y - df[m] for m in BASE_ALL})
    E.corr().to_csv(RES / "base_error_correlation_h1.csv", float_format="%.3f")


def main():
    F = load_all()
    pm = point_metrics(F)
    mcs(F)
    pr = primary(F)
    probabilistic(F)
    probabilistic(F, float(np.quantile(Y[:EVAL_START], 0.90)), "probabilistic_p90")
    stability(F)
    error_correlation(F)
    ev = pm[pm.period == "evaluation"].pivot(index="model", columns="h", values="RelRMSE_naive")
    print(ev.sort_values(1).round(4).to_string())
    print(pr[["h", "comparison", "stat", "p", "p_holm", "decision", "confirmatory_same_sign", "claim_supported"]].to_string())


if __name__ == "__main__":
    import sys
    if "--probabilistic" in sys.argv:   # re-run only the probabilistic scores
        F = load_all()
        probabilistic(F)
        probabilistic(F, float(np.quantile(Y[:EVAL_START], 0.90)), "probabilistic_p90")
    else:
        main()
