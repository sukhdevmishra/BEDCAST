#!/usr/bin/env python3
"""
Lag-level attribution, computed from the fitted models that produced the reported forecasts.

Lag convention: lag k = y[o - k + 1], the k-th most recent observation at origin o (lag 1 = y[o]).
Groups: {1}, {2}, {3-7}, {8-14}, {15-28}, {29-56}, {57-90}, plus the day-of-week group.

  * Classical learners: exact linear forecast weights, from the fit at the last refit.
      ARIMA / SARIMA: AR(inf) pi-weights (h = 1).
      ETS (simple exponential smoothing): alpha(1 - alpha)^(k-1).
      EMA: 2/(s+1) exponential weights.
  * Neural learners: signed grouped permutation importance (increase in MSE, beds^2), computed from the
    persisted, evaluated models over the evaluation span. Each refit block is permuted within itself, 10 repeats.
  * Meta-learner (ElasticNet-A): the prediction is y[o] + b + sum_j w_j (f_j - y[o]). The effective weights
    w_j and the persistence weight 1 - sum_j w_j therefore sum to 1. The propagated profile is
    sum_j w_j p_j + (1 - sum_j w_j) * delta(lag 1), where p_j is learner j's normalised lag-group profile,
    so the mass is conserved and the persistence channel is included.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma2ar

from common import (OUT, HORIZONS, EVAL_START, EVAL_END, LOOKBACK, SEED, REFITS, BASE_ALL, BASE_NN,
                    load_series, all_origins, refit_for_origin, load_json, ensure_dir, save_json)
import stat_models as sm

_, Y, DOW = load_series()
DOW0 = int(DOW[0])
ORIG = all_origins()
RES = ensure_dir(OUT / "results")
GROUPS = {"1": [1], "2": [2], "3-7": list(range(3, 8)), "8-14": list(range(8, 15)),
          "15-28": list(range(15, 29)), "29-56": list(range(29, 57)), "57-90": list(range(57, 91))}


def dow_fn(t):
    return (DOW0 + t) % 7


def group_profile(w_by_lag):
    """w_by_lag: array of length 90 (index k-1 = lag k). Returns group sums."""
    return {g: float(sum(w_by_lag[k - 1] for k in ks)) for g, ks in GROUPS.items()}


def classical_weights():
    params = load_json(OUT / "tuning" / "stat_params.json")
    R = REFITS[-1]
    out = {}
    # ARIMA: regression with ARIMA(p,d,q) errors -> weights of the error process on levels
    st = sm.arima_fit(Y, dow_fn, R, params["ARIMA"])
    p, d, q = st["p"], st["d"], st["q"]
    names = sm._arima_model(Y[:R + 1], sm.weekday_dummies(range(R + 1), dow_fn), p, d, q).param_names
    pv = dict(zip(names, st["params"]))
    ar = np.r_[1, [-pv[f"ar.L{i}"] for i in range(1, p + 1)]]
    ma = np.r_[1, [pv[f"ma.L{i}"] for i in range(1, q + 1)]]
    ar_full = ar
    for _ in range(d):                       # multiply by (1 - B)^d; coefficients ordered by increasing lag
        ar_full = np.convolve(ar_full, [1, -1])
    out["ARIMA"] = -arma2ar(ar_full, ma, lags=LOOKBACK + 1)[1:]
    # SARIMA
    st = sm.sarima_fit(Y, dow_fn, R, params["SARIMA"])
    s = params["SARIMA"]
    names = sm._sarima_model(Y[:R + 1], s).param_names
    pv = dict(zip(names, st["params"]))
    arp = np.r_[1, [-pv[f"ar.L{i}"] for i in range(1, s["p"] + 1)]]
    sar = np.zeros(7 * s["P"] + 1); sar[0] = 1
    for i in range(1, s["P"] + 1):
        sar[7 * i] = -pv[f"ar.S.L{7 * i}"]
    ma = np.r_[1, [pv[f"ma.L{i}"] for i in range(1, s["q"] + 1)]]
    sma = np.zeros(7 * s["Q"] + 1); sma[0] = 1
    for i in range(1, s["Q"] + 1):
        sma[7 * i] = pv[f"ma.S.L{7 * i}"]
    ar_all = np.convolve(arp, sar)
    for _ in range(s["d"]):
        ar_all = np.convolve(ar_all, [1, -1])
    for _ in range(s["D"]):
        ar_all = np.convolve(ar_all, np.r_[1, np.zeros(6), -1])
    out["SARIMA"] = -arma2ar(ar_all, np.convolve(ma, sma), lags=LOOKBACK + 1)[1:]
    # ETS: simple exponential smoothing (tuned configuration 0) or smoothing level of the chosen model
    st = sm.ets_fit(Y, dow_fn, R, params["ETS"])
    mdl = sm._ets_model(Y[:R + 1], st["cfg"])
    alpha = dict(zip(mdl.param_names, st["params"]))["smoothing_level"]
    k = np.arange(1, LOOKBACK + 1)
    out["ETS"] = alpha * (1 - alpha) ** (k - 1)
    a = 2 / (params["EMA"]["span"] + 1)
    out["EMA"] = a * (1 - a) ** (k - 1)
    rows = [{"model": m, "lag": int(j + 1), "weight": float(w)} for m, v in out.items() for j, w in enumerate(v)]
    pd.DataFrame(rows).to_csv(RES / "attribution_classical_lag_weights.csv", index=False, float_format="%.6f")
    return {m: group_profile(v) for m, v in out.items()}, {"ARIMA_order": (p, d, q), "ETS_alpha": float(alpha)}


def nn_importance(n_rep=10):
    import keras
    import nn_models as nn
    rng = np.random.default_rng(SEED)
    rows = []
    for arch in BASE_NN:
        for h in HORIZONS:
            tgt = np.arange(EVAL_START, EVAL_END + 1)
            o = tgt - h
            refit_of = np.array([refit_for_origin(x) for x in o])
            dmse = {g: 0.0 for g in list(GROUPS) + ["DOW"]}
            base_sse, n_tot = 0.0, 0
            for R in np.unique(refit_of):
                sel = o[refit_of == R]
                path = OUT / "models" / arch / f"h{h}" / f"R{R}.keras"
                m = keras.models.load_model(path)
                stt = np.load(str(path).replace(".keras", "_stats.npy"), allow_pickle=True).item()
                Xs, Xc = nn.make_xy(Y, dow_fn, sel, h, stt, with_target=False)
                truth = Y[sel + h]

                def pred(a, b):
                    return Y[sel] + m.predict([a, b], batch_size=1024, verbose=0).ravel() * stt["sd_h"]
                e0 = truth - pred(Xs, Xc)
                base_sse += np.sum(e0 ** 2); n_tot += len(sel)
                for g, lags in GROUPS.items():
                    pos = [LOOKBACK - k for k in lags]           # lag k sits at window position 90 - k
                    acc = 0.0
                    for _ in range(n_rep):
                        P = Xs.copy(); perm = rng.permutation(len(sel))
                        P[:, pos, :] = Xs[perm][:, pos, :]
                        acc += np.sum((truth - pred(P, Xc)) ** 2) - np.sum(e0 ** 2)
                    dmse[g] += acc / n_rep
                acc = 0.0
                for _ in range(n_rep):
                    perm = rng.permutation(len(sel))
                    acc += np.sum((truth - pred(Xs, Xc[perm])) ** 2) - np.sum(e0 ** 2)
                dmse["DOW"] += acc / n_rep
                keras.backend.clear_session()
            for g, v in dmse.items():
                rows.append({"model": arch, "h": h, "group": g, "delta_MSE": v / n_tot,
                             "baseline_MSE": base_sse / n_tot})
            print(arch, h, {g: round(v / n_tot, 3) for g, v in dmse.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "attribution_nn_grouped_permutation.csv", index=False, float_format="%.5f")
    return df


def meta_weights():
    cf = load_json(OUT / "models" / "elasticnet_coefficients.json")["ElasticNet-A"]
    rows = []
    for h in HORIZONS:
        for rec in cf[str(h)]:
            w = np.asarray(rec["coef"]) / np.asarray(rec["x_scale"])
            rows.append({"h": h, "R": rec["R"], **{m: float(v) for m, v in zip(BASE_ALL, w)},
                         "persistence": float(1 - w.sum()), "intercept": rec["intercept"]})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "attribution_meta_effective_weights.csv", index=False, float_format="%.5f")
    return df


def propagate(classical, nn_df, mw):
    rows = []
    for h in HORIZONS:
        w = mw[(mw.h == h) & (mw.R >= refit_for_origin(EVAL_START - h))][BASE_ALL + ["persistence"]].mean()
        prof = {}
        for m in BASE_ALL:
            if m in classical:
                p = classical[m]
            else:
                s = nn_df[(nn_df.model == m) & (nn_df.h == h)].set_index("group").delta_MSE.clip(lower=0)
                p = (s / s.sum()).to_dict() if s.sum() > 0 else {g: 0.0 for g in s.index}
            prof[m] = p
        groups = list(GROUPS) + ["DOW"]
        total = {g: 0.0 for g in groups}
        for m in BASE_ALL:
            for g in groups:
                total[g] += w[m] * prof[m].get(g, 0.0)
        total["1"] += w["persistence"]
        rows += [{"h": h, "group": g, "propagated_mass": v} for g, v in total.items()]
        rows.append({"h": h, "group": "TOTAL (should be ~1)", "propagated_mass": sum(total.values())})
    pd.DataFrame(rows).to_csv(RES / "attribution_propagated_ensemble.csv", index=False, float_format="%.4f")


def main():
    classical, info = classical_weights()
    save_json({"classical_group_profiles": classical, **info}, RES / "attribution_classical_summary.json")
    nn_df = nn_importance()
    mw = meta_weights()
    propagate(classical, nn_df, mw)


if __name__ == "__main__":
    main()


def contribution_decomposition():
    """Post hoc (not part of the experimental plan). The weights w_j are ill-conditioned because some
    base forecasts (ETS, and ARIMA at h >= 7) are numerically identical to the naive forecast. The ensemble
    forecast is y_o + D with D = b + sum_j w_j x_j, where x_j = f_j - y_o. We report (i) the departure ratio
    RMS(D) / RMS(y_{o+h} - y_o), i.e. how far the ensemble moves from persistence relative to how far occupancy
    actually moves, and (ii) each channel's share of the departure: mean |w_j (x_j - mean x_j)|, normalised
    (linear SHAP)."""
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cf = load_json(OUT / "models" / "elasticnet_coefficients.json")["ElasticNet-A"]
    rows = []
    for h in HORIZONS:
        B = np.column_stack([np.load(OUT / "forecasts" / f"{m}.npz")["F"][:, h - 1] for m in BASE_ALL])
        X = B - Y[ORIG][:, None]
        recs = {r["R"]: r for r in cf[str(h)]}
        tgt = np.arange(EVAL_START, EVAL_END + 1)
        o = tgt - h
        Fen = np.load(OUT / "forecasts" / "meta_ElasticNet-A.npz")["F"][:, h - 1]
        D = Fen[o - ORIG[0]] - Y[o]                      # exact departure of the ensemble from persistence
        phi = np.zeros((len(o), len(BASE_ALL)))
        for i, oo in enumerate(o):
            r = recs[refit_for_origin(oo)]
            w = np.asarray(r["coef"]) / np.asarray(r["x_scale"])
            x = X[oo - ORIG[0]]
            phi[i] = w * x
        # centre contributions within the evaluation span (linear SHAP)
        phi_c = phi - phi.mean(0)
        share = np.abs(phi_c).mean(0); share = share / share.sum()
        move = Y[tgt] - Y[o]
        rows.append({"h": h, "departure_ratio_RMS": float(np.sqrt(np.mean(D ** 2)) / np.sqrt(np.mean(move ** 2))),
                     **{f"share_{m}": float(s) for m, s in zip(BASE_ALL, share)}})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "attribution_contribution_shares_posthoc.csv", index=False, float_format="%.4f")
    return df


def propagate_posthoc(classical, nn_df, shares):
    """Post hoc: lag profile of the ensemble's departure from persistence,
    sum_j share_j * p_j, where p_j is learner j's normalised lag-group profile."""
    rows = []
    groups = list(GROUPS) + ["DOW"]
    for h in HORIZONS:
        sh = shares[shares.h == h].iloc[0]
        tot = {g: 0.0 for g in groups}
        for m in BASE_ALL:
            if m in classical:
                p = {g: max(0.0, v) for g, v in classical[m].items()}
            else:
                s = nn_df[(nn_df.model == m) & (nn_df.h == h)].set_index("group").delta_MSE.clip(lower=0)
                p = (s / s.sum()).to_dict() if s.sum() > 0 else {}
            z = sum(p.values()) or 1.0
            for g in groups:
                tot[g] += sh[f"share_{m}"] * p.get(g, 0.0) / z
        rows += [{"h": h, "group": g, "departure_mass": v} for g, v in tot.items()]
        rows.append({"h": h, "group": "departure_ratio_RMS", "departure_mass": sh.departure_ratio_RMS})
    pd.DataFrame(rows).to_csv(RES / "attribution_departure_profile_posthoc.csv", index=False, float_format="%.4f")
