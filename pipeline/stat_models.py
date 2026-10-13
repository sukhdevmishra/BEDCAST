"""
Statistical base learners and benchmarks. All are fitted on the UNSCALED series.

Each model exposes:
    fit(y, dow_fn, R, params)   -> state        (uses y[0..R] only)
    forecast(state, y, dow_fn, o) -> array[H_MAX] of forecasts for targets o+1 .. o+H_MAX (uses y[0..o] only)
dow_fn(t) returns the weekday of any index t, including indices beyond the data.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from common import H_MAX

warnings.filterwarnings("ignore")


def weekday_dummies(idx, dow_fn):
    """Six dummies (Tuesday..Sunday; Monday is the reference) for integer indices idx."""
    d = np.array([dow_fn(t) for t in idx])
    X = np.zeros((len(idx), 6))
    for j in range(1, 7):
        X[d == j, j - 1] = 1.0
    return X


# ── ARIMA: regression on weekday dummies with ARIMA(p,d,q) errors ─────────────
def _arima_model(y, X, p, d, q):
    return SARIMAX(y, exog=X, order=(p, d, q), trend="c" if d == 0 else "n")


def arima_fit(y, dow_fn, R, params):
    p, d, q = params["p"], params["d"], params["q"]
    X = weekday_dummies(range(R + 1), dow_fn)
    res = _arima_model(y[:R + 1], X, p, d, q).fit(disp=False, maxiter=200)
    return {"params": np.asarray(res.params), "p": p, "d": d, "q": q}


def arima_forecast(st, y, dow_fn, o):
    X = weekday_dummies(range(o + 1), dow_fn)
    Xf = weekday_dummies(range(o + 1, o + 1 + H_MAX), dow_fn)
    res = _arima_model(y[:o + 1], X, st["p"], st["d"], st["q"]).filter(st["params"])
    return np.asarray(res.forecast(H_MAX, exog=Xf))


# ── SARIMA(p,d,q)(P,D,Q)_7 ────────────────────────────────────────────────────
def _sarima_model(y, pr):
    trend = "c" if (pr["d"] == 0 and pr["D"] == 0) else "n"
    return SARIMAX(y, order=(pr["p"], pr["d"], pr["q"]),
                   seasonal_order=(pr["P"], pr["D"], pr["Q"], 7), trend=trend)


def sarima_fit(y, dow_fn, R, params):
    res = _sarima_model(y[:R + 1], params).fit(disp=False, maxiter=200)
    return {"params": np.asarray(res.params), "spec": params}


def sarima_forecast(st, y, dow_fn, o):
    res = _sarima_model(y[:o + 1], st["spec"]).filter(st["params"])
    return np.asarray(res.forecast(H_MAX))


# ── ETS (error A; trend N/A/Ad; seasonal N/A, m = 7) ──────────────────────────
ETS_CONFIGS = [
    {"trend": None, "damped": False, "seasonal": None},
    {"trend": "add", "damped": False, "seasonal": None},
    {"trend": "add", "damped": True, "seasonal": None},
    {"trend": None, "damped": False, "seasonal": "add"},
    {"trend": "add", "damped": False, "seasonal": "add"},
    {"trend": "add", "damped": True, "seasonal": "add"},
]


def _ets_model(y, cfg):
    return ETSModel(y, error="add", trend=cfg["trend"], damped_trend=cfg["damped"],
                    seasonal=cfg["seasonal"], seasonal_periods=7 if cfg["seasonal"] else None)


def ets_fit(y, dow_fn, R, params):
    cfg = ETS_CONFIGS[params["config"]]
    res = _ets_model(y[:R + 1], cfg).fit(disp=False, maxiter=500)
    return {"params": np.asarray(res.params), "cfg": cfg}


def ets_forecast(st, y, dow_fn, o):
    res = _ets_model(y[:o + 1], st["cfg"]).smooth(st["params"])
    return np.asarray(res.forecast(H_MAX))


# ── EMA of the weekday-adjusted series, re-seasonalised ───────────────────────
def weekday_offsets(y_train, dow_fn):
    """Classical decomposition: weekday means of y minus its centred 7-day moving average, centred to sum 0."""
    s = pd.Series(y_train)
    detr = s - s.rolling(7, center=True).mean()
    d = np.array([dow_fn(t) for t in range(len(y_train))])
    off = pd.Series(detr.values).groupby(d).mean().reindex(range(7)).fillna(0.0).values
    return off - off.mean()


def ema_fit(y, dow_fn, R, params):
    return {"span": int(params["span"]), "off": weekday_offsets(y[:R + 1], dow_fn)}


def ema_forecast(st, y, dow_fn, o):
    d = np.array([dow_fn(t) for t in range(o + 1)])
    adj = y[:o + 1] - st["off"][d]
    level = pd.Series(adj).ewm(span=st["span"], adjust=False).mean().values[-1]
    return np.array([level + st["off"][dow_fn(o + h)] for h in range(1, H_MAX + 1)])


# ── Benchmarks ────────────────────────────────────────────────────────────────
def naive_fit(y, dow_fn, R, params):
    return {}


def naive_forecast(st, y, dow_fn, o):
    return np.full(H_MAX, y[o])


def naivedow_fit(y, dow_fn, R, params):
    dy = np.diff(y[:R + 1])
    d = np.array([dow_fn(t) for t in range(1, R + 1)])
    m = pd.Series(dy).groupby(d).mean().reindex(range(7)).fillna(0.0).values
    return {"m": m}


def naivedow_forecast(st, y, dow_fn, o):
    return y[o] + np.cumsum([st["m"][dow_fn(o + j)] for j in range(1, H_MAX + 1)])


def seasonalnaive_fit(y, dow_fn, R, params):
    return {}


def seasonalnaive_forecast(st, y, dow_fn, o):
    return np.array([y[o + h - 7 * int(np.ceil(h / 7))] for h in range(1, H_MAX + 1)])


def _ar_design(x, p, pmax):
    n = len(x)
    X = np.column_stack([np.ones(n - pmax)] + [x[pmax - j - 1:n - j - 1] for j in range(p)])
    return X, x[pmax:]


def ar_fit(y, dow_fn, R, params, pmax=21):
    x = y[:R + 1]
    best = (np.inf, None, None)
    for p in range(1, pmax + 1):
        X, t = _ar_design(x, p, pmax)          # common estimation sample for AIC comparability
        beta = np.linalg.lstsq(X, t, rcond=None)[0]
        aic = len(t) * np.log(np.mean((t - X @ beta) ** 2)) + 2 * (p + 1)
        if aic < best[0]:
            best = (aic, p, None)
    p = best[1]
    X, t = _ar_design(x, p, p)                  # re-estimate on the full sample at the chosen order
    return {"p": p, "beta": np.linalg.lstsq(X, t, rcond=None)[0]}


def ar_forecast(st, y, dow_fn, o):
    p, b = st["p"], st["beta"]
    buf = list(y[o - p + 1:o + 1])
    out = []
    for _ in range(H_MAX):
        nxt = b[0] + sum(b[1 + j] * buf[-1 - j] for j in range(p))
        out.append(nxt)
        buf.append(nxt)
    return np.array(out)


MODELS = {
    "ARIMA": (arima_fit, arima_forecast),
    "SARIMA": (sarima_fit, sarima_forecast),
    "ETS": (ets_fit, ets_forecast),
    "EMA": (ema_fit, ema_forecast),
    "Naive": (naive_fit, naive_forecast),
    "NaiveDOW": (naivedow_fit, naivedow_forecast),
    "SeasonalNaive": (seasonalnaive_fit, seasonalnaive_forecast),
    "AR": (ar_fit, ar_forecast),
}
