"""Alignment tests for the pipeline. Run: python3 tests/test_alignment.py"""
import sys
sys.path.insert(0, ".")
import numpy as np
from common import load_series, LOOKBACK, dow_onehot, refit_for_origin, REFITS
import nn_models as nn

_, y, dow = load_series()
dow_fn = lambda t: (int(dow[0]) + t) % 7

def test_nn_window_and_target():
    st = nn.scaler_stats(y, 2000, 7)
    o = np.array([1500, 2100, 3000])
    Xs, Xc, t = nn.make_xy(y, dow_fn, o, 7, st)
    assert Xs.shape == (3, LOOKBACK, 1)
    assert np.allclose(Xs[:, -1, 0] * st["sd"] + st["mu"], y[o], atol=1e-3), "window must end at the origin"
    assert np.allclose(Xs[:, 0, 0] * st["sd"] + st["mu"], y[o - 89], atol=1e-3)
    assert np.allclose(t * st["sd_h"], y[o + 7] - y[o], atol=1e-3), "target must be the h-step increment"
    assert (Xc.argmax(1) == dow[o + 7]).all(), "calendar must describe the target date"

def test_scaler_uses_training_only():
    a = nn.scaler_stats(y, 2000, 1); b = nn.scaler_stats(np.r_[y[:2001], y[2001:] + 1000], 2000, 1)
    assert a == b

def test_refit_mapping():
    assert refit_for_origin(1368) == 1367 and refit_for_origin(1457) == 1367 and refit_for_origin(1458) == 1458
    assert all(refit_for_origin(R) == R for R in REFITS)



def test_stacking_uses_target_date_forecasts():
    """A linear meta-learner fed only SARIMA's target-date forecast must match SARIMA's accuracy
    (warm-up targets only)."""
    from sklearn.linear_model import LinearRegression
    from common import OUT, all_origins
    O = all_origins()
    F = np.load(OUT / "forecasts" / "SARIMA.npz")["F"][:, 0]
    tgt = O + 1
    tr = (tgt >= 1400) & (tgt <= 1900)
    te = (tgt > 1900) & (tgt <= 2282)
    X = (F - y[O])[:, None]
    m = LinearRegression().fit(X[tr], (y[tgt] - y[O])[tr])
    rmse_stack = np.sqrt(np.mean((y[tgt][te] - (y[O][te] + m.predict(X[te]))) ** 2))
    rmse_sarima = np.sqrt(np.mean((y[tgt][te] - F[te]) ** 2))
    assert abs(rmse_stack / rmse_sarima - 1) < 0.02, (rmse_stack, rmse_sarima)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
