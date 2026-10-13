"""
Neural base learners. Each network takes a 90-day window, standardised with training
statistics, plus a one-hot day-of-week for the TARGET date, and predicts the standardised h-step
increment (y[o+h] - y[o]) / sd_h. Early stopping uses the last 10% of training samples in time order.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import keras
from keras import layers

from common import LOOKBACK, dow_onehot

_DET = False


def set_determinism():
    global _DET
    if not _DET:
        import tensorflow as tf
        for g in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError:
                pass
        tf.config.experimental.enable_op_determinism()
        _DET = True


def scaler_stats(y, R, h):
    """Standardisation constants estimated on y[0..R] only."""
    tr = y[:R + 1]
    inc = tr[h:] - tr[:-h]
    return {"mu": float(tr.mean()), "sd": float(tr.std()), "sd_h": float(inc.std())}


def make_xy(y, dow_fn, origins, h, stats, with_target=True):
    """Windows ending at each origin o: y[o-89..o]; calendar = day of week of target o+h."""
    origins = np.asarray(origins)
    idx = origins[:, None] + np.arange(-LOOKBACK + 1, 1)[None, :]
    Xs = ((y[idx] - stats["mu"]) / stats["sd"])[..., None].astype("float32")
    Xc = dow_onehot(np.array([dow_fn(o + h) for o in origins])).astype("float32")
    if not with_target:
        return Xs, Xc
    t = ((y[origins + h] - y[origins]) / stats["sd_h"]).astype("float32")
    return Xs, Xc, t


def build(arch, units, n_layers, dropout, lr):
    seq = keras.Input((LOOKBACK, 1), name="window")
    cal = keras.Input((7,), name="dow")
    if arch == "MLP":
        x = layers.Flatten()(seq)
        x = layers.Concatenate()([x, cal])
        for _ in range(n_layers):
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(dropout)(x)
    elif arch == "CNN":
        x = layers.Conv1D(units, 3, activation="relu")(seq)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Concatenate()([x, cal])
        for _ in range(n_layers):
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(dropout)(x)
    elif arch == "TDNN":
        x = layers.Conv1D(units, 3, dilation_rate=2, activation="relu")(seq)
        x = layers.Flatten()(x)
        x = layers.Concatenate()([x, cal])
        for _ in range(n_layers):
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(dropout)(x)
    elif arch == "LSTM":
        x = seq
        for i in range(n_layers):
            x = layers.LSTM(units, return_sequences=(i < n_layers - 1))(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Concatenate()([x, cal])
        x = layers.Dense(units, activation="relu")(x)
    elif arch == "CNN-LSTM":
        x = layers.Conv1D(units, 3, activation="relu")(seq)
        x = layers.MaxPooling1D(2)(x)
        for i in range(n_layers):
            x = layers.LSTM(units, return_sequences=(i < n_layers - 1))(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Concatenate()([x, cal])
        x = layers.Dense(units, activation="relu")(x)
    else:
        raise ValueError(arch)
    out = layers.Dense(1)(x)
    m = keras.Model([seq, cal], out)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return m


def fit_predict(arch, hp, y, dow_fn, R, h, pred_origins, seed, save_path=None):
    """Train on origins o with o >= 89 and o + h <= R; return level forecasts for pred_origins."""
    set_determinism()
    keras.utils.set_random_seed(seed)
    stats = scaler_stats(y, R, h)
    tr_orig = np.arange(LOOKBACK - 1, R - h + 1)
    Xs, Xc, t = make_xy(y, dow_fn, tr_orig, h, stats)
    m = build(arch, hp["units"], hp["layers"], hp["dropout"], hp["lr"])
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    hist = m.fit([Xs, Xc], t, epochs=200, batch_size=hp["batch"], validation_split=0.1,
                 shuffle=True, callbacks=[es], verbose=0)
    Ps, Pc = make_xy(y, dow_fn, pred_origins, h, stats, with_target=False)
    inc = m.predict([Ps, Pc], batch_size=1024, verbose=0).ravel() * stats["sd_h"]
    if save_path is not None:
        m.save(save_path)
        np.save(str(save_path).replace(".keras", "_stats.npy"), stats)
    keras.backend.clear_session()
    return y[np.asarray(pred_origins)] + inc, len(hist.history["loss"])
