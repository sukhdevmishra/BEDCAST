"""
Shared configuration and helpers for the BEDCAST v2 pipeline.
Every constant here is fixed by the experimental plan.
"""
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "bed_occupancy1.csv"
DATA_SHA256 = "85af142497cbb0d0fc0c3996810dd0aa14c9f732719fa91122a3564a6c0bf902"
OUT = ROOT / "output" / "v2"

SEED = 42
LOOKBACK = 90
HORIZONS = [1, 7, 14, 28]
H_MAX = max(HORIZONS)

DEV_END = 1367            # last development/tuning target index (2014-09-29)
WARMUP_START = 1368       # first origin whose forecasts are genuinely out-of-sample
EVAL_START = 2283         # first evaluation target (2017-04-02)
EVAL_END = 3370           # last evaluation target (2020-03-24)
CONF_START = 3198         # first confirmatory target (2019-10-04)
LAST_INDEX = 3377         # 2020-03-31
REFIT_EVERY = 91
REFITS = list(range(DEV_END, LAST_INDEX, REFIT_EVERY))   # R_k = 1367 + 91k
META_FIRST_REFIT = DEV_END + 4 * REFIT_EVERY               # 1731

# Exceedance event (adopted after the results were examined; see the paper): occupancy above 85% of the sanctioned bed capacity.
BED_CAPACITY = 643        # sanctioned beds throughout 2011-2020 (institute annual reports)
EXCEED_RATE = 0.85        # threshold = 546.55 beds, i.e. the event is occupancy >= 547

# Tuning folds inside the development block: (train_end, val_start, val_end) as target indices
TUNE_FOLDS = [(1001, 1002, 1184), (1184, 1185, 1367)]
N_TUNE = 15

BASE_STAT = ["ARIMA", "SARIMA", "ETS", "EMA"]
BASE_NN = ["MLP", "CNN", "LSTM", "CNN-LSTM", "TDNN"]
BASE_ALL = ["ARIMA", "SARIMA", "EMA", "ETS"] + BASE_NN
# AR (levels) is estimated but excluded from evaluation and reporting.
BENCHMARKS = ["Naive", "NaiveDOW", "SeasonalNaive", "Comb"]
META = ["ElasticNet", "DecisionTree", "RandomForest", "LightGBM"]


def load_series():
    raw = DATA_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DATA_SHA256, "data file differs from the recorded checksum"
    df = pd.read_csv(DATA_PATH)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d-%b-%y")
    df = df.sort_values("DATE").reset_index(drop=True)
    assert len(df) == LAST_INDEX + 1 and df["BED_OCCUPANCY"].notna().all()
    assert (df["DATE"].diff().dropna() == pd.Timedelta(days=1)).all()
    y = df["BED_OCCUPANCY"].astype(float).values
    dow = df["DATE"].dt.dayofweek.values          # Monday = 0
    return df["DATE"], y, dow


def dow_onehot(dow_idx):
    out = np.zeros((len(dow_idx), 7))
    out[np.arange(len(dow_idx)), dow_idx] = 1.0
    return out


def refit_for_origin(o):
    """The model re-estimated at the latest R_k <= o issues the forecast from origin o."""
    k = (o - DEV_END) // REFIT_EVERY
    return DEV_END + k * REFIT_EVERY


def all_origins():
    return np.arange(WARMUP_START, LAST_INDEX)  # origins 1368 .. 3376


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


class Timer:
    """Append wall-clock timings to output/v2/timing.csv."""
    path = OUT / "timing.csv"

    @classmethod
    @contextmanager
    def log(cls, model, horizon, refit, stage):
        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
        ensure_dir(OUT)
        new = not cls.path.exists()
        with open(cls.path, "a") as fh:
            if new:
                fh.write("model,horizon,refit,stage,seconds\n")
            fh.write(f"{model},{horizon},{refit},{stage},{dt:.4f}\n")


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v))


def load_json(path):
    with open(path) as fh:
        return json.load(fh)
