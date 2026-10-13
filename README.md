# BEDCAST: a diagnostic stacking ensemble for hospital bed occupancy forecasting

Code, aggregate results and online supplementary material for the paper
"BEDCAST: A Diagnostic Stacking Ensemble for Hospital Bed Occupancy Forecasting with Lag-Level Explainability"
(Das, Chaudhary, Modi, Pachori and Mishra). [CITATION AND DOI TO BE ADDED ON PUBLICATION]

## Contents

| Path | What it holds |
|---|---|
| `ONLINE_SUPPLEMENT.md` | Figures and tables referred to in the paper and its supplement as being in the online repository |
| `pipeline/` | The analysis code, from model tuning to every reported table and figure |
| `output/v2/results/` | Aggregate results behind the reported tables and figures |
| `figures/` | Images used by `ONLINE_SUPPLEMENT.md` |
| `data/` | Placeholder: the occupancy data are available on request (see `data/README.md`) |

No file in this repository contains daily occupancy values or forecasts.

## Running the analysis

With the data file in `data/` and the packages in `requirements.txt` installed, run from `pipeline/`:

```bash
python3 run_stat.py tune && python3 run_stat.py roll        # statistical learners and benchmarks (CPU)
./launch_nn.sh                                              # 20 processes: 5 networks x 4 horizons, 2 GPUs
python3 run_nn.py merge MLP CNN LSTM CNN-LSTM TDNN
python3 run_meta.py                                         # stacking: Variants A and B, four meta-learners
python3 tests/test_alignment.py                             # must pass before evaluation
python3 evaluate.py && python3 attribution.py               # inference and lag-level attribution
python3 run_stat.py roll --start-index 425 --tag _S1        # sensitivity analysis (estimation from 1 March 2012)
python3 cadence.py && python3 cadence.py --nn               # re-estimation frequency
python3 secondary.py && python3 make_outputs.py --design && python3 make_outputs.py
python3 exceedance_capacity.py && python3 lag_correlation.py
python3 gantt.py && python3 accuracy_cost.py                # training schedule; accuracy against training compute
python3 inference_timing.py measure && python3 inference_timing.py plot
```

Forecasts, fitted models, figures and LaTeX tables are written under `output/v2/`. Neural training is
stochastic on GPUs; the statistical learners, benchmarks and meta-learners are deterministic (seed 42).

Notes:
- The scripts also fit an autoregressive benchmark on levels; it lies within the ARIMA search space and is not
  reported in the paper.
- Inference latency depends on the machine; the published latency figure was measured on the machine that
  trained the models. [AUTHORS TO ADD: `output/v2/results/inference_latency.csv` from that machine.]

## License

The software code in this repository is licensed under the **MIT License**. You are free to use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software, subject to the terms of the licence.

See the [`LICENSE`](LICENSE) file for the full licence text.

The hospital bed occupancy data used in this study are not included in this repository. Data access conditions are described in [`data/README.md`](data/README.md).

The licence for the software does not automatically extend to third-party materials or datasets referenced in this repository.