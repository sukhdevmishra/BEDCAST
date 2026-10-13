#!/usr/bin/env bash
# Launch one process per (architecture, horizon), alternating between the two GPUs.
cd "$(dirname "$0")"
i=0
for a in LSTM CNN-LSTM MLP TDNN CNN; do
  for h in 1 7 14 28; do
    g=$((i % 2))
    CUDA_VISIBLE_DEVICES=$g nohup python3 run_nn.py both "$a" --horizons "$h" > "../output/v2/logs/nn_${a}_h${h}.log" 2>&1 &
    i=$((i + 1))
  done
done
