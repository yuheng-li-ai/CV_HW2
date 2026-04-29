#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/2] CNN + Momentum improved"
python -u train_opt_momentum_cnn.py | tee opt_momentum_cnn_improved.log

echo "[2/2] CNN + Early stopping improved"
python -u train_earlystop_cnn.py | tee earlystop_cnn_improved.log

echo "finished cnn improved experiments"
