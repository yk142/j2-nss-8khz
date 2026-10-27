#!/usr/bin/env bash
# acc ノイズ学習（幅 64、seed 0、20 エポック）。σ は引数（rad/s²）。issue #46
cd "$(dirname "$0")/.." || exit 1
for sg in "$@"; do
  run=acc_noise${sg}_w64_s0
  .venv/bin/python python/train_nss_acc.py --width 64 --epochs 20 --seed 0 --acc-noise $sg --run $run > data/logs/train_$run.log 2>&1
  .venv/bin/python python/evaluate_nss_acc.py --run $run --report-dir reports/issue-46/$run > data/logs/eval_$run.log 2>&1
done
echo NOISEDONE $*
