#!/usr/bin/env bash
# acc ノイズ学習 σ=1.0 のシード違い（幅 64、20 エポック）。issue #46。引数: シード番号
cd "$(dirname "$0")/.." || exit 1
sg=1.0
for s in "$@"; do
  run=acc_noise${sg}_w64_s${s}
  .venv/bin/python python/train_nss_acc.py --width 64 --epochs 20 --seed $s --acc-noise $sg --run $run > data/logs/train_$run.log 2>&1
  .venv/bin/python python/evaluate_nss_acc.py --run $run --report-dir reports/issue-46/$run > data/logs/eval_$run.log 2>&1
done
echo SEEDSDONE $*
