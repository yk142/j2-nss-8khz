#!/usr/bin/env bash
# 加速度を状態に持つ残差型 NSS の本学習（幅 64、3 シード × 20 エポック）と評価。issue #45
cd "$(dirname "$0")/.." || exit 1
for s in 0 1 2; do
  run=acc_w64_s$s
  .venv/bin/python python/train_nss_acc.py --width 64 --epochs 20 --seed $s --run $run > data/logs/train_$run.log 2>&1
  .venv/bin/python python/evaluate_nss_acc.py --run $run --report-dir reports/issue-45/$run > data/logs/eval_$run.log 2>&1
done
echo ACCDONE
