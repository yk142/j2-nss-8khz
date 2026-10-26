#!/usr/bin/env bash
# アブレーション: 幅 256（seed 0、20 エポック）。issue #45
cd "$(dirname "$0")/.." || exit 1
run=acc_w256_s0
.venv/bin/python python/train_nss_acc.py --width 256 --epochs 20 --seed 0 --run $run > data/logs/train_$run.log 2>&1
.venv/bin/python python/evaluate_nss_acc.py --run $run --report-dir reports/issue-45/$run > data/logs/eval_$run.log 2>&1
echo W256DONE
