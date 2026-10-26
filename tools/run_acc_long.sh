#!/usr/bin/env bash
# エポックを増やした場合の確認（幅 64、seed 0、60 エポック、patience 15）。issue #45
cd "$(dirname "$0")/.." || exit 1
run=acc_w64_s0_e60
.venv/bin/python python/train_nss_acc.py --width 64 --epochs 60 --patience 15 --seed 0 --run $run > data/logs/train_$run.log 2>&1
.venv/bin/python python/evaluate_nss_acc.py --run $run --report-dir reports/issue-45/$run > data/logs/eval_$run.log 2>&1
echo LONGDONE
