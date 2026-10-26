#!/usr/bin/env bash
# アブレーション: 位置の入力を sin q, cos q に（幅 64、seed 0、20 エポック）。幅 256 の run の終了を待ってから走る。issue #45
cd "$(dirname "$0")/.." || exit 1
until grep -q W256DONE data/logs/acc_w256.log 2>/dev/null; do sleep 20; done
run=acc_sincos_w64_s0
.venv/bin/python python/train_nss_acc.py --width 64 --epochs 20 --seed 0 --feat sincos --run $run > data/logs/train_$run.log 2>&1
.venv/bin/python python/evaluate_nss_acc.py --run $run --report-dir reports/issue-45/$run > data/logs/eval_$run.log 2>&1
echo SINCOSDONE
