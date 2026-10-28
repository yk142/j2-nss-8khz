#!/usr/bin/env bash
# ロールアウト損失（幅 64、seed 0、20 エポック）。issue #49。引数: 「名前:モード:K:窓数:σ[:シード]」を並べる
cd "$(dirname "$0")/.." || exit 1
for spec in "$@"; do
  IFS=: read -r name mode k rb sg seed <<< "$spec"; seed=${seed:-0}
  run=acc_${name}_w64_s${seed}
  .venv/bin/python python/train_nss_acc.py --width 64 --epochs 20 --seed $seed --rollout-mode $mode --rollout-k $k --rollout-batch $rb --acc-noise $sg --run $run > data/logs/train_$run.log 2>&1
  .venv/bin/python python/evaluate_nss_acc.py --run $run --report-dir reports/issue-49/$run > data/logs/eval_$run.log 2>&1
done
echo ROLLDONE $*
