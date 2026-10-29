#!/usr/bin/env bash
# 弾性の加速度状態 NSS（幅 64、20 エポック）。issue #51。引数: 「名前:ノイズ倍率:ロールアウトK:シード」を並べる
cd "$(dirname "$0")/.." || exit 1
for spec in "$@"; do
  IFS=: read -r name rel k seed <<< "$spec"; seed=${seed:-0}
  run=accflex_${name}_w64_s${seed}
  .venv/bin/python python/train_nss_accflex.py --width 64 --epochs 20 --seed $seed --acc-noise-rel $rel --rollout-k $k --run $run > data/logs/train_$run.log 2>&1
  .venv/bin/python python/evaluate_nss_accflex.py --run $run --report-dir reports/issue-51/$run > data/logs/eval_$run.log 2>&1
done
echo FLEXDONE $*
