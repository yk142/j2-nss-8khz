#!/bin/bash
# 本学習: B w64 / B w128 / A w256 × シード {0,1,2} を 20 エポックで直列に学習し、学習後に評価する。
# 使い方: nohup tools/nss8k_main_train.sh [報告先] > data/logs/main_train.log 2>&1 &
# 学習済み（config.json あり）は学習をスキップ、評価済み（metrics.json あり）は評価をスキップする（中断後の再開用）。
cd "$(dirname "$0")/.." || exit 1
OUT=${1:-reports/issue-16}
for cfg in "B 64" "B 128" "A 256"; do
  set -- $cfg; m=$1; w=$2
  for s in 0 1 2; do
    run="${m}_w${w}_s${s}"
    if [ -f "data/train_nss8k/$run/config.json" ]; then echo "skip 学習 $run"; else
      echo "[$(date +%T)] 学習開始 $run"
      .venv/bin/python python/train_nss.py --model $m --width $w --epochs 20 --seed $s --run "$run" > "data/logs/train_$run.log" 2>&1 || echo "失敗 学習 $run"
      tail -1 "data/logs/train_$run.log"
    fi
    if [ -f "$OUT/$run/metrics.json" ]; then echo "skip 評価 $run"; else
      echo "[$(date +%T)] 評価開始 $run"
      .venv/bin/python python/evaluate_nss.py --run "$run" --report-dir "$OUT/$run" > "data/logs/eval_$run.log" 2>&1 || echo "失敗 評価 $run"
      grep -E "^ゲート" "data/logs/eval_$run.log" | cut -c1-110
    fi
  done
done
echo "[$(date +%T)] すべて終了"
