#!/bin/bash
# 学習済みの run を順に評価する。使い方: tools/nss8k_eval_all.sh <報告先ディレクトリ> <run>...
cd "$(dirname "$0")/.." || exit 1
OUT=$1; shift
for run in "$@"; do
  echo "[$(date +%T)] 評価開始 $run"
  .venv/bin/python python/evaluate_nss.py --run "$run" --report-dir "$OUT/$run" > "data/logs/eval_$run.log" 2>&1 || echo "失敗 $run"
  grep -E "^ゲート" "data/logs/eval_$run.log" | cut -c1-110
done
echo "[$(date +%T)] すべて終了"
