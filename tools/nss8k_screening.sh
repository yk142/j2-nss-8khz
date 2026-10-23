#!/bin/bash
# スクリーニング: {A, B} × 幅 {64, 128, 256} を 5 エポックで直列に学習する（seed 0）。
# 使い方: nohup tools/nss8k_screening.sh > data/logs/screening.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
for w in 64 128 256; do
  for m in A B; do
    run="${m}_w${w}_screen"
    if [ -f "data/train_nss8k/$run/config.json" ]; then echo "skip $run"; continue; fi
    echo "[$(date +%T)] 学習開始 $run"
    .venv/bin/python python/train_nss.py --model $m --width $w --epochs 5 --seed 0 --run "$run" > "data/logs/train_$run.log" 2>&1 || echo "失敗 $run"
    tail -1 "data/logs/train_$run.log"
  done
done
echo "[$(date +%T)] すべて終了"
