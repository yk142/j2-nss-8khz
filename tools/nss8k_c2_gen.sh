#!/bin/bash
# (c2) の真値を 2 ワーカーで生成する。使い方: nohup tools/nss8k_c2_gen.sh > data/logs/c2_gen.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
for i in 1 2; do
  matlab -batch "addpath('tools'); j2_gen_c2($i,2)" > "data/logs/c2_gen_$i.log" 2>&1 &
done
wait
echo "[$(date +%T)] すべて終了"
