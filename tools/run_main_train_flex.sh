#!/bin/bash
# 弾性 NSS の本学習: {A,B} x 幅 {64,128} x シード {0,1,2}、20 エポック。直列で学習 → 評価 → (c3)。
cd "$(dirname "$0")/.."
for w in 64 128; do
  for s in 0 1 2; do
    for k in A B; do
      r=${k}_w${w}_s${s}
      [ -f data/train_nss8kflex/$r/model.pt ] || .venv/bin/python python/train_nss_flex.py --model $k --width $w --epochs 20 --seed $s --run $r
      [ -f reports/issue-37/$r/metrics.json ] || .venv/bin/python python/evaluate_nss_flex.py --run $r --report-dir reports/issue-37/$r
      [ -f reports/issue-37/c3_$r.json ] || .venv/bin/python python/evaluate_c3.py --runs $r --out reports/issue-37
    done
  done
done
echo ALLDONE
