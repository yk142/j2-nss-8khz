#!/usr/bin/env python3
"""(c) PTP 閉ループの誤差（モデル − 真値の θ [°]）の時系列。評価ライブラリの閉ループを再実行して描く。

    .venv/bin/python python/plot_nss_acc_closed.py --run acc_w64_s0 --out reports/issue-45/plots/closed_loop_err_s0.png
"""
import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nss8k import evaluate_lib as ev  # noqa: E402
from nss8k.data import load  # noqa: E402
from nss8k.physics import Physics  # noqa: E402
from nss8kacc.model import build_model  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "TakaoPGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--out", required=True)
p.add_argument("--data", default="data/export/j2_nss8k.h5")
p.add_argument("--out-root", default="data/train_nss8kacc")
a = p.parse_args()
torch.set_num_threads(4)
ck = torch.load(os.path.join(a.out_root, a.run, "model.pt"), map_location="cpu")
model = build_model(ck["config"]); model.load_state_dict(ck["state_dict"]); model.eval(); model.reset()
d = load(a.data, splits=("benchmark",))
phys = Physics(d["physics"], d["dt"])
rows, seqs, res = ev.closed_loop_benchmark(model, phys, d)
n = len(seqs)
fig, axs = plt.subplots(int(np.ceil(n / 3)), 3, figsize=(13, 2.6 * int(np.ceil(n / 3))), squeeze=False)
for i, (s, ax) in enumerate(zip(seqs, axs.ravel())):
    K = len(s["tau"]); t = np.arange(K) / 8000.0
    for name, col, lab in (("model", "tab:red", "学習モデル"), ("analytic", "tab:blue", "解析モデル")):
        ax.plot(t, np.degrees(res[name][0][i, :K] - s["q"]), color=col, lw=0.8, label=lab)
    ax.axhline(0.1, color="g", ls="--", lw=0.6); ax.axhline(-0.1, color="g", ls="--", lw=0.6)
    ax.set_title(f"{s['name']}  RMSE {rows[i]['model']['vs_truth_rmse_q_deg']:.3f}°", fontsize=8)
    ax.grid(alpha=0.3)
axs[0, 0].legend(fontsize=7)
fig.suptitle(f"{a.run}: モデル − 真値の θ [°]（緑の破線 = ±0.1°、基準）", fontsize=10)
fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=100)
