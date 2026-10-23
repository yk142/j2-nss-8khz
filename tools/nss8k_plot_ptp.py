"""採用モデル（B w64 / A w256）の PTP 閉ループ時系列。使い方: .venv/bin/python tools/nss8k_plot_ptp.py <出力.png>"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from nss8k.data import load, sequences
from nss8k.physics import Physics
from nss8k import evaluate_lib as ev
from evaluate_nss import load_model
for f in ("Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "TakaoGothic", "DejaVu Sans"):
    from matplotlib import font_manager as fm
    if any(f in x.name for x in fm.fontManager.ttflist): plt.rcParams["font.family"] = f; break
d = load("data/export/j2_nss8k.h5", splits=("benchmark",)); phys = Physics(d["physics"], d["dt"])
seqs = [s for s in sequences(d, "benchmark") if s["name"] in ("ptp_v40_003", "ptp_v80_003")]
models = {"B (w64)": load_model("models/nss8k/B_w64_s2/model.pt")[0], "A (w256)": load_model("models/nss8k/A_w256_s2/model.pt")[0]}
fns = {k: m.step_physical for k, m in models.items()}; fns["解析モデル"] = phys.step
res = {k: ev.closed_loop_batch(fn, seqs, d["physics"], d["dt"]) for k, fn in fns.items()}
col = {"B (w64)": "tab:blue", "A (w256)": "tab:red", "解析モデル": "tab:green"}
fig, axs = plt.subplots(3, 2, figsize=(14, 9), sharex="col")
for j, s in enumerate(seqs):
    K = len(s["tau"]); t = np.arange(K) * d["dt"]
    axs[0, j].plot(t, np.degrees(s["qref"][:K]), "k--", lw=0.8, label="参照"); axs[0, j].plot(t, np.degrees(s["q"][:K]), "k", lw=1.2, label="真値 (Simscape)")
    for k, (Q, U) in res.items():
        axs[0, j].plot(t, np.degrees(Q[j, :K]), color=col[k], lw=0.8, label=k)
        axs[1, j].plot(t, np.degrees(Q[j, :K] - s["q"][:K]), color=col[k], lw=0.9, label=k)
        axs[2, j].plot(t, U[j, :K] - s["tau"], color=col[k], lw=0.7, label=k)
    rm = {k: np.degrees(np.sqrt(np.nanmean((Q[j, :K] - s["q"][:K]) ** 2))) for k, (Q, U) in res.items()}
    axs[0, j].set_title(f"{s['name']}（閉ループ PD+FF, wn=30, ζ=1）  真値との θ RMSE: " + "  ".join(f"{k} {v:.4f}°" for k, v in rm.items()), fontsize=8)
    axs[0, j].set_ylabel("θ [deg]"); axs[1, j].set_ylabel("θ − 真値 [deg]"); axs[2, j].set_ylabel("τ − 真値 [N·m]"); axs[2, j].set_xlabel("t [s]")
    for a in axs[:, j]: a.grid(alpha=0.3)
axs[0, 0].legend(fontsize=7); axs[1, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=100)
