#!/usr/bin/env python3
"""4 状態 NSS の時系列プロット（真値 / 学習モデル / 解析モデル）。docs/flex_nss_results.md 用。

    .venv/bin/python python/plot_nss_flex_timeseries.py --run B_w128_screen [--run-a A_w128_screen] --out-dir reports/issue-33/plots
出力: closed_loop_<名前>.png（PTP 閉ループ）、window_<名前>.png（窓の開ループ）、ddqm_floor.png（ddqm の下限の説明）
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
from nss8kflex import evaluate_lib as ev  # noqa: E402
from nss8kflex.data import load, sequences  # noqa: E402
from nss8kflex.model import build_model  # noqa: E402
from nss8kflex.physics import FlexPhysics  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "TakaoPGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
STY = {"truth": ("k", "-", 1.4), "B": ("tab:red", "-", 1.0), "A": ("tab:green", "-", 1.0), "analytic": ("tab:blue", "--", 1.0)}
LAB = {"truth": "真値（Simscape）", "B": "残差型 B", "A": "純 NN A", "analytic": "解析モデル"}


def load_model(root, run):
    ck = torch.load(os.path.join(root, run, "model.pt"), map_location="cpu")
    m = build_model(ck["config"]); m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def closed_loop_plot(name, seqs, rows, res_by_method, out, t_zoom=None):
    i = [s["name"] for s in seqs].index(name)
    s = seqs[i]; K = len(s["tau"]); t = np.arange(K) / 8000.0
    fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axs[0].plot(t, np.degrees(s["qref"]), "g--", lw=0.8, label="参照")
    axs[0].plot(t, np.degrees(s["q"]), color="k", lw=1.4, label=LAB["truth"])
    for k, (Q, QM, U) in res_by_method.items():
        c, ls, lw = STY[k]
        axs[0].plot(t, np.degrees(Q[i, :K]), color=c, ls=ls, lw=lw, label=LAB[k])
        axs[1].plot(t, np.degrees(Q[i, :K] - s["q"]), color=c, ls=ls, lw=lw, label=LAB[k])
        axs[2].plot(t, (QM[i, :K] - Q[i, :K]) * 1e3, color=c, ls=ls, lw=lw, label=LAB[k])
        axs[3].plot(t, U[i, :K], color=c, ls=ls, lw=lw, label=LAB[k])
    axs[2].plot(t, (s["qm"] - s["q"]) * 1e3, color="k", lw=1.4, label=LAB["truth"])
    axs[3].plot(t, s["tau"], color="k", lw=1.4, label=LAB["truth"])
    for ax, yl in zip(axs, ("リンク角 θ [度]", "θ − θ真値 [度]", "ねじり qm−θ [mrad]", "トルク [N·m]")):
        ax.set_ylabel(yl); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8, ncol=5); axs[-1].set_xlabel("時間 [s]")
    if t_zoom:
        for ax in axs:
            ax.set_xlim(*t_zoom)
    axs[0].set_title(f"{name}（PTP、モータ角 PD＋FF の閉ループ、8 kHz）")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def window_plot(seq, a0, n, models, phys, title, out):
    sl = slice(a0, a0 + n + 1)
    s0 = tuple(torch.tensor([seq[k][a0]], dtype=torch.float64) for k in ("q", "dq", "qm", "dqm"))
    tau = torch.from_numpy(seq["tau"][a0:a0 + n])[None]
    t = np.arange(n + 1) / 8000.0
    fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axs[0].plot(t, np.degrees(seq["q"][sl]), color="k", lw=1.4, label=LAB["truth"])
    axs[2].plot(t, (seq["qm"][sl] - seq["q"][sl]) * 1e3, color="k", lw=1.4, label=LAB["truth"])
    axs[3].plot(t[:-1], seq["tau"][a0:a0 + n], color="k", lw=1.0)
    fns = {"analytic": phys.step}
    fns.update({k: m.step_physical for k, m in models.items()})
    for k, fn in fns.items():
        q, _, qm, _ = (x.numpy()[0] for x in ev.rollout(fn, s0, tau))
        c, ls, lw = STY[k]
        axs[0].plot(t, np.degrees(q), color=c, ls=ls, lw=lw, label=LAB[k])
        axs[1].plot(t, np.degrees(q - seq["q"][sl]), color=c, ls=ls, lw=lw, label=LAB[k])
        axs[2].plot(t, (qm - q) * 1e3, color=c, ls=ls, lw=lw, label=LAB[k])
    for ax, yl in zip(axs, ("リンク角 θ [度]", "θ − θ真値 [度]", "ねじり qm−θ [mrad]", "トルク（再生）[N·m]")):
        ax.set_ylabel(yl); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8, ncol=4); axs[-1].set_xlabel("窓の先頭からの時間 [s]")
    axs[0].set_title(title)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def floor_plot(d, models, out):
    """ddqm の 1 ステップ予測: 真値 / 解析 / 残差型 B。`step` の 1 本の先頭 0.1 秒。"""
    seqs = sequences(d, "test")
    s = next(x for x in seqs if x["pattern"] == "step")
    f = d["flex"]; n = 800
    a0 = max(0, int(np.argmax(np.abs(np.diff(s["tau"])))) - 200)         # トルクの変化が最大の点の前後
    sl = slice(a0, a0 + n)
    R = np.c_[s["q"], s["dq"], s["qm"] - s["q"], s["dqm"] - s["dq"], s["tau"]][sl]
    s_all = d["splits"]["test"]
    idx = np.flatnonzero(s_all["scenario_id"] == s["sid"])[sl]
    Y = s_all["Y"][idx].astype(np.float64); A = s_all["a_phys"][idx].astype(np.float64)
    t = np.arange(n) / 8000.0
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axs[0].plot(t, s["tau"][sl], "k"); axs[0].set_ylabel("トルク τ [N·m]")
    axs[1].plot(t, Y[:, 1], "k", lw=1.4, label=LAB["truth"]); axs[1].plot(t, A[:, 1], color=STY["analytic"][0], ls="--", label=LAB["analytic"])
    for k, m in models.items():
        p = ev.predict_acc(m, R)
        axs[1].plot(t, p[:, 1], color=STY[k][0], lw=1.0, label=LAB[k])
    axs[1].set_ylabel("ddqm [rad/s²]"); axs[1].legend(fontsize=8)
    dtau = np.r_[np.diff(s["tau"][sl]), 0] * 0.5 / f["Jm"]
    axs[2].plot(t, Y[:, 1] - A[:, 1], "k", lw=1.4, label="真値 − 解析（残差）")
    axs[2].plot(t, dtau, color="tab:orange", ls="--", lw=1.0, label="0.5·Δτ/Jm（区間内のトルク変化）")
    axs[2].set_ylabel("残差 [rad/s²]"); axs[2].legend(fontsize=8); axs[2].set_xlabel("時間 [s]")
    for ax in axs:
        ax.grid(alpha=0.3)
    axs[0].set_title("ddqm の下限: 残差は区間内のトルク変化 0.5·Δτ/Jm でほぼ説明できる（step 系、トルクの変化点の前後 0.1 s）")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/export/j2_nss8kflex.h5")
    p.add_argument("--out-root", default="data/train_nss8kflex")
    p.add_argument("--run", required=True, help="残差型 B の run")
    p.add_argument("--run-a", default=None, help="純 NN A の run（比較用）")
    p.add_argument("--out-dir", default="reports/issue-33/plots")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch.set_num_threads(4)
    d = load(a.data, splits=("test", "benchmark"))
    phys = FlexPhysics(d["physics"], d["flex"], d["dt"])
    models = {"B": load_model(a.out_root, a.run)}
    if a.run_a:
        models["A"] = load_model(a.out_root, a.run_a)

    seqs = [s for s in sequences(d, "benchmark") if np.isfinite(s["qref"]).all()]
    fns = {"analytic": phys.step, **{k: m.step_physical for k, m in models.items()}}
    res = {k: ev.closed_loop_batch(fn, seqs, d["physics"], d["flex"], d["dt"]) for k, fn in fns.items()}
    for nm in ("ptp_v80_002", "ptp_v60_002", "ptp_v20_001"):
        closed_loop_plot(nm, seqs, None, res, os.path.join(a.out_dir, f"closed_loop_{nm}.png"))

    tseqs = sequences(d, "test")
    for pat in ("chirp", "step", "hold"):
        s = next(x for x in tseqs if x["pattern"].startswith(pat))
        n = 4000
        starts = list(range(0, len(s["tau"]) - n, 2000))
        tw = [np.std(s["qm"][a0:a0 + n] - s["q"][a0:a0 + n]) for a0 in starts]
        a0 = starts[int(np.argmax(tw))]
        window_plot(s, a0, n, models, phys, f"窓の開ループ {s['name']}（{pat}、{n} ステップ = 0.5 s、記録トルクを再生）",
                    os.path.join(a.out_dir, f"window_{pat}.png"))
    floor_plot(d, models, os.path.join(a.out_dir, "ddqm_floor.png"))
    print("保存:", a.out_dir)


if __name__ == "__main__":
    main()
