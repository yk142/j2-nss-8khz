#!/usr/bin/env python3
"""学習済みの 4 状態 NSS を評価し、ゲート (a)(b)(c)(d) を判定する（docs/requirements_nss_flex.md §5）。

    .venv/bin/python python/evaluate_nss_flex.py --run B_w128_s0 --report-dir reports/issue-33/B_w128_s0

(a) test の 1 ステップ [ddq, ddqm] の RMSE（解析モデルに対する倍率、2 出力とも）
(b) test の 4,000 ステップ窓の開ループ（記録トルクを再生）の θ RMSE とねじり RMSE
(c) benchmark（PTP 12 本）の閉ループ（8 kHz のモータ角 PD＋FF にモデルを接続）。真値との θ RMSE とねじり RMSE
(d) バッチ並列ロールアウトのスループット
出力: <report-dir>/{metrics.json, training_curves.png, rollout_horizon.png, closed_loop.png}
"""
import argparse
import json
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
from nss8kflex.data import load  # noqa: E402
from nss8kflex.model import build_model  # noqa: E402
from nss8kflex.physics import FlexPhysics  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "TakaoPGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
COL = {"model": "tab:red", "analytic": "tab:blue", "truth": "k"}
LAB = {"model": "学習モデル", "analytic": "解析モデル", "truth": "真値"}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/export/j2_nss8kflex.h5")
    p.add_argument("--run", required=True)
    p.add_argument("--out-root", default="data/train_nss8kflex")
    p.add_argument("--report-dir", default=None)
    p.add_argument("--horizon", type=int, default=4000)
    p.add_argument("--stride", type=int, default=2000)
    p.add_argument("--max-windows", type=int, default=0, help="窓の数の上限（動作確認用。0 は全件）")
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    m = build_model(ck["config"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def plot_training(run_dir, out):
    h = json.load(open(os.path.join(run_dir, "history.json")))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy([r["epoch"] for r in h], [r["train_loss"] for r in h], label="train")
    ax.semilogy([r["epoch"] for r in h], [r["val_loss"] for r in h], label="val")
    ax.set_xlabel("エポック"); ax.set_ylabel("損失（標準化した出力の MSE）"); ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_horizon(win, out):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for meth, r in win["methods"].items():
        axs[0].loglog(np.arange(1, len(r["curve_q_deg"])), np.maximum(r["curve_q_deg"][1:], 1e-9), color=COL[meth], label=LAB[meth])
        axs[1].loglog(np.arange(1, len(r["curve_twist_rad"])), np.maximum(r["curve_twist_rad"][1:], 1e-12), color=COL[meth], label=LAB[meth])
    axs[0].set_ylabel("θ の RMSE [度]"); axs[1].set_ylabel("ねじりの RMSE [rad]")
    for ax in axs:
        ax.set_xlabel("ステップ（8 kHz）"); ax.grid(True, which="both", alpha=0.3); ax.legend()
    axs[0].set_title(f"窓の開ループ（{win['n_windows']} 窓、記録トルクを再生）")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_closed(seqs, res, rows, out):
    n = len(seqs)
    fig, axs = plt.subplots(int(np.ceil(n / 3)), 3, figsize=(13, 2.6 * int(np.ceil(n / 3))), squeeze=False)
    for i, (s, ax) in enumerate(zip(seqs, axs.ravel())):
        K = len(s["tau"]); t = np.arange(K) / 8000.0
        ax.plot(t, np.degrees(s["qref"]), "g--", lw=0.8, label="参照")
        ax.plot(t, np.degrees(s["q"]), "k", lw=1, label=LAB["truth"])
        for name in res:
            ax.plot(t, np.degrees(res[name][0][i, :K]), color=COL[name], lw=0.8, label=LAB[name])
        ax.set_title(f"{s['name']}  真値との θ RMSE {rows[i]['model']['vs_truth_rmse_q_deg']:.4f}°", fontsize=8)
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def main(argv=None):
    a = parse(argv)
    torch.set_num_threads(a.threads)
    run_dir = os.path.join(a.out_root, a.run)
    rep = a.report_dir or os.path.join("reports", "nss8kflex", a.run)
    os.makedirs(rep, exist_ok=True)
    model, ck = load_model(os.path.join(run_dir, "model.pt"))
    kind = "B" if model.residual else "A"
    d = load(a.data, splits=("val", "test", "benchmark"))
    phys = FlexPhysics(d["physics"], d["flex"], d["dt"])
    print(f"run {a.run}（モデル {kind}、幅 {model.width}、best epoch {ck['epoch']}）")

    one = {sp: ev.one_step_table(model, d, sp) for sp in ("val", "test", "benchmark")}
    for sp, t in one.items():
        r = t["all"]
        print(f"[1 ステップ {sp}] " + "  ".join(
            f"{o}: モデル {r[o]['model']:.4f} 解析 {r[o]['analytic']:.4f} 倍率 {r[o]['ratio_to_analytic']:.3f}" for o in ("ddq", "ddqm")))
    win = ev.window_rollouts(model, phys, d, "test", a.horizon, a.stride, a.max_windows)
    if win:
        h = str(a.horizon)
        print(f"[窓の開ループ {win['n_windows']} 窓、{a.horizon} ステップ] " + "  ".join(
            f"{m}: θ {r['rmse_q_deg'][h]:.4f}° ねじり {r['rmse_twist_rad'][h]:.2e} rad（発散 {r['diverged_fraction']:.1%}）"
            for m, r in win["methods"].items()))
    rows, seqs, res = ev.closed_loop_benchmark(model, phys, d)
    for r in rows:
        print(f"[閉ループ {r['name']}] 真値との θ RMSE: モデル {r['model']['vs_truth_rmse_q_deg']:.4f}°  "
              f"解析 {r['analytic']['vs_truth_rmse_q_deg']:.4f}°  ねじり差 モデル {r['model']['vs_truth_rmse_twist_rad']:.2e}  発散 {r['model']['diverged']}")
    cc = ev.controller_check(seqs, d["physics"], d["flex"], d["dt"])
    print(f"[制御器の再現] 真値の状態に Python の PD＋FF を当てたトルクと記録の差 RMS: 最大 {max(c['rms_N_m'] for c in cc):.3f} N·m")
    thr = ev.throughput(model, phys, d)
    print(f"[スループット] モデル {thr['model']['steps_per_s']:,.0f} ステップ/s、解析 {thr['analytic']['steps_per_s']:,.0f}、"
          f"1 サンプル推論 中央値 {thr['model']['single_step_us']['median']:.0f} µs")
    gates = ev.judge(one, win, rows, thr)
    for k, v in gates.items():
        print(f"ゲート ({k}): {'合格' if v['pass'] else '不合格'}  {({kk: vv for kk, vv in v.items() if kk != 'pass'})}")

    json.dump({"run": a.run, "kind": kind, "width": model.width, "depth": model.depth, "best_epoch": ck["epoch"],
               "one_step": one, "window": win, "closed_loop": rows, "controller_check": cc, "throughput": thr,
               "gates": gates, "gate_limits": ev.GATES}, open(os.path.join(rep, "metrics.json"), "w"), indent=1, ensure_ascii=False)
    plot_training(run_dir, os.path.join(rep, "training_curves.png"))
    if win:
        plot_horizon(win, os.path.join(rep, "rollout_horizon.png"))
    plot_closed(seqs, res, rows, os.path.join(rep, "closed_loop.png"))
    print(f"保存: {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
