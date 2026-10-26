#!/usr/bin/env python3
"""加速度を状態に持つ残差型 NSS の評価。ゲート (a)(b)(c)(d)（evaluate_nss.py と同じ定義）。

    .venv/bin/python python/evaluate_nss_acc.py --run acc_w64_s0 --report-dir reports/issue-45/acc_w64_s0

(a) は教師強制（acc は真値 Y[k-1]）の 1 ステップ ddq RMSE。解析モデル・「加速度を変えない」予測（acc₁ = acc）との比も記録する。
(b)(c)(d) は自己回帰（acc はモデル自身の出力。最初の acc は解析モデルで初期化）。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_nss import plot_closed, plot_horizon, plot_training  # noqa: E402
from nss8k import evaluate_lib as ev  # noqa: E402
from nss8k.data import load  # noqa: E402
from nss8k.physics import Physics  # noqa: E402
from nss8kacc.data import pairs  # noqa: E402
from nss8kacc.model import build_model  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/export/j2_nss8k.h5")
    p.add_argument("--run", required=True)
    p.add_argument("--out-root", default="data/train_nss8kacc")
    p.add_argument("--report-dir", default=None)
    p.add_argument("--horizon", type=int, default=8000)
    p.add_argument("--stride", type=int, default=2000)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


@torch.no_grad()
def one_step_teacher(model, split, bs=262144):
    """教師強制の 1 ステップ ddq RMSE [rad/s²]（モデル / 解析モデル / acc を変えない予測）。step = 0 の行は除く。"""
    model.eval()
    acc, q, dq, tau, y = pairs(split)
    ana = split["a_phys"][np.flatnonzero(split["step"] > 0), 0].astype(np.float64)
    T = [torch.from_numpy(v) for v in (acc, q, dq, tau)]
    pred = np.concatenate([(T[0][i:i + bs] + model.delta_acc(*(t[i:i + bs] for t in T))).numpy()
                           for i in range(0, len(acc), bs)])
    rms = lambda e: float(np.sqrt(np.mean(e ** 2)))
    r = {"n": int(len(y)), "target_std": float(np.std(y)), "model": rms(pred - y), "analytic": rms(ana - y),
         "persistence": rms(acc - y)}
    r["ratio_to_analytic"] = r["model"] / r["analytic"]
    return {"all": r}


def main(argv=None):
    a = parse(argv)
    torch.set_num_threads(a.threads)
    run_dir = os.path.join(a.out_root, a.run)
    rep = a.report_dir or os.path.join("reports", "nss8kacc", a.run)
    os.makedirs(rep, exist_ok=True)
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu")
    model = build_model(ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    d = load(a.data, splits=("val", "test", "benchmark"))
    phys = Physics(d["physics"], d["dt"])
    print(f"run {a.run}（AccNSS、幅 {model.width}、best epoch {ck['epoch']}）")

    one = {sp: one_step_teacher(model, d["splits"][sp]) for sp in ("val", "test", "benchmark")}
    for sp, t in one.items():
        r = t["all"]
        print(f"[1 ステップ {sp}、教師強制] モデル {r['model']:.4f}  解析 {r['analytic']:.4f}  acc 据え置き {r['persistence']:.4f}  "
              f"倍率（対解析）{r['ratio_to_analytic']:.3f}  （rad/s²）")
    model.reset()
    win = ev.window_rollouts(model, phys, d, "test", a.horizon, a.stride, a.max_windows)
    if win:
        print("[窓の開ループ] θ RMSE [°] " + "  ".join(
            f"{m} {r['rmse_q_deg']['4000']:.4f}(4000) {r['rmse_q_deg'][str(a.horizon)]:.4f}({a.horizon})  発散 {r['diverged_fraction']:.1%}"
            for m, r in win["methods"].items()))
    model.reset()
    rows, seqs, res = ev.closed_loop_benchmark(model, phys, d)
    for r in rows:
        print(f"[閉ループ {r['name']}] 真値との θ RMSE: モデル {r['model']['vs_truth_rmse_q_deg']:.4f}°  "
              f"解析 {r['analytic']['vs_truth_rmse_q_deg']:.4f}°  発散 {r['model']['diverged']}")
    model.reset()
    thr = ev.throughput(model, phys, d)
    print(f"[スループット] モデル {thr['model']['steps_per_s']:,.0f} ステップ/s、解析 {thr['analytic']['steps_per_s']:,.0f}、"
          f"1 サンプル推論 中央値 {thr['model']['single_step_us']['median']:.0f} µs")
    gates = ev.judge("A", one, win, rows, thr)
    for k, v in gates.items():
        print(f"ゲート ({k}): {'合格' if v['pass'] else '不合格'}  {({kk: vv for kk, vv in v.items() if kk != 'pass'})}")

    json.dump({"run": a.run, "kind": "acc", "width": model.width, "depth": model.depth, "best_epoch": ck["epoch"],
               "one_step": one, "window": win, "closed_loop": rows, "throughput": thr,
               "gates": gates, "gate_limits": ev.GATES}, open(os.path.join(rep, "metrics.json"), "w"), indent=1, ensure_ascii=False)
    plot_training(run_dir, os.path.join(rep, "training_curves.png"))
    if win:
        plot_horizon(win, os.path.join(rep, "rollout_horizon.png"))
    plot_closed(seqs, res, rows, os.path.join(rep, "closed_loop.png"))
    print(f"保存: {rep}")


if __name__ == "__main__":
    main()
