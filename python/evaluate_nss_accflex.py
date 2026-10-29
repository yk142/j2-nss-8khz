#!/usr/bin/env python3
"""弾性 J2 の加速度状態 NSS の評価。ゲート (a)(b)(c)(d)（evaluate_nss_flex.py と同じ定義。(b) のねじり倍率は 2.0）。

    .venv/bin/python python/evaluate_nss_accflex.py --run accflex_w64_s0 --report-dir reports/issue-51/accflex_w64_s0

(a) は教師強制（acc は真値 Y[k-1]）の 1 ステップ RMSE。(b)(c)(d) は自己回帰（最初の acc は解析モデルで初期化）。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_nss_flex import plot_closed, plot_horizon, plot_training  # noqa: E402
from nss8kaccflex.data import feature_cols, pairs  # noqa: E402
from nss8kaccflex.model import build_model  # noqa: E402
from nss8kflex import evaluate_lib as ev  # noqa: E402
from nss8kflex.data import load  # noqa: E402
from nss8kflex.physics import FlexPhysics  # noqa: E402

GATES = dict(ev.GATES, b_twist_ratio=2.0)       # 要件 §5（#43 で確定）: ねじり RMSE ≤ 解析モデルの 2 倍


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/export/j2_nss8kflex.h5")
    p.add_argument("--run", required=True)
    p.add_argument("--out-root", default="data/train_nss8kaccflex")
    p.add_argument("--report-dir", default=None)
    p.add_argument("--horizon", type=int, default=4000)
    p.add_argument("--stride", type=int, default=2000)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


@torch.no_grad()
def one_step_teacher(model, split, tw_std, bs=262144):
    """教師強制の 1 ステップ RMSE [rad/s²]（ddq・ddqm それぞれ。モデル / 解析 / acc を変えない予測 / 0）。step = 0 の行は除く。"""
    model.eval()
    acc, R, y, ana = pairs(split)
    osc = model.out_scale.double().numpy()
    F = torch.from_numpy(feature_cols(acc, R, model.scale, osc, tw_std))
    pred = np.concatenate([(F[i:i + bs, :2].double() * model.out_scale.double()
                            + (model.net_out(F[i:i + bs]) * model.d_std).double()).numpy() for i in range(0, len(F), bs)])
    rms = lambda e: float(np.sqrt(np.mean(e ** 2)))
    row = {"n": int(len(y))}
    for j, o in enumerate(("ddq", "ddqm")):
        e = {"target_std": float(np.std(y[:, j])), "model": rms(pred[:, j] - y[:, j]), "analytic": rms(ana[:, j] - y[:, j]),
             "persistence": rms(acc[:, j] - y[:, j]), "zero": rms(y[:, j])}
        e["ratio_to_analytic"] = e["model"] / e["analytic"]
        row[o] = e
    return {"all": row}


def main(argv=None):
    a = parse(argv)
    torch.set_num_threads(a.threads)
    run_dir = os.path.join(a.out_root, a.run)
    rep = a.report_dir or os.path.join("reports", "nss8kaccflex", a.run)
    os.makedirs(rep, exist_ok=True)
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu")
    model = build_model(ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    d = load(a.data, splits=("val", "test", "benchmark"))
    phys = FlexPhysics(d["physics"], d["flex"], d["dt"])
    tw_std = model.tw_std.double().numpy()
    print(f"run {a.run}（AccFlexNSS、幅 {model.width}、best epoch {ck['epoch']}）")

    one = {sp: one_step_teacher(model, d["splits"][sp], tw_std) for sp in ("val", "test", "benchmark")}
    for sp, t in one.items():
        r = t["all"]
        print(f"[1 ステップ {sp}、教師強制] " + "  ".join(
            f"{o}: モデル {r[o]['model']:.4f} 解析 {r[o]['analytic']:.4f} acc 据え置き {r[o]['persistence']:.4f} 倍率 {r[o]['ratio_to_analytic']:.3f}"
            for o in ("ddq", "ddqm")))
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
    thr = ev.throughput(model, phys, d)
    print(f"[スループット] モデル {thr['model']['steps_per_s']:,.0f} ステップ/s、解析 {thr['analytic']['steps_per_s']:,.0f}、"
          f"1 サンプル推論 中央値 {thr['model']['single_step_us']['median']:.0f} µs")
    gates = ev.judge(one, win, rows, thr, GATES)
    for k, v in gates.items():
        print(f"ゲート ({k}): {'合格' if v['pass'] else '不合格'}  {({kk: vv for kk, vv in v.items() if kk != 'pass'})}")

    json.dump({"run": a.run, "kind": "accflex", "width": model.width, "depth": model.depth, "best_epoch": ck["epoch"],
               "one_step": one, "window": win, "closed_loop": rows, "throughput": thr,
               "gates": gates, "gate_limits": GATES}, open(os.path.join(rep, "metrics.json"), "w"), indent=1, ensure_ascii=False)
    plot_training(run_dir, os.path.join(rep, "training_curves.png"))
    if win:
        plot_horizon(win, os.path.join(rep, "rollout_horizon.png"))
    plot_closed(seqs, res, rows, os.path.join(rep, "closed_loop.png"))
    print(f"保存: {rep}")


if __name__ == "__main__":
    main()
