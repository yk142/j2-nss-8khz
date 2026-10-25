#!/usr/bin/env python3
"""(a)(b) の誤差と (c)・(c3) の関係を、解析モデルの係数をずらしたモデルで調べる（issue #34、剛体版 #16 と同じ方法）。

    .venv/bin/python python/sensitivity_flex.py --out reports/issue-34/sensitivity.json

プラント（NSS の代わり）= 二慣性の解析モデルの係数を 1 つだけずらしたもの。制御器（PD＋FF）は真値の係数のまま。
各モデルについて (a) test の 1 ステップ RMSE の解析モデルに対する倍率（ddq・ddqm）、(b) test の 4,000 ステップ窓の θ・ねじり RMSE、
(c) PTP 閉ループの真値との θ RMSE（最悪）、(c3) C4 の判定一致数と J の Spearman を測る。
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_c3 import truth_rows  # noqa: E402
from nss8kflex import c4, evaluate_lib as ev  # noqa: E402
from nss8kflex.data import load  # noqa: E402
from nss8kflex.physics import FlexPhysics  # noqa: E402


class PerturbedPlant:
    """FlexPhysics を、評価の部品が使うモデルの I/F（eval, accel, step_physical）で包む。"""

    def __init__(self, p, flex, dt, name, scales):
        p2, f2 = dict(p), dict(flex)
        for k, s in scales.items():
            (f2 if k in ("Jm", "JL", "k", "c") else p2)[k] = (f2 if k in ("Jm", "JL", "k", "c") else p2)[k] * s
        self.phys = FlexPhysics(p2, f2, dt)
        self.name, self.scales = name, scales

    def eval(self):
        return self

    def accel(self, q, dq, tw, dtw, tau):
        return self.phys.accel(q, dq, tw, dtw, tau)

    def step_physical(self, q, dq, qm, dqm, tau):
        return self.phys.step(q, dq, qm, dqm, tau)


def variants():
    v = [("nominal", {})]
    for k, ss in (("k", (0.9, 0.95, 1.05, 1.1, 1.2)), ("c", (0.5, 2.0)), ("Jm", (0.9, 1.1)), ("JL", (0.9, 1.1)),
                  ("mgL", (0.95, 1.05)), ("Fc", (0.5, 1.5)), ("Bv", (0.5, 2.0))):
        for s in ss:
            v.append((f"{k}x{s:g}", {k: s}))
    return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/export/j2_nss8kflex.h5")
    ap.add_argument("--refs", default="data/export/j2_flex_c4_refs.h5")
    ap.add_argument("--truth", default="reports/issue-25/eval_C4.json")
    ap.add_argument("--out", default="reports/issue-34/sensitivity.json")
    ap.add_argument("--max-windows", type=int, default=300)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    d = load(a.data, splits=("test", "benchmark"))
    p, fl, dt = d["physics"], d["flex"], d["dt"]
    refs = c4.load_refs(a.refs)
    truth = truth_rows(a.truth)
    base = FlexPhysics(p, fl, dt)
    out = []
    for name, sc in variants():
        t0 = time.time()
        m = PerturbedPlant(p, fl, dt, name, sc)
        one = ev.one_step_table(m, d, "test")["all"]
        win = ev.window_rollouts(m, base, d, "test", 4000, 2000, a.max_windows)
        mw = win["methods"]["model"]
        rows, _, _ = ev.closed_loop_benchmark(m, base, d)
        res = c4.run_c4(m.step_physical, refs, p, fl, dt)
        crow = c4.add_J(c4.metrics(res, refs, dt=dt))
        cmp = c4.compare(crow, truth)
        rec = {"name": name, "scales": sc,
               "a_ddq": one["ddq"]["ratio_to_analytic"], "a_ddqm": one["ddqm"]["ratio_to_analytic"],
               "a_ddq_rmse": one["ddq"]["model"], "a_ddqm_rmse": one["ddqm"]["model"],
               "b_q_deg": mw["rmse_q_deg"]["4000"], "b_twist_rad": mw["rmse_twist_rad"]["4000"],
               "c_worst_deg": max(r["model"]["vs_truth_rmse_q_deg"] for r in rows),
               "c_worst_twist_rad": max(r["model"]["vs_truth_rmse_twist_rad"] for r in rows),
               "c_diverged": int(sum(r["model"]["diverged"] for r in rows)),
               "c3_match": cmp["n_class_match"], "c3_spearman_J": cmp["spearman_J"], "c3_spearman_track": cmp["spearman_track"],
               "c3_mismatch": cmp["mismatch"]}
        out.append(rec)
        print(f"[{name:9s}] {time.time() - t0:.0f}s  (a) {rec['a_ddq']:.2f}/{rec['a_ddqm']:.2f}  (b) θ {rec['b_q_deg']:.3f}° ねじり {rec['b_twist_rad']:.1e}  "
              f"(c) {rec['c_worst_deg']:.4f}°  (c3) {rec['c3_match']}/21 ρ {rec['c3_spearman_J']:.3f}", flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
