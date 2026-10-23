"""(a)(b) と (c) の関係（ゲート限界の見極め、#16）: 解析モデルの係数を意図的にずらしたモデルと、学習した補正を λ 倍したモデルで (a)(b)(c) を測る。
使い方: .venv/bin/python tools/nss8k_gate_sensitivity.py <出力.json>（約 1 時間）"""
import json, os, sys, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np, torch
from nss8k.data import load
from nss8k.physics import Physics
from nss8k import evaluate_lib as ev
from evaluate_nss import load_model
torch.set_num_threads(4)
d = load("data/export/j2_nss8k.h5", splits=("test", "benchmark"))
P, dt = d["physics"], d["dt"]
true = Physics(P, dt)

class Wrap:
    residual = False
    def __init__(self, accel): self.accel = accel
    def eval(self): return self
    def ddq(self, q, dq, tau): return self.accel(q, dq, tau)
    def step_physical(self, q, dq, tau): return q + dt * dq, dq + dt * self.accel(q, dq, tau)

def measure(m, tag, extra):
    one = {sp: ev.one_step_table(m, d, sp)["all"]["ratio_to_analytic"] for sp in ("test", "benchmark")}
    win = ev.window_rollouts(m, true, d, "test", 8000, 2000, 300)
    rows, _, _ = ev.closed_loop_benchmark(m, true, d)
    r = dict(tag=tag, **extra, a_test=one["test"], a_bench=one["benchmark"],
             b8000=win["methods"]["model"]["rmse_q_deg"]["8000"], b4000=win["methods"]["model"]["rmse_q_deg"]["4000"],
             c_worst=max(x["model"]["vs_truth_rmse_q_deg"] for x in rows),
             c_div=sum(x["model"]["diverged"] for x in rows))
    print(json.dumps(r, ensure_ascii=False), flush=True)
    return r

out = []
for key in ("M", "mgL", "Bv", "Fc"):
    for sgn in (+1, -1):
        for dl in (0.02, 0.05, 0.1, 0.2, 0.4):
            p = dict(P); p[key] = P[key] * (1 + sgn * dl)
            ph = Physics(p, dt)
            out.append(measure(Wrap(ph.accel), f"{key}{sgn*dl:+.2f}", dict(kind="param", key=key, delta=sgn * dl)))
for run in ("B_w64_s0", "B_w64_s1"):
    model, _ = load_model(f"data/train_nss8k/{run}/model.pt")
    for lam in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        def acc(q, dq, tau, lam=lam, model=model):
            a0 = true.accel(q, dq, tau)
            with torch.no_grad(): return a0 + lam * (model.ddq(q, dq, tau) - a0)
        out.append(measure(Wrap(acc), f"{run}_lam{lam}", dict(kind="lambda", run=run, lam=lam)))
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
