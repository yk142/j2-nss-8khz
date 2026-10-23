"""ゲート (d) のスループットだけを測り直し、既存の metrics.json の throughput と gates.d を更新する。
使い方: .venv/bin/python tools/nss8k_remeasure_d.py <outdir> <run>...   （モデルは data/train/<run>/model.pt）"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import torch
from nss8k.data import load
from nss8k.physics import Physics
from nss8k import evaluate_lib as ev
from evaluate_nss import load_model

torch.set_num_threads(4)
out, runs = sys.argv[1], sys.argv[2:]
d = load("data/export/j2_nss8k.h5", splits=("test",))
phys = Physics(d["physics"], d["dt"])
for r in runs:
    model, _ = load_model(os.path.join("data/train_nss8k", r, "model.pt"))
    thr = ev.throughput(model, phys, d)
    p = os.path.join(out, r, "metrics.json")
    m = json.load(open(p))
    m["throughput"] = thr
    m["gates"]["d"].update(value=thr["model"]["steps_per_s"], pass_=None)
    m["gates"]["d"]["pass"] = thr["model"]["steps_per_s"] >= ev.GATES["d_steps_per_s"]
    m["gates"]["d"].pop("pass_")
    m["gates"]["d"]["analytic_steps_per_s"] = thr["analytic"]["steps_per_s"]
    json.dump(m, open(p, "w"), indent=1, ensure_ascii=False)
    t = thr["model"]
    print(f"{r}: 最速 {t['steps_per_s']:,.0f}  中央値 {t['steps_per_s_median']:,.0f}  "
          f"最遅 {min(t['steps_per_s_runs']):,.0f} ステップ/s  合否 {m['gates']['d']['pass']}", flush=True)
