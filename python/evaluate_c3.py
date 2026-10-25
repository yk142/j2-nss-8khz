#!/usr/bin/env python3
"""NSS で C4 を再現し、真値（reports/issue-25/eval_C4.json）と比べる（(c3)、issue #34）。

    .venv/bin/python python/evaluate_c3.py --runs B_w128_screen A_w128_screen --out reports/issue-34
'analytic' を runs に含めると解析モデル（基準線）も評価する。出力: <out>/c3_<run>.json、<out>/c3_summary.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nss8kflex import c4  # noqa: E402
from nss8kflex.data import load  # noqa: E402
from nss8kflex.model import build_model  # noqa: E402
from nss8kflex.physics import FlexPhysics  # noqa: E402


def truth_rows(path):
    rows = json.load(open(path))
    rows = [{"wn": r["wn"], "zeta": r["zeta"], "trackRmsDeg": r["trackRmsDeg"], "settleTwist": r["settleTwist"],
             "tauRms": r["tauRms"], "tauMax": r["tauMax"], "diverged": bool(r["diverged"])} for r in rows]
    return c4.add_J(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/export/j2_nss8kflex.h5")
    p.add_argument("--refs", default="data/export/j2_flex_c4_refs.h5")
    p.add_argument("--truth", default="reports/issue-25/eval_C4.json")
    p.add_argument("--out-root", default="data/train_nss8kflex")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="reports/issue-34")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    d = load(a.data, splits=("benchmark",))        # 物理定数と flex だけ使う
    p_, fl, dt = d["physics"], d["flex"], d["dt"]
    refs = c4.load_refs(a.refs)
    truth = truth_rows(a.truth)
    phys = FlexPhysics(p_, fl, dt)
    summary = {}
    for run in a.runs:
        if run == "analytic":
            fn = phys.step
        else:
            ck = torch.load(os.path.join(a.out_root, run, "model.pt"), map_location="cpu")
            m = build_model(ck["config"]); m.load_state_dict(ck["state_dict"]); m.eval()
            fn = m.step_physical
        t0 = time.time()
        res = c4.run_c4(fn, refs, p_, fl, dt)
        rows = c4.add_J(c4.metrics(res, refs, dt=dt))
        cmp = c4.compare(rows, truth)
        json.dump({"run": run, "rows": rows, "compare": cmp}, open(os.path.join(a.out, f"c3_{run}.json"), "w"), indent=1)
        summary[run] = cmp
        print(f"[{run}] {time.time() - t0:.0f}s  判定一致 {cmp['n_class_match']}/{cmp['n_gains']}  不一致 {cmp['mismatch']}  "
              f"Spearman J {cmp['spearman_J']:.3f}（追従 {cmp['spearman_track']:.3f}、安定 {cmp['n_stable_truth']} 組）", flush=True)
    json.dump(summary, open(os.path.join(a.out, "c3_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
