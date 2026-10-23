"""(c2) ゲイン違いの再現性の評価。docs/requirements_c2.md。
使い方: .venv/bin/python python/evaluate_c2.py --report-dir reports/issue-19 [--conds E1 E2]
真値は data/c2/（tools/j2_gen_c2.m）。モデルは models/nss8k/{B_w64_s2,A_w256_s2}、解析モデルも併記。"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from nss8k import c2
from nss8k.data import load
from nss8k.physics import Physics
from evaluate_nss import load_model

SPEARMAN_MIN, BEST_TOL = 0.9, 0.05            # 合否（真値の生成前に確定。docs/requirements_c2.md §3.6）
MODELS = {"B_w64_s2": "models/nss8k/B_w64_s2/model.pt", "A_w256_s2": "models/nss8k/A_w256_s2/model.pt"}


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/export/j2_nss8k.h5")
    p.add_argument("--c2-root", default="data/c2")
    p.add_argument("--report-dir", default="reports/issue-19")
    p.add_argument("--conds", nargs="+", default=["E1", "E2"])
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


def evaluate(a):
    torch.set_num_threads(a.threads)
    d = load(a.data, splits=("benchmark",))
    p, dt = d["physics"], d["dt"]
    phys = Physics(p, dt)
    fns = {"analytic": phys.step}
    for k, path in MODELS.items():
        fns[k] = load_model(path)[0].step_physical
    res = {"gains": c2.GAINS, "conds": {}}
    for cond in a.conds:
        truth, tagg = {}, {}
        for wn, z in c2.GAINS:
            tr = c2.load_truth(a.c2_root, cond, wn, z)
            assert len(tr) == 12, f"{cond} wn={wn} z={z}: 軌道 {len(tr)} 本（12 本のはず）"
            truth[(wn, z)] = tr
            K = np.array([len(s["tau"]) - 1 for s in tr]); T = int(K.max())
            Q = np.full((len(tr), T + 1), np.nan); U = np.full((len(tr), T), np.nan)
            for i, s in enumerate(tr):
                Q[i, :K[i] + 1] = s["q"]; U[i, :K[i]] = s["tau"][:K[i]]
            tagg[(wn, z)] = c2.aggregate(c2.trial_metrics(Q, U, K, tr, dt))
        ref = {k: float(np.median([tagg[g][k] for g in c2.GAINS])) for k in ("theta_rmse_deg", "tau_rms")}
        Jt = [c2.cost(tagg[g], ref) for g in c2.GAINS]
        out = {"ref": ref, "truth": {f"{g[0]}_{g[1]}": tagg[g] for g in c2.GAINS}, "J_truth": Jt, "models": {}}
        for name, fn in fns.items():
            agg, vs_truth, J = {}, {}, []
            for g in c2.GAINS:
                tr = truth[g]
                Q, U, K = c2.closed_loop(fn, tr, p, dt, g[0], g[1])
                rows = c2.trial_metrics(Q, U, K, tr, dt)
                agg[g] = c2.aggregate(rows)
                vs_truth[g] = float(np.nanmean([np.degrees(np.sqrt(np.nanmean((Q[i, :K[i] + 1] - s["q"]) ** 2))) for i, s in enumerate(tr)]))
                J.append(c2.cost(agg[g], ref))
                print(f"[{cond} {name} wn={g[0]} ζ={g[1]}] θ RMSE {agg[g]['theta_rmse_deg']:.4f}° (真値 {tagg[g]['theta_rmse_deg']:.4f}°)  "
                      f"J {J[-1]:.3f} (真値 {Jt[c2.GAINS.index(g)]:.3f})  真値との差 {vs_truth[g]:.4f}°  発散 {agg[g]['n_diverged']}", flush=True)
            best = int(np.nanargmin(J))
            rho = c2.spearman(J, Jt)
            within = Jt[best] <= (1 + BEST_TOL) * min(Jt)
            out["models"][name] = {
                "agg": {f"{g[0]}_{g[1]}": agg[g] for g in c2.GAINS}, "vs_truth_theta_deg": {f"{g[0]}_{g[1]}": vs_truth[g] for g in c2.GAINS},
                "J": J, "spearman_J": rho, "best_gain": c2.GAINS[best], "best_gain_truth": c2.GAINS[int(np.argmin(Jt))],
                "J_truth_at_best": Jt[best], "J_truth_min": min(Jt), "best_within_tol": bool(within),
                "spearman_theta": c2.spearman([agg[g]["theta_rmse_deg"] for g in c2.GAINS], [tagg[g]["theta_rmse_deg"] for g in c2.GAINS]),
                "spearman_tau": c2.spearman([agg[g]["tau_rms"] for g in c2.GAINS], [tagg[g]["tau_rms"] for g in c2.GAINS]),
                "spearman_max_err": c2.spearman([agg[g]["max_err_deg"] for g in c2.GAINS], [tagg[g]["max_err_deg"] for g in c2.GAINS]),
                "spearman_settle": c2.spearman([agg[g]["settle_s"] for g in c2.GAINS], [tagg[g]["settle_s"] for g in c2.GAINS]),
                "pass": bool(rho >= SPEARMAN_MIN and within and sum(v["n_diverged"] for v in agg.values()) == 0)}
            m = out["models"][name]
            print(f"== {cond} {name}: Spearman(J) {rho:.3f}  最適 {m['best_gain']}（真値 {m['best_gain_truth']}、J 比 {Jt[best] / min(Jt):.3f}）  合否 {m['pass']}", flush=True)
        res["conds"][cond] = out
    return res


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in ("Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "TakaoGothic"):
        if any(f in x.name for x in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    conds = list(res["conds"])
    fig, axs = plt.subplots(1, len(conds), figsize=(6.5 * len(conds), 4.8), squeeze=False)
    col = {"analytic": "tab:green", "B_w64_s2": "tab:blue", "A_w256_s2": "tab:red"}
    for ax, c in zip(axs[0], conds):
        o = res["conds"][c]
        x = np.arange(len(res["gains"]))
        ax.plot(x, o["J_truth"], "k-o", lw=2, label="真値")
        for n, m in o["models"].items():
            ax.plot(x, m["J"], "--s", color=col[n], label=f"{n}（ρ={m['spearman_J']:.2f}）")
        ax.set_xticks(x); ax.set_xticklabels([f"{g[0]}\nζ{g[1]}" for g in res["gains"]], fontsize=8)
        ax.set_xlabel("wn [rad/s] / ζ"); ax.set_ylabel("コスト J（真値の中央値で正規化）"); ax.set_title(f"条件 {c}"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=100)


def main(argv=None):
    a = parse(argv)
    os.makedirs(a.report_dir, exist_ok=True)
    res = evaluate(a)
    json.dump(res, open(os.path.join(a.report_dir, "c2_metrics.json"), "w"), indent=1, ensure_ascii=False)
    plot(res, os.path.join(a.report_dir, "c2_cost.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
