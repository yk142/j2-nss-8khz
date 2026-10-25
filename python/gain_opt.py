#!/usr/bin/env python3
"""NSS による C4 ゲインの最適化と、真値での検証（issue #39）。docs/plan_gain_opt.md

    gain_opt.py scan   --runs B_w64_s0 ... analytic k0.9     # NSS で 512 ゲインを走査 → reports/issue-39/scan_<run>.json
    gain_opt.py select                                         # 検証用ゲインを選ぶ → reports/issue-39/verify_gains.json
    （MATLAB: j2_flex_eval('reports/issue-39/truth_verify.json', {'C4'}, gains)）
    gain_opt.py judge                                          # 基準 1〜5 を判定
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

def key(wn, z):
    return (round(float(wn), 3), round(float(z), 3))


OUT = "reports/issue-41"                     # 出力先。#39 の走査結果は OUT_OLD から読み、新しい格子点だけを OUT に足す
OUT_OLD = "reports/issue-39"
WN0, WN_RATIO, N_WN = 15.0, 1.1, 41          # wn = 15〜680 rad/s（#39 は 32 点で 288 まで）
ZETAS = [round(0.1 + 0.1 * i, 1) for i in range(20)]     # ζ = 0.1〜2.0（#39 は 0.5〜2.0）
WNS = [WN0 * WN_RATIO ** i for i in range(N_WN)]
GRID = [(w, z) for w in WNS for z in ZETAS]
OLD_GRID = {key(WN0 * WN_RATIO ** i, round(0.5 + 0.1 * j, 1)) for i in range(32) for j in range(16)}
NORM = c4.NORM_GAIN
CHUNK = 96


def make_step(run, p, fl, dt, out_root):
    if run == "analytic":
        return FlexPhysics(p, fl, dt).step
    if run.startswith("k") and run[1:].replace(".", "").isdigit():       # 負の対照: k を倍率だけずらした解析モデル（例 k0.9）
        from sensitivity_flex import PerturbedPlant
        return PerturbedPlant(p, fl, dt, run, {"k": float(run[1:])}).step_physical
    ck = torch.load(os.path.join(out_root, run, "model.pt"), map_location="cpu")
    m = build_model(ck["config"]); m.load_state_dict(ck["state_dict"]); m.eval()
    return m.step_physical


def scan_rows(fn, refs, p, fl, dt, gains):
    rows = []
    for i in range(0, len(gains), CHUNK):
        g = gains[i:i + CHUNK]
        res = c4.run_c4(fn, refs, p, fl, dt, gains=g)
        rows += c4.metrics(res, refs, gains=g, dt=dt)
    return c4.add_J(rows)


def mark_margin(rows):
    """余裕あり: wn/1.1 と wn·1.1（同じ ζ）も安定。格子の端は余裕なしとみなす。"""
    st = {key(r["wn"], r["zeta"]): not r["unstable"] for r in rows}
    for r in rows:
        a, b = key(r["wn"] / WN_RATIO, r["zeta"]), key(r["wn"] * WN_RATIO, r["zeta"])
        r["margin"] = bool(st[key(r["wn"], r["zeta"])] and st.get(a, False) and st.get(b, False))
    return rows


def optimum(rows, margin):
    c = [r for r in rows if not r["unstable"] and (r["margin"] or not margin)]
    return min(c, key=lambda r: r["J"]) if c else None


def boundary(rows, zeta):
    """ζ ごとに、wn を上げていって最初に不安定になる wn（なければ None）。"""
    for r in sorted((r for r in rows if key(0, r["zeta"]) == key(0, zeta)), key=lambda r: r["wn"]):
        if r["unstable"]:
            return r["wn"]
    return None


def cmd_scan(a):
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    d = load(a.data, splits=("benchmark",))
    p, fl, dt = d["physics"], d["flex"], d["dt"]
    refs = c4.load_refs(a.refs)
    gains = [g for g in GRID if key(*g) not in OLD_GRID] + [NORM]       # #39 で走査済みの格子点は再利用
    for run in a.runs:
        path = os.path.join(OUT, f"scan_{run}.json")
        if os.path.exists(path):
            continue
        t0 = time.time()
        rows = scan_rows(make_step(run, p, fl, dt, a.out_root), refs, p, fl, dt, gains)
        json.dump({"run": run, "rows": rows}, open(path, "w"))
        rows = load_scan(run)
        o1, o0 = optimum(rows, True), optimum(rows, False)
        f = lambda r: f"(wn {r['wn']:.1f}, ζ {r['zeta']:.1f}, J {r['J']:.3f})" if r else "なし"
        print(f"[{run}] {time.time() - t0:.0f}s  安定 {sum(not r['unstable'] for r in rows)}/{len(rows)}  最適（余裕あり）{f(o1)}  最適（余裕なし）{f(o0)}", flush=True)


def load_scan(run):
    """#39 の走査（OUT_OLD）と、新しい格子点（OUT）をつないで返す。J は同じ (30, 1) で正規化済み。余裕は広い格子で付け直す。"""
    old = json.load(open(os.path.join(OUT_OLD, f"scan_{run}.json")))["rows"]
    new = json.load(open(os.path.join(OUT, f"scan_{run}.json")))["rows"]
    rows = [r for r in old if key(r["wn"], r["zeta"]) != key(*NORM)] + [r for r in new if key(r["wn"], r["zeta"]) != key(*NORM)]
    return mark_margin(rows)


def cmd_select(a):
    runs = a.runs
    pick = {}                                           # key -> 理由

    def add(wn, z, why):
        if min(WNS) * 0.999 <= wn <= max(WNS) * 1.001 and 0.099 <= z <= 2.001:
            pick.setdefault(key(wn, z), []).append(why)
    for run in runs:
        rows = load_scan(run)
        for margin, tag in ((True, "余裕あり最適"), (False, "余裕なし最適")):
            o = optimum(rows, margin)
            if o:
                add(o["wn"], o["zeta"], f"{run} {tag}")
                if margin:
                    add(o["wn"] / WN_RATIO, o["zeta"], f"{run} 最適の −10%")
                    add(o["wn"] * WN_RATIO, o["zeta"], f"{run} 最適の +10%")
    base = load_scan(a.base)
    for r in sorted((r for r in base if not r["unstable"] and r["margin"]), key=lambda r: r["J"])[:4]:
        add(r["wn"], r["zeta"], f"{a.base} J 上位")
    for z in (0.7, 1.0, 1.5):
        wb = boundary(base, z)
        if wb:
            add(wb / WN_RATIO, z, f"{a.base} 境界の手前 ζ={z}"); add(wb, z, f"{a.base} 境界の向こう ζ={z}")
    exist = set(truth_all())
    todo = [k for k in pick if k not in exist]
    out = {"gains": [list(k) for k in todo], "reasons": {f"{k[0]},{k[1]}": v for k, v in pick.items()}}
    json.dump(out, open(os.path.join(OUT, "verify_gains.json"), "w"), indent=1, ensure_ascii=False)
    print(f"検証用ゲイン {len(todo)} 点（既存の真値と重複 {len(pick) - len(todo)} 点は除く）")
    for k in todo:
        print(" ", k, "; ".join(pick[k]))


def truth_all():
    """既存の 21 ゲイン（#25）＋検証用ゲイン（issue #39）の真値。J は (30, 1) の真値で正規化。"""
    rows = json.load(open("reports/issue-25/eval_C4.json"))
    for d in (OUT_OLD, OUT):
        path = os.path.join(d, "truth_verify.json")
        if os.path.exists(path):
            rows += json.load(open(path))
    seen, uniq = set(), []
    for r in rows:
        k = key(r["wn"], r["zeta"])
        if k not in seen:
            seen.add(k); uniq.append({x: r[x] for x in ("wn", "zeta", "trackRmsDeg", "settleTwist", "tauRms", "tauMax", "diverged")})
    return {key(r["wn"], r["zeta"]): r for r in c4.add_J(uniq)}


def truth_margin(tm, r):
    k = key(r["wn"], r["zeta"])
    nb = [key(r["wn"] / WN_RATIO, r["zeta"]), key(r["wn"] * WN_RATIO, r["zeta"])]
    return (not r["unstable"]) and all(n in tm and not tm[n]["unstable"] for n in nb)


def truth_bracket(tm, zeta):
    """真値の安定境界の範囲: (評価済みの最大の安定 wn で、それより上がすべて不安定, 最小の不安定 wn)。"""
    pts = sorted((r["wn"], r["unstable"]) for r in tm.values() if key(0, r["zeta"]) == key(0, zeta))
    first_bad = next((w for w, u in pts if u), None)
    if first_bad is None:
        return None, None
    lo = max((w for w, u in pts if w < first_bad and not u), default=None)
    return lo, first_bad


def cmd_judge(a):
    tm = truth_all()
    best_any = min(r["J"] for r in tm.values() if not r["unstable"])
    best_margin = min((r["J"] for r in tm.values() if truth_margin(tm, r)), default=None)
    res = {"truth_n": len(tm), "truth_best_J_any": best_any, "truth_best_J_margin": best_margin, "models": {}}
    for run in a.runs + a.controls:
        rows = load_scan(run)
        m = {}
        for margin, tag in ((True, "margin"), (False, "nomargin")):
            o = optimum(rows, margin)
            k = key(o["wn"], o["zeta"])
            t = tm.get(k)
            e = {"wn": o["wn"], "zeta": o["zeta"], "J_nss": o["J"], "truth_evaluated": t is not None}
            if t is not None:
                ref = best_margin if margin else best_any
                e.update({"truth_stable": not t["unstable"], "J_truth": t["J"], "settleTwist_truth": t["settleTwist"],
                          "J_relerr": abs(o["J"] - t["J"]) / t["J"],
                          "regret": (t["J"] - ref) / ref if ref else None})
            m[tag] = e
        b = {}
        for z in (0.7, 1.0, 1.5):
            nb, (lo, hi) = boundary(rows, z), truth_bracket(tm, z)
            ok = None if (nb is None or hi is None) else bool((lo or 0) * 0.9 * (1 - 1e-4) <= nb <= hi * 1.1 * (1 + 1e-4))
            b[str(z)] = {"nss": nb, "truth_lo": lo, "truth_hi": hi, "within10pct": ok}
        m["boundary"] = b
        res["models"][run] = m
    for grp in (a.runs_B, a.runs_A):
        ws = [res["models"][r]["margin"]["wn"] for r in grp]
        res["spread_" + grp[0].split("_")[0]] = {"wn": ws, "ratio": max(ws) / min(ws), "pass": bool(max(ws) / min(ws) <= WN_RATIO + 1e-9)}
    json.dump(res, open(os.path.join(OUT, "judge.json"), "w"), indent=1, ensure_ascii=False)
    for run, m in res["models"].items():
        g = m["margin"]
        print(f"[{run}] 余裕あり最適 (wn {g['wn']:.1f}, ζ {g['zeta']:.1f}) J_nss {g['J_nss']:.3f}",
              (f"真値 安定 {g['truth_stable']} J {g['J_truth']:.3f} 後悔 {100 * g['regret']:.1f}% 相対誤差 {100 * g['J_relerr']:.1f}%" if g["truth_evaluated"] else "真値未評価"))
        n = m["nomargin"]
        print(f"      余裕なし最適 (wn {n['wn']:.1f}, ζ {n['zeta']:.1f})",
              (f"真値 安定 {n['truth_stable']} 後悔 {100 * n['regret']:.1f}%" if n["truth_evaluated"] else "真値未評価"),
              " 境界", {z: (v["nss"], v["truth_lo"], v["truth_hi"], v["within10pct"]) for z, v in m["boundary"].items()})
    for k in [k for k in res if k.startswith("spread_")]:
        print(k, res[k])


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "TakaoPGothic", "DejaVu Sans"]
    tm = truth_all()
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    fig, axs = plt.subplots(1, len(a.show), figsize=(5.6 * len(a.show), 4.6), sharey=True)
    for ax, run in zip(axs, a.show):
        rows = [r for r in load_scan(run) if key(r["wn"], r["zeta"]) in {key(w, z) for w, z in GRID}]
        W, Z = np.array(WNS), np.array(ZETAS)
        M = np.full((len(Z), len(W)), np.nan)
        for r in rows:
            if not r["unstable"]:
                M[ZETAS.index(round(r["zeta"], 1)), int(round(np.log(r["wn"] / WN0) / np.log(WN_RATIO)))] = r["J"]
        im = ax.pcolormesh(W, Z, M, norm=LogNorm(0.1, 2.0), cmap="viridis_r", shading="nearest")
        ax.set_facecolor("0.85")
        for r in tm.values():
            if r["unstable"]:
                ax.scatter(r["wn"], r["zeta"], marker="x", c="r", s=40)
            else:
                ax.scatter(r["wn"], r["zeta"], marker="o", s=50, c=[r["J"]], cmap="viridis_r", norm=LogNorm(0.1, 2.0), edgecolors="k")
        o = optimum(mark_margin(load_scan(run)), True)
        ax.scatter(o["wn"], o["zeta"], marker="*", s=260, c="orange", edgecolors="k", zorder=5)
        ax.set_xscale("log"); ax.set_xlabel("wn [rad/s]"); ax.set_title(f"{run}: NSS の J（灰: 不安定）\n丸=真値の J、×=真値で不安定、★=NSS 最適（余裕あり）", fontsize=9)
    axs[0].set_ylabel("ζ")
    fig.colorbar(im, ax=axs, label="J（対数）")
    fig.savefig(os.path.join(OUT, "plots", "gain_map.png"), dpi=130, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--data", default="data/export/j2_nss8kflex.h5")
    s.add_argument("--refs", default="data/export/j2_flex_c4_refs.h5")
    s.add_argument("--out-root", default="data/train_nss8kflex")
    s.add_argument("--threads", type=int, default=4)
    s.set_defaults(f=cmd_scan)
    s = sub.add_parser("select")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--base", default="B_w64_s0")
    s.set_defaults(f=cmd_select)
    s = sub.add_parser("judge")
    s.add_argument("--runs-B", nargs="+", default=["B_w64_s0", "B_w64_s1", "B_w64_s2"])
    s.add_argument("--runs-A", nargs="+", default=["A_w64_s0", "A_w64_s1", "A_w64_s2"])
    s.add_argument("--controls", nargs="+", default=["analytic", "k0.9"])
    s.set_defaults(f=cmd_judge)
    s = sub.add_parser("plot")
    s.add_argument("--show", nargs="+", default=["analytic", "B_w64_s0", "k0.9"])
    s.set_defaults(f=cmd_plot)
    a = ap.parse_args()
    if a.cmd == "judge":
        a.runs = a.runs_B + a.runs_A
    a.f(a)


if __name__ == "__main__":
    main()
