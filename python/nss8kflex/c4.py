"""C4（1 kHz サンプル・ホールド＋1 サンプル遅れのモータ角 PD）の再現。docs/requirements_nss_flex.md §5 (c3)、issue #34。

真値（Simulink、tools/j2_flex_eval.m の C4）と同じ制御: 偏差を 1 kHz でサンプル → PD → 1 サンプル遅らせて保持。
FF（`j2FeedforwardFlex` と同じ）は 8 kHz のまま。軌道 6 本 × ゲイン 21 組を、プラント（step_fn）に接続して同時に進める。
"""
import h5py
import numpy as np
import torch

from .evaluate_lib import flex_reference, fresh

WNS = (15, 30, 60, 90, 120, 180, 240)
ZETAS = (0.7, 1.0, 1.5)
GAINS = [(wn, z) for wn in WNS for z in ZETAS]
UNSTABLE_SETTLE = 3e-5          # 整定ねじり [rad] がこれ以上なら持続振動（真値: 安定 ≤ 3.9e-6、不安定 ≥ 2.6e-4）
NORM_GAIN = (30, 1.0)           # J の正規化に使うゲイン


def load_refs(path):
    out = []
    with h5py.File(path, "r") as f:
        dt = float(np.ravel(f["dt"][()])[0])
        for name in f:
            if name == "dt":
                continue
            g = f[name]
            out.append({"name": name, "qref": g["qr"][:], "tEnd": float(np.ravel(g.attrs["tEnd"])[0]), "dt": dt})
    return out


@torch.no_grad()
def run_c4(step_fn, refs, p, flex, dt, gains=GAINS, period=8, progress=False):
    """全（ゲイン × 軌道）を同時に全長進める。戻り値: dict（Q, QM, U は (B, T)、len (B,)、diverged (B,)、index [(gain_i, ref_i)]）。"""
    fresh(step_fn)
    index = [(gi, ri) for gi in range(len(gains)) for ri in range(len(refs))]
    B = len(index)
    K = np.array([len(refs[ri]["qref"]) for _, ri in index])
    T = int(K.max())
    QMR, DQMR, FF = (np.zeros((B, T)) for _ in range(3))
    cache = {}
    for ri, r in enumerate(refs):
        cache[ri] = flex_reference({"qref": r["qref"]}, p, flex, dt)
    for b, (_, ri) in enumerate(index):
        qmr, dqmr, ff = cache[ri]
        n = K[b]
        QMR[b, :n], DQMR[b, :n], FF[b, :n] = qmr, dqmr, ff
        QMR[b, n:], DQMR[b, n:], FF[b, n:] = qmr[-1], 0.0, ff[-1]
    QMR, DQMR, FF = (torch.from_numpy(a) for a in (QMR, DQMR, FF))
    Kp = torch.tensor([p["M"] * gains[gi][0] ** 2 for gi, _ in index], dtype=torch.float64)
    Kd = torch.tensor([2 * gains[gi][1] * p["M"] * gains[gi][0] for gi, _ in index], dtype=torch.float64)
    q0 = torch.tensor([refs[ri]["qref"][0] for _, ri in index], dtype=torch.float64)
    dq0 = torch.zeros(B, dtype=torch.float64)
    # 初期状態: q = qref[0]、dq = dqref[0]（= 0）、ねじり = qmRef[0] − qref[0]、dqm = dq
    tw0 = torch.tensor([cache[ri][0][0] - refs[ri]["qref"][0] for _, ri in index], dtype=torch.float64)
    st = (q0, dq0, q0 + tw0, dq0.clone())
    Q = torch.full((B, T), float("nan"), dtype=torch.float64)
    QM, U = Q.clone(), Q.clone()
    u_hold = torch.zeros(B, dtype=torch.float64)
    u_pend = torch.zeros(B, dtype=torch.float64)
    alive = torch.ones(B, dtype=torch.bool)
    for k in range(T):
        q, dq, qm, dqm = st
        if k % period == 0:
            u_hold = u_pend                                            # 1 サンプル前の PD 出力を保持
            u_pend = Kp * (QMR[:, k] - qm) + Kd * (DQMR[:, k] - dqm)   # 今のサンプルの PD（次のサンプルから出力）
        tau = (FF[:, k] + u_hold).clamp(-p["tauPeak"], p["tauPeak"])
        Q[:, k], QM[:, k], U[:, k] = q, qm, tau
        new = step_fn(q, dq, qm, dqm, tau)
        ok = torch.isfinite(new[0]) & (new[0].abs() < 10) & torch.isfinite(new[2]) & ((new[2] - new[0]).abs() < 1.0)
        alive = alive & ok
        st = tuple(torch.where(alive, x, s) for x, s in zip(new, st))   # 発散した軌跡は止める
        if progress and k % 10000 == 0:
            print(f"  C4 step {k}/{T}", flush=True)
    return {"Q": Q.numpy(), "QM": QM.numpy(), "U": U.numpy(), "len": K, "diverged": (~alive).numpy(), "index": index}


def metrics(res, refs, gains=GAINS, dt=1 / 8000):
    """ゲインごとの指標（軌道平均）。tools/j2_flex_eval.m と同じ定義: trackRmsDeg（軌道区間）、settleTwist（軌道終了 +0.15 s 以降、最終値からの偏差の RMS）。"""
    rows = []
    for gi, (wn, z) in enumerate(gains):
        tr, se, tr_ms, tmax, dyn, div = [], [], [], [], [], False
        for b, (g, ri) in enumerate(res["index"]):
            if g != gi:
                continue
            r = refs[ri]; n = res["len"][b]
            q, qm, u = res["Q"][b, :n], res["QM"][b, :n], res["U"][b, :n]
            t = np.arange(n) * dt
            div = div or bool(res["diverged"][b]) or not np.isfinite(q).all()
            tw = qm - q
            inTraj = t <= r["tEnd"]
            win = t >= r["tEnd"] + 0.15
            last = t >= t[-1] - 0.05
            dev = tw[win] - np.mean(tw[last])
            tr.append(np.degrees(np.sqrt(np.mean((q[inTraj] - r["qref"][inTraj]) ** 2))))
            se.append(np.sqrt(np.mean(dev ** 2)))
            tr_ms.append(np.sqrt(np.mean(u ** 2))); tmax.append(np.max(np.abs(u)))
        rows.append({"wn": wn, "zeta": z, "trackRmsDeg": float(np.mean(tr)), "settleTwist": float(np.mean(se)),
                     "tauRms": float(np.mean(tr_ms)), "tauMax": float(np.mean(tmax)), "diverged": div})
    return rows


def add_J(rows, norm=NORM_GAIN):
    """J = 追従と整定ねじりを、(30, 1) の値で割って平均。"""
    ref = next(r for r in rows if (r["wn"], r["zeta"]) == norm)
    for r in rows:
        r["J"] = 0.5 * (r["trackRmsDeg"] / ref["trackRmsDeg"] + r["settleTwist"] / ref["settleTwist"])
        r["unstable"] = bool(r["diverged"] or r["settleTwist"] >= UNSTABLE_SETTLE)
    return rows


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def compare(model_rows, truth_rows):
    """(c3): (i) 安定／不安定の判定の一致数、(ii) 真値で安定なゲインでの J・追従の Spearman。"""
    tm = {(r["wn"], r["zeta"]): r for r in truth_rows}
    match = [bool(r["unstable"]) == bool(tm[(r["wn"], r["zeta"])]["unstable"]) for r in model_rows]
    stable = [i for i, r in enumerate(model_rows) if not tm[(r["wn"], r["zeta"])]["unstable"]]
    sJ = spearman([model_rows[i]["J"] for i in stable], [tm[(model_rows[i]["wn"], model_rows[i]["zeta"])]["J"] for i in stable])
    sT = spearman([model_rows[i]["trackRmsDeg"] for i in stable],
                  [tm[(model_rows[i]["wn"], model_rows[i]["zeta"])]["trackRmsDeg"] for i in stable])
    return {"n_gains": len(model_rows), "n_class_match": int(sum(match)), "class_pass": bool(all(match)),
            "mismatch": [(r["wn"], r["zeta"]) for r, m in zip(model_rows, match) if not m],
            "n_stable_truth": len(stable), "spearman_J": sJ, "spearman_track": sT, "spearman_pass": bool(sJ >= 0.9)}
