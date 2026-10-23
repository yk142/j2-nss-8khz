"""(c2) ゲイン違いの再現性: 真値の読み込み、ゲイン付き閉ループ、指標・コスト・順位相関。docs/requirements_c2.md。"""
import glob
import os

import h5py
import numpy as np
import torch

SETTLE_DEG = 0.05            # 整定の閾値 [°]
HOLD_MIN_S = 0.15            # 停留とみなす最短の長さ [s]（PTP は各到達点で 0.2 s 停留）
GAINS = [(wn, z) for wn in (15, 30, 60) for z in (0.7, 1.0, 1.5)]


def dir_name(cond, wn, zeta):
    return f"{cond}_wn{wn}_z{round(10 * zeta)}"


def load_truth(root, cond, wn, zeta):
    """data/c2/<条件>_wn<wn>_z<ζ×10>/*.mat（v7.3）を読み、軌道名順のリストで返す。"""
    out = []
    for fn in sorted(glob.glob(os.path.join(root, dir_name(cond, wn, zeta), "*.mat"))):
        with h5py.File(fn, "r") as f:
            e = {k: np.ravel(f[k][()]).astype(np.float64) for k in ("q", "dq", "tau", "qref", "dqref", "tauFf", "tauD")}
        e["name"] = os.path.splitext(os.path.basename(fn))[0]
        out.append(e)
    return out


@torch.no_grad()
def closed_loop(step_fn, trials, p, dt, wn, zeta):
    """PD（wn, ζ）＋保存済みの FF・外乱でプラントを step_fn に置き換え、全軌道を同時に進める。
    τ = clip(τ_ff + τ_dist + Kp·(q_ref − q) + Kd·(dq_ref − dq), ±τ_peak)（Simulink の構成と同じ）。
    戻り値: Q (B, T+1)（発散以降は NaN、軌道長の先は参照の最終値で保持）、U (B, T)、K（各軌道のステップ数）。"""
    B = len(trials)
    K = np.array([len(s["tau"]) - 1 for s in trials])           # N 点 = K+1 → 遷移 K 本
    T = int(K.max())
    QR, DQR, FF, TD = (np.zeros((B, T)) for _ in range(4))
    for i, s in enumerate(trials):
        for a, key in ((QR, "qref"), (DQR, "dqref"), (FF, "tauFf"), (TD, "tauD")):
            a[i, :K[i]] = s[key][:K[i]]
        QR[i, K[i]:], FF[i, K[i]:] = s["qref"][K[i]], s["tauFf"][K[i]]
    QR, DQR, FF, TD = (torch.from_numpy(a) for a in (QR, DQR, FF, TD))
    Kp, Kd = p["M"] * wn ** 2, 2 * zeta * p["M"] * wn
    q = torch.tensor([s["qref"][0] for s in trials], dtype=torch.float64)
    dq = torch.zeros(B, dtype=torch.float64)
    Q = torch.full((B, T + 1), float("nan"), dtype=torch.float64)
    U = torch.full((B, T), float("nan"), dtype=torch.float64)
    Q[:, 0] = q
    alive = torch.ones(B, dtype=torch.bool)
    for k in range(T):
        tau = (FF[:, k] + TD[:, k] + Kp * (QR[:, k] - q) + Kd * (DQR[:, k] - dq)).clamp(-p["tauPeak"], p["tauPeak"])
        U[:, k] = torch.where(alive, tau, torch.full_like(tau, float("nan")))
        q1, dq1 = step_fn(q, dq, tau)
        alive = alive & torch.isfinite(q1) & (q1.abs() < 10)
        q = torch.where(alive, q1, torch.full_like(q1, float("nan")))
        dq = torch.where(alive, dq1, torch.zeros_like(dq1))
        Q[:, k + 1] = q
    return Q.numpy(), U.numpy(), K


def settling_time(err_deg, dqref, dt, thr=SETTLE_DEG, hold_min=HOLD_MIN_S):
    """各停留区間（参照速度がほぼ 0 の連続区間）で、誤差が thr 以内に入り区間の終わりまで保つまでの時間 [s] の平均。
    入らなければ区間の長さ。停留区間が無ければ NaN。"""
    still = np.abs(dqref) < 1e-3
    d = np.diff(np.concatenate([[0], still.astype(int), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    ts = []
    for a, b in zip(starts, ends):
        if (b - a) * dt < hold_min or a == 0:                    # 開始直後の静止は到達ではないので除く
            continue
        bad = np.where(~(np.abs(err_deg[a:b]) <= thr))[0]
        ts.append(((bad[-1] + 1) if len(bad) else 0) * dt)
    return float(np.mean(ts)) if ts else float("nan")


def trial_metrics(Q, U, K, trials, dt):
    """軌道ごとの指標: θ RMSE [°]、τ RMS [N·m]、最大誤差 [°]、整定時間 [s]、発散。"""
    rows = []
    for i, s in enumerate(trials):
        k = K[i]
        e = np.degrees(Q[i, :k + 1] - s["qref"][:k + 1])
        u = U[i, :k]
        rows.append({"name": s["name"], "diverged": bool(np.isnan(Q[i, :k + 1]).any()),
                     "theta_rmse_deg": float(np.sqrt(np.nanmean(e ** 2))), "tau_rms": float(np.sqrt(np.nanmean(u ** 2))),
                     "max_err_deg": float(np.nanmax(np.abs(e))),
                     "settle_s": settling_time(np.nan_to_num(e, nan=1e3), s["dqref"][:k + 1], dt)})
    return rows


def aggregate(rows):
    """12 軌道の平均。発散が 1 本でもあれば diverged = True（指標は NaN を除いた平均）。"""
    return {"theta_rmse_deg": float(np.nanmean([r["theta_rmse_deg"] for r in rows])),
            "tau_rms": float(np.nanmean([r["tau_rms"] for r in rows])),
            "max_err_deg": float(np.nanmean([r["max_err_deg"] for r in rows])),
            "settle_s": float(np.nanmean([r["settle_s"] for r in rows])),
            "n_diverged": int(sum(r["diverged"] for r in rows))}


def cost(agg, ref):
    """J = θ RMSE / ref.θ RMSE + τ RMS / ref.τ RMS。ref は同じ条件の 9 組の真値の中央値。"""
    return agg["theta_rmse_deg"] / ref["theta_rmse_deg"] + agg["tau_rms"] / ref["tau_rms"]


def ranks(x):
    x = np.asarray(x, float)
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    for v in np.unique(x):                                       # 同順位は平均順位
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
