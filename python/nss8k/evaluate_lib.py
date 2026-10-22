"""評価の部品: 1 ステップ誤差、窓の開ループ、PTP の閉ループ、スループット、ゲート判定。docs/training_8khz.md §4。"""
import time

import numpy as np
import torch

from .data import group_of, sequences
from .physics import Physics

# ゲート（暫定を含む）。(a) は解析モデルの RMS に対する倍率、(b) は度
GATES = {"a_ratio": {"A": 1.0, "B": 0.5}, "b_deg": {"A": 1.0, "B": 0.3}, "c_deg": 0.1, "d_steps_per_s": 145000.0}
# PD の設計（MATLAB の buildJ2ClosedLoop の既定: wn = 30 rad/s、ζ = 1、Kp = M·wn²、Kd = 2·ζ·M·wn、±tauPeak で飽和）
PD_WN, PD_ZETA = 30.0, 1.0


def _rms(a):
    return float(np.sqrt(np.mean(np.square(a)))) if np.size(a) else float("nan")


# ---------------------------------------------------------------- 1 ステップ誤差
@torch.no_grad()
def predict_ddq(model, X_raw, bs=262144):
    """X_raw = [q dq tau]（N, 3）→ モデルの ddq [rad/s²]（N,）。ロールアウトと同じ経路（物理量から特徴量を作る）。"""
    model.eval()
    R = torch.as_tensor(X_raw, dtype=torch.float64)
    return torch.cat([model.ddq(R[i:i + bs, 0], R[i:i + bs, 1], R[i:i + bs, 2]) for i in range(0, len(R), bs)]).numpy()


def one_step_table(model, d, split):
    """全体・群別の 1 ステップ ddq RMSE [rad/s²]（モデル / 解析モデル / 0 予測）。"""
    s = d["splits"][split]
    y = s["Y"][:, 0].astype(np.float64)
    pred = predict_ddq(model, s["X_raw"])
    ana = s["a_phys"][:, 0].astype(np.float64)
    grp = np.array([group_of(d["scenarios"]["pattern"][k]) for k in s["scenario_id"]])
    sel = {"all": np.ones(len(y), bool)}
    for g in sorted(set(grp)):
        sel[f"group:{g}"] = grp == g
    table = {}
    for name, m in sel.items():
        table[name] = {"n": int(m.sum()), "target_std": float(np.std(y[m])), "model": _rms(pred[m] - y[m]),
                       "analytic": _rms(ana[m] - y[m]), "zero": _rms(y[m])}
        table[name]["ratio_to_analytic"] = table[name]["model"] / max(table[name]["analytic"], 1e-12)
    return table


# ---------------------------------------------------------------- ロールアウト
@torch.no_grad()
def rollout(step_fn, q0, dq0, tau):
    """q0, dq0: (B,)、tau: (B, H)。H ステップ自己回帰で進め、(B, H+1) の q, dq を返す。"""
    B, H = tau.shape
    q = torch.empty(B, H + 1, dtype=q0.dtype)
    dq = torch.empty(B, H + 1, dtype=q0.dtype)
    q[:, 0], dq[:, 0] = q0, dq0
    for k in range(H):
        q[:, k + 1], dq[:, k + 1] = step_fn(q[:, k], dq[:, k], tau[:, k])
    return q, dq


def step_fns(model, phys):
    return {"model": model.step_physical, "analytic": phys.step, "persistence": lambda q, dq, tau: (q, dq)}


def make_windows(seqs, horizon, stride):
    """各シナリオから、開始点が stride 刻みで長さ horizon の窓を切り出す（q, dq は horizon+1 点、tau は horizon 点）。"""
    W = {"q": [], "dq": [], "tau": [], "group": [], "name": []}
    for s in seqs:
        K = len(s["tau"])
        for a in range(0, K - horizon, stride):          # 窓の最終点 a+horizon が行の範囲内
            W["q"].append(s["q"][a:a + horizon + 1]); W["dq"].append(s["dq"][a:a + horizon + 1])
            W["tau"].append(s["tau"][a:a + horizon]); W["group"].append(s["group"]); W["name"].append(s["name"])
    if not W["q"]:
        return None
    return {k: (np.array(v) if k != "name" else v) for k, v in W.items()}


def window_rollouts(model, phys, d, split, horizon=8000, stride=2000, max_windows=0, steps=(1, 10, 100, 1000, 2000, 4000, 8000)):
    """窓の開ループ（記録トルクを再生）。ステップごとの θ の RMSE [度]、最終ステップの群別 RMSE、発散の割合。"""
    W = make_windows(sequences(d, split), horizon, stride)
    if W is None:
        return None
    if max_windows and len(W["q"]) > max_windows:
        pick = np.sort(np.random.RandomState(0).permutation(len(W["q"]))[:max_windows])
        W = {k: ([v[i] for i in pick] if k == "name" else v[pick]) for k, v in W.items()}
    q0, dq0, tau = (torch.from_numpy(W[k][:, 0] if k != "tau" else W[k]) for k in ("q", "dq", "tau"))
    out = {"n_windows": int(len(W["q"])), "horizon": horizon, "methods": {}}
    grp = np.array(W["group"])
    for name, fn in step_fns(model, phys).items():
        q, _ = rollout(fn, q0, dq0, tau)
        q = q.numpy()
        e = q - W["q"]
        bad = ~np.isfinite(q).all(axis=1) | (np.abs(np.nan_to_num(q, nan=1e9)).max(axis=1) > 10)
        rm = np.sqrt(np.nanmean(e ** 2, axis=0))
        out["methods"][name] = {
            "rmse_q_deg": {str(h): float(np.degrees(rm[h])) for h in steps if h < len(rm)},
            "diverged_fraction": float(bad.mean()),
            "by_group_final_deg": {g: float(np.degrees(np.sqrt(np.nanmean(e[grp == g][:, -1] ** 2)))) for g in sorted(set(grp))},
            "curve_q_deg": np.degrees(rm).tolist()}
    return out


# ---------------------------------------------------------------- PTP の閉ループ（8 kHz、全軌跡をバッチで）
def pd_ff_reference(seq, p, dt):
    """参照角 qref（K 点）から、dqref とフィードフォワードトルクを作る。
    τ_ff = M·ddq_ref − mgL·sin q_ref + Bv·dq_ref + Fc·tanh(dq_ref/ε)（MATLAB の j2Feedforward と同じ）。
    qref は倍精度の 8 kHz 記録（5 次多項式で滑らか）なので、数値微分で十分な精度が出る。"""
    qr = seq["qref"]
    dqr = np.gradient(qr, dt)
    ddqr = np.gradient(dqr, dt)
    tau_ff = p["M"] * ddqr - p["mgL"] * np.sin(qr) + p["Bv"] * dqr + p["Fc"] * np.tanh(dqr / p["eps"])
    return qr, dqr, tau_ff


@torch.no_grad()
def closed_loop_batch(step_fn, seqs, p, dt):
    """制御器（PD＋FF）を閉ループに入れ、プラントを step_fn に置き換えて、全軌跡を同時に全長進める。

    開ループの再生は θ=0 まわりの不安定性（成長率 約 3.5/s）で正確なモデルでも発散するため使わない。
    戻り値: Q (B, T+1)（NaN 埋め）、U (B, T)。T は最長の軌跡のステップ数。
    """
    B = len(seqs)
    K = np.array([len(s["tau"]) for s in seqs])
    T = int(K.max())
    QR, DQR, FF = (np.zeros((B, T)) for _ in range(3))
    for i, s in enumerate(seqs):
        qr, dqr, ff = pd_ff_reference(s, p, dt)
        QR[i, :K[i]], DQR[i, :K[i]], FF[i, :K[i]] = qr, dqr, ff
        QR[i, K[i]:], DQR[i, K[i]:], FF[i, K[i]:] = qr[-1], 0.0, ff[-1]
    QR, DQR, FF = (torch.from_numpy(a) for a in (QR, DQR, FF))
    Kp, Kd = p["M"] * PD_WN ** 2, 2 * PD_ZETA * p["M"] * PD_WN
    q = torch.tensor([s["q"][0] for s in seqs], dtype=torch.float64)
    dq = torch.tensor([s["dq"][0] for s in seqs], dtype=torch.float64)
    Q = torch.full((B, T + 1), float("nan"), dtype=torch.float64)
    U = torch.full((B, T), float("nan"), dtype=torch.float64)
    Q[:, 0] = q
    alive = torch.ones(B, dtype=torch.bool)
    for k in range(T):
        tau = (FF[:, k] + Kp * (QR[:, k] - q) + Kd * (DQR[:, k] - dq)).clamp(-p["tauPeak"], p["tauPeak"])
        U[:, k] = torch.where(alive, tau, torch.full_like(tau, float("nan")))
        q1, dq1 = step_fn(q, dq, tau)
        alive = alive & torch.isfinite(q1) & (q1.abs() < 10)
        q = torch.where(alive, q1, torch.full_like(q1, float("nan")))
        dq = torch.where(alive, dq1, torch.zeros_like(dq1))
        Q[:, k + 1] = q
    return Q.numpy(), U.numpy()


def closed_loop_benchmark(model, phys, d, split="benchmark"):
    seqs = [s for s in sequences(d, split) if np.isfinite(s["qref"]).all()]
    p = d["physics"]
    fns = {k: v for k, v in step_fns(model, phys).items() if k != "persistence"}
    rows = []
    res = {name: closed_loop_batch(fn, seqs, p, d["dt"]) for name, fn in fns.items()}
    for i, s in enumerate(seqs):
        K = len(s["tau"])
        row = {"name": s["name"], "pattern": s["pattern"], "n_steps": K,
               "truth_track_rmse_deg": float(np.degrees(_rms(s["q"] - s["qref"])))}
        for name, (Q, U) in res.items():
            q = Q[i, :K]                                       # 行 k の状態に対応する予測（k = 0..K-1）
            row[name] = {"vs_truth_rmse_q_deg": float(np.degrees(np.sqrt(np.nanmean((q - s["q"]) ** 2)))),
                         "vs_truth_max_q_deg": float(np.degrees(np.nanmax(np.abs(q - s["q"])))),
                         "track_rmse_deg": float(np.degrees(np.sqrt(np.nanmean((q - s["qref"]) ** 2)))),
                         "tau_rmse_N_m": float(np.sqrt(np.nanmean((U[i, :K] - s["tau"]) ** 2))),
                         "diverged": bool(np.isnan(Q[i, :K]).any())}
        rows.append(row)
    return rows, seqs, res


def controller_check(seqs, p, dt):
    """Python の PD＋FF を真値の状態に当てたトルクと記録トルクの差（RMS、N·m）。制御器の再現の確認。"""
    Kp, Kd = p["M"] * PD_WN ** 2, 2 * PD_ZETA * p["M"] * PD_WN
    out = []
    for s in seqs:
        qr, dqr, ff = pd_ff_reference(s, p, dt)
        tau = np.clip(ff + Kp * (qr - s["q"]) + Kd * (dqr - s["dq"]), -p["tauPeak"], p["tauPeak"])
        out.append({"name": s["name"], "rms_N_m": _rms(tau - s["tau"]), "tau_rms_N_m": _rms(s["tau"])})
    return out


# ---------------------------------------------------------------- スループット
@torch.no_grad()
def throughput(model, phys, d, batch=1024, steps=2000, repeats=2, seed=0):
    """バッチ並列のロールアウト速度 [ステップ/s]（バッチ × ステップ数 / 実時間）と、1 サンプル推論の時間 [µs]。"""
    s = d["splits"]["test"] if "test" in d["splits"] else next(iter(d["splits"].values()))
    R = s["X_raw"].astype(np.float64)
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, len(R) - steps, size=batch)
    q0, dq0 = torch.from_numpy(R[idx, 0]), torch.from_numpy(R[idx, 1])
    tau = torch.from_numpy(np.stack([R[i:i + steps, 2] for i in idx]))
    out = {}
    for name, fn in step_fns(model, phys).items():
        if name == "persistence":
            continue
        best = float("inf")
        for _ in range(repeats):
            t = time.perf_counter()
            rollout(fn, q0, dq0, tau)
            best = min(best, time.perf_counter() - t)
        out[name] = {"steps_per_s": batch * steps / best}
    q1, dq1, t1 = q0[:1], dq0[:1], tau[:1, 0]
    for _ in range(200):
        model.step_physical(q1, dq1, t1)
    ts = []
    for _ in range(3000):
        t = time.perf_counter()
        model.step_physical(q1, dq1, t1)
        ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1e6
    out["model"]["single_step_us"] = {"median": float(np.median(ts)), "p99": float(np.percentile(ts, 99)),
                                      "p99_9": float(np.percentile(ts, 99.9))}
    out["batch"], out["steps"] = batch, steps
    return out


# ---------------------------------------------------------------- ゲート判定
def judge(kind, one_step, windows, closed, thr, gates=GATES):
    """ゲート (a)(b)(c)(d) の合否。kind は 'A' か 'B'。one_step は {split: 表}、windows は window_rollouts の出力。"""
    g = {}
    ratios = {sp: t["all"]["ratio_to_analytic"] for sp, t in one_step.items() if sp in ("test", "benchmark")}
    g["a"] = {"pass": all(r <= gates["a_ratio"][kind] for r in ratios.values()), "value": ratios, "limit": gates["a_ratio"][kind]}
    if windows is not None:
        h = str(windows["horizon"])
        v = windows["methods"]["model"]["rmse_q_deg"][h]
        g["b"] = {"pass": v <= gates["b_deg"][kind], "value": v, "analytic": windows["methods"]["analytic"]["rmse_q_deg"][h],
                  "limit": gates["b_deg"][kind]}
    nd = sum(r["model"]["diverged"] for r in closed)
    worst = max(r["model"]["vs_truth_rmse_q_deg"] for r in closed)
    g["c"] = {"pass": nd == 0 and worst <= gates["c_deg"], "n_diverged": int(nd), "worst_rmse_deg": worst,
              "mean_rmse_deg": float(np.mean([r["model"]["vs_truth_rmse_q_deg"] for r in closed])), "limit": gates["c_deg"],
              "analytic_worst_rmse_deg": max(r["analytic"]["vs_truth_rmse_q_deg"] for r in closed)}
    g["d"] = {"pass": thr["model"]["steps_per_s"] >= gates["d_steps_per_s"], "value": thr["model"]["steps_per_s"],
              "limit": gates["d_steps_per_s"], "analytic_steps_per_s": thr["analytic"]["steps_per_s"]}
    return g
