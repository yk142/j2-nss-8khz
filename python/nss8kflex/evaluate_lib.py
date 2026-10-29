"""4 状態 NSS の評価の部品: 1 ステップ誤差、窓の開ループ、PTP の閉ループ、スループット、ゲート判定。
docs/requirements_nss_flex.md §5、docs/plan_nss_flex_train.md §2。"""
import time

import numpy as np
import torch

from .data import group_of, sequences
from .physics import FlexPhysics

# ゲート（要件 §5）。(a) は 2 出力それぞれの RMSE の、解析モデルに対する倍率。(b) は 4,000 ステップ窓の θ RMSE [°] と、
# ねじり RMSE の解析モデルに対する倍率。(c) は PTP 閉ループの真値との θ RMSE [°]。(d) はバッチロールアウトの速度。
GATES = {"a_test_ratio": 1.0, "b_deg_4000": 0.4, "b_twist_ratio": 1.0, "c_deg": 0.1, "d_steps_per_s": 145000.0}
PD_WN, PD_ZETA = 30.0, 1.0


def fresh(step_fn):
    """状態（直前の加速度）を持つモデルの step_fn なら reset() を呼ぶ。ロールアウトの先頭で呼び、前の実行の持ち越しを防ぐ。"""
    m = getattr(step_fn, "__self__", None)
    if hasattr(m, "reset"):
        m.reset()
    return step_fn


def _rms(a):
    return float(np.sqrt(np.mean(np.square(a)))) if np.size(a) else float("nan")


# ---------------------------------------------------------------- 1 ステップ誤差
@torch.no_grad()
def predict_acc(model, X_raw, bs=262144):
    """X_raw = [q dq tw dtw tau]（N, 5）→ モデルの (ddq, ddqm) [rad/s²]（N, 2）。"""
    model.eval()
    R = torch.as_tensor(X_raw, dtype=torch.float64)
    out = []
    for i in range(0, len(R), bs):
        r = R[i:i + bs]
        out.append(torch.stack(model.accel(r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 4]), dim=1))
    return torch.cat(out).numpy()


def one_step_table(model, d, split):
    """全体・群別の 1 ステップ加速度 RMSE [rad/s²]。各出力（ddq, ddqm）ごと。"""
    s = d["splits"][split]
    y = s["Y"].astype(np.float64)
    pred = predict_acc(model, s["X_raw"])
    ana = s["a_phys"].astype(np.float64)
    grp = np.array([group_of(d["scenarios"]["pattern"][k]) for k in s["scenario_id"]])
    sel = {"all": np.ones(len(y), bool)}
    for g in sorted(set(grp)):
        sel[f"group:{g}"] = grp == g
    table = {}
    for name, m in sel.items():
        row = {"n": int(m.sum())}
        for j, out in enumerate(("ddq", "ddqm")):
            e = {"target_std": float(np.std(y[m, j])), "model": _rms(pred[m, j] - y[m, j]),
                 "analytic": _rms(ana[m, j] - y[m, j]), "zero": _rms(y[m, j])}
            e["ratio_to_analytic"] = e["model"] / max(e["analytic"], 1e-12)
            row[out] = e
        table[name] = row
    return table


# ---------------------------------------------------------------- ロールアウト
def step_fns(model, phys):
    return {"model": model.step_physical, "analytic": phys.step}


@torch.no_grad()
def rollout(step_fn, s0, tau):
    """s0 = (q, dq, qm, dqm) それぞれ (B,)、tau: (B, H)。H ステップ自己回帰で進め、(B, H+1) の 4 配列を返す。"""
    fresh(step_fn)
    B, H = tau.shape
    S = [torch.empty(B, H + 1, dtype=s0[0].dtype) for _ in range(4)]
    cur = tuple(s0)
    for j in range(4):
        S[j][:, 0] = cur[j]
    for k in range(H):
        cur = step_fn(*cur, tau[:, k])
        for j in range(4):
            S[j][:, k + 1] = cur[j]
    return S


def make_windows(seqs, horizon, stride):
    keys = ("q", "dq", "qm", "dqm")
    W = {k: [] for k in keys + ("tau", "group")}
    for s in seqs:
        K = len(s["tau"])
        for a in range(0, K - horizon, stride):
            for k in keys:
                W[k].append(s[k][a:a + horizon + 1])
            W["tau"].append(s["tau"][a:a + horizon]); W["group"].append(s["group"])
    return {k: np.array(v) for k, v in W.items()} if W["q"] else None


def window_rollouts(model, phys, d, split, horizon=4000, stride=2000, max_windows=0,
                    steps=(1, 10, 100, 1000, 2000, 4000)):
    """窓の開ループ（記録トルクを再生）。ステップごとの θ の RMSE [度] とねじりの RMSE [rad]、発散の割合。"""
    W = make_windows(sequences(d, split), horizon, stride)
    if W is None:
        return None
    if max_windows and len(W["q"]) > max_windows:
        pick = np.sort(np.random.RandomState(0).permutation(len(W["q"]))[:max_windows])
        W = {k: v[pick] for k, v in W.items()}
    s0 = tuple(torch.from_numpy(W[k][:, 0]) for k in ("q", "dq", "qm", "dqm"))
    tau = torch.from_numpy(W["tau"])
    out = {"n_windows": int(len(W["q"])), "horizon": horizon, "methods": {}}
    grp = np.array(W["group"])
    for name, fn in step_fns(model, phys).items():
        q, _, qm, _ = (x.numpy() for x in rollout(fn, s0, tau))
        eq = q - W["q"]
        etw = (qm - q) - (W["qm"] - W["q"])
        bad = ~np.isfinite(q).all(axis=1) | (np.abs(np.nan_to_num(q, nan=1e9)).max(axis=1) > 10)
        rq, rt = np.sqrt(np.nanmean(eq ** 2, axis=0)), np.sqrt(np.nanmean(etw ** 2, axis=0))
        out["methods"][name] = {
            "rmse_q_deg": {str(h): float(np.degrees(rq[h])) for h in steps if h < len(rq)},
            "rmse_twist_rad": {str(h): float(rt[h]) for h in steps if h < len(rt)},
            "diverged_fraction": float(bad.mean()),
            "by_group_final_deg": {g: float(np.degrees(np.sqrt(np.nanmean(eq[grp == g][:, -1] ** 2)))) for g in sorted(set(grp))},
            "curve_q_deg": np.degrees(rq).tolist(), "curve_twist_rad": rt.tolist()}
    return out


# ---------------------------------------------------------------- PTP の閉ループ（8 kHz、モータ角 PD＋FF）
def flex_reference(seq, p, flex, dt):
    """リンク角参照 qref（K 点）から、j2FeedforwardFlex と同じ FF トルクとモータ角の参照を作る。"""
    qr = seq["qref"]
    dqr = np.gradient(qr, dt)
    ddqr = np.gradient(dqr, dt)
    tau_ff = p["M"] * ddqr - p["mgL"] * np.sin(qr) + p["Bv"] * dqr + p["Fc"] * np.tanh(dqr / p["eps"])
    twist = (tau_ff - flex["Jm"] * ddqr) / flex["k"]
    return qr + twist, dqr + np.gradient(twist, dt), tau_ff


@torch.no_grad()
def closed_loop_batch(step_fn, seqs, p, flex, dt):
    """モータ角 PD＋FF を閉ループに入れ、プラントを step_fn に置き換えて、全軌跡を同時に全長進める。
    戻り値: Q, QM (B, T+1)（NaN 埋め）、U (B, T)。"""
    fresh(step_fn)
    B = len(seqs)
    K = np.array([len(s["tau"]) for s in seqs])
    T = int(K.max())
    QMR, DQMR, FF = (np.zeros((B, T)) for _ in range(3))
    for i, s in enumerate(seqs):
        qmr, dqmr, ff = flex_reference(s, p, flex, dt)
        QMR[i, :K[i]], DQMR[i, :K[i]], FF[i, :K[i]] = qmr, dqmr, ff
        QMR[i, K[i]:], DQMR[i, K[i]:], FF[i, K[i]:] = qmr[-1], 0.0, ff[-1]
    QMR, DQMR, FF = (torch.from_numpy(a) for a in (QMR, DQMR, FF))
    Kp, Kd = p["M"] * PD_WN ** 2, 2 * PD_ZETA * p["M"] * PD_WN
    st = tuple(torch.tensor([s[k][0] for s in seqs], dtype=torch.float64) for k in ("q", "dq", "qm", "dqm"))
    Q = torch.full((B, T + 1), float("nan"), dtype=torch.float64)
    QM = Q.clone()
    U = torch.full((B, T), float("nan"), dtype=torch.float64)
    Q[:, 0], QM[:, 0] = st[0], st[2]
    alive = torch.ones(B, dtype=torch.bool)
    for k in range(T):
        q, dq, qm, dqm = st
        tau = (FF[:, k] + Kp * (QMR[:, k] - qm) + Kd * (DQMR[:, k] - dqm)).clamp(-p["tauPeak"], p["tauPeak"])
        U[:, k] = torch.where(alive, tau, torch.full_like(tau, float("nan")))
        new = step_fn(q, dq, qm, dqm, tau)
        alive = alive & torch.isfinite(new[0]) & (new[0].abs() < 10) & torch.isfinite(new[2]) & ((new[2] - new[0]).abs() < 1.0)
        st = tuple(torch.where(alive, x, torch.zeros_like(x)) for x in new)
        Q[:, k + 1] = torch.where(alive, new[0], torch.full_like(new[0], float("nan")))
        QM[:, k + 1] = torch.where(alive, new[2], torch.full_like(new[2], float("nan")))
    return Q.numpy(), QM.numpy(), U.numpy()


def closed_loop_benchmark(model, phys, d, split="benchmark"):
    seqs = [s for s in sequences(d, split) if np.isfinite(s["qref"]).all()]
    p, flex = d["physics"], d["flex"]
    res = {name: closed_loop_batch(fn, seqs, p, flex, d["dt"]) for name, fn in step_fns(model, phys).items()}
    rows = []
    for i, s in enumerate(seqs):
        K = len(s["tau"])
        tw_truth = s["qm"] - s["q"]
        row = {"name": s["name"], "pattern": s["pattern"], "n_steps": K,
               "truth_track_rmse_deg": float(np.degrees(_rms(s["q"] - s["qref"]))), "truth_twist_rms_rad": _rms(tw_truth)}
        for name, (Q, QM, U) in res.items():
            q, qm = Q[i, :K], QM[i, :K]
            row[name] = {"vs_truth_rmse_q_deg": float(np.degrees(np.sqrt(np.nanmean((q - s["q"]) ** 2)))),
                         "vs_truth_max_q_deg": float(np.degrees(np.nanmax(np.abs(q - s["q"])))),
                         "vs_truth_rmse_twist_rad": float(np.sqrt(np.nanmean(((qm - q) - tw_truth) ** 2))),
                         "twist_rms_rad": float(np.sqrt(np.nanmean((qm - q) ** 2))),
                         "track_rmse_deg": float(np.degrees(np.sqrt(np.nanmean((q - s["qref"]) ** 2)))),
                         "tau_rmse_N_m": float(np.sqrt(np.nanmean((U[i, :K] - s["tau"]) ** 2))),
                         "diverged": bool(np.isnan(Q[i, :K]).any())}
        rows.append(row)
    return rows, seqs, res


def controller_check(seqs, p, flex, dt):
    """Python の PD＋FF を真値の状態に当てたトルクと記録トルクの差（RMS、N·m）。制御器の再現の確認。"""
    Kp, Kd = p["M"] * PD_WN ** 2, 2 * PD_ZETA * p["M"] * PD_WN
    out = []
    for s in seqs:
        qmr, dqmr, ff = flex_reference(s, p, flex, dt)
        tau = np.clip(ff + Kp * (qmr - s["qm"]) + Kd * (dqmr - s["dqm"]), -p["tauPeak"], p["tauPeak"])
        out.append({"name": s["name"], "rms_N_m": _rms(tau - s["tau"]), "tau_rms_N_m": _rms(s["tau"])})
    return out


# ---------------------------------------------------------------- スループット
@torch.no_grad()
def throughput(model, phys, d, batch=1024, steps=2000, repeats=10, seed=0):
    """バッチ並列のロールアウト速度 [ステップ/s]。ウォームアップ後、repeats 回のうち最速で判定（中央値・全回も記録）。"""
    s = d["splits"]["test"] if "test" in d["splits"] else next(iter(d["splits"].values()))
    R = s["X_raw"].astype(np.float64)
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, len(R) - steps, size=batch)
    s0 = (torch.from_numpy(R[idx, 0]), torch.from_numpy(R[idx, 1]),
          torch.from_numpy(R[idx, 0] + R[idx, 2]), torch.from_numpy(R[idx, 1] + R[idx, 3]))
    tau = torch.from_numpy(np.stack([R[i:i + steps, 4] for i in idx]))
    out = {}
    for name, fn in step_fns(model, phys).items():
        rollout(fn, s0, tau[:, :200])
        ts = []
        for _ in range(repeats):
            t = time.perf_counter()
            rollout(fn, s0, tau)
            ts.append(time.perf_counter() - t)
        out[name] = {"steps_per_s": batch * steps / float(np.min(ts)),
                     "steps_per_s_median": batch * steps / float(np.median(ts)),
                     "steps_per_s_runs": [batch * steps / x for x in ts]}
    one = tuple(x[:1] for x in s0)
    t1 = tau[:1, 0]
    for _ in range(200):
        model.step_physical(*one, t1)
    ts = []
    for _ in range(3000):
        t = time.perf_counter()
        model.step_physical(*one, t1)
        ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1e6
    out["model"]["single_step_us"] = {"median": float(np.median(ts)), "p99": float(np.percentile(ts, 99)),
                                      "p99_9": float(np.percentile(ts, 99.9))}
    out["batch"], out["steps"] = batch, steps
    return out


# ---------------------------------------------------------------- ゲート判定
def judge(one_step, windows, closed, thr, gates=GATES):
    g = {}
    r = one_step["test"]["all"]
    ratios = [r["ddq"]["ratio_to_analytic"], r["ddqm"]["ratio_to_analytic"]]
    g["a"] = {"pass": max(ratios) <= gates["a_test_ratio"], "ratios": ratios, "limit": gates["a_test_ratio"]}
    if windows is not None:
        m, an = windows["methods"]["model"], windows["methods"]["analytic"]
        h = str(windows["horizon"])
        tw_ratio = m["rmse_twist_rad"][h] / max(an["rmse_twist_rad"][h], 1e-12)
        g["b"] = {"pass": m["rmse_q_deg"][h] <= gates["b_deg_4000"] and tw_ratio <= gates["b_twist_ratio"],
                  "q_deg": m["rmse_q_deg"][h], "analytic_q_deg": an["rmse_q_deg"][h], "limit_q_deg": gates["b_deg_4000"],
                  "twist_rad": m["rmse_twist_rad"][h], "analytic_twist_rad": an["rmse_twist_rad"][h], "twist_ratio": tw_ratio,
                  "limit_twist_ratio": gates["b_twist_ratio"]}
    nd = sum(x["model"]["diverged"] for x in closed)
    worst = max(x["model"]["vs_truth_rmse_q_deg"] for x in closed)
    g["c"] = {"pass": nd == 0 and worst <= gates["c_deg"], "n_diverged": int(nd), "worst_rmse_deg": worst,
              "mean_rmse_deg": float(np.mean([x["model"]["vs_truth_rmse_q_deg"] for x in closed])), "limit": gates["c_deg"],
              "analytic_worst_rmse_deg": max(x["analytic"]["vs_truth_rmse_q_deg"] for x in closed),
              "worst_twist_rmse_rad": max(x["model"]["vs_truth_rmse_twist_rad"] for x in closed)}
    g["d"] = {"pass": thr["model"]["steps_per_s"] >= gates["d_steps_per_s"], "value": thr["model"]["steps_per_s"],
              "limit": gates["d_steps_per_s"], "analytic_steps_per_s": thr["analytic"]["steps_per_s"]}
    return g
