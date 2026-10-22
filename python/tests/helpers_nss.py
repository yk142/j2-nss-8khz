"""8 kHz NSS のテスト用の部品: j2ExportNss8k と同じ構成の小さな HDF5 を、解析モデルで作る。"""
import h5py
import numpy as np

PHYS = {"M": 6.0857, "mgL": 73.25, "Fc": 23.82, "Bv": 1.42, "eps": 0.0130,
        "qMin": -2.7576, "qMax": 1.1345, "qdMax": 5.236, "tauRated": 286.8, "tauPeak": 860.4}
SCALE = {"DTHETA_MAX": 5.236, "TAU_MAX": 860.4, "DDQ_SCALE": 860.4 / 6.0857}
DT = 1.0 / 8000


def accel(q, dq, tau, p=PHYS):
    return (tau + p["mgL"] * np.sin(q) - p["Bv"] * dq - p["Fc"] * np.tanh(dq / p["eps"])) / p["M"]


def simulate(q0, dq0, tau, extra=0.0):
    """解析モデルのオイラーで、トルク列 tau（K 点）から q, dq（K 点。行 k の状態）を作る。次状態は返さない。
    extra は解析モデルに無い加速度（0.5·extra·sin 2q）。学習用データに残差を持たせるために使う。"""
    K = len(tau)
    q, dq = np.empty(K), np.empty(K)
    q[0], dq[0] = q0, dq0
    for k in range(K - 1):
        q[k + 1] = q[k] + DT * dq[k]
        dq[k + 1] = dq[k] + DT * (accel(q[k], dq[k], tau[k]) + 0.5 * extra * np.sin(2 * q[k]))
    return q, dq


def ptp_reference(K, q_from=-1.0, q_to=-0.4, lead=200):
    """5 次多項式の滑らかな参照（開始と終了で速度・加速度 0）。"""
    t = np.clip((np.arange(K) - lead) / (K - 2 * lead), 0, 1)
    s = 10 * t**3 - 15 * t**4 + 6 * t**5
    return q_from + (q_to - q_from) * s


def closed_loop_truth(K, wn=30.0, zeta=1.0):
    """PD＋FF の閉ループを解析モデルのオイラーで回した、PTP の真値（q, dq, tau, qref）。"""
    p = PHYS
    qr = ptp_reference(K)
    dqr = np.gradient(qr, DT)
    ddqr = np.gradient(dqr, DT)
    ff = p["M"] * ddqr - p["mgL"] * np.sin(qr) + p["Bv"] * dqr + p["Fc"] * np.tanh(dqr / p["eps"])
    Kp, Kd = p["M"] * wn**2, 2 * zeta * p["M"] * wn
    q, dq, tau = np.empty(K), np.empty(K), np.empty(K)
    q[0], dq[0] = qr[0], 0.0
    for k in range(K):
        tau[k] = np.clip(ff[k] + Kp * (qr[k] - q[k]) + Kd * (dqr[k] - dq[k]), -p["tauPeak"], p["tauPeak"])
        if k + 1 < K:
            q[k + 1] = q[k] + DT * dq[k]
            dq[k + 1] = dq[k] + DT * accel(q[k], dq[k], tau[k])
    return q, dq, tau, qr


def write_tiny_h5(path, K=3000, seed=0):
    """train 2 本・val 1 本・test 1 本・benchmark 1 本（PTP、閉ループ）。Y は解析モデルのオイラーと厳密に一致する。"""
    rs = np.random.RandomState(seed)
    spec = [("train", "bln_normal"), ("train", "chirp"), ("val", "freefall"), ("test", "hold_-090"), ("benchmark", "ptp_v40")]
    seqs = []
    for sp, pat in spec:
        if pat.startswith("ptp"):
            q, dq, tau, qref = closed_loop_truth(K)
        else:
            tau = 80 * np.sin(2 * np.pi * rs.uniform(1, 5) * np.arange(K) * DT) + 20 * rs.randn(K)
            q, dq = simulate(rs.uniform(-1.5, 0.5), 0.2 * rs.randn(), tau, extra=1.0 if sp in ("train", "val") else 0.0)
            qref = np.full(K, np.nan)
        seqs.append((sp, pat, q, dq, tau, qref))
    names = [f"s{i}" for i in range(len(spec))]
    with h5py.File(path, "w") as f:
        for sp in ("train", "val", "test", "benchmark"):
            R, Y, A, sid, st, qr = [], [], [], [], [], []
            for i, (s, _, q, dq, tau, qref) in enumerate(seqs):
                if s != sp:
                    continue
                R.append(np.c_[q, dq, tau])
                a = accel(q, dq, tau)
                # 行 k の ddq = (dq[k+1] − dq[k])/dt。最終行は解析モデルの値を使う（次状態が行に無いため）
                y = np.r_[np.diff(dq) / DT, a[-1]]
                Y.append(y); A.append(a); sid += [i] * K; st += list(range(K)); qr.append(qref)
            R, Y, A = np.vstack(R), np.concatenate(Y), np.concatenate(A)
            X = np.c_[np.sin(R[:, 0]), np.cos(R[:, 0]), R[:, 1] / SCALE["DTHETA_MAX"], R[:, 2] / SCALE["TAU_MAX"]]
            g = f.create_group(sp)
            g["X"], g["X_raw"] = X.astype(np.float32), R.astype(np.float32)
            g["Y"], g["Y_scaled"], g["a_phys"] = (v[:, None].astype(np.float32) for v in (Y, Y / SCALE["DDQ_SCALE"], A))
            g["scenario_id"], g["step"] = np.array(sid, np.int32), np.array(st, np.int32)
            g["near_limit"], g["qref"] = np.zeros(len(sid), np.uint8), np.concatenate(qr)
        for grp, dct in (("scale", SCALE), ("physics", PHYS)):
            for k, v in dct.items():
                f[f"{grp}/{k}"] = np.array([v])              # MATLAB は長さ 1 の配列として書く
        for k, n in ("x_mean", 3), ("x_std", 3), ("ddq_mean", 1), ("ddq_std", 1), ("res_mean", 1), ("res_std", 1):
            f[f"stats/{k}"] = np.ones(n)
        sc = f.create_group("scenarios")
        dt = h5py.string_dtype()
        sc.create_dataset("name", data=names, dtype=dt); sc.create_dataset("pattern", data=[s[1] for s in seqs], dtype=dt)
        sc.create_dataset("type", data=["excitation"] * len(seqs), dtype=dt); sc.create_dataset("split", data=[s[0] for s in seqs], dtype=dt)
        sc["n_transition"] = np.full(len(seqs), K, np.int32)
        f.attrs["fs_hz"] = np.array([8000.0])
        f.attrs["exclude_near_limit"] = np.array([0.0])
    return seqs
