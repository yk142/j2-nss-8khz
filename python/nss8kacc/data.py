"""j2_nss8k.h5 の遷移から、(acc_k, q_k, dq_k, τ_k) → acc_{k+1} の対を作る。acc_k = Y[k-1]（直前の区間の加速度）。"""
import numpy as np
import torch


def pairs(split):
    """split の辞書 → (acc_prev, q, dq, tau, y)（いずれも倍精度 numpy）。各シナリオの最初の遷移（step = 0）は除く。"""
    ok = split["step"] > 0
    i = np.flatnonzero(ok)
    assert np.all(split["scenario_id"][i] == split["scenario_id"][i - 1]), "行が連続していません"
    R = split["X_raw"].astype(np.float64)
    Y = split["Y"][:, 0].astype(np.float64)
    return Y[i - 1], R[i, 0], R[i, 1], R[i, 2], Y[i]


def build_xy(split, scale, feat="q"):
    """特徴量 F（標準化前、float32）と標的 Δacc = Y[k] − Y[k-1]（rad/s²、float32）。"""
    acc, q, dq, tau, y = pairs(split)
    pos = [np.sin(q), np.cos(q)] if feat == "sincos" else [q]
    F = np.stack([acc / scale["DDQ_SCALE"], tau / scale["TAU_MAX"], *pos, dq / scale["DTHETA_MAX"]], axis=1).astype(np.float32)
    return torch.from_numpy(F), torch.from_numpy((y - acc).astype(np.float32))
