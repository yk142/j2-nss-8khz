"""j2_nss8kflex.h5 の遷移から、(acc_k, 状態_k, τ_k) → acc_{k+1} の対を作る。acc_k = Y[k-1]（直前の区間の [ddq, ddqm]）。"""
import numpy as np
import torch

from nss8kacc.data import window_starts  # noqa: F401  （step・scenario_id だけを使う汎用の関数）


def pairs(split):
    """split の辞書 → (acc_prev (N,2), X_raw (N,5)=[q dq tw dtw tau], y (N,2), a_phys (N,2))（倍精度 numpy）。step = 0 の行は除く。"""
    i = np.flatnonzero(split["step"] > 0)
    assert np.all(split["scenario_id"][i] == split["scenario_id"][i - 1]), "行が連続していません"
    Y = split["Y"].astype(np.float64)
    return Y[i - 1], split["X_raw"][i].astype(np.float64), Y[i], split["a_phys"][i].astype(np.float64)


def feature_cols(acc, R, scale, out_scale, tw_std):
    """特徴量 F（標準化前、float32）: [ddq/s_L, ddqm/s_m, τ/TAU_MAX, q, dq/DTHETA_MAX, tw/s_tw, dtw/s_dtw]。"""
    return np.stack([acc[:, 0] / out_scale[0], acc[:, 1] / out_scale[1], R[:, 4] / scale["TAU_MAX"], R[:, 0],
                     R[:, 1] / scale["DTHETA_MAX"], R[:, 2] / tw_std[0], R[:, 3] / tw_std[1]], axis=1).astype(np.float32)


def build_xy(split, scale, out_scale, tw_std):
    """特徴量 F（float32）と標的 Δacc = Y[k] − Y[k-1]（rad/s²、float32、(N,2)）。"""
    acc, R, y, _ = pairs(split)
    return torch.from_numpy(feature_cols(acc, R, scale, out_scale, tw_std)), torch.from_numpy((y - acc).astype(np.float32))


def targets(split):
    """pairs() の標的 y = acc_{k+1}（float32、(N,2)）。"""
    return torch.from_numpy(pairs(split)[2].astype(np.float32))
