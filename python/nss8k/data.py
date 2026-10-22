"""j2ExportNss8k が書き出した HDF5 の読み込みと、シナリオ系列の復元。"""
import h5py
import numpy as np

SPLITS = ("train", "val", "test", "benchmark")
KEYS = ("X", "X_raw", "Y", "Y_scaled", "a_phys")


def _s(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def _scalar(ds):
    return float(np.ravel(ds[()])[0])        # MATLAB が書く数値スカラーは長さ 1 の配列に見える


def load(path, splits=SPLITS, keys=KEYS):
    """HDF5 を読み、辞書で返す。

    d['splits'][split] : keys で指定した行列（X (N,4)、X_raw (N,3)、Y/Y_scaled/a_phys (N,1)）、
                         scenario_id, step, near_limit(bool), qref (N,)（倍精度）
    d['scale'], d['physics'] : スカラー定数の辞書（SI 単位）、d['stats'] : x_mean/x_std/ddq_*/res_*（train）
    d['scenarios'] : name, pattern, type, split, n_transition、d['dt'] : 1 ステップ [s]
    """
    d = {"splits": {}}
    with h5py.File(path, "r") as f:
        for sp in splits:
            g = f[sp]
            e = {k: g[k][:] for k in keys}
            e["scenario_id"], e["step"] = g["scenario_id"][:], g["step"][:]
            e["near_limit"] = g["near_limit"][:].astype(bool)
            e["qref"] = g["qref"][:].astype(np.float64)
            d["splits"][sp] = e
        d["scale"] = {k: _scalar(f["scale"][k]) for k in f["scale"]}
        d["physics"] = {k: _scalar(f["physics"][k]) for k in f["physics"]}
        d["stats"] = {k: f["stats"][k][:] for k in f["stats"]}
        sc = f["scenarios"]
        d["scenarios"] = {k: [_s(x) for x in sc[k][:]] for k in ("name", "pattern", "type", "split")}
        d["scenarios"]["n_transition"] = sc["n_transition"][:].astype(int)
        d["dt"] = 1.0 / float(np.ravel(f.attrs["fs_hz"])[0])
        d["exclude_near_limit"] = bool(np.ravel(f.attrs.get("exclude_near_limit", 0))[0])
    return d


def group_of(pattern):
    """パターン名を評価用の群にまとめる。"""
    for prefix, g in (("bln_", "bln"), ("hold_", "hold"), ("ptp_", "ptp")):
        if pattern.startswith(prefix):
            return g
    return pattern  # freefall, chirp, step, micro


def sequences(d, split):
    """split 内のシナリオごとの時系列を復元する。行 k の状態が (q, dq)[k]、トルクが tau[k]。

    行 = 遷移なので、q, dq は K 点（最後の遷移の次状態は含まない）、tau は K 点。
    エクスポートの行はシナリオごとに連続し、遷移番号の昇順に並ぶ（near_limit を除外していない場合）。
    """
    if d["exclude_near_limit"]:
        raise ValueError("near_limit を除外したエクスポートからは時系列を復元できません")
    s = d["splits"][split]
    sid, step, R = s["scenario_id"], s["step"], s["X_raw"].astype(np.float64)
    out = []
    bounds = np.flatnonzero(np.diff(sid)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(sid)]):
        k = int(sid[lo])
        assert np.array_equal(step[lo:hi], np.arange(hi - lo)), "遷移番号が連続していません"
        assert hi - lo == d["scenarios"]["n_transition"][k], "遷移数がシナリオ表と一致しません"
        out.append({"sid": k, "name": d["scenarios"]["name"][k], "pattern": d["scenarios"]["pattern"][k],
                    "group": group_of(d["scenarios"]["pattern"][k]),
                    "q": R[lo:hi, 0], "dq": R[lo:hi, 1], "tau": R[lo:hi, 2], "qref": s["qref"][lo:hi]})
    return out
