"""j2ExportNss8kFlex が書き出した HDF5 の読み込みと、シナリオ系列の復元。"""
import h5py
import numpy as np

SPLITS = ("train", "val", "test", "benchmark")
KEYS = ("X", "X_raw", "Y", "Y_scaled", "a_phys")


def _s(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def _scalar(ds):
    return float(np.ravel(ds[()])[0])


def load(path, splits=SPLITS, keys=KEYS):
    """HDF5 を読み、辞書で返す。

    d['splits'][split] : X (N,6)、X_raw (N,5)=[q dq tw dtw tau]、Y/Y_scaled/a_phys (N,2)、scenario_id、step、near_limit、qref
    d['scale'], d['physics'], d['flex'] : スカラー定数の辞書、d['stats'] : x_mean/x_std/res_*（train）、d['dt'] : 1 ステップ [s]
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
        for name in ("scale", "physics", "flex"):
            d[name] = {k: _scalar(f[name][k]) for k in f[name]}
        d["stats"] = {k: f["stats"][k][:] for k in f["stats"]}
        sc = f["scenarios"]
        d["scenarios"] = {k: [_s(x) for x in sc[k][:]] for k in ("name", "pattern", "type", "split")}
        d["scenarios"]["n_transition"] = sc["n_transition"][:].astype(int)
        d["dt"] = 1.0 / float(np.ravel(f.attrs["fs_hz"])[0])
    return d


def group_of(pattern):
    for prefix, g in (("bln_", "bln"), ("hold_", "hold"), ("ptp_", "ptp")):
        if pattern.startswith(prefix):
            return g
    return pattern


def sequences(d, split):
    """split 内のシナリオごとの時系列（行 k の状態が q[k], dq[k], qm[k], dqm[k]、トルクが tau[k]）を復元する。"""
    s = d["splits"][split]
    sid, step, R = s["scenario_id"], s["step"], s["X_raw"].astype(np.float64)
    out = []
    bounds = np.flatnonzero(np.diff(sid)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(sid)]):
        k = int(sid[lo])
        assert np.array_equal(step[lo:hi], np.arange(hi - lo)), "遷移番号が連続していません"
        assert hi - lo == d["scenarios"]["n_transition"][k], "遷移数がシナリオ表と一致しません"
        q, dq, tw, dtw = R[lo:hi, 0], R[lo:hi, 1], R[lo:hi, 2], R[lo:hi, 3]
        out.append({"sid": k, "name": d["scenarios"]["name"][k], "pattern": d["scenarios"]["pattern"][k],
                    "group": group_of(d["scenarios"]["pattern"][k]),
                    "q": q, "dq": dq, "qm": q + tw, "dqm": dq + dtw, "tau": R[lo:hi, 4], "qref": s["qref"][lo:hi]})
    return out
