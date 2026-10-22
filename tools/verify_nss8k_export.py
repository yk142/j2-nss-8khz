#!/usr/bin/env python3
"""j2ExportNss8k で書き出した HDF5 を Python（h5py）から読み、整合を検証する。

使い方:
    .venv/bin/python tools/verify_nss8k_export.py data/export/j2_nss8k.h5

検証内容:
  - 各 split の形状 X(N,4) X_raw(N,3) Y(N,1) Y_scaled(N,1) a_phys(N,1)、型、有限性
  - X が X_raw と /scale から再計算した値と一致（sin²+cos²=1 を含む）、Y_scaled = Y / DDQ_SCALE
  - a_phys が X_raw と /physics から再計算した解析モデルの加速度と一致
  - 正規化統計 /stats が train の値と一致（x_mean, x_std, ddq_std, res_std）
  - scenario_id / step / split の整合、遷移数の合計（near_limit を除外していない場合）
  - near_limit のフラグ数、at_limit 相当（リミットから 0.1° 以内）が 0 件
"""
import sys
import numpy as np
import h5py


def check(cond, msg):
    print(('  [OK] ' if cond else '  [NG] ') + msg)
    return bool(cond)


def scalar(g, k):
    return float(np.ravel(g[k][()])[0])


def main(path):
    ok = True
    with h5py.File(path, 'r') as f:
        print(f'ファイル: {path}')
        scale = {k: scalar(f['scale'], k) for k in f['scale']}
        phys = {k: scalar(f['physics'], k) for k in f['physics']}
        names = [x.decode() if isinstance(x, bytes) else x for x in f['scenarios/name'][:]]
        sc_split = np.array([x.decode() if isinstance(x, bytes) else x for x in f['scenarios/split'][:]])
        n_tr = f['scenarios/n_transition'][:]
        excl = int(np.ravel(f.attrs.get('exclude_near_limit', 0))[0])
        total = 0
        for sp in [s for s in ('train', 'val', 'test', 'benchmark') if s in f]:
            g = f[sp]
            X, R, Y, Ys, A = (g[k][:].astype(np.float64) for k in ('X', 'X_raw', 'Y', 'Y_scaled', 'a_phys'))
            N = X.shape[0]
            total += N
            print(f'[{sp}] N = {N:,}')
            ok &= check(X.shape == (N, 4) and R.shape == (N, 3) and Y.shape == (N, 1) and Ys.shape == (N, 1) and A.shape == (N, 1),
                        f'形状 X{X.shape} X_raw{R.shape} Y{Y.shape} Y_scaled{Ys.shape} a_phys{A.shape}')
            ok &= check(all(np.isfinite(a).all() for a in (X, R, Y, Ys, A)), '有限（NaN/Inf なし）')
            ok &= check(np.allclose(X[:, 0], np.sin(R[:, 0]), atol=1e-6) and np.allclose(X[:, 1], np.cos(R[:, 0]), atol=1e-6),
                        'X[:,0:2] = sin q, cos q')
            ok &= check(np.allclose(X[:, 2], R[:, 1] / scale['DTHETA_MAX'], atol=1e-6), 'X[:,2] = dq / DTHETA_MAX')
            ok &= check(np.allclose(X[:, 3], R[:, 2] / scale['TAU_MAX'], atol=1e-6), 'X[:,3] = tau / TAU_MAX')
            ok &= check(np.allclose(Ys, Y / scale['DDQ_SCALE'], rtol=1e-5, atol=1e-7), 'Y_scaled = Y / DDQ_SCALE')
            q, dq, tau = R[:, 0], R[:, 1], R[:, 2]
            a = (tau + phys['mgL'] * np.sin(q) - phys['Bv'] * dq - phys['Fc'] * np.tanh(dq / phys['eps'])) / phys['M']
            ok &= check(np.allclose(A[:, 0], a, rtol=1e-4, atol=1e-3), 'a_phys = 解析モデル（X_raw, /physics から再計算）')
            dmin = np.minimum(q - phys['qMin'], phys['qMax'] - q).min()
            ok &= check(dmin > np.deg2rad(0.1), f'リミットから 0.1° を超えて内側（最小 {np.rad2deg(dmin):.2f}°）')
            sid, stp, nl = g['scenario_id'][:], g['step'][:], g['near_limit'][:]
            ok &= check(sid.min() >= 0 and sid.max() < len(names), 'scenario_id は 0 始まりで範囲内')
            ok &= check(bool(np.all(sc_split[sid] == sp)), f"全行が split == '{sp}' のシナリオに属する")
            ok &= check(bool(np.all(stp >= 0) and np.all(stp < n_tr[sid])), 'step は 0 ≤ step < n_transition')
            print(f'     near_limit {int(nl.sum()):,} 件、ddq の標準偏差 {Y.std():.3f}、(Y − a_phys) の RMS {np.sqrt(np.mean((Y - A) ** 2)):.3f} rad/s²')
            if sp == 'train':
                st = {k: f['stats'][k][:] for k in f['stats']}
                ok &= check(np.allclose(st['x_mean'], R.mean(0), rtol=1e-3, atol=1e-4), '/stats/x_mean = train の X_raw の平均')
                ok &= check(np.allclose(st['x_std'], R.std(0, ddof=1), rtol=1e-3), '/stats/x_std = train の X_raw の標準偏差')
                ok &= check(np.allclose(st['ddq_std'], Y.std(ddof=1), rtol=1e-3), '/stats/ddq_std = train の ddq の標準偏差')
                ok &= check(np.allclose(st['res_std'], (Y - A).std(ddof=1), rtol=1e-2), '/stats/res_std = train の残差の標準偏差')
        if not excl:
            ok &= check(total == int(n_tr.sum()), f'遷移数の合計 {total:,} = シナリオ別 n_transition の合計 {int(n_tr.sum()):,}')
    print('結果:', '合格' if ok else '不合格')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
