"""1 kHz サンプル・ホールド＋1 サンプル遅れのモータ側 PD の線形安定性（docs/delay_vs_resonance.md）。
プラントは線形（重力・摩擦なし）。離散化したプラント＋遅れ状態の閉ループ行列のスペクトル半径 ρ を出す（ρ>1 で不安定）。
使い方: .venv/bin/python tools/delay_stability.py
"""
import numpy as np


def expm(A, n=20):
    s = max(0, int(np.ceil(np.log2(max(1e-12, np.linalg.norm(A, 1))))) + 1)
    A = A / 2 ** s
    E = np.eye(len(A)); T = np.eye(len(A))
    for i in range(1, n):
        T = T @ A / i; E = E + T
    for _ in range(s):
        E = E @ E
    return E


Jm, JL = 2.2896, 3.7961
M = Jm + JL
Ts = 1e-3


def rho(k, c, wn, z, rigid=False):
    if rigid:
        A = np.array([[0, 1], [0, 0.]]); B = np.array([0, 1 / M]); C = np.array([1, 0.]); D = np.array([0, 1.])
    else:
        A = np.array([[0, 1, 0, 0], [-k / JL, -c / JL, k / JL, c / JL], [0, 0, 0, 1], [k / Jm, c / Jm, -k / Jm, -c / Jm]])
        B = np.array([0, 0, 0, 1 / Jm]); C = np.array([0, 0, 1, 0.]); D = np.array([0, 0, 0, 1.])
    n = len(A)
    Mx = np.zeros((n + 1, n + 1)); Mx[:n, :n] = A * Ts; Mx[:n, n] = B * Ts
    E = expm(Mx); Ad = E[:n, :n]; Bd = E[:n, n]
    Kv = -(M * wn ** 2 * C + 2 * z * M * wn * D)        # 偏差の PD。サンプル k で計算し、次の周期に印加
    Z = np.zeros((n + 1, n + 1)); Z[:n, :n] = Ad; Z[:n, n] = Bd; Z[n, :n] = Kv
    return max(abs(np.linalg.eigvals(Z)))


if __name__ == '__main__':
    kN = (2 * np.pi * 20) ** 2 / (1 / Jm + 1 / JL); cN = 17.947
    wns = [15, 30, 60, 90, 120, 180, 240]
    for name, args in [('剛体', (0, 0, True)), ('弾性 20 Hz（C4）', (kN, cN, False)), ('剛性 100 倍 約 200 Hz（C5）', (kN * 100, cN * 10, False))]:
        print(name, ' wn=' + ' '.join(map(str, wns)))
        for z in [0.7, 1, 1.5]:
            print('  ζ=%.1f' % z, ' '.join('%.3f' % rho(args[0], args[1], w, z, args[2]) for w in wns))
