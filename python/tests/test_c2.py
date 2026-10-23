"""(c2) の単体テスト。実行: .venv/bin/python -m unittest discover -s python/tests -v"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers_nss as H  # noqa: E402
from nss8k import c2  # noqa: E402
from nss8k.physics import Physics  # noqa: E402


def make_trial(ff_err=0.0, dist=0.0, K=4000):
    """滑らかな参照（5 次多項式）と、解析モデルの FF（係数を (1+ff_err) 倍）・矩形の外乱を持つ軌道。"""
    t = np.clip((np.arange(K + 1) - 200) / (K - 400), 0, 1)
    s = 10 * t**3 - 15 * t**4 + 6 * t**5
    ds = (30 * t**2 - 60 * t**3 + 30 * t**4) / ((K - 400) * H.DT)
    dds = (60 * t - 180 * t**2 + 120 * t**3) / ((K - 400) * H.DT) ** 2
    qr, dqr, ddqr = -1.0 + 0.6 * s, 0.6 * ds, 0.6 * dds
    p = H.PHYS
    ff = (1 + ff_err) * (p["M"] * ddqr - p["mgL"] * np.sin(qr) + p["Bv"] * dqr + p["Fc"] * np.tanh(dqr / p["eps"]))
    td = np.zeros(K + 1)
    td[2500:2900] = dist
    return {"name": "t", "qref": qr, "dqref": dqr, "tauFf": ff, "tauD": td, "tau": np.zeros(K + 1)}


class TestC2(unittest.TestCase):
    def run_cl(self, trial, wn, zeta):
        ph = Physics(H.PHYS, H.DT)
        Q, U, K = c2.closed_loop(ph.step, [trial], H.PHYS, H.DT, wn, zeta)
        return Q, U, K

    def test_exact_ff_tracks_and_gain_barely_matters(self):
        tr = make_trial()
        for wn in (15, 60):
            Q, _, K = self.run_cl(tr, wn, 1.0)
            self.assertLess(np.degrees(np.nanmax(np.abs(Q[0, :K[0] + 1] - tr["qref"]))), 0.05)

    def test_gain_matters_with_ff_error_and_disturbance(self):
        """FF 誤差＋外乱があると、ゲインが高いほど追従誤差が小さい（wn 15 > 30 > 60）。"""
        tr = make_trial(ff_err=-0.15, dist=20.0)
        r = []
        for wn in (15, 30, 60):
            Q, U, K = self.run_cl(tr, wn, 1.0)
            r.append(c2.trial_metrics(Q, U, K, [tr], H.DT)[0]["theta_rmse_deg"])
        self.assertGreater(r[0], r[1]); self.assertGreater(r[1], r[2])

    def test_disturbance_is_added_before_saturation(self):
        tr = make_trial(dist=1e5)
        _, U, K = self.run_cl(tr, 30, 1.0)
        self.assertLessEqual(np.nanmax(np.abs(U)), H.PHYS["tauPeak"] + 1e-9)

    def test_ranks_and_spearman(self):
        self.assertTrue(np.allclose(c2.ranks([3, 1, 2, 2]), [3, 0, 1.5, 1.5]))
        self.assertAlmostEqual(c2.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(c2.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_settling_time(self):
        dt = 1e-3
        n = 1000
        dqref = np.ones(n); dqref[300:600] = 0.0                      # 300〜600 が停留（0.3 s）
        err = np.zeros(n); err[300:350] = 1.0                         # 停留の最初の 50 ms は誤差 1°
        self.assertAlmostEqual(c2.settling_time(err, dqref, dt), 0.05, places=6)
        err[:] = 1.0                                                  # 区間の最後まで入らない → 区間の長さ
        self.assertAlmostEqual(c2.settling_time(err, dqref, dt), 0.3, places=6)

    def test_cost_normalisation(self):
        agg = {"theta_rmse_deg": 0.02, "tau_rms": 50.0}
        self.assertAlmostEqual(c2.cost(agg, {"theta_rmse_deg": 0.01, "tau_rms": 100.0}), 2.5)


if __name__ == "__main__":
    unittest.main()
