"""4 状態 NSS（python/nss8kflex）の単体テスト。HDF5 は使わず、解析モデルと合成データで確かめる。"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nss8kflex import evaluate_lib as ev  # noqa: E402
from nss8kflex.model import NSS, ResidualNSS, build_model  # noqa: E402
from nss8kflex.physics import FlexPhysics  # noqa: E402

P = {"M": 6.0857, "mgL": 73.25, "Fc": 23.82, "Bv": 1.42, "eps": 0.0130, "tauPeak": 860.4}
JM = 2.2896
JL = P["M"] - JM
K_ = (2 * np.pi * 20) ** 2 / (1 / JM + 1 / JL)
FLEX = {"Jm": JM, "JL": JL, "k": K_, "c": 2 * 0.05 * np.sqrt(K_ * JM * JL / (JM + JL))}
SCALE = {"DTHETA_MAX": 5.236, "TAU_MAX": 860.4, "DDQ_SCALE": 860.4 / 6.0857}
DT = 1 / 8000


def make(cls, width=16):
    m = cls(SCALE, DT, P, FLEX, width, 2)
    m.tw_std.copy_(torch.tensor([3.5e-3, 0.22]))
    return m


class TestPhysics(unittest.TestCase):
    def test_static_hold_balances(self):
        """静止・ねじり一定のとき、リンク側の加速度が 0 になるトルクでモータ側の加速度も 0（τ = k·tw）。"""
        ph = FlexPhysics(P, FLEX, DT)
        q = torch.tensor([-0.5], dtype=torch.float64)
        tw = -(P["mgL"] * torch.sin(q)) / FLEX["k"]            # k·tw + mgL·sin q = 0
        aq, am = ph.accel(q, torch.zeros_like(q), tw, torch.zeros_like(q), FLEX["k"] * tw)
        self.assertAlmostEqual(float(aq), 0.0, places=9)
        self.assertAlmostEqual(float(am), 0.0, places=9)

    def test_step_euler(self):
        ph = FlexPhysics(P, FLEX, DT)
        q, dq, qm, dqm = (torch.tensor([v], dtype=torch.float64) for v in (0.3, 0.1, 0.302, 0.12))
        tau = torch.tensor([50.0], dtype=torch.float64)
        aq, am = ph.accel(q, dq, qm - q, dqm - dq, tau)
        q1, dq1, qm1, dqm1 = ph.step(q, dq, qm, dqm, tau)
        self.assertAlmostEqual(float(q1), float(q + DT * dq), places=12)
        self.assertAlmostEqual(float(dqm1), float(dqm + DT * am), places=12)
        self.assertAlmostEqual(float(dq1), float(dq + DT * aq), places=12)


class TestModels(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        n = 7
        self.s = [torch.rand(n, dtype=torch.float64) * 2 - 1 for _ in range(2)] + [torch.rand(n, dtype=torch.float64) * 0.01 for _ in range(2)]
        self.tau = torch.rand(n, dtype=torch.float64) * 400 - 200

    def test_shapes_and_features(self):
        for cls, n_in in ((NSS, 6), (ResidualNSS, 8)):
            m = make(cls)
            q, dq, tw, dtw = self.s
            f = m.features(q, dq, tw, dtw, self.tau)
            self.assertEqual(tuple(f.shape), (7, n_in))
            aq, am = m.accel(q, dq, tw, dtw, self.tau)
            self.assertEqual(tuple(aq.shape), (7,))

    def test_residual_starts_at_analytic(self):
        m = make(ResidualNSS)
        q, dq, tw, dtw = self.s
        aq, am = m.accel(q, dq, tw, dtw, self.tau)
        eq, em = m._phys.accel(q, dq, tw, dtw, self.tau)
        self.assertTrue(torch.allclose(aq, eq, rtol=1e-5, atol=1e-4))
        self.assertTrue(torch.allclose(am, em, rtol=1e-5, atol=1e-4))

    def test_step_matches_accel(self):
        m = make(NSS)
        q, dq, qm, dqm = self.s[0], self.s[1], self.s[0] + self.s[2], self.s[1] + self.s[3]
        aq, am = m.accel(q, dq, qm - q, dqm - dq, self.tau)
        q1, dq1, qm1, dqm1 = m.step_physical(q, dq, qm, dqm, self.tau)
        self.assertTrue(torch.allclose(dq1, dq + DT * aq))
        self.assertTrue(torch.allclose(qm1, qm + DT * dqm))
        self.assertTrue(torch.allclose(dqm1, dqm + DT * am))

    def test_build_model_roundtrip(self):
        m = make(ResidualNSS)
        m2 = build_model(m.config())
        m2.load_state_dict(m.state_dict())
        self.assertIsInstance(m2, ResidualNSS)
        self.assertEqual(m2.width, m.width)


def ptp(K=16000, q_from=-1.0, q_to=-0.4, lead=400):
    t = np.clip((np.arange(K) - lead) / (K - 2 * lead), 0, 1)
    return q_from + (q_to - q_from) * (10 * t**3 - 15 * t**4 + 6 * t**5)


class TestClosedLoop(unittest.TestCase):
    def test_analytic_plant_tracks_reference(self):
        """プラントを解析モデルにしたモータ角 PD＋FF が、参照に追従し発散しない。"""
        ph = FlexPhysics(P, FLEX, DT)
        qr = ptp()
        seq = {"q": np.array([qr[0]]), "dq": np.array([0.0]), "qm": np.array([qr[0]]), "dqm": np.array([0.0]),
               "tau": np.zeros(len(qr)), "qref": qr}
        Q, QM, U = ev.closed_loop_batch(ph.step, [seq], P, FLEX, DT)
        self.assertFalse(np.isnan(Q).any())
        err = np.degrees(np.sqrt(np.mean((Q[0, :len(qr)] - qr) ** 2)))
        self.assertLess(err, 0.5)
        self.assertLess(np.abs(U).max(), P["tauPeak"] + 1e-9)

    def test_judge_keys(self):
        row = {"model": {"diverged": False, "vs_truth_rmse_q_deg": 0.01, "vs_truth_rmse_twist_rad": 1e-4},
               "analytic": {"vs_truth_rmse_q_deg": 0.02}}
        one = {"test": {"all": {o: {"ratio_to_analytic": 0.5} for o in ("ddq", "ddqm")}}}
        thr = {"model": {"steps_per_s": 2e5}, "analytic": {"steps_per_s": 3e5}}
        g = ev.judge(one, None, [row], thr)
        self.assertTrue(all(g[k]["pass"] for k in "acd"))


if __name__ == "__main__":
    unittest.main()
