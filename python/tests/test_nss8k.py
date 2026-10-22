"""8 kHz NSS の単体テスト。実行: .venv/bin/python -m unittest discover -s python/tests -v"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers_nss as H  # noqa: E402
from nss8k import evaluate_lib as ev  # noqa: E402
from nss8k.data import group_of, load, sequences  # noqa: E402
from nss8k.model import NSS, ResidualNSS, build_model  # noqa: E402
from nss8k.physics import Physics  # noqa: E402
import train_nss  # noqa: E402

T = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731


class TestNss8k(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "tiny.h5")
        H.write_tiny_h5(cls.path)
        cls.d = load(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---- データ ----
    def test_load_shapes_and_dt(self):
        d = self.d
        self.assertAlmostEqual(d["dt"], 1 / 8000)
        s = d["splits"]["train"]
        N = len(s["X"])
        self.assertEqual(s["X"].shape, (N, 4)); self.assertEqual(s["X_raw"].shape, (N, 3))
        self.assertEqual(s["Y"].shape, (N, 1)); self.assertEqual(s["a_phys"].shape, (N, 1))
        self.assertEqual(s["qref"].dtype, np.float64)
        self.assertEqual(d["physics"]["M"], H.PHYS["M"])             # 長さ 1 の配列がスカラーになる

    def test_sequences_roundtrip(self):
        seqs = sequences(self.d, "benchmark")
        self.assertEqual(len(seqs), 1)
        q, dq, tau, qref = H.closed_loop_truth(len(seqs[0]["q"]))
        np.testing.assert_allclose(seqs[0]["q"], q, atol=1e-6)
        np.testing.assert_allclose(seqs[0]["qref"], qref, atol=1e-12)    # 倍精度のまま

    def test_group_of(self):
        self.assertEqual(group_of("hold_+030"), "hold"); self.assertEqual(group_of("ptp_v20"), "ptp")
        self.assertEqual(group_of("chirp"), "chirp")

    # ---- 解析モデル ----
    def test_physics_matches_numpy(self):
        ph = Physics(H.PHYS, H.DT)
        q, dq, tau = 0.3, -1.2, 40.0
        self.assertAlmostEqual(float(ph.accel(T(q), T(dq), T(tau))), H.accel(q, dq, tau), places=12)
        q1, dq1 = ph.step(T(q), T(dq), T(tau))
        self.assertAlmostEqual(float(q1), q + H.DT * dq, places=14)
        self.assertAlmostEqual(float(dq1), dq + H.DT * H.accel(q, dq, tau), places=12)

    def test_exported_a_phys_is_analytic(self):
        s = self.d["splits"]["test"]
        R = s["X_raw"].astype(np.float64)
        np.testing.assert_allclose(s["a_phys"][:, 0], H.accel(R[:, 0], R[:, 1], R[:, 2]), rtol=2e-4, atol=2e-2)

    # ---- モデル ----
    def test_residual_zero_init_equals_physics(self):
        m = ResidualNSS(H.SCALE, H.DT, H.PHYS, 16, 2)
        q, dq, tau = T([0.1, -1.0, 0.7]), T([0.5, -2.0, 0.0]), T([10.0, -200.0, 5.0])
        np.testing.assert_allclose(m.ddq(q, dq, tau).detach().numpy(), H.accel(q.numpy(), dq.numpy(), tau.numpy()), rtol=1e-5, atol=1e-4)

    def test_step_is_euler(self):
        m = NSS(H.SCALE, H.DT, H.PHYS, 16, 2)
        q, dq, tau = T([0.1, -1.0]), T([0.5, -2.0]), T([10.0, -200.0])
        with torch.no_grad():
            a = m.ddq(q, dq, tau)
            q1, dq1 = m.step_physical(q, dq, tau)
        np.testing.assert_allclose(q1.numpy(), (q + H.DT * dq).numpy(), atol=1e-14)
        np.testing.assert_allclose(dq1.numpy(), (dq + H.DT * a.double()).numpy(), atol=1e-12)

    def test_features_match_export(self):
        """モデルが物理量から作る特徴量が、エクスポートの X / a_phys と一致する（学習と評価で同じ入力）。"""
        s = self.d["splits"]["train"]
        R = torch.from_numpy(s["X_raw"].astype(np.float64))[:500]
        for cls, n in ((NSS, 4), (ResidualNSS, 5)):
            m = cls(H.SCALE, H.DT, H.PHYS, 8, 2)
            F = m.features(R[:, 0], R[:, 1], R[:, 2]).detach().numpy()
            self.assertEqual(F.shape[1], n)
            np.testing.assert_allclose(F[:, :4], s["X"][:500], atol=1e-5)
            if n == 5:
                np.testing.assert_allclose(F[:, 4], s["a_phys"][:500, 0] / H.SCALE["DDQ_SCALE"], atol=2e-4)

    # ---- 学習 ----
    def test_train_smoke_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as out:
            for kind in ("A", "B"):
                train_nss.main(["--data", self.path, "--model", kind, "--width", "16", "--depth", "2", "--epochs", "3",
                                "--batch", "256", "--run", kind, "--out-root", out, "--threads", "1"])
                ck = torch.load(os.path.join(out, kind, "model.pt"))
                with open(os.path.join(out, kind, "history.json")) as f:
                    hist = json.load(f)
                self.assertLess(hist[-1]["train_loss"], hist[0]["train_loss"] * 1.01)
                m = build_model(ck["config"]); m.load_state_dict(ck["state_dict"]); m.eval()
                self.assertEqual(m.residual, kind == "B")
                q, dq, tau = T([0.2]), T([0.3]), T([15.0])
                self.assertTrue(torch.isfinite(m.ddq(q, dq, tau)).all().item())

    # ---- 評価 ----
    def test_one_step_with_exact_physics(self):
        """解析モデルそのもの（残差 0）の B は、解析モデルで作ったデータで誤差 ≈ 0（最終行を除く）。"""
        m = ResidualNSS(H.SCALE, H.DT, H.PHYS, 8, 2)
        t = ev.one_step_table(m, self.d, "test")["all"]
        self.assertLess(t["model"], 0.1)                              # 最終行と float32 の丸めで厳密に 0 にはならない
        self.assertAlmostEqual(t["model"], t["analytic"], delta=0.05)

    def test_windows_with_exact_physics(self):
        m = ResidualNSS(H.SCALE, H.DT, H.PHYS, 8, 2)
        w = ev.window_rollouts(m, Physics(H.PHYS, H.DT), self.d, "test", horizon=500, stride=500, steps=(1, 100, 500))
        self.assertGreater(w["n_windows"], 0)
        self.assertLess(w["methods"]["analytic"]["rmse_q_deg"]["500"], 1e-3)
        self.assertLess(w["methods"]["model"]["rmse_q_deg"]["500"], 0.05)   # 学習前の B は解析モデルそのもの（float32 の丸め）
        self.assertGreater(w["methods"]["persistence"]["rmse_q_deg"]["500"], 0.01)

    def test_closed_loop_reproduces_truth(self):
        """解析モデルを閉ループに接続すると、同じ解析モデルで作った真値の PTP を再現する（制御器の実装の検証を兼ねる）。"""
        ph = Physics(H.PHYS, H.DT)
        rows, seqs, res = ev.closed_loop_benchmark(ResidualNSS(H.SCALE, H.DT, H.PHYS, 8, 2), ph, self.d)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["analytic"]["diverged"])
        self.assertLess(rows[0]["analytic"]["vs_truth_rmse_q_deg"], 1e-4)
        self.assertLess(rows[0]["model"]["vs_truth_rmse_q_deg"], 1e-2)
        cc = ev.controller_check(seqs, self.d["physics"], self.d["dt"])
        self.assertLess(cc[0]["rms_N_m"], 1e-2)                       # 記録トルクを制御器が再現する

    def test_closed_loop_flags_divergence(self):
        """壊れたモデル（トルクを無視して加速し続ける）は発散として検出される。"""
        ph = Physics(H.PHYS, H.DT)
        seqs = sequences(self.d, "benchmark")
        Q, _ = ev.closed_loop_batch(lambda q, dq, tau: (q + H.DT * dq, dq + 5000.0 * H.DT * torch.ones_like(dq)), seqs, self.d["physics"], H.DT)
        self.assertTrue(np.isnan(Q).any())

    def test_throughput_and_judge(self):
        m = ResidualNSS(H.SCALE, H.DT, H.PHYS, 8, 2)
        thr = ev.throughput(m, Physics(H.PHYS, H.DT), self.d, batch=16, steps=100, repeats=1)
        self.assertGreater(thr["model"]["steps_per_s"], 0)
        one = {sp: ev.one_step_table(m, self.d, sp) for sp in ("test", "benchmark")}
        rows, _, _ = ev.closed_loop_benchmark(m, Physics(H.PHYS, H.DT), self.d)
        g = ev.judge("B", one, None, rows, thr)
        self.assertIn("c", g); self.assertTrue(g["c"]["pass"])


if __name__ == "__main__":
    unittest.main()
