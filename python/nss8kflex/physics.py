"""弾性 J2 の二慣性の解析モデル（基準線）。摩擦・重力はリンク側。

  JL·ddq  = k·tw + c·dtw + mgL·sin q − Bv·dq − Fc·tanh(dq/ε)
  Jm·ddqm = τ − k·tw − c·dtw          （tw = qm − q、dtw = dqm − dq）
"""
import torch


class FlexPhysics:
    def __init__(self, p, flex, dt):
        self.mgL, self.Fc, self.Bv, self.eps, self.M = p["mgL"], p["Fc"], p["Bv"], p["eps"], p["M"]
        self.tauPeak = p["tauPeak"]
        self.Jm, self.JL, self.k, self.c = flex["Jm"], flex["JL"], flex["k"], flex["c"]
        self.dt = float(dt)

    def accel(self, q, dq, tw, dtw, tau):
        ts = self.k * tw + self.c * dtw
        aq = (ts + self.mgL * torch.sin(q) - self.Bv * dq - self.Fc * torch.tanh(dq / self.eps)) / self.JL
        am = (tau - ts) / self.Jm
        return aq, am

    def step(self, q, dq, qm, dqm, tau):
        """オイラーで 1 ステップ（NSS と同じ更新式）。"""
        aq, am = self.accel(q, dq, qm - q, dqm - dq, tau)
        return q + self.dt * dq, dq + self.dt * aq, qm + self.dt * dqm, dqm + self.dt * am
