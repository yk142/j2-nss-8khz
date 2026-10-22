"""J2 の解析モデル（基準線）。M·ddq = τ + mgL·sin θ − Bv·dq − Fc·tanh(dq/ε)。接触（ストッパ）は含まない。"""
import torch


class Physics:
    def __init__(self, p, dt):
        self.M, self.mgL, self.Fc, self.Bv, self.eps = p["M"], p["mgL"], p["Fc"], p["Bv"], p["eps"]
        self.tauPeak, self.dt = p["tauPeak"], float(dt)

    def accel(self, q, dq, tau):
        return (tau + self.mgL * torch.sin(q) - self.Bv * dq - self.Fc * torch.tanh(dq / self.eps)) / self.M

    def step(self, q, dq, tau):
        """オイラーで 1 ステップ（NSS と同じ更新式）。"""
        return q + self.dt * dq, dq + self.dt * self.accel(q, dq, tau)
