"""加速度を状態に持つ残差型 NSS（docs/requirements_nss_resid_state.md）。

状態 acc（直前の区間の加速度）、入力 [τ, q, dq]。NN の入力 [acc, τ, q, dq]、出力 Δacc、acc₁ = acc + Δacc。
位置の入力は feat="q"（生の q、基本形）か "sincos"（sin q, cos q の 2 列、アブレーション）。
ロールアウト: 速度は前進オイラー（dq₁ = dq + Δt·acc₁）、位置は台形則（q₁ = q + Δt/2·(dq + dq₁)）。
最後の層は 0 初期化（恒等写像 acc₁ = acc から始める）。入力は train の平均・標準偏差で標準化する。

step_physical は acc を内部状態として持つ（evaluate_lib のロールアウトと同じ呼び出し形にするため）。
ロールアウトの前に reset() を呼ぶ。最初の 1 回は acc を解析モデルの加速度で初期化する。
"""
import torch
import torch.nn as nn

from nss8k.physics import Physics


def _mlp(d_in, width, depth):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    last = nn.Linear(d, 1)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class AccNSS(nn.Module):
    kind = "acc"
    residual = False          # evaluate_nss の kind 判定用（解析モデルを使わない）
    def __init__(self, scale, dt, phys_params, width=64, depth=3, feat="q"):
        super().__init__()
        self.feat = feat
        self.n_in = 5 if feat == "sincos" else 4
        self.scale, self.dt = dict(scale), float(dt)
        self.phys_params = dict(phys_params)
        self.width, self.depth = int(width), int(depth)
        self.register_buffer("x_mean", torch.zeros(self.n_in))
        self.register_buffer("x_std", torch.ones(self.n_in))
        self.register_buffer("d_std", torch.ones(1))
        self.net = _mlp(self.n_in, width, depth)
        self._phys = Physics(self.phys_params, self.dt)
        self._acc = None

    def config(self):
        return {"kind": self.kind, "width": self.width, "depth": self.depth, "scale": self.scale,
                "dt": self.dt, "phys_params": self.phys_params, "feat": self.feat}

    def features(self, acc, q, dq, tau):
        s = self.scale
        pos = [torch.sin(q), torch.cos(q)] if self.feat == "sincos" else [q]
        return torch.stack([acc / s["DDQ_SCALE"], tau / s["TAU_MAX"], *pos, dq / s["DTHETA_MAX"]], dim=-1).float()

    def net_out(self, feat):
        return self.net((feat - self.x_mean) / self.x_std)

    def delta_acc(self, acc, q, dq, tau):
        """Δacc [rad/s²]（倍精度で返す）。"""
        return (self.net_out(self.features(acc, q, dq, tau))[..., 0] * self.d_std[0]).to(q.dtype)

    def reset(self):
        self._acc = None

    def step_physical(self, q, dq, tau):
        if self._acc is None or self._acc.shape != q.shape:
            self._acc = self._phys.accel(q, dq, tau)
        acc1 = self._acc + self.delta_acc(self._acc, q, dq, tau)
        dq1 = dq + self.dt * acc1
        q1 = q + 0.5 * self.dt * (dq + dq1)
        self._acc = acc1
        return q1, dq1


def build_model(config):
    return AccNSS(config["scale"], config["dt"], config["phys_params"], config["width"], config["depth"], config.get("feat", "q"))
