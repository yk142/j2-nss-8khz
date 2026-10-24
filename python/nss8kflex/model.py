"""4 状態 NSS のモデル。状態 (q, dq, qm, dqm)、出力 (ddq, ddqm) の 2 列、更新はオイラー。

  NSS          (A) 純 NN: 入力 X（6 列）、目標 Y_scaled = [ddq, ddqm] ./ [τpeak/JL, τpeak/Jm]
  ResidualNSS  (B) 残差型: 入力 X ＋ a_phys ./ out_scale（8 列）、[ddq, ddqm] = a_phys + NN(x)·res_std。最後の層を 0 初期化
特徴量のねじり tw = qm − q は倍精度で計算してから単精度にする（ねじりは 1e-3 rad と小さいため）。
入力は train の平均・標準偏差で標準化する（バッファとして保存）。
"""
import torch
import torch.nn as nn

from .physics import FlexPhysics


def _mlp(d_in, width, depth, zero_last=False):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    last = nn.Linear(d, 2)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class _Base(nn.Module):
    n_in = 6
    residual = False

    def __init__(self, scale, dt, phys_params, flex, width=128, depth=3):
        super().__init__()
        self.scale, self.dt = dict(scale), float(dt)
        self.phys_params, self.flex = dict(phys_params), dict(flex)
        self.width, self.depth = int(width), int(depth)
        self.register_buffer("x_mean", torch.zeros(self.n_in))
        self.register_buffer("x_std", torch.ones(self.n_in))
        self.register_buffer("res_std", torch.ones(2))
        self.register_buffer("tw_std", torch.ones(2))            # [s_tw, s_dtw]
        self.register_buffer("out_scale", torch.tensor([phys_params["tauPeak"] / flex["JL"],
                                                        phys_params["tauPeak"] / flex["Jm"]]))
        self._phys = FlexPhysics(self.phys_params, self.flex, self.dt)

    def config(self):
        return {"kind": self.kind, "width": self.width, "depth": self.depth, "scale": self.scale, "dt": self.dt,
                "phys_params": self.phys_params, "flex": self.flex}

    def features(self, q, dq, tw, dtw, tau):
        s = self.scale
        cols = [torch.sin(q), torch.cos(q), dq / s["DTHETA_MAX"], tw / self.tw_std[0].double(),
                dtw / self.tw_std[1].double(), tau / s["TAU_MAX"]]
        if self.residual:
            aq, am = self._phys.accel(q, dq, tw, dtw, tau)
            cols += [aq / self.out_scale[0].double(), am / self.out_scale[1].double()]
        return torch.stack(cols, dim=-1).float()

    def net_out(self, feat):
        return self.net((feat - self.x_mean) / self.x_std)

    def acc_from_features(self, feat):
        raise NotImplementedError

    def accel(self, q, dq, tw, dtw, tau):
        """(ddq, ddqm) [rad/s²]。"""
        a = self.acc_from_features(self.features(q, dq, tw, dtw, tau)).to(q.dtype)
        return a[..., 0], a[..., 1]

    def step_physical(self, q, dq, qm, dqm, tau):
        aq, am = self.accel(q, dq, qm - q, dqm - dq, tau)
        return q + self.dt * dq, dq + self.dt * aq, qm + self.dt * dqm, dqm + self.dt * am


class NSS(_Base):
    kind = "nss"
    residual = False
    n_in = 6

    def __init__(self, scale, dt, phys_params, flex, width=128, depth=3):
        super().__init__(scale, dt, phys_params, flex, width, depth)
        self.net = _mlp(self.n_in, width, depth)

    def acc_from_features(self, feat):
        return self.net_out(feat) * self.out_scale


class ResidualNSS(_Base):
    kind = "residual"
    residual = True
    n_in = 8

    def __init__(self, scale, dt, phys_params, flex, width=128, depth=3):
        super().__init__(scale, dt, phys_params, flex, width, depth)
        self.net = _mlp(self.n_in, width, depth, zero_last=True)

    def acc_from_features(self, feat):
        a_phys = feat[..., 6:8] * self.out_scale
        return a_phys + self.net_out(feat) * self.res_std


def build_model(config):
    cls = ResidualNSS if config["kind"] == "residual" else NSS
    return cls(config["scale"], config["dt"], config["phys_params"], config["flex"], config["width"], config["depth"])
