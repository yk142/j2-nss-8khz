"""弾性（二慣性）の加速度状態 NSS（docs/requirements_nss_acc_flex.md）。

状態 acc = [ddq, ddqm]（直前の区間の加速度）、入力 [τ, q, dq, tw, dtw]。NN の入力は 7 列
[ddq/s_L, ddqm/s_m, τ, q, dq, tw/s_tw, dtw/s_dtw]、出力は Δacc（2 列）、acc₁ = acc + Δacc。最後の層は 0 初期化。
ロールアウト: 速度は前進オイラー（リンク・モータとも）、位置は台形則。ねじり tw = qm − q は倍精度で作ってから単精度にする。
step_physical は acc を内部状態として持つ。ロールアウトの前に reset() を呼ぶ（evaluate_lib の rollout・run_c4 が呼ぶ）。
最初の acc は二慣性の解析モデルの加速度で初期化する。
"""
import torch
import torch.nn as nn

from nss8kflex.physics import FlexPhysics


def _mlp(d_in, width, depth):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    last = nn.Linear(d, 2)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class AccFlexNSS(nn.Module):
    kind = "accflex"
    residual = False          # evaluate の kind 判定用（解析モデルを使わない）
    n_in = 7

    def __init__(self, scale, dt, phys_params, flex, width=64, depth=3):
        super().__init__()
        self.scale, self.dt = dict(scale), float(dt)
        self.phys_params, self.flex = dict(phys_params), dict(flex)
        self.width, self.depth = int(width), int(depth)
        self.register_buffer("x_mean", torch.zeros(self.n_in))
        self.register_buffer("x_std", torch.ones(self.n_in))
        self.register_buffer("d_std", torch.ones(2))                 # Δacc の列ごとの標準偏差 [rad/s²]
        self.register_buffer("tw_std", torch.ones(2))                # [s_tw, s_dtw]
        self.register_buffer("out_scale", torch.tensor([phys_params["tauPeak"] / flex["JL"],
                                                        phys_params["tauPeak"] / flex["Jm"]]))
        self.net = _mlp(self.n_in, width, depth)
        self._phys = FlexPhysics(self.phys_params, self.flex, self.dt)
        self._acc = None

    def config(self):
        return {"kind": self.kind, "width": self.width, "depth": self.depth, "scale": self.scale, "dt": self.dt,
                "phys_params": self.phys_params, "flex": self.flex}

    def features(self, aq, am, q, dq, tw, dtw, tau):
        s = self.scale
        return torch.stack([aq / self.out_scale[0].double(), am / self.out_scale[1].double(), tau / s["TAU_MAX"], q,
                            dq / s["DTHETA_MAX"], tw / self.tw_std[0].double(), dtw / self.tw_std[1].double()],
                           dim=-1).float()

    def net_out(self, feat):
        return self.net((feat - self.x_mean) / self.x_std)

    def delta_acc(self, aq, am, q, dq, tw, dtw, tau):
        """(Δddq, Δddqm) [rad/s²]（倍精度、(..., 2)）。"""
        return (self.net_out(self.features(aq, am, q, dq, tw, dtw, tau)) * self.d_std).to(q.dtype)

    def reset(self):
        self._acc = None

    def step_physical(self, q, dq, qm, dqm, tau):
        tw, dtw = qm - q, dqm - dq
        if self._acc is None or self._acc[0].shape != q.shape:
            self._acc = self._phys.accel(q, dq, tw, dtw, tau)
        aq, am = self._acc
        d = self.delta_acc(aq, am, q, dq, tw, dtw, tau)
        aq1, am1 = aq + d[..., 0], am + d[..., 1]
        dq1, dqm1 = dq + self.dt * aq1, dqm + self.dt * am1
        q1, qm1 = q + 0.5 * self.dt * (dq + dq1), qm + 0.5 * self.dt * (dqm + dqm1)
        self._acc = (aq1, am1)
        return q1, dq1, qm1, dqm1


def build_model(config):
    return AccFlexNSS(config["scale"], config["dt"], config["phys_params"], config["flex"], config["width"], config["depth"])
