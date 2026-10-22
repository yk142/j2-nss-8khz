"""NSS のモデル。出力は 1 列（ddq）、更新はオイラー（q1 = q + Δt·dq、dq1 = dq + Δt·ddq）。

  NSS          (A) 純 NN: 入力 X = [sin q, cos q, dq/DTHETA_MAX, tau/TAU_MAX]（4 列）、目標 Y_scaled = ddq/DDQ_SCALE
  ResidualNSS  (B) 残差型: 入力 X + a_phys/DDQ_SCALE（5 列）、ddq = a_phys + NN(x)·res_std。最後の層を 0 初期化
                （学習開始時点は解析モデルそのもの）
入力は train の平均・標準偏差で標準化する（バッファとして保存）。
"""
import torch
import torch.nn as nn

from .physics import Physics


def _mlp(d_in, width, depth, zero_last=False):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    last = nn.Linear(d, 1)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class _Base(nn.Module):
    n_in = 4
    residual = False

    def __init__(self, scale, dt, phys_params, width=128, depth=3):
        super().__init__()
        self.scale, self.dt = dict(scale), float(dt)
        self.phys_params = dict(phys_params)
        self.width, self.depth = int(width), int(depth)
        self.register_buffer("x_mean", torch.zeros(self.n_in))
        self.register_buffer("x_std", torch.ones(self.n_in))
        self.register_buffer("res_std", torch.ones(1))
        self._phys = Physics(self.phys_params, self.dt)

    def config(self):
        return {"kind": self.kind, "width": self.width, "depth": self.depth, "scale": self.scale,
                "dt": self.dt, "phys_params": self.phys_params}

    # ---- 特徴量（モデル入力、標準化前）----
    def features(self, q, dq, tau):
        s = self.scale
        cols = [torch.sin(q), torch.cos(q), dq / s["DTHETA_MAX"], tau / s["TAU_MAX"]]
        if self.residual:
            cols.append(self._phys.accel(q, dq, tau) / s["DDQ_SCALE"])
        return torch.stack(cols, dim=-1).float()

    def net_out(self, feat):
        return self.net((feat - self.x_mean) / self.x_std)

    def ddq_from_features(self, feat):
        """特徴量 → ddq [rad/s²]。"""
        raise NotImplementedError

    def ddq(self, q, dq, tau):
        return self.ddq_from_features(self.features(q, dq, tau)).to(q.dtype)

    def step_physical(self, q, dq, tau):
        """物理量（rad, rad/s, N·m）の状態を 1 ステップ進める（オイラー）。"""
        return q + self.dt * dq, dq + self.dt * self.ddq(q, dq, tau)


class NSS(_Base):
    kind = "nss"
    residual = False
    n_in = 4

    def __init__(self, scale, dt, phys_params, width=128, depth=3):
        super().__init__(scale, dt, phys_params, width, depth)
        self.net = _mlp(self.n_in, width, depth)

    def ddq_from_features(self, feat):
        return self.net_out(feat)[..., 0] * self.scale["DDQ_SCALE"]


class ResidualNSS(_Base):
    kind = "residual"
    residual = True
    n_in = 5

    def __init__(self, scale, dt, phys_params, width=128, depth=3):
        super().__init__(scale, dt, phys_params, width, depth)
        self.net = _mlp(self.n_in, width, depth, zero_last=True)

    def ddq_from_features(self, feat):
        a_phys = feat[..., 4] * self.scale["DDQ_SCALE"]
        return a_phys + self.net_out(feat)[..., 0] * self.res_std[0]


def build_model(config):
    """checkpoint の config から、重みを読み込む前のモデルを作る。"""
    cls = ResidualNSS if config["kind"] == "residual" else NSS
    return cls(config["scale"], config["dt"], config["phys_params"], config["width"], config["depth"])
