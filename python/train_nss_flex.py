#!/usr/bin/env python3
"""弾性 J2 の 4 状態 NSS を学習する（docs/requirements_nss_flex.md、docs/plan_nss_flex_train.md）。

    .venv/bin/python python/train_nss_flex.py --model B --width 128 --epochs 5 --run B_w128_screen

--model A : 純 NN。[ddq, ddqm] を固定定数 [τpeak/JL, τpeak/Jm] で割った値を直接予測する。
--model B : 残差型。[ddq, ddqm] = 二慣性の解析モデル + NN の補正。補正は train の列ごとの標準偏差で標準化、最後の層は 0 初期化。
損失は標準化した 2 出力の MSE の平均。学習率はステップ数ベース（ウォームアップ付きコサイン）。
出力: data/train_nss8kflex/<run>/{model.pt, history.json, config.json}（git 管理外）
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nss8kflex.data import load  # noqa: E402
from nss8kflex.model import NSS, ResidualNSS  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/export/j2_nss8kflex.h5")
    p.add_argument("--run", default="run")
    p.add_argument("--out-root", default="data/train_nss8kflex")
    p.add_argument("--model", choices=("A", "B"), default="A")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-train", type=int, default=0, help="学習行数の上限（動作確認用。0 は全件）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


def build_xy(model_kind, split, out_scale):
    """split の辞書 → (特徴量 F, 標準化前の目標 T)。T は A: Y_scaled、B: Y − a_phys [rad/s²]（N, 2）。"""
    X = torch.from_numpy(split["X"])
    if model_kind == "A":
        return X, torch.from_numpy(split["Y_scaled"])
    a = torch.from_numpy(split["a_phys"])
    return torch.cat([X, a / out_scale], dim=1), torch.from_numpy(split["Y"]) - a


@torch.no_grad()
def eval_loss(model, F, T, bs=65536):
    """標準化した目標の列ごとの MSE（2 要素）。"""
    model.eval()
    tot = torch.zeros(2, dtype=torch.float64)
    for i in range(0, len(F), bs):
        tot += ((model.net_out(F[i:i + bs]) - T[i:i + bs]) ** 2).sum(0).double()
    return tot / len(F)


def main(argv=None):
    a = parse(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    out = os.path.join(a.out_root, a.run)
    os.makedirs(out, exist_ok=True)

    d = load(a.data, splits=("train", "val"))
    tr, va = d["splits"]["train"], d["splits"]["val"]
    if a.max_train and a.max_train < len(tr["X"]):
        idx = np.sort(np.random.permutation(len(tr["X"]))[:a.max_train])
        tr = {k: v[idx] for k, v in tr.items()}
    scale, phys, flex, dt = d["scale"], d["physics"], d["flex"], d["dt"]
    cls = ResidualNSS if a.model == "B" else NSS
    model = cls(scale, dt, phys, flex, a.width, a.depth)
    model.tw_std.copy_(torch.from_numpy(d["stats"]["x_std"][2:4]).float())
    Ftr, Ttr = build_xy(a.model, tr, model.out_scale)
    Fva, Tva = build_xy(a.model, va, model.out_scale)
    model.x_mean.copy_(Ftr.mean(0)); model.x_std.copy_(Ftr.std(0))
    if a.model == "B":
        model.res_std.copy_(Ttr.std(0).clamp_min(1e-6))
        Ttr, Tva = Ttr / model.res_std, Tva / model.res_std
        unit = model.res_std.double()
    else:
        unit = model.out_scale.double()                     # √損失 × unit = 各出力の RMS [rad/s²]
    n_param = sum(p.numel() for p in model.parameters())
    print(f"モデル {a.model}（{cls.__name__}）、幅 {a.width} × 深さ {a.depth}、パラメータ {n_param:,}、"
          f"学習 {len(Ftr):,} 行 / 検証 {len(Fva):,} 行、単位 {unit.tolist()} rad/s²", flush=True)
    if a.model == "B":
        print(f"開始時点（解析モデルそのもの）の val RMS {(eval_loss(model, Fva, Tva).sqrt() * unit).tolist()} rad/s²", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    spe = math.ceil(len(Ftr) / a.batch)
    total = a.epochs * spe
    warm = min(200, total // 10)

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        t = (s - warm) / max(1, total - warm)
        return a.lr_min + 0.5 * (a.lr - a.lr_min) * (1 + math.cos(math.pi * t))

    hist, best, bad, step = [], float("inf"), 0, 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(Ftr))
        run, cnt = 0.0, 0
        for i in range(0, len(Ftr), a.batch):
            idx = perm[i:i + a.batch]
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss = torch.mean((model.net_out(Ftr[idx]) - Ttr[idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss.detach()) * len(idx); cnt += len(idx); step += 1
        vlc = eval_loss(model, Fva, Tva)
        vl = float(vlc.mean())
        rec = {"epoch": ep, "train_loss": run / cnt, "val_loss": vl, "val_rms": (vlc.sqrt() * unit).tolist(),
               "lr": lr_at(step - 1), "time_s": time.time() - t0}
        hist.append(rec)
        flag = ""
        if vl < best:
            best, bad = vl, 0
            torch.save({"state_dict": model.state_dict(), "config": model.config(), "epoch": ep, "val_loss": vl},
                       os.path.join(out, "model.pt"))
            flag = " *"
        else:
            bad += 1
        print(f"epoch {ep:3d}  train {rec['train_loss']:.3e}  val {vl:.3e}  val RMS {[round(x, 4) for x in rec['val_rms']]} rad/s²  "
              f"lr {rec['lr']:.2e}  {rec['time_s']:.0f}s{flag}", flush=True)
        with open(os.path.join(out, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)
        if bad >= a.patience:
            print(f"早期終了（{a.patience} エポック改善なし）")
            break
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"args": vars(a), "n_param": n_param, "best_val_loss": best, "steps": step,
                   "train_time_s": time.time() - t0, "n_train": int(len(Ftr)), "n_val": int(len(Fva))}, f, indent=1)
    print(f"完了: best val loss {best:.3e}、{step} ステップ、学習時間 {time.time() - t0:.0f}s、保存 {out}")


if __name__ == "__main__":
    main()
