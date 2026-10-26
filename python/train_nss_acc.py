#!/usr/bin/env python3
"""加速度を状態に持つ残差型 NSS を学習する（docs/requirements_nss_resid_state.md）。

    .venv/bin/python python/train_nss_acc.py --width 64 --run acc_w64_s0

NN の入力 [acc, τ, q, dq]（4 列、標準化）、出力 Δacc（train の標準偏差で標準化）。損失は標準化した Δacc の MSE。
学習率・早期終了などは train_nss.py と同じ。出力: data/train_nss8kacc/<run>/{model.pt, history.json, config.json}（git 管理外）
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
from nss8k.data import load  # noqa: E402
from nss8kacc.data import build_xy  # noqa: E402
from nss8kacc.model import AccNSS  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/export/j2_nss8k.h5")
    p.add_argument("--run", default="run")
    p.add_argument("--out-root", default="data/train_nss8kacc")
    p.add_argument("--width", type=int, default=64)
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


@torch.no_grad()
def eval_loss(model, F, T, bs=65536):
    model.eval()
    tot = 0.0
    for i in range(0, len(F), bs):
        tot += float(((model.net_out(F[i:i + bs])[:, 0] - T[i:i + bs]) ** 2).sum())
    return tot / T.numel()


def main(argv=None):
    a = parse(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    out = os.path.join(a.out_root, a.run)
    os.makedirs(out, exist_ok=True)

    d = load(a.data, splits=("train", "val"))
    scale, phys, dt = d["scale"], d["physics"], d["dt"]
    Ftr, Dtr = build_xy(d["splits"]["train"], scale)
    Fva, Dva = build_xy(d["splits"]["val"], scale)
    del d["splits"]
    if a.max_train and a.max_train < len(Ftr):
        idx = torch.from_numpy(np.sort(np.random.permutation(len(Ftr))[:a.max_train]))
        Ftr, Dtr = Ftr[idx], Dtr[idx]
    model = AccNSS(scale, dt, phys, a.width, a.depth)
    model.x_mean.copy_(Ftr.mean(0)); model.x_std.copy_(Ftr.std(0))
    model.d_std.copy_(Dtr.std().clamp_min(1e-6))
    unit = float(model.d_std)                            # 損失の平方根 × unit = Δacc の RMS [rad/s²]
    Ttr, Tva = Dtr / unit, Dva / unit
    n_param = sum(p.numel() for p in model.parameters())
    print(f"AccNSS、幅 {a.width} × 深さ {a.depth}、パラメータ {n_param:,}、学習 {len(Ftr):,} 行 / 検証 {len(Fva):,} 行、"
          f"Δacc の標準偏差 {unit:.4f} rad/s²", flush=True)
    print(f"開始時点（acc₁ = acc）の val RMS {math.sqrt(eval_loss(model, Fva, Tva)) * unit:.4f} rad/s²", flush=True)

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
            loss = torch.mean((model.net_out(Ftr[idx])[:, 0] - Ttr[idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss.detach()) * len(idx); cnt += len(idx); step += 1
        vl = eval_loss(model, Fva, Tva)
        rec = {"epoch": ep, "train_loss": run / cnt, "val_loss": vl, "val_rms_ddq": math.sqrt(vl) * unit,
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
        print(f"epoch {ep:3d}  train {rec['train_loss']:.3e}  val {vl:.3e}  val RMS {rec['val_rms_ddq']:.4f} rad/s²  "
              f"lr {rec['lr']:.2e}  {rec['time_s']:.0f}s{flag}", flush=True)
        with open(os.path.join(out, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)
        if bad >= a.patience:
            print(f"早期終了（{a.patience} エポック改善なし）")
            break
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"args": vars(a), "n_param": n_param, "best_val_loss": best, "steps": step,
                   "train_time_s": time.time() - t0, "n_train": int(len(Ftr)), "n_val": int(len(Fva))}, f, indent=1)
    print(f"完了: best val RMS {math.sqrt(best) * unit:.4f} rad/s²、{step} ステップ、学習時間 {time.time() - t0:.0f}s、保存 {out}")


if __name__ == "__main__":
    main()
