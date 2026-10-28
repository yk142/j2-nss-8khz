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
from nss8kacc.data import build_xy, targets, window_starts  # noqa: E402
from nss8kacc.model import AccNSS  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/export/j2_nss8k.h5")
    p.add_argument("--run", default="run")
    p.add_argument("--out-root", default="data/train_nss8kacc")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--acc-noise", type=float, default=0.0,
                   help="学習時に入力 acc へ加えるガウスノイズの標準偏差 [rad/s²]。標的は y − acc_ノイズ入り（0 は基本形）")
    p.add_argument("--rollout-k", type=int, default=0, help="ロールアウト損失の窓長 K（0 は使わない）")
    p.add_argument("--rollout-mode", choices=("teacher", "full"), default="teacher",
                   help="teacher: pos/vel は真値、acc だけ自己回帰。full: acc から積分した pos/vel も自己回帰（トルクは真値）")
    p.add_argument("--rollout-batch", type=int, default=256, help="1 ステップあたりの窓の数")
    p.add_argument("--rollout-weight", type=float, default=1.0, help="ロールアウト損失の重み（1 ステップ損失に加える）")
    p.add_argument("--feat", choices=("q", "sincos"), default="q", help="位置の入力: 生の q、または sin q・cos q")
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


@torch.no_grad()
def eval_rollout(model, F, Y, starts, k, unit, mode):
    model.eval()
    return sum(float(rollout_loss(model, F, Y, starts[i:i + 4096], k, unit, mode)) * len(starts[i:i + 4096])
               for i in range(0, len(starts), 4096)) / len(starts)


def rollout_loss(model, F, Y, starts, k, unit, mode, noise=0.0):
    """starts から k ステップ自己回帰した損失（標準化した acc 誤差の 2 乗の平均）。窓の先頭の acc は真値（noise > 0 なら加える）。
    full では pos/vel の真値からのずれ（dq_dev, q_dev）を誤差から積分して入力へ戻し、状態の誤差も損失に足す
    （一定の誤差 e に対する成長 j·dt·e、(j²/2)·dt²·e で割って、acc 誤差と同じ大きさにそろえる）。"""
    sc, dt = model.scale, model.dt
    ddq, dth = sc["DDQ_SCALE"], sc["DTHETA_MAX"]
    acc = F[starts, 0] * ddq
    if noise > 0:
        acc = acc + torch.randn_like(acc) * noise
    dq_dev = q_dev = torch.zeros_like(acc)
    loss = 0.0
    for j in range(k):
        idx = starts + j
        f = F[idx].clone()
        f[:, 0] = acc / ddq
        if mode == "full":
            f[:, 2] = f[:, 2] + q_dev
            f[:, 3] = f[:, 3] + dq_dev / dth
        acc = acc + model.net_out(f)[:, 0] * unit
        e = acc - Y[idx]
        loss = loss + torch.mean((e / unit) ** 2)
        if mode == "full":
            dq_new = dq_dev + dt * e
            q_dev = q_dev + 0.5 * dt * (dq_dev + dq_new)
            dq_dev = dq_new
            n = j + 1
            loss = loss + torch.mean((dq_dev / (n * dt * unit)) ** 2) + torch.mean((q_dev / (0.5 * n * n * dt * dt * unit)) ** 2)
    return loss / (k * (3 if mode == "full" else 1))


def main(argv=None):
    a = parse(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    out = os.path.join(a.out_root, a.run)
    os.makedirs(out, exist_ok=True)

    d = load(a.data, splits=("train", "val"))
    scale, phys, dt = d["scale"], d["physics"], d["dt"]
    Ftr, Dtr = build_xy(d["splits"]["train"], scale, a.feat)
    Fva, Dva = build_xy(d["splits"]["val"], scale, a.feat)
    if a.max_train and a.max_train < len(Ftr):
        idx = torch.from_numpy(np.sort(np.random.permutation(len(Ftr))[:a.max_train]))
        Ftr, Dtr = Ftr[idx], Dtr[idx]
    roll = a.rollout_k > 0
    if roll:
        assert a.feat == "q", "ロールアウト損失は feat=q のみ"
        Ytr, Yva = targets(d["splits"]["train"]), targets(d["splits"]["val"])
        Str, Sva = window_starts(d["splits"]["train"], a.rollout_k), window_starts(d["splits"]["val"], a.rollout_k)
        g = torch.Generator().manual_seed(1234)
        Sva = Sva[torch.randperm(len(Sva), generator=g)[:20000]]
        assert not a.max_train, "ロールアウト損失と --max-train は併用しない"
    del d["splits"]
    model = AccNSS(scale, dt, phys, a.width, a.depth, a.feat)
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
            Fb, Tb = Ftr[idx], Ttr[idx]
            if a.acc_noise > 0:                          # acc にノイズを足し、出力 acc₁ が真値になるよう標的を補正する
                n = torch.randn(len(idx)) * a.acc_noise
                Fb = Fb.clone(); Fb[:, 0] += n / scale["DDQ_SCALE"]
                Tb = Tb - n / unit
            loss = torch.mean((model.net_out(Fb)[:, 0] - Tb) ** 2)
            if roll:
                st = Str[torch.randint(len(Str), (a.rollout_batch,))]
                loss = loss + a.rollout_weight * rollout_loss(model, Ftr, Ytr, st, a.rollout_k, unit, a.rollout_mode, a.acc_noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss.detach()) * len(idx); cnt += len(idx); step += 1
        vl = eval_loss(model, Fva, Tva)
        vr = eval_rollout(model, Fva, Yva, Sva, a.rollout_k, unit, a.rollout_mode) if roll else 0.0
        vl1 = vl
        vl = vl + a.rollout_weight * vr                    # 最良エポックの選択は 1 ステップ + ロールアウトの検証損失
        rec = {"epoch": ep, "train_loss": run / cnt, "val_loss": vl, "val_roll": vr, "val_rms_ddq": math.sqrt(vl1) * unit,
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
