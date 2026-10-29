#!/usr/bin/env python3
"""弾性 J2 の加速度状態 NSS を学習する（docs/requirements_nss_acc_flex.md。剛体版は train_nss_acc.py）。

    .venv/bin/python python/train_nss_accflex.py --width 64 --run accflex_w64_s0 [--acc-noise-rel 5] [--rollout-k 16]

NN の入力 [acc(2), τ, q, dq, tw, dtw]（7 列、標準化）、出力 Δacc（2 列、列ごとの標準偏差で標準化）。損失は標準化した 2 列の MSE の平均。
--acc-noise-rel r: 入力 acc の各列へ、Δacc の列ごとの標準偏差の r 倍のガウスノイズを加える（目標は y − acc_ノイズ入り）。
--rollout-k K: 同じシナリオの連続 K 行を、acc だけ自己回帰（pos/vel は真値）で進めた損失を足す（剛体 #49 の案 1）。
出力: data/train_nss8kaccflex/<run>/{model.pt, history.json, config.json}（git 管理外）
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
from nss8kaccflex.data import build_xy, targets, window_starts  # noqa: E402
from nss8kaccflex.model import AccFlexNSS  # noqa: E402
from nss8kflex.data import load  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/export/j2_nss8kflex.h5")
    p.add_argument("--run", default="run")
    p.add_argument("--out-root", default="data/train_nss8kaccflex")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--acc-noise-rel", type=float, default=0.0, help="入力 acc へのノイズの標準偏差（Δacc の標準偏差の倍数。0 は基本形）")
    p.add_argument("--rollout-k", type=int, default=0, help="ロールアウト損失の窓長 K（0 は使わない）")
    p.add_argument("--rollout-batch", type=int, default=128, help="1 ステップあたりの窓の数")
    p.add_argument("--rollout-weight", type=float, default=1.0)
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
    """標準化した Δacc の列ごとの MSE の平均（スカラー）と列ごとの値。"""
    model.eval()
    tot = torch.zeros(2, dtype=torch.float64)
    for i in range(0, len(F), bs):
        tot += ((model.net_out(F[i:i + bs]) - T[i:i + bs]) ** 2).sum(0).double()
    per = tot / len(F)
    return float(per.mean()), per


def rollout_loss(model, F, Y, starts, k, noise=None):
    """starts から k ステップ、acc だけ自己回帰した損失（標準化した acc 誤差の 2 乗の平均）。窓の先頭の acc は真値。
    noise: (2,) 先頭の acc に加えるノイズの標準偏差 [rad/s²]（None は加えない）。"""
    sc, ds = model.out_scale, model.d_std
    acc = F[starts, :2] * sc
    if noise is not None:
        acc = acc + torch.randn_like(acc) * noise
    loss = 0.0
    for j in range(k):
        idx = starts + j
        f = F[idx].clone()
        f[:, :2] = acc / sc
        acc = acc + model.net_out(f) * ds
        loss = loss + torch.mean(((acc - Y[idx]) / ds) ** 2)
    return loss / k


@torch.no_grad()
def eval_rollout(model, F, Y, starts, k):
    model.eval()
    return sum(float(rollout_loss(model, F, Y, starts[i:i + 4096], k)) * len(starts[i:i + 4096])
               for i in range(0, len(starts), 4096)) / len(starts)


def main(argv=None):
    a = parse(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    out = os.path.join(a.out_root, a.run)
    os.makedirs(out, exist_ok=True)

    d = load(a.data, splits=("train", "val"))
    scale, phys, flex, dt = d["scale"], d["physics"], d["flex"], d["dt"]
    model = AccFlexNSS(scale, dt, phys, flex, a.width, a.depth)
    tw_std = d["stats"]["x_std"][2:4].astype(np.float64)
    model.tw_std.copy_(torch.from_numpy(tw_std).float())
    osc = model.out_scale.double().numpy()
    Ftr, Dtr = build_xy(d["splits"]["train"], scale, osc, tw_std)
    Fva, Dva = build_xy(d["splits"]["val"], scale, osc, tw_std)
    roll = a.rollout_k > 0
    if roll:
        assert not a.max_train, "ロールアウト損失と --max-train は併用しない"
        Ytr, Yva = targets(d["splits"]["train"]), targets(d["splits"]["val"])
        Str, Sva = window_starts(d["splits"]["train"], a.rollout_k), window_starts(d["splits"]["val"], a.rollout_k)
        Sva = Sva[torch.randperm(len(Sva), generator=torch.Generator().manual_seed(1234))[:20000]]
    del d["splits"]
    if a.max_train and a.max_train < len(Ftr):
        idx = torch.from_numpy(np.sort(np.random.permutation(len(Ftr))[:a.max_train]))
        Ftr, Dtr = Ftr[idx], Dtr[idx]
    model.x_mean.copy_(Ftr.mean(0)); model.x_std.copy_(Ftr.std(0))
    model.d_std.copy_(Dtr.std(0).clamp_min(1e-6))
    unit = model.d_std                                    # 損失の平方根 × unit = 各列の Δacc の RMS [rad/s²]
    Ttr, Tva = Dtr / unit, Dva / unit
    noise = a.acc_noise_rel * unit if a.acc_noise_rel > 0 else None
    n_param = sum(p.numel() for p in model.parameters())
    print(f"AccFlexNSS、幅 {a.width} × 深さ {a.depth}、パラメータ {n_param:,}、学習 {len(Ftr):,} 行 / 検証 {len(Fva):,} 行、"
          f"Δacc の標準偏差 {unit.tolist()} rad/s²、ノイズ σ {None if noise is None else noise.tolist()}", flush=True)
    print(f"開始時点（acc₁ = acc）の val RMS {(eval_loss(model, Fva, Tva)[1].sqrt() * unit.double()).tolist()} rad/s²", flush=True)

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
            if noise is not None:                         # acc にノイズを足し、出力 acc₁ が真値になるよう標的を補正する
                n = torch.randn(len(idx), 2) * noise
                Fb = Fb.clone(); Fb[:, :2] += n / model.out_scale
                Tb = Tb - n / unit
            loss = torch.mean((model.net_out(Fb) - Tb) ** 2)
            if roll:
                st = Str[torch.randint(len(Str), (a.rollout_batch,))]
                loss = loss + a.rollout_weight * rollout_loss(model, Ftr, Ytr, st, a.rollout_k, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss.detach()) * len(idx); cnt += len(idx); step += 1
        vl1, vper = eval_loss(model, Fva, Tva)
        vr = eval_rollout(model, Fva, Yva, Sva, a.rollout_k) if roll else 0.0
        vl = vl1 + a.rollout_weight * vr                 # 最良エポックの選択は 1 ステップ + ロールアウトの検証損失
        rec = {"epoch": ep, "train_loss": run / cnt, "val_loss": vl, "val_roll": vr,
               "val_rms": (vper.sqrt() * unit.double()).tolist(), "lr": lr_at(step - 1), "time_s": time.time() - t0}
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
