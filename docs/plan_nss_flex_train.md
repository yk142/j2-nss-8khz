# 実装計画 — 弾性 J2 の 4 状態 NSS（issue #33）

要件は `docs/requirements_nss_flex.md`（確定）。ここでは実装の分割と、要件に書いていない細部の決めだけを書く。

## 1. 構成

| 部品 | 内容 |
|---|---|
| `src/data/j2ExportNss8kFlex.m` | `nss8kflex` を HDF5 に書き出す。X_raw = [q dq tw dtw tau]、Y = [ddq ddqm]、a_phys 2 列、`/flex/{Jm,JL,k,c}` |
| `python/nss8kflex/` | `data.py`（読み込み・系列復元）、`physics.py`（二慣性の解析モデル）、`model.py`（A / B）、`evaluate_lib.py`（ゲート） |
| `python/train_nss_flex.py` | 学習（剛体版 `train_nss.py` と同じステップ数ベースの学習率・早期終了） |
| `python/evaluate_nss_flex.py` | ゲート (a)(b)(c)(d) の判定と JSON 出力 |
| `python/tests/test_nss8kflex.py` | 単体テスト |

## 2. 細部の決め

- **特徴量**: `[sin q, cos q, dq/5.236, tw/s_tw, dtw/s_dtw, tau/860.4]`。`s_tw`, `s_dtw` は train の標準偏差（`/stats/x_std`）。B はさらに `a_phys/DDQ_SCALE` の 2 列を足す（8 列）
- **出力の標準化**: A は `[τpeak/JL, τpeak/Jm]` で割る固定定数。B は残差を train の列ごとの標準偏差で割る
- **損失**: 標準化した 2 出力の MSE の平均
- **状態の更新**: `q, qm` を倍精度で持ち、`tw = qm − q` は倍精度で計算してから単精度の特徴量にする
- **HDF5 の精度**: X_raw は tw, dtw を倍精度で計算してから single で保存（q, qm を別々に保存すると、ねじり 1e-3 rad が single の丸めに埋もれるため）
- **(c) 閉ループ**: モータ角 PD（wn = 30、ζ = 1、8 kHz、Kp = M·wn²、Kd = 2ζM·wn、±τpeak で飽和）＋ `j2FeedforwardFlex` と同じ FF を、qref の数値微分から Python で作る。制御器の再現は、真値の状態に当てたトルクと記録トルクの差で確認する
- **(b) 窓**: 4,000 ステップ。θ と twist の RMSE を測る。ねじりの閾値は、解析モデルの値の 1.0 倍（要件 §5 の「解析モデルの値から決める」）
- **(a)**: 2 出力それぞれの RMSE が、解析モデルの 1.0 倍以内
- **(d)**: 剛体版と同じバッチ 1,024 のロールアウト。出力が 2 列に増えるため、満たさなければ幅を下げる

## 3. 実行順

1. エクスポートと単体テスト
2. スクリーニング（{A, B} × 幅 {64, 128, 256} × 5 エポック × 1 シード）
3. 本学習（勝った幅 × 20 エポック × 3 シード）。長いので、スクリーニングの結果を報告してから判断を仰ぐ
4. 評価（ゲート）
