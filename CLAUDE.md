# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 言語

ユーザーとのやり取りは日本語で行うこと。

## 参照先

- **`docs/handover.md`（フォーク元 j2-state-data-gen からの引継ぎ。最初に読む）**
- `README.md`（使い方）、`docs/dataset.md`（データセット仕様と注意点）、`docs/plan.md`（進捗）、`docs/phase*_report.md`（各フェーズの結果）
- 再現確認: `matlab -batch "addpath('tools'); j2_repro_check()"`（キャッシュなしで再生成し、保存済み 8 kHz データとのビット一致を確認）

## 作業サイクル（必須）

1. issue 作成 → ブランチ作成 → 作業
2. 作業終了後、**ユーザーに報告**（ここで止まる）
3. **ユーザー指示があって初めて**コミットし、結果を issue にコメント投稿
4. **ユーザー指示があって初めて** PR を作成 → ユーザーが承認

- **main への直接 push は禁止**。ユーザーの明示指示なしにコミット・push・PR 作成・issue 投稿をしない。
- GitHub リポジトリは現時点で未作成（ユーザーが用意する）。この間は git 操作・issue 作成をしない。
- 実装は「要件定義 → 計画 → 実装」の順（`~/matlab-projects/robotArmSurrogate-matlab` の流儀）。いきなり実装しない。

## 決定事項

- 実装は **MATLAB/Simulink**（このリポジトリ内）。`robotArmSurrogate-matlab` の `c8_params` やプラント関連は path 参照のみ（コピーしない）。Deep Learning / System Identification / Statistics and ML Toolbox は使用禁止。
- 対象は EPSON C8-A901S の J2。**他軸を固定した J2 単軸の別モデル**を作る。固定姿勢は J1, J4〜J6 = 0、**J3 = 75.0684°**（θ=0 を倒立平衡点にする値。J3=0 だと平衡点が θ≈+11° にずれる）。この姿勢で J2 の重力トルクは −mgL·sin θ（mgL=73.25 N·m）。
- 固定軸の実現は **URDF の該当 joint を fixed に後処理して smimport する方式（A）**に決定（`spike/phase0_fixA.m`、固定角は origin の rpy に畳み込む）。
- リミット接触: 基本の 338 本はバリアでリミットに接触しない。接触は `contact_fall/drive/bln`（56 本、バリアをリミットの外側に移し、速度バリアは 80%）で取る。接触遷移（`atLimit`）は既定で学習対象に含め、`j2BuildFlat` の `excludeAtLimit=true` で除外できる。
- 引継ぎ資料の `simulate_joint2` と Python + matlab.engine 構成は仮のもので、採用しない。

## コマンド（MATLAB R2025a）

```matlab
addpath('config'); addpath(genpath('src'));
c   = j2_setup_path();                      % 親資産と本リポジトリを path に追加し、設定を返す
mdl = buildJ2Plant();                       % J2 単軸プラントを生成（data/models/j2_plant.slx、約 15 s）
out = simJ2(mdl, tVec, tau, q0, dq0);       % トルク列を与えて実行 → out.t, q, dq, tau
results = runtests('test/test_j2_plant.m'); % 物理検証（13 件、約 3 分）
S   = j2ScenarioSet('full');                % シナリオ表（338 本）。'small' は動作確認用（19 本）
raw = runJ2Scenario(S(k), struct('excite',mE,'closed',mC));   % 1 シナリオを 8 kHz で実行
%   mE = buildJ2Excitation(); mC = buildJ2ClosedLoop();  （開ループ励振 / PD+FF 閉ループ）
```

- テスト: `test_j2_signals`（約 3 s）、`test_j2_scenarioset`（約 10 s）、`test_j2_scenarios`（約 250 s、実行を伴う）
- J2 の解析モデル: `M·ddq = τ + mgL·sin θ − Bv·dq − Fc·tanh(dq/ε)`（M=6.09 kg·m²（armature 込み）、mgL=73.25 N·m）。係数は `j2Params()`

- データセット生成: `ds = genJ2Dataset(struct('preset','small'))`（約 6 分）。`'full'` は約 6〜7 時間かかる。シナリオ単位で `data/cache/` にキャッシュされ、中断しても同じ呼び出しで再開できる
- 長時間生成は別プロセスで: `matlab -batch "addpath('tools'); j2_gen_worker('full', i, n)"`（n 分割の i 番目を担当）。ワーカー 1 本で約 3.7 GB 使うため、メモリが足りなければ並列にしない。生成後は `j2_finish('full')` で組み立て・検証・カバレッジ評価
- 他言語向け: `j2ExportFlat(ds, 'x.h5')`（HDF5、h5py から (N,d) で見える。インデックスは 0 始まり）または `x.mat`（v7）。Python 側の検証は `python tools/verify_flat_export.py x.h5 x.mat`
- 検証: `validateJ2Data(ds)`、フラット化: `[X,Y] = j2BuildFlat(ds,'train')`（`features='spec'` で引継ぎ資料の正規化形式）、カバレッジ: `j2Coverage(ds, outDir)`（PNG 7 枚と指標）
- 1 kHz 変換は、状態は瞬時値の間引き、トルクは区間平均（フィルタなし。遷移データのため）。リミット拘束（リミットから 0.1° 以内または外側）の遷移にはフラグを付けるが、既定では除外しない
- 学習・評価（Python）: `.venv`（システムの PyTorch ＋ h5py）。`.venv/bin/python python/train.py --run <名前>` → `data/train/<名前>/`、`.venv/bin/python python/evaluate.py --run <名前> --report-dir reports/...`。単体テストは `.venv/bin/python -m unittest discover -s python/tests`。学習は CPU で 1 エポック約 1 分（40 エポックで約 40 分）。長いので `nohup` でバックグラウンド実行する
- 学習の損失・早期終了: 接触の外れ値で val 損失が大きく揺れる。残差型では `--patience` を大きくする（既定 8 では接触を学ぶ前に打ち切られうる）。`residual` の損失・標準化の選び方は `docs/residual_report.md`
- **全長の開ループ再生で評価しない**: θ=0 まわりが不安定（成長率 約 3.5/s）で、正確なモデルでも数秒で発散する。長い区間は制御器（PD＋FF）を閉ループにして評価する（`python/j2nsm/evaluate_lib.py`）
- 単一テストは `runtests('test/test_j2_plant.m','Name','test_j2_plant/gravityIsPendulum')`
- 親資産のパスは環境変数 `J2_PARENT_DIR` で上書き可（既定 `~/matlab-projects/robotArmSurrogate-matlab`）
- MATLAB MCP サーバー経由の実行は 120 s でバックグラウンド化される。長い実行は完了通知を待つ
- 8 kHz のシミュレーションは実時間の約 5〜6 倍かかる（Parallel Computing Toolbox 未許諾のため直列）

## このリポジトリの位置づけ
`yk142/j2-state-data-gen`（J2 単軸の学習データ生成〜学習・評価まで完了。2026-10-10 時点の main = `834680a`）から**フォーク**した、**8 kHz の NSS（Neural State Space）**の検討用リポジトリ。フォーク元は remote `upstream`。上記「決定事項」「コマンド」は J2 版のもので、8 kHz 版で変える点は `docs/handover.md` を参照。

## 現状
フォーク直後。8 kHz 版の要件は未確定（`docs/handover.md` の「未決事項」）。git 管理外のデータ（`data/cache/` の 8 kHz 生データなど）はフォーク元にある。
