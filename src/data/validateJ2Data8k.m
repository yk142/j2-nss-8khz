function R = validateJ2Data8k(ds, verbose, forkCacheDir)
%VALIDATEJ2DATA8K 8 kHz 接触なしデータセット（preset nss8k）の妥当性を検証する。
%   R = VALIDATEJ2DATA8K(ds, verbose, forkCacheDir)   docs/requirements_8khz.md §5
%   forkCacheDir（任意）: フォーク元の 8 kHz キャッシュのフォルダ。あれば、同名のシナリオ
%     （full_<name>.mat。nss8k_small は small_<name>.mat）と single に丸めて一致するかを調べる。
%   R.checks: 検査名・合否・値の構造体配列、R.nErrors: 不合格数、R.metrics: 参考指標。
%   検査:
%     有限性 / トルク飽和内 / 速度制限内 / 全サンプルがリミットから 0.1° を超えて内側 /
%     接触・nearlimit のパターンを含まない / 配列長の整合 / シナリオ名の一意 / split の整合 /
%     正規化統計が train 由来 / 解析モデルとの 1 ステップ整合 / フォーク元キャッシュとの一致
if nargin < 2 || isempty(verbose), verbose = true; end
if nargin < 3, forkCacheDir = ''; end
jp = j2Params();
S = ds.scen;
dt = 1/jp.fsSim;
R.checks = struct('name',{},'pass',{},'value',{});
add = @(name, pass, value) addCheck(name, pass, value, verbose);
chk = {};

nBad = 0;  maxTau = 0;  maxDq = 0;  minDist = Inf;  nLim = 0;  nNear = 0;
for k = 1:numel(S)
    sc = S(k);
    nBad = nBad + nnz(~isfinite([sc.q; sc.dq; sc.tau]));
    maxTau = max(maxTau, max(abs(sc.tau)));  maxDq = max(maxDq, max(abs(sc.dq)));
    minDist = min(minDist, min(min(sc.q - jp.qMin, jp.qMax - sc.q)));
    nLim = nLim + nnz(sc.atLimit);  nNear = nNear + nnz(sc.nearLimit);
end
chk{end+1} = add('すべて有限', nBad == 0, nBad);
chk{end+1} = add('|tau| <= tauPeak', maxTau <= jp.tauPeak*(1 + 1e-9), maxTau);
chk{end+1} = add('|dq| < qdMax', maxDq < jp.qdMax, maxDq);
chk{end+1} = add('全サンプルがリミットから 0.1° を超えて内側（接触なし）', minDist > deg2rad(0.1), rad2deg(minDist));
chk{end+1} = add('atLimit の遷移が 0 件', nLim == 0, nLim);
R.metrics.nearLimitTransitions = nNear;          % 参考（リミットから 1° 以内。現状 0 の見込み。変更検知用）
R.metrics.minDistToLimitDeg = rad2deg(minDist);
isExcluded = startsWith({S.pattern},'contact_') | strcmp({S.pattern},'nearlimit');
chk{end+1} = add('contact_* と nearlimit を含まない', ~any(isExcluded), nnz(isExcluded));

okLen = all(arrayfun(@(s) numel(s.q) == numel(s.tau)+1 && numel(s.dq) == numel(s.tau)+1 ...
    && numel(s.atLimit) == numel(s.tau) && numel(s.nearLimit) == numel(s.tau) && s.nTransition == numel(s.tau), S));
chk{end+1} = add('配列長の整合 (q,dq = K+1, tau,atLimit,nearLimit = K)', okLen, nnz(~okLen));

names = {S.name};
chk{end+1} = add('シナリオ名が一意', numel(unique(names)) == numel(names), numel(names) - numel(unique(names)));
isBench = strcmp({S.phase},'benchmark');
chk{end+1} = add('benchmark は split=benchmark のみ', all(strcmp({S(isBench).split},'benchmark')) && ...
    all(ismember({S(~isBench).split},{'train','val','test'})), nnz(isBench));
% 小さい preset（本数の少ないパターンは train に寄せる）では val/test が無くてよい。100 本以上で必須
chk{end+1} = add('train/val/test がそろっている（100 本以上のとき）', numel(S) < 100 || all(ismember({'train','val','test'}, {S.split})), 0);
Sx = j2ScenarioSet(ds.manifest.preset);
chk{end+1} = add('シナリオ表が j2ScenarioSet(preset) と一致（名前・split・シード）', ...
    isequal({Sx.name},names) && isequal({Sx.split},{S.split}) && isequal([Sx.seed],[S.seed]), numel(Sx) - numel(S));

% 正規化統計: train の全遷移から再計算（累積で組み立てたものと照合）
n = 0;  sx = zeros(1,3);  sxx = zeros(1,3);  sy = 0;  syy = 0;
for k = find(strcmp({S.split},'train'))
    sc = S(k);  K = sc.nTransition;
    X = [sc.q(1:K), sc.dq(1:K), sc.tau];  y = diff(sc.dq)/dt;
    n = n + K;  sx = sx + sum(X,1);  sxx = sxx + sum(X.^2,1);  sy = sy + sum(y);  syy = syy + sum(y.^2);
end
xStd = sqrt((sxx - sx.^2/n)/(n-1));  yStd = sqrt((syy - sy^2/n)/(n-1));
chk{end+1} = add('正規化統計 = train の統計', max(abs(ds.stats.xMean - sx/n)./max(abs(sx/n),1e-9)) < 1e-9 && ...
    max(abs(ds.stats.xStd - xStd)./xStd) < 1e-9 && abs(ds.stats.ddqStd - yStd)/yStd < 1e-9, 0);

% 解析モデルとの 1 ステップ整合（x_k, tau_k で評価した加速度と実測の差）
num = 0;  den = 0;  numStep = 0;
for k = 1:numel(S)
    sc = S(k);  K = sc.nTransition;
    aMeas  = diff(sc.dq)/dt;
    aModel = (sc.tau + jp.mgL*sin(sc.q(1:K)) - jp.Bv*sc.dq(1:K) - jp.Fc*tanh(sc.dq(1:K)/jp.eps)) / jp.M;
    num = num + sum((aMeas - aModel).^2);  den = den + sum(aMeas.^2);
end
R.metrics.modelResidualRatio = sqrt(num/den);
R.metrics.modelResidualRMS = sqrt(num/sum([S.nTransition]));     % 暫定ゲート (a) の基準線（8 kHz、rad/s²）
chk{end+1} = add('解析モデルの 1 ステップ加速度残差 / 加速度 RMS < 0.5', R.metrics.modelResidualRatio < 0.5, R.metrics.modelResidualRatio);

% フォーク元キャッシュとの一致（single に丸めて完全一致）
if ~isempty(forkCacheDir)
    nCmp = 0;  nDiff = 0;  nMissing = 0;
    forkPrefix = 'full';  if strcmp(ds.manifest.preset,'nss8k_small'), forkPrefix = 'small'; end
    for k = 1:numel(S)
        fn = fullfile(forkCacheDir, sprintf('%s_%s.mat', forkPrefix, S(k).name));
        if ~isfile(fn), nMissing = nMissing + 1; continue; end
        F = load(fn);  e = F.e;  nCmp = nCmp + 1;
        same = numel(e.q) == numel(S(k).q) && isequal(single(S(k).q), e.q(:)) && ...
            isequal(single(S(k).dq), e.dq(:)) && isequal(single(S(k).tau), e.tau(1:numel(S(k).tau)));
        if ~same, nDiff = nDiff + 1; end
    end
    R.metrics.forkCompared = nCmp;  R.metrics.forkMissing = nMissing;
    chk{end+1} = add('フォーク元のキャッシュと一致（single に丸めて、比較した全シナリオ）', nCmp > 0 && nDiff == 0, [nCmp nDiff nMissing]);
end

R.checks = [chk{:}];
R.nErrors = nnz(~[R.checks.pass]);
R.metrics.nTransition = sum([S.nTransition]);
if verbose, fprintf('検証: %d 項目中 %d 件不合格（遷移 %d、リミットまでの最小距離 %.2f°、モデル残差 RMS %.3f rad/s²）\n', ...
    numel(R.checks), R.nErrors, R.metrics.nTransition, R.metrics.minDistToLimitDeg, R.metrics.modelResidualRMS); end
end

function c = addCheck(name, pass, value, verbose)
c = struct('name',name,'pass',logical(pass),'value',value);
if verbose, fprintf('  [%s] %s  (%s)\n', ternary(pass,'OK','NG'), name, mat2str(value,4)); end
end

function s = ternary(c,a,b)
if c, s = a; else, s = b; end
end
