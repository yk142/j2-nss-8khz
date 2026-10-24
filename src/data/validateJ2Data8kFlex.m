function R = validateJ2Data8kFlex(ds, verbose)
%VALIDATEJ2DATA8KFLEX 弾性 J2 の 8 kHz データセット（preset nss8kflex）の妥当性を検証する（docs/requirements_nss_flex.md §4.4）。
%   R = VALIDATEJ2DATA8KFLEX(ds, verbose)   R.checks: 名前・合否・値、R.nErrors、R.metrics
%   検査: 有限性 / トルク飽和内 / 速度制限内 / リンク角がリミットから 0.1° を超えて内側 / ねじりの上限 /
%         接触・nearlimit を含まない / 配列長 / シナリオ名の一意 / split / シナリオ表との一致 /
%         正規化統計 = train の統計 / 二慣性の解析モデルとの 1 ステップ整合（ddq・ddqm それぞれ）
if nargin < 2 || isempty(verbose), verbose = true; end
jp = j2Params();  flex = ds.flex;  S = ds.scen;  dt = 1/jp.fsSim;
R.checks = struct('name',{},'pass',{},'value',{});
add = @(name, pass, value) addCheck(name, pass, value, verbose);
chk = {};
nBad = 0;  maxTau = 0;  maxDq = 0;  maxDqm = 0;  minDist = Inf;  maxTw = 0;  nLim = 0;  nNear = 0;
for k = 1:numel(S)
    sc = S(k);
    nBad = nBad + nnz(~isfinite([sc.q; sc.dq; sc.qm; sc.dqm; sc.tau]));
    maxTau = max(maxTau, max(abs(sc.tau)));  maxDq = max(maxDq, max(abs(sc.dq)));  maxDqm = max(maxDqm, max(abs(sc.dqm)));
    minDist = min(minDist, min(min(sc.q - jp.qMin, jp.qMax - sc.q)));
    maxTw = max(maxTw, max(abs(sc.qm - sc.q)));
    nLim = nLim + nnz(sc.atLimit);  nNear = nNear + nnz(sc.nearLimit);
end
twMax = jp.tauPeak/flex.k;      % 最大トルクの静的たわみ
chk{end+1} = add('すべて有限', nBad == 0, nBad);
chk{end+1} = add('|tau| <= tauPeak', maxTau <= jp.tauPeak*(1 + 1e-9), maxTau);
chk{end+1} = add('|dq| < 1.3·qdMax（リンク。弾性で励振バリアを超えうる）', maxDq < 1.3*jp.qdMax, maxDq);
chk{end+1} = add('|dqm| < qdMax（モータ。バリアの制御対象）', maxDqm < jp.qdMax, maxDqm);
chk{end+1} = add('リンク角がリミットから 0.1° を超えて内側（接触なし）', minDist > deg2rad(0.1), rad2deg(minDist));
chk{end+1} = add('atLimit の遷移が 0 件', nLim == 0, nLim);
chk{end+1} = add('|ねじり| <= 3 × 最大トルクの静的たわみ（破綻の検出）', maxTw <= 3*twMax, [maxTw twMax]);
R.metrics.nearLimitTransitions = nNear;  R.metrics.minDistToLimitDeg = rad2deg(minDist);
R.metrics.maxTwistRad = maxTw;  R.metrics.staticTwistAtPeakRad = twMax;
isExcluded = startsWith({S.pattern},'contact_') | strcmp({S.pattern},'nearlimit');
chk{end+1} = add('contact_* と nearlimit を含まない', ~any(isExcluded), nnz(isExcluded));
okLen = all(arrayfun(@(s) numel(s.q) == numel(s.tau)+1 && numel(s.dq) == numel(s.tau)+1 && numel(s.qm) == numel(s.tau)+1 ...
    && numel(s.dqm) == numel(s.tau)+1 && numel(s.atLimit) == numel(s.tau) && s.nTransition == numel(s.tau), S));
chk{end+1} = add('配列長の整合', okLen, nnz(~okLen));
names = {S.name};
chk{end+1} = add('シナリオ名が一意', numel(unique(names)) == numel(names), numel(names) - numel(unique(names)));
isBench = strcmp({S.phase},'benchmark');
chk{end+1} = add('benchmark は split=benchmark のみ', all(strcmp({S(isBench).split},'benchmark')) && ...
    all(ismember({S(~isBench).split},{'train','val','test'})), nnz(isBench));
chk{end+1} = add('train/val/test がそろっている（100 本以上のとき）', numel(S) < 100 || all(ismember({'train','val','test'}, {S.split})), 0);
Sx = j2ScenarioSet(ds.manifest.preset);
chk{end+1} = add('シナリオ表が j2ScenarioSet(preset) と一致（名前・split・シード）', ...
    isequal({Sx.name},names) && isequal({Sx.split},{S.split}) && isequal([Sx.seed],[S.seed]), numel(Sx) - numel(S));

% 正規化統計と、解析モデルとの 1 ステップ整合
n = 0;  sx = zeros(1,5);  sxx = zeros(1,5);
num = zeros(1,2);  den = zeros(1,2);
for k = 1:numel(S)
    sc = S(k);  K = sc.nTransition;
    tw = sc.qm - sc.q;  dtw = sc.dqm - sc.dq;
    [aq, am] = j2FlexAnalyticAccel(sc.q(1:K), sc.dq(1:K), tw(1:K), dtw(1:K), sc.tau, jp, flex);
    a1 = diff(sc.dq)/dt;  a2 = diff(sc.dqm)/dt;
    num = num + [sum((a1 - aq).^2), sum((a2 - am).^2)];  den = den + [sum(a1.^2), sum(a2.^2)];
    if strcmp(sc.split,'train')
        X = [sc.q(1:K), sc.dq(1:K), tw(1:K), dtw(1:K), sc.tau];
        n = n + K;  sx = sx + sum(X,1);  sxx = sxx + sum(X.^2,1);
    end
end
xStd = sqrt((sxx - sx.^2/n)/(n-1));
chk{end+1} = add('正規化統計 = train の統計', max(abs(ds.stats.xMean - sx/n)./max(abs(sx/n),1e-9)) < 1e-9 && ...
    max(abs(ds.stats.xStd - xStd)./xStd) < 1e-9, 0);
nT = sum([S.nTransition]);
R.metrics.modelResidualRatio = sqrt(num./den);          % [ddq, ddqm]
R.metrics.modelResidualRMS = sqrt(num/nT);              % 暫定ゲート (a) の基準線（rad/s²）
chk{end+1} = add('二慣性の解析モデルの 1 ステップ加速度残差 / 加速度 RMS < 0.5（ddq・ddqm）', all(R.metrics.modelResidualRatio < 0.5), R.metrics.modelResidualRatio);
R.checks = [chk{:}];
R.nErrors = nnz(~[R.checks.pass]);
R.metrics.nTransition = nT;
if verbose, fprintf('検証: %d 項目中 %d 件不合格（遷移 %d、リミットまでの最小距離 %.2f°、最大ねじり %.4f rad、モデル残差 RMS [%.3f %.3f] rad/s²）\n', ...
    numel(R.checks), R.nErrors, nT, R.metrics.minDistToLimitDeg, maxTw, R.metrics.modelResidualRMS); end
end

function c = addCheck(name, pass, value, verbose)
c = struct('name',name,'pass',logical(pass),'value',value);
if verbose, fprintf('  [%s] %s  (%s)\n', ternary(pass,'OK','NG'), name, mat2str(value,4)); end
end

function s = ternary(c,a,b)
if c, s = a; else, s = b; end
end
