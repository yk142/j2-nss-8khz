function ds = assembleJ2Dataset8kFlex(S, cacheFn, opts, jp)
%ASSEMBLEJ2DATASET8KFLEX 弾性（二慣性）J2 の 8 kHz 生データキャッシュから、遷移データセットを組み立てる。
%   ds = ASSEMBLEJ2DATASET8KFLEX(S, cacheFn, opts, jp)   preset 'nss8kflex'（docs/requirements_nss_flex.md）
%   assembleJ2Dataset8k（剛体）との違い:
%     - 状態は (q, dq, qm, dqm)。入力特徴は X = [q dq twist dtwist tau]（twist = qm − q、dtwist = dqm − dq）
%     - 出力は Y = [ddq ddqm]（Δ/Δt）。解析モデル（二慣性）の加速度との残差も保存
%     - 解析モデル: JL·ddq = k·tw + c·dtw + mgL·sin q − Bv·dq − Fc·tanh(dq/ε)、Jm·ddqm = τ − k·tw − c·dtw
%   ds.scen の要素: name pattern type phase split seed、q dq qm dqm [K+1]、tau [K]、qref [K+1]、atLimit nearLimit [K]（リンク角）、nTransition
%   ds.stats: xMean xStd（[q dq tw dtw tau]）、yMean yStd・resMean resStd（各 1x2）。ds.flex: 弾性パラメータ
dt = 1/jp.fsSim;  flex = j2FlexParams();
tol1 = deg2rad(0.1);  tol2 = deg2rad(1);
scen = repmat(struct('name','','pattern','','type','','phase',[],'split','','seed',0, ...
    'q',[],'dq',[],'qm',[],'dqm',[],'tau',[],'qref',[],'atLimit',[],'nearLimit',[],'nTransition',0), numel(S), 1);
n = 0;  sx = zeros(1,5);  sxx = zeros(1,5);  sy = zeros(1,4);  syy = zeros(1,4);   % y = [ddq ddqm res_q res_m]
for k = 1:numel(S)
    s = S(k);  C = load(cacheFn(s));  e = C.e;
    q = double(e.q(:));  dq = double(e.dq(:));  qm = double(e.qm(:));  dqm = double(e.dqm(:));  tau = double(e.tau(:));
    K = numel(q) - 1;
    tauK = tau(1:K);
    d = min(q - jp.qMin, jp.qMax - q);
    atLimit   = d(1:K) <= tol1 | d(2:end) <= tol1;
    nearLimit = d(1:K) <= tol2 | d(2:end) <= tol2;
    qref = [];
    if isfield(e,'qref') && ~isempty(e.qref), qref = double(e.qref(:)); end
    scen(k) = struct('name',s.name,'pattern',s.pattern,'type',s.type,'phase',s.phase,'split',s.split,'seed',s.seed, ...
        'q',q,'dq',dq,'qm',qm,'dqm',dqm,'tau',tauK,'qref',qref,'atLimit',atLimit,'nearLimit',nearLimit,'nTransition',K);
    if strcmp(s.split,'train')
        tw = qm - q;  dtw = dqm - dq;
        [aq, am] = j2FlexAnalyticAccel(q(1:K), dq(1:K), tw(1:K), dtw(1:K), tauK, jp, flex);
        ddq = diff(dq)/dt;  ddqm = diff(dqm)/dt;
        X = [q(1:K), dq(1:K), tw(1:K), dtw(1:K), tauK];  Y = [ddq ddqm ddq-aq ddqm-am];
        n = n + K;  sx = sx + sum(X,1);  sxx = sxx + sum(X.^2,1);  sy = sy + sum(Y,1);  syy = syy + sum(Y.^2,1);
    end
end
sd = @(sum1, sum2) sqrt(max(sum2 - sum1.^2/n, 0)/(n-1));
stats.xMean = sx/n;  stats.xStd = sd(sx, sxx);
stats.yMean = sy(1:2)/n;  stats.yStd = sd(sy(1:2), syy(1:2));
stats.resMean = sy(3:4)/n;  stats.resStd = sd(sy(3:4), syy(3:4));
stats.xStd(stats.xStd < eps) = 1;
stats.note = '正規化統計は train の全遷移から算出。X=[q dq tw dtw tau]、Y=[ddq ddqm]（Δ/Δt）、res=Y−二慣性の解析モデル（x_k, tau_k で評価）';
scale.DTHETA_MAX = jp.qdMax;  scale.TAU_MAX = jp.tauPeak;  scale.DDQ_SCALE = jp.tauPeak/jp.M;

isTrain = strcmp({scen.split},'train');
manifest.created = char(datetime('now'));
manifest.preset = opts.preset;  manifest.fsRaw = jp.fsSim;  manifest.fsData = jp.fsSim;
manifest.nScenario = numel(scen);
manifest.nTrain = nnz(isTrain);  manifest.nVal = nnz(strcmp({scen.split},'val'));
manifest.nTest = nnz(strcmp({scen.split},'test'));  manifest.nBenchmark = nnz(strcmp({scen.split},'benchmark'));
manifest.nTransition = sum([scen.nTransition]);
manifest.qFixedDeg = rad2deg(jp.qFixed);
manifest.torqueInput = '区間先頭の瞬時値 tau(t_k)（モータ側）';
manifest.downsample = 'なし（8 kHz のまま）';
manifest.flex = flex;
manifest.params = jp;
manifest.matlab = version;
ds.scen = scen;  ds.stats = stats;  ds.scale = scale;  ds.manifest = manifest;  ds.flex = flex;
if opts.save
    if ~isfolder(opts.outDir), mkdir(opts.outDir); end
    fn = fullfile(opts.outDir, sprintf('j2_dataset_%s.mat', opts.preset));
    save(fn, '-struct', 'ds', '-v7.3');
    fprintf('保存: %s（%d シナリオ、%d 遷移、train/val/test/bench = %d/%d/%d/%d）\n', fn, numel(scen), ...
        manifest.nTransition, manifest.nTrain, manifest.nVal, manifest.nTest, manifest.nBenchmark);
end
end
