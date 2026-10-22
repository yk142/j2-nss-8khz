function ds = assembleJ2Dataset8k(S, cacheFn, opts, jp)
%ASSEMBLEJ2DATASET8K 8 kHz の生データキャッシュから、8 kHz のままの遷移データセットを組み立てる。
%   ds = ASSEMBLEJ2DATASET8K(S, cacheFn, opts, jp)   preset 'nss8k' 用（docs/requirements_8khz.md）
%     S        j2ScenarioSet('nss8k')、cacheFn(s) はキャッシュのパス、opts は genJ2Dataset の opts
%
%   1 kHz 版（genJ2Dataset の組み立て）との違い:
%     - 間引きも区間平均もしない。遷移 k は (q(k), dq(k), tau(k)) → (q(k+1), dq(k+1))、Δt = 125 µs
%     - トルクは区間先頭の瞬時値 tau(t_k)（因果的。制御器が x_k から出して 1 ステップ保持する形）
%     - 出力は ddq = (dq(k+1) − dq(k))/Δt。正規化統計は ddq と、解析モデルの残差の両方を保存
%     - 生データは double（キャッシュのまま）
%   ds.scen の要素: name pattern type phase split seed、q dq [K+1]、tau [K]、qref [K+1]、
%     atLimit [K]（リミットから 0.1° 以内または外側。nss8k では 0 件のはず）、
%     nearLimit [K]（同 1° 以内。参考）、nTransition
%   ds.stats: xMean xStd（[q dq tau]）、ddqMean ddqStd、resMean resStd（残差 = ddq − 解析モデル）
%   ds.scale: DTHETA_MAX TAU_MAX DDQ_SCALE（= tauPeak/M）
dt = 1/jp.fsSim;
tol1 = deg2rad(0.1);  tol2 = deg2rad(1);
scen = repmat(struct('name','','pattern','','type','','phase',[],'split','','seed',0, ...
    'q',[],'dq',[],'tau',[],'qref',[],'atLimit',[],'nearLimit',[],'nTransition',0), numel(S), 1);
n = 0;  sx = zeros(1,3);  sxx = zeros(1,3);  sy = zeros(1,2);  syy = zeros(1,2);   % train の累積（y = [ddq res]）
for k = 1:numel(S)
    s = S(k);  C = load(cacheFn(s));  e = C.e;
    q = double(e.q(:));  dq = double(e.dq(:));  tau = double(e.tau(:));
    K = numel(q) - 1;
    tauK = tau(1:K);                                   % 遷移 k の入力 = サンプル k の瞬時値
    dist = @(x) min(x - jp.qMin, jp.qMax - x);          % リミットまでの距離（外側は負）
    d = dist(q);
    atLimit   = d(1:K) <= tol1 | d(2:end) <= tol1;
    nearLimit = d(1:K) <= tol2 | d(2:end) <= tol2;
    qref = [];
    if isfield(e,'qref') && ~isempty(e.qref), qref = double(e.qref(:)); end
    scen(k) = struct('name',s.name,'pattern',s.pattern,'type',s.type,'phase',s.phase,'split',s.split, ...
        'seed',s.seed,'q',q,'dq',dq,'tau',tauK,'qref',qref,'atLimit',atLimit,'nearLimit',nearLimit,'nTransition',K);
    if strcmp(s.split,'train')
        ddq = diff(dq)/dt;
        res = ddq - j2AnalyticAccel(q(1:K), dq(1:K), tauK, jp);
        X = [q(1:K), dq(1:K), tauK];  Y = [ddq res];
        n = n + K;  sx = sx + sum(X,1);  sxx = sxx + sum(X.^2,1);  sy = sy + sum(Y,1);  syy = syy + sum(Y.^2,1);
    end
end
sd = @(sum1, sum2) sqrt(max(sum2 - sum1.^2/n, 0)/(n-1));
stats.xMean = sx/n;  stats.xStd = sd(sx, sxx);
stats.ddqMean = sy(1)/n;  stats.ddqStd = sd(sy(1), syy(1));
stats.resMean = sy(2)/n;  stats.resStd = sd(sy(2), syy(2));
stats.xStd(stats.xStd < eps) = 1;
stats.note = '正規化統計は train の全遷移から算出。X=[q dq tau]（tau は区間先頭の瞬時値）、ddq=Δdq/Δt、res=ddq−解析モデル（x_k, tau_k で評価）';
scale.DTHETA_MAX = jp.qdMax;  scale.TAU_MAX = jp.tauPeak;  scale.DDQ_SCALE = jp.tauPeak/jp.M;

isTrain = strcmp({scen.split},'train');
manifest.created = char(datetime('now'));
manifest.preset = opts.preset;  manifest.fsRaw = jp.fsSim;  manifest.fsData = jp.fsSim;
manifest.nScenario = numel(scen);
manifest.nTrain = nnz(isTrain);  manifest.nVal = nnz(strcmp({scen.split},'val'));
manifest.nTest = nnz(strcmp({scen.split},'test'));  manifest.nBenchmark = nnz(strcmp({scen.split},'benchmark'));
manifest.nTransition = sum([scen.nTransition]);
manifest.qFixedDeg = rad2deg(jp.qFixed);
manifest.torqueInput = '区間先頭の瞬時値 tau(t_k)';
manifest.downsample = 'なし（8 kHz のまま。遷移は隣接サンプルの組）';
manifest.params = jp;
manifest.parentCommit = parentCommit();
manifest.matlab = version;
ds.scen = scen;  ds.stats = stats;  ds.scale = scale;  ds.manifest = manifest;
if opts.save
    if ~isfolder(opts.outDir), mkdir(opts.outDir); end
    fn = fullfile(opts.outDir, sprintf('j2_dataset_%s.mat', opts.preset));
    save(fn, '-struct', 'ds', '-v7.3');
    fprintf('保存: %s（%d シナリオ、%d 遷移、train/val/test/bench = %d/%d/%d/%d）\n', fn, numel(scen), ...
        manifest.nTransition, manifest.nTrain, manifest.nVal, manifest.nTest, manifest.nBenchmark);
end
end

function a = j2AnalyticAccel(q, dq, tau, jp)
% 解析モデルの加速度 M·ddq = τ + mgL·sin θ − Bv·dq − Fc·tanh(dq/ε)
a = (tau + jp.mgL*sin(q) - jp.Bv*dq - jp.Fc*tanh(dq/jp.eps)) / jp.M;
end

function h = parentCommit()
c = j2_config();
[st, out] = system(sprintf('git -C "%s" rev-parse HEAD 2>/dev/null', c.parentDir));
if st == 0, h = strtrim(out); else, h = 'unknown'; end
end
