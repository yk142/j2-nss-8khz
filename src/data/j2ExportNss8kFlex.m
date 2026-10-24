function info = j2ExportNss8kFlex(ds, outFile, opts)
%J2EXPORTNSS8KFLEX 弾性 J2 の 8 kHz データセット（preset nss8kflex）を HDF5 で書き出す（Python/h5py 向け）。
%   info = J2EXPORTNSS8KFLEX(ds, outFile, opts)   docs/requirements_nss_flex.md §2、docs/plan_nss_flex_train.md
%     ds       genJ2Dataset('nss8kflex') の出力（または data/j2_dataset_nss8kflex.mat を load した構造体）
%     opts.splits  書き出す split（既定 {'train','val','test','benchmark'}）
%     opts.dtype   行列の型 'single'（既定）| 'double'。加速度とねじりは倍精度で計算してから丸める
%     opts.deflate HDF5 の圧縮レベル（既定 4）
%
%   HDF5 の構成（h5py から (N, d) に見える。インデックスは 0 始まり）:
%     /<split>/X          (N, 6)  [sin q, cos q, dq/DTHETA_MAX, tw/s_tw, dtw/s_dtw, tau/TAU_MAX]   ← モデル入力
%     /<split>/X_raw      (N, 5)  [q, dq, tw, dtw, tau]   tw = qm − q [rad]、dtw = dqm − dq [rad/s]（倍精度で作ってから丸める）
%     /<split>/Y          (N, 2)  [ddq, ddqm] [rad/s^2] = Δ/Δt、Δt = 125 µs
%     /<split>/Y_scaled   (N, 2)  Y ./ [tauPeak/JL, tauPeak/Jm]
%     /<split>/a_phys     (N, 2)  二慣性の解析モデルの加速度（x_k, tau_k で評価）。残差 = Y − a_phys
%     /<split>/scenario_id, step, near_limit, qref（j2ExportNss8k と同じ）
%     /scenarios/*、/stats/{x_mean,x_std}（X_raw の列）、/stats/{y_mean,y_std,res_mean,res_std}（各 2 要素）
%     /scale/*、/physics/*、/flex/{Jm,JL,k,c,fRes,zeta}、ルート属性
if nargin < 3, opts = struct(); end
def = struct('splits',{{'train','val','test','benchmark'}}, 'dtype','single', 'deflate',4);
f = fieldnames(def);
for i = 1:numel(f)
    if ~isfield(opts,f{i}) || isempty(opts.(f{i})), opts.(f{i}) = def.(f{i}); end
end
[outDir,~,ext] = fileparts(outFile);
assert(strcmpi(ext,'.h5'), 'j2ExportNss8kFlex:ext', '.h5 のみ対応です');
if ~isempty(outDir) && ~isfolder(outDir), mkdir(outDir); end
jp = j2Params();  sc = ds.scale;  S = ds.scen;  flex = ds.flex;  dt = 1/jp.fsSim;  st = ds.stats;
cast = @(A) cast_(A, opts.dtype);
if isfile(outFile), delete(outFile); end
info.file = outFile;  info.dtype = opts.dtype;
yScale = [jp.tauPeak/flex.JL, jp.tauPeak/flex.Jm];

for i = 1:numel(opts.splits)
    sp = opts.splits{i};
    idx = find(strcmp({S.split}, sp));
    Xc = cell(numel(idx),1);  Rc = Xc;  Yc = Xc;  Ac = Xc;  Sid = Xc;  Stp = Xc;  Nl = Xc;  Qr = Xc;
    for j = 1:numel(idx)
        s = S(idx(j));  K = s.nTransition;
        q = s.q(1:K);  dq = s.dq(1:K);  tau = s.tau;
        tw = s.qm(1:K) - q;  dtw = s.dqm(1:K) - dq;
        Y = [diff(s.dq)/dt, diff(s.dqm)/dt];
        [aq, am] = j2FlexAnalyticAccel(q, dq, tw, dtw, tau, jp, flex);
        qref = nan(K,1);  if ~isempty(s.qref), qref = s.qref(1:K); end
        Xc{j} = [sin(q), cos(q), dq/sc.DTHETA_MAX, tw/st.xStd(3), dtw/st.xStd(4), tau/sc.TAU_MAX];
        Rc{j} = [q, dq, tw, dtw, tau];
        Yc{j} = Y;  Ac{j} = [aq am];
        Sid{j} = repmat(int32(idx(j)-1), K, 1);  Stp{j} = int32((0:K-1)');
        Nl{j} = uint8(s.nearLimit);  Qr{j} = qref;
    end
    X = vertcat(Xc{:});  R = vertcat(Rc{:});  Y = vertcat(Yc{:});  A = vertcat(Ac{:});
    writeMat(outFile, ['/' sp '/X'],        cast(X), opts, 'sin q, cos q, dq/DTHETA_MAX, tw/s_tw, dtw/s_dtw, tau/TAU_MAX');
    writeMat(outFile, ['/' sp '/X_raw'],    cast(R), opts, 'q [rad], dq [rad/s], tw=qm-q [rad], dtw=dqm-dq [rad/s], tau [N*m]');
    writeMat(outFile, ['/' sp '/Y'],        cast(Y), opts, 'ddq, ddqm [rad/s^2] = (next - now)/dt, dt = 125 us');
    writeMat(outFile, ['/' sp '/Y_scaled'], cast(Y./yScale), opts, 'Y ./ [tauPeak/JL, tauPeak/Jm]');
    writeMat(outFile, ['/' sp '/a_phys'],   cast(A), opts, 'two-mass analytic accelerations [ddq ddqm] at (x_k, tau_k)');
    writeVec(outFile, ['/' sp '/scenario_id'], vertcat(Sid{:}), opts, '0-based index into /scenarios');
    writeVec(outFile, ['/' sp '/step'],        vertcat(Stp{:}), opts, '0-based transition index within the scenario');
    writeVec(outFile, ['/' sp '/near_limit'],  vertcat(Nl{:}),  opts, '1 = within 1 deg of a joint limit');
    writeVec(outFile, ['/' sp '/qref'],        double(vertcat(Qr{:})), opts, 'reference link angle [rad] at step start (hold/ptp), NaN otherwise');
    info.n.(sp) = size(X,1);
    clear X R Y A Xc Rc Yc Ac Sid Stp Nl Qr
end
writeStr(outFile, '/scenarios/name',    string({S.name}));
writeStr(outFile, '/scenarios/pattern', string({S.pattern}));
writeStr(outFile, '/scenarios/type',    string({S.type}));
writeStr(outFile, '/scenarios/split',   string({S.split}));
writeVec(outFile, '/scenarios/seed',         int64([S.seed]'), opts, '');
writeVec(outFile, '/scenarios/n_transition', int32([S.nTransition]'), opts, '');
for nm = {'xMean','x_mean'; 'xStd','x_std'; 'yMean','y_mean'; 'yStd','y_std'; 'resMean','res_mean'; 'resStd','res_std'}'
    writeVec(outFile, ['/stats/' nm{2}], double(st.(nm{1})(:)), opts, 'train, all transitions');
end
for nm = fieldnames(sc)', writeVec(outFile, ['/scale/' nm{1}], double(sc.(nm{1})), opts, ''); end
for nm = {'M','mgL','Fc','Bv','eps','qMin','qMax','qdMax','tauRated','tauPeak'}
    writeVec(outFile, ['/physics/' nm{1}], double(jp.(nm{1})), opts, '');
end
for nm = fieldnames(flex)', writeVec(outFile, ['/flex/' nm{1}], double(flex.(nm{1})), opts, ''); end
m = ds.manifest;  put = @(k, v) h5writeatt(outFile, '/', k, v);
put('preset', char(m.preset));  put('created', char(m.created));  put('fs_hz', double(m.fsData));
put('q_fixed_deg', double(m.qFixedDeg(:)));  put('matlab', char(m.matlab));
put('layout', 'row = transition; HDF5 datasets are (N, d) as seen from h5py. indices are 0-based.');
put('description', 'EPSON C8-A901S J2 two-mass (flexible) single-axis, contact-free, 8 kHz one-step dataset for 4-state NSS');
d = dir(outFile);  info.bytes = d.bytes;
end

function A = cast_(A, dtype)
if strcmp(dtype,'single'), A = single(A); else, A = double(A); end
end

function writeMat(file, path, A, opts, columns)
[N, d] = size(A);
if N == 0, error('j2ExportNss8kFlex:empty', '%s は空です', path); end
h5create(file, path, [d N], 'Datatype', class(A), 'ChunkSize', [d min(N, 65536)], 'Deflate', opts.deflate);
h5write(file, path, A');
if ~isempty(columns), h5writeatt(file, path, 'columns', columns); end
end

function writeVec(file, path, v, opts, note)
v = v(:);
if isempty(v), return; end
if numel(v) > 1 && opts.deflate > 0
    h5create(file, path, numel(v), 'Datatype', class(v), 'ChunkSize', min(numel(v), 65536), 'Deflate', opts.deflate);
else
    h5create(file, path, numel(v), 'Datatype', class(v));
end
h5write(file, path, v);
if ~isempty(note), h5writeatt(file, path, 'note', note); end
end

function writeStr(file, path, s)
h5create(file, path, numel(s), 'Datatype', 'string');
h5write(file, path, s(:));
end
