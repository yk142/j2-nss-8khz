function info = j2ExportNss8k(ds, outFile, opts)
%J2EXPORTNSS8K 8 kHz 接触なしデータセット（preset nss8k）を HDF5 で書き出す（Python/h5py 向け）。
%   info = J2EXPORTNSS8K(ds, outFile, opts)   docs/requirements_8khz.md §3 の入出力定義
%     ds       genJ2Dataset('nss8k') の出力（または data/j2_dataset_nss8k.mat を load したもの）
%     outFile  出力先（.h5）
%     opts.splits           書き出す split（既定 {'train','val','test','benchmark'}）
%     opts.excludeNearLimit リミットから 1° 以内の遷移を除く（既定 false。フラグのみ付ける）
%     opts.dtype            行列の型 'single'（既定）| 'double'。ddq は倍精度で計算してから丸める
%     opts.deflate          HDF5 の圧縮レベル 0〜9（既定 4）
%
%   HDF5 の構成（h5py から (N, d) に見える。インデックスは 0 始まり）:
%     /<split>/X          (N, 4)  [sin q, cos q, dq/DTHETA_MAX, tau/TAU_MAX]   ← モデル入力
%     /<split>/X_raw      (N, 3)  [q rad, dq rad/s, tau N·m]（tau は区間先頭の瞬時値）
%     /<split>/Y          (N, 1)  ddq [rad/s²] = (dq_next − dq)/Δt、Δt = 125 µs   ← 目標（評価はこの単位）
%     /<split>/Y_scaled   (N, 1)  ddq / DDQ_SCALE（= tauPeak/M = 141.4 rad/s²）   ← 純 NN (A) の目標
%     /<split>/a_phys     (N, 1)  解析モデルの加速度 [rad/s²]（x_k, tau_k で評価）  ← 残差型 (B) の土台。残差 = Y − a_phys
%     /<split>/scenario_id (N,)   int32、/scenarios/* の添字
%     /<split>/step        (N,)   int32、シナリオ内の遷移番号
%     /<split>/near_limit  (N,)   uint8、リミットから 1° 以内なら 1
%     /<split>/qref        (N,)   参照角 [rad]（姿勢保持・PTP のみ。励振は NaN）。遷移 k の開始時刻の値
%     /scenarios/{name,pattern,type,phase,split}、/scenarios/{seed,n_transition}
%     /stats/{x_mean,x_std}（X_raw の列）、/stats/{ddq_mean,ddq_std,res_mean,res_std}
%     /scale/*、/physics/*（M mgL Fc Bv eps qMin qMax qdMax tauRated tauPeak）、ルート属性
if nargin < 3, opts = struct(); end
def = struct('splits',{{'train','val','test','benchmark'}}, 'excludeNearLimit',false, 'dtype','single', 'deflate',4);
f = fieldnames(def);
for i = 1:numel(f)
    if ~isfield(opts,f{i}) || isempty(opts.(f{i})), opts.(f{i}) = def.(f{i}); end
end
[outDir,~,ext] = fileparts(outFile);
assert(strcmpi(ext,'.h5'), 'j2ExportNss8k:ext', '.h5 のみ対応です');
if ~isempty(outDir) && ~isfolder(outDir), mkdir(outDir); end
jp = j2Params();  sc = ds.scale;  S = ds.scen;  dt = 1/jp.fsSim;
cast = @(A) cast_(A, opts.dtype);
if isfile(outFile), delete(outFile); end
info.file = outFile;  info.dtype = opts.dtype;

for i = 1:numel(opts.splits)
    sp = opts.splits{i};
    idx = find(strcmp({S.split}, sp));
    Xc = cell(numel(idx),1);  Rc = Xc;  Yc = Xc;  Ac = Xc;  Sid = Xc;  Stp = Xc;  Nl = Xc;  Qr = Xc;
    for j = 1:numel(idx)
        s = S(idx(j));  K = s.nTransition;
        q = s.q(1:K);  dq = s.dq(1:K);  tau = s.tau;
        ok = true(K,1);  if opts.excludeNearLimit, ok = ~s.nearLimit; end
        ddq = diff(s.dq)/dt;
        aPhys = (tau + jp.mgL*sin(q) - jp.Bv*dq - jp.Fc*tanh(dq/jp.eps)) / jp.M;
        qref = nan(K,1);  if ~isempty(s.qref), qref = s.qref(1:K); end
        Xc{j} = [sin(q(ok)), cos(q(ok)), dq(ok)/sc.DTHETA_MAX, tau(ok)/sc.TAU_MAX];
        Rc{j} = [q(ok), dq(ok), tau(ok)];
        Yc{j} = ddq(ok);  Ac{j} = aPhys(ok);
        Sid{j} = repmat(int32(idx(j)-1), nnz(ok), 1);  Stp{j} = int32(find(ok)-1);
        Nl{j} = uint8(s.nearLimit(ok));  Qr{j} = qref(ok);
    end
    X = vertcat(Xc{:});  R = vertcat(Rc{:});  Y = vertcat(Yc{:});  A = vertcat(Ac{:});
    writeMat(outFile, ['/' sp '/X'],        cast(X), opts, 'sin q, cos q, dq/DTHETA_MAX, tau/TAU_MAX');
    writeMat(outFile, ['/' sp '/X_raw'],    cast(R), opts, 'q [rad], dq [rad/s], tau [N*m] (instantaneous at step start)');
    writeMat(outFile, ['/' sp '/Y'],        cast(Y), opts, 'ddq [rad/s^2] = (dq_next - dq)/dt, dt = 125 us');
    writeMat(outFile, ['/' sp '/Y_scaled'], cast(Y/sc.DDQ_SCALE), opts, 'ddq / DDQ_SCALE');
    writeMat(outFile, ['/' sp '/a_phys'],   cast(A), opts, 'analytic-model acceleration [rad/s^2] at (q, dq, tau)');
    writeVec(outFile, ['/' sp '/scenario_id'], vertcat(Sid{:}), opts, '0-based index into /scenarios');
    writeVec(outFile, ['/' sp '/step'],        vertcat(Stp{:}), opts, '0-based transition index within the scenario');
    writeVec(outFile, ['/' sp '/near_limit'],  vertcat(Nl{:}),  opts, '1 = within 1 deg of a joint limit');
    writeVec(outFile, ['/' sp '/qref'],        cast(vertcat(Qr{:})), opts, 'reference angle [rad] at step start (hold/ptp), NaN otherwise');
    info.n.(sp) = size(X,1);
end
writeStr(outFile, '/scenarios/name',    string({S.name}));
writeStr(outFile, '/scenarios/pattern', string({S.pattern}));
writeStr(outFile, '/scenarios/type',    string({S.type}));
writeStr(outFile, '/scenarios/phase',   string(cellfun(@num2str, {S.phase}, 'UniformOutput', false)));
writeStr(outFile, '/scenarios/split',   string({S.split}));
writeVec(outFile, '/scenarios/seed',         int64([S.seed]'), opts, '');
writeVec(outFile, '/scenarios/n_transition', int32([S.nTransition]'), opts, '');
st = ds.stats;
for nm = {'xMean','x_mean'; 'xStd','x_std'; 'ddqMean','ddq_mean'; 'ddqStd','ddq_std'; 'resMean','res_mean'; 'resStd','res_std'}'
    writeVec(outFile, ['/stats/' nm{2}], double(st.(nm{1})(:)), opts, 'train, all transitions');
end
for nm = fieldnames(sc)', writeVec(outFile, ['/scale/' nm{1}], double(sc.(nm{1})), opts, ''); end
for nm = {'M','mgL','Fc','Bv','eps','qMin','qMax','qdMax','tauRated','tauPeak'}
    writeVec(outFile, ['/physics/' nm{1}], double(jp.(nm{1})), opts, '');
end
m = ds.manifest;  put = @(k, v) h5writeatt(outFile, '/', k, v);
put('preset', char(m.preset));  put('created', char(m.created));  put('fs_hz', double(m.fsData));
put('q_fixed_deg', double(m.qFixedDeg(:)));  put('parent_commit', char(m.parentCommit));  put('matlab', char(m.matlab));
put('exclude_near_limit', double(opts.excludeNearLimit));  put('dtype', opts.dtype);
put('layout', 'row = transition; HDF5 datasets are (N, d) as seen from h5py. indices are 0-based.');
put('description', 'EPSON C8-A901S J2 single-axis (J3 fixed at 75.0684 deg), contact-free, 8 kHz one-step dataset for NSS');
d = dir(outFile);  info.bytes = d.bytes;
end

function A = cast_(A, dtype)
if strcmp(dtype,'single'), A = single(A); else, A = double(A); end
end

function writeMat(file, path, A, opts, columns)
% A は (N, d)。MATLAB は列優先なので (d, N) に転置して書く → h5py からは (N, d) に見える
[N, d] = size(A);
if N == 0, error('j2ExportNss8k:empty', '%s は空です', path); end
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
