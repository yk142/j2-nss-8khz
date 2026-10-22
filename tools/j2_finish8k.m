function j2_finish8k(preset, forkCacheDir, outDir)
%J2_FINISH8K 8 kHz（nss8k 系）の全シナリオ生成後に、組み立て・検証・フォーク元との照合を行う。
%   別プロセスから実行: matlab -batch "addpath('tools'); j2_finish8k('nss8k')"
%   forkCacheDir の既定は ../j2-state-data-gen/data/cache、outDir の既定は reports/issue-6。
%   結果の指標は outDir/metrics_<preset>.json に残す。
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if nargin < 1, preset = 'nss8k'; end
if nargin < 2 || isempty(forkCacheDir), forkCacheDir = fullfile(fileparts(repo),'j2-state-data-gen','data','cache'); end
if nargin < 3, outDir = fullfile(repo,'reports','issue-6'); end
ds = genJ2Dataset(struct('preset',preset));
assert(~isempty(ds), 'j2_finish8k:incomplete', '未生成のシナリオが残っています');
R = validateJ2Data8k(ds, true, forkCacheDir);
if ~isfolder(outDir), mkdir(outDir); end
out.validation = struct('nErrors',R.nErrors,'metrics',R.metrics, ...
    'checks',arrayfun(@(c) struct('name',c.name,'pass',c.pass,'value',c.value), R.checks, 'UniformOutput', false));
out.stats = ds.stats;  out.scale = ds.scale;  out.manifest = ds.manifest;
fid = fopen(fullfile(outDir,sprintf('metrics_%s.json',preset)),'w');  fwrite(fid, jsonencode(out,'PrettyPrint',true));  fclose(fid);
fprintf('完了: 検証 不合格 %d 件、遷移 %d、解析モデルの 1 ステップ RMS %.3f rad/s²\n', R.nErrors, R.metrics.nTransition, R.metrics.modelResidualRMS);
end
