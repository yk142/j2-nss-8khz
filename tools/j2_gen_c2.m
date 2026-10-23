function j2_gen_c2(shardI, shardN, only)
%J2_GEN_C2 (c2) の真値を生成する: PD ゲイン 9 組 × 条件 E1/E2 × PTP 12 軌道（docs/requirements_c2.md）。
%   別プロセスから: matlab -batch "addpath('tools'); j2_gen_c2(1, 2)"   （shardI/shardN で担当を分ける）
%   only = 'nominal' なら、検証用に wn=30, ζ=1, FF 正確・外乱なしの条件 N（1 組）だけを作る。
%   出力: data/c2/<条件>_wn<wn>_z<ζ×10>/<軌道名>.mat（git 管理外）。生成済みはスキップ（中断後に再開できる）。
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
c = j2_setup_path();
if nargin < 1, shardI = 1; shardN = 1; end
if ischar(shardI), shardI = str2double(shardI); end
if ischar(shardN), shardN = str2double(shardN); end
if nargin < 3, only = ''; end
conds = struct('name',{'E1','E2'}, 'ffErr',{-0.15, +0.15}, 'distAmp',{+20, -20}, 'distDur',{0.5, 0.5});
if strcmp(only,'nominal'), conds = struct('name','N','ffErr',0,'distAmp',0,'distDur',0); end
wns = [15 30 60];  zs = [0.7 1 1.5];
gains = [repelem(wns,numel(zs)); repmat(zs,1,numel(wns))]';       % 9 x 2
if strcmp(only,'nominal'), gains = [30 1]; end
S = j2ScenarioSet('nss8k');  S = S(strcmp({S.phase},'benchmark'));
outRoot = fullfile(c.dataDir,'c2');
workDir = fullfile(c.dataDir,'models');
mdl = sprintf('j2_cl_c2_s%d', shardI);
mC = buildJ2ClosedLoop(struct('modelName',mdl,'slxPath',fullfile(workDir,[mdl '.slx'])));
cleanup = onCleanup(@() close_if_loaded(mC));
jp = j2Params();
for g = shardI:shardN:size(gains,1)
    wn = gains(g,1);  zeta = gains(g,2);
    set_param([mC '/Kp'],'Gain',sprintf('%.10g',jp.M*wn^2));
    set_param([mC '/Kd'],'Gain',sprintf('%.10g',2*zeta*jp.M*wn));
    for ic = 1:numel(conds)
        cd_ = conds(ic);
        dirn = fullfile(outRoot, sprintf('%s_wn%d_z%d', cd_.name, wn, round(10*zeta)));
        if ~isfolder(dirn), mkdir(dirn); end
        for k = 1:numel(S)
            fn = fullfile(dirn, [S(k).name '.mat']);
            if isfile(fn), continue; end
            t0 = tic;
            raw = runJ2PtpC2(S(k), mC, cd_);
            meta = struct('wn',wn,'zeta',zeta,'cond',cd_,'scenario',S(k).name,'dt',1/jp.fsSim);
            save(fn, '-struct', 'raw', '-v7.3'); save(fn, 'meta', '-append');
            fprintf('[%s] %s %s (%.0f s)\n', datestr(now,'HH:MM:SS'), cd_.name, S(k).name, toc(t0));
        end
    end
end
end

function close_if_loaded(m)
if bdIsLoaded(m), close_system(m,0); end
end
