function j2_flex_dump_refs(outFile)
%J2_FLEX_DUMP_REFS C4 評価（tools/j2_flex_eval.m）の参照軌道 6 本を HDF5 に書き出す（Python の C4 再現用、issue #34）。
%   j2_flex_dump_refs('data/export/j2_flex_c4_refs.h5')
%   /<名前>/{qr,dqr,ddqr}（倍精度、8 kHz、軌道の後ろに 0.6 s の静止）と属性 tEnd, seed。刻み幅は /dt。
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if nargin < 1, outFile = fullfile(repo,'data','export','j2_flex_c4_refs.h5'); end
jp = j2Params();  dt = 1/jp.fsSim;
names = {'ptp_v40_001','ptp_v40_003','ptp_v60_001','ptp_v60_002','ptp_v80_001','ptp_v80_003'};
S = j2ScenarioSet('nss8k');
if isfile(outFile), delete(outFile); end
for i = 1:numel(names)
    s = S(strcmp({S.name},names{i}));
    rs = RandStream('twister','Seed',s.seed);
    WP = jp.qTrainMin + (jp.qTrainMax - jp.qTrainMin)*rand(rs, s.nWp, 1);
    T = j2PtpSegTime(WP, s.speedScale, 0.5);
    WP2 = WP(1);  T2 = [];
    for k = 1:numel(T)
        WP2(end+1,1) = WP(k+1);  T2(end+1,1) = T(k);        %#ok<AGROW>
        WP2(end+1,1) = WP(k+1);  T2(end+1,1) = 0.2;         %#ok<AGROW>
    end
    tEnd = min(sum(T2), s.duration);
    t = (0:dt:tEnd+0.6)';
    [qr, dqr, ddqr] = j2PtpTrajectory(min(t,tEnd), WP2, T2);
    after = t > tEnd;  dqr(after) = 0;  ddqr(after) = 0;
    for nm = {'qr','dqr','ddqr'}
        v = eval(nm{1});
        h5create(outFile, ['/' names{i} '/' nm{1}], numel(v), 'Datatype','double');
        h5write(outFile, ['/' names{i} '/' nm{1}], v);
    end
    h5writeatt(outFile, ['/' names{i}], 'tEnd', tEnd);
    h5writeatt(outFile, ['/' names{i}], 'seed', double(s.seed));
    fprintf('%s: %d 点、tEnd %.3f s\n', names{i}, numel(t), tEnd);
end
h5create(outFile, '/dt', 1);  h5write(outFile, '/dt', dt);
end
