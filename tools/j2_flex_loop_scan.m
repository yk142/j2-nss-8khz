function j2_flex_loop_scan(outJson)
%J2_FLEX_LOOP_SCAN 弾性 J2 のモータ角 PD 閉ループで、ゲイン（wn × ζ）ごとの追従・ねじり・残留振動を測る（確認 3・4）。
%   j2_flex_loop_scan('reports/issue-23/flex_loop_scan.json')
%   PTP 3 本（ptp_v40_003、ptp_v60_002、ptp_v80_003）× wn ∈ {15, 30, 60, 90, 120} × ζ ∈ {0.7, 1, 1.5}。約 25 分。
%   指標（3 本の平均）:
%     trackRmsDeg   リンク角の追従 RMS [°]
%     twistRms      動的ねじり（θm − θ − twist_ref）の RMS [rad]
%     residualRms   停止区間（参照速度がほぼ 0 で、停止から 0.05 s 以後）での、動的ねじりの RMS [rad]（残留振動）
%     tauRms        印加トルクの RMS [N·m]
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
jp = j2Params();  dt = 1/jp.fsSim;
tmp = tempname;  mkdir(tmp);
[mC, flex] = buildJ2FlexClosedLoop(struct('modelName','jfs_loop','slxPath',fullfile(tmp,'jfs_loop.slx')));
cleanup = onCleanup(@() close_if_loaded(mC));
names = {'ptp_v40_003','ptp_v60_002','ptp_v80_003'};
S = j2ScenarioSet('nss8k');
refs = cell(1,numel(names));
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
    t = (0:dt:min(sum(T2), s.duration))';
    [qr, dqr, ddqr] = j2PtpTrajectory(t, WP2, T2);
    refs{i} = struct('t',t,'qr',qr,'dqr',dqr,'ddqr',ddqr);
end
res = struct('wn',{},'zeta',{},'trackRmsDeg',{},'twistRms',{},'residualRms',{},'tauRms',{},'diverged',{});
for wn = [15 30 60 90 120]
    for zeta = [0.7 1 1.5]
        m = nan(numel(names),4);  div = false;
        for i = 1:numel(names)
            r = refs{i};
            o = simJ2FlexLoop(mC, r.t, r.qr, r.dqr, r.ddqr, wn, zeta, flex);
            if ~all(isfinite([o.q; o.qm; o.tau])), div = true; continue; end
            tw = o.qm - o.q - (o.qmRef - o.qr);                         % 動的ねじり
            still = abs(r.dqr) < 1e-3;  still = still & [false; still(1:end-1)];
            % 停止から 0.05 s 以後だけを残す
            idx = false(size(still));  run = 0;
            for k = 1:numel(still)
                if still(k), run = run + 1; else, run = 0; end
                idx(k) = run > 0.05/dt;
            end
            m(i,:) = [rad2deg(sqrt(mean((o.q - r.qr).^2))), sqrt(mean(tw.^2)), sqrt(mean(tw(idx).^2)), sqrt(mean(o.tau.^2))];
        end
        a = mean(m, 1, 'omitnan');
        res(end+1) = struct('wn',wn,'zeta',zeta,'trackRmsDeg',a(1),'twistRms',a(2),'residualRms',a(3),'tauRms',a(4),'diverged',div); %#ok<AGROW>
        fprintf('[%s] wn=%3d ζ=%.1f  追従 %.4f°  ねじり %.5f  残留 %.5f  τ %.1f  発散 %d\n', datestr(now,'HH:MM:SS'), wn, zeta, a, div);
    end
end
if nargin > 0 && ~isempty(outJson)
    fid = fopen(outJson,'w');  fwrite(fid, jsonencode(res, 'PrettyPrint', true));  fclose(fid);
end
end

function close_if_loaded(m)
if bdIsLoaded(m), close_system(m,0); end
end
