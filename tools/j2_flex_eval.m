function j2_flex_eval(outJson, conds)
%J2_FLEX_EVAL 弾性 J2 のモータ角 PD 閉ループを、共振を励起する条件で評価する（docs/requirements_flex_eval.md）。
%   j2_flex_eval('reports/issue-25/flex_eval.json' [, conds])   conds 既定 {'C0','C1','C2','C3'}。約 2 時間
%   条件: C0 公称 / C1 静止区間の衝撃外乱 100 N·m×2 ms / C2 真値の剛性 0.7 倍（FF は公称） / C3 FF −15% ＋ 外乱 +20 N·m（0.5 s）
%   軌道 6 本 × wn ∈ {15,30,60,90,120} × ζ ∈ {0.7,1,1.5}。軌道の後ろに 0.6 s の静止を付ける。
%   C4 は PD を 1 kHz のサンプル・ホールド＋1 サンプル遅れにし、wn ∈ {180, 240} も走査する（docs/requirements_flex_eval.md §6）。
%   指標（軌道平均）: trackRmsDeg（軌道区間）、settleTwist（軌道終了 +0.15 s 以降、最終値からの偏差の RMS [rad]）、
%     decayTime（C1 のみ。パルス後、ねじりが最初のピークの 10% 未満に収まるまで [s]、上限 0.5）、tauRms、tauMax、diverged
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if nargin < 2, conds = {'C0','C1','C2','C3'}; end
jp = j2Params();  dt = 1/jp.fsSim;
names = {'ptp_v40_001','ptp_v40_003','ptp_v60_001','ptp_v60_002','ptp_v80_001','ptp_v80_003'};
tmp = tempname;  mkdir(tmp);
[mN, flexN] = buildJ2FlexClosedLoop(struct('modelName','jfe_nom','slxPath',fullfile(tmp,'jfe_nom.slx')));
[mD, flexD] = buildJ2FlexClosedLoop(struct('modelName','jfe_dig','slxPath',fullfile(tmp,'jfe_dig.slx'),'ctrlTs',1e-3));
[mR, flexR] = buildJ2FlexClosedLoop(struct('modelName','jfe_rigid','slxPath',fullfile(tmp,'jfe_rigid.slx'),'ctrlTs',1e-3,'kScale',100,'cScale',10));
[mS, flexS] = buildJ2FlexClosedLoop(struct('modelName','jfe_soft','slxPath',fullfile(tmp,'jfe_soft.slx'),'kScale',0.7));
cleanup = onCleanup(@() cellfun(@close_if_loaded, {mN, mS, mD, mR}));
refs = makeRefs(names, jp, dt);
res = struct('cond',{},'wn',{},'zeta',{},'trackRmsDeg',{},'settleTwist',{},'decayTime',{},'tauRms',{},'tauMax',{},'twistDyn',{},'diverged',{});
for ci = 1:numel(conds)
    c = conds{ci};
    mdl = mN;  fl = flexN;  ff = struct();
    if strcmp(c,'C2'), mdl = mS; fl = flexS; ff.flexFf = flexN; end
    if strcmp(c,'C3'), ff.ffScale = 0.85; end
    if strcmp(c,'C4'), mdl = mD; fl = flexD; end
    if strcmp(c,'C5'), mdl = mR; fl = flexR; end      % C4 と同じ遅れで、剛性 100 倍（共振 約 200 Hz、ほぼ剛体）
    wns = [15 30 60 90 120];  if any(strcmp(c,{'C4','C5'})), wns = [wns 180 240]; end
    for wn = wns
        for zeta = [0.7 1 1.5]
            m = nan(numel(names),6);  div = false;
            for i = 1:numel(names)
                r = refs{i};
                tauD = zeros(size(r.t));
                if strcmp(c,'C1'), tauD(r.t >= r.tEnd+0.1 & r.t < r.tEnd+0.1+0.002) = 100; end
                if strcmp(c,'C3')
                    t0 = rand(RandStream('twister','Seed',r.seed+7))*(r.tEnd - 0.5);
                    tauD(r.t >= t0 & r.t < t0+0.5) = 20;
                end
                o = simJ2FlexLoop(mdl, r.t, r.qr, r.dqr, r.ddqr, wn, zeta, fl, tauD, ff);
                if ~all(isfinite([o.q; o.qm; o.tau])), div = true; continue; end
                tw = o.qm - o.q;
                inTraj = r.t <= r.tEnd;
                win = r.t >= r.tEnd + 0.15;
                last = r.t >= r.t(end) - 0.05;
                dev = tw(win) - mean(tw(last));
                dec = nan;
                if strcmp(c,'C1'), dec = decayTime(r.t, tw, r.tEnd+0.1, mean(tw(last))); end
                m(i,:) = [rad2deg(sqrt(mean((o.q(inTraj) - r.qr(inTraj)).^2))), sqrt(mean(dev.^2)), dec, sqrt(mean(o.tau.^2)), max(abs(o.tau)), sqrt(mean((tw(inTraj) - (o.qmRef(inTraj) - o.qr(inTraj))).^2))];
            end
            a = mean(m, 1, 'omitnan');
            res(end+1) = struct('cond',c,'wn',wn,'zeta',zeta,'trackRmsDeg',a(1),'settleTwist',a(2),'decayTime',a(3),'tauRms',a(4),'tauMax',a(5),'twistDyn',a(6),'diverged',div); %#ok<AGROW>
            fprintf('[%s] %s wn=%3d ζ=%.1f  追従 %.4f°  整定ねじり %.2e  減衰 %.3f s  τrms %.1f  動的ねじり %.2e  発散 %d\n', ...
                datestr(now,'HH:MM:SS'), c, wn, zeta, a(1), a(2), a(3), a(4), a(6), div);
            if nargin > 0 && ~isempty(outJson)
                fid = fopen(outJson,'w');  fwrite(fid, jsonencode(res, 'PrettyPrint', true));  fclose(fid);
            end
        end
    end
end
end

function refs = makeRefs(names, jp, dt)
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
    tEnd = min(sum(T2), s.duration);
    t = (0:dt:tEnd+0.6)';
    [qr, dqr, ddqr] = j2PtpTrajectory(min(t,tEnd), WP2, T2);
    after = t > tEnd;  dqr(after) = 0;  ddqr(after) = 0;
    refs{i} = struct('t',t,'qr',qr,'dqr',dqr,'ddqr',ddqr,'tEnd',tEnd,'seed',s.seed);
end
end

function d = decayTime(t, tw, t0, twFinal)
% パルス後のねじり（最終値からの偏差）が、最初の 50 ms のピークの 10% 未満に収まるまでの時間
x = abs(tw - twFinal);
idx = t >= t0;  tt = t(idx) - t0;  xx = x(idx);
pk = max(xx(tt <= 0.05));
w = round(0.05/(t(2)-t(1)));
env = movmax(xx, [0 w]);
k = find(env < 0.1*pk & tt > 0.05, 1);
if isempty(k), d = 0.5; else, d = min(tt(k), 0.5); end
end

function close_if_loaded(m)
if bdIsLoaded(m), close_system(m,0); end
end
