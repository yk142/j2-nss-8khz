function raw = runJ2ScenarioFlex(s, models)
%RUNJ2SCENARIOFLEX 1 シナリオを弾性（二慣性）J2 で 8 kHz 実行し、生データを返す（docs/requirements_nss_flex.md §4）。
%   raw = RUNJ2SCENARIOFLEX(s, models)
%     s       j2ScenarioSet の要素（剛体版 nss8k と同じ表・シード）
%     models  .excite（buildJ2FlexExcitation）、.closed（buildJ2FlexClosedLoop）、.flex（弾性パラメータ）。ロード済み
%   raw: .t .q .dq .qm .dqm .tau（実印加トルク）、.qref .dqref、.q0 .dq0、.contact（リンク角のリミット拘束の割合）
%   runJ2Scenario（剛体）との違い:
%     excitation  バリアはモータ角フィードバック（buildJ2FlexExcitation）。初期ねじりは 0
%     hold / ptp  モータ角 PD（wn = 30 rad/s、ζ = 1）＋ j2FeedforwardFlex（simJ2FlexLoop）。参照・外乱の作り方は剛体版と同じ
jp = j2Params();
dt = 1/jp.fsSim;
raw = struct('t',[],'q',[],'dq',[],'qm',[],'dqm',[],'tau',[],'qref',[],'dqref',[],'q0',0,'dq0',0,'contact',0);
wn = 30;  zeta = 1;

switch s.type
    case 'excitation'
        tVec = (0:dt:s.duration)';
        amp  = s.ampRel * jp.tauRef;
        switch s.pattern
            case 'step'
                rs = RandStream('twister','Seed',s.seed);
                u = j2Signal('step', tVec, jp.tauPeak, struct('tOn',0.2,'sign',1-2*(rand(rs)<0.5)));
            case 'chirp'
                u = j2Signal('chirp', tVec, amp, struct('f0',0.1,'f1',s.fMax));
            case 'micro'
                u = -jp.mgL*sin(s.q0) + j2Signal('bln', tVec, amp, struct('fMax',s.fMax,'seed',s.seed));
            otherwise    % bln_normal / bln_high / bln_low / freefall（接触系は nss8k に無い）
                u = j2Signal('bln', tVec, amp, struct('fMax',s.fMax,'seed',s.seed));
        end
        setJ2Barrier(models.excite, s.qSoftFrac, s.dqSoftFrac);
        o = runJ2Sim(models.excite, tVec, u, s.q0, s.dq0);
        raw.q0 = s.q0;  raw.dq0 = s.dq0;

    case 'hold'
        tVec = (0:dt:s.duration)';
        [qr, dqr, ddqr] = j2SmoothRef(tVec, s.posture, s.refAmp, 1.0, s.seed);
        qr = min(max(qr, jp.qMin + s.refMargin), jp.qMax - s.refMargin);
        tauD = j2Signal('bln', tVec, s.ampRel*jp.tauRef, struct('fMax',s.fMax,'seed',s.seed+1));
        o = simJ2FlexLoop(models.closed, tVec, qr, dqr, ddqr, wn, zeta, models.flex, tauD);
        raw.qref = qr;  raw.dqref = dqr;  raw.q0 = qr(1);

    case 'ptp'
        rs = RandStream('twister','Seed',s.seed);
        WP = jp.qTrainMin + (jp.qTrainMax - jp.qTrainMin)*rand(rs, s.nWp, 1);
        T  = j2PtpSegTime(WP, s.speedScale, 0.5);
        WP2 = WP(1);  T2 = [];
        for k = 1:numel(T)
            WP2(end+1,1) = WP(k+1);  T2(end+1,1) = T(k);        %#ok<AGROW>
            WP2(end+1,1) = WP(k+1);  T2(end+1,1) = 0.2;         %#ok<AGROW>
        end
        tVec = (0:dt:min(sum(T2), s.duration))';
        [qr, dqr, ddqr] = j2PtpTrajectory(tVec, WP2, T2);
        o = simJ2FlexLoop(models.closed, tVec, qr, dqr, ddqr, wn, zeta, models.flex);
        raw.qref = qr;  raw.dqref = dqr;  raw.q0 = qr(1);

    otherwise
        error('runJ2ScenarioFlex:type', '未知の type: %s', s.type);
end
raw.t = o.t;  raw.q = o.q;  raw.dq = o.dq;  raw.qm = o.qm;  raw.dqm = o.dqm;  raw.tau = o.tau;
if strcmp(s.type,'excitation'), raw.dq0 = s.dq0; end
raw.contact = mean(raw.q <= jp.qMin + deg2rad(0.1) | raw.q >= jp.qMax - deg2rad(0.1));
end
