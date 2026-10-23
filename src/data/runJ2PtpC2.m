function raw = runJ2PtpC2(s, mC, cond)
%RUNJ2PTPC2 PTP ベンチマークの 1 軌道を、FF の係数誤差と外乱パルスを加えて閉ループで実行する（(c2) の真値）。
%   raw = RUNJ2PTPC2(s, mC, cond)
%     s     j2ScenarioSet の ptp 要素（参照軌道は runJ2Scenario の 'ptp' と同一）
%     mC    ロード済みの閉ループモデル名（PD ゲインは呼び出し側で設定済み）
%     cond  .ffErr   FF の係数（M, mgL, Bv, Fc）の相対誤差（例 −0.15）
%           .distAmp 外乱パルスの大きさ [N·m]、.distDur パルス長 [s]（0 で外乱なし）
%   外乱パルスの開始時刻は s.seed から決める（モデルと真値で同じ）。入力は [q_ref dq_ref tau_ff tau_dist]。
%   raw: .t .q .dq .tau（実印加トルク）、.qref .dqref .tauFf .tauD（入力。Python 側の閉ループにそのまま使う）
jp = j2Params();
dt = 1/jp.fsSim;
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
f = 1 + cond.ffErr;
tauFf = f*(jp.M*ddqr - jp.mgL*sin(qr) + jp.Bv*dqr + jp.Fc*tanh(dqr/jp.eps));
tauD = zeros(size(tVec));
if cond.distAmp ~= 0 && cond.distDur > 0
    t0 = rand(RandStream('twister','Seed',s.seed+7))*(tVec(end) - cond.distDur);
    tauD(tVec >= t0 & tVec < t0 + cond.distDur) = cond.distAmp;
end
o = runJ2Sim(mC, tVec, [qr dqr tauFf tauD], qr(1), 0);
raw = struct('t',o.t,'q',o.q,'dq',o.dq,'tau',o.tau,'qref',qr,'dqref',dqr,'tauFf',tauFf,'tauD',tauD);
end
