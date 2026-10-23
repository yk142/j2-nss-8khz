function out = simJ2FlexLoop(mdl, tVec, qr, dqr, ddqr, wn, zeta, flex, tauD)
%SIMJ2FLEXLOOP 弾性 J2 の閉ループ（モータ角 PD＋FF）を、参照軌道から作って実行する。
%   out = SIMJ2FLEXLOOP(mdl, tVec, qr, dqr, ddqr, wn, zeta, flex [, tauD])
%     mdl   buildJ2FlexClosedLoop のモデル（ロード済み）、qr/dqr/ddqr リンク角の参照（tVec と同じ長さ）
%     tauD  外乱トルク列（既定 0）
%   out: .t .q .dq .tau（飽和後の実印加）.qm .dqm、.qr .qmRef（参照）、.tauFf .tauD
jp = j2Params();
if nargin < 9 || isempty(tauD), tauD = zeros(size(tVec)); end
tVec = tVec(:);  qr = qr(:);  dqr = dqr(:);  ddqr = ddqr(:);  tauD = tauD(:);
[tauFf, qmRef, dqmRef] = j2FeedforwardFlex(qr, dqr, ddqr, tVec(2)-tVec(1), flex);
set_param([mdl '/Kp'],'Gain',sprintf('%.10g',jp.M*wn^2));
set_param([mdl '/Kd'],'Gain',sprintf('%.10g',2*zeta*jp.M*wn));
twist0 = qmRef(1) - qr(1);
setJ2InitialState(mdl, qr(1), dqr(1), twist0);
assignin('base','J2_in',[tVec qmRef dqmRef tauFf tauD]);
o = sim(mdl,'StopTime',num2str(tVec(end),'%.12g'),'LoadExternalInput','on','ExternalInput','J2_in', ...
    'SaveOutput','on','SaveFormat','Dataset');
e = o.yout;
out.t = e.getElement(1).Values.Time;
out.q = e.getElement(1).Values.Data(:,1);   out.dq = e.getElement(2).Values.Data(:,1);
out.tau = e.getElement(3).Values.Data(:,1);
out.qm = e.getElement(4).Values.Data(:,1);  out.dqm = e.getElement(5).Values.Data(:,1);
out.qr = qr;  out.qmRef = qmRef;  out.tauFf = tauFf;  out.tauD = tauD;
end
