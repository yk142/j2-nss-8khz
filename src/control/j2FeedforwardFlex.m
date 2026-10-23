function [tauFf, qmRef, dqmRef] = j2FeedforwardFlex(qr, dqr, ddqr, dt, flex)
%J2FEEDFORWARDFLEX 弾性 J2 の閉ループ用の FF トルクとモータ角の参照（docs/requirements_flex_loop.md §2）。
%   [tauFf, qmRef, dqmRef] = J2FEEDFORWARDFLEX(qr, dqr, ddqr, dt, flex)
%     tauFf   剛体の逆動力学（j2Feedforward と同じ。M = Jm + JL）
%     qmRef   θ_ref + twist_ref、twist_ref = (tauFf − Jm·ddq_ref)/k（剛体運動のときのばねのたわみ）
%     dqmRef  qmRef の数値微分（dt = 刻み幅）
tauFf = j2Feedforward(qr, dqr, ddqr);
twist = (tauFf - flex.Jm*ddqr)/flex.k;
qmRef = qr + twist;
dqmRef = dqr + gradient(twist, dt);
end
