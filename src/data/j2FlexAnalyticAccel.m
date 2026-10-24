function [aq, am] = j2FlexAnalyticAccel(q, dq, tw, dtw, tau, jp, flex)
%J2FLEXANALYTICACCEL 弾性（二慣性）J2 の解析モデルの加速度（摩擦・重力はリンク側）。
%   [aq, am] = J2FLEXANALYTICACCEL(q, dq, tw, dtw, tau, jp, flex)   tw = qm − q、dtw = dqm − dq
%     JL·ddq  = k·tw + c·dtw + mgL·sin q − Bv·dq − Fc·tanh(dq/ε)
%     Jm·ddqm = τ − k·tw − c·dtw
ts = flex.k*tw + flex.c*dtw;
aq = (ts + jp.mgL*sin(q) - jp.Bv*dq - jp.Fc*tanh(dq/jp.eps)) / flex.JL;
am = (tau - ts) / flex.Jm;
end
