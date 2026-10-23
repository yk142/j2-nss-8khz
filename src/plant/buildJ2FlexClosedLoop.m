function [mdl, flex] = buildJ2FlexClosedLoop(opts)
%BUILDJ2FLEXCLOSEDLOOP 弾性（二慣性）J2 の PD＋フィードフォワード閉ループモデル（モータ角フィードバック、トルク飽和付き）。
%   [mdl, flex] = BUILDJ2FLEXCLOSEDLOOP(opts)   docs/requirements_flex_loop.md
%     opts は buildJ2FlexPlant のものに加え:
%       opts.modelName  既定 'j2_flex_closedloop'
%       opts.wn         PD の固有角周波数 [rad/s]（既定 30）、opts.zeta 減衰比（既定 1）
%   モデル I/F:
%     Inport 'ref'[4] = [qm_ref, dqm_ref, tau_ff, tau_dist]
%     Outport 1 'q'、2 'dq'（リンク）、3 'tauOut'（飽和後の実印加トルク）、4 'qm'、5 'dqm'（モータ側）
%     τ = sat( tau_ff + tau_dist + Kp·(qm_ref − qm) + Kd·(dqm_ref − dqm) )，±tauPeak で飽和
%   Kp = M·wn²、Kd = 2·ζ·M·wn（M = Jm + JL = 実効慣性。剛体の buildJ2ClosedLoop と同じ式）。
%   ゲインは set_param([mdl '/Kp'],'Gain',...) / [mdl '/Kd'] で後から変えられる（simJ2FlexLoop が行う）。
if nargin < 1, opts = struct(); end
if ~isfield(opts,'modelName') || isempty(opts.modelName), opts.modelName = 'j2_flex_closedloop'; end
wn = 30;   if isfield(opts,'wn') && ~isempty(opts.wn), wn = opts.wn; end
zeta = 1;  if isfield(opts,'zeta') && ~isempty(opts.zeta), zeta = opts.zeta; end
for f = {'wn','zeta'}, if isfield(opts,f{1}), opts = rmfield(opts,f{1}); end, end
[mdl, flex] = buildJ2FlexPlant(opts);
jp = j2Params();

set_param([mdl '/tau'],'Name','ref');
set_param([mdl '/ref'],'PortDimensions','4');
delete_line(mdl,'ref/1','tau_demux/1');
add_block('simulink/Signal Routing/Demux',[mdl '/refDemux'],'Outputs','4');
add_block('simulink/Math Operations/Sum',[mdl '/eq'],'Inputs','+-');
add_block('simulink/Math Operations/Sum',[mdl '/edq'],'Inputs','+-');
add_block('simulink/Math Operations/Gain',[mdl '/Kp'],'Gain',sprintf('%.10g',jp.M*wn^2));
add_block('simulink/Math Operations/Gain',[mdl '/Kd'],'Gain',sprintf('%.10g',2*zeta*jp.M*wn));
add_block('simulink/Math Operations/Sum',[mdl '/tauSum'],'Inputs','++++');
add_block('simulink/Discontinuities/Saturation',[mdl '/tauSat'], ...
    'UpperLimit',sprintf('%.10g',jp.tauPeak),'LowerLimit',sprintf('%.10g',-jp.tauPeak));
add_line(mdl,'ref/1','refDemux/1');
add_line(mdl,'refDemux/1','eq/1');   add_line(mdl,'negQm/1','eq/2');          % モータ角
add_line(mdl,'refDemux/2','edq/1');  add_line(mdl,'negDqm/1','edq/2');        % モータ角速度
add_line(mdl,'eq/1','Kp/1');         add_line(mdl,'edq/1','Kd/1');
add_line(mdl,'Kp/1','tauSum/1');     add_line(mdl,'Kd/1','tauSum/2');
add_line(mdl,'refDemux/3','tauSum/3'); add_line(mdl,'refDemux/4','tauSum/4');
add_line(mdl,'tauSum/1','tauSat/1'); add_line(mdl,'tauSat/1','tau_demux/1');   % tau_demux/1 → S2PS_1 と tauOut（飽和後の実印加）
save_system(mdl, get_param(mdl,'FileName'));
end
