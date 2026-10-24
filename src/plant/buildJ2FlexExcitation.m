function [mdl, flex] = buildJ2FlexExcitation(opts)
%BUILDJ2FLEXEXCITATION 弾性（二慣性）J2 の開ループ励振モデル（励振トルク＋ソフトリミットバリア＋トルク飽和）。
%   [mdl, flex] = BUILDJ2FLEXEXCITATION(opts)   docs/requirements_nss_flex.md §4.2
%     opts は buildJ2FlexPlant のものに加え:
%       opts.modelName  既定 'j2_flex_excite'
%       opts.barrierWn  バリアの固有角周波数 [rad/s]（既定 60）
%   モデル I/F: Inport 'tauEx'[スカラ] → Outport 1 'q'、2 'dq'（リンク）、3 'tauOut'（飽和後の実印加トルク）、4 'qm'、5 'dqm'
%   buildJ2Excitation（剛体）との違い: バリアは**モータ角・モータ角速度**のフィードバック（リンク角では wn ≳ 7 rad/s で不安定）。
%     τ = sat( τ_ex − Kp_bar·deadzone(qm; qSoftMin, qSoftMax) − Kd_bar·deadzone(dqm; ±dqSoft) )
%   Kp_bar = M·wn²、Kd_bar = 2·M·wn。安全帯は setJ2Barrier(mdl, qSoftFrac, dqSoftFrac)（ブロック名 qDead/dqDead は剛体版と同じ）。
if nargin < 1, opts = struct(); end
if ~isfield(opts,'modelName') || isempty(opts.modelName), opts.modelName = 'j2_flex_excite'; end
if ~isfield(opts,'barrierWn') || isempty(opts.barrierWn), opts.barrierWn = 60; end
bw = opts.barrierWn;  opts = rmfield(opts,'barrierWn');
[mdl, flex] = buildJ2FlexPlant(opts);
jp = j2Params();

set_param([mdl '/tau'],'Name','tauEx');
delete_line(mdl,'tauEx/1','tau_demux/1');
add_block('simulink/Discontinuities/Dead Zone',[mdl '/qDead'],'LowerValue','-1','UpperValue','1');
add_block('simulink/Discontinuities/Dead Zone',[mdl '/dqDead'],'LowerValue','-1','UpperValue','1');
add_block('simulink/Math Operations/Gain',[mdl '/KpBar'],'Gain',sprintf('-%.10g',jp.M*bw^2));
add_block('simulink/Math Operations/Gain',[mdl '/KdBar'],'Gain',sprintf('-%.10g',2*jp.M*bw));
add_block('simulink/Math Operations/Sum',[mdl '/tauSum'],'Inputs','+++');
add_block('simulink/Discontinuities/Saturation',[mdl '/tauSat'], ...
    'UpperLimit',sprintf('%.10g',jp.tauPeak),'LowerLimit',sprintf('%.10g',-jp.tauPeak));
add_line(mdl,'negQm/1','qDead/1');   add_line(mdl,'qDead/1','KpBar/1');
add_line(mdl,'negDqm/1','dqDead/1'); add_line(mdl,'dqDead/1','KdBar/1');
add_line(mdl,'tauEx/1','tauSum/1');  add_line(mdl,'KpBar/1','tauSum/2'); add_line(mdl,'KdBar/1','tauSum/3');
add_line(mdl,'tauSum/1','tauSat/1'); add_line(mdl,'tauSat/1','tau_demux/1');   % tau_demux/1 → S2PS_1 と tauOut
setJ2Barrier(mdl, 0.85, 0.60);
save_system(mdl, get_param(mdl,'FileName'));
end
