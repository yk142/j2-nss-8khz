function [mdl, flex] = buildJ2FlexPlant(opts)
%BUILDJ2FLEXPLANT 減速機のねじり弾性（二慣性）を入れた J2 単軸プラントを生成する（docs/requirements_flex.md）。
%   [mdl, flex] = BUILDJ2FLEXPLANT(opts)
%     opts は buildJ2Plant のものに加え:
%       opts.modelName  既定 'j2_flex_plant'、opts.slxPath 既定 <dataDir>/models/j2_flex_plant.slx
%       opts.fRes       共振周波数 [Hz]（既定 20）。k を (1/2π)·√(k·(1/Jm + 1/JL)) = fRes から決める
%       opts.zetaTors   ねじりモードの減衰比（既定 0.05）。c = 2·ζ·√(k·Jeq)、Jeq = Jm·JL/(Jm + JL)
%       opts.kScale, opts.cScale  k, c の倍率（剛体極限の確認用。既定 1）
%   モータ側慣性 Jm = 親資産の armature、リンク側 JL = j2Params().M − Jm（M = JL + Jm なので剛体極限で現行と一致）。
%   I/F: Inport 'tau'（モータ側へ入るトルク）、Outport 'q','dq'（リンク）、'tauOut'（トルクの素通し）、'qm','dqm'（モータ側）。
%   flex: 使った値（Jm, JL, k, c, fRes, zeta）。
if nargin < 1, opts = struct(); end
def = struct('modelName','j2_flex_plant','fRes',20,'zetaTors',0.05,'kScale',1,'cScale',1);
f = fieldnames(def);
for i = 1:numel(f)
    if ~isfield(opts,f{i}) || isempty(opts.(f{i})), opts.(f{i}) = def.(f{i}); end
end
c = j2_setup_path();
if ~isfield(opts,'slxPath') || isempty(opts.slxPath), opts.slxPath = fullfile(c.dataDir,'models',[opts.modelName '.slx']); end
flex = j2FlexParams(opts.fRes, opts.zetaTors);
flex.k = flex.k*opts.kScale;  flex.c = flex.c*opts.cScale;
bo = rmfield(opts, {'fRes','zetaTors','kScale','cScale'});
bo.flex = flex;
mdl = buildJ2Plant(bo);
end
