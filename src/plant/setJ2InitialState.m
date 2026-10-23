function setJ2InitialState(mdl, q0, dq0, twist0)
%SETJ2INITIALSTATE J2 単軸プラントの初期角度・初期角速度を設定する。
%   SETJ2INITIALSTATE(mdl, q0)        初期角度 q0 [rad]（初期角速度は 0）
%   SETJ2INITIALSTATE(mdl, q0, dq0)   初期角速度 dq0 [rad/s] も指定
%   SETJ2INITIALSTATE(mdl, q0, dq0, twist0)   弾性プラント（buildJ2FlexPlant）の初期ねじり角 θ − θm [rad]（既定 0 = ばねが自然長）。
%     弾性プラントでは、モータ慣性の初期角速度も dq0 に合わせる。剛体プラントには twist0 は無視される。
%
%   可動関節 'joint1'（buildJ2Plant が J2 を改名したもの）の Position/Velocity Target を設定する。
if nargin < 3, dq0 = 0; end
if nargin < 4, twist0 = 0; end
blk = [mdl '/joint1'];
set_param(blk, ...
    'PositionTargetSpecify','on', 'PositionTargetValue',sprintf('%.12g',q0), 'PositionTargetValueUnits','rad', ...
    'VelocityTargetSpecify','on', 'VelocityTargetValue',sprintf('%.12g',dq0), 'VelocityTargetValueUnits','rad/s');
if ~isempty(find_system(mdl,'SearchDepth',1,'Name','torsSpring'))
    set_param([mdl '/torsSpring'], 'phi_specify','on', 'phi_priority','high', 'phi',sprintf('%.12g',twist0), 'phi_unit','rad');
    set_param([mdl '/motorInertia'], 'w_specify','on', 'w_priority','high', 'w',sprintf('%.12g',-dq0), 'w_unit','rad/s');   % 機械回転ドメインは −θm の座標
    % 機械回転ドメインの角度は RMI が速度を積分したもの（−θ の座標、初期値 0）。qm = q0 − センサ値（出力側で符号反転）
    set_param([mdl '/motorSensor'], 'offset',sprintf('%.12g',-(q0 + twist0)), 'offset_unit','rad');
end
end
