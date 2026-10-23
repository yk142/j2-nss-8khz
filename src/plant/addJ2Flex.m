function addJ2Flex(mdl, p, flex)
%ADDJ2FLEX J2 に減速機のねじり弾性（二慣性）を加える。addC8Armature の代わりに呼ぶ（docs/requirements_flex.md）。
%   ADDJ2FLEX(mdl, p, flex)  addC8Actuation 適用済み・addC8Friction 適用前のモデルに対して、
%     flex.Jm  モータ側慣性 [kg·m²]、flex.k  ねじりばね [N·m/rad]、flex.c  ねじりダンパ [N·m·s/rad]
%   構成（RMI の R–C は 1-D 機械回転ドメイン。R の角速度は −dθ（関節角速度と逆符号。実測）なので、モータ側は −θm の座標で組む）:
%     RMI.R —[ばね k ∥ ダンパ c]— node(モータ角 θm) —[慣性 Jm]— 回転リファレンス
%     制御トルク τ → 理想トルク源 → node（モータ側に入る）
%     リンク側の関節へ入るトルク = RMI の反力（= ばね・ダンパのトルク）。addC8Friction は tauAdd_1 の出力に摩擦を足す
%   追加の出力: 3 番目 'tauOut'（制御トルクの素通し）、4 番目 'qm'、5 番目 'dqm'（モータ角・角速度）。
assert(p.nJoints == 1, 'addJ2Flex:nJoints', 'J2 単軸（nJoints=1）専用です');
RMI   = ['fl_lib/Mechanical/Multibody Interfaces/Rotational Multibody' newline 'Interface'];
INER  = 'fl_lib/Mechanical/Rotational Elements/Inertia';
SPR   = 'fl_lib/Mechanical/Rotational Elements/Rotational Spring';
DMP   = 'fl_lib/Mechanical/Rotational Elements/Rotational Damper';
REF   = ['fl_lib/Mechanical/Rotational Elements/Mechanical' newline 'Rotational Reference'];
TSRC  = 'fl_lib/Mechanical/Mechanical Sources/Ideal Torque Source';
SENS  = 'fl_lib/Mechanical/Mechanical Sensors/Ideal Rotational Motion Sensor';
PSADD = 'fl_lib/Physical Signals/Functions/PS Add';
PSCON = 'fl_lib/Physical Signals/Sources/PS Constant';
PS2S  = ['nesl_utility/PS-Simulink' newline 'Converter'];
jb = 'joint1';  sc = 'S2PS_1';  yb = 700;   % ブロックの位置は見やすさだけの目安

% 制御トルク → 関節 の直結を外す（制御トルクはモータ側へ入れる）
delete_line(mdl, [sc '/RConn1'], [jb '/LConn2']);

% 関節へ入るトルク: RMI の反力 + 0（addC8Friction が tauAdd_1 を見つけて、その出力に摩擦を足す）
add_block(PSADD, [mdl '/tauAdd_1'], 'Position', [230 yb 260 yb+30]);
add_block(PSCON, [mdl '/zeroTau'], 'Position', [150 yb+60 190 yb+80]);
set_param([mdl '/zeroTau'], 'constant', '0', 'constant_unit', 'N*m');
add_line(mdl, 'zeroTau/RConn1', 'tauAdd_1/LConn2', 'autorouting', 'on');
add_line(mdl, 'tauAdd_1/RConn1', [jb '/LConn2'], 'autorouting', 'on');

% RMI（リンク角に追従する R と、反力トルクの出力 t）
add_block(RMI, [mdl '/RMI_1'], 'Position', [330 yb 380 yb+40]);
add_line(mdl, [jb '/RConn3'], 'RMI_1/RConn1', 'autorouting', 'on');
add_line(mdl, 'RMI_1/LConn1', 'tauAdd_1/LConn1', 'autorouting', 'on');

% 機械回転ドメイン: 回転リファレンス、ばね・ダンパ（R–node）、モータ慣性（node–リファレンス）、トルク源、センサ
add_block(REF,  [mdl '/rotRef_1'],   'Position', [560 yb+120 600 yb+150]);
add_block(SPR,  [mdl '/torsSpring'], 'Position', [450 yb-20 490 yb+10]);
add_block(DMP,  [mdl '/torsDamper'], 'Position', [450 yb+30 490 yb+60]);
add_block(INER, [mdl '/motorInertia'], 'Position', [600 yb 640 yb+30]);
add_block(TSRC, [mdl '/motorTorque'], 'Position', [600 yb+60 640 yb+100]);
add_block(SENS, [mdl '/motorSensor'], 'Position', [700 yb 740 yb+40]);
set_param([mdl '/torsSpring'], 'spr_rate', sprintf('%.10g', flex.k), 'spr_rate_unit', 'N*m/rad');
set_param([mdl '/torsDamper'], 'D', sprintf('%.10g', flex.c), 'D_unit', 'N*m/(rad/s)');
set_param([mdl '/motorInertia'], 'inertia', sprintf('%.10g', flex.Jm));

% 結線（ポートは実機検証: トルク源 = LConn1 保存(R)・RConn1 = 入力 S・RConn2 保存(C)、センサ = LConn1 保存・RConn1 保存・RConn2 = 速度・RConn3 = 角度）
% RMI の R 端の角速度は関節角速度と逆符号（実測）。モータ側の機械回転ドメインは「−θm」の座標で組み、
% 制御トルクは符号を反転して入れ（PS Gain −1）、モータ角・角速度の出力は Simulink Gain −1 で戻す。
add_block('fl_lib/Physical Signals/Functions/PS Gain', [mdl '/motorSign'], 'Position', [540 yb+60 570 yb+90]);
set_param([mdl '/motorSign'], 'gain', '-1', 'gain_unit', '1');
add_line(mdl, [sc '/RConn1'], 'motorSign/LConn1', 'autorouting', 'on');          % 制御トルク → 符号反転 → モータ側
add_line(mdl, 'motorSign/RConn1', 'motorTorque/RConn1', 'autorouting', 'on');
add_line(mdl, 'RMI_1/LConn2', 'torsSpring/LConn1', 'autorouting', 'on');           % リンク側（R）→ ばね・ダンパ
add_line(mdl, 'RMI_1/LConn2', 'torsDamper/LConn1', 'autorouting', 'on');
add_line(mdl, 'torsSpring/RConn1', 'motorInertia/LConn1', 'autorouting', 'on');    % node（モータ角 θm）
add_line(mdl, 'torsDamper/RConn1', 'motorInertia/LConn1', 'autorouting', 'on');
add_line(mdl, 'motorTorque/LConn1', 'motorInertia/LConn1', 'autorouting', 'on');
add_line(mdl, 'motorSensor/LConn1', 'motorInertia/LConn1', 'autorouting', 'on');
add_line(mdl, 'RMI_1/RConn2', 'rotRef_1/LConn1', 'autorouting', 'on');             % 基準（C）
add_line(mdl, 'motorTorque/RConn2', 'rotRef_1/LConn1', 'autorouting', 'on');
add_line(mdl, 'motorSensor/RConn1', 'rotRef_1/LConn1', 'autorouting', 'on');

% 追加の出力: 3 'tauOut'（制御トルクの素通し）、4 'qm'、5 'dqm'
add_block('simulink/Sinks/Out1', [mdl '/tauOut'], 'Position', [960 600 990 614]);  set_param([mdl '/tauOut'], 'Port', '3');
add_line(mdl, 'tau_demux/1', 'tauOut/1', 'autorouting', 'on');
add_block('simulink/Sinks/Out1', [mdl '/qm'], 'Position', [960 660 990 674]);   set_param([mdl '/qm'], 'Port', '4');
add_block('simulink/Sinks/Out1', [mdl '/dqm'], 'Position', [960 720 990 734]);  set_param([mdl '/dqm'], 'Port', '5');
add_block(PS2S, [mdl '/PS2S_qm'], 'Position', [800 660 840 674]);
add_block(PS2S, [mdl '/PS2S_dqm'], 'Position', [800 720 840 734]);
add_line(mdl, 'motorSensor/RConn3', 'PS2S_qm/LConn1', 'autorouting', 'on');
add_line(mdl, 'motorSensor/RConn2', 'PS2S_dqm/LConn1', 'autorouting', 'on');
add_block('simulink/Math Operations/Gain', [mdl '/negQm'], 'Gain', '-1', 'Position', [860 660 890 674]);
add_block('simulink/Math Operations/Gain', [mdl '/negDqm'], 'Gain', '-1', 'Position', [860 720 890 734]);
add_line(mdl, 'PS2S_qm/1', 'negQm/1', 'autorouting', 'on');   add_line(mdl, 'negQm/1', 'qm/1', 'autorouting', 'on');
add_line(mdl, 'PS2S_dqm/1', 'negDqm/1', 'autorouting', 'on'); add_line(mdl, 'negDqm/1', 'dqm/1', 'autorouting', 'on');
end
