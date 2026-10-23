classdef test_j2_flex_plant < matlab.unittest.TestCase
%TEST_J2_FLEX_PLANT 弾性（二慣性）J2 プラントの物理検証（docs/requirements_flex.md §4、検証 1〜6）。
%   実行: results = runtests('test/test_j2_flex_plant.m')   （プラント 4 本を一時フォルダに生成。約 3 分）
%   出力の順: 1 q、2 dq、3 tauOut、4 qm、5 dqm（弾性）／ 1 q、2 dq（剛体）

    properties
        jp
        dt
        flex
        mFlex       % 弾性・摩擦あり
        mFree       % 弾性・摩擦なし（自由振動の周波数・減衰・エネルギー）
        mStiff      % 弾性（k を 100 倍・c を 10 倍）・摩擦あり（剛体極限）
        mRigid      % 剛体（現行）
        tmpDir
    end

    methods (TestClassSetup)
        function build(tc)
            repo = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(repo,'config'), fullfile(repo,'src'), genpath(fullfile(repo,'src')));
            tc.jp = j2Params();  tc.dt = 1/tc.jp.fsSim;
            tc.tmpDir = tempname;  mkdir(tc.tmpDir);
            mk = @(name, o) fullfile(tc.tmpDir, [name '.slx']);
            o = struct('modelName','jf_flex','slxPath',mk('jf_flex'));
            [tc.mFlex, tc.flex] = buildJ2FlexPlant(o);
            tc.mFree = buildJ2FlexPlant(struct('modelName','jf_free','slxPath',mk('jf_free'),'noFriction',true));
            tc.mStiff = buildJ2FlexPlant(struct('modelName','jf_stiff','slxPath',mk('jf_stiff'),'kScale',100,'cScale',10));
            tc.mRigid = buildJ2Plant(struct('modelName','jf_rigid','slxPath',mk('jf_rigid')));
        end
    end

    methods (TestClassTeardown)
        function cleanup(tc)
            for m = {tc.mFlex, tc.mFree, tc.mStiff, tc.mRigid}
                if bdIsLoaded(m{1}), close_system(m{1},0); end
            end
            if isfolder(tc.tmpDir), rmdir(tc.tmpDir,'s'); end
        end
    end

    methods (Access = private)
        function y = simPlant(tc, mdl, tVec, u, q0, dq0, twist0)
            % 実行して q dq（と弾性なら tau qm dqm）を列で返す
            if nargin < 7, twist0 = 0; end
            setJ2InitialState(mdl, q0, dq0, twist0);
            assignin('base','J2_in',[tVec u]);
            o = sim(mdl,'StopTime',num2str(tVec(end),'%.12g'),'LoadExternalInput','on','ExternalInput','J2_in', ...
                'SaveOutput','on','SaveFormat','Dataset');
            e = o.yout;  y.t = e.getElement(1).Values.Time;
            y.q = e.getElement(1).Values.Data(:,1);  y.dq = e.getElement(2).Values.Data(:,1);
            if e.numElements >= 5
                y.qm = e.getElement(4).Values.Data(:,1);  y.dqm = e.getElement(5).Values.Data(:,1);
            end
        end

        function [t, u, tauHold, q0] = pulseCase(tc, amp, tEnd)
            % 安定な釣合い（リンクが下向きに近い θ=−2.2 rad: cosθ<0）で、保持トルクに短いパルスを重ねる
            q0 = -2.2;  tauHold = -tc.jp.mgL*sin(q0);
            t = (0:tc.dt:tEnd)';  u = tauHold*ones(size(t));
            k = t >= 0.1 & t < 0.102;  u(k) = u(k) + amp;
        end
    end

    methods (Test)
        function rigidLimit(tc)
            % 検証 1: k を 100 倍にした弾性プラントは、現行の剛体プラントと同じ運動（θ の差が変化量の 1% 以内）
            [t, u, ~, q0] = tc.pulseCase(0, 1.0);   % 保持トルクのまま、0.1〜0.4 s に 40 N·m のステップを足す
            k = t >= 0.1 & t < 0.4;  u(k) = u(k) + 40;
            a = tc.simPlant(tc.mStiff, t, u, q0, 0);   b = tc.simPlant(tc.mRigid, t, u, q0, 0);
            n = min(numel(a.q), numel(b.q));
            swing = max(b.q(1:n)) - min(b.q(1:n));
            tc.verifyGreaterThan(swing, 0.05, '剛体の運動が小さすぎて比較にならない');
            tc.verifyLessThan(max(abs(a.q(1:n) - b.q(1:n)))/swing, 0.01);
        end

        function resonanceFrequency(tc)
            % 検証 2: 自由振動のねじり角（θm − θ）の周波数が 20 Hz の ±5%
            [t, u, ~, q0] = tc.pulseCase(100, 1.0);
            a = tc.simPlant(tc.mFree, t, u, q0, 0);
            i = a.t > 0.11;  w = a.qm(i) - a.q(i);  w = w - mean(w);
            nf = 2^17;  Y = abs(fft(w.*hann(numel(w)), nf));  f = (0:nf-1)'/(nf*tc.dt);
            h = f < 100;  [~, ix] = max(Y.*h);
            fpk = f(ix);
            tc.verifyEqual(fpk, tc.flex.fRes, 'RelTol', 0.05);
        end

        function modalDamping(tc)
            % 検証 3: ねじり振動の包絡線の減衰から求めた減衰比が 0.05 の ±30%（摩擦なしの別条件）。
            % 初期ねじり 0.05 rad から、トルクなしの自由振動（静的なねじり 0.003 rad より十分大きい）
            t = (0:tc.dt:0.5)';  u = zeros(size(t));
            a = tc.simPlant(tc.mFree, t, u, -2.2, 0, 0.05);
            w = a.qm - a.q;
            [pk, loc] = findpeaks(abs(w), 'MinPeakDistance', round(0.02/tc.dt));
            tc.assertGreaterThan(numel(pk), 6);
            n = min(numel(pk), 12);
            pf = polyfit(a.t(loc(1:n)), log(pk(1:n)), 1);        % 包絡線の指数減衰 exp(−σ t)
            zeta = -pf(1)/(2*pi*tc.flex.fRes);
            tc.verifyEqual(zeta, tc.flex.zeta, 'RelTol', 0.30);
        end

        function staticTwist(tc)
            % 検証 4: 保持トルクで静止したときのねじり角 = 保持トルク / k（5% 以内）。摩擦なしで十分長く待って減衰させる
            [t, u, tauHold, q0] = tc.pulseCase(0, 2.0);
            a = tc.simPlant(tc.mFree, t, u, q0, 0);
            tw = mean(a.qm(a.t > 1.8) - a.q(a.t > 1.8));
            tc.verifyEqual(tw, tauHold/tc.flex.k, 'RelTol', 0.05);
        end

        function energyNotIncreasing(tc)
            % 検証 5: 摩擦なしで、パルス後の力学エネルギー（運動 + ばね + 重力 − トルクの仕事）が増えない
            [t, u, tauHold, q0] = tc.pulseCase(100, 1.0);
            a = tc.simPlant(tc.mFree, t, u, q0, 0);
            f = tc.flex;  jp = tc.jp;
            E = 0.5*f.Jm*a.dqm.^2 + 0.5*f.JL*a.dq.^2 + 0.5*f.k*(a.qm - a.q).^2 + jp.mgL*cos(a.q) - tauHold*a.qm;
            i = a.t >= 0.11;  Ei = E(i);
            Evib = max(Ei) - min(Ei);
            tc.verifyLessThan(max(diff(Ei)), 1e-3*Evib, 'エネルギーが増えている');
            tc.verifyLessThan(Ei(end), Ei(1), 'エネルギーが減っていない（ダンパが効いていない）');
        end

        function stableUnderLargeInputs(tc)
            % 検証 6: トルク上限のステップと、定格振幅のチャープで、発散・NaN なし（8 kHz、ode14x）
            t = (0:tc.dt:0.5)';
            q0 = -2.2;
            u1 = tc.jp.tauPeak*(t >= 0.05);
            a = tc.simPlant(tc.mFlex, t, u1, q0, 0);
            tc.verifyTrue(all(isfinite([a.q; a.dq; a.qm; a.dqm])));
            u2 = tc.jp.tauRated*sin(2*pi*(1 + 49*t/0.5/2).*t);     % 1→50 Hz のチャープ
            b = tc.simPlant(tc.mFlex, t, u2, q0, 0);
            tc.verifyTrue(all(isfinite([b.q; b.dq; b.qm; b.dqm])));
            tc.verifyLessThan(max(abs(b.qm - b.q)), 0.5, 'ねじりが過大');
        end
    end
end
