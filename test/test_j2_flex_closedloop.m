classdef test_j2_flex_closedloop < matlab.unittest.TestCase
%TEST_J2_FLEX_CLOSEDLOOP 弾性 J2 のモータ角 PD 閉ループの確認（docs/requirements_flex_loop.md §3、確認 1・2）。
%   実行: results = runtests('test/test_j2_flex_closedloop.m')   （約 6 分）

    properties
        jp
        dt
        flex
        mLoop       % 摩擦あり（確認 2: 全ゲインで安定）
        mFree       % 摩擦なし（確認 1: リンギングの減衰を線形モデルと比べる）
        tmpDir
    end

    properties (TestParameter)
        zeta = {0.7, 1.0, 1.5};
    end

    methods (TestClassSetup)
        function build(tc)
            repo = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(repo,'config'), fullfile(repo,'src'), genpath(fullfile(repo,'src')));
            tc.jp = j2Params();  tc.dt = 1/tc.jp.fsSim;
            tc.tmpDir = tempname;  mkdir(tc.tmpDir);
            [tc.mLoop, tc.flex] = buildJ2FlexClosedLoop(struct('modelName','jfc_loop','slxPath',fullfile(tc.tmpDir,'jfc_loop.slx')));
            tc.mFree = buildJ2FlexClosedLoop(struct('modelName','jfc_free','slxPath',fullfile(tc.tmpDir,'jfc_free.slx'),'noFriction',true));
        end
    end

    methods (TestClassTeardown)
        function cleanup(tc)
            for m = {tc.mLoop, tc.mFree}
                if bdIsLoaded(m{1}), close_system(m{1},0); end
            end
            if isfolder(tc.tmpDir), rmdir(tc.tmpDir,'s'); end
        end
    end

    methods (Access = private)
        function s = linearPoles(tc, wn, zeta)
            % 線形（重力・摩擦なし）の閉ループの極: 状態 [θ θm dθ dθm]、モータ角・角速度フィードバック
            f = tc.flex;  M = f.Jm + f.JL;  Kp = M*wn^2;  Kd = 2*zeta*M*wn;
            A = zeros(4);  A(1,3) = 1;  A(2,4) = 1;
            A(3,:) = [-f.k f.k -f.c f.c]/f.JL;
            A(4,:) = [f.k -f.k f.c -f.c]/f.Jm + [0 -Kp 0 -Kd]/f.Jm;
            s = eig(A);
        end
    end

    methods (Test)
        function ringdownMatchesLinearModel(tc)
            % 確認 1: 停止中に外乱パルスを入れたときの、ねじりのリンギングの減衰率が、線形モデルの振動モードの実部と ±30%。
            % wn = 90、ζ = 1 を使う（振動モードの減衰が小さく、約 11 Hz で長く鳴るので、ピークの減衰を読める）
            wn = 90;  zeta = 1;
            q0 = -2.2;  t = (0:tc.dt:0.8)';
            qr = q0*ones(size(t));  zr = zeros(size(t));
            tauD = zeros(size(t));  tauD(t >= 0.1 & t < 0.102) = 300;
            o = simJ2FlexLoop(tc.mFree, t, qr, zr, zr, wn, zeta, tc.flex, tauD);
            tw = o.qm - o.q;  tw = tw - tw(end);
            i = o.t > 0.12;  tt = o.t(i);
            [pk, loc] = findpeaks(abs(tw(i)), 'MinPeakDistance', round(0.03/tc.dt));
            tc.assertGreaterThan(numel(pk), 6);
            n = min(numel(pk), 10);
            pf = polyfit(tt(loc(1:n)), log(pk(1:n)), 1);
            sigmaSim = -pf(1);
            p = tc.linearPoles(wn, zeta);  r = p(abs(imag(p)) > 40);
            sigmaLin = -max(real(r));
            tc.verifyEqual(sigmaSim, sigmaLin, 'RelTol', 0.30);
        end

        function stableForAllGains(tc, zeta)
            % 確認 2: wn = 15〜120 rad/s の全ゲインで、PTP の最初の 1.5 s が発散しない（ねじり < 0.3 rad、追従誤差 < 5°）
            for wn = [15 30 60 90 120]
                [t, qr, dqr, ddqr] = tc.ptpHead(1.5);
                o = simJ2FlexLoop(tc.mLoop, t, qr, dqr, ddqr, wn, zeta, tc.flex);
                ok = all(isfinite([o.q; o.qm; o.tau]));
                tc.verifyTrue(ok, sprintf('wn=%d ζ=%.1f: 発散', wn, zeta));
                tc.verifyLessThan(max(abs(o.qm - o.q)), 0.3, sprintf('wn=%d ζ=%.1f: ねじり過大', wn, zeta));
                tc.verifyLessThan(rad2deg(max(abs(o.q - qr))), 5, sprintf('wn=%d ζ=%.1f: 追従誤差過大', wn, zeta));
            end
        end
    end

    methods (Access = private)
        function [t, qr, dqr, ddqr] = ptpHead(tc, tEnd)
            % benchmark の ptp_v80_003 の参照軌道の先頭 tEnd 秒
            S = j2ScenarioSet('nss8k');  s = S(strcmp({S.name},'ptp_v80_003'));
            rs = RandStream('twister','Seed',s.seed);
            WP = tc.jp.qTrainMin + (tc.jp.qTrainMax - tc.jp.qTrainMin)*rand(rs, s.nWp, 1);
            T = j2PtpSegTime(WP, s.speedScale, 0.5);
            WP2 = WP(1);  T2 = [];
            for k = 1:numel(T)
                WP2(end+1,1) = WP(k+1);  T2(end+1,1) = T(k);        %#ok<AGROW>
                WP2(end+1,1) = WP(k+1);  T2(end+1,1) = 0.2;         %#ok<AGROW>
            end
            t = (0:tc.dt:min(tEnd, sum(T2)))';
            [qr, dqr, ddqr] = j2PtpTrajectory(t, WP2, T2);
        end
    end
end
