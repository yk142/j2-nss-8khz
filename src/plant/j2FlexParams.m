function flex = j2FlexParams(fRes, zetaTors)
%J2FLEXPARAMS 弾性（二慣性）J2 の公称パラメータ（docs/flex_plant.md）。
%   flex = J2FLEXPARAMS([fRes [, zetaTors]])   既定 fRes = 20 Hz、zetaTors = 0.05
%   flex: Jm（モータ側＝親資産の armature）、JL = M − Jm、k（ねじり剛性）、c（ねじり減衰）、fRes、zeta
if nargin < 1 || isempty(fRes), fRes = 20; end
if nargin < 2 || isempty(zetaTors), zetaTors = 0.05; end
c = j2_setup_path();
p = c8_params(c.meshDir);  jp = j2Params();
flex.Jm = p.armature(c.jointIdx);  flex.JL = jp.M - flex.Jm;
Jeq = flex.Jm*flex.JL/(flex.Jm + flex.JL);
flex.k = (2*pi*fRes)^2/(1/flex.Jm + 1/flex.JL);
flex.c = 2*zetaTors*sqrt(flex.k*Jeq);
flex.fRes = fRes;  flex.zeta = zetaTors;
end
