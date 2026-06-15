
@alias_sitetype Qubit => SpinHalf
@alias_sitetype "S=1/2" => SpinHalf

siteind(::SiteType{:SpinHalf}) = Index(2, :Site; sitetype=SiteType{:SpinHalf})

state(::SiteType{:SpinHalf}, ::StateName{:Up}) = [1.0, 0.0]
state(::SiteType{:SpinHalf}, ::StateName{:Dn}) = [0.0, 1.0]

# Unicode aliases
const _Sp_Up = StateName{Symbol("↑")}
const _Sp_Dn = StateName{Symbol("↓")}
state(st::SiteType{:SpinHalf}, ::_Sp_Up) = state(st, StateName(:Up))
state(st::SiteType{:SpinHalf}, ::_Sp_Dn) = state(st, StateName(:Dn))

# Bit-label aliases
state(st::SiteType{:SpinHalf}, ::StateName{Symbol("0")}) = state(st, StateName(:Up))
state(st::SiteType{:SpinHalf}, ::StateName{Symbol("1")}) = state(st, StateName(:Dn))

state(::SiteType{:SpinHalf}, ::StateName{:Coherent}; θ::Real, ϕ::Real=0.0) =
    ComplexF64[cos(θ/2), exp(im*ϕ)*sin(θ/2)]

op(::SiteType{:SpinHalf}, ::OpName{:Id})  = Float64[1   0;   0   1 ]
op(::SiteType{:SpinHalf}, ::OpName{:Sz})  = Float64[0.5 0;   0  -0.5]
op(::SiteType{:SpinHalf}, ::OpName{:Sx})  = Float64[0   0.5; 0.5  0 ]
op(::SiteType{:SpinHalf}, ::OpName{:Sy})  = ComplexF64[0  -0.5im; 0.5im  0]
op(::SiteType{:SpinHalf}, ::OpName{:S2})  = Float64[0.75 0; 0  0.75]

# S+, S- need Symbol() since +/- are not valid identifier characters
const _OpSp = OpName{Symbol("S+")}
const _OpSm = OpName{Symbol("S-")}
op(::SiteType{:SpinHalf}, ::_OpSp) = Float64[0  1;  0  0]
op(::SiteType{:SpinHalf}, ::_OpSm) = Float64[0  0;  1  0]

# Projectors — unicode and ASCII aliases both defined
op(::SiteType{:SpinHalf}, ::OpName{:ProjUp}) = Float64[1  0;  0  0]
op(::SiteType{:SpinHalf}, ::OpName{:ProjDn}) = Float64[0  0;  0  1]
const _OpProjUp = OpName{Symbol("Proj↑")}
const _OpProjDn = OpName{Symbol("Proj↓")}
op(st::SiteType{:SpinHalf}, ::_OpProjUp) = op(st, OpName(:ProjUp))
op(st::SiteType{:SpinHalf}, ::_OpProjDn) = op(st, OpName(:ProjDn))


"""
    op(::SiteType{:SpinHalf}, ::OpName{:Rx}; θ::Real)

Rotation by `θ` around x: exp(-iθSx) = cos(θ/2)𝟙 - i sin(θ/2)σx.
"""
op(::SiteType{:SpinHalf}, ::OpName{:Rx}; θ::Real) =
    ComplexF64[cos(θ/2)      -im*sin(θ/2);
               -im*sin(θ/2)   cos(θ/2)   ]

"""
    op(::SiteType{:SpinHalf}, ::OpName{:Ry}; θ::Real)

Rotation by `θ` around y: exp(-iθSy) = cos(θ/2)𝟙 - i sin(θ/2)σy.
"""
op(::SiteType{:SpinHalf}, ::OpName{:Ry}; θ::Real) =
    Float64[ cos(θ/2)  -sin(θ/2);
                sin(θ/2)   cos(θ/2)]

"""
    op(::SiteType{:SpinHalf}, ::OpName{:Rz}; θ::Real)

Rotation by `θ` around z: exp(-iθSz) = diag(e^{-iθ/2}, e^{iθ/2}).
"""
op(::SiteType{:SpinHalf}, ::OpName{:Rz}; θ::Real) =
    ComplexF64[exp(-im*θ/2)  0;
               0              exp(im*θ/2)]
