mutable struct ProjMPO{T}
    H::MPO
    env::Vector{T}  # Env[i] = left-environment "through site i", for i <= lpos
                    #         right-environment "through site i", for i >= rpos
    nsite::Int      # 1 or 2
    lpos::Int       # rightmost site index whose LEFT environment is cached/valid
    rpos::Int       # leftmost site index whose RIGHT environment is cached/valid
end

function ProjMPO(H::MPO{T}, nsite::Int=2) where {T}
    L = length(H)
    return ProjMPO{DenseTensor{T, 3}}(H, Vector{DenseTensor{T, 3}}(undef, L), nsite, 0, L+1)
end

function _extend_env(::Nothing, ψi::MPSTensor, Hi::MPOTensor)
    ψi2 = _drop_trivial_dims(ψi)
    Hi2 = _drop_trivial_dims(Hi)
    return ψi2 * Hi2 * dag(ψi2)
end

function _extend_env(E::DenseTensor{T, 3}, ψi::MPSTensor, Hi::MPOTensor) where {T}
    return E * ψi * Hi * dag(ψi)
end

function position!(P::ProjMPO, ψ::MPS, pos::Int)
    L = length(ψ)
    lpos_target = pos - 1
    rpos_target = pos + P.nsite

    for i in (P.lpos+1):lpos_target
        prev = i == 1 ? nothing : P.env[i-1]
        P.env[i] = _extend_env(prev, ψ[i], P.H[i])
    end
    P.lpos = lpos_target

    for i in (P.rpos-1):-1:rpos_target
        nxt = i == L ? nothing : P.env[i+1]
        P.env[i] = _extend_env(nxt, ψ[i], P.H[i])
    end
    P.rpos = rpos_target

    return P
end

function _align(t::DenseTensor, ref_inds::NTuple{N,Index}) where N
    perm = ntuple(k -> findfirst(==(ref_inds[k]), inds(t)), N)
    return DenseTensor(ref_inds, _maybe_permute(t.storage, perm))
end

function product(P::ProjMPO, ϕ::AbstractTensor)
    L  = length(P.H)
    pos, last = P.lpos + 1, P.rpos - 1

    Hϕ = ϕ
    P.lpos != 0 && (Hϕ = P.env[P.lpos] * Hϕ)
    for i in pos:last
        Hi = (i == 1 || i == L) ? _drop_trivial_dims(P.H[i]) : P.H[i]
        Hϕ = Hϕ * Hi
    end
    P.rpos != L+1 && (Hϕ = Hϕ * P.env[P.rpos])

    return  _align(noprime(Hϕ), inds(ϕ))
end


"""
    _eigsolve_map(P::ProjMPO, ϕ0::AbstractTensor) -> Function
Returns a function that takes in input a vector `v`, 
reshape it to match `ϕ0` and convert to `DenseTensor`,
then send it to `product`, and then convert the result 
back to a vector. 
"""
function _eigsolve_map(P::ProjMPO, ϕ0::AbstractTensor)
    ix    = inds(ϕ0)
    shape = size(ϕ0.storage)
    return v -> vec(product(P, DenseTensor(ix, reshape(v, shape))).storage)
end