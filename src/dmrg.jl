const _DEFAULT_EIGSOLVE_KWARGS = (krylovdim=6, maxiter=5)

mutable struct ProjMPO{T,N}
    H::MPO
    env::Vector{T}  # Env[i] = left-environment "through site i", for i <= lpos
                    #         right-environment "through site i", for i >= rpos
    lpos::Int       # rightmost site index whose LEFT environment is cached/valid
    rpos::Int       # leftmost site index whose RIGHT environment is cached/valid
    function ProjMPO{T}(H::MPO, env::Vector{T}, nsite::Int, lpos::Int, rpos::Int) where {T}
        nsite ∉ (1,2) && error("nsite must be 1 or 2")
        return new{T,nsite}(H, env, lpos, rpos)
    end
end
nsite(::ProjMPO{T,N}) where {T,N} = N
Base.length(P::ProjMPO) = length(P.H)

function ProjMPO(H::MPO{T}, nsite::Int=2) where {T}
    L = length(H)
    ET = DenseTensor{T, 3, Array{T, 3}}
    return ProjMPO{ET}(H, Vector{ET}(undef, L), nsite, 0, L+1)
end

function range(P::ProjMPO{T,1}, direction::SVDDirection) where {T} 
    L = length(P)
    return direction == LeftOrthogonal ? (1:(L-1)) : (L:-1:2)
end

function range(P::ProjMPO{T,2}, direction::SVDDirection) where {T} 
    L = length(P)
    return direction == LeftOrthogonal ? (1:(L-1)) : ((L-1):-1:1)
end

function _drop_trivial_left_link(t::MPSTensor)
    return DenseTensor((t.site, t.right), dropdims(t.storage, dims=1))
end

function _drop_trivial_left_link(t::MPOTensor)
    return DenseTensor((t.site_in, t.site_out, t.right), dropdims(t.storage, dims=1))
end

function _drop_trivial_right_link(t::MPSTensor)
    return DenseTensor((t.left, t.site), dropdims(t.storage, dims=3))
end

function _drop_trivial_right_link(t::MPOTensor)
    return DenseTensor((t.left, t.site_in, t.site_out), dropdims(t.storage, dims=4))
end

function _extend_env(::Nothing, ψi::MPSTensor, Hi::MPOTensor, edge::Symbol)
    ψi2, Hi2 = edge === :left ?
        (_drop_trivial_left_link(ψi),  _drop_trivial_left_link(Hi)) :
        (_drop_trivial_right_link(ψi), _drop_trivial_right_link(Hi))
    return ψi2 * Hi2 * dag(ψi2)
end

function _extend_env(E::DenseTensor{T, 3}, ψi::MPSTensor, Hi::MPOTensor) where {T}
    return E * ψi * Hi * dag(ψi)
end

function position!(P::ProjMPO, ψ::MPS, pos::Int)
    L = length(ψ)
    lpos_target = pos - 1
    rpos_target = pos + nsite(P)


    for i in (P.lpos+1):lpos_target
        P.env[i] = i == 1 ?
            _extend_env(nothing, ψ[i], P.H[i], :left) :
            _extend_env(P.env[i-1], ψ[i], P.H[i])
    end
    P.lpos = lpos_target
    for i in (P.rpos-1):-1:rpos_target
        P.env[i] = i == L ?
            _extend_env(nothing, ψ[i], P.H[i], :right) :
            _extend_env(P.env[i+1], ψ[i], P.H[i])
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
    local Hi
    for i in pos:last
        if i == 1
            Hi = _drop_trivial_left_link(P.H[i])
        elseif i == L
            Hi = _drop_trivial_right_link(P.H[i])
        else
            Hi = P.H[i]
        end
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

function _update_local_tensors!(P::ProjMPO{T,1}, ψ::MPS, ϕ::MPSTensor, pos::Int, direction::SVDDirection;
                                 maxdim=nothing, cutoff=nothing, noise=nothing) where {T}
    Hi = P.H[pos]
    χl, χr, d, wl, wr = dim.((ϕ.left, ϕ.right, ϕ.site, Hi.left, Hi.right))
    M = ϕ.storage
    local truncerr
    if direction == LeftOrthogonal
        M = reshape(M, χl * d, χr)
        if !isnothing(noise)
            perturbation = P.lpos == 0 ?
                ϕ * _drop_trivial_left_link(Hi) :
                P.env[P.lpos] * ϕ * Hi
            @assert noprime.(inds(perturbation)) == (ϕ.left, ϕ.right, ϕ.site, Hi.right)
            perturbation = noise * perturbation.storage
            perturbation = reshape(permutedims(perturbation, (1, 3, 2, 4)), χl * d, χr * wr)
            M = reshape(M, χl * d, χr)
            M = hcat(M, perturbation)
        end
        U_mat, s, Vt_mat = _mat_svd(M)
        U_mat, s, Vt_mat, truncerr = _truncate(s, U_mat, Vt_mat; maxdim, cutoff)
        χ = length(s)
        bond = Index(χ, :Link)
        ψ.tensors[pos] = MPSTensor(reshape(U_mat, χl, d, χ), ϕ.left, ϕ.site, bond)
        R = (Diagonal(s) * Vt_mat)[:, 1:χr]      # (χ, χr(1+wr)) -> (χ, χr); χ may exceed χr
        R = DenseTensor((bond, ϕ.right), R)
        ψ.tensors[pos+1] = _to_mpstensor(R * ψ[pos+1])
        
        ψ.llim, ψ.rlim = pos, pos + 2
    else
        M = reshape(M, χl, d * χr)
        if !isnothing(noise)
            perturbation = P.rpos == length(ψ)+1 ?
                    _drop_trivial_right_link(Hi) * ϕ :
                    Hi * (ϕ * P.env[P.rpos])
            @assert noprime.(inds(perturbation)) == (Hi.left, ϕ.site, ϕ.left, ϕ.right)
            perturbation = noise * perturbation.storage
            perturbation = reshape(permutedims(perturbation, (1, 3, 2, 4)), wl * χl, d * χr)
            M = reshape(M, χl, d * χr)
            M = vcat(M, perturbation)
        end
        U_mat, s, Vt_mat = _mat_svd(M)
        U_mat, s, Vt_mat, truncerr = _truncate(s, U_mat, Vt_mat; maxdim, cutoff)
        χ = length(s)
        bond = Index(χ, :Link)
        L = (U_mat * Diagonal(s))[1:χl, :]       # (χl(1+wl), χ) -> (χl, χ); χ may exceed χl
        L = DenseTensor((ϕ.left, bond), L)
        ψ.tensors[pos-1] = _to_mpstensor(ψ[pos-1] * L)
        ψ.tensors[pos] = MPSTensor(reshape(Vt_mat, χ, d, χr), bond, ϕ.site, ϕ.right)
        ψ.llim, ψ.rlim = pos - 2, pos
    end
    return truncerr
end

function _update_local_tensors!(::ProjMPO{T,2}, ψ::MPS, ϕ::DenseTensor, pos::Int, direction::SVDDirection;
                                 maxdim=nothing, cutoff=nothing, noise=nothing) where {T}
    left_inds = (ψ[pos].left, ψ[pos].site)
    U, S, V, truncerr = svd(ϕ, left_inds; maxdim, cutoff)
    if direction == LeftOrthogonal
        ψ.tensors[pos]   = _to_mpstensor(U)
        ψ.tensors[pos+1] = _to_mpstensor(S * V)
        ψ.llim, ψ.rlim = pos, pos + 2
    else
        ψ.tensors[pos]   = _to_mpstensor(U * S)
        ψ.tensors[pos+1] = _to_mpstensor(V)
        ψ.llim, ψ.rlim = pos - 1, pos + 1
    end
    return truncerr
end

_local_tensor(::ProjMPO{T,1}, ψ::MPS, pos::Int) where {T} = ψ[pos]
_local_tensor(::ProjMPO{T,2}, ψ::MPS, pos::Int) where {T} = ψ[pos] * ψ[pos+1]

function _align_eigenvector(ϕ0::MPSTensor, storage::AbstractVector)
    return MPSTensor(reshape(storage, size(ϕ0.storage)), ϕ0.left, ϕ0.site, ϕ0.right)
end

function _align_eigenvector(ϕ0::DenseTensor{T,4}, storage::AbstractVector) where {T}
    return DenseTensor(inds(ϕ0), reshape(storage, size(ϕ0.storage)))
end

_realtype(::Type{DenseTensor{T,N,S}}) where {T,N,S<:AbstractArray{T,N}} = real(T)
_realtype(::ProjMPO{Tens}) where {Tens} = _realtype(Tens)

_sweep_param(p, sw::Int) = p isa AbstractVector ? p[min(sw, length(p))] : p
_flip(d::SVDDirection) = d == LeftOrthogonal ? RightOrthogonal : LeftOrthogonal

function dmrg_sweep!(ψ::MPS, P::ProjMPO, direction::SVDDirection;
                      maxdim=nothing, cutoff=nothing, noise=nothing, eigsolve_kwargs=(;))
    rng = range(P, direction)
    energies, truncerrs, converged = Float64[], Float64[], Bool[]

    for pos in rng
        orthogonalize!(ψ, pos)
        position!(P, ψ, pos)
        ϕ0 = _local_tensor(P, ψ, pos)

        vals, vecs, info = eigsolve(_eigsolve_map(P, ϕ0), vec(ϕ0.storage), 1, :SR;
                                     ishermitian=true, verbosity=0, eigsolve_kwargs...)

        ϕ = _align_eigenvector(ϕ0, vecs[1])
        truncerr = _update_local_tensors!(P, ψ, ϕ, pos, direction; maxdim, cutoff, noise)

        push!(energies, real(vals[1]))
        push!(truncerrs, truncerr)
        push!(converged, info.converged >= 1)
    end
    return energies, truncerrs, converged
end

function dmrg!(ψ::MPS, P::ProjMPO, nsweeps::Int;
               maxdim=nothing, cutoff=nothing, noise=nothing, eigsolve_kwargs=(;),
               start_direction::SVDDirection=LeftOrthogonal,
               H_scale::Real=1.0, tol_power::Real=0.5)
    F = _realtype(P)
    tol, E_scale = H_scale * eps(F)^0.25, H_scale
    sweep_data = NamedTuple{(:energies,:truncerrs,:converged),
                             Tuple{Vector{Float64},Vector{Float64},Vector{Bool}}}[]

    for sw in 0:(nsweeps-1)
        direction = iseven(sw) ? start_direction : _flip(start_direction)
        kw = merge((tol=tol,), _DEFAULT_EIGSOLVE_KWARGS, _sweep_param(eigsolve_kwargs, sw+1))
        
        t0 = time()
        energies, truncerrs, converged = dmrg_sweep!(ψ, P, direction;
                        maxdim=_sweep_param(maxdim, sw+1), cutoff=_sweep_param(cutoff, sw+1),
                        noise=_sweep_param(noise, sw+1),
                        eigsolve_kwargs=kw)
        elapsed = time() - t0
        push!(sweep_data, (energies=energies, truncerrs=truncerrs, converged=converged))

        E_scale = maximum(abs, energies)
        next_tol = clamp(E_scale * maximum(truncerrs)^tol_power, E_scale*sqrt(eps(F)), tol)

        @debug "dmrg sweep $(sw+1)/$nsweeps" direction time=elapsed energy=energies[end] max_truncerr=maximum(truncerrs) n_unconverged=count(!, converged) tol=tol next_tol=next_tol maxdim=_sweep_param(maxdim,sw+1) maxlinkdim=maxlinkdim(ψ)

        tol = next_tol
    end
    return ψ, P, sweep_data
end

dmrg!(ψ::MPS, H::MPO, nsweeps::Int; nsite::Int=1, kwargs...) =
    dmrg!(ψ, ProjMPO(H, nsite), nsweeps; kwargs...)