"""
    MPO{T, A}

A Matrix Product Operator: a chain of `MPOTensor`s sharing bond indices.

## Constructors

    MPO(tensors, llim, rlim)   # raw; validates bond connectivity
    MPO(tensors)               # unorthogonalized (llim=0, rlim=L+1)
    MPO(tensors, center)       # orthogonality center at `center`
    MPO(opsum::OpSum, sites)   # build from an operator sum via finite-state-machine construction
"""
mutable struct MPO{T, A <: AbstractArray{T,4}} <: AbstractTensorTrain{T}
    tensors :: Vector{MPOTensor{T,A}}
    llim    :: Int
    rlim    :: Int

    function MPO(tensors::Vector{MPOTensor{T,A}}, llim::Int, rlim::Int) where {T, A}
        _validate_tensor_train(tensors, llim, rlim)
        new{T,A}(tensors, llim, rlim)
    end
end

MPO(tensors::Vector{<:MPOTensor{T}}) where {T} = 
    MPO(tensors, 0, length(tensors) + 1)

MPO(tensors::Vector{<:MPOTensor{T}}, center::Int) where {T} = 
    MPO(tensors, center - 1, center + 1)

Base.copy(H::MPO) = MPO(copy(H.tensors), H.llim, H.rlim)
Base.deepcopy(H::MPO) = MPO(deepcopy(H.tensors), H.llim, H.rlim)


"""
    expect(ψ::MPS, O::AbstractMatrix, i::Int) -> Real

Compute the single-site expectation value `⟨ψ|O_i|ψ⟩` where `O` is a
`d×d` matrix acting on site `i`.  `ψ` is first orthogonalized to site `i`.
"""
function expect(ψ::MPS{T}, O::AbstractMatrix, i::Int) where {T}
    s     = siteinds(ψ)[i]
    s_out = s'
    @assert size(O, 1) == s.dim "Operator dimension $(size(O,1)) doesn't match site dimension $(s.dim)"
    @assert size(O, 2) == s.dim "Operator dimension $(size(O,2)) doesn't match site dimension $(s.dim)"
    l = Index(1, :Link)
    r = Index(1, :Link)
    O_tensor = MPOTensor(reshape(O, 1, s.dim, s.dim, 1), l, s_out, s, r)
    return expect(ψ, O_tensor; i_site=i)
end

function expect(ψ::MPS{T}, O::MPOTensor{T}; i_site=nothing) where {T}
    sites  = siteinds(ψ)
    if isnothing(i_site)
        i_site = findfirst(==(O.site_in), sites)
    else
        @assert O.site_in == sites[i_site] "Operator site_in mismatch"
    end
    @assert !isnothing(i_site) "Operator site_in not found in MPS"
    @assert O.left.dim == 1 && O.right.dim == 1 "Single-site operator must have bond dim 1"

    ψ_ortho  = orthogonalize(ψ, i_site)
    ψ_i      = ψ_ortho[i_site]

    # contract O with ψ_i over site_in (s)
    # result indices: (l_O, site_out, r_O, left, right) dims (1,d,1,χl,χr)
    Oψ_i = contract(to_dense(O), to_dense(ψ_i))

    # bra: conj(ψ_i) with site index replaced by site_out (s')
    # so it contracts with Oψ_i over (site_out, left, right)
    # l_O and r_O (dim 1) will be left free → result is (1,1) → scalar after squeeze
    ψ_bra = DenseTensor(
        (ψ_i.left, O.site_out, ψ_i.right),
        conj(ψ_i.storage)
    )

    # contract bra with Oψ_i
    # contracts: left, site_out, right
    # free: l_O, r_O (both dim 1)
    result = contract(ψ_bra, Oψ_i)

    # result has shape (1,1) from trivial MPO bonds — squeeze to scalar
    return real(result.storage[])
end

"""
    expect(ψ::MPS, O::AbstractMatrix; sites=1:length(ψ)) -> Vector{Real}

Compute single-site expectation values of operator matrix `O` over the
given `sites`, returning a `Vector` of real scalars.
"""
function expect(ψ::MPS{T}, O::AbstractMatrix; sites::AbstractVector{Int} = 1:length(ψ)) where {T}
    return [expect(ψ, O, i) for i in sites]
end

"""
    inner(ψ::MPS, H::MPO, φ::MPS) -> Number

Compute the matrix element `⟨ψ|H|φ⟩` by contracting the bra, MPO, and ket
left-to-right in a single sweep.  `ψ` is automatically conjugated.

It is safe to call `inner(ψ, H, ψ)` (bra === ket): link indices of `ψ` are
refreshed via `sim_linkinds` before contraction to avoid index conflicts.

Both MPS must have the same length as the MPO, and their physical indices must
match `siteinds(H)`.
"""
function inner(ψ::MPS, H::MPO, φ::MPS)
    L = length(H)
    @assert length(ψ) == L && length(φ) == L "MPS/MPO length mismatch"

    ψ_sim = sim_linkinds(ψ')

    E = DenseTensor(
        (ψ_sim[1].left, H[1].left, φ[1].left),
        fill(one(eltype(ψ_sim[1].storage)), 1, 1, 1)
    )

    for i in 1:L
        E = E * ψ_sim[i] * H[i] * φ[i]
    end

    return E.storage[]
end

"""
    apply!(H::MPO, ψ::MPS, ::Val{:zipup};
           maxdim=nothing, cutoff=nothing,
           sweep_maxdim=2*maxdim, sweep_cutoff=cutoff/10) -> MPS

Compute `H|ψ⟩` in-place using the zip-up algorithm, overwriting `ψ`.

The left-to-right pass contracts `H[i] * ψ[i]` site by site, fusing the
left and right link indices via `Combiner` and factorizing via QR (or SVD
if `sweep_maxdim`/`sweep_cutoff` are provided). The right-to-left pass
compresses the result with `compress!` using `maxdim`/`cutoff`.

For best numerical accuracy, `H` should be in left-canonical form
(`orthogonalize!(H, 1)` called beforehand). A warning is issued if `H`
is not in canonical form.

Returns `ψ` (now representing `H|ψ⟩`) in left-canonical form
(orthogonality center at site 1).

# Arguments
- `maxdim`: maximum bond dimension for the right-to-left compression pass.
- `cutoff`: singular value cutoff for the right-to-left compression pass.
- `sweep_maxdim`: maximum bond dimension during the left-to-right pass
  (default: `2*maxdim`, following Paeckel et al. 2019 — a loose intermediate
  truncation, refined by the final compression pass. `nothing` if `maxdim`
  is `nothing` — no truncation anywhere, exact contract-then-compress).
- `sweep_cutoff`: singular value cutoff during the left-to-right pass
  (default: `cutoff/10`, same rationale; `nothing` if `cutoff` is `nothing`).

See also: Paeckel et al., *Time-evolution methods for matrix-product states*,
Ann. Phys. 411, 167998 (2019), [arXiv:1901.05824](https://arxiv.org/abs/1901.05824)
— section on the zip-up algorithm, for the rationale behind the default
`sweep_maxdim`/`sweep_cutoff` values.

# Example
```julia
sites = siteinds(:SpinHalf, 10)
H = MPO(opsum, sites)
orthogonalize!(H, 1)
ψ = random_mps(Float64, sites, 8)
apply!(H, ψ; maxdim=16, cutoff=1e-10)
```
"""
function apply!(H::MPO, ψ::MPS, ::Val{:zipup};
                maxdim=nothing, cutoff=nothing,
                sweep_maxdim = isnothing(maxdim) ? nothing : 2 * maxdim,
                sweep_cutoff = isnothing(cutoff) ? nothing : cutoff / 10)
    L = length(H)
    @assert length(ψ) == L "MPO/MPS length mismatch"
    if !isortho(H) || orthocenter(H) != 1
        @warn "apply!(H, ψ, Val(:zipup)): H is not left-canonicalized at site 1. " *
            "Call orthogonalize!(H, 1) before apply! for best numerical accuracy."
    end
    orthogonalize!(ψ, 1)
    T = promote_type(eltype(H[1].storage), eltype(ψ[1].storage))

    R = DenseTensor(
        (Index(1, :Link), H[1].left, ψ[1].left),
        ones(T, 1, 1, 1)
    )

    for i in 1:L
        θ = R * H[i] * ψ[i]
        c_right, _ = combine(H[i].right, ψ[i].right; tags=:Link)
        θ = noprime(_to_traintensor(θ * c_right))
        if i == L
            ψ.tensors[i] = θ
        else
            U, R = factorize(θ, LeftOrthogonal; maxdim=sweep_maxdim, cutoff=sweep_cutoff)
            ψ.tensors[i] = U
            R = R * dag(c_right)   # → (link_svd, H[i].right, ψ[i].right)
        end
    end

    ψ.llim = L - 1
    ψ.rlim = L + 1

    # ── right-to-left pass: compress ─────────────────────────────────────
    compress!(ψ, 1; maxdim, cutoff)

    return ψ
end

"""
    apply!(H::MPO, ψ::MPS; alg=:zipup, kwargs...) -> MPS

Dispatch to the algorithm selected by `alg`. Currently supported: `:zipup`.
See [`apply!(H, ψ, Val(:zipup))`](@ref) for keyword arguments.
"""
apply!(H::MPO, ψ::MPS; alg=:zipup, kwargs...) = apply!(H, ψ, Val(alg); kwargs...)

"""
    apply(H::MPO, ψ::MPS; kwargs...) -> MPS

Non-mutating version of [`apply!`](@ref): returns a new MPS representing
`H|ψ⟩` without modifying `ψ`. See `apply!` for keyword arguments.
"""
apply(H::MPO,  ψ::MPS; kwargs...)              = apply!(H, copy(ψ); kwargs...)