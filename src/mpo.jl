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