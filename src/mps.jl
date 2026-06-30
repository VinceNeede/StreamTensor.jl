
"""
    MPS{T, A}

A Matrix Product State: a chain of `MPSTensor`s sharing bond indices.

The orthogonality window `(llim, rlim)` tracks which sites are known to be
orthogonalized.  Use [`orthogonalize!`](@ref) / [`orthogonalize`](@ref) to
move the center, and [`leftlim`](@ref) / [`rightlim`](@ref) to query it.

## Constructors

    MPS(tensors, llim, rlim)           # raw constructor, validates bond connectivity
    MPS(tensors)                        # assumes unorthogonalized (llim=0, rlim=L+1)
    MPS(T, sites, labels)              # product state from site labels (strings or (name,params) tuples)
    MPS(sites, labels)                 # defaults to Float64
"""
mutable struct MPS{T, A <: AbstractArray{T,3}} <: AbstractTensorTrain{T}
    tensors :: Vector{MPSTensor{T,A}}
    llim    :: Int
    rlim    :: Int

    function MPS(tensors::Vector{<:MPSTensor{T, A}}, llim::Int, rlim::Int) where {T, A <: AbstractArray{T,3}}
        _validate_tensor_train(tensors, llim, rlim)
        new{T, A}(tensors, llim, rlim)
    end
end

# Constructors
MPS(tensors::Vector{<:MPSTensor{T}}) where {T} = 
    MPS(tensors, 0, length(tensors) + 1)

MPS(tensors::Vector{<:MPSTensor{T}}, center::Int) where {T} = 
    MPS(tensors, center - 1, center + 1)

function MPS(
    ::Type{T},
    sites::Vector{<:Index},
    labels::Vector{<:Tuple{AbstractString, NamedTuple}}
) where {T}
    L = length(sites)
    @assert length(labels) == L "Number of labels doesn't match number of sites"
    tensors = Vector{MPSTensor{T, Array{T,3}}}(undef, L)

    left = Index(1, :Link)
    for (i, (site, label)) in enumerate(zip(sites, labels))
        tensors[i] = state(T, site, label; left_link=left)
        left = tensors[i].right
    end
    return MPS(tensors, 0, L+1)
end

function MPS(
    ::Type{T},
    sites::Vector{<:Index},
    labels::Vector{<:AbstractString}
) where {T}
    labels_tuple = [(label, (;)) for label in labels]
    return MPS(T, sites, labels_tuple)
end

function MPS(
    sites::Vector{<:Index},
    labels::Vector{<:AbstractString}
)
    MPS(Float64, sites, labels)
end

function MPS(
    sites::Vector{<:Index},
    labels::Vector{<:Tuple{AbstractString, NamedTuple}}
)
    return MPS(Float64, sites, labels)   
end

Base.copy(ψ::MPS) = MPS(copy(ψ.tensors), ψ.llim, ψ.rlim)
Base.deepcopy(ψ::MPS) = MPS(deepcopy(ψ.tensors), ψ.llim, ψ.rlim)

"""
    siteinds(ψ::MPS) -> Vector{Index}

Return the physical (site) indices of `ψ` in site order.
"""
siteinds(ψ::MPS) = [t.site for t in ψ.tensors]

"""
    random_mps([T=Float64,] sites, linkdim) -> MPS

Construct a random MPS over `sites` with maximum bond dimension `linkdim`.
The state is built in mixed-canonical form with the orthogonality center at
`L÷2 + 1`: the left half is left-orthogonalized via QR, the right half is
right-orthogonalized via QR, and the center tensor is normalized.

`T` sets the element type (default `Float64`).
"""
function random_mps(::Type{T}, sites::Vector{<:Index}, linkdim::Int) where {T}
    L = length(sites)
    tensors = Vector{MPSTensor{T, Array{T, 3}}}(undef, L)
    mid = L ÷ 2

    # left half: left-orthogonal tensors 1..mid
    left = Index(1, :Link)
    for i in 1:mid
        d  = sites[i].dim
        χl = left.dim
        χr = min(linkdim, χl * d)
        right = Index(χr, :Link)

        Q, _ = qr(MPSTensor(randn(T, χl, d, χr), left, sites[i], right))

        tensors[i] = Q
        left = Q.right
    end

    # right half: right-orthogonal tensors L..mid+2
    right = Index(1, :Link)
    for i in L:-1:mid+2
        d  = sites[i].dim
        χr = right.dim
        χl = min(linkdim, χr * d)
        left_i = Index(χl, :Link)

        _, Q = qr(MPSTensor(randn(T, χl, d, χr), left_i, sites[i], right); direction=RightOrthogonal)
        tensors[i] = Q
        right = Q.left
    end

    # center tensor: random, connects left and right halves
    χl_c = tensors[mid].right.dim
    χr_c = tensors[mid+2].left.dim
    d_c  = sites[mid+1].dim
    center_storage = randn(T, χl_c, d_c, χr_c)
    center_storage ./= norm(center_storage)  # normalize
    tensors[mid+1] = MPSTensor(center_storage,
                                tensors[mid].right,
                                sites[mid+1],
                                tensors[mid+2].left)

    return MPS(tensors, mid, mid + 2)
end

function random_mps(sites::Vector{<:Index}, linkdim::Int)
    return random_mps(Float64, sites, linkdim)
end

function _needs_truncation(t::MPSTensor, maxdim, cutoff, direction::SVDDirection)
    isnothing(maxdim) && isnothing(cutoff) && return false
    !isnothing(cutoff) && return true
    χl, d, χr = size(t.storage)
    actual_dim = direction == LeftOrthogonal ? min(χl * d, χr) : min(χl, d * χr)
    return maxdim < actual_dim
end

function _shift_center_right!(mps::MPS, i::Int; maxdim=nothing, cutoff=nothing)
    if _needs_truncation(mps.tensors[i], maxdim, cutoff, LeftOrthogonal)
        U, S, V, _ = svd(mps.tensors[i]; direction=LeftOrthogonal, maxdim, cutoff)
        mps.tensors[i] = U
        mps.tensors[i+1] = _to_mpstensor(S * V * mps.tensors[i+1])
    else
        Q, R = qr(mps.tensors[i]; direction=LeftOrthogonal)
        mps.tensors[i] = Q
        mps.tensors[i+1] = _to_mpstensor(R * mps.tensors[i+1])
    end
end

function _shift_center_left!(mps::MPS, i::Int; maxdim=nothing, cutoff=nothing)
    if _needs_truncation(mps.tensors[i], maxdim, cutoff, RightOrthogonal)
        U, S, V, _ = svd(mps.tensors[i]; direction=RightOrthogonal, maxdim, cutoff)
        mps.tensors[i] = V
        mps.tensors[i-1] = _to_mpstensor(mps.tensors[i-1] * U * S)
    else
        L, Q = qr(mps.tensors[i]; direction=RightOrthogonal)
        mps.tensors[i] = Q
        mps.tensors[i-1] = _to_mpstensor(mps.tensors[i-1] * L)
    end
end

"""
    orthogonalize!(mps::MPS, center::Int) -> MPS

In-place: sweep left and right to bring the orthogonality center to site
`center` using QR decompositions.  Updates `mps.llim` and `mps.rlim`.
"""
function orthogonalize!(mps::MPS, center::Int)
    L = length(mps)
    @assert 1 <= center <= L "Center $center out of bounds for MPS of length $L"

    # left sweep: llim+1 up to center-1
    for i in mps.llim+1 : center-1
        _shift_center_right!(mps, i)
    end

    # right sweep: rlim-1 down to center+1
    for i in mps.rlim-1 : -1 : center+1
        _shift_center_left!(mps, i)
    end

    # update limits
    mps.llim = center - 1
    mps.rlim = center + 1

    return mps
end

function compress!(mps::MPS, center::Int; maxdim=nothing, cutoff=nothing)
    L = length(mps)
    @assert 1 <= center <= L "Center $center out of bounds for MPS of length $L"

    # left sweep: 1 up to center-1
    for i in 1 : center-1
        _shift_center_right!(mps, i, maxdim=maxdim, cutoff=cutoff)
    end

    # right sweep: L-1 down to center+1
    for i in L : -1 : center+1
        _shift_center_left!(mps, i, maxdim=maxdim, cutoff=cutoff)
    end

    # update limits
    mps.llim = center - 1
    mps.rlim = center + 1

    return mps
end

"""
    orthogonalize(mps::MPS, center::Int) -> MPS

Non-mutating version of [`orthogonalize!`](@ref): returns a copy of `mps`
with orthogonality center at site `center`.
"""
function orthogonalize(mps::MPS, center::Int)
    return orthogonalize!(copy(mps), center)
end

function compress(mps::MPS, center::Int; maxdim=nothing, cutoff=nothing)
    return compress!(copy(mps), center; maxdim=maxdim, cutoff=cutoff)
end

linkdim(ψ::MPS, i::Int) = ψ[i].right.dim

# dag: conjugate the MPS storage
function Base.conj(ψ::MPS{T}) where {T}
    tensors = [MPSTensor(conj(t.storage), t.left, t.site, t.right) 
               for t in ψ.tensors]
    return MPS(tensors, ψ.llim, ψ.rlim)
end

Base.adjoint(ψ::MPS) = conj(ψ) # to use ', even if no transpose is performed

# sim: replace link indices with fresh ones of same dimension
function sim_linkinds(ψ::MPS{T}) where {T}
    L = length(ψ)
    tensors = Vector{MPSTensor{T, Array{T,3}}}(undef, L)
    
    # create new link indices
    new_links = [Index(linkdim(ψ, i), :Link) for i in 1:L-1]
    
    # boundary indices
    left_bdry  = Index(1, :Link)
    right_bdry = Index(1, :Link)
    
    for i in 1:L
        left  = i == 1   ? left_bdry    : new_links[i-1]
        right = i == L   ? right_bdry   : new_links[i]
        tensors[i] = MPSTensor(ψ[i].storage, left, ψ[i].site, right)
    end
    
    return MPS(tensors, ψ.llim, ψ.rlim)
end

"""
    inner(ψ::MPS, φ::MPS) -> Number

Compute the inner product `⟨ψ|φ⟩` by contracting the two MPS left-to-right.
`ψ` is automatically conjugated.  Both MPS must have the same length and
matching site indices.
"""
function inner(ψ::MPS{T}, φ::MPS{T}) where {T}
    L = length(ψ)
    @assert length(φ) == L "MPS must have the same length"
    for i in 1:L
        @assert ψ[i].site == φ[i].site "Site index mismatch at site $i"
    end

    # sim the link indices of φ to avoid conflicts with ψ
    ψ = sim_linkinds(ψ')

    # initialize scalar environment
    E = DenseTensor(
        (ψ[1].left, φ[1].left),
        fill(one(T), 1, 1)
    )

    for i in 1:L
        Eψ = contract(E, to_dense(ψ[i]))
        E  = contract(Eψ, to_dense(φ[i]))
    end

    return E.storage[]
end