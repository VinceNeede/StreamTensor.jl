
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

linkdim(ψ::MPS, i::Int) = ψ[i].right.dim

function Base.conj(ψ::MPS{T}) where {T}
    tensors = [MPSTensor(conj(t.storage), t.left, t.site, t.right) 
               for t in ψ.tensors]
    return MPS(tensors, ψ.llim, ψ.rlim)
end

"""
    prime(ψ::MPS) -> MPS
Returns a view of the same storage with the primed version of the site indices
"""
function prime(ψ::MPS)
    tensors = [MPSTensor(t.storage, t.left, t.site', t.right)
               for t in ψ.tensors]
    return MPS(tensors, ψ.llim, ψ.rlim)
end

"""
    noprime(ψ::MPS) -> MPS
Returns a view of the same storage with the unprimed version of the site indices
"""
function noprime(ψ::MPS)
    tensors = [MPSTensor(t.storage, t.left, noprime(t.site), t.right)
               for t in ψ.tensors]
    return MPS(tensors, ψ.llim, ψ.rlim)
end

# dag = conj + prime
dag(ψ::MPS) = prime(conj(ψ))

# adjoint = dag
Base.adjoint(ψ::MPS) = dag(ψ)

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
    ψ = sim_linkinds(conj(ψ))

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