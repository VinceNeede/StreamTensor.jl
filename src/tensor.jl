"""
    AbstractTensor{T, N}
Abstract type for tensors with `N` legs and eltype `T`.
"""
abstract type AbstractTensor{T, N} end

"""
    DenseTensor{T, N, S <: AbstractArray{T,N}}
Concrete type for dense tensors.
"""
struct DenseTensor{T, N, S <: AbstractArray{T,N}} <: AbstractTensor{T, N}
    inds    :: NTuple{N, Index}
    storage :: S
    function DenseTensor(inds::NTuple{N, Index}, storage::S) where {T, N, S <: AbstractArray{T,N}}
        @assert allunique(inds) "DenseTensor has repeated indices: $inds"
        @assert size(storage) == map(i -> i.dim, inds) "Storage shape $(size(storage)) doesn't match index dims $(map(i -> i.dim, inds))"
        new{T, N, S}(inds, storage)
    end
end

"""
    DiagTensor{T, N, S <: AbstractVector{T}}
Concrete type for diagonal tensors.
"""
struct DiagTensor{T, N, S <: AbstractVector{T}} <: AbstractTensor{T, N}
    inds    :: NTuple{N, Index}
    storage :: S
    function DiagTensor(inds::NTuple{N, Index}, storage::S) where {T, N, S <: AbstractVector{T}}
        @assert allunique(inds) "DiagTensor has repeated indices: $inds"
        @assert allequal(i.dim for i in inds) "DiagTensor requires all indices to have equal dimension, got $(map(i -> i.dim, inds))"
        @assert length(storage) == first(inds).dim "Storage length $(length(storage)) doesn't match index dim $(first(inds).dim)"
        new{T, N, S}(inds, storage)
    end
end

"""
    DeltaTensor(inds::NTuple{N, Index})
Helper constructor for a `DiagTensor` with only ones on the diagonal.
"""
function DeltaTensor(inds::NTuple{N, Index}) where {N}
    return DiagTensor(inds, ones(first(inds).dim))
end

"""
    MPSTensor{T, A}

A rank-3 tensor representing one site of an MPS.  The storage axes follow the
fixed convention `(χ_left, d, χ_right)` so that reshape / BLAS operations
never need a permutation for the common left- and right-orthogonalization
sweeps.

Fields: `storage`, `left`, `site`, `right`.
"""
struct MPSTensor{T, A <: AbstractArray{T,3}} <: AbstractTensor{T, 3}
    storage :: A                  # (χ_left, d, χ_right) — fixed convention
    left    :: Index              # always leg 1
    site    :: Index              # always leg 2  
    right   :: Index              # always leg 3
    function MPSTensor(storage::A, left::Index, site::Index, right::Index) where {T, A <: AbstractArray{T,3}}
        @assert allunique((left, site, right)) "MPSTensor has repeated indices: $((left, site, right))"
        @assert size(storage) == (left.dim, site.dim, right.dim) "Storage shape $(size(storage)) doesn't match index dims $((left.dim, site.dim, right.dim))"
        new{T, A}(storage, left, site, right)
    end
end

"""
    MPOTensor{T, A <: AbstractArray{T,4}}
Concrete type of a tensor with 4 ordered legs. The canonical order is:
    left link - site_in - site_out - right link
"""
struct MPOTensor{T, A <: AbstractArray{T,4}} <: AbstractTensor{T, 4}
    storage  :: A                  # (χ_left, d_in, d_out, χ_right)
    left     :: Index
    site_in  :: Index
    site_out :: Index
    right    :: Index
    function MPOTensor(storage::A, left::Index, site_in::Index, site_out::Index, right::Index) where {T, A <: AbstractArray{T,4}}
        @assert allunique((left, site_in, site_out, right)) "MPOTensor has repeated indices: $((left, site_in, site_out, right))"
        @assert size(storage) == (left.dim, site_in.dim, site_out.dim, right.dim) "Storage shape $(size(storage)) doesn't match index dims $((left.dim, right.dim, site_in.dim, site_out.dim))"
        new{T, A}(storage, left, site_in, site_out, right)
    end
end

"""
    inds(t::AbstractTensor) -> NTuple{N, Index}
Return the indices of `t`.
"""
inds(t::AbstractTensor) = t.inds
inds(t::MPSTensor) = (t.left, t.site, t.right)
inds(t::MPOTensor) = (t.left, t.site_in, t.site_out, t.right)
"""
    siteind(t::MPSTensor) -> Index
Return the site index of `t`.
"""
siteind(t::MPSTensor) = t.site
"""
    siteinds(t::MPSTensor) -> NTuple{2, Index}
Return the site indices of `t`.
"""
siteinds(t::MPOTensor) = (t.site_in, t.site_out)
"""
    linkinds(t::MPSTensor) -> NTuple{2, Index}
Return the link indices of `t`.
"""
linkinds(t::Union{MPSTensor, MPOTensor}) = (t.left, t.right)

"""
    Base.ndims(t::AbstractTensor) -> Int
Return the number of legs of `t`.
"""
Base.ndims(::AbstractTensor{T,N}) where {T,N} = N
"""
    Base.eltype(t::AbstractTensor) -> T
Return the element type of `t`.
"""
Base.eltype(::AbstractTensor{T,N}) where {T,N} = T

Base.size(t::AbstractTensor) = dims(inds(t))
Base.size(t::AbstractTensor, n::Int) = dim(inds(t)[n])

"""
    to_dense(t::AbstractTensor) -> DenseTensor
Utility function to convert `t` to a `DenseTensor`.
"""
function to_dense(t::AbstractTensor)
    DenseTensor(inds(t), t.storage)
end

function to_dense(D::DiagTensor{T, N}) where {T, N}
    out_size = size(D)
    storage = fill(zero(T), out_size...)
    n = first(inds(D)).dim
    
    for k in 1:n
        idx = ntuple(_ -> k, N)
        storage[idx...] = D.storage[k]
    end
    return DenseTensor(inds(D), storage)
end

"""
    _to_mpstensor(t::DenseTensor{T, 3}) -> MPSTensor

Internal helper: reinterpret a rank-3 `DenseTensor` whose indices are
`(left, site, right)` as an `MPSTensor` without copying storage.
"""
function _to_mpstensor(t::DenseTensor{T, 3}) where {T}
    l, s, r = inds(t)
    MPSTensor(t.storage, l, s, r)
end

"""
    Base.conj(t::AbstractTensor) -> AbstractTensor
Return a tensor with same indices as `t` but with the conjugated storage.
"""
function Base.conj(t::DenseTensor{T}) where {T}
    return DenseTensor(inds(t), conj(t.storage))
end

function Base.conj(t::DiagTensor)
    return DiagTensor(inds(t), conj(t.storage))
end

function Base.conj(t::MPSTensor{T}) where {T}
    return MPSTensor(conj(t.storage), t.left, t.site, t.right)
end

function Base.conj(t::MPOTensor{T}) where {T}
    return MPOTensor(conj(t.storage), t.left, t.site_in, t.site_out, t.right)
end

"""
    prime(t::AbstractTensor) -> AbstractTensor
Return a tensor with same storage as `t` but with the primed indices. The storage
is a view of the original tensor.
"""
prime(t::MPSTensor) = MPSTensor(t.storage, t.left', t.site', t.right')
prime(t::MPOTensor) = MPOTensor(t.storage, t.left', t.site_in', t.site_out', t.right')
prime(t::DenseTensor) = DenseTensor(prime.(inds(t)), t.storage)
prime(t::DiagTensor) = DiagTensor(prime.(inds(t)), t.storage)
prime(t::AbstractTensor) = prime(to_dense(t))

"""
    noprime(t::AbstractTensor) -> AbstractTensor
Return a tensor with same storage as `t` but with the unprimed indices. The storage
is a view of the original tensor.
"""
noprime(t::MPSTensor) = MPSTensor(t.storage, noprime.(inds(t))...)
noprime(t::MPOTensor) = MPOTensor(t.storage, noprime.(inds(t))...)
noprime(t::DenseTensor) = DenseTensor(noprime.(inds(t)), t.storage)
noprime(t::DiagTensor) = DiagTensor(noprime.(inds(t)), t.storage)
noprime(t::AbstractTensor) = noprime(to_dense(t))

"""
    dag(t::AbstractTensor) -> AbstractTensor

Return the "dagger" of `t`: conjugated storage with all indices primed.
Equivalent to `prime(conj(t))`.
"""
dag(t::AbstractTensor) = prime(conj(t))

function _drop_trivial_dims(t::AbstractTensor)
    td = to_dense(t)
    inds_td = inds(td)
    new_inds = filter(i -> dim(i) != 1, inds_td)
    drop_dims = Tuple(findall(==(1), size(td.storage)))
    new_storage = dropdims(td.storage; dims = drop_dims)
    return DenseTensor(new_inds, new_storage)
end

"""
    Combiner <: AbstractTensor{Bool, N}

A bookkeeping pseudo-tensor that fuses/splits a contiguous block of indices.
`legs[1]` is the combined index; `legs[2:end]` are the original indices it
replaces, in order. Carries no real storage — contracting with a `DenseTensor`
performs a reshape, not arithmetic.

Construct via `combine(inds_to_combine...)`, which generates the combined
`Index` and returns it alongside the `Combiner`. Apply with `t * c` (fuse)
or `t * dag(c)` (expand) on a `DenseTensor`. `dag(c::Combiner)` flips its
role from fusing to expanding.
"""
struct Combiner{N} <: AbstractTensor{Bool, N}
    legs      :: NTuple{N, Index}   # legs[1] = combined, legs[2:end] = original
    expanding :: Bool               # false: combine direction; true: dag'd, expand direction
end

Combiner(new_index::Index, original::Index...) =
    Combiner((new_index, original...), false)

"""
    dag(c::Combiner) -> Combiner

Flip a `Combiner` from fusing direction to expanding direction (or back).
Contracting a tensor against `dag(c)` replaces the combined index with the
original indices it was fused from.
"""
dag(c::Combiner) = Combiner(c.legs, !c.expanding)

inds(c::Combiner) = c.legs

"""
    combine(inds_to_combine::Index...; kwargs...) -> (Combiner, Index)

Create a new `Index` of dimension `prod(dim, inds_to_combine)` and return a
`Combiner` that fuses `inds_to_combine` into it (or, via `dag`, expands it
back out). `kwargs` are forwarded to the `Index` constructor for the new
combined index (e.g. `tags`).

Does not touch any tensor — apply the result via `t * c` (fuse) or
`t * dag(c)` (expand) on a `DenseTensor` whose `inds` contain
`inds_to_combine` as a contiguous block, in the order given.
"""
function combine(inds_to_combine::Index...; kwargs...)
    n = length(inds_to_combine)
    @assert n >= 1 "combine needs at least one index"
    combined_dim = prod(dim, inds_to_combine)
    combined = Index(combined_dim; kwargs...)
    return Combiner(combined, inds_to_combine...), combined
end