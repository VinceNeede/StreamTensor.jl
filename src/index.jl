# ── Index counter ────────────────────────────────────────────────────────────

const _INDEX_COUNTER = Threads.Atomic{UInt64}(0)

# ── Index ────────────────────────────────────────────────────────────────────

"""
    Index(dim, tags...)

A labeled tensor leg.  `tags` is an unordered, deduplicated set of `Symbol`s
stored as a sorted `NTuple` so that equality is order-independent.

    Index(2, :Site, :SpinHalf)
    Index(4, :Link)
"""
struct Index{N,ST<:Union{Nothing,SiteType}}
    id       :: UInt64
    dim      :: Int
    tags     :: NTuple{N, Symbol}   # unchanged — generic categorization (:Site, :Link, display)
    primed   :: Bool
end

# ── Internal tag normalisation ───────────────────────────────────────────────

function _make_tags(tags)
    syms = unique(sort(collect(Symbol, tags)))   # sort + dedup
    return (syms...,)                             # NTuple
end

# ── Constructors ─────────────────────────────────────────────────────────────

function Index(dim::Int, tags...; sitetype::Type{T}=Nothing) where {T<:Union{SiteType,Nothing}}
    @assert dim > 0 "Index dimension must be positive, got $dim"
    id = Threads.atomic_add!(_INDEX_COUNTER, UInt64(1))
    t  = _make_tags(tags)
    Index{length(t),sitetype}(id, dim, t, false)
end

sitetype(::Index{N,Nothing}) where {N}          = nothing
sitetype(::Index{N,ST}) where {N,ST<:SiteType}  = ST()

# ── Priming ──────────────────────────────────────────────────────────────────

Base.adjoint(i::Index{N,ST}) where {N,ST} =
    Index{N,ST}(i.id, i.dim, i.tags, !i.primed)

noprime(i::Index{N}) where {N} = Index{N}(i.id, i.dim, i.tags, false)
prime(i::Index)                 = i'
isprime(i::Index)               = i.primed

# ── Equality & hashing ───────────────────────────────────────────────────────

Base.:(==)(a::Index, b::Index) = a.id == b.id && a.primed == b.primed
Base.hash(a::Index, h::UInt)   = hash((a.id, a.primed), h)

# ── Tag helpers ──────────────────────────────────────────────────────────────

"""
    hastag(i::Index, tag::Symbol) -> Bool

Return `true` if `tag` is present in `i`'s tag set.
"""
hastag(i::Index, tag::Symbol) = tag ∈ i.tags

"""
    addtags(i::Index, tags...) -> Index

Return a new `Index` with `tags` added (same id, dim, prime level).
"""
function addtags(i::Index, tags...)
    new_tags = _make_tags((i.tags..., tags...))
    Index{length(new_tags)}(i.id, i.dim, new_tags, i.primed)
end

"""
    removetags(i::Index, tags...) -> Index

Return a new `Index` with `tags` removed (same id, dim, prime level).
"""
function removetags(i::Index, tags...)
    to_remove = Set{Symbol}(tags)
    kept      = filter(t -> t ∉ to_remove, collect(i.tags))
    new_tags  = _make_tags(kept)
    Index{length(new_tags)}(i.id, i.dim, new_tags, i.primed)
end

"""
    tags(i::Index) -> NTuple

Return the (sorted) tag tuple of `i`.
"""
tags(i::Index) = i.tags

# ── Display ──────────────────────────────────────────────────────────────────

function Base.show(io::IO, i::Index)
    tag_str = join(i.tags, ",")
    print(io, "Index($(i.dim), \"$(tag_str)\"$(i.primed ? "'" : ""), id=$(i.id))")
end

# ── Dimension helpers ────────────────────────────────────────────────────────

dim(i::Index)   = i.dim
dims(inds)      = ntuple(k -> inds[k].dim, length(inds))