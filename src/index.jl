# ── Index counter ────────────────────────────────────────────────────────────

const _INDEX_COUNTER = Threads.Atomic{UInt64}(0)

# ── Tag capacity ─────────────────────────────────────────────────────────────

const MAX_TAGS = 4

# ── Index ────────────────────────────────────────────────────────────────────

"""
    Index(dim, tags...)

A labeled tensor leg.  `tags` is an unordered, deduplicated set of `Symbol`s
stored as a sorted, fixed-capacity `NTuple{MAX_TAGS,Symbol}` (padded with
`Symbol()`), so that `Index` is a single concrete type — equality is
order-independent on the active tags.

    Index(2, :Site, :SpinHalf)
    Index(4, :Link)
"""
struct Index
    id       :: UInt64
    dim      :: Int
    tags     :: NTuple{MAX_TAGS, Symbol}   # sorted, padded with Symbol()
    ntags    :: Int8                       # number of active tags (<= MAX_TAGS)
    primed   :: Bool
    sitetype :: Symbol                     # Symbol() (empty) means "no site type"
end

# ── Internal tag normalisation ───────────────────────────────────────────────

function _make_tags(tags)
    syms = unique(sort(collect(Symbol, tags)))   # sort + dedup
    n = length(syms)
    @assert n <= MAX_TAGS "too many tags (max $MAX_TAGS): $syms"
    padded = ntuple(i -> i <= n ? syms[i] : Symbol(), MAX_TAGS)
    return padded, Int8(n)
end

# ── Constructors ─────────────────────────────────────────────────────────────

function Index(dim::Int, tags...; sitetype::Type{T}=Nothing) where {T<:Union{SiteType,Nothing}}
    @assert dim > 0 "Index dimension must be positive, got $dim"
    id = Threads.atomic_add!(_INDEX_COUNTER, UInt64(1))
    padded_tags, ntags = _make_tags(tags)
    st = sitetype === Nothing ? Symbol() : _sitetype_symbol(sitetype)
    Index(id, dim, padded_tags, ntags, false, st)
end

_sitetype_symbol(::Type{SiteType{S}}) where {S} = S

sitetype(i::Index) = i.sitetype === Symbol() ? nothing : SiteType{i.sitetype}()

# ── Priming ──────────────────────────────────────────────────────────────────
"""
    Base.adjoint(i::Index) -> Index
    prime(i::Index) -> Index
Return a new index that is primed.
    prime(i) == i' ≠ i
    prime(prime(i)) == i
"""
function Base.adjoint(i::Index)
    Index(i.id, i.dim, i.tags, i.ntags, !i.primed, i.sitetype)
end

"""
    noprime(i::Index) -> Index
Return a new index that is unprimed.
"""
function noprime(i::Index) 
    Index(i.id, i.dim, i.tags, i.ntags, false, i.sitetype)
end

prime(i::Index)   = i'
isprime(i::Index) = i.primed

# ── Equality & hashing ───────────────────────────────────────────────────────

Base.:(==)(a::Index, b::Index) = a.id == b.id && a.primed == b.primed
Base.hash(a::Index, h::UInt)   = hash((a.id, a.primed), h)

# ── Tag helpers ──────────────────────────────────────────────────────────────

"""
    hastag(i::Index, tag::Symbol) -> Bool

Return `true` if `tag` is present in `i`'s tag set.
"""
hastag(i::Index, tag::Symbol) = tag in i.tags

"""
    addtags(i::Index, tags...) -> Index

Return a new `Index` with `tags` added (same id, dim, prime level).
"""
function addtags(i::Index, newtags...)
    new_tags, n = _make_tags((tags(i)..., newtags...))
    Index(i.id, i.dim, new_tags, n, i.primed, i.sitetype)
end

"""
    removetags(i::Index, tags...) -> Index

Return a new `Index` with `tags` removed (same id, dim, prime level).
"""
function removetags(i::Index, removed...)
    to_remove = Set{Symbol}(removed)
    kept      = filter(t -> t ∉ to_remove, collect(tags(i)))
    new_tags, n = _make_tags(kept)
    Index(i.id, i.dim, new_tags, n, i.primed, i.sitetype)
end

"""
    tags(i::Index) -> NTuple

Return the (sorted) tuple of active tags of `i`.
"""
tags(i::Index) = ntuple(k -> i.tags[k], i.ntags)

# ── Display ──────────────────────────────────────────────────────────────────

function Base.show(io::IO, i::Index)
    tag_str = join(tags(i), ",")
    print(io, "Index($(i.dim), \"$(tag_str)\"$(i.primed ? "'" : ""), id=$(i.id))")
end

# ── Dimension helpers ────────────────────────────────────────────────────────
"""
    dim(i::Index) -> Int
Return the dimension of `i`.
"""
function dim(i::Index)
    return i.dim
end

"""
    dims(inds) -> NTuple
For a collection of indices `inds`, returns their dimension.
"""
dims(inds)      = ntuple(k -> inds[k].dim, length(inds))