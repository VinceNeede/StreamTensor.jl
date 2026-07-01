
# ── siteind ──────────────────────────────────────────────────────────────────

"""
    siteind(st::SiteType) :: Index

Return a fresh `Index` carrying `:Site` and the site-type tag, with the
dimension matching the local Hilbert space.

    siteind(SiteType(:SpinHalf))  →  Index(2, :Site, :SpinHalf)
"""
function siteind end

siteind(s::Union{Symbol, AbstractString}) = siteind(SiteType(s))

"""
    siteinds(st, L::Int) :: Vector{Index}

Return `L` independent site indices for site type `st`.
"""
siteinds(st::Union{SiteType, Symbol, AbstractString}, L::Int) =
    [siteind(st) for _ in 1:L]

# ── state ─────────────────────────────────────────────────────────────────────

"""
    state(st::SiteType, sn::StateName; kwargs...) :: Vector

Return the basis-state vector for state name `sn` on site type `st`.
Keyword arguments are forwarded, allowing parametric states.

    state(SiteType(:SpinHalf), StateName(:Up))
    state(SiteType(:SpinHalf), StateName(:Coherent); θ=π/3)

The string entry point converts automatically:

    state(SiteType(:SpinHalf), "Up")
"""
function state end

# String entry point
state(st::SiteType, name::AbstractString; kwargs...) =
    state(st, StateName(name); kwargs...)

state(s::Union{Symbol, AbstractString}, name::AbstractString; kwargs...) =
    state(SiteType(s), StateName(name); kwargs...)

# ── op ───────────────────────────────────────────────────────────────────────

"""
    op(st::SiteType, on::OpName; kwargs...) :: Matrix

Return the local operator matrix for site type `st` and operator `on`.
Keyword arguments are forwarded, enabling parametric operators:

    op(SiteType(:SpinHalf), OpName(:Rz); θ=π/2)

The string entry point converts automatically:

    op(SiteType(:SpinHalf), "Rz"; θ=π/2)
"""
function op end

# -- String / Symbol entry points ---------------------------------------------

op(st::SiteType, name::AbstractString; kwargs...) =
    op(st, OpName(name); kwargs...)

op(s::Union{Symbol, AbstractString}, name::AbstractString; kwargs...) =
    op(SiteType(s), OpName(name); kwargs...)

function op(
    s::Index,
    opname::OpName;
    left_link::Index=Index(1, :Link), 
    right_link::Index=Index(1, :Link), 
    kwargs...
)
    O_mat = _dispatch_op(s, opname; kwargs...)
    s_out = s'
    return MPOTensor(reshape(O_mat, 1, s.dim, s.dim, 1), left_link, s_out, s, right_link)
end

function state(
    ::Type{T},
    s::Index,
    statename::StateName;
    left_link::Index=Index(1, :Link), 
    right_link::Index=Index(1, :Link), 
    kwargs...
) where{T}
    ψ_mat = T.(_dispatch_state(s, statename; kwargs...))
    return MPSTensor(reshape(ψ_mat, 1, s.dim, 1), left_link, s, right_link)
end

function op(s::Index, name::AbstractString; kwargs...)
    return op(s, OpName(name); kwargs...)
end

function op(s::Index, label_tuple::Tuple{<:AbstractString, <:NamedTuple}; kwargs...)
    return op(s, OpName(label_tuple[1]); label_tuple[2]..., kwargs...)
end

function state(::Type{T}, s::Index, name::AbstractString; kwargs...) where{T}
    return state(T, s, StateName(name); kwargs...)
end

function state(::Type{T},s::Index, label_tuple::Tuple{<:AbstractString, <:NamedTuple}; kwargs...) where{T}
    return state(T, s, StateName(label_tuple[1]); label_tuple[2]..., kwargs...)
end

function _dispatch_op(s::Index, on::OpName; kwargs...)
    st = sitetype(s)
    st === nothing && error("Index $s has no associated site type " *
                             "(e.g. a :Link index); cannot resolve operator $on")
    return op(st, on; kwargs...)
end

function _dispatch_state(s::Index, sn::StateName; kwargs...)
    st = sitetype(s)
    st === nothing && error("Index $s has no associated site type " *
                             "(e.g. a :Link index); cannot resolve state $sn")
    return state(st, sn; kwargs...)
end
