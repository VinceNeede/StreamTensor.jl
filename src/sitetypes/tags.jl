# ── SiteType ─────────────────────────────────────────────────────────────────

"""
    SiteType{T}

Zero-cost tag struct identifying a local Hilbert space for dispatch.
Construct with `SiteType(:SpinHalf)` or `SiteType("SpinHalf")`.
"""
struct SiteType{T} end

SiteType(s::Symbol)                    = SiteType{s}()
SiteType(s::AbstractString)            = SiteType(Symbol(s))
SiteType(s::Symbol, params...)         = SiteType{(s, params...)}()
SiteType(t::Tuple)                     = SiteType{t}()

Base.show(io::IO, ::SiteType{T}) where {T <: Symbol} = print(io, "SiteType(:$T)")
Base.show(io::IO, ::SiteType{T}) where {T <: Tuple}  = print(io, "SiteType$(T)")

# ── OpName ───────────────────────────────────────────────────────────────────

"""
    OpName{N}

Zero-cost tag struct identifying a local operator for dispatch.
Construct with `OpName(:Sz)` or `OpName("Sz")`.

Define new operators by adding methods:

    import StreamTensor: op, SiteType, OpName
    # parameter-free
    op(::SiteType{:MyType}, ::OpName{:MyOp}) = ...
    # parametric
    op(::SiteType{:MyType}, ::OpName{:MyOp}; param::Real) = ...
"""
struct OpName{N} end

OpName(s::Symbol)         = OpName{s}()
OpName(s::AbstractString) = OpName(Symbol(s))

Base.show(io::IO, ::OpName{N}) where {N} = print(io, "OpName(:$N)")

# ── StateName ─────────────────────────────────────────────────────────────────

"""
    StateName{N}

Zero-cost tag struct identifying a basis state for dispatch.
Construct with `StateName(:Up)` or `StateName("Up")`.

Define new states by adding methods:

    import StreamTensor: state, SiteType, StateName
    # parameter-free
    state(::SiteType{:MyType}, ::StateName{:MyState}) = [1.0, 0.0, 0.0]
    # parametric (e.g. a coherent state parameterised by an angle)
    state(::SiteType{:MyType}, ::StateName{:Coherent}; θ::Real) = [cos(θ/2), sin(θ/2)]
"""
struct StateName{N} end

StateName(s::Symbol)         = StateName{s}()
StateName(s::AbstractString) = StateName(Symbol(s))

Base.show(io::IO, ::StateName{N}) where {N} = print(io, "StateName(:$N)")

macro alias_sitetype(expr)
    @assert expr.head == :call && expr.args[1] == :(=>) "Usage: @alias_sitetype Alias => Canonical"

    function to_sym(arg)
        if arg isa String
            QuoteNode(Symbol(arg))   # "S=1/2" → literal :S=1/2, not a variable lookup
        elseif arg isa Expr && arg.head == :tuple
            Expr(:tuple, map(a -> a isa QuoteNode ? a : QuoteNode(a), arg.args)...)
        else
            QuoteNode(arg)
        end
    end
    alias     = to_sym(expr.args[2])
    canonical = to_sym(expr.args[3])

    return esc(quote
        siteind(::SiteType{$(alias)}) =
            siteind(SiteType{$(canonical)}())
        op(::SiteType{$(alias)}, on::OpName; kwargs...) =
            op(SiteType{$(canonical)}(), on; kwargs...)
        state(::SiteType{$(alias)}, sn::StateName; kwargs...) =
            state(SiteType{$(canonical)}(), sn; kwargs...)
    end)
end