"""
    OpTerm{C<:Number}

A single term in an [`OpSum`](@ref): a scalar coefficient `coeff` and an
ordered list of `site => (opname, params)` pairs (sorted by site).

Not normally constructed directly — use [`add!`](@ref) or `opsum + (...)`.
"""
struct OpTerm{C<:Number}
    coeff::C
    ops::Vector{Pair{Int, Tuple{String,NamedTuple}}}  # site => (opname, params)
end

"""
    OpSum

A sum of operator terms used to build an [`MPO`](@ref).

```julia
H = OpSum()
H += (J, "Sz", 1, "Sz", 2)
H += (h, "Sx", 1)
mpo = MPO(H, sites)
```

Each term is a `Tuple` whose first element is the numeric coefficient, followed
by alternating `opname, site` pairs.  Parametric operators use a
`(name, NamedTuple)` pair in place of the bare name string.

See also: [`add!`](@ref), [`MPO`](@ref).
"""
struct OpSum
    terms::Vector{OpTerm}
end

OpSum() = OpSum(OpTerm[])

"""
    add!(opsum::OpSum, coeff, ops::Pair{Int,Tuple{String,NamedTuple}}...) -> OpSum
    add!(opsum::OpSum, term::Tuple) -> OpSum

Append a new term to `opsum` (mutating).

The low-level form takes a numeric `coeff` and explicit `site => (name, params)`
pairs.  The high-level tuple form parses `(coeff, name, site, name, site, …)`,
normalising bare strings to `(name, (;))` automatically.
"""
function add!(opsum::OpSum, coeff::Number, ops::Pair{Int,Tuple{String,NamedTuple}}...)
    push!(opsum.terms, OpTerm(coeff, sort(collect(ops); by=first)))
    return opsum
end

function add!(opsum::OpSum, term::Tuple{Vararg})
    if isempty(term)
        return opsum
    end

    # 1. Extract the coefficient safely
    coeff = first(term)
    @assert coeff isa Number "The first element of an OpSum tuple must be a numeric coefficient."

    ops = Pair{Int, Tuple{String, NamedTuple}}[]
    
    # 2. Parse the remaining elements structurally (Op, Site, Op, Site...)
    i = 2
    while i <= length(term)
        opdata = term[i]
        
        # Normalize the operator data format
        if opdata isa String
            op = (opdata, (;))
        elseif opdata isa Tuple && length(opdata) == 2 && opdata[1] isa String && opdata[2] isa NamedTuple
            op = opdata
        else
            error("Expected operator name (String) or a (String, NamedTuple) pair at position $i, got: $opdata")
        end
        
        i += 1
        if i > length(term)
            error("Malformed term: Missing a physical site index for operator '$(op[1])'.")
        end
        
        site = term[i]
        @assert site isa Int "Expected an integer site index at position $i, got: $site"
        
        push!(ops, site => op)
        i += 1
    end
    
    # 3. Hand off to the safe, pre-existing raw broad method
    return add!(opsum, coeff, ops...)
end

"""
    opsum + term -> OpSum

Syntactic sugar for `add!(opsum, term)`.  Mutates `opsum` in place and
returns it (despite the `+` spelling) so that the idiom `H += (coeff, ...)` works.
"""
function Base.:+(opsum::OpSum, term::Tuple{Vararg})
    return add!(opsum, term)
end

const FSMLabel = Union{Symbol, Tuple{Int,Int}}

function _fsm_states(opsum::OpSum, N::Int)
    states = [FSMLabel[] for _ in 0:N]   # states[n+1] = label set at bond n
    states[1]   = FSMLabel[:I]
    states[end] = FSMLabel[:F]
    # initialize
    for n in 1:(N-1)
        push!(states[n+1], :I, :F)
    end

    for (α, term) in enumerate(opsum.terms)
        ops = term.ops
        for j in 1:(length(ops)-1)
            site_j, site_j1 = ops[j][1], ops[j+1][1]
            for n in site_j:(site_j1-1)
                push!(states[n+1], (α, j))
            end
        end
    end
    return states
end

function _fsm_site_tensor(opsum::OpSum, n::Int, site::Index,
                           S_prev::Vector{FSMLabel}, S_curr::Vector{FSMLabel},
                           op_cache, C::Type)
    d  = dim(site)
    T  = zeros(C, length(S_prev), d, d, length(S_curr))
    ip = Dict(l => i for (i,l) in enumerate(S_prev))
    ic = Dict(l => i for (i,l) in enumerate(S_curr))
    Id = Matrix{C}(LinearAlgebra.I, d, d)

    for (label, i) in ip
        j = get(ic, label, nothing)
        isnothing(j) || (T[i,:,:,j] .+= Id)
    end

    for (α, term) in enumerate(opsum.terms)
        ops = term.ops
        for (j, (site_j, opdata)) in enumerate(ops)
            site_j == n || continue
            r = j == 1 ? :I : (α, j-1)
            c = j == length(ops) ? :F : (α, j)
            opmat = op_cache[(site_j, opdata)]
            coeff = j == 1 ? term.coeff : one(C)
            T[ip[r], :, :, ic[c]] .+= coeff .* opmat
        end
    end
    return T
end

function _opsum_cache_and_eltype(opsum::OpSum, sites)
    cache = Dict{Tuple{Int,Tuple{String,NamedTuple}}, Matrix}()
    C = Float64
    for term in opsum.terms
        C = promote_type(C, typeof(term.coeff))
        for (site_j, opdata) in term.ops
            key = (site_j, opdata)
            mat = get!(cache, key) do
                _dispatch_op(sites[site_j], OpName(opdata[1]); opdata[2]...)
            end
            C = promote_type(C, eltype(mat))
        end
    end
    return cache, C
end

function MPO(opsum::OpSum, sites::Vector{<:Index})
    N = length(sites)
    states = _fsm_states(opsum, N)
    cache, C = _opsum_cache_and_eltype(opsum, sites)
    links = [Index(length(states[n+1]), :Link) for n in 0:N]

    tensors = map(1:N) do n
        T = _fsm_site_tensor(opsum, n, sites[n], states[n], states[n+1], cache, C)
        MPOTensor(T, links[n], sites[n], sites[n]', links[n+1])
    end
    return MPO(tensors)
end