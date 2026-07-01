"""
    AbstractTensorTrain{T}

Abstract base type for one-dimensional tensor networks (MPS, MPO, …) with
element type `T`.  Concrete subtypes must provide fields `tensors`, `llim`,
and `rlim` so that all accessor functions defined here work out of the box.

The orthogonality window `(llim, rlim)` satisfies `0 ≤ llim < rlim ≤ L+1`:
- `llim == i-1, rlim == i+1` means site `i` is the orthogonality center.
- `llim == 0, rlim == L+1` means the state is unorthogonalized.
"""
abstract type AbstractTensorTrain{T} end

function _validate_tensor_train(tensors::Vector, llim::Int, rlim::Int)
    L = length(tensors)
    @assert L > 0 "TensorTrain must have at least one tensor"
    @assert 0 <= llim < rlim <= L + 1 "Invalid orthogonality limits: llim=$llim, rlim=$rlim"
    for i in 1:L-1
        @assert tensors[i].right == tensors[i+1].left "Bond index mismatch between sites $i and $(i+1)"
    end
end

Base.length(tt::AbstractTensorTrain)               = length(tt.tensors)
Base.getindex(tt::AbstractTensorTrain, i::Int)     = tt.tensors[i]
Base.setindex!(tt::AbstractTensorTrain, t, i::Int) = (tt.tensors[i] = t)
Base.firstindex(tt::AbstractTensorTrain)           = 1
Base.lastindex(tt::AbstractTensorTrain)            = length(tt.tensors)
Base.iterate(tt::AbstractTensorTrain)              = iterate(tt.tensors)
Base.iterate(tt::AbstractTensorTrain, state)       = iterate(tt.tensors, state)

function Base.show(io::IO, tt::AbstractTensorTrain)
    println(io, "$(typeof(tt)):")
    for (i, t) in enumerate(tt)
        println(io, "\t[$i]: $(inds(t))")
    end
end

"""
    nsites(tt::AbstractTensorTrain) -> Int

Return the number of sites (tensors) in `tt`.
"""
nsites(tt::AbstractTensorTrain) = length(tt.tensors)

"""
    leftlim(tt::AbstractTensorTrain) -> Int
    rightlim(tt::AbstractTensorTrain) -> Int

Return the left / right orthogonality limit of `tt`.
All tensors at sites `1:leftlim(tt)` are left-orthogonal and all tensors at
sites `rightlim(tt):end` are right-orthogonal.
"""
leftlim(tt::AbstractTensorTrain)  = tt.llim
rightlim(tt::AbstractTensorTrain) = tt.rlim

"""
    setleftlim!(tt::AbstractTensorTrain, l::Int)
    setrightlim!(tt::AbstractTensorTrain, r::Int)

Mutate the stored orthogonality limits without touching the tensors.
"""
setleftlim!(tt::AbstractTensorTrain, l::Int)  = (tt.llim = l)
setrightlim!(tt::AbstractTensorTrain, r::Int) = (tt.rlim = r)

"""
    isortho(tt::AbstractTensorTrain) -> Bool

Return `true` if `tt` has a well-defined single orthogonality center,
i.e. `leftlim(tt) + 2 == rightlim(tt)`.
"""
isortho(tt::AbstractTensorTrain) = tt.llim + 2 == tt.rlim

"""
    orthocenter(tt::AbstractTensorTrain) -> Int

Return the site index of the orthogonality center.
Throws an assertion error if `!isortho(tt)`.
"""
function orthocenter(tt::AbstractTensorTrain)
    @assert isortho(tt) "No well-defined orthogonality center"
    return tt.llim + 1
end

Base.eltype(::AbstractTensorTrain{T}) where {T} = T

"""
    linkind(tt::AbstractTensorTrain, i::Int) -> Index

Return the right link `Index` of site `i` (i.e. the bond between sites `i`
and `i+1`).
"""
linkind(tt::AbstractTensorTrain, i::Int) = tt[i].right   # was i::int — fixed

"""
    linkinds(tt::AbstractTensorTrain) -> Vector{Index}

Return all internal bond indices of `tt` in site order.
"""
linkinds(tt::AbstractTensorTrain) = [linkind(tt, i) for i in 1:length(tt)]

"""
    maxlinkdim(tt::AbstractTensorTrain) -> Int

Return the maximum bond dimension across all internal bonds of `tt`.
"""
maxlinkdim(tt::AbstractTensorTrain) = maximum(dims(linkinds(tt)))


function _shift_center_right!(tt::AbstractTensorTrain, i::Int;
                               maxdim=nothing, cutoff=nothing)
    Q, R = factorize(tt[i], LeftOrthogonal; maxdim, cutoff)
    tt[i]   = Q
    tt[i+1] = _to_traintensor(R * tt[i+1])
end

function _shift_center_left!(tt::AbstractTensorTrain, i::Int;
                              maxdim=nothing, cutoff=nothing)
    L, Q = factorize(tt[i], RightOrthogonal; maxdim, cutoff)
    tt[i]   = Q
    tt[i-1] = _to_traintensor(tt[i-1] * L)
end

"""
    orthogonalize!(tt::AbstractTensorTrain, center::Int) -> AbstractTensorTrain

In-place: sweep left and right to bring the orthogonality center to site
`center` using QR decompositions. Updates `tt.llim` and `tt.rlim`.
Works for both MPS and MPO.
"""
function orthogonalize!(tt::AbstractTensorTrain, center::Int)
    L = length(tt)
    @assert 1 <= center <= L "Center $center out of bounds"

    for i in tt.llim+1 : center-1
        _shift_center_right!(tt, i)
    end
    for i in tt.rlim-1 : -1 : center+1
        _shift_center_left!(tt, i)
    end

    setleftlim!(tt, center - 1)
    setrightlim!(tt, center + 1)
    return tt
end

"""
    compress!(tt::AbstractTensorTrain, center::Int; maxdim, cutoff) -> AbstractTensorTrain

In-place: sweep over all sites with SVD truncation to bring the orthogonality
center to `center`. Unlike `orthogonalize!`, always processes all sites
regardless of current `llim`/`rlim`. Works for both MPS and MPO.
"""
function compress!(tt::AbstractTensorTrain, center::Int;
                   maxdim=nothing, cutoff=nothing)
    L = length(tt)
    @assert 1 <= center <= L "Center $center out of bounds"

    for i in 1:center-1
        _shift_center_right!(tt, i; maxdim, cutoff)
    end
    for i in L:-1:center+1
        _shift_center_left!(tt, i; maxdim, cutoff)
    end

    setleftlim!(tt, center - 1)
    setrightlim!(tt, center + 1)
    return tt
end

orthogonalize(tt::AbstractTensorTrain, center::Int) =
    orthogonalize!(copy(tt), center)

compress(tt::AbstractTensorTrain, center::Int; kwargs...) =
    compress!(copy(tt), center; kwargs...)