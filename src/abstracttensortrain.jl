abstract type AbstractTensorTrain{T} end

function _validate_tensor_train(tensors::Vector, llim::Int, rlim::Int)
    L = length(tensors)
    @assert L > 0 "TensorTrain must have at least one tensor"
    @assert 0 <= llim < rlim <= L + 1 "Invalid orthogonality limits: llim=$llim, rlim=$rlim"
    for i in 1:L-1
        @assert tensors[i].right == tensors[i+1].left "Bond index mismatch between sites $i and $(i+1)"
    end
end

Base.length(tt::AbstractTensorTrain)            = length(tt.tensors)
Base.getindex(tt::AbstractTensorTrain, i::Int)  = tt.tensors[i]
Base.setindex!(tt::AbstractTensorTrain, t, i::Int) = (tt.tensors[i] = t)
Base.firstindex(tt::AbstractTensorTrain)        = 1
Base.lastindex(tt::AbstractTensorTrain)         = length(tt.tensors)
Base.iterate(tt::AbstractTensorTrain)           = iterate(tt.tensors)
Base.iterate(tt::AbstractTensorTrain, state)    = iterate(tt.tensors, state)

function Base.show(io::IO, tt::AbstractTensorTrain)
    println(io, "$typeof(tt):")
    for (i, t) in enumerate(tt)
        println(io, "\t[$i]: $(inds(t))")
    end
end

nsites(tt::AbstractTensorTrain)                 = length(tt.tensors)
Base.eltype(::AbstractTensorTrain{T}) where {T} = T
leftlim(tt::AbstractTensorTrain)                = tt.llim
rightlim(tt::AbstractTensorTrain)               = tt.rlim
setleftlim!(tt::AbstractTensorTrain, l::Int)    = (tt.llim = l)
setrightlim!(tt::AbstractTensorTrain, r::Int)   = (tt.rlim = r)

isortho(tt::AbstractTensorTrain)                = tt.llim + 2 == tt.rlim

function orthocenter(tt::AbstractTensorTrain)
    @assert isortho(tt) "No well-defined orthogonality center"
    return tt.llim + 1
end