module StreamTensor

using LinearAlgebra
using StaticArrays
import KrylovKit: eigsolve

include("sitetypes/tags.jl")
include("index.jl")
include("tensor.jl")
include("contraction.jl")
include("decomposition.jl")
include("abstracttensortrain.jl")
include("mps.jl")
include("mpo.jl")
include("opsum.jl")
include("sitetypes/sitetypes.jl")
include("sitetypes/qubit.jl")
include("dmrg.jl")

# ==================
# Index
# ==================
export Index
export dim, dims
export hastag, addtags, removetags, tags, prime, noprime, isprime, sitetype

# ==================
# Tensor types
# ==================
export AbstractTensor
export DenseTensor, DiagTensor, DeltaTensor
export MPSTensor, MPOTensor

# Tensor accessors
export inds, siteind, siteinds, linkinds
export to_dense, size, ndims, eltype

# ==================
# Contraction
# ==================
export contract

# ==================
# Decomposition
# ==================
export SVDDirection, LeftOrthogonal, RightOrthogonal
export svd, qr

# ==================
# AbstractTensorTrain
# ==================
export AbstractTensorTrain
export nsites, leftlim, rightlim, isortho, orthocenter
export setleftlim!, setrightlim!
export linkind, linkinds, maxlinkdim

# ==================
# MPS
# ==================
export MPS, random_mps
export orthogonalize, orthogonalize!
export inner

# ==================
# MPO
# ==================
export MPO
export expect
export OpSum, add!

# ==================
# Site types
# ==================
export SiteType, OpName, StateName
export siteind, siteinds
export state
export op
export product_state

# ==================
# DMRG
# ==================
export ProjMPO, dmrg!

end