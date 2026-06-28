module StreamTensor

using LinearAlgebra
using StaticArrays
import LinearAlgebra: mul!, norm, Diagonal
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

# Tensor accessors / operations
export inds, siteind, siteinds, linkinds
export to_dense, dag
export size, ndims, eltype

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
export siteinds, linkdim, sim_linkinds
export orthogonalize, orthogonalize!
export inner

# ==================
# MPO
# ==================
export MPO
export expect

# ==================
# OpSum
# ==================
export OpSum, OpTerm, add!

# ==================
# Site types
# ==================
export SiteType, OpName, StateName
export @alias_sitetype
export siteind, siteinds
export state
export op

# ==================
# DMRG
# ==================
export ProjMPO, nsite
export dmrg_sweep!, dmrg!

end