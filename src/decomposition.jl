
# ==================
# Internal helpers
# ==================

"""
    _truncate(s, U_mat, Vt_mat; maxdim, cutoff) -> (U, s, Vt, truncerr)

Truncate a thin SVD (singular values `s` assumed sorted descending) to at most
`maxdim` values and/or discard values below `cutoff`.  Returns the sliced
factors and the relative truncation error `‖s_discarded‖² / ‖s_full‖²`.
"""
function _truncate(s::AbstractVector, U_mat::AbstractMatrix, Vt_mat::AbstractMatrix;
                   maxdim::Union{Int, Nothing}=nothing,
                   cutoff::Union{Real, Nothing}=nothing)
    χ = length(s)
    if !isnothing(cutoff)
        χ = something(findlast(>=(cutoff), s), 0)
    end
    if !isnothing(maxdim)
        χ = min(χ, maxdim)
    end
    χ = max(χ, 1)

    total = sum(abs2, s)
    truncerr = total > 0 ? sum(abs2, @view s[(χ+1):end]) / total : zero(eltype(s))

    return U_mat[:, 1:χ], s[1:χ], Vt_mat[1:χ, :], truncerr
end

function _mat_svd(A_mat::AbstractMatrix)
    F = LinearAlgebra.svd(A_mat; full=false)
    return F.U, F.S, F.Vt
end

function _new_bonds(χ::Int, left_inds::NTuple{NL,Index}, right_inds::NTuple{NR,Index}) where {NL,NR}
    bond_u = Index(χ, :Link)
    bond_v = Index(χ, :Link)
    u_inds = ntuple(i -> i <= NL ? left_inds[i] : bond_u, NL + 1)
    v_inds = ntuple(i -> i == 1 ? bond_v : right_inds[i - 1], NR + 1)

    return bond_u, bond_v, u_inds, v_inds
end

function _new_bonds(χ::Int)
    return Index(χ, :Link), Index(χ, :Link)
end

# ==================
# SVD methods
# ==================

"""
    svd(A::AbstractTensor, left_inds::NTuple; maxdim, cutoff)
        -> (U::DenseTensor, S::DiagTensor, V::DenseTensor, truncerr::Real)

Tensor SVD splitting `A` into `U * S * V`.  `left_inds` identifies which legs
go onto `U`; the remaining legs go onto `V`.  `S` is a rank-2 `DiagTensor`
with a fresh pair of bond `Index`es.

Keyword arguments:
- `maxdim::Int` — keep at most this many singular values.
- `cutoff::Real` — discard singular values below this threshold.

Returns the truncation error as a fourth value.
"""
function LinearAlgebra.svd(A::AbstractTensor, left_inds::NTuple{NL, Index}; kwargs...) where {NL}
    LinearAlgebra.svd(to_dense(A), left_inds; kwargs...)
end

function LinearAlgebra.svd(A::DenseTensor{T,NA}, left_inds::NTuple{NL, Index};
             maxdim::Union{Int, Nothing}=nothing,
             cutoff::Union{Real, Nothing}=nothing) where {T, NA, NL}

    inds_a = inds(A)
    NR = NA - NL

    right_inds = ntuple(Val(NR)) do i
        count = 0
        for idx in inds_a
            if idx ∉ left_inds
                count += 1
                if count == i
                    return idx
                end
            end
        end
        error("Index missing: left_inds is not a strict subset of A's indices.")
    end

    perm_mut = zero(MVector{NA, Int})
    @inbounds begin
        for i in 1:NL
            perm_mut[i] = findfirst(==(left_inds[i]), inds_a)
        end
        for i in 1:NR
            perm_mut[i + NL] = findfirst(==(right_inds[i]), inds_a)
        end
    end
    perm = Tuple(perm_mut)
    
    A_p       = _maybe_permute(A.storage, perm)
    left_dim  = prod(dims(left_inds))
    right_dim = prod(dims(right_inds))
    A_mat     = reshape(A_p, left_dim, right_dim)

    U_mat, s, Vt_mat = _mat_svd(A_mat)
    U_mat, s, Vt_mat, truncerr = _truncate(s, U_mat, Vt_mat; maxdim, cutoff)
    χ = length(s)
    bond_u, bond_v, u_inds, v_inds = _new_bonds(χ, left_inds, right_inds)

    U = DenseTensor(u_inds, reshape(U_mat, dims(left_inds)..., χ))
    S = DiagTensor((bond_u, bond_v), s)
    V = DenseTensor(v_inds, reshape(Vt_mat, χ, dims(right_inds)...))

    return U, S, V, truncerr
end

@enum SVDDirection LeftOrthogonal RightOrthogonal

"""
    svd(A::MPSTensor; maxdim, cutoff, direction) -> (U, S, V, truncerr)

SVD of a single MPS tensor.

- `direction = LeftOrthogonal` (default): splits `(χl·d | χr)`, returning a
  left-orthogonal `MPSTensor` `U` and a `DenseTensor` `V` absorbing `S`.
- `direction = RightOrthogonal`: splits `(χl | d·χr)`, returning a `DenseTensor`
  `U` and a right-orthogonal `MPSTensor` `V`.
"""
function LinearAlgebra.svd(A::MPSTensor{T};
             maxdim::Union{Int, Nothing}=nothing,
             cutoff::Union{Real, Nothing}=nothing,
             direction::SVDDirection=LeftOrthogonal) where {T}

    χl, d, χr = size(A.storage)

    if direction == LeftOrthogonal
        # split (χl * d | χr) — U is left-orthogonal MPSTensor
        U_mat, s, Vt_mat = _mat_svd(reshape(A.storage, χl * d, χr))
        U_mat, s, Vt_mat, truncerr = _truncate(s, U_mat, Vt_mat; maxdim, cutoff)
        χ = length(s)
        bond_u, bond_v = _new_bonds(χ)

        U = MPSTensor(reshape(U_mat,  χl, d, χ), A.left, A.site, bond_u)
        S = DiagTensor((bond_u, bond_v), s)
        V = DenseTensor((bond_v, A.right), reshape(Vt_mat, χ, χr))
    else
        # split (χl | d * χr) — V is right-orthogonal MPSTensor
        U_mat, s, Vt_mat = _mat_svd(reshape(A.storage, χl, d * χr))
        U_mat, s, Vt_mat, truncerr = _truncate(s, U_mat, Vt_mat; maxdim, cutoff)
        χ = length(s)
        bond_u, bond_v = _new_bonds(χ)

        U = DenseTensor((A.left, bond_u), reshape(U_mat, χl, χ))
        S = DiagTensor((bond_u, bond_v), s)
        V = MPSTensor(reshape(Vt_mat, χ, d, χr), bond_v, A.site, A.right)
    end

    return U, S, V, truncerr
end

"""
    svd(A::MPOTensor; maxdim, cutoff, direction) -> (U, S, V, truncerr)

SVD of a single MPO tensor.  Mirrors the `MPSTensor` convention:

- `LeftOrthogonal`: splits `(χl·d_out·d_in | χr)`.
- `RightOrthogonal`: splits `(χl | d_out·d_in·χr)`.

Both directions require no storage permutation because the canonical axis order
`(χl, d_out, d_in, χr)` already groups left and right legs contiguously.
"""
function LinearAlgebra.svd(A::MPOTensor{T};
             maxdim::Union{Int, Nothing}=nothing,
             cutoff::Union{Real, Nothing}=nothing,
             direction::SVDDirection=LeftOrthogonal) where {T}

    χl, do_, di, χr = size(A.storage)

    if direction == LeftOrthogonal
        # split (χl * do_ * di | χr) — free reshape, no permutation
        U_mat, s, Vt_mat = _mat_svd(reshape(A.storage, χl * do_ * di, χr))
        U_mat, s, Vt_mat, truncerr = _truncate(s, U_mat, Vt_mat; maxdim, cutoff)
        χ = length(s)
        bond_u, bond_v = _new_bonds(χ)

        U = MPOTensor(reshape(U_mat, χl, do_, di, χ), A.left, A.site_out, A.site_in, bond_u)
        S = DiagTensor((bond_u, bond_v), s)
        V = DenseTensor((bond_v, A.right), reshape(Vt_mat, χ, χr))
    else
        # split (χl | do_ * di * χr) — free reshape, no permutation
        U_mat, s, Vt_mat = _mat_svd(reshape(A.storage, χl, do_ * di * χr))
        U_mat, s, Vt_mat, truncerr = _truncate(s, U_mat, Vt_mat; maxdim, cutoff)
        χ = length(s)
        bond_u, bond_v = _new_bonds(χ)

        U = DenseTensor((A.left, bond_u), reshape(U_mat, χl, χ))
        S = DiagTensor((bond_u, bond_v), s)
        V = MPOTensor(reshape(Vt_mat, χ, do_, di, χr), bond_v, A.site_out, A.site_in, A.right)
    end

    return U, S, V, truncerr
end

"""
    qr(A::MPSTensor; direction) -> (Q, R) or (L, Q)

Thin QR decomposition of a single MPS tensor.

- `direction = LeftOrthogonal` (default): splits `(χl·d | χr)`, returning a
  left-orthogonal `MPSTensor` `Q` and a `DenseTensor` `R` (upper-triangular).
- `direction = RightOrthogonal`: splits `(χl | d·χr)` via LQ, returning a
  `DenseTensor` `L` (lower-triangular) and a right-orthogonal `MPSTensor` `Q`.
"""
function LinearAlgebra.qr(A::MPSTensor{T};
             direction::SVDDirection=LeftOrthogonal) where {T}

    χl, d, χr = size(A.storage)

    if direction == LeftOrthogonal
        # QR of (χl*d × χr) — thin, direct LAPACK
        A_mat    = reshape(A.storage, χl * d, χr)  # must be plain Matrix for LAPACK
        χ        = min(χl * d, χr)
        F = qr(A_mat)
        R = F.R[1:χ, :]
        Q = Matrix(F.Q[:, 1:χ])

        bond = Index(χ, :Link)

        Q_tensor = MPSTensor(reshape(Q, χl, d, χ), A.left, A.site, bond)
        R_tensor = DenseTensor((bond, A.right), R)
        return Q_tensor, R_tensor

    else
        A_mat    = reshape(A.storage, χl, d * χr)
        χ        = min(χl, d * χr)
        bond     = Index(χ, :Link)
        F = qr(A_mat')  # QR of transpose to get LQ factors
        L = Matrix(F.R[1:χ, :]')
        Q = Matrix(F.Q[:, 1:χ]')

        L_tensor = DenseTensor((A.left, bond), L)
        Q_tensor = MPSTensor(reshape(Q, χ, d, χr), bond, A.site, A.right)
        return L_tensor, Q_tensor
    end
end
