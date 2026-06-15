
# ==================
# Internal helpers
# ==================

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
    u_inds = ntuple(i -> i <= NL ? left_inds[i] : bond_u, NL + 1)
    v_inds = ntuple(i -> i == 1 ? bond_v : right_inds[i - 1], NR + 1)

    return Index(χ, :Link), Index(χ, :Link), u_inds, v_inds
end

function _new_bonds(χ::Int)
    return Index(χ, :Link), Index(χ, :Link)
end

# ==================
# SVD methods
# ==================

function LinearAlgebra.svd(A::AbstractTensor, left_inds::NTuple{NL, Index}; kwargs...) where {NL}
    LinearAlgebra.svd(to_dense(A), left_inds; kwargs...)
end

function LinearAlgebra.svd(A::DenseTensor{T,NA}, left_inds::NTuple{NL, Index};
             maxdim::Union{Int, Nothing}=nothing,
             cutoff::Union{Real, Nothing}=nothing) where {T, NA, NL}

    inds_a = inds(A)
    NR = NA - NL

    right_inds_mut = MVector{NA-NL, Index}(undef)
    ir = 0
    for idx in inds_a
        if idx ∉ left_inds
            ir += 1 
            right_inds_mut[ir] = idx
        end
    end
    right_inds = Tuple(right_inds_mut)

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
    bond_u, bond_v, u_inds, v_inds = _new_bonds(χ)

    U = DenseTensor(u_inds, reshape(U_mat, dims(left_inds)..., χ))
    S = DiagTensor((bond_u, bond_v), s)
    V = DenseTensor(v_inds, reshape(Vt_mat, χ, dims(right_inds)...))

    return U, S, V, truncerr
end

@enum SVDDirection LeftOrthogonal RightOrthogonal

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

function LinearAlgebra.svd(A::MPOTensor{T};
             maxdim::Union{Int, Nothing}=nothing,
             cutoff::Union{Real, Nothing}=nothing,
             direction::SVDDirection=LeftOrthogonal) where {T}

    χl, di, do_, χr = size(A.storage)

    if direction == LeftOrthogonal
        # split (χl * di * do_ | χr) — free reshape, no permutation
        U_mat, s, Vt_mat = _mat_svd(reshape(A.storage, χl * di * do_, χr))
        U_mat, s, Vt_mat, truncerr = _truncate(s, U_mat, Vt_mat; maxdim, cutoff)
        χ = length(s)
        bond_u, bond_v = _new_bonds(χ)

        U = MPOTensor(reshape(U_mat, χl, di, do_, χ), A.left, A.site_in, A.site_out, bond_u)
        S = DiagTensor((bond_u, bond_v), s)
        V = DenseTensor((bond_v, A.right), reshape(Vt_mat, χ, χr))
    else
        # split (χl | di * do_ * χr) — free reshape, no permutation
        U_mat, s, Vt_mat = _mat_svd(reshape(A.storage, χl, di * do_ * χr))
        U_mat, s, Vt_mat, truncerr = _truncate(s, U_mat, Vt_mat; maxdim, cutoff)
        χ = length(s)
        bond_u, bond_v = _new_bonds(χ)

        U = DenseTensor((A.left, bond_u), reshape(U_mat, χl, χ))
        S = DiagTensor((bond_u, bond_v), s)
        V = MPOTensor(reshape(Vt_mat, χ, di, do_, χr), bond_v, A.site_in, A.site_out, A.right)
    end

    return U, S, V, truncerr
end

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
