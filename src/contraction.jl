function _find_contracted_free(
    inds_a::NTuple{NA, Index},
    inds_b::NTuple{NB, Index}
) where {NA, NB}
    
    c_a = zero(MVector{NA, Int})
    c_b = zero(MVector{NB, Int})   # indices into B — needs NB capacity, not NA
    f_a = zero(MVector{NA, Int})
    f_b = zero(MVector{NB, Int})
    
    nc = nfa = nfb = 0
    
    for ia in 1:NA
        a = inds_a[ia]
        ib = findfirst(==(a), inds_b)
        if ib !== nothing
            nc += 1
            c_a[nc] = ia
            c_b[nc] = ib
        else
            nfa += 1
            f_a[nfa] = ia
        end
    end
    
    for ib in 1:NB
        if !any(==(inds_b[ib]), inds_a)
            nfb += 1
            f_b[nfb] = ib
        end
    end
    
    # 2. Build output permutations safely using standard loops.
    # Slicing causes heap allocation
    perm_a_mut = zero(MVector{NA, Int})
    perm_b_mut = zero(MVector{NB, Int})
    
    @inbounds begin
        # For A: Free first, then Contracted
        for i in 1:nfa;          perm_a_mut[i] = f_a[i];       end
        for i in 1:nc;           perm_a_mut[nfa + i] = c_a[i]; end

        # For B: Contracted first, then Free
        for i in 1:nc;           perm_b_mut[i] = c_b[i];       end
        for i in 1:nfb;          perm_b_mut[nc + i] = f_b[i];  end
    end
    perm_a = Tuple(perm_a_mut)
    perm_b = Tuple(perm_b_mut)
    return perm_a, perm_b, nc, nfa, nfb
end
function _maybe_permute(storage::AbstractArray{T,N}, perm::NTuple{N, Int}) where {T,N}
    issorted(perm) && return storage
    return permutedims(storage, collect(perm))
end

function _matricize(storage::AbstractArray{T,N}, perm::NTuple{N,Int}, ngroup1::Int, d_first::Int, d_second::Int) where {T,N}
    if issorted(perm)
        return reshape(storage, d_first, d_second)
    end
    if perm == (ntuple(i -> ngroup1+i, N-ngroup1)..., ntuple(identity, ngroup1)...)
        return transpose(reshape(storage, d_second, d_first))
    end
    return reshape(_maybe_permute(storage, perm), d_first, d_second)
end

"""
    contract(A::DenseTensor, B::DenseTensor) -> DenseTensor

Contract `A` and `B` over all shared indices.  Free indices of `A` come first
in the output, followed by free indices of `B`, in their original order.

Uses `_matricize` to avoid a copy when the required permutation is an identity
or a clean two-block swap, falling back to `permutedims` otherwise.
"""
function contract(A::DenseTensor{TA,NA}, B::DenseTensor{TB,NB}) where {TA,TB,NA,NB}
    inds_a = inds(A)
    inds_b = inds(B)

    perm_a, perm_b, nc, nfa, nfb = _find_contracted_free(inds_a, inds_b)

    inds_a_p = ntuple(i -> inds_a[perm_a[i]], Val(NA))
    inds_b_p = ntuple(i -> inds_b[perm_b[i]], Val(NB))

    free_dim_a = 1
    for i in 1:nfa;      free_dim_a *= inds_a_p[i].dim; end
    cont_dim   = 1
    for i in (nfa+1):NA; cont_dim   *= inds_a_p[i].dim; end
    free_dim_b = 1
    for i in (nc+1):NB;  free_dim_b *= inds_b_p[i].dim; end

    a_mat = _matricize(A.storage, perm_a, nc, free_dim_a, cont_dim)
    b_mat = _matricize(B.storage, perm_b, nfb, cont_dim, free_dim_b)

    T = promote_type(TA, TB)
    a_mat = TA === T ? a_mat : T.(a_mat)
    b_mat = TB === T ? b_mat : T.(b_mat)

    c_storage = Matrix{T}(undef, free_dim_a, free_dim_b)
    mul!(c_storage, a_mat, b_mat)

    NOUT = nfa + nfb
    out_inds = ntuple(NOUT) do i
        i <= nfa ? inds_a_p[i] : inds_b_p[nc + (i - nfa)]
    end

    out_storage = reshape(c_storage, dims(out_inds)...)

    return DenseTensor(out_inds, out_storage)
end

function _contract_dense_diag(
    A::DenseTensor{TA,NA}, 
    D::DiagTensor{TB,ND}, 
    ::Val{A_is_left}
) where {TA,TB,NA,ND,A_is_left}
    T = promote_type(TA, TB)
    inds_a = inds(A)
    inds_d = inds(D)

    if A_is_left
        perm_a, perm_d, nc, nfa, nfd = _find_contracted_free(inds_a, inds_d)
    else
        perm_d, perm_a, nc, nfd, nfa = _find_contracted_free(inds_d, inds_a)
    end

    if nc == 0
        if A_is_left
            return contract(A, to_dense(D))
        else
            return contract(to_dense(D), A)
        end
    end

    inds_a_p = ntuple(i -> inds_a[perm_a[i]], Val(NA))
    inds_d_p = ntuple(i -> inds_d[perm_d[i]], Val(ND))

    NOUT = nfa + nfd
    out_inds = if A_is_left
        ntuple(NOUT) do i
            i <= nfa ? inds_a_p[i] : inds_d_p[nc + (i - nfa)]
        end
    else
        ntuple(NOUT) do i
            i <= nfd ? inds_d_p[i] : inds_a_p[nc + (i - nfd)]
        end
    end

    out_size = dims(out_inds)
    out = zeros(T, out_size...)

    n = first(inds_d).dim

    free_dim_a = 1
    for i in 1:nfa; free_dim_a *= inds_a_p[A_is_left ? i : (nc + i)].dim; end

    cont_dim_a = n^nc
    free_dim_d = n^nfd

    a_p = _maybe_permute(A.storage, perm_a)
    
    stride_a = sum(n^i for i in 0:(nc-1); init=0)
    stride_d = sum(n^i for i in 0:(nfd-1); init=0)

    if A_is_left
        a_mat = reshape(a_p, free_dim_a, cont_dim_a)
        out_mat = reshape(out, free_dim_a, free_dim_d)
        
        for k in 1:n
            col_a = 1 + (k-1)*stride_a
            col_out = 1 + (k-1)*stride_d
            d_val = D.storage[k]
            for i in 1:free_dim_a
                out_mat[i, col_out] += a_mat[i, col_a] * d_val
            end
        end
    else
        a_mat = reshape(a_p, cont_dim_a, free_dim_a)
        out_mat = reshape(out, free_dim_d, free_dim_a)

        for i in 1:free_dim_a
            for k in 1:n
                row_a = 1 + (k-1)*stride_a
                row_out = 1 + (k-1)*stride_d
                d_val = D.storage[k]
                out_mat[row_out, i] += a_mat[row_a, i] * d_val
            end
        end
    end

    return DenseTensor(out_inds, out)
end

function contract(A::DenseTensor{TA,NA}, D::DiagTensor{TD,ND}) where {TA,TD,NA,ND}
    return _contract_dense_diag(A, D, Val(true))
end

function contract(D::DiagTensor{TD,ND}, A::DenseTensor{TA,NA}) where {TA,TD,NA,ND}
    return _contract_dense_diag(A, D, Val(false))
end

function contract(A::DiagTensor{TA,NA}, B::DiagTensor{TB,NB}) where {TA,TB,NA,NB}
    T = promote_type(TA, TB)
    inds_a = inds(A)
    inds_b = inds(B)

    perm_a, perm_b, nc, nfa, nfb = _find_contracted_free(inds_a, inds_b)

    if nc == 0
        return contract(to_dense(A), to_dense(B))
    end

    inds_a_p = ntuple(i -> inds_a[perm_a[i]], Val(NA))
    inds_b_p = ntuple(i -> inds_b[perm_b[i]], Val(NB))

    NOUT = nfa + nfb
    out_inds = ntuple(NOUT) do i
        i <= nfa ? inds_a_p[i] : inds_b_p[nc + (i - nfa)]
    end

    n = first(inds_a).dim
    
    if NOUT == 0
        val = sum(A.storage[k] * B.storage[k] for k in 1:n; init=zero(T))
        return DenseTensor((), fill(val))
    end

    out_storage = A.storage .* B.storage
    return DiagTensor(out_inds, out_storage)
end

function contract(A::AbstractTensor, B::AbstractTensor)
    contract(to_dense(A), to_dense(B))
end

function contract(A::AbstractTensor, B::DiagTensor)
    contract(to_dense(A), B)
end

function contract(A::DiagTensor, B::AbstractTensor)
    contract(A, to_dense(B))
end

Base.:*(A::AbstractTensor, B::AbstractTensor) = contract(A, B)

function contract(t::DenseTensor, c::Combiner{N}) where {N}
    ti = inds(t)
    if !c.expanding
        # fuse: c.legs[2:end] (contiguous in t) -> c.legs[1]
        original, combined = c.legs[2:end], c.legs[1]
        pos = findfirst(==(original[1]), ti)
        @assert !isnothing(pos) "Combiner's original indices not found in tensor"
        @assert pos + N - 2 <= length(ti) "Combiner's indices run past the end of inds(t)"
        @assert ti[pos:pos+N-2] == original "Indices to combine are not contiguous in inds(t) in Combiner's order"

        old_shape = size(t.storage)
        new_shape = (old_shape[1:pos-1]..., dim(combined), old_shape[pos+N-1:end]...)
        new_inds  = (ti[1:pos-1]..., combined, ti[pos+N-1:end]...)
        return DenseTensor(new_inds, reshape(t.storage, new_shape))
    else
        # expand: c.legs[1] (present in t) -> c.legs[2:end]
        combined, original = c.legs[1], c.legs[2:end]
        pos = findfirst(==(combined), ti)
        @assert !isnothing(pos) "Combined index $combined not found in tensor"

        old_shape = size(t.storage)
        new_shape = (old_shape[1:pos-1]..., dim.(original)..., old_shape[pos+1:end]...)
        new_inds  = (ti[1:pos-1]..., original..., ti[pos+1:end]...)
        return DenseTensor(new_inds, reshape(t.storage, new_shape))
    end
end

contract(c::Combiner, t::DenseTensor) = contract(t, c)