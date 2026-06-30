@testset "Contraction" begin
    @testset "matrix multiplication" begin
        i = Index(2, :Site)
        j = Index(3, :Link)
        k = Index(4, :Link)
        A = DenseTensor((i, j), rand(2, 3))
        B = DenseTensor((j, k), rand(3, 4))
        C = contract(A, B)
        @test inds(C) == (i, k)
        @test size(C) == (2, 4)
        # correctness: compare with plain matrix multiply
        @test C.storage ≈ A.storage * B.storage
    end

    @testset "outer product" begin
        i = Index(2, :Site)
        j = Index(3, :Link)
        A = DenseTensor((i,), rand(2))
        B = DenseTensor((j,), rand(3))
        C = contract(A, B)
        @test size(C) == (2, 3)
        @test inds(C) == (i, j)
    end

    @testset "full contraction to scalar" begin
        i = Index(2, :Site)
        j = Index(3, :Link)
        A = DenseTensor((i, j), rand(2, 3))
        B = DenseTensor((i, j), rand(2, 3))
        C = contract(A, B)
        @test ndims(C) == 0
        @test C.storage[] ≈ sum(A.storage .* B.storage)
    end

    @testset "index order in output" begin
        i = Index(2, :Site)
        j = Index(3, :Link)
        k = Index(4, :Link)
        l = Index(5, :Link)
        A = DenseTensor((i, j, k), rand(2, 3, 4))
        B = DenseTensor((k, l), rand(4, 5))
        C = contract(A, B)
        # free indices of A come first, then free of B
        @test inds(C) == (i, j, l)
        @test size(C) == (2, 3, 5)
    end

    @testset "correctness against einsum" begin
        i = Index(2, :Site)
        j = Index(3, :Link)
        k = Index(4, :Link)
        A = DenseTensor((i, j), rand(2, 3))
        B = DenseTensor((j, k), rand(3, 4))
        C = contract(A, B)
        # manual einsum: C[a,c] = sum_b A[a,b] * B[b,c]
        C_ref = zeros(2, 4)
        for a in 1:2, b in 1:3, c in 1:4
            C_ref[a,c] += A.storage[a,b] * B.storage[b,c]
        end
        @test C.storage ≈ C_ref
    end
end

@testset "DiagTensor contraction" begin
    @testset "rank-2 diagonal × dense: scales rows" begin
        i = Index(3, :Link)
        j = Index(4, :Link)
        k = Index(3, :Link)  # same dim as i, will be the free diag index

        # D is diagonal: D[m,n] = d[m] * δ_mn, contracts i with A
        d = [2.0, 3.0, 4.0]
        D = DiagTensor((i, k), d)
        A = DenseTensor((i, j), rand(3, 4))

        C = contract(A, D)

        @test inds(C) == (j, k)
        @test size(C) == (4, 3)

        # C[b, n] = sum_m A[m,b] * d[m] * δ_mn = A[n,b] * d[n]
        C_ref = zeros(4, 3)
        for n in 1:3, b in 1:4
            C_ref[b, n] = A.storage[n, b] * d[n]
        end
        @test C.storage ≈ C_ref
    end

    @testset "rank-2 delta: acts as index rename" begin
        i = Index(3, :Site)
        j = Index(4, :Link)
        k = Index(3, :Site)  # replacement index

        δ = DeltaTensor((i, k))   # ones on diagonal
        A = DenseTensor((i, j), rand(3, 4))

        C = contract(A, δ)

        # delta just renames i → k, data unchanged
        @test inds(C) == (j, k)
        @test size(C) == (4, 3)
        @test C.storage ≈ A.storage'
    end

    @testset "rank-3 diagonal contracts two indices of A" begin
        i = Index(3, :Link)
        j = Index(3, :Link)
        k = Index(3, :Link)

        d = [1.0, 2.0, 3.0]
        D = DiagTensor((i, j, k), d)   # D[m,n,p] = d[m] * δ_mnp
        A = DenseTensor((i, j), rand(3, 3))

        C = contract(A, D)

        # contracts i and j, k is free
        # C[k] = sum_{i=j=k} A[i,j] * d[k] = A[k,k] * d[k]
        @test size(C) == (3,)
        C_ref = [A.storage[k,k] * d[k] for k in 1:3]
        @test C.storage ≈ C_ref
    end

    @testset "commutativity of argument order" begin
        i = Index(3, :Link)
        j = Index(4, :Link)
        k = Index(3, :Link)

        d = [1.0, 2.0, 3.0]
        D = DiagTensor((i, k), d)
        A = DenseTensor((i, j), rand(3, 4))

        C1 = contract(A, D)
        C2 = contract(D, A)

        @test inds(C1) == (j, k)
        @test inds(C2) == (k, j)
        @test C1.storage ≈ transpose(C2.storage)
    end

    @testset "full contraction to diagonal" begin
        i = Index(3, :Link)
        j = Index(3, :Link)

        d = [2.0, 3.0, 4.0]
        D = DiagTensor((i, j), d)
        A = DenseTensor((i, j), rand(3, 3))

        C = contract(A, D)

        # all indices contracted: scalar = sum_k d[k] * A[k,k]
        @test ndims(C) == 0
        @test C.storage[] ≈ sum(d[k] * A.storage[k,k] for k in 1:3)
    end

    @testset "DiagTensor × DiagTensor: element-wise product" begin
        i = Index(3, :Link)
        j = Index(3, :Link)

        d1 = [1.0, 2.0, 3.0]
        d2 = [2.0, 3.0, 4.0]
        D1 = DiagTensor((i, j), d1)
        D2 = DiagTensor((j, i), d2)

        C = contract(D1, D2)

        # fully contracted, returns scalar DenseTensor
        @test ndims(C) == 0
        @test C.storage[] ≈ sum(d1 .* d2)
        
        k = Index(3, :Link)
        D3 = DiagTensor((j, k), d2)
        C2 = contract(D1, D3)
        
        # partially contracted, returns DiagTensor
        @test C2 isa DiagTensor
        @test inds(C2) == (i, k)
        @test C2.storage ≈ d1 .* d2
    end

    @testset "Mixed eltype contraction" begin
        @testset "DenseTensor{ComplexF64} × DenseTensor{Float64}: matrix multiply" begin
            i = Index(2, :Site)
            j = Index(3, :Link)
            k = Index(4, :Link)
            A = DenseTensor((i, j), rand(ComplexF64, 2, 3))
            B = DenseTensor((j, k), rand(Float64, 3, 4))
            C = contract(A, B)

            @test eltype(C) == ComplexF64
            @test inds(C) == (i, k)
            @test size(C) == (2, 4)
            @test C.storage ≈ A.storage * B.storage
        end

        @testset "DenseTensor{Float64} × DenseTensor{ComplexF64}: matrix multiply (order swapped)" begin
            i = Index(2, :Site)
            j = Index(3, :Link)
            k = Index(4, :Link)
            A = DenseTensor((i, j), rand(Float64, 2, 3))
            B = DenseTensor((j, k), rand(ComplexF64, 3, 4))
            C = contract(A, B)

            @test eltype(C) == ComplexF64
            @test inds(C) == (i, k)
            @test C.storage ≈ A.storage * B.storage
        end

        @testset "DenseTensor{ComplexF64} × DiagTensor{Float64}: scales columns" begin
            # mirrors the S (real singular values) * V (complex tensor) pattern from
            # _shift_center_left!/_shift_center_right! that triggered the original
            # infinite-recursion bug
            i = Index(3, :Link)
            j = Index(4, :Link)
            k = Index(3, :Link)

            d = [2.0, 3.0, 4.0]
            D = DiagTensor((i, k), d)             # real singular values
            A = DenseTensor((i, j), rand(ComplexF64, 3, 4))

            C = contract(A, D)

            @test eltype(C) == ComplexF64
            @test inds(C) == (j, k)
            @test size(C) == (4, 3)

            C_ref = zeros(ComplexF64, 4, 3)
            for n in 1:3, b in 1:4
                C_ref[b, n] = A.storage[n, b] * d[n]
            end
            @test C.storage ≈ C_ref
        end

        @testset "DiagTensor{Float64} × DenseTensor{ComplexF64}: order swapped" begin
            i = Index(3, :Link)
            j = Index(4, :Link)
            k = Index(3, :Link)

            d = [2.0, 3.0, 4.0]
            D = DiagTensor((i, k), d)
            A = DenseTensor((i, j), rand(ComplexF64, 3, 4))

            C1 = contract(A, D)
            C2 = contract(D, A)

            @test eltype(C2) == ComplexF64
            @test inds(C1) == (j, k)
            @test inds(C2) == (k, j)
            @test C1.storage ≈ transpose(C2.storage)
        end

        @testset "DenseTensor{ComplexF64} × DiagTensor{Float64}: outer product (nc==0) path" begin
            i = Index(2, :Site)
            j = Index(3, :Link)
            k = Index(3, :Link)

            d = [1.0, 2.0, 3.0]
            D = DiagTensor((j, k), d)
            A = DenseTensor((i,), rand(ComplexF64, 2))

            C = contract(A, D)

            @test eltype(C) == ComplexF64
            @test inds(C) == (i, j, k)
        end

        @testset "DenseTensor{ComplexF64} × DiagTensor{Float64}: full contraction to scalar" begin
            i = Index(3, :Link)
            j = Index(3, :Link)

            d = [2.0, 3.0, 4.0]
            D = DiagTensor((i, j), d)
            A = DenseTensor((i, j), rand(ComplexF64, 3, 3))

            C = contract(A, D)

            @test ndims(C) == 0
            @test eltype(C) == ComplexF64
            @test C.storage[] ≈ sum(d[k] * A.storage[k,k] for k in 1:3)
        end

        @testset "DiagTensor{ComplexF64} × DiagTensor{Float64}: element-wise product" begin
            i = Index(3, :Link)
            j = Index(3, :Link)
            k = Index(3, :Link)

            d1 = ComplexF64.([1.0, 2.0, 3.0]) .* im
            d2 = [2.0, 3.0, 4.0]
            D1 = DiagTensor((i, j), d1)
            D2 = DiagTensor((j, k), d2)

            C = contract(D1, D2)

            @test C isa DiagTensor
            @test eltype(C) == ComplexF64
            @test inds(C) == (i, k)
            @test C.storage ≈ d1 .* d2
        end

        @testset "DiagTensor{ComplexF64} × DiagTensor{Float64}: full contraction to scalar" begin
            i = Index(3, :Link)
            j = Index(3, :Link)

            d1 = ComplexF64.([1.0, 2.0, 3.0]) .* im
            d2 = [2.0, 3.0, 4.0]
            D1 = DiagTensor((i, j), d1)
            D2 = DiagTensor((j, i), d2)

            C = contract(D1, D2)

            @test ndims(C) == 0
            @test eltype(C) == ComplexF64
            @test C.storage[] ≈ sum(d1 .* d2)
        end

        @testset "no infinite recursion regression: repeated DenseTensor{Complex} * DiagTensor{Float64}" begin
            # the original bug manifested as unbounded recursion through
            # contract(A::AbstractTensor, B::DiagTensor) = contract(to_dense(A), B)
            # when eltype(A) != eltype(B). Chaining several contractions in a row
            # is a regression guard against that recursion reappearing.
            i = Index(2, :Link)
            j = Index(2, :Link)
            k = Index(2, :Link)
            l = Index(2, :Link)

            d = [1.0, 2.0]
            D1 = DiagTensor((i, j), d)
            D2 = DiagTensor((k, l), d)
            A = DenseTensor((i,), rand(ComplexF64, 2))

            B = contract(A, D1)         # (j,)
            B = DenseTensor((k,), B.storage)
            C = contract(B, D2)         # (l,)

            @test eltype(C) == ComplexF64
            @test inds(C) == (l,)
        end
    end
end