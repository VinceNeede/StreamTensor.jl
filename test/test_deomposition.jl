import LinearAlgebra: I, diagm

@testset "Decomposition" begin

    @testset "DenseTensor SVD" begin
        @testset "reconstruction" begin
            i = Index(4, :Site)
            j = Index(3, :Link)
            k = Index(5, :Link)
            A = DenseTensor((i, j, k), rand(4, 3, 5))
            U, S, V = svd(A, (i, j))

            # reconstruct via contraction
            US  = contract(U, S)
            USV = contract(US, V)

            @test size(USV) == size(A)
            @test Set(inds(USV)) == Set(inds(A))
            @test USV.storage ≈ A.storage
        end

        @testset "left orthogonality" begin
            i = Index(4, :Site)
            j = Index(5, :Link)
            A = DenseTensor((i, j), rand(4, 5))
            U, S, V = svd(A, (i,))

            # U'U = I — work directly on the matrix
            χl = size(U, 1)
            χ  = size(U, 2)
            U_mat = reshape(U.storage, χl, χ)
            @test U_mat' * U_mat ≈ I(χ)
        end

        @testset "maxdim truncation" begin
            i = Index(6, :Site)
            j = Index(6, :Link)
            A = DenseTensor((i, j), rand(6, 6))
            U, S, V = svd(A, (i,); maxdim=3)

            @test length(S.storage) == 3
            @test size(U, 2) == 3
            @test size(V, 1) == 3
        end

        @testset "cutoff truncation" begin
            i = Index(4, :Site)
            j = Index(4, :Link)
            # construct matrix with known singular values
            A_mat = diagm([1.0, 0.5, 1e-10, 1e-12])
            A = DenseTensor((i, j), A_mat)
            U, S, V = svd(A, (i,); cutoff=1e-8)

            @test length(S.storage) == 2
        end

        @testset "index connectivity" begin
            i = Index(3, :Site)
            j = Index(4, :Link)
            A = DenseTensor((i, j), rand(3, 4))
            U, S, V = svd(A, (i,))

            # bond indices connect U→S→V
            bond_u = last(inds(U))
            bond_v = first(inds(V))
            @test inds(S) == (bond_u, bond_v)
            @test bond_u != bond_v
        end
    end

    @testset "MPSTensor SVD" begin
        @testset "left orthogonal reconstruction" begin
            l = Index(4, :Link)
            s = Index(2, :Site)
            r = Index(4, :Link)
            ψ = MPSTensor(rand(4, 2, 4), l, s, r)
            U, S, V = svd(ψ; direction=LeftOrthogonal)

            US  = contract(to_dense(U), S)
            USV = contract(US, V)

            @test Set(inds(USV)) == Set(inds(ψ))
            @test USV.storage ≈ reshape(ψ.storage, 4*2, 4) |> x -> reshape(x, 4, 2, 4)
        end

        @testset "right orthogonal reconstruction" begin
            l = Index(4, :Link)
            s = Index(2, :Site)
            r = Index(4, :Link)
            ψ = MPSTensor(rand(4, 2, 4), l, s, r)
            U, S, V = svd(ψ; direction=RightOrthogonal)

            US  = contract(U, S)
            USV = contract(US, to_dense(V))

            @test Set(inds(USV)) == Set(inds(ψ))
        end

        @testset "left orthogonal U" begin
            l = Index(4, :Link)
            s = Index(2, :Site)
            r = Index(6, :Link)
            ψ = MPSTensor(rand(4, 2, 6), l, s, r)
            U, S, V = svd(ψ; direction=LeftOrthogonal)

            # U should be an MPSTensor
            @test U isa MPSTensor
            @test U.left == l
            @test U.site == s

            # left orthogonality: reshape U as (χl*d, χ), then U'U = I
            χl, d, χ = size(U.storage)
            U_mat = reshape(U.storage, χl * d, χ)
            @test U_mat' * U_mat ≈ I(χ)
        end

        @testset "right orthogonal V" begin
            l = Index(6, :Link)
            s = Index(2, :Site)
            r = Index(4, :Link)
            ψ = MPSTensor(rand(6, 2, 4), l, s, r)
            U, S, V = svd(ψ; direction=RightOrthogonal)

            # V should be an MPSTensor
            @test V isa MPSTensor
            @test V.site == s
            @test V.right == r

            # right orthogonality: reshape V as (χ, d*χr), then VV' = I
            χ, d, χr = size(V.storage)
            V_mat = reshape(V.storage, χ, d * χr)
            @test V_mat * V_mat' ≈ I(χ)
        end

        @testset "maxdim" begin
            l = Index(4, :Link)
            s = Index(2, :Site)
            r = Index(4, :Link)
            ψ = MPSTensor(rand(4, 2, 4), l, s, r)
            U, S, V = svd(ψ; maxdim=3)

            @test length(S.storage) <= 3
        end
    end

    @testset "MPOTensor SVD" begin
        @testset "left orthogonal reconstruction" begin
            l  = Index(4, :Link)
            r  = Index(4, :Link)
            si = Index(2, :Site)
            so = Index(2, :Site)
            W  = MPOTensor(rand(4, 2, 2, 4), l, so, si, r)
            U, S, V = svd(W; direction=LeftOrthogonal)

            US  = contract(U, S)
            USV = contract(US, to_dense(V))

            @test Set(inds(USV)) == Set(inds(W))
        end

        @testset "right orthogonal reconstruction" begin
            l  = Index(4, :Link)
            r  = Index(4, :Link)
            si = Index(2, :Site)
            so = Index(2, :Site)
            W  = MPOTensor(rand(4, 2, 2, 4), l, so, si, r)
            U, S, V = svd(W; direction=RightOrthogonal)

            US  = contract(U, S)
            USV = contract(US, to_dense(V))

            @test Set(inds(USV)) == Set(inds(W))
        end

        @testset "left orthogonal U" begin
            l  = Index(4, :Link)
            r  = Index(6, :Link)
            si = Index(2, :Site)
            so = Index(2, :Site)
            W  = MPOTensor(rand(4, 2, 2, 6), l, so, si, r)
            U, S, V = svd(W; direction=LeftOrthogonal)

            @test U isa MPOTensor
            @test U.left == l
            @test U.site_in == si
            @test U.site_out == so

            χl, do_, di, χ = size(U.storage)
            U_mat = reshape(U.storage, χl * do_ * di, χ)
            @test U_mat' * U_mat ≈ I(χ)
        end

        @testset "right orthogonal V" begin
            l  = Index(6, :Link)
            r  = Index(4, :Link)
            si = Index(2, :Site)
            so = Index(2, :Site)
            W  = MPOTensor(rand(6, 2, 2, 4), l, so, si, r)
            U, S, V = svd(W; direction=RightOrthogonal)

            @test V isa MPOTensor
            @test V.right == r
            @test V.site_in == si
            @test V.site_out == so

            χ, do_, di, χr = size(V.storage)
            V_mat = reshape(V.storage, χ, do_ * di * χr)
            @test V_mat * V_mat' ≈ I(χ)
        end
    end
end