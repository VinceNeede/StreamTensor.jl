@testset "Tensor" begin
    @testset "DenseTensor construction" begin
        i = Index(2, :Site)
        j = Index(3, :Link)
        A = DenseTensor((i, j), rand(2, 3))
        @test size(A) == (2, 3)
        @test ndims(A) == 2
        @test eltype(A) == Float64
    end

    @testset "DenseTensor shape mismatch" begin
        i = Index(2, :Site)
        j = Index(3, :Link)
        @test_throws AssertionError DenseTensor((i, j), rand(3, 3))
    end

    @testset "DeltaTensor construction" begin
        i = Index(3, :Site)
        j = Index(3, :Link)
        δ = DeltaTensor((i, j))
        @test ndims(δ) == 2
    end

    @testset "DeltaTensor dim mismatch" begin
        i = Index(2, :Site)
        j = Index(3, :Link)
        @test_throws AssertionError DeltaTensor((i, j))
    end

    @testset "MPSTensor" begin
        l = Index(4, :Link)
        s = Index(2, :Site)
        r = Index(4, :Link)
        ψ = MPSTensor(rand(4, 2, 4), l, s, r)
        @test siteind(ψ) == s
        @test linkinds(ψ) == (l, r)
        @test size(ψ) == (4, 2, 4)
    end

    @testset "MPOTensor" begin
        l  = Index(4, :Link)
        r  = Index(4, :Link)
        si = Index(2, :Site)
        so = Index(2, :Site)
        W = MPOTensor(rand(4, 2, 2, 4), l, so, si, r)
        @test inds(W) == (l, so, si, r)
        @test siteinds(W) == (so, si)
        @test linkinds(W) == (l, r)
        @test size(W) == (4, 2, 2, 4)
    end
end