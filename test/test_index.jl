@testset "Index" begin
    @testset "construction" begin
        i = Index(3, :Site)
        @test i.dim == 3
        @test hastag(i, :Site)

        j = Index(4, :Link)
        @test j.dim == 4
        @test hastag(j, :Link)
    end

    @testset "uniqueness" begin
        i = Index(3, :Site)
        j = Index(3, :Site)   # same dim and tag, different id
        @test i != j
    end

    @testset "equality" begin
        i = Index(3, :Site)
        @test i == i
    end

    @testset "hash consistency" begin
        i = Index(3, :Site)
        @test hash(i) == hash(i)

        # usable as dict key
        d = Dict(i => 1)
        @test d[i] == 1
    end

    @testset "invalid dim" begin
        @test_throws AssertionError Index(0, :Site)
        @test_throws AssertionError Index(-1, :Link)
    end

    @testset "adjoint" begin
        i = Index(3, :Site)
        ip = i'
        @test ip.dim == i.dim
        @test tags(ip) == tags(i)
        @test ip != i
    end
end