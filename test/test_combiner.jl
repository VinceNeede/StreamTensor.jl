@testset "Combiner" begin

    @testset "combine: basic construction" begin
        i1 = Index(2, :Site)
        i2 = Index(3, :Site)
        i3 = Index(4, :Site)

        c, combined = combine(i1, i2, i3)

        @test dim(combined) == 2 * 3 * 4
        @test c isa Combiner{4}                  # legs = (combined, i1, i2, i3)
        @test inds(c) == (combined, i1, i2, i3)
        @test c.expanding == false
    end

    @testset "combine: single index (degenerate case)" begin
        i1 = Index(5, :Site)
        c, combined = combine(i1)

        @test dim(combined) == 5
        @test inds(c) == (combined, i1)
    end

    @testset "combine: kwargs forwarded to Index constructor" begin
        i1 = Index(2, :Site)
        i2 = Index(3, :Site)
        c, combined = combine(i1, i2; sitetype=Nothing)

        @test dim(combined) == 6
        # combined index carries no leftover tag collision with site tags
        @test combined != i1
        @test combined != i2
    end

    @testset "dag: flips expanding flag, preserves legs" begin
        i1 = Index(2, :Site)
        i2 = Index(3, :Site)
        c, combined = combine(i1, i2)

        cd = dag(c)
        @test cd.expanding == true
        @test inds(cd) == inds(c)

        # dag is an involution
        @test dag(cd).expanding == c.expanding
        @test inds(dag(cd)) == inds(c)
    end

    @testset "fuse: contiguous block in the middle" begin
        l  = Index(2, :Link)
        i1 = Index(3, :Site)
        i2 = Index(4, :Site)
        r  = Index(5, :Link)

        data = reshape(collect(1:(2*3*4*5)), 2, 3, 4, 5)
        t = DenseTensor((l, i1, i2, r), data)

        c, combined = combine(i1, i2)
        fused = t * c

        @test inds(fused) == (l, combined, r)
        @test size(fused.storage) == (2, 12, 5)
        @test fused.storage == reshape(data, 2, 12, 5)
    end

    @testset "fuse: contiguous block at the start" begin
        i1 = Index(2, :Site)
        i2 = Index(3, :Site)
        r  = Index(4, :Link)

        data = reshape(collect(1:(2*3*4)), 2, 3, 4)
        t = DenseTensor((i1, i2, r), data)

        c, combined = combine(i1, i2)
        fused = t * c

        @test inds(fused) == (combined, r)
        @test size(fused.storage) == (6, 4)
        @test fused.storage == reshape(data, 6, 4)
    end

    @testset "fuse: contiguous block at the end" begin
        l  = Index(2, :Link)
        i1 = Index(3, :Site)
        i2 = Index(4, :Site)

        data = reshape(collect(1:(2*3*4)), 2, 3, 4)
        t = DenseTensor((l, i1, i2), data)

        c, combined = combine(i1, i2)
        fused = t * c

        @test inds(fused) == (l, combined)
        @test size(fused.storage) == (2, 12)
        @test fused.storage == reshape(data, 2, 12)
    end

    @testset "fuse: Combiner on the left of *  (c * t == t * c)" begin
        l  = Index(2, :Link)
        i1 = Index(3, :Site)
        i2 = Index(4, :Site)

        data = reshape(collect(1:(2*3*4)), 2, 3, 4)
        t = DenseTensor((l, i1, i2), data)

        c, combined = combine(i1, i2)
        fused_right = t * c
        fused_left  = c * t

        @test inds(fused_right) == inds(fused_left)
        @test fused_right.storage == fused_left.storage
    end

    @testset "expand: inverse of fuse (round trip)" begin
        l  = Index(2, :Link)
        i1 = Index(3, :Site)
        i2 = Index(4, :Site)
        r  = Index(5, :Link)

        data = reshape(collect(1:(2*3*4*5)), 2, 3, 4, 5)
        t = DenseTensor((l, i1, i2, r), data)

        c, combined = combine(i1, i2)
        fused = t * c
        expanded = fused * dag(c)

        @test inds(expanded) == (l, i1, i2, r)
        @test expanded.storage == data
    end

    @testset "expand: combined index not present throws" begin
        i1 = Index(2, :Site)
        i2 = Index(3, :Site)
        other = Index(7, :Link)

        c, combined = combine(i1, i2)
        t = DenseTensor((other,), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        @test_throws AssertionError t * dag(c)
    end

    @testset "fuse: indices not found in tensor throws" begin
        i1 = Index(2, :Site)
        i2 = Index(3, :Site)
        unrelated = Index(6, :Link)

        c, combined = combine(i1, i2)
        t = DenseTensor((unrelated,), collect(1.0:6.0))

        @test_throws AssertionError t * c
    end

    @testset "fuse: non-contiguous indices throws" begin
        i1 = Index(2, :Site)
        mid = Index(10, :Link)
        i2 = Index(3, :Site)

        data = reshape(collect(1:(2*10*3)), 2, 10, 3)
        t = DenseTensor((i1, mid, i2), data)

        c, combined = combine(i1, i2)   # i1, i2 are not adjacent in t

        @test_throws AssertionError t * c
    end

    @testset "fuse: wrong order (contiguous but reversed) throws" begin
        l  = Index(2, :Link)
        i1 = Index(3, :Site)
        i2 = Index(4, :Site)

        data = reshape(collect(1:(2*3*4)), 2, 3, 4)
        t = DenseTensor((l, i1, i2), data)

        # combine built with the reversed order relative to inds(t)
        c, combined = combine(i2, i1)

        @test_throws AssertionError t * c
    end

    @testset "fuse then expand restores MPO-MPS zip-up shape (3 indices)" begin
        # mimics fusing (link_H_right, link_ψ_right) after R * H[i] * ψ[i]
        s   = Index(2, :Site)
        lh  = Index(3, :Link)
        lp  = Index(4, :Link)

        data = reshape(collect(1:(2*3*4)), 2, 3, 4)
        t = DenseTensor((s, lh, lp), data)

        c, combined = combine(lh, lp)
        fused = t * c
        @test inds(fused) == (s, combined)
        @test size(fused.storage) == (2, 12)

        restored = fused * dag(c)
        @test inds(restored) == (s, lh, lp)
        @test restored.storage == data
    end

end