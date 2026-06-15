@testset "MPS" begin

    @testset "construction" begin
        s1 = Index(2, :Site)
        s2 = Index(2, :Site)
        s3 = Index(2, :Site)

        l1 = Index(1, :Link)
        l2 = Index(4, :Link)
        l3 = Index(4, :Link)
        l4 = Index(1, :Link)

        t1 = MPSTensor(randn(1, 2, 4), l1, s1, l2)
        t2 = MPSTensor(randn(4, 2, 4), l2, s2, l3)
        t3 = MPSTensor(randn(4, 2, 1), l3, s3, l4)

        mps = MPS([t1, t2, t3])

        @test length(mps) == 3
        @test mps.llim == 0
        @test mps.rlim == 4
    end

    @testset "bond index mismatch" begin
        s1 = Index(2, :Site)
        s2 = Index(2, :Site)

        l1 = Index(1, :Link)
        l2 = Index(4, :Link)
        l3 = Index(4, :Link)  # different id from l2
        l4 = Index(1, :Link)

        t1 = MPSTensor(randn(1, 2, 4), l1, s1, l2)
        t2 = MPSTensor(randn(4, 2, 1), l3, s2, l4)  # l3 != l2

        @test_throws AssertionError MPS([t1, t2])
    end

    @testset "random_mps" begin
        L = 10
        sites = [Index(2, :Site) for _ in 1:L]
        mps = random_mps(Float64, sites, 4)
        mid = L ÷ 2

        @test length(mps) == L
        @test nsites(mps) == L

        # check bond dimensions don't exceed linkdim
        for i in 1:L-1
            @test mps[i].right.dim <= 4
        end

        # check bond connectivity
        for i in 1:L-1
            @test mps[i].right == mps[i+1].left
        end

        # check center is at mid+1
        @test mps.llim == mid
        @test mps.rlim == mid + 2
        @test orthocenter(mps) == mid + 1

        # check bond dimensions grow and shrink symmetrically
        for i in 1:mid-1
            @test mps[i].right.dim <= mps[i+1].right.dim
        end
        for i in mid+1:L-1
            @test mps[i].right.dim >= mps[i+1].right.dim
        end

        # check left orthogonality of sites 1..mid
        for i in 1:mid
            χl, d, χr = size(mps[i].storage)
            U_mat = reshape(mps[i].storage, χl * d, χr)
            @test U_mat' * U_mat ≈ I(χr) atol=1e-12
        end

        # check right orthogonality of sites mid+2..L
        for i in mid+2:L
            χl, d, χr = size(mps[i].storage)
            V_mat = reshape(mps[i].storage, χl, d * χr)
            @test V_mat * V_mat' ≈ I(χl) atol=1e-12
        end

        # check normalization
        @test inner(mps, mps) ≈ 1.0
    end

    @testset "orthogonalize!" begin
        sites = [Index(2, :Site) for _ in 1:5]
        mps = random_mps(Float64, sites, 4)

        # move center from site 5 to site 3
        orthogonalize!(mps, 3)

        @test mps.llim == 2
        @test mps.rlim == 4

        # check left orthogonality of sites 1,2
        for i in 1:2
            χl, d, χr = size(mps[i].storage)
            U_mat = reshape(mps[i].storage, χl * d, χr)
            @test U_mat' * U_mat ≈ I(χr) atol=1e-12
        end

        # check right orthogonality of sites 4,5
        for i in 4:5
            χl, d, χr = size(mps[i].storage)
            V_mat = reshape(mps[i].storage, χl, d * χr)
            @test V_mat * V_mat' ≈ I(χl) atol=1e-12
        end
    end

    @testset "orthogonalize non-destructive" begin
        sites = [Index(2, :Site) for _ in 1:5]
        mps   = random_mps(Float64, sites, 4)
        mid   = 5 ÷ 2  # = 2

        # original has center at mid+1 = 3
        @test mps.llim == mid
        @test mps.rlim == mid + 2

        mps2 = orthogonalize(mps, 3)

        # original unchanged
        @test mps.llim == mid
        @test mps.rlim == mid + 2

        # copy has new center
        @test mps2.llim == 2
        @test mps2.rlim == 4
        @test orthocenter(mps2) == 3
    end

    @testset "orthocenter" begin
        sites = [Index(2, :Site) for _ in 1:5]
        mps   = random_mps(Float64, sites, 4)
        orthogonalize!(mps, 3)
        @test orthocenter(mps) == 3
    end

    @testset "copy" begin
        sites = [Index(2, :Site) for _ in 1:5]
        mps   = random_mps(Float64, sites, 4)
        mps2  = copy(mps)

        # modifying tensor reference in copy doesn't affect original
        s     = Index(2, :Site)
        l     = mps[1].left
        r     = mps[1].right
        mps2.tensors[1] = MPSTensor(randn(1, 2, r.dim), l, s, r)
        @test mps[1].storage !== mps2[1].storage
    end

    @testset "inner" begin
        sites = [Index(2, :Site) for _ in 1:10]
        ψ = random_mps(Float64, sites, 4)
        orthogonalize!(ψ, 1)

        norm_sq = sum(abs2, ψ[1].storage)
        @test inner(ψ, ψ) ≈ norm_sq atol=1e-10
    end

    @testset "expect" begin

        σz = [1.0  0.0; 0.0 -1.0]
        σy = [0.0 -1.0im; 1.0im 0.0]

        @testset "σz on |0⟩ product state → all +1" begin
            sites = [Index(2, :Site) for _ in 1:4]
            up    = [1.0, 0.0]

            tensors = Vector{MPSTensor{Float64, Array{Float64,3}}}(undef, 4)
            left = Index(1, :Link)
            for i in 1:4
                right = Index(1, :Link)
                tensors[i] = MPSTensor(reshape(up, 1, 2, 1), left, sites[i], right)
                left = right
            end
            ψ = MPS(tensors, 0, 5)

            for i in 1:4
                @test expect(ψ, σz, i) ≈ 1.0 atol=1e-10
            end
        end

        @testset "σz on |+⟩ product state → all 0" begin
            sites = [Index(2, :Site) for _ in 1:4]
            plus  = [1.0, 1.0] / sqrt(2)

            tensors = Vector{MPSTensor{Float64, Array{Float64,3}}}(undef, 4)
            left = Index(1, :Link)
            for i in 1:4
                right = Index(1, :Link)
                tensors[i] = MPSTensor(reshape(plus, 1, 2, 1), left, sites[i], right)
                left = right
            end
            ψ = MPS(tensors, 0, 5)

            for i in 1:4
                @test expect(ψ, σz, i) ≈ 0.0 atol=1e-10
            end
        end

        @testset "σy on |+⟩ product state → all 0 (real part)" begin
            # |+⟩ = (|0⟩ + |1⟩)/√2
            # ⟨+|σy|+⟩ = 0 since σy is purely imaginary and |+⟩ is real
            sites = [Index(2, :Site) for _ in 1:4]
            plus  = [1.0, 1.0] / sqrt(2)

            tensors = Vector{MPSTensor{ComplexF64, Array{ComplexF64,3}}}(undef, 4)
            left = Index(1, :Link)
            for i in 1:4
                right = Index(1, :Link)
                tensors[i] = MPSTensor(reshape(ComplexF64.(plus), 1, 2, 1), left, sites[i], right)
                left = right
            end
            ψ = MPS(tensors, 0, 5)

            for i in 1:4
                result = expect(ψ, σy, i)
                @test result isa Float64       # real() should return Float64
                @test result ≈ 0.0 atol=1e-10
            end
        end

    end
end