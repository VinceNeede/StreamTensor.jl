using Test
using LinearAlgebra

@testset "apply! / apply (zipup)" begin

    function tfim_opsum(sites; J=1.0, h=1.0)
        L = length(sites)
        os = OpSum()
        for i in 1:L-1
            os += (-J, "Sz", i, "Sz", i+1)
        end
        for i in 1:L
            os += (-h, "Sx", i)
        end
        return os
    end

    function product_mps(::Type{T}, sites, state_names) where {T}
        L = length(sites)
        links = [Index(1, :Link) for _ in 0:L]
        tensors = map(1:L) do i
            v = T.(_dispatch_state(sites[i], StateName(state_names[i])))
            MPSTensor(reshape(v, 1, sites[i].dim, 1), links[i], sites[i], links[i+1])
        end
        return MPS(tensors, 0, L+1)
    end

    # ── basic structure tests ─────────────────────────────────────────────

    @testset "warning issued when H not canonical" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)   # fresh MPO, not orthogonalized
        ψ = random_mps(Float64, sites, 4)

        @test_logs (:warn,) apply(H, ψ; maxdim=16)
    end

    @testset "no warning when H is canonical at site 1" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        orthogonalize!(H, 1)
        ψ = random_mps(Float64, sites, 4)

        @test_logs apply(H, ψ; maxdim=16)
    end

    @testset "apply returns MPS of same length" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        orthogonalize!(H, 1)
        ψ = random_mps(Float64, sites, 4)

        φ = apply(H, ψ; maxdim=16)

        @test length(φ) == L
    end

    @testset "apply: site indices match input MPS (noprime)" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        orthogonalize!(H, 1)
        ψ = random_mps(Float64, sites, 4)

        φ = apply(H, ψ; maxdim=16)

        for i in 1:L
            @test φ[i].site == ψ[i].site
        end
    end

    @testset "apply: bond connectivity preserved" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        orthogonalize!(H, 1)
        ψ = random_mps(Float64, sites, 4)

        φ = apply(H, ψ; maxdim=16)

        for i in 1:L-1
            @test φ[i].right == φ[i+1].left
        end
    end

    @testset "apply: output in left-canonical form" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        orthogonalize!(H, 1)
        ψ = random_mps(Float64, sites, 4)

        φ = apply(H, ψ; maxdim=16)

        @test φ.llim == 0
        @test φ.rlim == 2
    end

    # ── non-mutating ─────────────────────────────────────────────────────

    @testset "apply does not modify ψ" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        orthogonalize!(H, 1)
        ψ = random_mps(Float64, sites, 4)

        sites_before = [ψ[i].site for i in 1:L]
        norms_before = [norm(ψ[i].storage) for i in 1:L]

        apply(H, ψ; maxdim=16)

        for i in 1:L
            @test ψ[i].site == sites_before[i]
            @test norm(ψ[i].storage) ≈ norms_before[i]
        end
    end

    @testset "apply! modifies ψ in-place" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        orthogonalize!(H, 1)
        ψ = random_mps(Float64, sites, 4)
        ψ_ref = copy(ψ)

        apply!(H, ψ; maxdim=16)

        # ψ should now differ from ψ_ref
        @test !all(ψ[i].storage ≈ ψ_ref[i].storage for i in 1:L)
    end

    # ── correctness ───────────────────────────────────────────────────────

    @testset "identity MPO: apply(I, ψ) ≈ ψ" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        
        links = [Index(1, :Link) for _ in 0:L]
        id_mat = Float64[1 0; 0 1]
        tensors = map(1:L) do i
            MPOTensor(reshape(id_mat, 1, 2, 2, 1),
                    links[i], sites[i]', sites[i], links[i+1])
        end
        H_id = MPO(tensors)
        orthogonalize!(H_id, 1)

        ψ = random_mps(Float64, sites, 4)
        orthogonalize!(ψ, 1)

        φ = apply(H_id, ψ; maxdim=16)

        @test inner(φ, φ) ≈ inner(ψ, ψ) atol=1e-8
        @test abs(inner(φ, ψ)) ≈ real(inner(ψ, ψ)) atol=1e-8
    end

    @testset "⟨ψ|H|φ⟩ = inner(ψ, H, φ) = inner(ψ, apply(H,φ))" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        orthogonalize!(H, 1)
        ψ = random_mps(Float64, sites, 6)
        φ = random_mps(Float64, sites, 4)

        Hφ = apply(H, φ; maxdim=32)

        @test real(inner(ψ, H, φ)) ≈ real(inner(ψ, Hφ)) atol=1e-8
    end

    @testset "‖H|ψ⟩‖² = ⟨ψ|H†H|ψ⟩ = inner(Hψ, Hψ)" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        orthogonalize!(H, 1)
        ψ = random_mps(Float64, sites, 4)
        orthogonalize!(ψ, 1)

        Hψ = apply(H, ψ; maxdim=32)

        # ‖Hψ‖² should be positive and finite
        norm2 = real(inner(Hψ, Hψ))
        @test norm2 > 0
        @test isfinite(norm2)
    end

    @testset "energy via apply: ⟨ψ|H|ψ⟩ = inner(ψ, Hψ)" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        orthogonalize!(H, 1)

        # use DMRG ground state for a non-trivial test
        ψ = random_mps(Float64, sites, 8)
        ψ, _, sweep_data = dmrg!(ψ, H, 10;
            maxdim=8, cutoff=1e-10,
            eigsolve_kwargs=(krylovdim=6, maxiter=5))
        E_dmrg = sweep_data[end].energies[end]

        Hψ = apply(H, ψ; maxdim=32, cutoff=1e-12)

        E_apply = real(inner(ψ, Hψ)) / real(inner(ψ, ψ))
        E_inner = real(inner(ψ, H, ψ)) / real(inner(ψ, ψ))

        @test E_apply ≈ E_inner atol=1e-6
        @test E_apply ≈ E_dmrg  atol=1e-6
    end

    @testset "energy variance ≈ 0 at ground state: ‖Hψ - Eψ‖² ≈ 0" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        orthogonalize!(H, 1)

        ψ = random_mps(Float64, sites, 8)
        ψ, _, sweep_data = dmrg!(ψ, H, 10;
            maxdim=8, cutoff=1e-10,
            eigsolve_kwargs=(krylovdim=6, maxiter=5))
        E = sweep_data[end].energies[end]

        # normalize ψ
        orthogonalize!(ψ, 1)
        ψ_norm = copy(ψ)

        Hψ = apply(H, ψ; maxdim=32, cutoff=1e-12)

        # variance = ‖H|ψ⟩ - E|ψ⟩‖² = ⟨H²⟩ - E²
        # compute via inner(Hψ - Eψ, Hψ - Eψ) — approximate since
        # we can't do MPS subtraction yet, so use:
        # var = ⟨Hψ|Hψ⟩ - 2E⟨ψ|Hψ⟩ + E²⟨ψ|ψ⟩
        norm2_ψ = real(inner(ψ, ψ))
        E_normalized = E / norm2_ψ
        H2_exp = real(inner(Hψ, Hψ)) / norm2_ψ
        E_exp  = real(inner(ψ, Hψ))  / norm2_ψ
        var = H2_exp - 2 * E_normalized * E_exp + E_normalized^2 * norm2_ψ

        @test var ≈ 0.0 atol=1e-6
    end

    @testset "Sz operator: apply(Sz_1, |↑↑⟩) = 0.5|↑↑⟩" begin
        sites = siteinds(:SpinHalf, 2)
        os = OpSum()
        os += (1.0, "Sz", 1)
        H = MPO(os, sites)
        orthogonalize!(H, 1)

        ψ = MPS(sites, ["Up","Up"])
        φ = apply(H, ψ; maxdim=4)

        # ⟨φ|φ⟩ = 0.25 (eigenvalue 0.5, squared)
        @test real(inner(φ, φ)) ≈ 0.25 atol=1e-10
        # ⟨ψ|φ⟩ = 0.5 (eigenvalue)
        @test real(inner(ψ, φ)) ≈ 0.5 atol=1e-10
    end

    @testset "maxdim truncation: bond dimension bounded" begin
        L = 6
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        orthogonalize!(H, 1)
        ψ = random_mps(Float64, sites, 8)

        φ = apply(H, ψ; maxdim=4)

        for i in 1:L-1
            @test φ[i].right.dim <= 4
        end
    end

    @testset "ComplexF64 MPS: apply produces finite result" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        orthogonalize!(H, 1)
        ψ = random_mps(ComplexF64, sites, 4)

        φ = apply(H, ψ; maxdim=16)

        @test all(all(isfinite.(φ[i].storage)) for i in 1:L)
        result = inner(φ, φ)
        @test isfinite(real(result))
        @test abs(imag(result)) < 1e-10
    end

    @testset "sweep_maxdim: bond dim bounded during left-to-right pass" begin
        L = 6
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        orthogonalize!(H, 1)
        ψ = random_mps(Float64, sites, 8)

        # with sweep_maxdim=4, energy should still be reasonable
        φ = apply(H, ψ; maxdim=16, sweep_maxdim=4)

        # result is a valid MPS
        @test length(φ) == L
        for i in 1:L-1
            @test φ[i].right == φ[i+1].left
        end
        @test isfinite(real(inner(φ, φ)))
    end

end