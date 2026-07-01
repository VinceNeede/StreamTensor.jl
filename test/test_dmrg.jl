using Test
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Build a product MPS from a list of state names, chaining link indices properly
function product_mps(::Type{T}, sites, state_names) where {T}
    L = length(sites)
    links = [Index(1, :Link) for _ in 0:L]
    tensors = map(1:L) do i
        v = T.(_dispatch_state(sites[i], StateName(state_names[i])))
        MPSTensor(reshape(v, 1, sites[i].dim, 1), links[i], sites[i], links[i+1])
    end
    return MPS(tensors, 0, L+1)
end

# Exact TFIM matrix for open boundary conditions
# H = -J Σ Sz_i Sz_{i+1} - h Σ Sx_i
function tfim_matrix(L::Int; J=1.0, h=1.0)
    sz = [0.5 0.0; 0.0 -0.5]
    sx = [0.0 0.5; 0.5  0.0]
    id = [1.0 0.0; 0.0  1.0]

    H = zeros(2^L, 2^L)
    for i in 1:L-1
        ops = [k == i ? sz : k == i+1 ? sz : id for k in 1:L]
        H .-= J .* foldl(kron, ops)
    end
    for i in 1:L
        ops = [k == i ? sx : id for k in 1:L]
        H .-= h .* foldl(kron, ops)
    end
    return H
end

# Exact Heisenberg matrix for open boundary conditions
# H = Σ (Sz_i Sz_{i+1} + 0.5*(S+_i S-_{i+1} + S-_i S+_{i+1}))
function heisenberg_matrix(L::Int)
    sz = [0.5 0.0; 0.0 -0.5]
    sp = [0.0 1.0; 0.0  0.0]
    sm = [0.0 0.0; 1.0  0.0]
    id = Matrix{Float64}(I, 2, 2)

    H = zeros(2^L, 2^L)
    for i in 1:L-1
        zz = [k==i ? sz : k==i+1 ? sz : id for k in 1:L]
        pm = [k==i ? sp : k==i+1 ? sm : id for k in 1:L]
        mp = [k==i ? sm : k==i+1 ? sp : id for k in 1:L]
        H .+= foldl(kron, zz)
        H .+= 0.5 .* foldl(kron, pm)
        H .+= 0.5 .* foldl(kron, mp)
    end
    return H
end

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

function heisenberg_opsum(sites)
    L = length(sites)
    os = OpSum()
    for i in 1:L-1
        os += (1.0, "Sz", i, "Sz", i+1)
        os += (0.5, "S+", i, "S-", i+1)
        os += (0.5, "S-", i, "S+", i+1)
    end
    return os
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "StreamTensor DMRG" begin

# ─────────────────────────────────────────────────────────────────────────────
@testset "OpSum construction" begin

    @testset "single-site term" begin
        os = OpSum()
        os += (0.5, "Sz", 1)
        @test length(os.terms) == 1
        @test os.terms[1].coeff == 0.5
        @test os.terms[1].ops == [1 => ("Sz", (;))]
    end

    @testset "two-site term sorted by site" begin
        os = OpSum()
        os += (1.0, "Sz", 3, "Sz", 1)   # deliberately out of order
        @test os.terms[1].ops == [1 => ("Sz", (;)), 3 => ("Sz", (;))]
    end

    @testset "multiple terms accumulate" begin
        os = OpSum()
        for i in 1:4
            os += (1.0, "Sz", i, "Sz", i+1)
        end
        @test length(os.terms) == 4
    end

    @testset "malformed term (missing site) throws" begin
        os = OpSum()
        @test_throws Exception add!(os, (1.0, "Sz"))
    end

    @testset "coefficient type preserved" begin
        os = OpSum()
        os += (1.0 + 0.0im, "Sz", 1)
        @test os.terms[1].coeff isa ComplexF64
    end

end  # OpSum construction

# ─────────────────────────────────────────────────────────────────────────────
@testset "MPO construction from OpSum" begin

    @testset "single-site Sz: structure and matrix elements" begin
        sites = siteinds(:SpinHalf, 3)
        os = OpSum()
        os += (1.0, "Sz", 2)
        H = MPO(os, sites)

        @test length(H) == 3
        @test H[1].left.dim == 1
        @test H[3].right.dim == 1

        # ⟨↑↑↑|Sz₂|↑↑↑⟩ = +0.5, ⟨↑↓↑|Sz₂|↑↓↑⟩ = -0.5
        ψ_uuu = product_mps(Float64, sites, ["Up","Up","Up"])
        ψ_udu = product_mps(Float64, sites, ["Up","Dn","Up"])
        @test inner(ψ_uuu, H, ψ_uuu) ≈  0.5 atol=1e-10
        @test inner(ψ_udu, H, ψ_udu) ≈ -0.5 atol=1e-10
    end

    @testset "two-site SzSz: all basis state expectation values" begin
        sites = siteinds(:SpinHalf, 2)
        os = OpSum()
        os += (1.0, "Sz", 1, "Sz", 2)
        H = MPO(os, sites)

        ψ_uu = product_mps(Float64, sites, ["Up","Up"])
        ψ_ud = product_mps(Float64, sites, ["Up","Dn"])
        ψ_du = product_mps(Float64, sites, ["Dn","Up"])
        ψ_dd = product_mps(Float64, sites, ["Dn","Dn"])

        @test inner(ψ_uu, H, ψ_uu) ≈  0.25 atol=1e-10
        @test inner(ψ_ud, H, ψ_ud) ≈ -0.25 atol=1e-10
        @test inner(ψ_du, H, ψ_du) ≈ -0.25 atol=1e-10
        @test inner(ψ_dd, H, ψ_dd) ≈  0.25 atol=1e-10
    end

    @testset "TFIM bulk bond dimension = 3" begin
        L = 6
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)

        for i in 2:L-1
            @test H[i].left.dim == 3
            @test H[i].right.dim == 3
        end
        @test H[1].left.dim == 1
        @test H[L].right.dim == 1
    end

    @testset "TFIM L=4 full Hamiltonian matrix via all product states" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        H_exact = tfim_matrix(L)

        states = [["Up","Up","Up","Up"], ["Up","Up","Up","Dn"],
                  ["Up","Up","Dn","Up"], ["Up","Dn","Up","Up"],
                  ["Dn","Up","Up","Up"]]

        for s in states
            ψ = product_mps(Float64, sites, s)
            v = Float64[s[i] == "Up" ? 1.0 : 0.0 for i in 1:L]
            # Kronecker product ordering: site 1 is leftmost (most significant)
            idx = sum((s[i]=="Dn" ? 1 : 0) * 2^(L-i) for i in 1:L) + 1
            E_mpo  = inner(ψ, H, ψ)
            E_diag = H_exact[idx, idx]
            @test E_mpo ≈ E_diag atol=1e-10
        end
    end

end  # MPO construction

# ─────────────────────────────────────────────────────────────────────────────
@testset "inner(ψ, H, φ)" begin

    @testset "identity MPO: inner(ψ,H,ψ) ≈ inner(ψ,ψ)" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        os = OpSum()
        for i in 1:L
            os += (1.0, "Id", i)
        end
        H = MPO(os, sites)
        ψ = random_mps(Float64, sites, 4)

        @test inner(ψ, H, ψ) ≈ inner(ψ, ψ) atol=1e-10
    end

    @testset "off-diagonal: ⟨↑↑|SzSz|↓↓⟩ = 0" begin
        sites = siteinds(:SpinHalf, 2)
        os = OpSum()
        os += (1.0, "Sz", 1, "Sz", 2)
        H = MPO(os, sites)

        ψ = product_mps(Float64, sites, ["Up","Up"])
        φ = product_mps(Float64, sites, ["Dn","Dn"])

        @test inner(ψ, H, φ) ≈ 0.0 atol=1e-10
    end

    @testset "bra === ket: no index conflict (ψ passed as both bra and ket)" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        ψ = random_mps(Float64, sites, 4)

        result = inner(ψ, H, ψ)
        @test isfinite(real(result))
    end

    @testset "ComplexF64 MPS: ⟨ψ|H|ψ⟩ is real for Hermitian H" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        ψ = random_mps(ComplexF64, sites, 4)

        result = inner(ψ, H, ψ)
        @test abs(imag(result)) < 1e-10
        @test isfinite(real(result))
    end

    @testset "inner vs expect for single-site observable" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        os_sz = OpSum()
        for i in 1:L
            os_sz += (1.0, "Sz", i)
        end
        H_sz = MPO(os_sz, sites)

        ψ = random_mps(Float64, sites, 8)
        orthogonalize!(ψ, 1)

        sz_mat = op(SiteType(:SpinHalf), "Sz")
        E_inner  = real(inner(ψ, H_sz, ψ)) / real(inner(ψ, ψ))
        E_expect = sum(expect(ψ, sz_mat, i) for i in 1:L)

        @test E_inner ≈ E_expect atol=1e-10
    end

    @testset "TFIM L=4: inner vs exact diagonalization" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        E_exact = minimum(eigvals(Symmetric(tfim_matrix(L))))

        ψ = random_mps(Float64, sites, 8)
        E_dmrg, ψ = dmrg!(ψ, H; nsweeps=10,
            maxdim=[2,4,8,8,8,8,8,8,8,8],
            cutoff=1e-10,
            eigsolve_kwargs=(krylovdim=6, maxiter=5))

        E_inner = real(inner(ψ, H, ψ)) / real(inner(ψ, ψ))
        @test E_inner ≈ E_exact atol=1e-8
        @test E_dmrg ≈ E_exact  atol=1e-8
    end

end  # inner(ψ, H, φ)

# ─────────────────────────────────────────────────────────────────────────────
@testset "DMRG" begin

    @testset "TFIM L=4 open: energy vs exact diagonalization" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        E_exact = minimum(eigvals(Symmetric(tfim_matrix(L))))

        ψ = random_mps(Float64, sites, 8)
        E, _ = dmrg!(ψ, H; nsweeps=10,
            maxdim=[2,4,8,8,8,8,8,8,8,8],
            cutoff=1e-10,
            eigsolve_kwargs=(krylovdim=6, maxiter=5))

        @test E ≈ E_exact atol=1e-8
    end

    @testset "TFIM L=6 open: energy vs exact diagonalization" begin
        L = 6
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        E_exact = minimum(eigvals(Symmetric(tfim_matrix(L))))

        ψ = random_mps(Float64, sites, 16)
        E, _ = dmrg!(ψ, H; nsweeps=12,
            maxdim=[2,4,8,16,16,16,16,16,16,16,16,16],
            cutoff=1e-12,
            eigsolve_kwargs=(krylovdim=6, maxiter=5))

        @test E ≈ E_exact atol=1e-8
    end

    @testset "TFIM L=20 open: energy vs known reference" begin
        # Reference value from dmrg_status_notes.md: verified against ITensor
        L = 20
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)

        ψ = random_mps(Float64, sites, 40)
        E, _ = dmrg!(ψ, H; nsweeps=10,
            maxdim=[4,8,16,32,40,40,40,40,40,40],
            cutoff=1e-12,
            eigsolve_kwargs=(krylovdim=6, maxiter=5))

        @test E ≈ -10.6354441534 atol=1e-6
    end

    @testset "nsite=1 and nsite=2 agree on TFIM L=6" begin
        L = 6
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)

        ψ1 = random_mps(Float64, sites, 16)
        E1, _ = dmrg!(ψ1, H; nsweeps=12, nsite=1,
            maxdim=16, cutoff=1e-12, noise=1e-4,
            eigsolve_kwargs=(krylovdim=6, maxiter=5))

        ψ2 = random_mps(Float64, sites, 16)
        E2, _ = dmrg!(ψ2, H; nsweeps=12, nsite=2,
            maxdim=16, cutoff=1e-12,
            eigsolve_kwargs=(krylovdim=6, maxiter=5))

        @test E1 ≈ E2 atol=1e-8
    end

    @testset "variational principle: DMRG energy >= exact ground state" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        E_exact = minimum(eigvals(Symmetric(tfim_matrix(L))))

        ψ = random_mps(Float64, sites, 8)
        E, _ = dmrg!(ψ, H; nsweeps=10,
            maxdim=8, cutoff=1e-10,
            eigsolve_kwargs=(krylovdim=6, maxiter=5))

        @test E >= E_exact - 1e-10
    end

    @testset "Heisenberg L=4 open: energy vs exact diagonalization" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(heisenberg_opsum(sites), sites)
        E_exact = minimum(real.(eigvals(Hermitian(heisenberg_matrix(L)))))

        ψ = random_mps(Float64, sites, 8)
        E, _ = dmrg!(ψ, H; nsweeps=10,
            maxdim=8, cutoff=1e-10,
            eigsolve_kwargs=(krylovdim=6, maxiter=5))

        @test E ≈ E_exact atol=1e-8
    end

    @testset "Heisenberg L=6 open: energy vs exact diagonalization" begin
        L = 6
        sites = siteinds(:SpinHalf, L)
        H = MPO(heisenberg_opsum(sites), sites)
        E_exact = minimum(real.(eigvals(Hermitian(heisenberg_matrix(L)))))

        ψ = random_mps(Float64, sites, 16)
        E, _ = dmrg!(ψ, H; nsweeps=12,
            maxdim=16, cutoff=1e-12,
            eigsolve_kwargs=(krylovdim=6, maxiter=5))

        @test E ≈ E_exact atol=1e-8
    end

    @testset "ComplexF64 MPS: DMRG converges on TFIM L=4" begin
        L = 4
        sites = siteinds(:SpinHalf, L)
        H = MPO(tfim_opsum(sites), sites)
        E_exact = minimum(eigvals(Symmetric(tfim_matrix(L))))

        ψ = random_mps(ComplexF64, sites, 8)
        E, _ = dmrg!(ψ, H; nsweeps=10,
            maxdim=8, cutoff=1e-10,
            eigsolve_kwargs=(krylovdim=6, maxiter=5))

        @test real(E) ≈ E_exact atol=1e-8
        @test abs(imag(E)) < 1e-10
    end

end  # DMRG

end  # StreamTensor DMRG