# StreamTensor.jl

A Julia library for tensor-network simulations of one-dimensional quantum systems.
StreamTensor provides an MPS/MPO stack built from scratch, with a finite-state-machine
`OpSum → MPO` compiler and a DMRG solver competitive with ITensor in both speed and
memory usage.

> **Status:** active development. The current release covers closed-system ground-state
> search via DMRG (single-site and two-site, including DMRG3S subspace expansion)
> and MPO–MPS contraction via the zip-up algorithm (`apply!`/`apply`).
> Two further contraction algorithms are planned: the naïve contract-then-compress
> method and Successive Randomized Compression (SRC, Camaño, Epperly & Tropp 2025),
> which achieves accuracy comparable to contract-then-compress at zip-up speed.
> Planned extensions also include GPU-accelerated contraction (windowed device memory,
> without copying the full MPS/MPO to VRAM) and open quantum systems
> (Lindblad master equation, matrix product density operators).

---

## Features

- **Concrete `Index` type** — single, fully-concrete struct with a fixed-capacity
  `NTuple{4,Symbol}` tag set; no type parameters, no boxing of individual indices in
  `NTuple{N,Index}` fields, no heap allocation on `prime`/`noprime`/`contract` calls.
- **`OpSum` → `MPO` compiler** — write Hamiltonians in second-quantised notation;
  the finite-state-machine builder produces the exact MPO automatically.
- **MPO–MPS contraction** — `apply!(H, ψ)` / `apply(H, ψ)` compute `H|ψ⟩` as a
  compressed MPS using the zip-up algorithm (Stoudenmire & White 2010): a
  left-to-right sweep contracts and optionally truncates site by site, followed by
  a right-to-left SVD compression pass. Naïve contract-then-compress and Successive
  Randomized Compression (SRC) are planned.
- **DMRG** — single-site (`nsite=1`) with DMRG3S subspace expansion (Hubig et al. 2015)
  and two-site (`nsite=2`), adaptive `eigsolve` tolerance schedule, per-sweep
  `maxdim`/`cutoff`/`noise` schedules.
- **Efficient contraction** — `_matricize` avoids unnecessary copies for two-block-swap
  permutations, feeding `mul!`/`gemm!` directly.
- **SpinHalf / Qubit site type** included; extend with `@alias_sitetype` and custom
  `op`/`state` methods.

---

## Installation

StreamTensor.jl is not yet registered in the Julia General Registry.
Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/VinceNeede/StreamTensor.jl")
```

---

## Background

### Matrix Product States (MPS)

A quantum state of $L$ sites can be written exactly as a product of matrices:

$$|\psi\rangle = \sum_{\sigma_1,\ldots,\sigma_L} A^{\sigma_1}_1 A^{\sigma_2}_2 \cdots A^{\sigma_L}_L \, |\sigma_1 \cdots \sigma_L\rangle$$

where each $A^{\sigma_i}_i$ is a $\chi_{i-1} \times \chi_i$ matrix (the *bond dimension*
$\chi$ controls the amount of entanglement the ansatz can represent). Ground states of
gapped 1D Hamiltonians satisfy an *area law* for entanglement entropy, meaning a moderate
$\chi$ is sufficient for high accuracy.

### Matrix Product Operators (MPO)

A local Hamiltonian $H = \sum_\alpha c_\alpha \prod_j O^\alpha_j$ is represented as an
MPO — the operator analogue of an MPS — whose bond dimension is determined by the range
and number of interaction terms. StreamTensor builds the MPO automatically from an
`OpSum` via a finite-state-machine (automaton) construction: each term traces a unique
path through the automaton from a shared start state $I$ to a shared end state $F$,
and the MPO tensors encode all paths simultaneously.

### DMRG

The Density Matrix Renormalization Group (White 1992) variationally minimizes
$\langle\psi|H|\psi\rangle$ over the MPS manifold. At each step a small *local*
eigenvalue problem is solved (the effective Hamiltonian acting on one or two sites),
and the solution is folded back into the MPS via SVD truncation. Sweeping left and
right to convergence yields the ground state.

StreamTensor implements:

- **Two-site DMRG** (`nsite=2`): the standard algorithm; bond dimension grows
  naturally through the SVD.
- **Single-site DMRG with subspace expansion** (`nsite=1`, `noise≠nothing`):
  the DMRG3S enrichment of Hubig et al. [[3]](#references),
  which adds a perturbative correction to escape local minima while keeping the
  per-step cost lower than two-site.

---

## Quickstart

### 1. Define sites and a Hamiltonian

```julia
using StreamTensor

L = 20
sites = siteinds("SpinHalf", L)

# Transverse-field Ising model with PBC: H = -J ΣSzSz - h ΣSx
J, h = 1.0, 1.0
H = OpSum()
for i in 1:L-1
    H += (-J, "Sz", i, "Sz", i+1)
end
H += (-J, "Sz", 1, "Sz", L)   # periodic boundary
for i in 1:L
    H += (-h, "Sx", i)
end

mpo = MPO(H, sites)
```

### 2. Build an initial MPS and run DMRG

```julia
ψ0 = random_mps(sites, 10)   # random initial state, bond dim 10

ψ, _, sweep_data = dmrg!(ψ0, mpo, 10;
    nsite  = 2,
    maxdim = [10, 20, 40],    # per-sweep schedule
    cutoff = 1e-10)

E = sweep_data[end].energies[end]
println("Ground state energy: $E")
```

### 3. Apply an MPO to an MPS

```julia
# compute H|ψ⟩ as a compressed MPS (zip-up algorithm)
orthogonalize!(mpo, 1)          # recommended before apply!
Hψ = apply(mpo, ψ; maxdim=40, cutoff=1e-10)

# energy via inner product: ⟨ψ|H|ψ⟩ = ⟨ψ|Hψ⟩
E = real(inner(ψ, Hψ)) / real(inner(ψ, ψ))

# in-place version (destroys ψ, avoids copying — useful in time evolution)
apply!(mpo, ψ; maxdim=40, cutoff=1e-10)
```

### 4. Measure observables

```julia
using LinearAlgebra

Sz = op(SiteType("SpinHalf"), "Sz")
sz_profile = expect(ψ, Sz)        # ⟨Sz_i⟩ at every site
println("⟨Sz⟩ = ", sz_profile)

norm2 = inner(ψ, ψ)
println("‖ψ‖² = ", norm2)
```

### 5. Parametric and custom operators

```julia
# Rotation gate (built-in parametric operator)
Rx_op = op(sites[1], "Rx"; θ = π/4)

# Define a new site type
import StreamTensor: op, state, siteind, SiteType, OpName, StateName

siteind(::SiteType{:MyQutrit}) = Index(3, :Site; sitetype=SiteType{:MyQutrit})

op(::SiteType{:MyQutrit}, ::OpName{:Lz}) = diagm([1.0, 0.0, -1.0])

state(::SiteType{:MyQutrit}, ::StateName{:Zero}) = [0.0, 1.0, 0.0]
```

---

## Performance

Benchmarks on the periodic TFIM ($L=20$, $J=h=1$, PBC,
`maxdim=[2,2,2,4,8,10,20,40]`, 10 sweeps).
`nsite=1` uses `noise=[1, 0.1, 0.01, 0.001]` (DMRG3S subspace expansion).
Both StreamTensor variants converge to $E_0 = -10.6354441534\ldots$

| | nsite | krylovdim | maxiter | median time | min time | memory | allocations |
|---|---|---|---|---|---|---|---|
| StreamTensor | 2 | 6 | 5 | 220 ms | 191 ms | 279 MiB | 873 K |
| StreamTensor | 2 | 3 | 1 | 138 ms | 112 ms | 165 MiB | 702 K |
| StreamTensor | 1 | 6 | 5 | 147 ms | 120 ms | 137 MiB | 183 K |
| StreamTensor | 1 | 3 | 1 | 125 ms | 111 ms | 115 MiB | 145 K |
| ITensor      | 2 | 3 | 1 | 338 ms | 328 ms | 480 MiB | 864 K |

All benchmarks run with `BenchmarkTools.@benchmark` (20–37 samples).
Median times are reported; GC pressure causes high variance across all runs
(observed range: 0%–70% of wall time for both libraries).

The apples-to-apples comparison (krylovdim=3, maxiter=1) shows StreamTensor
roughly **2.5× faster and 3× lower memory** than ITensor for `nsite=2`.
The allocation count is comparable (~700K vs ~864K), suggesting ITensor's
overhead comes from larger individual allocations rather than more frequent ones.

---

## Project structure

```
src/
├── StreamTensor.jl          # module entry point, all exports
├── index.jl                 # Index type (concrete, isbitstype)
├── tensor.jl                # DenseTensor, DiagTensor, MPSTensor, MPOTensor
├── contraction.jl           # contract, _matricize, Combiner
├── decomposition.jl         # svd, qr, factorize (tensor-train conventions)
├── abstracttensortrain.jl   # AbstractTensorTrain base, orthogonalize!, compress!
├── mps.jl                   # MPS, random_mps, inner
├── mpo.jl                   # MPO, expect, inner(ψ,H,φ)
├── apply.jl                 # apply!/apply, zip-up MPO–MPS contraction
├── opsum.jl                 # OpSum, OpTerm, FSM MPO builder
├── dmrg.jl                  # ProjMPO, dmrg_sweep!, dmrg!
└── sitetypes/
    ├── tags.jl              # SiteType, OpName, StateName, @alias_sitetype
    ├── sitetypes.jl         # siteind, op, state dispatch layer
    └── qubit.jl             # SpinHalf / Qubit site type
```

---

## References

- [1] S. R. White, *Density matrix formulation for quantum renormalization groups*,
  Phys. Rev. Lett. **69**, 2863 (1992).
- [2] U. Schollwöck, *The density-matrix renormalization group in the age of matrix
  product states*, Ann. Phys. **326**, 96 (2011).
- [3] C. Hubig, I. P. McCulloch, U. Schollwöck, F. A. Wolf,
  *Strictly single-site DMRG algorithm with subspace expansion*,
  Phys. Rev. B **91**, 155115 (2015).
  [10.1103/PhysRevB.91.155115](https://link.aps.org/doi/10.1103/PhysRevB.91.155115)
- [4] E. M. Stoudenmire and S. R. White,
  *Minimally entangled typical thermal state algorithms*,
  New J. Phys. **12**, 055026 (2010). *(zip-up algorithm)*
- [5] C. Camaño, E. N. Epperly, and J. A. Tropp,
  *Successive randomized compression: A randomized algorithm for the compressed
  MPO–MPS product*, Quantum (2025).
  [arXiv:2504.06475](https://arxiv.org/abs/2504.06475)

---

## License

MIT — see [LICENSE](LICENSE).