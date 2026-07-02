# StreamTensor.jl

A Julia library for tensor-network simulations of one-dimensional quantum systems.
StreamTensor provides an MPS/MPO stack built from scratch, with a finite-state-machine
`OpSum → MPO` compiler and a DMRG solver benchmarked against ITensor for speed and
memory usage (see [Performance](#performance)).

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

I benchmark StreamTensor against [ITensor](https://itensor.org/) (via `ITensorMPS.jl`),
the reference tensor-network library, on physically meaningful DMRG problems rather than
synthetic random tensors. Full code is in [`benchmark/`](./benchmark), reproducible with:

```bash
julia --project=benchmark benchmark/compare_itensor.jl
```

**Methodology:**
- Same Hamiltonian, same physical parameters, same initial conditions on both sides
  (see `benchmark/problems.jl` for the exact `maxdim`/`cutoff`/`noise` schedules).
- Same Krylov solver settings passed explicitly to both libraries
  (`krylovdim=6, maxiter=5`) — comparing default-vs-default would conflate "solver
  tuning" with "implementation efficiency", which isn't what's being measured here.
- `nsite=2` only (the mode directly comparable between the two libraries).
- Single-threaded Julia. Absolute numbers depend on hardware — the point of
  reporting the ratio is that it's what stays roughly stable across machines;
  run `benchmark/compare_itensor.jl` yourself to reproduce on your own setup.

**Results** (TFIM, $J=h=1$):

| Problem | StreamTensor | ITensor | Speedup | Memory ratio |
|---|---|---|---|---|
| $L=20$, open      | 66.9 ms, 30.7 MiB   | 663.1 ms, 744.1 MiB  | 9.9×  | 24.2× |
| $L=20$, periodic  | 159.1 ms, 144.1 MiB | 698.3 ms, 864.7 MiB  | 4.4×  | 6.0×  |
| $L=50$, periodic  | 1057.2 ms, 1.61 GiB | 2369.1 ms, 2.74 GiB  | 2.2×  | 1.7×  |

Energies agree between the two libraries to the precision shown, **except** for
$L=50$ periodic:

**Known caveat:** for $L=50$ periodic, the two libraries converge to different
energies (StreamTensor: $-26.5886\ldots$, ITensor: $-26.5557\ldots$). This isn't a
bug — it's DMRG getting stuck in a well-documented local-minimum plateau under
periodic boundary conditions (the plateau value coincides with the open-boundary
ground energy). With matched parameters in this configuration, StreamTensor's sweep
schedule escapes it and ITensor's doesn't; this is a known, studied phenomenon in
PBC-DMRG, not evidence of a general accuracy advantage. Flagged here rather than
cherry-picked out.

Internal regression tracking (StreamTensor vs itself across commits, via
[`PkgBenchmark.jl`](https://github.com/JuliaCI/PkgBenchmark.jl)) also lives in
[`benchmark/`](./benchmark); see `benchmark/benchmarks.jl`.

### `apply` (MPO–MPS contraction)

Same methodology as above, this time on a fixed 2-qubit brickwork circuit
(non-overlapping gates on odd bonds, then even bonds, repeated for $N$ steps —
a standard quantum-circuit / quench setup) applied to a fully polarized initial
state. Both libraries build the layer MPOs once (outside the timed region) and
run `apply`/`apply` with the **same zip-up algorithm** and the same truncation
schedule recommended by Paeckel et al. (2019): a loose intermediate truncation
during the sweep (`sweep_maxdim = 2·maxdim`, `sweep_cutoff = cutoff/10`), then
a tight final compression pass at the target `maxdim`/`cutoff`.

**Methodology check:** before trusting the timings, the two trajectories were
verified to produce the *same physical state* — not just similar norms —
by materializing both to dense vectors (small $L$) and checking
$|\langle\psi_\text{ST}|\psi_\text{IT}\rangle| \approx 1$ (numerically, to machine
precision). See `benchmark/verify_circuit_equivalence.jl`.

| $L$ | maxdim | StreamTensor | ITensor |
|---|---|---|---|
| 20 | 5  | 13.0 ms, 7.0 MiB  | 51.9 ms, 57.0 MiB |
| 20 | 10 | 22.6 ms, 12.6 MiB | 58.3 ms, 62.3 MiB |
| 20 | 20 | 54.5 ms, 27.2 MiB | 71.3 ms, 74.7 MiB |
| 20 | 40 | 92.2 ms, 49.6 MiB | 93.0 ms, 93.4 MiB |
| 50 | 5  | 24.1 ms, 13.2 MiB | 98.6 ms, 111.6 MiB |
| 50 | 10 | 39.8 ms, 22.6 MiB | 106.7 ms, 118.6 MiB |
| 50 | 20 | 79.4 ms, 42.7 MiB | 114.9 ms, 127.0 MiB |
| 50 | 40 | 93.4 ms, 49.4 MiB | 114.9 ms, 127.6 MiB |

The gap narrows as `maxdim` grows (StreamTensor and ITensor converge to
comparable timings at `maxdim=40` for $L=20$) — the exact cause of the
residual gap at low `maxdim` hasn't been isolated. Reproducible via
`benchmark/compare_itensor.jl`.

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