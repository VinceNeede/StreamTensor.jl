# ---------------------------------------------------------------------------
# Shared physical problem definitions, reused by suites/dmrg.jl, suites/apply.jl,
# suites/inner.jl and benchmark/compare_itensor.jl.
# ---------------------------------------------------------------------------

struct HamiltonianSpec
    name::String
    L::Int
    periodic::Bool
    build_sites::Function   # () -> sites
    build_opsum::Function   # () -> OpSum
end

struct DMRGProblem
    hamiltonian::HamiltonianSpec
    nsite::Int
    maxdim_schedule::Vector{Int}
    cutoff::Float64
    noise::Union{Nothing,Vector{Float64}}
    eigsolve_kwargs::NamedTuple
    expected_energy::Float64
    energy_tol::Float64
end

# --- TFIM, J = h = 1, H = -J * sum(Sx_i Sx_j) - h * sum(Sz_i) --------------

function _tfim_opsum(L; periodic::Bool)
    os = StreamTensor.OpSum()
    range_ = periodic ? (1:L) : (1:L-1)
    for i in range_
        j = periodic ? mod1(i + 1, L) : i + 1
        os += (-1.0, "Sx", i, "Sx", j)
    end
    for i in 1:L
        os += (-1.0, "Sz", i)
    end
    return os
end

const tfim_L20_open = HamiltonianSpec(
    "tfim_L20_open", 20, false,
    () -> StreamTensor.siteinds("SpinHalf", 20),
    () -> _tfim_opsum(20; periodic=false),
)

const tfim_L50_periodic = HamiltonianSpec(
    "tfim_L50_periodic", 50, true,
    () -> StreamTensor.siteinds("SpinHalf", 50),
    () -> _tfim_opsum(50; periodic=true),
)

const tfim_L20_periodic = HamiltonianSpec(
    "tfim_L20_periodic", 20, true,
    () -> StreamTensor.siteinds("SpinHalf", 20),
    () -> _tfim_opsum(20; periodic=true),
)

# --- DMRG problem instances --------------------------------------------------

const _EIGSOLVE_KWARGS = (krylovdim=6, maxiter=5)   # StreamTensor hardcoded default, vedi dmrg_status_notes.md

# Valori reali usati nel benchmark pubblicato nel README per tfim_L20_periodic.
const _MAXDIM_SCHEDULE_L20 = [2, 2, 2, 4, 8, 10, 20, 40]
const _NOISE_SCHEDULE = [1.0, 0.1, 0.01, 0.001]   # solo nsite=1; oltre l'ultimo sweep elencato resta clampato a 0.001
const _CUTOFF = 1e-10

# L=50 è allo stesso punto critico (J=h=1). Questa schedule è quella che ha
# già funzionato empiricamente in un run precedente (converge a -26.5886...,
# scappa dal plateau OBC) — a differenza di una ricostruzione "proporzionale"
# con partenza più bassa (2,2,4,4,...), che invece resta bloccata nel plateau
# per diversi sweep (vedi dmrg_status_notes.md sul fenomeno). Partire da un
# maxdim iniziale più alto sembra aiutare DMRG a scappare prima dal minimo
# locale, non solo il numero di sweep.
const _MAXDIM_SCHEDULE_L50 = [10, 10, 20, 20, 40, 40, 40, 40, 40, 40]

const tfim_L20_open_nsite1 = DMRGProblem(
    tfim_L20_open, 1, _MAXDIM_SCHEDULE_L20, _CUTOFF, _NOISE_SCHEDULE,
    _EIGSOLVE_KWARGS, -10.602551828567528, 1e-6,
)

const tfim_L20_open_nsite2 = DMRGProblem(
    tfim_L20_open, 2, _MAXDIM_SCHEDULE_L20, _CUTOFF, nothing,
    _EIGSOLVE_KWARGS, -10.602551828567528, 1e-6,
)

const tfim_L20_periodic_nsite1 = DMRGProblem(
    tfim_L20_periodic, 1, _MAXDIM_SCHEDULE_L20, _CUTOFF, _NOISE_SCHEDULE,
    _EIGSOLVE_KWARGS, -10.635444153459572, 1e-6,
)

const tfim_L20_periodic_nsite2 = DMRGProblem(
    tfim_L20_periodic, 2, _MAXDIM_SCHEDULE_L20, _CUTOFF, nothing,
    _EIGSOLVE_KWARGS, -10.635444153459572, 1e-6,
)

const tfim_L50_periodic_nsite1 = DMRGProblem(
    tfim_L50_periodic, 1, _MAXDIM_SCHEDULE_L50, _CUTOFF, _NOISE_SCHEDULE,
    _EIGSOLVE_KWARGS, -26.5886, 1e-4,   # tolleranza più larga: le note danno solo 6 cifre
)

const tfim_L50_periodic_nsite2 = DMRGProblem(
    tfim_L50_periodic, 2, _MAXDIM_SCHEDULE_L50, _CUTOFF, nothing,
    _EIGSOLVE_KWARGS, -26.5886, 1e-4,   # tolleranza più larga: le note danno solo 6 cifre
)

# --- Shared helpers (usati sia da suites/dmrg.jl che dal check di correttezza
#     in benchmarks.jl) -------------------------------------------------------

"""
    build_dmrg_inputs(problem::DMRGProblem) -> (sites, H)

Costruisce `sites` e l'`MPO` una volta sola: non fa parte del tempo misurato
dal benchmark (vedi discussione in chat).
"""
function build_dmrg_inputs(problem::DMRGProblem)
    sites = problem.hamiltonian.build_sites()
    H = StreamTensor.MPO(problem.hamiltonian.build_opsum(), sites)
    return sites, H
end

"""
    run_dmrg!(ψ, H, problem::DMRGProblem)

Esegue `dmrg!` con i parametri del problema. Questa è la parte effettivamente
cronometrata da `@benchmarkable`.
"""
function run_dmrg!(ψ, H, problem::DMRGProblem)
    nsweeps = length(problem.maxdim_schedule)
    return StreamTensor.dmrg!(ψ, H, nsweeps;
                 nsite=problem.nsite, maxdim=problem.maxdim_schedule,
                 cutoff=problem.cutoff, noise=problem.noise,
                 eigsolve_kwargs=problem.eigsolve_kwargs)
end

const DMRG_PROBLEMS = (
    tfim_L20_open_nsite1, tfim_L20_open_nsite2,
    tfim_L20_periodic_nsite1, tfim_L20_periodic_nsite2,
    tfim_L50_periodic_nsite1, tfim_L50_periodic_nsite2,
)