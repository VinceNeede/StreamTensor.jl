using BenchmarkTools
using StreamTensor

include(joinpath(@__DIR__, "problems.jl"))

# ---------------------------------------------------------------------------
# Check di correttezza bloccante: un run NON cronometrato di ogni DMRGProblem,
# per assicurarsi che l'energia converga al valore atteso (vedi
# dmrg_status_notes.md) prima di fidarsi dei tempi misurati dalla SUITE.
# Se questo fallisce, benchmarkpkg si interrompe con un errore.
# ---------------------------------------------------------------------------

function _check_dmrg_correctness(problem::DMRGProblem)
    sites, H = build_dmrg_inputs(problem)
    ψ0 = random_mps(sites, 1)
    _, _, sweep_data = run_dmrg!(ψ0, H, problem)
    energy = sweep_data[end].energies[end]
    @assert isapprox(energy, problem.expected_energy; atol=problem.energy_tol) """
        DMRG correctness check failed for $(problem.hamiltonian.name) (nsite=$(problem.nsite)):
        got $energy, expected $(problem.expected_energy) ± $(problem.energy_tol)
        """
    return nothing
end

for problem in DMRG_PROBLEMS
    _check_dmrg_correctness(problem)
end

# ---------------------------------------------------------------------------
# SUITE
# ---------------------------------------------------------------------------

const SUITE = BenchmarkGroup()

include(joinpath(@__DIR__, "suites", "dmrg.jl"))