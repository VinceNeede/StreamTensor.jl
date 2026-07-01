# ---------------------------------------------------------------------------
# DMRG benchmarks. Popola SUITE["dmrg"][hamiltonian_name]["nsite=$n"].
#
# H e sites sono costruiti una volta sola per problema, fuori dal blocco
# cronometrato: non fanno parte dell'algoritmo DMRG in sé (vedi
# opsum_mpo_design_notes.md). Lo stato iniziale ψ viene invece ricostruito
# ad ogni sample (setup=...), perché dmrg! lo muta in-place.
# ---------------------------------------------------------------------------

SUITE["dmrg"] = BenchmarkGroup()

for name in unique(p.hamiltonian.name for p in DMRG_PROBLEMS)
    SUITE["dmrg"][name] = BenchmarkGroup()
end

for problem in DMRG_PROBLEMS
    sites, H = build_dmrg_inputs(problem)

    SUITE["dmrg"][problem.hamiltonian.name]["nsite=$(problem.nsite)"] =
        @benchmarkable(
            run_dmrg!(ψ, $H, $problem),
            setup = (ψ = random_mps($sites, 1)),
        )
end