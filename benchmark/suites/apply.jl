# ---------------------------------------------------------------------------
# apply benchmarks. Popola SUITE["apply"][circuit_name]["maxdim=$χ"].
#
# Cronometra run_circuit_trajectory: N passi del circuito brickwork (vedi
# circuit.jl), ognuno troncato a un dato maxdim. H_odd/H_even/ψ0 si
# costruiscono una volta sola, fuori dal tempo cronometrato (stessi gate
# fissi ad ogni passo — circuito "Floquet").
#
# Questo file traccia SOLO velocità/memoria (regressione nel tempo). La
# curva errore-vs-maxdim è un report separato, non bloccante: vedi
# report_apply_accuracy.jl.
# ---------------------------------------------------------------------------

SUITE["apply"] = BenchmarkGroup()

for name in unique(p.name for p in CIRCUIT_PROBLEMS)
    SUITE["apply"][name] = BenchmarkGroup()
end

for problem in CIRCUIT_PROBLEMS
    sites, ψ0, H_odd, H_even = build_circuit_inputs(problem)

    for χ in problem.maxdim_values
        SUITE["apply"][problem.name]["maxdim=$χ"] =
            @benchmarkable(
                run_circuit_trajectory($ψ0, $H_odd, $H_even, $(problem.n_steps);
                    maxdim=$χ, cutoff=$(problem.cutoff),
                    sweep_maxdim=$(2χ), sweep_cutoff=$(problem.cutoff / 10))
            )
    end
end