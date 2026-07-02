# ---------------------------------------------------------------------------
# Curva errore-vs-maxdim per apply(), su una traiettoria di N passi del
# circuito brickwork (vedi circuit.jl). Script standalone, NON bloccante e
# NON parte della SUITE di PkgBenchmark — è un report informativo.
#
# Metodologia: φ_exact = run_circuit_trajectory(ψ0, H_odd, H_even, n_steps)
# SENZA troncamento (maxdim=nothing, bond dimension esatta 4^n_steps), poi
# per ogni maxdim testato calcola l'errore relativo tra la traiettoria
# troncata e quella esatta, allo stesso numero di passi, tramite inner
# products (evita di materializzare la differenza esplicita).
#
# Uso:
#   julia --project=benchmark benchmark/report_apply_accuracy.jl
# ---------------------------------------------------------------------------

using LinearAlgebra
using StreamTensor
using Printf

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "circuit.jl"))

struct AccuracyRow
    name::String
    maxdim::Int
    rel_error::Float64
end

function _accuracy_curve(problem::CircuitProblem)
    sites, ψ0, H_odd, H_even = build_circuit_inputs(problem)

    φ_exact = run_circuit_trajectory(ψ0, H_odd, H_even, problem.n_steps)   # nessun troncamento
    norm_exact = real(inner(φ_exact, φ_exact))

    rows = AccuracyRow[]
    for χ in problem.maxdim_values
        φ = run_circuit_trajectory(ψ0, H_odd, H_even, problem.n_steps;
                            maxdim=χ, cutoff=problem.cutoff,
                            sweep_maxdim=2χ, sweep_cutoff=problem.cutoff / 10)
        err = sqrt(abs(inner(φ_exact, φ_exact) - 2 * real(inner(φ_exact, φ)) + inner(φ, φ))) / sqrt(norm_exact)
        push!(rows, AccuracyRow(problem.name, χ, err))
    end
    return rows
end

rows = reduce(vcat, (_accuracy_curve(p) for p in CIRCUIT_PROBLEMS))

function _print_table(rows)
    @printf("%-20s %8s %14s\n", "problem", "maxdim", "rel. error")
    for r in rows
        @printf("%-20s %8d %14.3e\n", r.name, r.maxdim, r.rel_error)
    end
end

_print_table(rows)

function _write_csv(path, rows)
    open(path, "w") do io
        println(io, "problem,maxdim,rel_error")
        for r in rows
            println(io, join((r.name, r.maxdim, r.rel_error), ","))
        end
    end
end

mkpath(joinpath(@__DIR__, "results"))
_write_csv(joinpath(@__DIR__, "results", "apply_accuracy.csv"), rows)

println("\nSaved: benchmark/results/apply_accuracy.csv")