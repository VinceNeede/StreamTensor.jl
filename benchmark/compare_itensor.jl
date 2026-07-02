# ---------------------------------------------------------------------------
# Confronto StreamTensor vs ITensor su DMRG. Script standalone, NON fa parte
# della SUITE di PkgBenchmark (vedi apply_benchmark_notes.md / discussione in
# chat: ITensor non cambia con i commit di StreamTensor, quindi il judge/
# compare di PkgBenchmark non è il meccanismo giusto per questo confronto).
#
# NOTA sui nomi qualificati: StreamTensor e ITensorMPS esportano entrambi
# `siteinds`, `OpSum`, `MPO`, `random_mps`, `dmrg`/`dmrg!` — con entrambi i
# pacchetti caricati questi nomi sono ambigui a livello globale, quindi vanno
# sempre qualificati con il modulo (`StreamTensor.X` / `ITensorMPS.X`), anche
# dentro problems.jl che viene incluso qui.
#
# NOTA sui parametri Krylov: per un confronto equo passiamo a ITensor
# (eigsolve_krylovdim, eigsolve_maxiter) uguali a quelli usati da StreamTensor
# (problem.eigsolve_kwargs), invece di lasciare i default di ITensor
# (krylovdim=3, maxiter=1), che sono diversi (vedi dmrg_status_notes.md).
#
# Uso:
#   julia --project=benchmark benchmark/compare_itensor.jl
# ---------------------------------------------------------------------------

using BenchmarkTools
using LinearAlgebra
using StreamTensor
using ITensors, ITensorMPS
using Printf

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "circuit.jl"))
include(joinpath(@__DIR__, "itensor_circuit.jl"))

# Solo nsite=2: ITensor non supporta DMRG a singolo sito nello stesso modo
# (vedi discussione in chat).
const _ITENSOR_COMPARABLE_PROBLEMS = filter(p -> p.nsite == 2, DMRG_PROBLEMS)

# --- Lato ITensor: stessa fisica (L, periodic), sintassi/nomi ITensor ------

function _tfim_opsum_itensor(L; periodic::Bool)
    os = ITensorMPS.OpSum()
    range_ = periodic ? (1:L) : (1:L-1)
    for i in range_
        j = periodic ? mod1(i + 1, L) : i + 1
        os += -1.0, "Sx", i, "Sx", j
    end
    for i in 1:L
        os += -1.0, "Sz", i
    end
    return os
end

"""
    build_itensor_inputs(problem::DMRGProblem) -> (sites, H)

Analogo a `build_dmrg_inputs` ma lato ITensor: stessi L/periodic, presi da
`problem.hamiltonian`, così la fisica non può divergere tra le due librerie.
"""
function build_itensor_inputs(problem::DMRGProblem)
    sites = ITensorMPS.siteinds("S=1/2", problem.hamiltonian.L)
    H = ITensorMPS.MPO(_tfim_opsum_itensor(problem.hamiltonian.L; periodic=problem.hamiltonian.periodic), sites)
    return sites, H
end

"""
    run_itensor_dmrg(psi0, H, problem::DMRGProblem) -> (energy, psi)

Analogo a `run_dmrg!` ma lato ITensor, con `eigsolve_krylovdim`/
`eigsolve_maxiter` presi da `problem.eigsolve_kwargs` per allinearsi a
StreamTensor.
"""
function run_itensor_dmrg(psi0, H, problem::DMRGProblem)
    nsweeps = length(problem.maxdim_schedule)
    return ITensorMPS.dmrg(H, psi0; nsweeps, maxdim=problem.maxdim_schedule,
                            cutoff=problem.cutoff, outputlevel=0,
                            eigsolve_krylovdim=problem.eigsolve_kwargs.krylovdim,
                            eigsolve_maxiter=problem.eigsolve_kwargs.maxiter)
end

# --- Benchmark dei due lati ---------------------------------------------------

struct ComparisonRow
    name::String
    st_time_ms::Float64
    st_memory_mib::Float64
    st_energy::Float64
    it_time_ms::Float64
    it_memory_mib::Float64
    it_energy::Float64
end

function _compare(problem::DMRGProblem)
    name = "$(problem.hamiltonian.name) (nsite=$(problem.nsite))"
    println("Benchmarking $name ...")

    # StreamTensor
    sites, H = build_dmrg_inputs(problem)
    st_trial = @benchmark(
        run_dmrg!(ψ, $H, $problem),
        setup = (ψ = StreamTensor.random_mps($sites, 1)),
    )
    _, _, sweep_data = run_dmrg!(StreamTensor.random_mps(sites, 1), H, problem)
    st_energy = sweep_data[end].energies[end]

    # ITensor
    it_sites, it_H = build_itensor_inputs(problem)
    it_trial = @benchmark(
        run_itensor_dmrg(psi0, $it_H, $problem),
        setup = (psi0 = ITensorMPS.random_mps($it_sites; linkdims=1)),
    )
    it_energy, _ = run_itensor_dmrg(ITensorMPS.random_mps(it_sites; linkdims=1), it_H, problem)

    st_m = median(st_trial)
    it_m = median(it_trial)

    return ComparisonRow(
        name,
        time(st_m) / 1e6, memory(st_m) / 2^20, st_energy,
        time(it_m) / 1e6, memory(it_m) / 2^20, it_energy,
    )
end

rows = [_compare(p) for p in _ITENSOR_COMPARABLE_PROBLEMS]

# --- Report ------------------------------------------------------------------

function _print_table(rows)
    @printf("%-30s %12s %12s %10s %12s %12s %10s\n",
            "problem", "ST time(ms)", "ST mem(MiB)", "ST E", "IT time(ms)", "IT mem(MiB)", "IT E")
    for r in rows
        @printf("%-30s %12.2f %12.2f %10.6f %12.2f %12.2f %10.6f\n",
                r.name, r.st_time_ms, r.st_memory_mib, r.st_energy,
                r.it_time_ms, r.it_memory_mib, r.it_energy)
    end
end

_print_table(rows)

function _write_markdown(path, rows)
    open(path, "w") do io
        println(io, "| problem | ST time (ms) | ST mem (MiB) | ST energy | IT time (ms) | IT mem (MiB) | IT energy |")
        println(io, "|---|---|---|---|---|---|---|")
        for r in rows
            @printf(io, "| %s | %.2f | %.2f | %.6f | %.2f | %.2f | %.6f |\n",
                    r.name, r.st_time_ms, r.st_memory_mib, r.st_energy,
                    r.it_time_ms, r.it_memory_mib, r.it_energy)
        end
    end
end

function _write_csv(path, rows)
    open(path, "w") do io
        println(io, "problem,st_time_ms,st_memory_mib,st_energy,it_time_ms,it_memory_mib,it_energy")
        for r in rows
            println(io, join((r.name, r.st_time_ms, r.st_memory_mib, r.st_energy,
                               r.it_time_ms, r.it_memory_mib, r.it_energy), ","))
        end
    end
end

mkpath(joinpath(@__DIR__, "results"))
_write_markdown(joinpath(@__DIR__, "results", "compare_itensor.md"), rows)
_write_csv(joinpath(@__DIR__, "results", "compare_itensor.csv"), rows)

println("\nSaved: benchmark/results/compare_itensor.md, benchmark/results/compare_itensor.csv")

# ---------------------------------------------------------------------------
# Confronto zip-up vs zip-up: apply() su una traiettoria del circuito
# brickwork. Vedi itensor_circuit.jl per _itensor_gate_layer_mpo /
# run_itensor_circuit_trajectory (condivise con verify_circuit_equivalence.jl).
# ---------------------------------------------------------------------------

struct CircuitComparisonRow
    name::String
    maxdim::Int
    st_time_ms::Float64
    st_memory_mib::Float64
    it_time_ms::Float64
    it_memory_mib::Float64
end

function _compare_circuit(problem::CircuitProblem)
    println("Benchmarking $(problem.name) ...")
    sites, ψ0, H_odd, H_even = build_circuit_inputs(problem)
    it_sites, it_ψ0, it_H_odd, it_H_even = build_itensor_circuit_inputs(problem)

    rows = CircuitComparisonRow[]
    for χ in problem.maxdim_values
        st_trial = @benchmark(
            run_circuit_trajectory($ψ0, $H_odd, $H_even, $(problem.n_steps);
                                    maxdim=$χ, cutoff=$(problem.cutoff),
                                    sweep_maxdim=$(2χ), sweep_cutoff=$(problem.cutoff / 10)),
        )
        it_trial = @benchmark(
            run_itensor_circuit_trajectory($it_ψ0, $it_H_odd, $it_H_even, $(problem.n_steps);
                                            maxdim=$χ, cutoff=$(problem.cutoff),
                                            sweep_maxdim=$(2χ), sweep_cutoff=$(problem.cutoff / 10)),
        )
        st_m, it_m = median(st_trial), median(it_trial)
        push!(rows, CircuitComparisonRow(
            problem.name, χ,
            time(st_m) / 1e6, memory(st_m) / 2^20,
            time(it_m) / 1e6, memory(it_m) / 2^20,
        ))
    end
    return rows
end

circuit_rows = reduce(vcat, (_compare_circuit(p) for p in CIRCUIT_PROBLEMS))

function _print_circuit_table(rows)
    @printf("%-16s %8s %12s %12s %12s %12s\n",
            "problem", "maxdim", "ST time(ms)", "ST mem(MiB)", "IT time(ms)", "IT mem(MiB)")
    for r in rows
        @printf("%-16s %8d %12.2f %12.2f %12.2f %12.2f\n",
                r.name, r.maxdim, r.st_time_ms, r.st_memory_mib, r.it_time_ms, r.it_memory_mib)
    end
end

_print_circuit_table(circuit_rows)

function _write_circuit_csv(path, rows)
    open(path, "w") do io
        println(io, "problem,maxdim,st_time_ms,st_memory_mib,it_time_ms,it_memory_mib")
        for r in rows
            println(io, join((r.name, r.maxdim, r.st_time_ms, r.st_memory_mib, r.it_time_ms, r.it_memory_mib), ","))
        end
    end
end

_write_circuit_csv(joinpath(@__DIR__, "results", "compare_itensor_apply.csv"), circuit_rows)

println("\nSaved: benchmark/results/compare_itensor_apply.csv")