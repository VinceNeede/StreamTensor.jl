# ---------------------------------------------------------------------------
# Lato ITensor del circuito brickwork — analogo di circuit.jl, condiviso tra
# compare_itensor.jl (benchmark velocità/memoria) e
# verify_circuit_equivalence.jl (verifica di correttezza fisica).
#
# ITensorMPS.apply(H::MPO, ψ::MPS; ...) usa alg="densitymatrix" per default
# — va richiesto esplicitamente alg="zipup" per usare lo stesso algoritmo di
# StreamTensor.apply (Stoudenmire & White 2010).
# ---------------------------------------------------------------------------

"""
    _itensor_gate_layer_mpo(sites, gate; start=1) -> MPO

Analogo ITensor di `build_gate_layer_mpo` (circuit.jl): stesso gate fisso,
decomposto via `ITensors.svd`. A differenza di StreamTensor (dove
`MPOTensor` è sempre rank-4, bordi inclusi), in ITensor il primo/ultimo
tensore della catena sono rank-3 — niente link a sinistra del primo, niente
link a destra dell'ultimo (esattamente come nel Rule54 di riferimento).
I link fittizi (dim 1) vanno quindi SOLO tra segmenti consecutivi interni,
mai ai due estremi assoluti. `length(sites)` deve essere pari.
"""
function _itensor_gate_layer_mpo(sites, gate::AbstractMatrix; start::Int=1)
    L = length(sites)
    T = permutedims(reshape(gate, 2, 2, 2, 2), (2, 1, 4, 3))   # (out1,out2,in1,in2)

    A = ITensorMPS.MPO(sites)
    i = 1

    if start == 2
        A[1] = ITensors.op("Id", sites[1])
        i = 2
    end

    while i <= L
        if i == L
            A[i] = ITensors.op("Id", sites[i])
            i += 1
        else
            gate_tensor = ITensors.ITensor(T, sites[i]', sites[i+1]', sites[i], sites[i+1])
            A[i], A[i+1] = ITensorMPS.factorize(gate_tensor, sites[i]', sites[i])
            i += 2
        end
    end

    return A
end

"""
    build_itensor_circuit_inputs(problem::CircuitProblem) -> (sites, ψ0, H_odd, H_even)

Analogo ITensor di `build_circuit_inputs` (circuit.jl): stessa struttura,
stesso stato di quench, MPO canonicalizzati una volta — fuori dal tempo
cronometrato, esattamente come lato StreamTensor.
"""
function build_itensor_circuit_inputs(problem::CircuitProblem)
    sites = ITensorMPS.siteinds("S=1/2", problem.L)
    ψ0 = ITensorMPS.MPS(sites, fill("Up", problem.L))
    H_odd = _itensor_gate_layer_mpo(sites, _CIRCUIT_GATE; start=1)
    ITensorMPS.orthogonalize!(H_odd, 1)
    H_even = _itensor_gate_layer_mpo(sites, _CIRCUIT_GATE; start=2)
    ITensorMPS.orthogonalize!(H_even, 1)
    return sites, ψ0, H_odd, H_even
end

"""
    run_itensor_circuit_trajectory(ψ0, sites, n_steps; maxdim=nothing, cutoff=nothing,
                                    sweep_maxdim=nothing, sweep_cutoff=nothing) -> MPS

Analogo ITensor di `run_circuit_trajectory` (circuit.jl).

Il metodo zipup di ITensor ha due livelli di troncamento distinti (vedi
`ITensors.contract(::Algorithm"zipup", A::MPO, B::AbstractMPS; cutoff,
maxdim, mindim, truncate_kwargs=(;cutoff,maxdim,mindim), kwargs...)`):
- `cutoff`/`maxdim`/`mindim` diretti: troncamento durante lo sweep di
  contrazione zip-up stesso — corrisponde a `sweep_cutoff`/`sweep_maxdim`
  di StreamTensor.
- `truncate_kwargs`: pass di compressione finale — corrisponde a
  `maxdim`/`cutoff` di StreamTensor.

Con `sweep_maxdim=nothing, sweep_cutoff=nothing` (default, come usato in
circuit.jl — modalità CTC, nessun troncamento nello sweep) i parametri
diretti sono impostati a "praticamente nessun troncamento"
(`cutoff=0.0, maxdim=typemax(Int)`), e tutto il troncamento vero avviene
in `truncate_kwargs`, esattamente come lato StreamTensor.
"""
function run_itensor_circuit_trajectory(ψ0, H_odd, H_even, n_steps;
                                         maxdim=nothing, cutoff=nothing,
                                         sweep_maxdim=nothing, sweep_cutoff=nothing)
    apply_kwargs = (
        alg = "zipup",
        cutoff = sweep_cutoff,
        maxdim = sweep_maxdim,
        mindim = 1,
        truncate_kwargs = (cutoff=cutoff, maxdim=maxdim, mindim=1),
    )
    
    ψ = ψ0
    for _ in 1:n_steps
        ψ = ITensorMPS.apply(H_odd, ψ; apply_kwargs...)
        ψ = ITensorMPS.apply(H_even, ψ; apply_kwargs...)
    end
    return ψ
end