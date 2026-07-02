# ---------------------------------------------------------------------------
# Circuito quantistico random a bricchi ("brickwork"), gate fisso.
#
# Gate 2x2x2x2 fisso (seed=42, ortogonale reale, generato offline — vedi
# discussione in chat), applicato in due layer alternati:
#   - layer "odd":  bond (1,2), (3,4), (5,6), ...
#   - layer "even": bond (2,3), (4,5), ...  (siti 1 e L, se scoperti, identità)
# Un passo completo = apply(layer_even, apply(layer_odd, ψ)).
# ---------------------------------------------------------------------------

# Gate fisso, seed=42, ortogonale reale 4×4 (generato una volta offline, non a runtime).
const _CIRCUIT_GATE = [
    0.154219769592948  -0.661326125434211   0.307225002353379  -0.666690945201626;
   -0.987434631181200  -0.079398578030011   0.069569904996381  -0.117595677087443;
   -0.008503201203209  -0.456837201326657   0.545528812124499   0.702585071144668;
    0.033418668136880   0.589612918030985   0.776640934660152  -0.219245657017761;
]

const _ID2 = [1.0 0.0; 0.0 1.0]

"""
    _gate_to_mpo_tensors(gate, l, r, s1, s2) -> (MPOTensor, MPOTensor)

Decompone un gate 2-siti (matrice 4×4, convenzione indice combinato
`k = (i1-1)*2 + i2`, `i2` più veloce) in due `MPOTensor` via SVD, con bond
interno di dimensione `rank(gate)` (tipicamente 4, dato che `_CIRCUIT_GATE`
è ortogonale generico). `l`/`r` sono gli Index di bordo del layer; `s1`/`s2`
i due site Index accoppiati dal gate.
"""
function _gate_to_mpo_tensors(gate::AbstractMatrix, l::StreamTensor.Index, r::StreamTensor.Index, s1::StreamTensor.Index, s2::StreamTensor.Index)
    d = 2
    T = permutedims(reshape(gate, d, d, d, d), (2, 1, 4, 3))   # (out1,out2,in1,in2)
    M = reshape(permutedims(T, (1, 3, 2, 4)), d * d, d * d)     # rows=(out1,in1), cols=(out2,in2)
    F = LinearAlgebra.svd(M)
    χ = length(F.S)
    bond = StreamTensor.Index(χ, :Link)

    L_storage = reshape(F.U, 1, d, d, χ)
    R_storage = reshape(LinearAlgebra.Diagonal(F.S) * F.Vt, χ, d, d, 1)

    left_tensor  = StreamTensor.MPOTensor(L_storage, l, s1', s1, bond)
    right_tensor = StreamTensor.MPOTensor(R_storage, bond, s2', s2, r)
    return left_tensor, right_tensor
end

_identity_mpo_tensor(l::StreamTensor.Index, r::StreamTensor.Index, s::StreamTensor.Index) =
    StreamTensor.MPOTensor(reshape(_ID2, 1, 2, 2, 1), l, s', s, r)

"""
    build_gate_layer_mpo(sites, gate; start=1) -> MPO

Un layer del circuito: `gate` applicato ai bond non sovrapposti a partire
da `start` (`start=1` → bond (1,2),(3,4),...; `start=2` → bond (2,3),(4,5),...,
con siti 1 e/o L scoperti trattati a identità). `length(sites)` deve essere pari.
"""
function build_gate_layer_mpo(sites, gate; start::Int=1)
    L = length(sites)
    @assert iseven(L) "build_gate_layer_mpo richiede L pari"
    @assert start in (1, 2) "start deve essere 1 o 2"

    tensors = Vector{StreamTensor.MPOTensor{Float64,Array{Float64,4}}}(undef, L)
    prev_r = StreamTensor.Index(1, :Link)
    i = 1

    if start == 2
        r = StreamTensor.Index(1, :Link)
        tensors[1] = _identity_mpo_tensor(prev_r, r, sites[1])
        prev_r = r
        i = 2
    end

    while i <= L
        if i == L
            r = StreamTensor.Index(1, :Link)
            tensors[i] = _identity_mpo_tensor(prev_r, r, sites[i])
            prev_r = r
            i += 1
        else
            l = prev_r
            r = StreamTensor.Index(1, :Link)
            t1, t2 = _gate_to_mpo_tensors(gate, l, r, sites[i], sites[i+1])
            tensors[i], tensors[i+1] = t1, t2
            prev_r = r
            i += 2
        end
    end

    return StreamTensor.MPO(tensors)
end

"""
    circuit_step(ψ, sites) -> MPS

Un passo completo del circuito brickwork: layer dispari poi layer pari.
"""
function circuit_step(ψ, sites)
    H_odd = build_gate_layer_mpo(sites, _CIRCUIT_GATE; start=1)
    StreamTensor.orthogonalize!(H_odd, 1)
    ψ1 = StreamTensor.apply(H_odd, ψ)

    H_even = build_gate_layer_mpo(sites, _CIRCUIT_GATE; start=2)
    StreamTensor.orthogonalize!(H_even, 1)
    return StreamTensor.apply(H_even, ψ1)
end

# ---------------------------------------------------------------------------
# CircuitProblem: benchmark di apply() su una traiettoria di N passi del
# circuito, invece che su una singola applicazione di un'Hamiltoniana (vedi
# discussione in chat: applicare H ripetutamente non ha significato fisico,
# e un singolo apply su uno stato prodotto non genera abbastanza bond
# dimension da rendere interessante la curva errore-vs-maxdim).
# ---------------------------------------------------------------------------

struct CircuitProblem
    name::String
    L::Int
    n_steps::Int
    maxdim_values::Vector{Int}
    cutoff::Float64
end

const _CIRCUIT_MAXDIM_VALUES = [5, 10, 20, 40]
const _CIRCUIT_CUTOFF = 1e-12

# N=4 per L=20 (bond esatta finale 4^4=256, gestibile); N=3 per L=50 (4^3=64,
# più basso perché il costo per passo cresce anche con L, non solo la bond
# dimension).
const circuit_L20 = CircuitProblem("circuit_L20", 20, 4, _CIRCUIT_MAXDIM_VALUES, _CIRCUIT_CUTOFF)
const circuit_L50 = CircuitProblem("circuit_L50", 50, 3, _CIRCUIT_MAXDIM_VALUES, _CIRCUIT_CUTOFF)

const CIRCUIT_PROBLEMS = (circuit_L20, circuit_L50)

"""
    build_quench_state(sites) -> MPS

Stato prodotto completamente polarizzato lungo z (tutti "Up"), bond dimension 1
— lo stato iniziale standard per un quench.
"""
function build_quench_state(sites)
    return StreamTensor.MPS(sites, fill("Up", length(sites)))
end

"""
    build_circuit_inputs(problem::CircuitProblem) -> (sites, ψ0, H_odd, H_even)

Costruisce sites, stato iniziale di quench, e i due layer MPO (canonicalizzati
una volta — stessi gate fissi ad ogni passo, "Floquet"). Non fa parte del
tempo cronometrato.
"""
function build_circuit_inputs(problem::CircuitProblem)
    sites = StreamTensor.siteinds("SpinHalf", problem.L)
    ψ0 = build_quench_state(sites)
    H_odd = build_gate_layer_mpo(sites, _CIRCUIT_GATE; start=1)
    StreamTensor.orthogonalize!(H_odd, 1)
    H_even = build_gate_layer_mpo(sites, _CIRCUIT_GATE; start=2)
    StreamTensor.orthogonalize!(H_even, 1)
    return sites, ψ0, H_odd, H_even
end

"""
    run_circuit_trajectory(ψ0, H_odd, H_even, n_steps; maxdim=nothing, cutoff=nothing) -> MPS

Applica `n_steps` passi del circuito (layer dispari poi pari ad ogni passo),
troncando ad ogni `apply` con `maxdim`/`cutoff`. Questa è la parte
effettivamente cronometrata da `@benchmarkable`.
"""
function run_circuit_trajectory(ψ0, H_odd, H_even, n_steps;
                                 maxdim=nothing, cutoff=nothing,
                                 sweep_maxdim=nothing, sweep_cutoff=nothing)
    ψ = ψ0
    for _ in 1:n_steps
        ψ = StreamTensor.apply(H_odd, ψ; maxdim, cutoff, sweep_maxdim, sweep_cutoff)
        ψ = StreamTensor.apply(H_even, ψ; maxdim, cutoff, sweep_maxdim, sweep_cutoff)
    end
    return ψ
end