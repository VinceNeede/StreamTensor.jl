# ---------------------------------------------------------------------------
# Verifica diretta: ψ_st e ψ_it rappresentano davvero lo stesso stato fisico?
#
# Materializza entrambe le MPS a vettore denso (2^L complessi/reali) e
# confronta via overlap (invariante a fase globale, che non ha significato
# fisico). Pensato per L piccolo (4-8), NON per i problemi di benchmark veri
# (2^20, 2^50 sono intrattabili come vettore denso).
#
# Convenzione di appiattimento usata per ENTRAMBE le materializzazioni,
# per garantire che siano confrontabili: sito 1 è la componente a variazione
# più veloce nel vettore appiattito (site1 fastest, siteL slowest).
#
# Uso:
#   julia --project=benchmark benchmark/verify_circuit_equivalence.jl
# ---------------------------------------------------------------------------

using LinearAlgebra
using StreamTensor
using ITensors, ITensorMPS

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "circuit.jl"))
include(joinpath(@__DIR__, "itensor_circuit.jl"))

"""
    mps_to_dense(ψ) -> Vector

Materializza una StreamTensor.MPS a vettore denso (site1 fastest-varying).
"""
function mps_to_dense(ψ)
    L = length(ψ.tensors)
    acc = dropdims(ψ.tensors[1].storage; dims=1)   # (d1, χ1) — via left boundary dim1
    for i in 2:L
        T = ψ.tensors[i].storage                    # (χ_{i-1}, d_i, χ_i)
        χprev, d_i, χi = size(T)
        M = reshape(acc, :, χprev) * reshape(T, χprev, d_i * χi)
        acc = reshape(M, :, χi)
    end
    return vec(acc)   # shape finale (2^L, 1) — via right boundary dim1
end

"""
    itensor_mps_to_dense(ψ, sites) -> Vector

Materializza una ITensorMPS.MPS a vettore denso (site1 fastest-varying),
stessa convenzione di `mps_to_dense`.
"""
function itensor_mps_to_dense(ψ, sites)
    T = prod(ψ)                  # ITensor unico con tutti gli indici di sito
    A = Array(T, sites...)       # ordine assi = sites[1],...,sites[L]
    return vec(A)
end

"""
    compare_states(v_st, v_it; label="") -> NamedTuple

Confronta due vettori a meno di fase globale (irrilevante fisicamente):
overlap ≈ 1 ⟹ stesso stato fisico. Stampa anche ‖v_st - v_it‖ grezzo, utile
solo se l'overlap è già ≈1 (altrimenti la fase lo rende poco informativo).
"""
function compare_states(v_st, v_it; label="")
    overlap = abs(dot(v_st, v_it)) / (norm(v_st) * norm(v_it))
    println("── $label ──")
    println("  ‖v_st‖ = ", norm(v_st), "   ‖v_it‖ = ", norm(v_it))
    println("  |⟨v_st|v_it⟩| / (‖v_st‖‖v_it‖) = ", overlap, "   (atteso ≈ 1 se stesso stato fisico)")
    println("  ‖v_st - v_it‖ (grezzo, sensibile a fase globale) = ", norm(v_st - v_it))
    return (; overlap, raw_diff = norm(v_st - v_it))
end

# ---------------------------------------------------------------------------
# Test 1: stato iniziale ψ0 (nessun gate applicato) — verifica che "Up" abbia
# lo stesso significato fisico nelle due librerie, prima di introdurre
# qualsiasi complicazione del circuito.
# ---------------------------------------------------------------------------

L = 4
sites_st = StreamTensor.siteinds("SpinHalf", L)
sites_it = ITensorMPS.siteinds("S=1/2", L)

ψ0_st = build_quench_state(sites_st)
ψ0_it = ITensorMPS.MPS(sites_it, fill("Up", L))

compare_states(mps_to_dense(ψ0_st), itensor_mps_to_dense(ψ0_it, sites_it); label="ψ0 (stato iniziale)")

# ---------------------------------------------------------------------------
# Test 2: dopo 1 solo layer (start=1), nessun troncamento — isola la
# correttezza della costruzione del gate/MPO dal resto della traiettoria.
# ---------------------------------------------------------------------------

H_odd_st = build_gate_layer_mpo(sites_st, _CIRCUIT_GATE; start=1)
StreamTensor.orthogonalize!(H_odd_st, 1)
ψ1_st = StreamTensor.apply(H_odd_st, ψ0_st)

H_odd_it = _itensor_gate_layer_mpo(sites_it, _CIRCUIT_GATE; start=1)
ψ1_it = ITensorMPS.apply(H_odd_it, ψ0_it; alg="zipup", cutoff=0.0, maxdim=typemax(Int), mindim=1)

compare_states(mps_to_dense(ψ1_st), itensor_mps_to_dense(ψ1_it, sites_it); label="dopo 1 layer (start=1)")

# ---------------------------------------------------------------------------
# Test 3: dopo un passo completo (odd poi even), nessun troncamento.
# ---------------------------------------------------------------------------

ψ2_st = run_circuit_trajectory(ψ0_st, H_odd_st, let
    H = build_gate_layer_mpo(sites_st, _CIRCUIT_GATE; start=2)
    StreamTensor.orthogonalize!(H, 1)
    H
end, 1)

ψ2_it = run_itensor_circuit_trajectory(ψ0_it, sites_it, 1)

compare_states(mps_to_dense(ψ2_st), itensor_mps_to_dense(ψ2_it, sites_it); label="dopo 1 passo completo (odd+even)")