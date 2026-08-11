# src/simulator.py
"""Ideal and noisy execution of a Clifford circuit via Qiskit Aer.

method="stabilizer" is required, not just preferred: it is what keeps
simulation polynomial-time for Clifford circuits.

Readout noise model: correlated-pair crosstalk, applied as PYTHON POST-PROCESSING on raw
per-shot samples, NOT via Aer's NoiseModel/ReadoutError. Reason: Qiskit Aer silently ignores
multi-qubit (joint) ReadoutError instructions — confirmed via Qiskit/qiskit-aer#319 and via a
direct calibration-circuit smoke test in this project (paired qubits showed zero error across
100,000 shots while the one independently-registered qubit showed the expected ~2% error rate).
Relying on NoiseModel for the correlated component would silently produce uncorrelated data.

A fraction of qubits are randomly grouped into pairs (NOT tied to any coupling map / nearest-
neighbor structure — see Maciejewski, Baccari, Zimboras, Oszmaniec, "Modeling and mitigation of
cross-talk effects in readout noise...", Quantum 5, 464 (2021): measured correlations on real
hardware do not follow the physical qubit layout). Each pair shares a single crosstalk event
that flips both qubits together with probability SHARED_FLIP_PROB, mixed with the independent
per-qubit error at weight CORRELATION_RHO. Unpaired qubits keep independent single-qubit error.
"""
import random

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

_BACKEND = AerSimulator(method="stabilizer")

# --- correlated noise model parameters ---
P1_GIVEN_0 = 0.02           # P(measure '1' | true state |0>) — per-qubit marginal
P0_GIVEN_1 = 0.03           # P(measure '0' | true state |1>) — per-qubit marginal
CORRELATION_RHO = 0.5       # 0.0 = fully independent, 1.0 = fully correlated (shared-flip only)
SHARED_FLIP_PROB = 0.05     # probability the shared crosstalk event fires on a pair
CORRELATED_PAIR_FRACTION = 1.0  # fraction of qubits placed into correlated pairs
NOISE_MODEL_SEED = 20260811     # fixed so pair TOPOLOGY is reproducible per num_qubits


def _build_pair_topology(num_qubits: int):
    """Deterministic (seeded) pair topology for a given qubit count — the WHICH-QUBITS-ARE-
    CORRELATED structure, not the actual noise sampling."""
    rng = random.Random(NOISE_MODEL_SEED + num_qubits)
    qubit_order = list(range(num_qubits))
    rng.shuffle(qubit_order)

    num_correlated_pairs = int(round((num_qubits // 2) * CORRELATED_PAIR_FRACTION))

    correlated_pairs = []
    idx = 0
    for _ in range(num_correlated_pairs):
        a, b = qubit_order[idx], qubit_order[idx + 1]
        correlated_pairs.append((a, b))
        idx += 2

    return tuple(correlated_pairs)


def get_correlated_pairs(num_qubits: int):
    """Ground-truth metadata: which qubit pairs are crosstalk-correlated for this qubit count."""
    return [list(p) for p in _build_pair_topology(num_qubits)]


def _apply_correlated_noise(bits_matrix: np.ndarray, correlated_pairs, rng: np.random.Generator):
    """bits_matrix: (shots, num_qubits) int8 array of ideal (noiseless) measurement outcomes.
    Returns a new array with correlated-pair + independent readout noise applied."""
    num_qubits = bits_matrix.shape[1]
    shots = bits_matrix.shape[0]
    noisy = bits_matrix.copy()

    paired_qubits = {q for pair in correlated_pairs for q in pair}
    unpaired_qubits = [q for q in range(num_qubits) if q not in paired_qubits]

    for q in unpaired_qubits:
        col = bits_matrix[:, q]
        p_flip = np.where(col == 0, P1_GIVEN_0, P0_GIVEN_1)
        flip_mask = rng.random(shots) < p_flip
        noisy[flip_mask, q] = 1 - noisy[flip_mask, q]

    for a, b in correlated_pairs:
        use_shared = rng.random(shots) < CORRELATION_RHO

        shared_fires = use_shared & (rng.random(shots) < SHARED_FLIP_PROB)
        noisy[shared_fires, a] = 1 - noisy[shared_fires, a]
        noisy[shared_fires, b] = 1 - noisy[shared_fires, b]

        indep_mask = ~use_shared
        for q in (a, b):
            col = bits_matrix[:, q]
            p_flip = np.where(col == 0, P1_GIVEN_0, P0_GIVEN_1)
            flip_mask = indep_mask & (rng.random(shots) < p_flip)
            noisy[flip_mask, q] = 1 - noisy[flip_mask, q]

    return noisy


def execute_circuit_pipeline(qc: QuantumCircuit, shots: int, use_noise: bool) -> dict:
    """Run a measured Clifford circuit and return bitstring -> count dict."""
    if not use_noise:
        result = _BACKEND.run(qc, shots=shots).result()
        return result.get_counts(qc)

    result = _BACKEND.run(qc, shots=shots, memory=True).result()
    raw_shots = result.get_memory(qc)

    num_qubits = qc.num_qubits
    correlated_pairs = _build_pair_topology(num_qubits)

    bits_matrix = np.array(
        [[int(b) for b in reversed(bs)] for bs in raw_shots],
        dtype=np.int8,
    )

    rng = np.random.default_rng()
    noisy_matrix = _apply_correlated_noise(bits_matrix, correlated_pairs, rng)

    noisy_counter = {}
    for row in noisy_matrix:
        bitstring = "".join(str(b) for b in row[::-1])
        noisy_counter[bitstring] = noisy_counter.get(bitstring, 0) + 1

    return noisy_counter
