# src/simulator.py
"""Ideal and noisy execution of a Clifford circuit via Qiskit Aer.

method="stabilizer" is required, not just preferred: it is what keeps
simulation polynomial-time for Clifford circuits.

Readout noise model: correlated-pair crosstalk, not independent per-qubit.
A fraction of qubits are randomly grouped into pairs (NOT tied to any coupling
map / nearest-neighbor structure — see Maciejewski, Baccari, Zimboras, Oszmaniec,
"Modeling and mitigation of cross-talk effects in readout noise...", Quantum 5, 464
(2021): measured correlations on real hardware do not follow the physical qubit
layout, so random pairing is the faithful choice, not a simplification). Each pair
shares a single crosstalk event that flips both qubits together with probability
`SHARED_FLIP_PROB`, mixed with the old independent per-qubit error at weight `rho`.
Unpaired qubits keep the original independent single-qubit error.
"""
from functools import lru_cache
import random

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError

_BACKEND = AerSimulator(method="stabilizer")

# --- correlated noise model parameters ---
P1_GIVEN_0 = 0.02           # P(measure '1' | true state |0>) — per-qubit marginal
P0_GIVEN_1 = 0.03           # P(measure '0' | true state |1>) — per-qubit marginal
CORRELATION_RHO = 0.5       # 0.0 = fully independent (old model), 1.0 = fully correlated
SHARED_FLIP_PROB = 0.05     # probability the shared crosstalk event fires on a pair
CORRELATED_PAIR_FRACTION = 0.5  # fraction of qubits placed into correlated pairs
NOISE_MODEL_SEED = 20260811     # fixed so pair topology is reproducible per num_qubits


def _single_qubit_matrix(p1_given_0: float, p0_given_1: float):
    return [[1 - p1_given_0, p1_given_0], [p0_given_1, 1 - p0_given_1]]


def _independent_pair_matrix(p1_given_0: float, p0_given_1: float):
    """4x4 joint matrix = product of two independent single-qubit marginals."""
    m = _single_qubit_matrix(p1_given_0, p0_given_1)
    mat = [[0.0] * 4 for _ in range(4)]
    for true_a in range(2):
        for true_b in range(2):
            row = true_a | (true_b << 1)
            for obs_a in range(2):
                for obs_b in range(2):
                    col = obs_a | (obs_b << 1)
                    mat[row][col] = m[true_a][obs_a] * m[true_b][obs_b]
    return mat


def _shared_flip_pair_matrix(p_shared_flip: float):
    """4x4 joint matrix for a single shared crosstalk event: with probability
    p_shared_flip BOTH qubits flip together; otherwise both read correctly.
    Not decomposable into independent per-qubit noise for any p_shared_flip > 0 —
    this is the genuinely correlated component."""
    mat = [[0.0] * 4 for _ in range(4)]
    for true_a in range(2):
        for true_b in range(2):
            row = true_a | (true_b << 1)
            flipped_col = (1 - true_a) | ((1 - true_b) << 1)
            mat[row][row] += 1 - p_shared_flip
            mat[row][flipped_col] += p_shared_flip
    return mat


def _correlated_pair_matrix(p1_given_0, p0_given_1, rho, p_shared_flip):
    indep = _independent_pair_matrix(p1_given_0, p0_given_1)
    shared = _shared_flip_pair_matrix(p_shared_flip)
    return [
        [(1 - rho) * indep[i][j] + rho * shared[i][j] for j in range(4)]
        for i in range(4)
    ]


@lru_cache(maxsize=None)
def _build_correlated_noise_model(num_qubits: int):
    """Fixed correlated-pair readout noise model for a given qubit count. Cached + seeded
    so every circuit in a dataset run shares the same 'device' topology (which qubit pairs
    are crosstalk-correlated), instead of re-randomizing per circuit."""
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
    unpaired_qubits = qubit_order[idx:]

    noise_model = NoiseModel()

    pair_matrix = _correlated_pair_matrix(P1_GIVEN_0, P0_GIVEN_1, CORRELATION_RHO, SHARED_FLIP_PROB)
    pair_error = ReadoutError(pair_matrix)
    for a, b in correlated_pairs:
        noise_model.add_readout_error(pair_error, [a, b])

    single_matrix = _single_qubit_matrix(P1_GIVEN_0, P0_GIVEN_1)
    single_error = ReadoutError(single_matrix)
    for q in unpaired_qubits:
        noise_model.add_readout_error(single_error, [q])

    return noise_model, tuple(correlated_pairs)


def get_correlated_pairs(num_qubits: int):
    """Ground-truth metadata: which qubit pairs are crosstalk-correlated for this qubit
    count. Needed later to check whether the GAT's attention actually learns the
    correlation structure, not just to build the noise model."""
    _, pairs = _build_correlated_noise_model(num_qubits)
    return [list(p) for p in pairs]


def execute_circuit_pipeline(qc: QuantumCircuit, shots: int, use_noise: bool) -> dict:
    """Run a measured Clifford circuit and return bitstring -> count dict."""
    if use_noise:
        noise_model, _ = _build_correlated_noise_model(qc.num_qubits)
        result = _BACKEND.run(qc, shots=shots, noise_model=noise_model).result()
    else:
        result = _BACKEND.run(qc, shots=shots).result()
    return result.get_counts(qc)