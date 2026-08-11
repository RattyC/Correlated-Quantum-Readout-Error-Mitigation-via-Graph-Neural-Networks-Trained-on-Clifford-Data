# src/simulator.py
"""Ideal and noisy execution of a Clifford circuit via Qiskit Aer.

method="stabilizer" is required, not just preferred: it is what keeps
simulation polynomial-time for Clifford circuits. Falling back to a
statevector/automatic method would defeat the reason Clifford circuits
were chosen in the first place.
"""
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError

_BACKEND = AerSimulator(method="stabilizer")


def _build_readout_noise_model(num_qubits: int, p1_given_0: float = 0.02, p0_given_1: float = 0.03) -> NoiseModel:
    """Build a simple per-qubit, uncorrelated readout (bit-flip) noise model.

    p1_given_0: P(measure '1' | true state |0>)
    p0_given_1: P(measure '0' | true state |1>)

    Values are asymmetric (p0_given_1 > p1_given_0) because real
    superconducting hardware typically shows higher |1>->|0> decay-driven
    readout error than the reverse. This model is intentionally simple
    (uncorrelated per-qubit) — it does NOT reproduce cross-qubit
    crosstalk correlation; it only gives the GNN something non-trivial
    to learn to invert. Swap in a correlated / device-calibrated noise
    model before making any claim about real hardware performance.
    """
    noise_model = NoiseModel()
    error_matrix = [[1 - p1_given_0, p1_given_0], [p0_given_1, 1 - p0_given_1]]
    readout_error = ReadoutError(error_matrix)
    for qubit in range(num_qubits):
        noise_model.add_readout_error(readout_error, [qubit])
    return noise_model


def execute_circuit_pipeline(qc: QuantumCircuit, shots: int, use_noise: bool) -> dict:
    """Run a measured Clifford circuit and return bitstring -> count dict.

    Args:
        qc: measured QuantumCircuit (Clifford-only).
        shots: number of measurement shots.
        use_noise: if True, apply the per-qubit readout noise model.

    Returns:
        dict mapping bitstring -> count.
    """
    if use_noise:
        noise_model = _build_readout_noise_model(qc.num_qubits)
        result = _BACKEND.run(qc, shots=shots, noise_model=noise_model).result()
    else:
        result = _BACKEND.run(qc, shots=shots).result()
    return result.get_counts(qc)
