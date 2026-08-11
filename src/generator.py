# src/generator.py
"""Random Clifford-dominated circuit generation.

Clifford circuits are classically simulable in polynomial time
(Gottesman-Knill theorem, via stabilizer-generator tracking rather
than full 2^n statevector amplitudes). This is what makes it possible
to generate large {ideal, noisy} label pairs cheaply for this project.
"""
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_clifford_circuit

# Full single/two-qubit Clifford gate set. Restricting to these gates
# is what keeps every generated circuit classically simulable.
CLIFFORD_GATE_SET = ["cx", "cz", "swap", "h", "s", "sdg", "x", "y", "z"]


def generate_random_clifford_circuit(num_qubits: int, depth: int, seed: int | None = None) -> QuantumCircuit:
    """Generate a random circuit built entirely from Clifford gates and append measurement.

    Args:
        num_qubits: number of qubits in the circuit.
        depth: number of gate operations to place (passed as num_gates).
        seed: optional RNG seed for reproducibility.

    Returns:
        A measured QuantumCircuit ready to be passed to the simulator.
    """
    qc = random_clifford_circuit(
        num_qubits=num_qubits,
        num_gates=depth,
        gates=CLIFFORD_GATE_SET,
        seed=seed,
    )
    qc.measure_all()
    return qc
