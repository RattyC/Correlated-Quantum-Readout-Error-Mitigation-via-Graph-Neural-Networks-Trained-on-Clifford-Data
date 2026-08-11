# calibrate_pair_matrix.py
"""Phase B2: build the empirical 4x4 pair assignment matrix from dedicated calibration
circuits (prepare |00>, |01>, |10>, |11> on each correlated pair), then compute the classical
analytical-inversion correction (Moore-Penrose pseudo-inverse) — the "classical optimum"
ceiling any learned model must be compared against.

This deliberately reproduces the naive-inversion weakness the thesis motivation describes:
inverting A can and will produce negative "probabilities" — that's the literature-documented
failure mode being used as the baseline to beat, not a bug in this script.
"""
import argparse
import json

import numpy as np
from qiskit import QuantumCircuit

from src.simulator import execute_circuit_pipeline, get_correlated_pairs


def _prepared_state_circuit(num_qubits, qubit_a, state_a, qubit_b, state_b):
    qc = QuantumCircuit(num_qubits, num_qubits)
    if state_a == 1:
        qc.x(qubit_a)
    if state_b == 1:
        qc.x(qubit_b)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def _pair_marginal(counts, qubit_a, qubit_b):
    total = sum(counts.values())
    joint = np.zeros(4)
    for bitstring, count in counts.items():
        bits = list(reversed(bitstring))
        bit_a = int(bits[qubit_a])
        bit_b = int(bits[qubit_b])
        idx = bit_a | (bit_b << 1)
        joint[idx] += count / total
    return joint


def calibrate_pair_matrix(num_qubits, qubit_a, qubit_b, shots):
    """A[m] = P(observed | true=m), m in {00,01,10,11}, idx = a|(b<<1)."""
    A = np.zeros((4, 4))
    for m in range(4):
        state_a = m & 1
        state_b = (m >> 1) & 1
        qc = _prepared_state_circuit(num_qubits, qubit_a, state_a, qubit_b, state_b)
        counts = execute_circuit_pipeline(qc, shots=shots, use_noise=True)
        A[m] = _pair_marginal(counts, qubit_a, qubit_b)
    return A


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-qubits", type=int, required=True)
    parser.add_argument("--shots", type=int, default=50000)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = get_correlated_pairs(args.num_qubits)
    print(f"Correlated pairs: {pairs}")

    result = {}
    for a, b in pairs:
        A = calibrate_pair_matrix(args.num_qubits, a, b, args.shots)
        M = np.linalg.pinv(A)
        print(f"\nPair ({a},{b}) assignment matrix A (rows=true, cols=observed):\n{A}")
        print(f"Inversion matrix M = pinv(A):\n{M}")
        result[f"{a}_{b}"] = {"A": A.tolist(), "M": M.tolist()}

    output_path = args.output or f"output_data/pair_matrices_q{args.num_qubits}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()