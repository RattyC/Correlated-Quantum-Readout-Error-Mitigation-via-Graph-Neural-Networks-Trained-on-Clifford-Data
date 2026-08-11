# calibrate_single_qubit_m3.py
"""Calibrate per-qubit 2x2 matrices the same way M3's 'independent' method does internally
(confirmed from mthree source: prep |0>, measure -> P00/P10; prep |1>, measure -> P01/P11).
Saved matrices feed mthree via cals_from_matrices() — no live backend needed.
"""
import argparse
import json

import numpy as np
from qiskit import QuantumCircuit

from src.simulator import execute_circuit_pipeline


def _prep_circuit(num_qubits, qubit, state):
    qc = QuantumCircuit(num_qubits, num_qubits)
    if state == 1:
        qc.x(qubit)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def calibrate_qubit(num_qubits, qubit, shots):
    qc0 = _prep_circuit(num_qubits, qubit, 0)
    counts0 = execute_circuit_pipeline(qc0, shots=shots, use_noise=True)
    total0 = sum(counts0.values())
    p10 = sum(v for k, v in counts0.items() if list(reversed(k))[qubit] == '1') / total0
    p00 = 1 - p10

    qc1 = _prep_circuit(num_qubits, qubit, 1)
    counts1 = execute_circuit_pipeline(qc1, shots=shots, use_noise=True)
    total1 = sum(counts1.values())
    p01 = sum(v for k, v in counts1.items() if list(reversed(k))[qubit] == '0') / total1
    p11 = 1 - p01

    cal = np.zeros((2, 2), dtype=np.float64)
    cal[:, 0] = [p00, p10]
    cal[:, 1] = [p01, p11]
    return cal


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-qubits", type=int, required=True)
    parser.add_argument("--shots", type=int, default=50000)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    matrices = []
    for q in range(args.num_qubits):
        cal = calibrate_qubit(args.num_qubits, q, args.shots)
        print(f"Qubit {q}: cal =\n{cal}")
        matrices.append(cal.tolist())

    output_path = args.output or f"output_data/m3_single_qubit_cals_q{args.num_qubits}.json"
    with open(output_path, "w") as f:
        json.dump(matrices, f)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()