# calibrate_correlations.py
"""Calibration step: estimate which qubit pairs have correlated readout error using ONLY
measurement statistics from a dedicated calibration circuit — no ground-truth pair info.

Mirrors the calibration cost any device-specific method (including M3) has to pay before use.
Prepares |0...0>, measures many shots under the SAME noise model the training data will be
generated with, and estimates per-pair correlation from the resulting bitstring counts.
"""
import argparse
import itertools
import json

from qiskit import QuantumCircuit

from src.simulator import execute_circuit_pipeline, get_correlated_pairs


def _bit_moments(counts: dict, num_qubits: int):
    """E[bit_i] and E[bit_i * bit_j] for all i, j, computed exactly from aggregated counts."""
    total = sum(counts.values())
    mean = [0.0] * num_qubits
    joint = [[0.0] * num_qubits for _ in range(num_qubits)]

    for bitstring, count in counts.items():
        prob = count / total
        bits = [int(b) for b in reversed(bitstring)]  # qubit 0 = rightmost, matches rest of repo
        for i in range(num_qubits):
            mean[i] += prob * bits[i]
        for i in range(num_qubits):
            for j in range(num_qubits):
                joint[i][j] += prob * bits[i] * bits[j]

    return mean, joint


def _pearson_correlation(mean, joint, i, j):
    cov = joint[i][j] - mean[i] * mean[j]
    var_i = joint[i][i] - mean[i] ** 2
    var_j = joint[j][j] - mean[j] ** 2
    denom = (var_i * var_j) ** 0.5
    if denom < 1e-12:
        return 0.0
    return cov / denom


def calibrate(num_qubits: int, shots: int, threshold: float):
    calib_circuit = QuantumCircuit(num_qubits, num_qubits)
    calib_circuit.measure(range(num_qubits), range(num_qubits))  # |0...0> prep, no gates

    counts = execute_circuit_pipeline(calib_circuit, shots=shots, use_noise=True)
    mean, joint = _bit_moments(counts, num_qubits)

    scored_pairs = []
    for i, j in itertools.combinations(range(num_qubits), 2):
        score = abs(_pearson_correlation(mean, joint, i, j))
        scored_pairs.append(((i, j), score))
    scored_pairs.sort(key=lambda x: -x[1])
    total = sum(counts.values())
    top_counts = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
    print(f"Total distinct outcomes: {len(counts)} | total shots counted: {total}")
    print("Top-10 raw outcomes (bitstring: count, fraction):")
    for bitstring, count in top_counts:
        print(f"  {bitstring}: {count} ({count/total:.5f})")
    all_zero = "0" * num_qubits
    print(f"P(all-zero outcome) = {counts.get(all_zero, 0)/total:.5f}  (should be ~0.95 if noise is applying, ~1.0 if not)")

    detected_pairs = [list(pair) for pair, score in scored_pairs if score > threshold]
    return detected_pairs, scored_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate correlated-pair readout structure from measurement statistics only.")
    parser.add_argument("--num-qubits", type=int, required=True)
    parser.add_argument("--shots", type=int, default=100000, help="Calibration shots — run once per 'device', much cheaper than full dataset generation.")
    parser.add_argument("--threshold", type=float, default=0.02, help="Absolute Pearson correlation threshold to flag a pair as correlated.")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    detected_pairs, scored_pairs = calibrate(args.num_qubits, args.shots, args.threshold)

    print(f"Calibration shots: {args.shots}")
    print("Top-scoring pairs (|Pearson correlation|):")
    for (i, j), score in scored_pairs[:10]:
        print(f"  ({i}, {j}): {score:.4f}")

    print(f"\nDetected pairs (score > {args.threshold}): {detected_pairs}")

    # Diagnostic-only comparison against ground truth — would NOT exist on real hardware,
    # this is purely to validate the calibration method itself while still in simulation.
    ground_truth = get_correlated_pairs(args.num_qubits)
    ground_truth_set = {tuple(sorted(p)) for p in ground_truth}
    detected_set = {tuple(sorted(p)) for p in detected_pairs}
    true_positives = ground_truth_set & detected_set
    print(f"\n[diagnostic-only, uses ground truth] True pairs: {ground_truth}")
    print(f"[diagnostic-only] Precision: {len(true_positives)}/{max(len(detected_set),1)} | Recall: {len(true_positives)}/{len(ground_truth_set)}")

    output_path = args.output or f"output_data/detected_pairs_q{args.num_qubits}.json"
    with open(output_path, "w") as f:
        json.dump(detected_pairs, f)
    print(f"\nSaved detected pairs to {output_path}")


if __name__ == "__main__":
    main()