"""
eval_ibu_baseline.py

Iterative Bayesian Unfolding (IBU) baseline for joint-pair readout error correction.
Reference: Nachman et al., "Unfolding quantum computer readout noise" (2020);
Pokharel/Srinivasan et al., "Scalable measurement error mitigation via iterative
Bayesian unfolding", Phys. Rev. Research 6, 013187 (2024).

Reuses the same assignment-matrix construction and joint-distribution extraction
as calibrate_pair_matrix.py / eval_analytical_baseline.py, so results are
directly comparable in the same KL/TV table as pinv, M3, and the learned MLP.

Unlike naive pinv, IBU is guaranteed to return a valid probability distribution
(non-negative, normalized) at every iteration -- no clipping/renormalization hacks
needed. This makes it a materially stronger classical baseline than pinv.
"""
import argparse
import json
import numpy as np

from src.simulator import execute_circuit_pipeline
from prepare_joint_dataset import marginal_joint_distribution


def build_assignment_matrix(qubit_a, qubit_b, num_qubits, shots):
    """Empirical 4x4 assignment matrix A[measured_idx, prepared_idx].
    Index convention matches marginal_joint_distribution: idx = bit_a | (bit_b << 1).
    """
    from qiskit import QuantumCircuit

    A = np.zeros((4, 4))
    for prep_idx in range(4):
        bit_a = prep_idx & 1
        bit_b = (prep_idx >> 1) & 1
        qc = QuantumCircuit(num_qubits, num_qubits)
        if bit_a:
            qc.x(qubit_a)
        if bit_b:
            qc.x(qubit_b)
        qc.measure(range(num_qubits), range(num_qubits))
        counts = execute_circuit_pipeline(qc, shots=shots, use_noise=True)
        A[:, prep_idx] = marginal_joint_distribution(counts, qubit_a, qubit_b).detach().numpy()
    return A


def ibu_correct(p_noisy, A, iterations=100, tol=1e-9):
    """D'Agostini iterative Bayesian unfolding.
    A[i,j] = P(measured=i | true=j). Guarantees a valid probability distribution
    at every step -- no negative entries, no post-hoc clipping.
    """
    p_est = np.clip(p_noisy.copy(), 1e-12, None)
    p_est /= p_est.sum()

    for _ in range(iterations):
        denom = np.clip(A @ p_est, 1e-12, None)
        ratio = p_noisy / denom
        p_new = p_est * (A.T @ ratio)
        p_new = np.clip(p_new, 0.0, None)
        s = p_new.sum()
        if s > 0:
            p_new /= s
        if np.max(np.abs(p_new - p_est)) < tol:
            p_est = p_new
            break
        p_est = p_new

    return p_est


def kl_divergence(p_ideal, p_pred, eps=1e-10):
    p = np.clip(p_ideal, eps, None)
    q = np.clip(p_pred, eps, None)
    return float(np.sum(p * np.log(p / q)))


def total_variation(p_ideal, p_pred):
    return float(0.5 * np.sum(np.abs(p_ideal - p_pred)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to dataset JSON")
    parser.add_argument("--cal-shots", type=int, default=100000,
                         help="Shots used to build the assignment matrix A per pair")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", default=None, help="Optional path to save per-pair summary JSON")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        raw_data = json.load(f)

    num_qubits = raw_data[0]["num_qubits"]
    print(f"Loaded {len(raw_data)} circuits, num_qubits={num_qubits}")

    # Collect the set of correlated pairs present in the dataset
    all_pairs = set()
    for item in raw_data:
        for pair in item["correlated_pairs"]:
            all_pairs.add(tuple(pair))

    print(f"Found {len(all_pairs)} correlated pair(s): {sorted(all_pairs)}")

    summary = {}
    for (qa, qb) in sorted(all_pairs):
        print(f"\n--- Pair ({qa},{qb}) ---")
        A = build_assignment_matrix(qa, qb, num_qubits, args.cal_shots)
        print("Assignment matrix A:\n", np.round(A, 4))

        kl_list, tv_list = [], []
        for item in raw_data:
            if [qa, qb] not in item["correlated_pairs"] and (qa, qb) not in item["correlated_pairs"]:
                continue
            p_noisy = marginal_joint_distribution(item["noisy_outputs"], qa, qb).detach().numpy()
            p_ideal = marginal_joint_distribution(item["ideal_outputs"], qa, qb).detach().numpy()
            p_corrected = ibu_correct(p_noisy, A, iterations=args.iterations)
            kl_list.append(kl_divergence(p_ideal, p_corrected))
            tv_list.append(total_variation(p_ideal, p_corrected))

        mean_kl = float(np.mean(kl_list))
        mean_tv = float(np.mean(tv_list))
        print(f"n={len(kl_list)}  mean KL={mean_kl:.5f}  mean TV={mean_tv:.5f}")
        summary[f"{qa}_{qb}"] = {"n": len(kl_list), "mean_kl": mean_kl, "mean_tv": mean_tv}

    print("\n=== IBU Baseline Summary ===")
    for pair, stats in summary.items():
        print(f"pair {pair}: KL={stats['mean_kl']:.5f}  TV={stats['mean_tv']:.5f}  (n={stats['n']})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()