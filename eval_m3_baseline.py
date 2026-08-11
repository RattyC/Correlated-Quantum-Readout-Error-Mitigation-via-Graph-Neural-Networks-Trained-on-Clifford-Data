# eval_m3_baseline.py
"""Phase C: M3 baseline via mthree.cals_from_matrices() (no live backend needed).

Key thesis-relevant fact, confirmed directly from mthree source (mitigation.py):
single_qubit_cals is a list of independent 2x2 matrices, one per qubit — M3's correction
model is a TENSOR PRODUCT of per-qubit calibrations with no cross-qubit term anywhere in the
formula. This means M3 cannot represent or correct genuinely correlated crosstalk error by
construction, regardless of how it's calibrated or how much data it's given.
"""
import argparse
import json

import numpy as np
import mthree

from prepare_joint_dataset import marginal_joint_distribution


def counts_to_bitstring_keys(joint4: np.ndarray) -> dict:
    """[P00,P01,P10,P11] (idx = bit_a | (bit_b<<1)) -> mthree counts dict, qubits=[a,b]
    convention (position 0 in qubits list = classical bit 0 = rightmost bitstring char)."""
    counts = {}
    for idx, p in enumerate(joint4):
        bit_a = idx & 1
        bit_b = (idx >> 1) & 1
        key = f"{bit_b}{bit_a}"
        counts[key] = counts.get(key, 0.0) + float(p) * 100000  # nominal shot scale
    return counts


def quasi_to_joint4(quasi, qubit_a_pos=0, qubit_b_pos=1) -> np.ndarray:
    """mthree QuasiDistribution -> our [P00,P01,P10,P11] vector, renormalized (quasi-probs
    can be slightly negative/unnormalized by design — clip + renormalize for fair KL/TV)."""
    joint4 = np.zeros(4)
    for bitstring, prob in quasi.items():
        bit_b = int(bitstring[0])
        bit_a = int(bitstring[1])
        idx = bit_a | (bit_b << 1)
        joint4[idx] = prob
    joint4 = np.clip(joint4, 0, None)
    total = joint4.sum()
    if total > 0:
        joint4 = joint4 / total
    return joint4


def kl_divergence(p, q, eps=1e-8):
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


def total_variation(p, q):
    return float(0.5 * np.sum(np.abs(p - q)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--single-qubit-cals", type=str, required=True)
    parser.add_argument("--num-qubits", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.single_qubit_cals, "r") as f:
        cal_list = json.load(f)
    matrices = [np.array(m, dtype=np.float32) for m in cal_list]

    mit = mthree.M3Mitigation(system=None)
    mit.num_qubits = args.num_qubits
    mit.cals_from_matrices(matrices)

    with open(args.input, "r") as f:
        raw_data = json.load(f)

    kl_scores, tv_scores = [], []
    for item in raw_data:
        for a, b in item.get("correlated_pairs", []):
            noisy4 = marginal_joint_distribution(item["noisy_outputs"], a, b).numpy()
            ideal4 = marginal_joint_distribution(item["ideal_outputs"], a, b).numpy()

            counts_dict = counts_to_bitstring_keys(noisy4)
            quasi = mit.apply_correction(counts_dict, qubits=[a, b])
            corrected4 = quasi_to_joint4(quasi)

            kl_scores.append(kl_divergence(ideal4, corrected4))
            tv_scores.append(total_variation(ideal4, corrected4))

    print(f"Evaluated {len(kl_scores)} (circuit, pair) examples")
    print(f"Mean KL divergence (ideal || M3-corrected): {np.mean(kl_scores):.6f}")
    print(f"Mean total variation distance: {np.mean(tv_scores):.6f}")


if __name__ == "__main__":
    main()