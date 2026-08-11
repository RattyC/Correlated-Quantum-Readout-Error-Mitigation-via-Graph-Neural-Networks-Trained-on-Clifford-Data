# eval_analytical_baseline.py
import argparse
import json

import numpy as np

from prepare_joint_dataset import marginal_joint_distribution


def kl_divergence(p, q, eps=1e-8):
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


def total_variation(p, q):
    return float(0.5 * np.sum(np.abs(p - q)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--matrices", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.matrices, "r") as f:
        matrices = json.load(f)
    with open(args.input, "r") as f:
        raw_data = json.load(f)

    kl_scores, tv_scores = [], []
    negative_entry_count, total_pairs = 0, 0

    for item in raw_data:
        for a, b in item.get("correlated_pairs", []):
            key = f"{a}_{b}"
            if key not in matrices:
                continue
            M = np.array(matrices[key]["M"])
            noisy = marginal_joint_distribution(item["noisy_outputs"], a, b).numpy()
            ideal = marginal_joint_distribution(item["ideal_outputs"], a, b).numpy()

            corrected = M @ noisy
            total_pairs += 1
            if np.any(corrected < 0):
                negative_entry_count += 1

            corrected_clipped = np.clip(corrected, 0, None)
            corrected_clipped = corrected_clipped / corrected_clipped.sum()

            kl_scores.append(kl_divergence(ideal, corrected_clipped))
            tv_scores.append(total_variation(ideal, corrected_clipped))

    print(f"Total (circuit, pair) examples evaluated: {total_pairs}")
    print(f"Mean KL divergence (ideal || corrected): {np.mean(kl_scores):.6f}")
    print(f"Mean total variation distance: {np.mean(tv_scores):.6f}")
    print(f"Fraction with negative raw entries before clipping: {negative_entry_count/total_pairs:.4f}")


if __name__ == "__main__":
    main()