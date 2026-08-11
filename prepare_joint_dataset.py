# prepare_joint_dataset.py
"""v2: extract per-pair joint distributions for correlated qubit pairs (k=2 local correction).
Unpaired qubits are NOT included here — they keep the v1 per-qubit <Z> pipeline.

Each example = one (circuit, pair) combination:
  x = noisy joint 4-vector [P(00), P(01), P(10), P(11)]
  y = ideal joint 4-vector, same marginalization on ideal_outputs.

Bit convention: qubit index i = reversed(bitstring)[i]. Vector index = bit_a | (bit_b << 1).
"""
import argparse
import json

import torch
from torch.utils.data import Dataset


def marginal_joint_distribution(counts: dict, qubit_a: int, qubit_b: int) -> torch.Tensor:
    total = sum(counts.values())
    joint = torch.zeros(4)
    for bitstring, count in counts.items():
        bits = list(reversed(bitstring))
        bit_a = int(bits[qubit_a])
        bit_b = int(bits[qubit_b])
        idx = bit_a | (bit_b << 1)
        joint[idx] += count / total
    return joint


class JointPairDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            raw_data = json.load(f)

        self.examples = []
        for item in raw_data:
            correlated_pairs = item.get("correlated_pairs", [])
            if not correlated_pairs:
                continue
            noisy_counts = item["noisy_outputs"]
            ideal_counts = item["ideal_outputs"]
            for a, b in correlated_pairs:
                x = marginal_joint_distribution(noisy_counts, a, b)
                y = marginal_joint_distribution(ideal_counts, a, b)
                self.examples.append((x, y))

        print(f"Extracted {len(self.examples)} (circuit, pair) joint-distribution examples "
              f"from {len(raw_data)} circuits.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def _unit_test():
    """Hand-verified — 3 qubits, 1 pair (0,1), known counts."""
    counts = {"000": 50, "001": 30, "010": 15, "011": 5}
    expected = torch.tensor([0.5, 0.3, 0.15, 0.05])
    result = marginal_joint_distribution(counts, qubit_a=0, qubit_b=1)
    assert torch.allclose(result, expected, atol=1e-6), f"Unit test FAILED: got {result}, expected {expected}"
    print("Unit test PASSED:", result.tolist())


def parse_args():
    parser = argparse.ArgumentParser(description="Extract per-pair joint distributions (v2, k=2 local correction).")
    parser.add_argument("--input", type=str, default="output_data/quantum_dataset_q7_correlated_v2.json")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _unit_test()
    if args.self_test:
        raise SystemExit(0)

    dataset = JointPairDataset(args.input)
    x0, y0 = dataset[0]
    print(f"\nExample 0 - Noisy joint: {x0.tolist()} | Ideal joint: {y0.tolist()}")
    print(f"Noisy joint sums to: {x0.sum().item():.6f} (should be 1.0)")
    print(f"Ideal joint sums to: {y0.sum().item():.6f} (should be 1.0)")