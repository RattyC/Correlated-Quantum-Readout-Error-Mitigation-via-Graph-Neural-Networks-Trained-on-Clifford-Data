# eval_zero_shot_transfer.py
"""Zero-shot cross-qubit-count transfer check: load a JointCorrectionMLP trained on one
qubit count's pair data, evaluate directly (no retraining) on a different qubit count's pair
data. The model's input is a fixed 4-dim joint vector with no N-dependence anywhere in its
architecture, so this should work near-identically to same-N training by construction — this
script exists to CONFIRM that claim empirically, not to discover new behavior.
"""
import argparse

import torch
from torch.utils.data import DataLoader

from prepare_joint_dataset import JointPairDataset
from train_joint_mlp import JointCorrectionMLP, kl_per_example, tv_per_example


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved model (from --save-model)")
    parser.add_argument("--eval-input", type=str, required=True, help="Dataset JSON to evaluate on (different N than training)")
    parser.add_argument("--hidden-dim", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()

    model = JointCorrectionMLP(hidden_dim=args.hidden_dim)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    dataset = JointPairDataset(args.eval_input)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    total_kl, total_tv, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x)
            total_kl += kl_per_example(pred, y).sum().item()
            total_tv += tv_per_example(pred, y).sum().item()
            n += x.size(0)

    print(f"Zero-shot eval on {args.eval_input} ({n} examples, no retraining):")
    print(f"  KL divergence: {total_kl / n:.6f}")
    print(f"  TV distance:   {total_tv / n:.6f}")


if __name__ == "__main__":
    main()