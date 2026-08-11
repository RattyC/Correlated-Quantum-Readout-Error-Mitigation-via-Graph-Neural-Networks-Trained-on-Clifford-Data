# train_baseline_mlp.py
"""Trivial baseline: per-qubit MLP with zero access to other qubits (no edge_index used).

Purpose: isolate whether the GAT's attention/graph structure is adding anything on the
current (uncorrelated-noise) dataset, or whether the ~0.0015 MSE floor seen across every
GAT ablation (clamp/tanh, 7q/28q, shot counts, learning rate, hidden dim) is simply the best
achievable by looking at each qubit's own noisy <Z> value in isolation. If this baseline lands
at the same floor, the floor is a data-information-content ceiling, not a GAT/training defect.
"""
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from prepare_gnn_dataset import load_and_convert_dataset


class PerQubitMLPBaseline(nn.Module):
    def __init__(self, in_channels, hidden_dim: int = 16, output_activation: str = "clamp"):
        super().__init__()
        if output_activation not in ("clamp", "tanh"):
            raise ValueError(f"Unknown output_activation: {output_activation!r}")
        self.output_activation = output_activation

        # ตั้งใจให้ depth เท่า GAT (2 hidden layers + linear head) เพื่อเทียบแบบยุติธรรม
        # แต่ไม่มี message passing ข้าม node เลย — แต่ละ qubit เห็นแค่ feature ของตัวเอง
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index=None):  # edge_index accepted but unused, for call-site parity
        raw_z = x[:, 0].view(-1, 1)

        out = F.elu(self.fc1(x))
        out = F.elu(self.fc2(out))
        out = self.fc_out(out)

        mitigated_x = raw_z + out

        if self.output_activation == "clamp":
            return torch.clamp(mitigated_x, min=-1.0, max=1.0)
        return torch.tanh(mitigated_x)


def _evaluate(model, loader, criterion, num_samples):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch.x)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
    return total_loss / num_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Train the trivial per-qubit MLP baseline (no graph).")
    parser.add_argument("--input", type=str, default="output_data/quantum_large_dataset.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--activation", type=str, choices=["clamp", "tanh"], default="clamp")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(42)
    random.seed(42)

    full_dataset = load_and_convert_dataset(args.input)
    random.shuffle(full_dataset)

    train_size = int(0.8 * len(full_dataset))
    train_dataset = full_dataset[:train_size]
    test_dataset = full_dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    in_channels = train_dataset[0].x.shape[1]
    model = PerQubitMLPBaseline(in_channels=in_channels, hidden_dim=args.hidden_dim, output_activation=args.activation)
    num_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"\n Baseline Architecture (PerQubitMLP, hidden_dim={args.hidden_dim}, params={num_params}, activation={args.activation}, lr={args.lr}) ")
    print(model)
    print("\n Training Loop (no edge_index used) ")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch.num_graphs
        average_train_loss = total_train_loss / len(train_dataset)

        average_test_loss = _evaluate(model, test_loader, criterion, len(test_dataset))

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs} | Train MSE: {average_train_loss:.6f} | Test MSE: {average_test_loss:.6f}")

    model.eval()
    with torch.no_grad():
        sample = test_dataset[0]

        prediction = model(sample.x).numpy().flatten()
        ideal_target = sample.y.numpy().flatten()
        noisy_input = sample.x[:, 0].numpy().flatten()

        num_q = len(ideal_target)

        print(f"\n--- Sample Prediction Contrast <Z> (Qubit 0 to {num_q - 1}) ---")
        for i in range(num_q):
            print(f"Qubit {i:02d} -> Noisy Input: {noisy_input[i]:+.4f} | Mitigated: {prediction[i]:+.4f} | Ideal Target: {ideal_target[i]:+.4f}")


if __name__ == "__main__":
    main()