# train_pair_feature_mlp.py
"""Pair-feature MLP: same trivial per-qubit MLP as train_baseline_mlp.py, but each paired
qubit also receives its partner's noisy Z value as a plain concatenated input feature (not
via graph message passing / attention). Isolates whether partner information is useful in
principle, separately from whether GATConv is a good mechanism for extracting it.
"""
import argparse
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from prepare_gnn_dataset import bitstrings_to_pauli_z, sinusoidal_positional_encoding


def load_and_convert_with_partner_feature(json_path):
    print(f"Loading dataset from {json_path}... (pair-feature mode)")
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    pyg_dataset = []
    for item in raw_data:
        num_qubits = item["num_qubits"]
        correlated_pairs = item.get("correlated_pairs", [])

        partner = {q: None for q in range(num_qubits)}
        for a, b in correlated_pairs:
            partner[a] = b
            partner[b] = a

        noisy_z = bitstrings_to_pauli_z(item["noisy_outputs"], num_qubits)
        ideal_z = bitstrings_to_pauli_z(item["ideal_outputs"], num_qubits)
        pe = sinusoidal_positional_encoding(num_qubits)

        partner_feature = torch.zeros(num_qubits, 1)
        for q in range(num_qubits):
            p = partner[q]
            if p is not None:
                partner_feature[q, 0] = noisy_z[p]

        x = torch.cat([noisy_z.view(num_qubits, 1), partner_feature, pe], dim=1)
        y = ideal_z.view(num_qubits, 1)

        edge_index = torch.zeros((2, 0), dtype=torch.long)  # unused, PyG Data just wants a placeholder
        pyg_dataset.append(Data(x=x, edge_index=edge_index, y=y))

    print(f"Successfully converted {len(pyg_dataset)} samples.")
    return pyg_dataset


class PairFeatureMLP(nn.Module):
    def __init__(self, in_channels, hidden_dim: int = 16, output_activation: str = "clamp"):
        super().__init__()
        self.output_activation = output_activation
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        raw_z = x[:, 0].view(-1, 1)  # column 0 = own noisy Z; column 1 = partner noisy Z (0 if unpaired)
        out = F.elu(self.fc1(x))
        out = F.elu(self.fc2(out))
        out = self.fc_out(out)
        mitigated = raw_z + out
        if self.output_activation == "clamp":
            return torch.clamp(mitigated, min=-1.0, max=1.0)
        return torch.tanh(mitigated)


def _evaluate(model, loader, criterion, num_samples):
    model.eval()
    total = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch.x)
            loss = criterion(out, batch.y)
            total += loss.item() * batch.num_graphs
    return total / num_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Pair-feature MLP: partner qubit's noisy Z as a direct input feature.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--activation", type=str, choices=["clamp", "tanh"], default="clamp")
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    random.seed(42)

    full_dataset = load_and_convert_with_partner_feature(args.input)
    random.shuffle(full_dataset)

    train_size = int(0.8 * len(full_dataset))
    train_dataset = full_dataset[:train_size]
    test_dataset = full_dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    in_channels = train_dataset[0].x.shape[1]
    model = PairFeatureMLP(in_channels=in_channels, hidden_dim=args.hidden_dim, output_activation=args.activation)
    num_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"\n Pair-Feature MLP Architecture (hidden_dim={args.hidden_dim}, params={num_params}, activation={args.activation}, lr={args.lr}) ")
    print(model)

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
        avg_train = total_train_loss / len(train_dataset)
        avg_test = _evaluate(model, test_loader, criterion, len(test_dataset))
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs} | Train MSE: {avg_train:.6f} | Test MSE: {avg_test:.6f}")

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