# train_gnn.py
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

from prepare_gnn_dataset import load_and_convert_dataset


class QuantumReadoutMitigationGAT(nn.Module):
    def __init__(self, in_channels, output_activation: str = "clamp"):
        super(QuantumReadoutMitigationGAT, self).__init__()

        if output_activation not in ("clamp", "tanh"):
            raise ValueError(f"Unknown output_activation: {output_activation!r}")
        self.output_activation = output_activation

        # ขาเข้า: 1 (Noisy Z) + D_MODEL (Sinusoidal Positional Encoding, มิติคงที่
        # ไม่ผูกกับจำนวนคิวบิตจริง — นี่คือจุดที่ทำให้ zero-shot ข้าม qubit count ทำได้)
        # in_channels ถูกกำหนดจาก data โดยตรง ไม่ hardcode ในนี้

        self.gat1 = GATConv(in_channels, 16, heads=4, concat=True)
        self.gat2 = GATConv(64, 16, heads=1, concat=False)
        self.fc = nn.Linear(16, 1)

    def forward(self, x, edge_index):
        raw_z = x[:, 0].view(-1, 1)

        out = self.gat1(x, edge_index)
        out = F.elu(out)

        out = self.gat2(out, edge_index)
        out = F.elu(out)

        out = self.fc(out)

        mitigated_x = raw_z + out

        # ablation switch:
        # - "clamp": ตัดขอบที่ -1.0/1.0 พอดี แต่ gradient = 0 นอกช่วง [-1, 1] (dead zone)
        # - "tanh":  gradient ไม่เป็นศูนย์เด็ดขาด แต่เข้าใกล้ ±1 แบบ asymptotic เท่านั้น
        if self.output_activation == "clamp":
            return torch.clamp(mitigated_x, min=-1.0, max=1.0)
        return torch.tanh(mitigated_x)


def _evaluate(model, loader, criterion, num_samples):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
    return total_loss / num_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Train the QuantumReadoutMitigationGAT model.")
    parser.add_argument(
        "--input",
        type=str,
        default="output_data/quantum_large_dataset.json",
        help="Path to the raw dataset JSON produced by main.py.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument(
        "--activation",
        type=str,
        choices=["clamp", "tanh"],
        default="clamp",
        help="Output activation for the residual correction — used for the clamp-vs-tanh ablation.",
    )
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

    # in_channels มาจากขนาด feature จริงของ node (1 Noisy Z + D_MODEL PE)
    # ไม่ใช่จำนวนคิวบิต — คงที่แม้ dataset จะมี graph หลายขนาด (4/6/8/10 qubits) ปนกัน
    in_channels = train_dataset[0].x.shape[1]
    model = QuantumReadoutMitigationGAT(in_channels=in_channels, output_activation=args.activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"\n Model Architecture (output_activation={args.activation}) ")
    print(model)
    print("\n Residual GAT Training Loop on Pauli-Z Domain ")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch.num_graphs
        average_train_loss = total_train_loss / len(train_dataset)

        average_test_loss = _evaluate(model, test_loader, criterion, len(test_dataset))

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs} | Train MSE: {average_train_loss:.6f} | Test MSE: {average_test_loss:.6f}")

    model.eval()
    with torch.no_grad():
        sample = test_dataset[0]

        prediction = model(sample.x, sample.edge_index).numpy().flatten()
        ideal_target = sample.y.numpy().flatten()
        noisy_input = sample.x[:, 0].numpy().flatten()

        num_q = len(ideal_target)

        print(f"\n--- Sample Prediction Contrast <Z> (Qubit 0 to {num_q - 1}) ---")
        for i in range(num_q):
            print(f"Qubit {i:02d} -> Noisy Input: {noisy_input[i]:+.4f} | Mitigated: {prediction[i]:+.4f} | Ideal Target: {ideal_target[i]:+.4f}")


if __name__ == "__main__":
    main()