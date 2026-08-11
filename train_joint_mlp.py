# train_joint_mlp.py
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from prepare_joint_dataset import JointPairDataset


class JointCorrectionMLP(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        out = F.elu(self.fc1(x))
        out = F.elu(self.fc2(out))
        logits = self.fc_out(out) + torch.log(x.clamp_min(1e-6))  # residual in logit space
        return F.softmax(logits, dim=-1)


def kl_loss(pred, target, eps=1e-8):
    pred = pred.clamp_min(eps)
    target = target.clamp_min(eps)
    return (target * (target.log() - pred.log())).sum(dim=-1).mean()


def total_variation(pred, target):
    return (0.5 * (pred - target).abs().sum(dim=-1)).mean()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    random.seed(42)

    dataset = JointPairDataset(args.input)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    model = JointCorrectionMLP(hidden_dim=args.hidden_dim)
    num_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\n Joint Correction MLP (hidden_dim={args.hidden_dim}, params={num_params}, lr={args.lr}) ")
    print(model)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = kl_loss(pred, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * x.size(0)
        avg_train = total_train_loss / len(train_ds)

        model.eval()
        total_test_kl, total_test_tv = 0.0, 0.0
        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x)
                total_test_kl += kl_loss(pred, y).item() * x.size(0)
                total_test_tv += total_variation(pred, y).item() * x.size(0)
        avg_test_kl = total_test_kl / len(test_ds)
        avg_test_tv = total_test_tv / len(test_ds)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs} | Train KL: {avg_train:.6f} | Test KL: {avg_test_kl:.6f} | Test TV: {avg_test_tv:.6f}")

    model.eval()
    with torch.no_grad():
        x0, y0 = test_ds[0]
        pred0 = model(x0.unsqueeze(0)).squeeze(0)
        print(f"\n--- Sample joint distribution ---")
        print(f"Noisy:     {x0.tolist()}")
        print(f"Corrected: {pred0.tolist()}")
        print(f"Ideal:     {y0.tolist()}")


if __name__ == "__main__":
    main()