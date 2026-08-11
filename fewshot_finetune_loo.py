"""
fewshot_finetune_loo.py

Few-shot fine-tuning of the (already synthetic-realistic-trained) JointCorrectionMLP on
real hardware data, evaluated via leave-one-out cross-validation over the 15 real test
circuits collected in Run 16 (sim_to_real_test_raw.json) -- no new hardware calls.

For each of the 15 examples: fine-tune a fresh copy of the pretrained model on the other
14 real examples (few epochs, small LR), evaluate KL/TV on the held-out one. Average across
all 15 folds. This tests whether a handful of real calibration examples can close the
remaining sim-to-real gap without a full real dataset.
"""
import copy
import json
import numpy as np
import torch
import torch.nn as nn

from train_joint_mlp import JointCorrectionMLP

FINETUNE_EPOCHS = 30
FINETUNE_LR = 1e-3


def counts_to_joint4(counts):
    total = sum(counts.values())
    vec = np.zeros(4)
    for bitstr, c in counts.items():
        b_q1 = int(bitstr[0])
        b_q0 = int(bitstr[-1])
        idx = b_q0 | (b_q1 << 1)
        vec[idx] = c / total
    return vec


def kl_divergence(p_ideal, p_pred, eps=1e-10):
    p = np.clip(p_ideal, eps, None)
    q = np.clip(p_pred, eps, None)
    return float(np.sum(p * np.log(p / q)))


def total_variation(p_ideal, p_pred):
    return float(0.5 * np.sum(np.abs(p_ideal - p_pred)))


with open("sim_to_real_test_raw.json") as f:
    raw = json.load(f)

test_items = [it for it in raw["labels_and_counts"] if it["label"][0] == "test"]
noisy_vecs = [counts_to_joint4(it["counts"]) for it in test_items]
ideal_vecs = [np.array(v) for v in raw["ideal_probs"]]
n = len(test_items)
print(f"n={n} real examples available for leave-one-out fine-tuning")

base_model = JointCorrectionMLP(hidden_dim=16)
base_state = torch.load("output_data/joint_mlp_trained_q4_realistic.pt", map_location="cpu")
base_model.load_state_dict(base_state)

kls_before, tvs_before = [], []
kls_after, tvs_after = [], []

for held_out in range(n):
    train_idx = [i for i in range(n) if i != held_out]

    model = copy.deepcopy(base_model)

    x_test = torch.tensor(noisy_vecs[held_out], dtype=torch.float32).unsqueeze(0)
    p_ideal = ideal_vecs[held_out]

    model.eval()
    with torch.no_grad():
        p_before = model(x_test).squeeze(0).numpy()
    kls_before.append(kl_divergence(p_ideal, p_before))
    tvs_before.append(total_variation(p_ideal, p_before))

    x_train = torch.tensor(np.array([noisy_vecs[i] for i in train_idx]), dtype=torch.float32)
    y_train = torch.tensor(np.array([ideal_vecs[i] for i in train_idx]), dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)
    model.train()
    for epoch in range(FINETUNE_EPOCHS):
        optimizer.zero_grad()
        pred = model(x_train)
        loss = torch.sum(y_train * torch.log(y_train.clamp_min(1e-6) / pred.clamp_min(1e-6)), dim=1).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        p_after = model(x_test).squeeze(0).numpy()
    kls_after.append(kl_divergence(p_ideal, p_after))
    tvs_after.append(total_variation(p_ideal, p_after))

    print(f"fold {held_out}: KL before={kls_before[-1]:.4f} after={kls_after[-1]:.4f}  "
          f"TV before={tvs_before[-1]:.4f} after={tvs_after[-1]:.4f}")

print("\n=== Leave-one-out few-shot fine-tune summary (n=%d) ===" % n)
print(f"BEFORE fine-tune: mean KL={np.mean(kls_before):.5f}  mean TV={np.mean(tvs_before):.5f}")
print(f"AFTER  fine-tune: mean KL={np.mean(kls_after):.5f}  mean TV={np.mean(tvs_after):.5f}")
print(f"\nReference (Run 16/17, no fine-tune): M3 KL=0.00430 TV=0.02362 | IBU KL=0.00823 TV=0.02476")