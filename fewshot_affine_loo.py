"""
fewshot_affine_loo.py

Constrained few-shot adaptation: freeze the pretrained JointCorrectionMLP entirely,
learn only a 2-parameter affine correction in logit-space on top of its frozen output
(scale + bias, shared across the 4 classes -- matches the identity-preserving affine
correction idea used in GEM, arXiv:2604.16815). Much less prone to overfitting on 14
examples than fine-tuning all 420 base parameters (see fewshot_finetune_loo.py result).

corrected = softmax( log(p_frozen).clamp_min(log(1e-6)) * exp(scale) + bias )

Same leave-one-out protocol over the 15 real examples in sim_to_real_test_raw.json.
"""
import copy
import json
import numpy as np
import torch
import torch.nn as nn

from train_joint_mlp import JointCorrectionMLP

FINETUNE_EPOCHS = 200
FINETUNE_LR = 5e-2


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

base_model = JointCorrectionMLP(hidden_dim=16)
base_state = torch.load("output_data/joint_mlp_trained_q4_realistic.pt", map_location="cpu")
base_model.load_state_dict(base_state)
base_model.eval()
for p in base_model.parameters():
    p.requires_grad = False


def frozen_logits(x):
    with torch.no_grad():
        p = base_model(x)
    return torch.log(p.clamp_min(1e-6))


kls_before, tvs_before = [], []
kls_after, tvs_after = [], []

for held_out in range(n):
    train_idx = [i for i in range(n) if i != held_out]

    x_test = torch.tensor(noisy_vecs[held_out], dtype=torch.float32).unsqueeze(0)
    p_ideal = ideal_vecs[held_out]

    with torch.no_grad():
        p_before = base_model(x_test).squeeze(0).numpy()
    kls_before.append(kl_divergence(p_ideal, p_before))
    tvs_before.append(total_variation(p_ideal, p_before))

    x_train = torch.tensor(np.array([noisy_vecs[i] for i in train_idx]), dtype=torch.float32)
    y_train = torch.tensor(np.array([ideal_vecs[i] for i in train_idx]), dtype=torch.float32)

    scale = torch.zeros(1, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([scale, bias], lr=FINETUNE_LR)

    logits_train = frozen_logits(x_train)
    for epoch in range(FINETUNE_EPOCHS):
        optimizer.zero_grad()
        adjusted = logits_train * torch.exp(scale) + bias
        pred = torch.softmax(adjusted, dim=1)
        loss = torch.sum(y_train * torch.log(y_train.clamp_min(1e-6) / pred.clamp_min(1e-6)), dim=1).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits_test = frozen_logits(x_test)
        adjusted_test = logits_test * torch.exp(scale) + bias
        p_after = torch.softmax(adjusted_test, dim=1).squeeze(0).numpy()
    kls_after.append(kl_divergence(p_ideal, p_after))
    tvs_after.append(total_variation(p_ideal, p_after))

    print(f"fold {held_out}: scale={scale.item():.3f} bias={bias.item():.3f}  "
          f"KL before={kls_before[-1]:.4f} after={kls_after[-1]:.4f}  "
          f"TV before={tvs_before[-1]:.4f} after={tvs_after[-1]:.4f}")

print("\n=== Leave-one-out AFFINE-ONLY adaptation summary (n=%d) ===" % n)
print(f"BEFORE: mean KL={np.mean(kls_before):.5f}  mean TV={np.mean(tvs_before):.5f}")
print(f"AFTER:  mean KL={np.mean(kls_after):.5f}  mean TV={np.mean(tvs_after):.5f}")
print(f"\nReference: M3 KL=0.00430 TV=0.02362 | IBU KL=0.00823 TV=0.02476")