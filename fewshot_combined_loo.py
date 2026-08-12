"""
fewshot_combined_loo.py

Merges Run 16 (15 examples) + round 2 (20 examples) = 35 real hardware examples on
pair (0,1), then re-runs BOTH few-shot adaptation strategies (full fine-tune, and
constrained 2-param affine-only) via leave-one-out cross-validation, to check whether
more real data (vs. algorithm choice) closes the sim-to-real gap.
"""
import copy
import json
import numpy as np
import torch

from train_joint_mlp import JointCorrectionMLP

FULL_FT_EPOCHS = 30
FULL_FT_LR = 1e-3
AFFINE_EPOCHS = 200
AFFINE_LR = 5e-2


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


# --- load and merge both rounds ---
with open("sim_to_real_test_raw.json") as f:
    round1 = json.load(f)
with open("sim_to_real_test_raw_round2.json") as f:
    round2 = json.load(f)

r1_items = [it for it in round1["labels_and_counts"] if it["label"][0] == "test"]
noisy_vecs = [counts_to_joint4(it["counts"]) for it in r1_items] + \
             [counts_to_joint4(it["counts"]) for it in round2["labels_and_counts"]]
ideal_vecs = [np.array(v) for v in round1["ideal_probs"]] + \
             [np.array(v) for v in round2["ideal_probs"]]
n = len(noisy_vecs)
print(f"Combined dataset: n={n} real examples (15 round1 + 20 round2)")

base_model_state = torch.load("output_data/joint_mlp_trained_q4_realistic.pt", map_location="cpu")


def make_base_model():
    m = JointCorrectionMLP(hidden_dim=16)
    m.load_state_dict(base_model_state)
    return m


def frozen_logits(model, x):
    with torch.no_grad():
        p = model(x)
    return torch.log(p.clamp_min(1e-6))


results = {
    "before": {"kl": [], "tv": []},
    "full_ft": {"kl": [], "tv": []},
    "affine": {"kl": [], "tv": []},
}

for held_out in range(n):
    train_idx = [i for i in range(n) if i != held_out]
    x_test = torch.tensor(noisy_vecs[held_out], dtype=torch.float32).unsqueeze(0)
    p_ideal = ideal_vecs[held_out]
    x_train = torch.tensor(np.array([noisy_vecs[i] for i in train_idx]), dtype=torch.float32)
    y_train = torch.tensor(np.array([ideal_vecs[i] for i in train_idx]), dtype=torch.float32)

    base_model = make_base_model()
    base_model.eval()
    with torch.no_grad():
        p_before = base_model(x_test).squeeze(0).numpy()
    results["before"]["kl"].append(kl_divergence(p_ideal, p_before))
    results["before"]["tv"].append(total_variation(p_ideal, p_before))

    # --- full fine-tune ---
    ft_model = copy.deepcopy(base_model)
    ft_model.train()
    optimizer = torch.optim.Adam(ft_model.parameters(), lr=FULL_FT_LR)
    for _ in range(FULL_FT_EPOCHS):
        optimizer.zero_grad()
        pred = ft_model(x_train)
        loss = torch.sum(y_train * torch.log(y_train.clamp_min(1e-6) / pred.clamp_min(1e-6)), dim=1).mean()
        loss.backward()
        optimizer.step()
    ft_model.eval()
    with torch.no_grad():
        p_ft = ft_model(x_test).squeeze(0).numpy()
    results["full_ft"]["kl"].append(kl_divergence(p_ideal, p_ft))
    results["full_ft"]["tv"].append(total_variation(p_ideal, p_ft))

    # --- affine-only ---
    for p in base_model.parameters():
        p.requires_grad = False
    scale = torch.zeros(1, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    opt_affine = torch.optim.Adam([scale, bias], lr=AFFINE_LR)
    logits_train = frozen_logits(base_model, x_train)
    for _ in range(AFFINE_EPOCHS):
        opt_affine.zero_grad()
        adjusted = logits_train * torch.exp(scale) + bias
        pred = torch.softmax(adjusted, dim=1)
        loss = torch.sum(y_train * torch.log(y_train.clamp_min(1e-6) / pred.clamp_min(1e-6)), dim=1).mean()
        loss.backward()
        opt_affine.step()
    with torch.no_grad():
        logits_test = frozen_logits(base_model, x_test)
        p_affine = torch.softmax(logits_test * torch.exp(scale) + bias, dim=1).squeeze(0).numpy()
    results["affine"]["kl"].append(kl_divergence(p_ideal, p_affine))
    results["affine"]["tv"].append(total_variation(p_ideal, p_affine))

    print(f"fold {held_out}: KL before={results['before']['kl'][-1]:.4f} "
          f"full_ft={results['full_ft']['kl'][-1]:.4f} affine={results['affine']['kl'][-1]:.4f}")

print(f"\n=== Combined (n={n}) leave-one-out summary ===")
for method in ["before", "full_ft", "affine"]:
    kl_mean = np.mean(results[method]["kl"])
    tv_mean = np.mean(results[method]["tv"])
    print(f"{method:10s}: mean KL={kl_mean:.5f}  mean TV={tv_mean:.5f}")
print("\nReference: M3 KL=0.00430 TV=0.02362 | IBU KL=0.00823 TV=0.02476")