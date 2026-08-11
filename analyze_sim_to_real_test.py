"""
analyze_sim_to_real_test.py

Applies 4 correction methods (raw/pinv/IBU/M3/pretrained-MLP) to real hardware
noisy joint distributions collected in build_sim_to_real_test.py, compared
against the classically-simulated ideal distribution for each test circuit.

Reuses the pair assignment matrix A for (0,1) from Run 14 (real_hw_calibration.json)
-- no new calibration circuits needed.
"""
import json
import numpy as np
import torch

from train_joint_mlp import JointCorrectionMLP


def ibu_correct(p_noisy, A, iterations=100, tol=1e-9):
    p_est = np.clip(p_noisy.copy(), 1e-12, None)
    p_est /= p_est.sum()
    for _ in range(iterations):
        denom = np.clip(A @ p_est, 1e-12, None)
        ratio = p_noisy / denom
        p_new = p_est * (A.T @ ratio)
        p_new = np.clip(p_new, 0.0, None)
        s = p_new.sum()
        if s > 0:
            p_new /= s
        if np.max(np.abs(p_new - p_est)) < tol:
            p_est = p_new
            break
        p_est = p_new
    return p_est


def kl_divergence(p_ideal, p_pred, eps=1e-10):
    p = np.clip(p_ideal, eps, None)
    q = np.clip(p_pred, eps, None)
    return float(np.sum(p * np.log(p / q)))


def total_variation(p_ideal, p_pred):
    return float(0.5 * np.sum(np.abs(p_ideal - p_pred)))


def counts_to_joint4(counts):
    """key format 'q1q0' (leftmost=q1, rightmost=q0), idx = bit_q0 | (bit_q1<<1)"""
    total = sum(counts.values())
    vec = np.zeros(4)
    for bitstr, c in counts.items():
        b_q1 = int(bitstr[0])
        b_q0 = int(bitstr[-1])
        idx = b_q0 | (b_q1 << 1)
        vec[idx] = c / total
    return vec


with open("sim_to_real_test_raw.json") as f:
    raw = json.load(f)
with open("real_hw_calibration.json") as f:
    calib = json.load(f)

# --- reuse pair assignment matrix A(0,1) from Run 14 ---
A = np.zeros((4, 4))
for entry in calib:
    if entry["q_a"] == 0 and entry["q_b"] == 1:
        prep = entry["prep_state"]
        prep_idx = int(prep[0]) | (int(prep[1]) << 1)
        A[:, prep_idx] = counts_to_joint4(entry["counts"])
print("Reused A(0,1) from Run 14:\n", np.round(A, 4))
M_pinv = np.linalg.pinv(A)

# --- single-qubit M3 matrices from this run's m3_cal circuits ---
m3_mats = {0: np.zeros((2, 2)), 1: np.zeros((2, 2))}
for item in raw["labels_and_counts"]:
    label = item["label"]
    if label[0] == "m3_cal":
        _, qubit, prep = label
        counts = item["counts"]
        total = sum(counts.values())
        for bit, c in counts.items():
            m3_mats[qubit][int(bit), prep] = c / total
print("M3 q0:\n", np.round(m3_mats[0], 4))
print("M3 q1:\n", np.round(m3_mats[1], 4))

import mthree
mit = mthree.M3Mitigation(system=None)
mit.cals_from_matrices([m3_mats[0], m3_mats[1]])

# --- pretrained MLP (trained on simulated data only, N=4) ---
model = JointCorrectionMLP(hidden_dim=16)
state = torch.load("output_data/joint_mlp_trained_q4_realistic.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

test_items = [it for it in raw["labels_and_counts"] if it["label"][0] == "test"]
results = {"raw": [], "pinv": [], "ibu": [], "m3": [], "mlp": []}

for i, item in enumerate(test_items):
    counts = item["counts"]
    p_noisy = counts_to_joint4(counts)
    p_ideal = np.array(raw["ideal_probs"][i])

    kl_raw, tv_raw = kl_divergence(p_ideal, p_noisy), total_variation(p_ideal, p_noisy)

    p_pinv = M_pinv @ p_noisy
    p_pinv = np.clip(p_pinv, 0, None)
    if p_pinv.sum() > 0:
        p_pinv /= p_pinv.sum()
    kl_pinv, tv_pinv = kl_divergence(p_ideal, p_pinv), total_variation(p_ideal, p_pinv)

    p_ibu = ibu_correct(p_noisy, A)
    kl_ibu, tv_ibu = kl_divergence(p_ideal, p_ibu), total_variation(p_ideal, p_ibu)

    quasi = mit.apply_correction(counts, qubits=[0, 1])
    p_m3 = np.zeros(4)
    for k, v in quasi.items():
        if isinstance(k, str):
            b_q1, b_q0 = int(k[0]), int(k[-1])
        else:
            b_q0, b_q1 = k & 1, (k >> 1) & 1
        idx = b_q0 | (b_q1 << 1)
        p_m3[idx] += v
    p_m3 = np.clip(p_m3, 0, None)
    if p_m3.sum() > 0:
        p_m3 /= p_m3.sum()
    kl_m3, tv_m3 = kl_divergence(p_ideal, p_m3), total_variation(p_ideal, p_m3)

    with torch.no_grad():
        x = torch.tensor(p_noisy, dtype=torch.float32).unsqueeze(0)
        p_mlp = model(x).squeeze(0).numpy()
    kl_mlp, tv_mlp = kl_divergence(p_ideal, p_mlp), total_variation(p_ideal, p_mlp)

    results["raw"].append((kl_raw, tv_raw))
    results["pinv"].append((kl_pinv, tv_pinv))
    results["ibu"].append((kl_ibu, tv_ibu))
    results["m3"].append((kl_m3, tv_m3))
    results["mlp"].append((kl_mlp, tv_mlp))

    print(f"test {i}: ideal={np.round(p_ideal,3)} noisy={np.round(p_noisy,3)}  "
          f"KL[raw={kl_raw:.4f} pinv={kl_pinv:.4f} ibu={kl_ibu:.4f} m3={kl_m3:.4f} mlp={kl_mlp:.4f}]")

print(f"\n=== Sim-to-Real Transfer Summary (real hardware, pair (0,1), n={len(test_items)}) ===")
for method in ["raw", "pinv", "ibu", "m3", "mlp"]:
    kls = [x[0] for x in results[method]]
    tvs = [x[1] for x in results[method]]
    print(f"{method:6s}: mean KL={np.mean(kls):.5f}  mean TV={np.mean(tvs):.5f}")