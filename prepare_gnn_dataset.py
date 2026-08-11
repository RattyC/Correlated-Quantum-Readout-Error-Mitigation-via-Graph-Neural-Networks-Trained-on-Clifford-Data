# prepare_gnn_dataset.py
import json
import math
from functools import lru_cache

import torch
from torch_geometric.data import Data

D_MODEL = 8  # fixed positional-encoding dim — import this into train_gnn.py too


def bitstrings_to_pauli_z(counts_dict, num_qubits):
    """แปลงข้อมูล Bitstring Counts ให้กลายเป็น Pauli-Z Expectation [-1.0, 1.0]"""
    total_shots = sum(counts_dict.values())

    # จองพื้นที่สำหรับเก็บค่าคาดหวัง <Z> ของแต่ละคิวบิต
    z_exp = torch.zeros(num_qubits, dtype=torch.float32)

    for bitstring, count in counts_dict.items():
        prob = count / total_shots
        # Qiskit นับ Qubit ตัวที่ 0 จากขวาสุด (Rightmost)
        for i, bit in enumerate(reversed(bitstring)):
            if bit == '0':
                z_exp[i] += prob  # สถานะ |0> ให้ค่า Eigenvalue เป็น +1
            else:
                z_exp[i] -= prob  # สถานะ |1> ให้ค่า Eigenvalue เป็น -1

    return z_exp


@lru_cache(maxsize=None)
def build_fully_connected_edge_index(num_qubits):
    """Fully-connected (all-to-all) edge index, vectorized + cached per qubit count.
    เดิมเป็น O(n^2) pure-Python nested loop ต่อ circuit — ตอนนี้ทำครั้งเดียวต่อ num_qubits
    ที่พบ แล้ว reuse tensor เดิม (ใช้ .clone() ตอนคืนค่าเพื่อกัน in-place mutation)."""
    idx = torch.arange(num_qubits)
    sources, targets = torch.meshgrid(idx, idx, indexing="ij")
    return torch.stack([sources.reshape(-1), targets.reshape(-1)], dim=0).long()


def sinusoidal_positional_encoding(num_qubits, d_model=D_MODEL):
    """Fixed-dimension positional encoding แทน one-hot qubit ID — ทำให้ in_channels
    ของโมเดลไม่ผูกกับ num_qubits อีกต่อไป (จำเป็นสำหรับ zero-shot scaling study)."""
    position = torch.arange(num_qubits).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(num_qubits, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def load_and_convert_dataset(json_path):
    print(f"Loading dataset from {json_path}...")
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    pyg_dataset = []

    for item in raw_data:
        num_qubits = item["num_qubits"]

        # 1. Edge Index แบบ Fully-Connected (All-to-All) — vectorized + cached
        edge_index = build_fully_connected_edge_index(num_qubits).clone()

        # 2. ดึงค่า Noisy <Z>
        noisy_z = bitstrings_to_pauli_z(item["noisy_outputs"], num_qubits)
        noisy_z_view = noisy_z.view(num_qubits, 1)

        # 3. Positional Encoding ขนาดคงที่ (D_MODEL) แทน one-hot ที่ขนาดผูกกับ num_qubits
        pe = sinusoidal_positional_encoding(num_qubits)

        # 4. ประกอบร่าง Input (X) = Noisy Z (มิติ 1) ต่อกับ Positional Encoding (มิติ D_MODEL)
        x = torch.cat([noisy_z_view, pe], dim=1)

        # 5. ดึงค่า Ideal <Z> เป็นเป้าหมาย (Y)
        ideal_z = bitstrings_to_pauli_z(item["ideal_outputs"], num_qubits)
        y = ideal_z.view(num_qubits, 1)

        # 6. สร้าง PyG Data Object แล้วเก็บเข้า Dataset
        graph_data = Data(x=x, edge_index=edge_index, y=y)
        pyg_dataset.append(graph_data)

    print(f"Successfully converted {len(pyg_dataset)} samples into Pauli-Z PyG Data objects.")
    return pyg_dataset


if __name__ == "__main__":
    json_file = "output_data/quantum_large_dataset.json"
    dataset = load_and_convert_dataset(json_file)

    sample = dataset[0]
    print("\n--- Example of Pauli-Z PyG Graph Data Object ---")
    print(f"Noisy Pauli-Z (Input X):\n{sample.x.numpy().flatten()}")
    print(f"Ideal Pauli-Z (Target Y):\n{sample.y.numpy().flatten()}")