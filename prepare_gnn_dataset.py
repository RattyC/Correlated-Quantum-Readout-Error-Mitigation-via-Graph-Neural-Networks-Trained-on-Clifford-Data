# prepare_gnn_dataset.py
import json
import torch
from torch_geometric.data import Data

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

def load_and_convert_dataset(json_path):
    print(f"Loading dataset from {json_path}...")
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    pyg_dataset = []

    for item in raw_data:
        num_qubits = item["num_qubits"]
        # 1. สร้าง Edge Index แบบ Fully-Connected (All-to-All)
        # ให้ทุก Qubit เชื่อมโยงถึงกันหมด เพื่อแก้ปัญหา Over-squashing
        sources = []
        targets = []
        for i in range(num_qubits):
            for j in range(num_qubits):
                sources.append(i)
                targets.append(j)
        edge_index = torch.tensor([sources, targets], dtype=torch.long)

        # 2. ดึงค่า Noisy <Z>
        noisy_z = bitstrings_to_pauli_z(item["noisy_outputs"], num_qubits)
        noisy_z_view = noisy_z.view(num_qubits, 1)

        # 3. สร้าง One-Hot Encoding Matrix ขนาด (num_qubits, num_qubits)
        identity_matrix = torch.eye(num_qubits)

        # 4. ประกอบร่าง Input (X) = Noisy Z (มิติ 1) ต่อกับ One-Hot (มิติ num_qubits)
        x = torch.cat([noisy_z_view, identity_matrix], dim=1)

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
