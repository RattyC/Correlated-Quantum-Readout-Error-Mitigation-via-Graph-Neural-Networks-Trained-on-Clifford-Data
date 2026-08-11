# prepare_gnn_dataset.py
import argparse
import json
import math
from functools import lru_cache

import torch
from torch_geometric.data import Data

D_MODEL = 8  # fixed positional-encoding dim — import this into train_gnn.py too


def bitstrings_to_pauli_z(counts_dict, num_qubits):
    """แปลงข้อมูล Bitstring Counts ให้กลายเป็น Pauli-Z Expectation [-1.0, 1.0]"""
    total_shots = sum(counts_dict.values())

    z_exp = torch.zeros(num_qubits, dtype=torch.float32)

    for bitstring, count in counts_dict.items():
        prob = count / total_shots
        for i, bit in enumerate(reversed(bitstring)):
            if bit == '0':
                z_exp[i] += prob
            else:
                z_exp[i] -= prob

    return z_exp


@lru_cache(maxsize=None)
def build_fully_connected_edge_index(num_qubits):
    """Fully-connected (all-to-all) edge index, vectorized + cached per qubit count."""
    idx = torch.arange(num_qubits)
    sources, targets = torch.meshgrid(idx, idx, indexing="ij")
    return torch.stack([sources.reshape(-1), targets.reshape(-1)], dim=0).long()


def build_sparse_edge_index(correlated_pairs, num_qubits):
    """Edge index built ONLY from correlated qubit pairs (ground-truth OR calibration-estimated),
    plus self-loops (each qubit always attends to itself so it keeps its own signal)."""
    sources = list(range(num_qubits))  # self-loops
    targets = list(range(num_qubits))
    for a, b in correlated_pairs:
        sources += [a, b]
        targets += [b, a]  # undirected: both directions for message passing
    return torch.tensor([sources, targets], dtype=torch.long)


def sinusoidal_positional_encoding(num_qubits, d_model=D_MODEL):
    """Fixed-dimension positional encoding แทน one-hot qubit ID."""
    position = torch.arange(num_qubits).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(num_qubits, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def load_and_convert_dataset(json_path, edge_mode: str = "full", external_pairs=None):
    if edge_mode not in ("full", "sparse"):
        raise ValueError(f"Unknown edge_mode: {edge_mode!r}")

    pairs_source = "external file" if external_pairs is not None else "dataset ground-truth metadata"
    print(f"Loading dataset from {json_path}... (edge_mode={edge_mode}, pairs_source={pairs_source if edge_mode == 'sparse' else 'n/a'})")
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    pyg_dataset = []

    for item in raw_data:
        num_qubits = item["num_qubits"]

        if edge_mode == "full":
            edge_index = build_fully_connected_edge_index(num_qubits).clone()
        else:
            pairs = external_pairs if external_pairs is not None else item.get("correlated_pairs", [])
            if not pairs:
                raise ValueError("edge_mode='sparse' requires correlated pairs (via --pairs-file or dataset metadata).")
            edge_index = build_sparse_edge_index(pairs, num_qubits)

        noisy_z = bitstrings_to_pauli_z(item["noisy_outputs"], num_qubits)
        noisy_z_view = noisy_z.view(num_qubits, 1)

        pe = sinusoidal_positional_encoding(num_qubits)

        x = torch.cat([noisy_z_view, pe], dim=1)

        ideal_z = bitstrings_to_pauli_z(item["ideal_outputs"], num_qubits)
        y = ideal_z.view(num_qubits, 1)

        graph_data = Data(x=x, edge_index=edge_index, y=y)
        pyg_dataset.append(graph_data)

    print(f"Successfully converted {len(pyg_dataset)} samples into Pauli-Z PyG Data objects.")
    return pyg_dataset


def _load_pairs_file(path):
    if path is None:
        return None
    with open(path, "r") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert raw Clifford dataset JSON into PyG Data objects.")
    parser.add_argument("--input", type=str, default="output_data/quantum_large_dataset.json")
    parser.add_argument(
        "--edge-mode", type=str, choices=["full", "sparse"], default="full",
        help="'full' = all-to-all edges. 'sparse' = correlated pairs + self-loops (ground-truth or --pairs-file).",
    )
    parser.add_argument(
        "--pairs-file", type=str, default=None,
        help="JSON file of [[a,b], ...] pairs (e.g. from calibrate_correlations.py) to use instead of the dataset's ground-truth correlated_pairs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    external_pairs = _load_pairs_file(args.pairs_file)
    dataset = load_and_convert_dataset(args.input, edge_mode=args.edge_mode, external_pairs=external_pairs)

    sample = dataset[0]
    print("\n--- Example of Pauli-Z PyG Graph Data Object ---")
    print(f"Noisy Pauli-Z (Input X):\n{sample.x.numpy().flatten()}")
    print(f"Ideal Pauli-Z (Target Y):\n{sample.y.numpy().flatten()}")
    print(f"Edge index shape: {sample.edge_index.shape}, edges:\n{sample.edge_index}")