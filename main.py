# main.py
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from src.generator import generate_random_clifford_circuit
from src.simulator import execute_circuit_pipeline, get_correlated_pairs
from src.data_exporter import save_dataset_to_json


def generate_single_data(circuit_idx, num_qubits, depth, shots):
    qc = generate_random_clifford_circuit(num_qubits=num_qubits, depth=depth, seed=circuit_idx)
    ideal_counts = execute_circuit_pipeline(qc, shots=shots, use_noise=False)
    noisy_counts = execute_circuit_pipeline(qc, shots=shots, use_noise=True)
    return {
        "circuit_index": circuit_idx,
        "num_qubits": num_qubits,
        "depth": depth,
        "ideal_outputs": ideal_counts,
        "noisy_outputs": noisy_counts,
        "correlated_pairs": get_correlated_pairs(num_qubits),  # ground truth, same for all circuits at this qubit count
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Clifford readout-error dataset.")
    parser.add_argument("--num-qubits", type=int, default=28)
    parser.add_argument("--num-circuits", type=int, default=10000)
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def run_parallel_pipeline(args):
    print("--- Starting Parallel Qiskit Heavy Pipeline ---")
    print(f"num_qubits={args.num_qubits} | num_circuits={args.num_circuits} | depth={args.depth} | shots={args.shots}")
    print(f"Correlated pairs (ground truth): {get_correlated_pairs(args.num_qubits)}")

    max_workers = os.cpu_count()
    print(f"Spawning jobs across {max_workers} CPU workers...")

    dataset = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_single_data, i, args.num_qubits, args.depth, args.shots)
            for i in range(args.num_circuits)
        ]

        for idx, future in enumerate(futures):
            dataset.append(future.result())
            if (idx + 1) % 500 == 0:
                print(f"Progress: {idx + 1}/{args.num_circuits} datasets generated.")

    output_dir = "output_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_name = args.output or f"quantum_dataset_q{args.num_qubits}.json"
    save_dataset_to_json(dataset, os.path.join(output_dir, output_name))
    print("--- Parallel Pipeline Completed ---")


if __name__ == "__main__":
    run_parallel_pipeline(parse_args())