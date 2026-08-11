# main.py
import os
from concurrent.futures import ProcessPoolExecutor
from src.generator import generate_random_clifford_circuit
from src.simulator import execute_circuit_pipeline
from src.data_exporter import save_dataset_to_json

def generate_single_data(circuit_idx, num_qubits, depth, shots):
    # 1. สร้างวงจร Clifford
    qc = generate_random_clifford_circuit(num_qubits=num_qubits, depth=depth, seed=circuit_idx)

    # 2. จำลองแบบ Ideal และ Noisy
    ideal_counts = execute_circuit_pipeline(qc, shots=shots, use_noise=False)
    noisy_counts = execute_circuit_pipeline(qc, shots=shots, use_noise=True)

    return {
        "circuit_index": circuit_idx,
        "num_qubits": num_qubits,
        "depth": depth,
        "ideal_outputs": ideal_counts,
        "noisy_outputs": noisy_counts
    }

def run_parallel_pipeline():
    print("--- Starting Parallel Qiskit Heavy Pipeline ---")

    NUM_CIRCUITS = 10000  # จำนวนวงจร 10,000 วงจรสำหรับ Stress Test ฮาร์ดแวร์
    NUM_QUBITS = 7
    CIRCUIT_DEPTH = 30
    SHOTS = 4096

    max_workers = os.cpu_count()
    print(f"Spawning jobs across {max_workers} CPU workers...")

    dataset = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_single_data, i, NUM_QUBITS, CIRCUIT_DEPTH, SHOTS)
            for i in range(NUM_CIRCUITS)
        ]

        for idx, future in enumerate(futures):
            dataset.append(future.result())
            if (idx + 1) % 500 == 0:
                print(f"Progress: {idx + 1}/{NUM_CIRCUITS} datasets generated.")

    output_dir = "output_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_dataset_to_json(dataset, os.path.join(output_dir, "quantum_large_dataset.json"))
    print("--- Parallel Pipeline Completed ---")

if __name__ == "__main__":
    run_parallel_pipeline()
