"""
build_sim_to_real_test_round2.py

Second round of real hardware test circuits on qubits (0,1) of ibm_marrakesh, to expand
the held-out real dataset used for few-shot adaptation (Run 16's 15 examples were shown
insufficient for both full fine-tuning and constrained affine adaptation).

Skips M3 single-qubit calibration circuits (already have them from Run 16) -- pure test
circuits only, to spend the limited QPU budget entirely on data that helps few-shot training.

Appends results into a NEW file (does not overwrite sim_to_real_test_raw.json), then a
follow-up step merges both rounds before re-running the few-shot experiments.
"""
import json
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Batch
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

NUM_TEST_CIRCUITS = 20   # adjust down if budget is tight -- see printed dry-run circuit count
SHOTS = 2048
SEED_OFFSET = 500  # different from round 1 (which used SEED+i for i in 0..14) to avoid duplicate circuits

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, min_num_qubits=5, simulator=False)
print("Backend:", backend.name)

test_circuits, test_labels, ideal_probs = [], [], []
for i in range(NUM_TEST_CIRCUITS):
    cliff = random_clifford(2, seed=SEED_OFFSET + i)
    sub_qc = cliff.to_circuit()

    ideal_qc = sub_qc.copy()
    ideal_qc.save_statevector()
    ideal_result = AerSimulator(method="statevector").run(ideal_qc).result()
    sv = ideal_result.get_statevector()
    probs = sv.probabilities_dict()
    vec = [0.0, 0.0, 0.0, 0.0]
    for bitstr, p in probs.items():
        b_q1 = int(bitstr[0])
        b_q0 = int(bitstr[-1])
        idx = b_q0 | (b_q1 << 1)
        vec[idx] += p
    ideal_probs.append(vec)

    hw_qc = QuantumCircuit(backend.num_qubits, 2)
    hw_qc.compose(sub_qc, qubits=[0, 1], inplace=True)
    hw_qc.measure(0, 0)
    hw_qc.measure(1, 1)
    test_circuits.append(hw_qc)
    test_labels.append(("test_r2", i))

print(f"Total circuits to submit: {len(test_circuits)} (round 2, test only)")

pm_local = generate_preset_pass_manager(optimization_level=1, backend=backend)
transpiled = pm_local.run(test_circuits)
aer_sampler = AerSampler()
dry = aer_sampler.run(transpiled, shots=100).result()
print("Dry run OK")

with Batch(backend=backend) as batch:
    sampler = SamplerV2(mode=batch)
    job = sampler.run(transpiled, shots=SHOTS)
    print("Job ID:", job.job_id())
    result = job.result()

output = []
for label, pub_result in zip(test_labels, result):
    counts = pub_result.data.c.get_counts()
    output.append({"label": label, "counts": counts})
    print(label, counts)

with open("sim_to_real_test_raw_round2.json", "w") as f:
    json.dump({"labels_and_counts": output, "ideal_probs": ideal_probs}, f, indent=2)

print("\nSaved to sim_to_real_test_raw_round2.json")