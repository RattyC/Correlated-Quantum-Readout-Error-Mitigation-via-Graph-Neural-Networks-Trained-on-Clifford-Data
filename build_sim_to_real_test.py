"""
build_sim_to_real_test.py

Submits a small set of 2-qubit random Clifford test circuits on real qubits (0,1)
of ibm_marrakesh, plus 4 single-qubit M3-style calibration circuits.
Combined into ONE Batch job to minimize QPU minute overhead.

Reuses the pair assignment matrix A for (0,1) already collected in Run 14
(real_hw_calibration.json) -- no need to re-run pair calibration.

Budget: 4 (M3 single-qubit cal) + 15 (test circuits) = 19 circuits, 2048 shots each.
Check dashboard remaining minutes BEFORE running this.
"""
import json
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Batch
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

NUM_TEST_CIRCUITS = 15
SHOTS = 2048
SEED = 20260812

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, min_num_qubits=5, simulator=False)
print("Backend:", backend.name)

# --- 1. M3-style single-qubit calibration circuits (prep |0>,|1> on q0 and q1) ---
m3_circuits, m3_labels = [], []
for qubit in [0, 1]:
    for prep in [0, 1]:
        qc = QuantumCircuit(backend.num_qubits, 1)
        if prep == 1:
            qc.x(qubit)
        qc.measure(qubit, 0)
        m3_circuits.append(qc)
        m3_labels.append(("m3_cal", qubit, prep))

# --- 2. Random 2-qubit Clifford test circuits on q0,q1 ---
test_circuits, test_labels, ideal_probs = [], [], []
for i in range(NUM_TEST_CIRCUITS):
    cliff = random_clifford(2, seed=SEED + i)
    sub_qc = cliff.to_circuit()

    # ideal distribution: noiseless local simulation of the same circuit
    ideal_qc = sub_qc.copy()
    ideal_qc.save_statevector()
    ideal_result = AerSimulator(method="statevector").run(ideal_qc).result()
    sv = ideal_result.get_statevector()
    probs = sv.probabilities_dict()  # keys like '00','01','10','11' for the 2-qubit circuit
    # normalize into our idx = bit_a | (bit_b<<1) convention, bit_a=q0,bit_b=q1
    vec = [0.0, 0.0, 0.0, 0.0]
    for bitstr, p in probs.items():
        b_q1 = int(bitstr[0])  # qiskit statevector key is little-endian already for 2 qubits: bitstr[-1]=q0
        b_q0 = int(bitstr[-1])
        idx = b_q0 | (b_q1 << 1)
        vec[idx] += p
    ideal_probs.append(vec)

    # hardware circuit: same unitary, mapped onto physical qubits 0,1
    hw_qc = QuantumCircuit(backend.num_qubits, 2)
    hw_qc.compose(sub_qc, qubits=[0, 1], inplace=True)
    hw_qc.measure(0, 0)
    hw_qc.measure(1, 1)
    test_circuits.append(hw_qc)
    test_labels.append(("test", i))

all_circuits = m3_circuits + test_circuits
all_labels = m3_labels + test_labels
print(f"Total circuits to submit: {len(all_circuits)} ({len(m3_circuits)} M3-cal + {len(test_circuits)} test)")

# --- dry run first ---
pm_local = generate_preset_pass_manager(optimization_level=1, backend=backend)
transpiled = pm_local.run(all_circuits)
aer_sampler = AerSampler()
dry = aer_sampler.run(transpiled, shots=100).result()
print("Dry run OK, first result:", dry[0].data.c.get_counts() if hasattr(dry[0].data, 'c') else "check field name")

# --- real submission ---
with Batch(backend=backend) as batch:
    sampler = SamplerV2(mode=batch)
    job = sampler.run(transpiled, shots=SHOTS)
    print("Job ID:", job.job_id())
    result = job.result()

output = []
for label, pub_result in zip(all_labels, result):
    counts = pub_result.data.c.get_counts()
    output.append({"label": label, "counts": counts})
    print(label, counts)

with open("sim_to_real_test_raw.json", "w") as f:
    json.dump({"labels_and_counts": output, "ideal_probs": ideal_probs}, f, indent=2)

print("\nSaved to sim_to_real_test_raw.json")