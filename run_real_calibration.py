
import json

from qiskit import QuantumCircuit

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Batch

service = QiskitRuntimeService()

backend = service.least_busy(operational=True, min_num_qubits=5, simulator=False)

print("Backend:", backend.name)

edges = backend.coupling_map.get_edges()

seen, candidate_pairs = set(), []

for a, b in edges:

    key = tuple(sorted((a, b)))

    if key not in seen:

        seen.add(key)

        candidate_pairs.append(key)

candidate_pairs = candidate_pairs[:3]

def build_calib_circuits(q_a, q_b, num_qubits):

    circuits = []

    for prep in ["00", "01", "10", "11"]:

        qc = QuantumCircuit(num_qubits, 2)

        if prep[0] == "1":

            qc.x(q_a)

        if prep[1] == "1":

            qc.x(q_b)

        qc.measure(q_a, 0)

        qc.measure(q_b, 1)

        circuits.append(qc)

    return circuits

all_circuits, labels = [], []

for q_a, q_b in candidate_pairs:

    all_circuits += build_calib_circuits(q_a, q_b, backend.num_qubits)

    labels += [(q_a, q_b, p) for p in ["00", "01", "10", "11"]]

pm = generate_preset_pass_manager(optimization_level=1, backend=backend)

transpiled = pm.run(all_circuits)

# --- ยิงจริง ---

with Batch(backend=backend) as batch:

    sampler = SamplerV2(mode=batch)

    job = sampler.run(transpiled, shots=4096)

    print("Job ID:", job.job_id())

    result = job.result()

output = []

for label, pub_result in zip(labels, result):

    q_a, q_b, prep = label

    counts = pub_result.data.c.get_counts()

    output.append({"q_a": q_a, "q_b": q_b, "prep_state": prep, "counts": counts})

    print(label, counts)

with open("real_hw_calibration.json", "w") as f:

    json.dump(output, f, indent=2)

print("\nบันทึกผลไว้ที่ real_hw_calibration.json")

print("เช็คงบเวลาที่เหลือได้ที่ IBM Quantum Platform dashboard > Usage")

