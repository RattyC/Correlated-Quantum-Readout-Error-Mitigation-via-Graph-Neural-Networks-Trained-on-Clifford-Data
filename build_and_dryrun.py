from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.primitives import SamplerV2 as AerSampler

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, min_num_qubits=5, simulator=False)
print("Backend:", backend.name, "| num_qubits:", backend.num_qubits)

# เลือก 3 คู่ qubit ที่ติดกันจริงบนชิป (undirected, ตัดคู่ซ้ำ)
edges = backend.coupling_map.get_edges()
seen, candidate_pairs = set(), []
for a, b in edges:
    key = tuple(sorted((a, b)))
    if key not in seen:
        seen.add(key)
        candidate_pairs.append(key)
candidate_pairs = candidate_pairs[:3]
print("Candidate pairs:", candidate_pairs)

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

all_circuits = []
for q_a, q_b in candidate_pairs:
    all_circuits += build_calib_circuits(q_a, q_b, backend.num_qubits)

print("Total circuits:", len(all_circuits))  # ควรได้ 12 (3 คู่ x 4 prep)

pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
transpiled = pm.run(all_circuits)

# --- DRY RUN บน Aer ก่อน ห้ามข้าม ---
aer_sampler = AerSampler()
dry_result = aer_sampler.run(transpiled, shots=100).result()
for i, pub_result in enumerate(dry_result):
    counts = pub_result.data.c.get_counts()
    print(f"circuit {i}: {counts}")

print("\nDRY RUN PASSED — pipeline พร้อมยิงจริง")
