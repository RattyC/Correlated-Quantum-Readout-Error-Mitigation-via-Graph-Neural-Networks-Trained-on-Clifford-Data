from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, min_num_qubits=5, simulator=False)
print("Auth OK, backend:", backend.name, "| pending jobs:", backend.status().pending_jobs)
