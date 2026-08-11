
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(

    token="wAnVvHYQcdrA_TDFW3J6OCABNn5yKABBcgVdmEFJ66hP",

    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/57a507735ff84537bdd5fd6e2742faa3:05fad123-c434-4859-b161-511aef2b8640::",

    channel="ibm_quantum_platform",

    overwrite=True,

    set_as_default=True,

)

print("saved OK")

