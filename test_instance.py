# --*-- conding:utf-8 --*--
# @time:12/3/25 18:47
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:test_instance.py

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

BACKEND_NAME = "ibm_cleveland"

def build_bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


def main():
    print("Loading saved IBM Runtime account...")

    # Load saved credentials
    service = QiskitRuntimeService()
    print("Account loaded successfully.")

    # List available backends
    backends = service.backends()
    print(f"Found {len(backends)} backends.")
    for backend in backends:
        status = backend.status()
        print(
            f"- {backend.name}: "
            f"operational={getattr(status, 'operational', None)}, "
            f"pending_jobs={getattr(status, 'pending_jobs', None)}, "
            f"status_msg={getattr(status, 'status_msg', None)}"
        )

    # Select backend
    backend = service.backend(BACKEND_NAME)
    print(f"\nSelected backend: {backend.name}")

    # Build Bell circuit
    bell = build_bell_circuit()
    print("Original circuit:")
    print(bell.draw())

    # Transpile for the backend
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(bell)
    print("\nTranspiled circuit:")
    print(isa_circuit.draw())

    # Create sampler bound to this backend
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 1024

    print("\nRunning sampler job on backend...")
    job = sampler.run([isa_circuit])
    print(f"Job ID: {job.job_id()}")

    pub_result = job.result()[0]
    counts = pub_result.data.get_counts()

    print("\nMeasurement results:")
    print(counts)


if __name__ == "__main__":
    main()

