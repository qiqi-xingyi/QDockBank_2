# --*-- conding:utf-8 --*--
# @time:12/3/25 18:47
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:test_instance.py

from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

BACKEND_NAME = "ibm_cleveland"


def build_bell_circuit() -> QuantumCircuit:
    """Build a 2-qubit Bell state circuit with default 'meas' classical register."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()  # this creates a classical register named 'meas'
    return qc


def main():
    print("Loading saved IBM Runtime account...")
    service = QiskitRuntimeService()
    print("Account loaded successfully.")

    # Select backend
    backend = service.backend(BACKEND_NAME)
    print(f"Selected backend: {backend.name}")

    # Build circuit
    bell = build_bell_circuit()
    print("Circuit classical registers:", bell.cregs)
    print("Original circuit:")
    print(bell.draw())

    # Transpile to ISA circuit for this backend
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(bell)
    print("\nTranspiled circuit:")
    print(isa_circuit.draw())

    # Create SamplerV2 bound to this backend
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 1024

    print("\nRunning sampler job on backend...")
    job = sampler.run([isa_circuit])
    print(f"Job ID: {job.job_id()}")

    result = job.result()
    pub_result = result[0]

    # Qiskit 2.0 / SamplerV2: data is organized by classical register name.
    # We used measure_all(), so the default register name is 'meas'.
    print("\nPubResult data keys:", list(pub_result.data.keys()))
    counts = pub_result.data.meas.get_counts()

    print("\nMeasurement counts (meas):")
    print(counts)


if __name__ == "__main__":
    main()


