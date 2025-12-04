# --*-- conding:utf-8 --*--
# @Time : 11/3/25 1:49 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : vqe_top5.py


import time
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
from qiskit.circuit.library import EfficientSU2
from scipy.optimize import minimize

from qiskit_ibm_runtime import Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives.containers import PrimitiveResult, PubResult, DataBin


class VQE5:
    """
    Variational Quantum Eigensolver (VQE) using IBM Quantum Runtime (Qiskit 2.0 style).
    This class is instrumented to record full timing and raw job results for analysis.
    """

    def __init__(
            self,
            service,
            hamiltonian,
            optimization_level: int = 3,
            shots: int = 200,
            min_qubit_num: int = 100,
            maxiter: int = 20,
    ):
        """
        Parameters
        ----------
        service : QiskitRuntimeService
            Runtime service instance (already configured with your account/instance).
        hamiltonian : SparsePauliOp or similar
            The Hamiltonian to minimize.
        optimization_level : int
            Transpiler optimization level (0-3).
        shots : int
            Default shots for estimator and sampler.
        min_qubit_num : int
            Minimal number of qubits required on the backend.
        maxiter : int
            Maximum number of COBYLA iterations.
        """
        self.service = service
        self.shots = shots
        self.backend = self._select_backend(min_qubits=min_qubit_num)
        self.hamiltonian = hamiltonian
        self.ansatz = EfficientSU2(self.hamiltonian.num_qubits)
        self.optimization_level = optimization_level

        self.cost_history_dict = {"prev_vector": None, "iters": 0, "cost_history": []}
        self.energy_list: List[float] = []
        self.maxiter = maxiter
        self.iteration_results: List[tuple[float, np.ndarray]] = []

        # New: detailed logs per estimator evaluation
        self.iter_logs: List[Dict[str, Any]] = []
        self.iter_raw_results: List[Dict[str, Any]] = []

        # New: detailed logs for sampler calls
        self.sampler_logs: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Backend / transpiler utilities
    # -------------------------------------------------------------------------
    def _select_backend(self, min_qubits: int):
        """
        Select a backend with enough qubits. In your CC environment
        this will typically resolve to ibm_cleveland.
        """
        backend = self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=min_qubits,
        )
        return backend

    def _generate_pass_manager(self):
        """
        Generate a preset pass manager for the selected backend.
        """
        pm = generate_preset_pass_manager(
            backend=self.backend,
            optimization_level=self.optimization_level,
        )
        return pm

    # -------------------------------------------------------------------------
    # Serialization helpers (for saving all raw data)
    # -------------------------------------------------------------------------
    @staticmethod
    def _to_serializable(obj: Any):
        """
        Convert numpy / datetime / DataBin / containers into JSON-serializable objects.
        """
        # numpy scalar
        if isinstance(obj, np.generic):
            return obj.item()
        # numpy array
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # datetime
        if isinstance(obj, datetime):
            return obj.isoformat()
        # DataBin
        if isinstance(obj, DataBin):
            return {k: VQE5._to_serializable(v) for k, v in obj.items()}
        # dict / list / tuple
        if isinstance(obj, dict):
            return {k: VQE5._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [VQE5._to_serializable(v) for v in obj]
        return obj

    @staticmethod
    def _primitive_result_to_dict(result: PrimitiveResult) -> Dict[str, Any]:
        """
        Convert PrimitiveResult into a nested dict containing metadata and per-pub data.
        """
        out: Dict[str, Any] = {
            "global_metadata": VQE5._to_serializable(result.metadata),
            "pub_results": [],
        }
        for idx, pub in enumerate(result):  # type: ignore
            pub: PubResult
            pub_entry: Dict[str, Any] = {
                "index": idx,
                "metadata": VQE5._to_serializable(pub.metadata),
                "data": VQE5._to_serializable(pub.data),
            }
            out["pub_results"].append(pub_entry)
        return out

    # -------------------------------------------------------------------------
    # VQE core: cost function
    # -------------------------------------------------------------------------
    def cost_func(self, params, ansatz_isa, hamiltonian_isa, estimator):
        """
        Cost function for the VQE optimization. One call corresponds to
        one EstimatorV2 evaluation (one Runtime job).
        """
        # Build pub
        pub = (ansatz_isa, [hamiltonian_isa], [params])

        iter_index = self.cost_history_dict["iters"] + 1
        wall_start = time.time()
        ts_start = datetime.now().isoformat()

        # Submit job
        job = estimator.run(pubs=[pub])

        # Collect job information (where supported)
        job_id = None
        creation_date = None
        usage = None
        try:
            job_id = job.job_id()
        except Exception:
            pass
        try:
            creation_date = job.creation_date
        except Exception:
            pass
        try:
            usage = job.usage_estimation()
        except Exception:
            usage = None

        # Wait for result
        result: PrimitiveResult = job.result()
        wall_end = time.time()
        ts_end = datetime.now().isoformat()

        # Extract energy (assume 1 pub, 1 observable)
        energy = float(result[0].data.evs[0])

        # Original bookkeeping
        self.energy_list.append(energy)
        self.iteration_results.append((energy, np.array(params, copy=True)))
        self.cost_history_dict["iters"] = iter_index
        self.cost_history_dict["prev_vector"] = np.array(params, copy=True)
        self.cost_history_dict["cost_history"].append(energy)

        print(f"Iter {iter_index} done. Energy = {energy}")

        # Detailed log for this iteration
        iter_log: Dict[str, Any] = {
            "iter": iter_index,
            "timestamp_start": ts_start,
            "timestamp_end": ts_end,
            "wall_time_sec": wall_end - wall_start,
            "energy": energy,
            "params_norm": float(np.linalg.norm(params)),
            "backend": getattr(self.backend, "name", None),
            "shots": self.shots,
            "job_id": job_id,
            "job_creation_date": creation_date.isoformat() if hasattr(creation_date, "isoformat") else None,
            "usage_estimation": self._to_serializable(usage),
        }
        self.iter_logs.append(iter_log)

        # Store full raw primitive result for this iteration
        raw_dict = self._primitive_result_to_dict(result)
        self.iter_raw_results.append(
            {
                "iter": iter_index,
                "job_id": job_id,
                "primitive_result": raw_dict,
            }
        )

        return energy

    # -------------------------------------------------------------------------
    # Probability distribution (Sampler) with logging
    # -------------------------------------------------------------------------
    def get_probability_distribution(self, optimized_params, tag: str | None = None) -> Dict[str, float]:
        """
        Compute probability distribution of bitstrings for given parameters using SamplerV2
        on the same backend. Also log full job timing and raw sampler result.

        Parameters
        ----------
        optimized_params : array-like
            Parameters to assign to the ansatz.
        tag : str, optional
            Optional label to identify this sampler call (e.g., 'final', 'top1', etc.).
        """
        circuit = self.ansatz.assign_parameters(optimized_params)
        circuit.measure_all()

        wall_start = time.time()
        ts_start = datetime.now().isoformat()

        with Session(backend=self.backend) as session:
            sampler = RuntimeSampler(mode=session)
            sampler.options.default_shots = self.shots

            job = sampler.run([circuit])

            job_id = None
            creation_date = None
            usage = None
            try:
                job_id = job.job_id()
            except Exception:
                pass
            try:
                creation_date = job.creation_date
            except Exception:
                pass
            try:
                usage = job.usage_estimation()
            except Exception:
                usage = None

            result: PrimitiveResult = job.result()

        wall_end = time.time()
        ts_end = datetime.now().isoformat()

        # Qiskit 2.x SamplerV2: measurement register is usually 'meas' when using measure_all()
        pub = result[0]
        quasi_list = pub.data.meas.get_quasi_dists()
        quasi = quasi_list[0]
        prob_dict = {bitstr: float(prob) for bitstr, prob in quasi.items()}

        # Log sampler call
        sampler_log: Dict[str, Any] = {
            "tag": tag,
            "timestamp_start": ts_start,
            "timestamp_end": ts_end,
            "wall_time_sec": wall_end - wall_start,
            "backend": getattr(self.backend, "name", None),
            "shots": self.shots,
            "job_id": job_id,
            "job_creation_date": creation_date.isoformat() if hasattr(creation_date, "isoformat") else None,
            "usage_estimation": self._to_serializable(usage),
        }
        self.sampler_logs.append(sampler_log)

        # Store raw sampler result
        raw_sampler_dict = self._primitive_result_to_dict(result)
        self.sampler_logs[-1]["primitive_result"] = raw_sampler_dict

        return prob_dict

    # -------------------------------------------------------------------------
    # Main entry: run VQE
    # -------------------------------------------------------------------------
    def run_vqe(self):
        """
        Run the VQE optimization loop fully on real hardware (through EstimatorV2).

        Returns
        -------
        energy_list : list[float]
            Energies observed during optimization, in order of iterations.
        res.x : np.ndarray
            Optimized parameter vector from the classical optimizer.
        self.ansatz : QuantumCircuit
            The ansatz circuit used (logical layout).
        top_results : list[tuple[float, np.ndarray]]
            Top-6 (energy, params) pairs collected across all iterations.
        """
        pm = self._generate_pass_manager()
        ansatz_isa = pm.run(self.ansatz)
        hamiltonian_isa = self.hamiltonian.apply_layout(layout=ansatz_isa.layout)

        rng = np.random.default_rng()
        x0 = rng.random(self.ansatz.num_parameters)

        with Session(backend=self.backend) as session:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = self.shots

            res = minimize(
                self.cost_func,
                x0,
                args=(ansatz_isa, hamiltonian_isa, estimator),
                method="cobyla",
                options={"maxiter": self.maxiter},
            )

        sorted_results = sorted(self.iteration_results, key=lambda x: x[0])
        top_6_results = sorted_results[:6]

        return self.energy_list, res.x, self.ansatz, top_6_results
