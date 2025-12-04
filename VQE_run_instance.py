# --*-- conding:utf-8 --*--
# @time:12/3/25 19:19
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:VQE_run_instance.py

import os
import csv
import time

from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem

from qiskit_ibm_runtime import QiskitRuntimeService
from Qiskit_VQE import VQE5, StateCalculator


def predict_protein_structure(
    main_chain_sequence: str,
    protein_id: str,
    service: QiskitRuntimeService,
    max_iter: int = 150,
):
    """
    Run the quantum VQE workflow for a protein fragment and save all results into
    Quantum_result/{protein_id}/
    """

    print(f"Starting prediction for protein fragment: {protein_id}")
    print(f"Sequence: {main_chain_sequence}")

    # Side-chain placeholder (empty)
    side_chain_sequences = ['' for _ in range(len(main_chain_sequence))]

    chain_length = len(main_chain_sequence)
    print(f"Number of amino acids: {chain_length}")

    # Build peptide and Hamiltonian
    peptide = Peptide(main_chain_sequence, side_chain_sequences)
    mj_interaction = MiyazawaJerniganInteraction()
    penalty_terms = PenaltyParameters(10, 10, 10)
    protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)

    hamiltonian = protein_folding_problem.qubit_op()
    qubits_num = hamiltonian.num_qubits + 5
    print(f"Number of qubits (w/ padding): {qubits_num}")

    # VQE instance
    vqe_instance = VQE5(
        service=service,
        hamiltonian=hamiltonian,
        min_qubit_num=qubits_num,
        maxiter=max_iter,
    )

    print("Running VQE optimization...")
    energy_list, res, ansatz, top_results = vqe_instance.run_vqe()
    print("VQE optimization finished.")

    # === Output directory ===
    output_dir = os.path.join("Quantum_result", protein_id)
    os.makedirs(output_dir, exist_ok=True)

    # 1) Save energy trajectory
    energy_file = os.path.join(output_dir, "energy_list.txt")
    with open(energy_file, "w") as f:
        for val in energy_list:
            f.write(f"{val}\n")
    print(f"Energy list saved: {energy_file}")

    # 2) Compute probability distribution for the final parameters
    state_calculator = StateCalculator(service, qubits_num, ansatz)
    print("Computing probability distribution for final parameters...")
    probability_distribution = state_calculator.get_probability_distribution(res)

    prob_file = os.path.join(output_dir, "prob_distribution.txt")
    with open(prob_file, "w") as f:
        for bitstring, prob in probability_distribution.items():
            f.write(f"{bitstring}: {prob}\n")
    print(f"Probability distribution saved: {prob_file}")

    # 3) Interpret final structure and save as XYZ
    protein_result = protein_folding_problem.interpret(probability_distribution)
    protein_result.save_xyz_file(name=protein_id, path=output_dir)
    print(f"Main XYZ saved: {protein_id}.xyz")

    # 4) Top-K VQE results
    for rank, (energy_val, best_params) in enumerate(top_results, start=1):
        print(f"Top {rank} energy = {energy_val}")

        prob_dist_best = state_calculator.get_probability_distribution(best_params)
        protein_result_best = protein_folding_problem.interpret(prob_dist_best)
        xyz_name = f"{protein_id}_top_{rank}"

        protein_result_best.save_xyz_file(name=xyz_name, path=output_dir)
        print(f"Top {rank} XYZ saved: {xyz_name}.xyz")

    print(f"Fragment {protein_id} finished.\n")


def load_first_fragment_from_csv(csv_path: str):
    """Load the first fragment from QDB2 CSV."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row["pdb_id"].strip()
            chain_id = row["chain_id"].strip()
            res_start = row["res_start"].strip()
            res_end = row["res_end"].strip()
            sequence = row["sequence"].strip()

            fragment_id = f"{pdb_id}{chain_id}_{res_start}_{res_end}"

            print("CSV fragment loaded:")
            print(f"  pdb_id   = {pdb_id}")
            print(f"  chain_id = {chain_id}")
            print(f"  res      = {res_start}-{res_end}")
            print(f"  sequence = {sequence}")
            print(f"  fragment_id = {fragment_id}")

            return sequence, fragment_id

    raise RuntimeError("No fragment rows found in CSV.")


if __name__ == "__main__":
    print("Loading IBM Runtime account...")
    service = QiskitRuntimeService()
    print("IBM Runtime account loaded.")

    # Check backend
    backends = service.backends()
    print(f"Available backends: {[b.name for b in backends]}")

    csv_path = os.path.join("Input_Data", "qdb2_fragments.csv")
    sequence, fragment_id = load_first_fragment_from_csv(csv_path)

    # Log runtime
    log_file = "execution_time_log_qdb2_test.txt"
    with open(log_file, "w") as lf:
        lf.write("Fragment_ID\tExecution_Time(s)\n")

        start_time = time.time()

        predict_protein_structure(
            main_chain_sequence=sequence,
            protein_id=fragment_id,
            service=service,
            max_iter=150,
        )

        end_time = time.time()
        lf.write(f"{fragment_id}\t{end_time - start_time:.2f}\n")

    print("Done.")

