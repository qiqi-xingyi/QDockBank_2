# --*-- conding:utf-8 --*--
# @time:12/2/25 22:19
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:run_qubit_scaling_test.py


import os
import time
import csv

import matplotlib.pyplot as plt

from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem

from qiskit_ibm_runtime import QiskitRuntimeService
from Qiskit_VQE import VQE5
from Qiskit_VQE import StateCalculator


def predict_protein_structure(
    main_chain_sequence: str,
    protein_id: str,
    service: QiskitRuntimeService,
    max_iter: int = 150
):
    """
    Use the given quantum VQE workflow to predict a protein structure based on the
    specified amino acid sequence, and store the results in corresponding directories.
    """

    print(f"Starting prediction for protein: {protein_id}, sequence: {main_chain_sequence}")

    side_chain_sequences = ['' for _ in range(len(main_chain_sequence))]

    chain_length = len(main_chain_sequence)
    print(f"Number of amino acids: {chain_length}")

    side_chain_count = len(side_chain_sequences)
    print(f"Number of side chain sites: {side_chain_count}")

    peptide = Peptide(main_chain_sequence, side_chain_sequences)
    mj_interaction = MiyazawaJerniganInteraction()
    penalty_terms = PenaltyParameters(10, 10, 10)

    protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
    hamiltonian = protein_folding_problem.qubit_op()

    qubits_num = hamiltonian.num_qubits + 5
    print(f"Number of qubits: {qubits_num}")

    vqe_instance = VQE5(
        service=service,
        hamiltonian=hamiltonian,
        min_qubit_num=qubits_num,
        maxiter=max_iter
    )

    energy_list, res, ansatz, top_results = vqe_instance.run_vqe()

    output_energy_path = f"Result/process_data/best_group/{protein_id}/System_Enegry"
    os.makedirs(output_energy_path, exist_ok=True)
    with open(f"{output_energy_path}/energy_list_{protein_id}.txt", 'w') as file:
        for item in energy_list:
            file.write(str(item) + '\n')

    state_calculator = StateCalculator(service, qubits_num, ansatz)
    probability_distribution = state_calculator.get_probability_distribution(res)

    protein_result = protein_folding_problem.interpret(probability_distribution)

    output_prob_path = f"Result/process_data/best_group/{protein_id}/Prob_distribution"
    os.makedirs(output_prob_path, exist_ok=True)
    with open(f"{output_prob_path}/prob_distribution.txt", 'w') as file:
        for key, value in probability_distribution.items():
            file.write(f'{key}: {value}\n')

    output_dir = f"Result/process_data/best_group/{protein_id}"
    os.makedirs(output_dir, exist_ok=True)
    protein_result.save_xyz_file(name=protein_id, path=output_dir)
    print("Protein structure saved as .xyz file")

    for rank, (energy_val, best_params) in enumerate(top_results, start=1):
        print(f"Top {rank} best energy = {energy_val}")

        prob_dist_best = state_calculator.get_probability_distribution(best_params)
        protein_result_best = protein_folding_problem.interpret(prob_dist_best)
        protein_result_best.save_xyz_file(
            name=f"{protein_id}_top_{rank}",
            path=output_dir
        )
        print(f"Protein structure for top {rank} best result has been saved.")

    print(f"Finished processing: {protein_id} \n")


def run_qubit_scaling_test(
    service: QiskitRuntimeService,
    max_qubits: int = 157,
    max_iter: int = 150,
    output_root: str = "Result/qubit_limit_test"
):
    """
    Increment sequence length starting from 1 amino acid.
    For each length:
      - Build a simple amino-acid sequence (e.g., all 'A')
      - Construct the protein folding Hamiltonian
      - Run VQE once
      - Record: sequence length, qubit count, VQE runtime, best energy
    Stop when the required number of qubits (hamiltonian.num_qubits + 5)
    exceeds max_qubits.

    Results:
      - Save CSV in output_root
      - Save plots (qubits vs length, time vs length) in output_root
    """

    os.makedirs(output_root, exist_ok=True)

    csv_path = os.path.join(output_root, "qubit_scaling_results.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "sequence_length",
            "num_qubits",
            "vqe_execution_time_s",
            "best_energy"
        ])

        sequence_length_list = []
        qubit_num_list = []
        time_list = []
        energy_list = []

        length = 1

        while True:
            main_chain_sequence = "A" * length
            side_chain_sequences = ['' for _ in range(length)]

            print("=" * 60)
            print(f"Testing sequence length: {length}")
            print(f"Sequence: {main_chain_sequence}")

            peptide = Peptide(main_chain_sequence, side_chain_sequences)
            mj_interaction = MiyazawaJerniganInteraction()
            penalty_terms = PenaltyParameters(10, 10, 10)
            protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
            hamiltonian = protein_folding_problem.qubit_op()

            qubits_num = hamiltonian.num_qubits + 5
            print(f"Number of qubits (hamiltonian + ancilla): {qubits_num}")

            if qubits_num > max_qubits:
                print(f"Reached qubit limit: {qubits_num} > {max_qubits}, stop.")
                break

            vqe_instance = VQE5(
                service=service,
                hamiltonian=hamiltonian,
                min_qubit_num=qubits_num,
                maxiter=max_iter
            )

            start_time = time.time()
            energy_vals, res, ansatz, top_results = vqe_instance.run_vqe()
            end_time = time.time()

            execution_time = end_time - start_time
            best_energy = min(energy_vals) if len(energy_vals) > 0 else None

            print(f"VQE execution time: {execution_time:.2f} s")
            print(f"Best energy: {best_energy}")

            writer.writerow([
                length,
                qubits_num,
                f"{execution_time:.6f}",
                best_energy
            ])

            sequence_length_list.append(length)
            qubit_num_list.append(qubits_num)
            time_list.append(execution_time)
            energy_list.append(best_energy)

            length += 1

    plt.figure()
    plt.plot(sequence_length_list, qubit_num_list, marker='o')
    plt.xlabel("Sequence length (number of amino acids)")
    plt.ylabel("Number of qubits")
    plt.title("Qubit usage vs amino acid sequence length")
    plt.grid(True)
    qubit_plot_path = os.path.join(output_root, "qubits_vs_sequence_length.png")
    plt.savefig(qubit_plot_path, dpi=300)
    plt.close()

    plt.figure()
    plt.plot(sequence_length_list, time_list, marker='o')
    plt.xlabel("Sequence length (number of amino acids)")
    plt.ylabel("VQE execution time (s)")
    plt.title("VQE execution time vs amino acid sequence length")
    plt.grid(True)
    time_plot_path = os.path.join(output_root, "vqe_time_vs_sequence_length.png")
    plt.savefig(time_plot_path, dpi=300)
    plt.close()

    print("\nQubit scaling test finished.")
    print(f"CSV saved to: {csv_path}")
    print(f"Plots saved to:\n  {qubit_plot_path}\n  {time_plot_path}")


if __name__ == '__main__':

    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance=' ',  # Replace with your real instance
        token=' '      # Replace with your real token
    )

    # For qubit limit and scaling test:
    run_qubit_scaling_test(
        service=service,
        max_qubits=157,
        max_iter=150,
        output_root="Result/qubit_limit_test"
    )


