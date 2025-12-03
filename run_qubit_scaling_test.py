# -*- coding: utf-8 -*-
# @Time : 1/17/25 11:14 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : run_qubit_scaling_test.py

import os
import csv
import time
import matplotlib.pyplot as plt

from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem


def run_qubit_encoding_test(
    max_qubits: int = 157,
    output_root: str = "Result/qubit_encoding_test",
    start_length: int = 5
):
    os.makedirs(output_root, exist_ok=True)

    csv_path = os.path.join(output_root, "qubit_encoding_results.csv")

    sequence_length_list = []
    hamiltonian_qubits_list = []
    total_qubits_list = []
    build_time_list = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sequence_length",
                "hamiltonian_qubits",
                "total_qubits",
                "hamiltonian_build_time_s",
            ]
        )

        length = start_length

        while True:
            main_chain_sequence = "A" * length
            side_chain_sequences = [""] * length

            print("=" * 60)
            print(f"Testing sequence length: {length}")
            print(f"Sequence: {main_chain_sequence}")

            start_build = time.time()

            peptide = Peptide(main_chain_sequence, side_chain_sequences)
            mj_interaction = MiyazawaJerniganInteraction()
            penalty_terms = PenaltyParameters(10, 10, 10)
            protein_folding_problem = ProteinFoldingProblem(
                peptide, mj_interaction, penalty_terms
            )
            hamiltonian = protein_folding_problem.qubit_op()

            end_build = time.time()
            build_time = end_build - start_build

            hamiltonian_qubits = hamiltonian.num_qubits
            total_qubits = hamiltonian_qubits + 5

            print(f"Hamiltonian qubits: {hamiltonian_qubits}")
            print(f"Total qubits (hamiltonian + 5): {total_qubits}")
            print(f"Hamiltonian build time: {build_time:.4f} s")

            if total_qubits > max_qubits:
                print(
                    f"Reached qubit limit: total_qubits {total_qubits} > max_qubits {max_qubits}. Stop."
                )
                break

            writer.writerow(
                [
                    length,
                    hamiltonian_qubits,
                    total_qubits,
                    f"{build_time:.6f}",
                ]
            )

            sequence_length_list.append(length)
            hamiltonian_qubits_list.append(hamiltonian_qubits)
            total_qubits_list.append(total_qubits)
            build_time_list.append(build_time)

            length += 1

    # Plot: total qubits vs sequence length
    plt.figure()
    plt.plot(sequence_length_list, total_qubits_list, marker="o")
    plt.xlabel("Sequence length (number of amino acids)")
    plt.ylabel("Total qubits (hamiltonian + 5)")
    plt.title("Total qubits vs amino acid sequence length")
    plt.grid(True)
    plt.savefig(os.path.join(output_root, "total_qubits_vs_sequence_length.png"), dpi=300)
    plt.close()

    # Plot: Hamiltonian qubits vs sequence length
    plt.figure()
    plt.plot(sequence_length_list, hamiltonian_qubits_list, marker="o")
    plt.xlabel("Sequence length (number of amino acids)")
    plt.ylabel("Hamiltonian qubits")
    plt.title("Hamiltonian qubits vs amino acid sequence length")
    plt.grid(True)
    plt.savefig(os.path.join(output_root, "hamiltonian_qubits_vs_sequence_length.png"), dpi=300)
    plt.close()

    # Plot: Hamiltonian build time vs sequence length
    plt.figure()
    plt.plot(sequence_length_list, build_time_list, marker="o")
    plt.xlabel("Sequence length (number of amino acids)")
    plt.ylabel("Hamiltonian build time (s)")
    plt.title("Hamiltonian build time vs amino acid sequence length")
    plt.grid(True)
    plt.savefig(os.path.join(output_root, "hamiltonian_build_time_vs_sequence_length.png"), dpi=300)
    plt.close()

    print("\nQubit encoding test finished.")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    run_qubit_encoding_test(
        max_qubits=157,
        output_root="Result/qubit_encoding_test",
        start_length=5
    )
