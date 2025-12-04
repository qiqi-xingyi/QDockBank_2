# --*-- coding:utf-8 --*--
# @time:12/3/25 19:19
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:VQE_run_instance.py

import os
import csv
import time
import json
from datetime import datetime
import argparse
import traceback

from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem

from qiskit_ibm_runtime import QiskitRuntimeService
from Qiskit_VQE import VQE5, StateCalculator


def predict_protein_structure(
    main_chain_sequence: str,
    fragment_id: str,
    pdb_id: str,
    service: QiskitRuntimeService,
    max_iter: int,
):
    """
    Run the VQE workflow on IBM Quantum hardware and store all outputs under
    Quantum_result/{pdb_id}/ with file names based on fragment_id.
    """

    print(f"Starting prediction for fragment: {fragment_id}")
    print(f"PDB ID: {pdb_id}")
    print(f"Sequence: {main_chain_sequence}")
    print(f"Max iterations: {max_iter}")

    # Build side chain placeholder
    side_chain_sequences = ['' for _ in range(len(main_chain_sequence))]

    chain_length = len(main_chain_sequence)
    print(f"Number of amino acids: {chain_length}")

    # Construct peptide, interaction model, and penalty terms
    peptide = Peptide(main_chain_sequence, side_chain_sequences)
    mj_interaction = MiyazawaJerniganInteraction()
    penalty_terms = PenaltyParameters(10, 10, 10)
    protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)

    # Construct Hamiltonian
    hamiltonian = protein_folding_problem.qubit_op()
    qubits_num = hamiltonian.num_qubits + 5
    print(f"Number of qubits (with padding): {qubits_num}")

    # Create VQE instance
    vqe_instance = VQE5(
        service=service,
        hamiltonian=hamiltonian,
        min_qubit_num=qubits_num,
        maxiter=max_iter,
    )

    print("Running VQE optimization...")
    energy_list, res, ansatz, top_results = vqe_instance.run_vqe()
    print("VQE optimization finished.")

    # Output directory
    output_dir = os.path.join("Quantum_result", pdb_id)
    os.makedirs(output_dir, exist_ok=True)

    # Save energy trajectory
    energy_file = os.path.join(output_dir, f"{fragment_id}_energy_list.txt")
    with open(energy_file, "w") as f:
        for val in energy_list:
            f.write(f"{val}\n")
    print(f"Energy list saved: {energy_file}")

    # Save iteration logs
    iter_log_file = os.path.join(output_dir, f"{fragment_id}_vqe_iter_logs.json")
    if hasattr(vqe_instance, "iter_logs"):
        with open(iter_log_file, "w") as f:
            json.dump(vqe_instance.iter_logs, f, indent=2, default=str)
        print(f"Iteration logs saved: {iter_log_file}")

    # Save raw primitive results (optional but valuable for analysis)
    raw_result_file = os.path.join(output_dir, f"{fragment_id}_vqe_raw_results.json")
    if hasattr(vqe_instance, "iter_raw_results"):
        with open(raw_result_file, "w") as f:
            json.dump(vqe_instance.iter_raw_results, f, indent=2, default=str)
        print(f"Raw primitive results saved: {raw_result_file}")

    # Compute measurement probabilities for final parameters (local simulator)
    state_calculator = StateCalculator(service, qubits_num, ansatz)
    print("Computing probability distribution for final parameters...")
    probability_distribution = state_calculator.get_probability_distribution(res)

    prob_file = os.path.join(output_dir, f"{fragment_id}_prob_distribution.txt")
    with open(prob_file, "w") as f:
        for bitstring, prob in probability_distribution.items():
            f.write(f"{bitstring}: {prob}\n")
    print(f"Probability distribution saved: {prob_file}")

    # Interpret final structure and save XYZ
    protein_result = protein_folding_problem.interpret(probability_distribution)
    protein_result.save_xyz_file(name=fragment_id, path=output_dir)
    print(f"XYZ saved: {fragment_id}.xyz")

    # Save top-k XYZ structures
    for rank, (energy_val, best_params) in enumerate(top_results, start=1):
        print(f"Top {rank} energy = {energy_val}")

        prob_dist_best = state_calculator.get_probability_distribution(best_params)
        protein_result_best = protein_folding_problem.interpret(prob_dist_best)

        xyz_name = f"{fragment_id}_top_{rank}"
        protein_result_best.save_xyz_file(name=xyz_name, path=output_dir)

        print(f"Top {rank} XYZ saved: {xyz_name}.xyz")

    print(f"Fragment {fragment_id} finished.\n")


def load_fragments_from_csv(csv_path: str):
    """
    Load all fragments from a QDB2-style CSV.
    Returns a list of dictionaries containing sequence, coordinates, and IDs.
    """
    fragments = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row["pdb_id"].strip()
            chain_id = row["chain_id"].strip()
            res_start = row["res_start"].strip()
            res_end = row["res_end"].strip()
            sequence = row["sequence"].strip()

            fragment_id = f"{pdb_id}{chain_id}_{res_start}_{res_end}"

            fragments.append(
                {
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "res_start": res_start,
                    "res_end": res_end,
                    "sequence": sequence,
                    "fragment_id": fragment_id,
                }
            )
    return fragments


def is_fragment_completed(fragment_id: str, pdb_id: str) -> bool:
    """
    Determine whether a fragment already completed processing.
    A fragment is considered complete if:
      - energy_list exists
      - probability distribution exists
      - final XYZ exists
    """
    out_dir = os.path.join("Quantum_result", pdb_id)
    energy_file = os.path.join(out_dir, f"{fragment_id}_energy_list.txt")
    prob_file = os.path.join(out_dir, f"{fragment_id}_prob_distribution.txt")
    xyz_file = os.path.join(out_dir, f"{fragment_id}.xyz")

    return os.path.exists(energy_file) and os.path.exists(prob_file) and os.path.exists(xyz_file)


def run_batch(
    csv_path: str,
    service: QiskitRuntimeService,
    sleep_minutes: int = 30,
    max_fragments: int | None = None,
    pdb_filter: str | None = None,
    resume: bool = True,
    log_file: str = "execution_time_log_qdb2_batch.txt",
    error_log_file: str = "execution_error_log_qdb2_batch.txt",
):
    """
    Batch executor for running VQE on all fragments in a CSV.
    Includes:
      - automatic skipping of completed fragments
      - automatic iteration number selection (len(sequence) * 10)
      - automatic error catching and error logging
      - sleep delay between fragments for load balancing on hardware
    """

    fragments = load_fragments_from_csv(csv_path)
    print(f"Total fragments in CSV: {len(fragments)}")

    # Optional PDB ID filter
    if pdb_filter is not None:
        fragments = [f for f in fragments if f["pdb_id"] == pdb_filter]
        print(f"Fragments after pdb_filter={pdb_filter}: {len(fragments)}")

    # Optional fragment limit
    if max_fragments is not None:
        fragments = fragments[:max_fragments]
        print(f"Limiting to first {max_fragments} fragments.")

    if not fragments:
        print("No fragments to process.")
        return

    # Open runtime log file
    new_log = not os.path.exists(log_file)
    lf = open(log_file, "a")
    if new_log:
        lf.write("Fragment_ID\tPDB_ID\tChain_ID\tRes_Start\tRes_End\tStart_Time\tEnd_Time\tExecution_Time(s)\n")

    # Open error log file
    new_err_log = not os.path.exists(error_log_file)
    ef = open(error_log_file, "a")
    if new_err_log:
        ef.write("Fragment_ID\tPDB_ID\tChain_ID\tRes_Start\tRes_End\tError_Time\tError_Type\tError_Message\n")

    try:
        for idx, frag in enumerate(fragments, start=1):
            fragment_id = frag["fragment_id"]
            pdb_id = frag["pdb_id"]
            chain_id = frag["chain_id"]
            res_start = frag["res_start"]
            res_end = frag["res_end"]
            sequence = frag["sequence"]

            print("=" * 80)
            print(f"[{idx}/{len(fragments)}] Processing fragment: {fragment_id}")

            # Resume check
            if resume and is_fragment_completed(fragment_id, pdb_id):
                print(f"Fragment {fragment_id} already complete. Skipping.")
                continue

            # Automatic iteration based on sequence length
            max_iter = len(sequence) * 10
            print(f"Auto-selected max_iter = {max_iter}, based on length={len(sequence)}")

            start_ts = datetime.now().isoformat()
            start_time = time.time()

            try:
                # VQE execution
                predict_protein_structure(
                    main_chain_sequence=sequence,
                    fragment_id=fragment_id,
                    pdb_id=pdb_id,
                    service=service,
                    max_iter=max_iter,
                )

                # Runtime logging
                end_time = time.time()
                end_ts = datetime.now().isoformat()
                elapsed = end_time - start_time

                lf.write(
                    f"{fragment_id}\t{pdb_id}\t{chain_id}\t{res_start}\t{res_end}\t"
                    f"{start_ts}\t{end_ts}\t{elapsed:.2f}\n"
                )
                lf.flush()

            except Exception as e:
                # Error logging
                err_ts = datetime.now().isoformat()
                err_type = type(e).__name__
                err_msg = str(e).replace("\n", " ")[:4000]

                print(f"[ERROR] Fragment {fragment_id} failed with {err_type}: {err_msg}")

                ef.write(
                    f"{fragment_id}\t{pdb_id}\t{chain_id}\t{res_start}\t{res_end}\t"
                    f"{err_ts}\t{err_type}\t{err_msg}\n"
                )
                ef.flush()
                continue  # Continue with next fragment

            # Sleep between fragments
            if idx < len(fragments):
                print(f"Sleeping for {sleep_minutes} minutes...")
                time.sleep(sleep_minutes * 60)

    finally:
        lf.close()
        ef.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Batch VQE runner for QDB2 fragments.")
    parser.add_argument("--csv", type=str, default="Input_Data/qdb2_fragments.csv")
    parser.add_argument("--sleep-minutes", type=int, default=30)
    parser.add_argument("--max-fragments", type=int, default=None)
    parser.add_argument("--pdb-filter", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--log-file", type=str, default="execution_time_log_qdb2_batch.txt")
    parser.add_argument("--error-log-file", type=str, default="execution_error_log_qdb2_batch.txt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Loading IBM Runtime account...")
    service = QiskitRuntimeService()
    print("IBM Runtime account loaded.")

    # Optional backend listing
    backends = service.backends()
    print(f"Available backends: {[b.name for b in backends]}")

    run_batch(
        csv_path=args.csv,
        service=service,
        sleep_minutes=args.sleep_minutes,
        max_fragments=args.max_fragments,
        pdb_filter=args.pdb_filter,
        resume=not args.no_resume,
        log_file=args.log_file,
        error_log_file=args.error_log_file,
    )

    print("Batch run finished.")
