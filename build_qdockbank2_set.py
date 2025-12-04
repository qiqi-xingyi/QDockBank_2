# --*-- conding:utf-8 --*--
# @time:12/2/25 23:37
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:build_qdockbank2_set.py

import random
import os
import re
import csv
import json
import math
import shutil
from collections import defaultdict

import requests


# ------------------------
# Config
# ------------------------

V1_INDEX_PATH = "Data/v1_index.txt"
PDBBIND_PL_ROOT = "PDBbind/P-L"
OUTPUT_ROOT = "QDockBank2_PDBbind_subset"
FRAGMENT_CSV = "qdb2_fragments.csv"
INFO_CSV = "qdb2_protein_info.csv"

MIN_LEN = 5
MAX_LEN = 18

# Extra targets per length group
# For lengths already present in v1: target = v1_count + EXTRA_PER_EXISTING
# For new lengths: target = EXTRA_PER_NEW
EXTRA_PER_EXISTING = 13  # previously 9, now +4 per group
EXTRA_PER_NEW = 19       # previously 15, now +4 per group

# Global cap on total number of fragments (v1 + new)
MAX_TOTAL_FRAGMENTS = 320

# Standard 20-aa mapping
AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# ------------------------
# Step 1: parse v1_index.txt
# ------------------------

def parse_v1_index(path):
    """
    Parse v1_index.txt and return:
      - fragments: list of dicts
      - used_sequences: set of aa sequences (single-letter)
      - length_counts: dict length -> count
      - pdb_source_map: pdb_id -> None (source path will be filled later)
    """
    fragments = []
    used_sequences = set()
    length_counts = defaultdict(int)
    pdb_ids = set()

    pdb_line_pattern = re.compile(
        r"^\s*([0-9a-zA-Z]{4})\tChain\s+(\w)\tResidues\s+(\d+)-(\d+)\tlength=(\d+)\t(\S+)\s*$"
    )

    with open(path, "r") as f:
        for line in f:
            m = pdb_line_pattern.match(line)
            if not m:
                continue
            pdb_id = m.group(1).lower()
            chain_id = m.group(2)
            res_start = int(m.group(3))
            res_end = int(m.group(4))
            length = int(m.group(5))
            seq = m.group(6).strip().upper()

            frag = {
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "res_start": res_start,
                "res_end": res_end,
                "length": length,
                "sequence": seq,
                "source_pocket_path": None,
                "source_dir": None,
                "origin": "v1",
            }
            fragments.append(frag)
            used_sequences.add(seq)
            length_counts[length] += 1
            pdb_ids.add(pdb_id)

    pdb_source_map = {pid: None for pid in pdb_ids}
    return fragments, used_sequences, length_counts, pdb_source_map


# ------------------------
# Step 2: scan pockets and generate new fragments
# ------------------------

def parse_pocket_pdb(path):
    """
    Parse a pocket PDB file and return:
      chain_id -> {resSeq -> resName3}
    """
    chains = defaultdict(dict)
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip().upper()
            chain_id = line[21].strip() or " "
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            if resseq not in chains[chain_id]:
                chains[chain_id][resseq] = resname
    return chains


def residues_to_fragments(chain_residues, pdb_id, chain_id):
    """
    Given dict resSeq->resName3, generate contiguous runs as list of
    (start_resSeq, end_resSeq, [aa1, aa1, ...]).
    """
    if not chain_residues:
        return []

    resnums = sorted(chain_residues.keys())
    runs = []
    run = [resnums[0]]

    for r in resnums[1:]:
        if r == run[-1] + 1:
            run.append(r)
        else:
            runs.append(run)
            run = [r]
    runs.append(run)

    contiguous_fragments = []
    for run_resnums in runs:
        aa1_list = []
        valid = True
        for r in run_resnums:
            res3 = chain_residues[r]
            if res3 not in AA3_TO_AA1:
                valid = False
                break
            aa1_list.append(AA3_TO_AA1[res3])
        if not valid:
            continue
        contiguous_fragments.append(
            (run_resnums[0], run_resnums[-1], aa1_list)
        )

    return contiguous_fragments


def generate_candidate_subfragments(pdb_id, chain_id, run_start, run_end, aa_list):
    """
    From a contiguous run, generate all subfragments with lengths in [MIN_LEN, MAX_LEN].
    """
    candidates = []
    run_len = len(aa_list)
    for L in range(MIN_LEN, MAX_LEN + 1):
        if L > run_len:
            continue
        for i in range(0, run_len - L + 1):
            sub_seq = "".join(aa_list[i:i + L])
            res_start = run_start + i
            res_end = res_start + L - 1
            candidates.append((L, res_start, res_end, sub_seq))
    return candidates


def build_target_counts(length_counts_v1):
    """
    For each length in [MIN_LEN, MAX_LEN], define a target count.
    If length appears in v1 -> v1_count + EXTRA_PER_EXISTING
    else -> EXTRA_PER_NEW
    """
    targets = {}
    for L in range(MIN_LEN, MAX_LEN + 1):
        base = length_counts_v1.get(L, 0)
        if base > 0:
            targets[L] = base + EXTRA_PER_EXISTING
        else:
            targets[L] = EXTRA_PER_NEW
    return targets


def scan_pdbbind_for_new_fragments(
    base_root,
    existing_fragments,
    used_sequences,
    length_counts_v1,
    max_total=MAX_TOTAL_FRAGMENTS,
):
    """
    Scan PDBbind/P-L folders, find candidate fragments, and select new ones.

    Rules:
      - Start from existing v1 fragments.
      - For v2_new, at most ONE fragment per pdb_id.
      - Only pick fragments with length in [MIN_LEN, MAX_LEN].
      - Avoid duplicate sequences.
      - Approximate per-length targets and per-year balance.
    """
    random.seed(123)

    target_counts = build_target_counts(length_counts_v1)
    selected_fragments = list(existing_fragments)  # include v1
    current_counts = dict(length_counts_v1)
    total_fragments = len(existing_fragments)

    # Record where each pdb comes from
    pdb_source_map = {}
    for frag in existing_fragments:
        pdb_source_map.setdefault(frag["pdb_id"], frag.get("source_dir", None))

    # v2_new constraint: at most one fragment per pdb_id
    used_new_pdb_ids = set()

    # Year directories
    year_dirs = [
        d for d in sorted(os.listdir(base_root))
        if os.path.isdir(os.path.join(base_root, d))
    ]

    num_years = max(len(year_dirs), 1)
    max_new_global = max_total - len(existing_fragments)
    per_year_cap = max(1, math.ceil(max_new_global / num_years))
    year_new_count = {yd: 0 for yd in year_dirs}

    for year_dir in year_dirs:
        year_path = os.path.join(base_root, year_dir)
        if not os.path.isdir(year_path):
            continue

        for pdb_id in sorted(os.listdir(year_path)):
            pdb_dir = os.path.join(year_path, pdb_id)
            if not os.path.isdir(pdb_dir):
                continue

            pdb_id_lower = pdb_id.lower()
            pocket_path = os.path.join(pdb_dir, f"{pdb_id_lower}_pocket.pdb")
            if not os.path.exists(pocket_path):
                continue

            # global cap
            if total_fragments >= max_total:
                break

            # per-year cap
            if year_new_count[year_dir] >= per_year_cap:
                continue

            # only one v2_new fragment per pdb_id
            if pdb_id_lower in used_new_pdb_ids:
                continue

            # record source dir (for copying later)
            pdb_source_map.setdefault(pdb_id_lower, pdb_dir)

            # parse residues and collect all candidate subfragments for this pdb
            chains = parse_pocket_pdb(pocket_path)
            candidates = []  # (chain_id, L, res_start, res_end, seq)

            for chain_id, res_map in chains.items():
                runs = residues_to_fragments(res_map, pdb_id_lower, chain_id)
                for run_start, run_end, aa_list in runs:
                    subfrags = generate_candidate_subfragments(
                        pdb_id_lower, chain_id, run_start, run_end, aa_list
                    )
                    for L, res_start, res_end, seq in subfrags:
                        if L < MIN_LEN or L > MAX_LEN:
                            continue
                        candidates.append((chain_id, L, res_start, res_end, seq))

            if not candidates:
                continue

            # randomize candidates so we do not always pick the same pattern
            random.shuffle(candidates)

            # try to pick exactly ONE fragment for this pdb
            chosen = None
            for chain_id, L, res_start, res_end, seq in candidates:
                if seq in used_sequences:
                    continue
                if current_counts.get(L, 0) >= target_counts[L]:
                    continue
                if total_fragments >= max_total:
                    break
                if year_new_count[year_dir] >= per_year_cap:
                    break

                chosen = {
                    "pdb_id": pdb_id_lower,
                    "chain_id": chain_id,
                    "res_start": res_start,
                    "res_end": res_end,
                    "length": L,
                    "sequence": seq,
                    "source_pocket_path": pocket_path,
                    "source_dir": pdb_dir,
                    "origin": "v2_new",
                }
                break

            if chosen is None:
                continue

            selected_fragments.append(chosen)
            used_sequences.add(chosen["sequence"])
            current_counts[chosen["length"]] = current_counts.get(chosen["length"], 0) + 1
            total_fragments += 1
            year_new_count[year_dir] += 1
            used_new_pdb_ids.add(pdb_id_lower)

        if total_fragments >= max_total:
            break

    return selected_fragments, pdb_source_map



# ------------------------
# Step 3: fetch RCSB info
# ------------------------

def fetch_rcsb_info(pdb_ids):
    """
    Fetch basic protein info from RCSB for each PDB ID.
    """
    info_list = []
    for pid in sorted(pdb_ids):
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pid}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                info_list.append(
                    {
                        "pdb_id": pid,
                        "title": None,
                        "classification": None,
                        "organism_ids": None,
                        "resolution": None,
                        "experimental_method": None,
                    }
                )
                continue
            data = r.json()
            title = data.get("struct", {}).get("title")
            classification = data.get("struct_keywords", {}).get("pdbx_keywords")
            entry_info = data.get("rcsb_entry_info", {})
            organism_ids = entry_info.get("source_organism_ids")
            expt = data.get("exptl", [{}])[0]
            method = expt.get("method")
            refinement = data.get("refine", [{}])[0]
            resolution = refinement.get("ls_d_res_high")

            info_list.append(
                {
                    "pdb_id": pid,
                    "title": title,
                    "classification": classification,
                    "organism_ids": organism_ids,
                    "resolution": resolution,
                    "experimental_method": method,
                }
            )
        except Exception:
            info_list.append(
                {
                    "pdb_id": pid,
                    "title": None,
                    "classification": None,
                    "organism_ids": None,
                    "resolution": None,
                    "experimental_method": None,
                }
            )
    return info_list


# ------------------------
# Step 4: copy selected PDBbind folders into flat subset
# ------------------------

def copy_selected_folders(pdb_source_map, output_root):
    os.makedirs(output_root, exist_ok=True)
    for pdb_id, src in pdb_source_map.items():
        if src is None:
            continue
        dst = os.path.join(output_root, pdb_id)
        if os.path.exists(dst):
            continue
        shutil.copytree(src, dst)


# ------------------------
# Step 5: main
# ------------------------

def main():
    # Step 1: v1 fragments
    v1_frags, used_seqs, length_counts_v1, pdb_source_map_v1 = parse_v1_index(
        V1_INDEX_PATH
    )
    print(f"Loaded {len(v1_frags)} v1 fragments.")
    print("Length distribution in v1:", dict(sorted(length_counts_v1.items())))

    # Step 2: scan PDBbind and select new fragments
    selected_frags, pdb_source_map_all = scan_pdbbind_for_new_fragments(
        PDBBIND_PL_ROOT,
        existing_fragments=v1_frags,
        used_sequences=used_seqs,
        length_counts_v1=length_counts_v1,
        max_total=MAX_TOTAL_FRAGMENTS,
    )
    print(f"Total selected fragments (v1 + new): {len(selected_frags)}")

    # Step 3: write fragment CSV
    with open(FRAGMENT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pdb_id",
                "chain_id",
                "res_start",
                "res_end",
                "length",
                "sequence",
                "origin",
                "source_pocket_path",
            ]
        )
        for frag in selected_frags:
            writer.writerow(
                [
                    frag["pdb_id"],
                    frag["chain_id"],
                    frag["res_start"],
                    frag["res_end"],
                    frag["length"],
                    frag["sequence"],
                    frag["origin"],
                    frag["source_pocket_path"],
                ]
            )
    print(f"Fragment summary written to {FRAGMENT_CSV}")

    # Sanity check: ensure v2_new fragments have unique pdb_id
    v2_new_pdb_ids = set()
    v2_new_duplicates = set()
    with open(FRAGMENT_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["origin"] != "v2_new":
                continue
            pid = row["pdb_id"]
            if pid in v2_new_pdb_ids:
                v2_new_duplicates.add(pid)
            else:
                v2_new_pdb_ids.add(pid)

    print(f"Number of v2_new fragments: {len(v2_new_pdb_ids)}")
    if v2_new_duplicates:
        print("WARNING: duplicated pdb_id in v2_new:", sorted(v2_new_duplicates))
    else:
        print("All v2_new pdb_id are unique.")

    # Step 4: fetch RCSB info
    pdb_ids = set(pdb_source_map_all.keys())
    info_list = fetch_rcsb_info(pdb_ids)
    with open(INFO_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pdb_id",
                "title",
                "classification",
                "organism_ids",
                "resolution",
                "experimental_method",
            ]
        )
        for info in info_list:
            writer.writerow(
                [
                    info["pdb_id"],
                    info["title"],
                    info["classification"],
                    json.dumps(info["organism_ids"]),
                    info["resolution"],
                    info["experimental_method"],
                ]
            )
    print(f"Protein info written to {INFO_CSV}")

    # Step 5: copy selected folders into flat subset
    copy_selected_folders(pdb_source_map_all, OUTPUT_ROOT)
    print(f"Selected PDBbind folders copied into {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
