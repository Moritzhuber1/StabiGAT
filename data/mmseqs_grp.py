#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def run(cmd: List[str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_first_fasta_sequence(fasta_path: Path) -> str:
    """
    Reads the first FASTA record's sequence from a file.
    Assumes standard FASTA format.
    """
    seq_lines: List[str] = []
    in_record = False
    with fasta_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if in_record and seq_lines:
                    break
                in_record = True
                continue
            if in_record:
                seq_lines.append(line)
    seq = "".join(seq_lines).replace(" ", "").replace("\t", "")
    if not seq:
        raise ValueError(f"No sequence found in {fasta_path}")
    return seq


def build_combined_fasta(fasta_dir: Path, combined_fasta: Path) -> List[str]:
    """
    Concatenates per-sample FASTAs into a single FASTA.
    Uses file stem as sequence ID (e.g., H_Pin1_W11A from H_Pin1_W11A.fasta).
    Returns list of names included.
    """
    fasta_files = sorted(list(fasta_dir.glob("*.fasta"))) + sorted(list(fasta_dir.glob("*.fa")))
    if not fasta_files:
        raise FileNotFoundError(f"No .fasta/.fa files found in {fasta_dir}")

    names: List[str] = []
    with combined_fasta.open("w") as out:
        for fp in fasta_files:
            name = fp.stem
            seq = read_first_fasta_sequence(fp)
            out.write(f">{name}\n{seq}\n")
            names.append(name)

    return names


def mmseqs_cluster_to_tsv(
    combined_fasta: Path,
    out_dir: Path,
    min_seq_id: float,
    coverage: float,
    threads: int,
) -> Path:
    """
    Runs MMseqs2 clustering and returns path to clusters TSV (rep<tab>member).
    Uses:
      mmseqs createdb
      mmseqs cluster
      mmseqs createtsv
    """
    mmseqs = shutil.which("mmseqs")
    if mmseqs is None:
        raise RuntimeError("mmseqs not found in PATH. Install MMseqs2 or add it to PATH.")

    out_dir.mkdir(parents=True, exist_ok=True)
    db = out_dir / "seqdb"
    clu = out_dir / "clu"
    tmp = out_dir / "tmp"
    tsv = out_dir / "clusters.tsv"

    # Clean previous artifacts (optional but avoids stale outputs)
    for p in [db, clu, tmp, tsv]:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()

    run(["mmseqs", "createdb", str(combined_fasta), str(db)])

    # cov-mode:
    # 0 = coverage of the alignment over query/target depends on -c
    # You can tune this later; it's a reasonable default for identity clustering.
    run([
        "mmseqs", "cluster",
        str(db), str(clu), str(tmp),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", "0",
        "--threads", str(threads),
    ])

    run(["mmseqs", "createtsv", str(db), str(db), str(clu), str(tsv)])

    if not tsv.exists():
        raise RuntimeError("MMseqs finished but clusters.tsv was not created. Check MMseqs output above.")
    return tsv


def make_group_mapping_from_clusters_tsv(tsv_path: Path) -> Dict[str, int]:
    """
    clusters.tsv contains lines: representative \t member
    We map each representative to an integer cluster id and assign members to it.
    """
    rep_to_id: Dict[str, int] = {}
    name_to_group: Dict[str, int] = {}

    with tsv_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rep, member = line.split("\t")[:2]
            if rep not in rep_to_id:
                rep_to_id[rep] = len(rep_to_id)
            cid = rep_to_id[rep]
            name_to_group[member] = cid

    if not name_to_group:
        raise RuntimeError(f"No clusters parsed from {tsv_path}")
    return name_to_group


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("data/raw/final.csv"))
    ap.add_argument("--fasta_dir", type=Path, default=Path("data/fasta"))
    ap.add_argument("--out_tsv", type=Path, default=Path("data/cache/groups_seqid.tsv"))
    ap.add_argument("--out_npy", type=Path, default=Path("data/cache/groups_seqid.npy"))
    ap.add_argument("--work_dir", type=Path, default=Path("data/cache/mmseqs_work"))
    ap.add_argument("--min_seq_id", type=float, default=0.90, help="Sequence identity threshold, e.g. 0.9")
    ap.add_argument("--coverage", type=float, default=0.80, help="Alignment coverage threshold, e.g. 0.8")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no_npy", action="store_true", help="Do not write the aligned .npy groups array")
    args = ap.parse_args()

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "name" not in df.columns:
        raise ValueError(f"'name' column not found in {args.csv}")

    # 1) Build combined FASTA from folder
    combined_fasta = args.work_dir / "all_sequences.fasta"
    fasta_names = set(build_combined_fasta(args.fasta_dir, combined_fasta))
    print(f"[INFO] Combined FASTA written: {combined_fasta}  (n={len(fasta_names)})")

    # 2) Run MMseqs2 clustering
    clusters_tsv = mmseqs_cluster_to_tsv(
        combined_fasta=combined_fasta,
        out_dir=args.work_dir,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
        threads=args.threads,
    )
    print(f"[INFO] MMseqs clusters TSV: {clusters_tsv}")

    # 3) Parse mapping: name -> cluster_id
    name_to_group = make_group_mapping_from_clusters_tsv(clusters_tsv)

    # 4) Write groups_seqid.tsv for all names in your dataset
    dataset_names = df["name"].astype(str).tolist()
    missing = [n for n in dataset_names if n not in name_to_group]
    if missing:
        # common cause: fasta filenames don't match df["name"]
        raise ValueError(
            f"{len(missing)} dataset names are missing from MMseqs clustering results.\n"
            f"Examples: {missing[:15]}\n"
            f"Tip: ensure FASTA file stems match df['name'] OR adjust this script to map them."
        )

    out_rows = [(n, int(name_to_group[n])) for n in dataset_names]
    out_df = pd.DataFrame(out_rows, columns=["name", "cluster"])
    out_df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"[OK] Wrote: {args.out_tsv}  (n={len(out_df)}, clusters={out_df['cluster'].nunique()})")

    # 5) Optional: aligned numpy array (same order as df)
    if not args.no_npy:
        groups = np.array([int(name_to_group[n]) for n in dataset_names], dtype=int)
        np.save(args.out_npy, groups)
        print(f"[OK] Wrote: {args.out_npy}  shape={groups.shape}")


if __name__ == "__main__":
    main()
