#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find electrophilic and nucleophilic sites from SMILES and export two DataFrames.
Also optionally export a unique (name, smiles) mapping table.

Usage:
  python sites_cli.py --input input.csv --smiles-col smiles \
      --output-elec elec_sites.csv --output-nuc nuc_sites.csv \
      [--keep-meta] [--meta-cols colA,colB] [--name-col name] \
      [--output-unique unique_map.csv]

Notes:
  - Comments are written in English as requested.
  - Atom indices correspond to RDKit molecules after AddHs() and Kekulize()
    inside the provided atom-finding functions.
"""

import os
import sys
import argparse
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Set, Tuple as Tup
from rdkit import Chem
from tqdm import tqdm
import hashlib

from core_modules.atom import find_electrophilic_sites, find_nucleophilic_sites

encode_cache: Dict[bytes, str] = {}

def hash_smiles(smiles: str, canonicalize: bool = True) -> Optional[str]:
    # Deterministic name generator based on SMILES hash.
    # Falls back to non-canonical hashing if canonicalization fails.
    try:
        if canonicalize:
            m = Chem.MolFromSmiles(smiles)
            if m is None:
                raise ValueError("MolFromSmiles failed")
            base = Chem.MolToSmiles(m)
        else:
            base = smiles
    except Exception:
        base = smiles
    try:
        b = base.encode("utf-8")
        if b in encode_cache:
            return encode_cache[b]
        name = hashlib.md5(b).hexdigest()
        encode_cache[b] = name
        return name
    except Exception:
        return None

# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------

def read_table(path: str, smiles_col: str) -> pd.DataFrame:
    # Auto-detect by extension
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv"]:
        df = pd.read_csv(path)
    elif ext in [".tsv", ".tab", ".txt"]:
        df = pd.read_csv(path, sep="\t")
    elif ext in [".parquet", ".pq", ".pqt"]:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in input file.")
    return df

def write_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ""]:
        df.to_csv(path, index=False)
    elif ext in [".tsv", ".tab", ".txt"]:
        df.to_csv(path, index=False, sep="\t")
    elif ext in [".parquet", ".pq", ".pqt"]:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

# ------------------------------------------------------------
# Core
# ------------------------------------------------------------

def process_smiles(smiles: str) -> Tuple[List[int], List[str], List[int], List[str]]:
    # Return four lists: elec_sites, elec_names, nuc_sites, nuc_names
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], [], [], []
    try:
        e_sites, e_names, _ = find_electrophilic_sites(mol)
    except Exception:
        e_sites, e_names = [], []
    try:
        n_sites, n_names, _ = find_nucleophilic_sites(mol)
    except Exception:
        n_sites, n_names = [], []
    return e_sites, e_names, n_sites, n_names

def expand_records(
    smiles: str,
    name_value: Optional[str],
    meta: Dict[str, Any],
    sites: List[int],
    labels: List[str],
    site_col: str,
    label_col: str,
    include_meta_cols: List[str],
) -> List[Dict[str, Any]]:
    # Expand parallel lists into row dicts
    out: List[Dict[str, Any]] = []
    if not name_value:
        name_value = hash_smiles(smiles)
    for s, lab in zip(sites, labels):
        row = {
            "name": name_value,
            "smiles": smiles,
            site_col: int(s),
            label_col: lab,
        }
        for c in include_meta_cols:
            row[c] = meta.get(c, None)
        out.append(row)
    return out

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract electrophilic and nucleophilic sites from SMILES.")
    parser.add_argument("--input", required=True, help="Input table (csv/tsv/parquet) containing SMILES.")
    parser.add_argument("--smiles-col", default="smiles", help="Column name containing SMILES. Default: smiles")
    parser.add_argument("--name-col", default="name", help="Column to use as pre-existing name if present. Default: name")
    parser.add_argument("--meta-cols", default=None,
                        help="Comma-separated list of extra columns to carry over. Overrides --keep-meta if set.")
    parser.add_argument("--keep-meta", action="store_true",
                        help="Include all non-SMILES/non-name columns in outputs.")
    parser.add_argument("--output-elec", required=True, help="Output file for electrophilic sites (csv/tsv/parquet).")
    parser.add_argument("--output-nuc", required=True, help="Output file for nucleophilic sites (csv/tsv/parquet).")
    parser.add_argument("--output-unique", required=True,
                        help="Optional output file for unique (name, smiles) mapping (csv/tsv/parquet).")
    parser.add_argument("--drop-duplicates", action="store_true",
                        help="Drop duplicate rows in outputs.")
    args = parser.parse_args()

    df = read_table(args.input, args.smiles_col)

    # Determine which meta columns to include
    if args.meta_cols:
        include_meta_cols = [c.strip() for c in args.meta_cols.split(",") if c.strip()]
        missing = [c for c in include_meta_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in input: {missing}")
    elif args.keep_meta:
        include_meta_cols = [c for c in df.columns if c not in [args.smiles_col, args.name_col]]
    else:
        include_meta_cols = []

    elec_rows: List[dict] = []
    nuc_rows: List[dict] = []
    unique_pairs: Set[Tup[str, str]] = set()

    # Row-wise processing to keep per-row meta intact and collect unique mapping
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        smiles = str(row[args.smiles_col])
        name_value = row[args.name_col] if (args.name_col in df.columns) else None
        if not isinstance(name_value, str) or not name_value:
            name_value = hash_smiles(smiles)

        # Track unique mapping only if name is available
        if isinstance(name_value, str) and name_value:
            unique_pairs.add((name_value, smiles))

        meta_payload = {c: row[c] for c in include_meta_cols}

        e_sites, e_names, n_sites, n_names = process_smiles(smiles)

        elec_rows.extend(
            expand_records(
                smiles=smiles,
                name_value=name_value,
                meta=meta_payload,
                sites=e_sites,
                labels=e_names,
                site_col="elec_sites",
                label_col="elec_names",
                include_meta_cols=include_meta_cols,
            )
        )
        nuc_rows.extend(
            expand_records(
                smiles=smiles,
                name_value=name_value,
                meta=meta_payload,
                sites=n_sites,
                labels=n_names,
                site_col="nuc_sites",
                label_col="nuc_names",
                include_meta_cols=include_meta_cols,
            )
        )

    # Build DataFrames with stable column order
    elec_cols = ["name", "smiles", "elec_sites", "elec_names"] + include_meta_cols
    nuc_cols  = ["name", "smiles", "nuc_sites",  "nuc_names"]  + include_meta_cols

    elec_df = pd.DataFrame(elec_rows, columns=elec_cols)
    nuc_df  = pd.DataFrame(nuc_rows,  columns=nuc_cols)

    if args.drop_duplicates:
        elec_df = elec_df.drop_duplicates().reset_index(drop=True)
        nuc_df  = nuc_df.drop_duplicates().reset_index(drop=True)

    write_table(elec_df, args.output_elec)
    write_table(nuc_df, args.output_nuc)

    # Optional unique mapping output
    unique_df = pd.DataFrame(sorted(unique_pairs), columns=["name", "smiles"])
    write_table(unique_df, args.output_unique)
    print(f"Unique mapping: {len(unique_df)} rows -> {args.output_unique}")

    print(f"Electrophilic hits: {len(elec_df)} rows -> {args.output_elec}")
    print(f"Nucleophilic hits: {len(nuc_df)} rows -> {args.output_nuc}")

if __name__ == "__main__":
    main()
