# -*- coding: utf-8 -*-
"""
End-to-end conformer generation and xTB optimization from SMILES.
All comments are in English.

Key changes:
- Allow choosing archive format: tar.xz (default), tar.gz, or zip.
- Allow controlling the number of threads for xTB via function argument.
- Keep avoiding rdDetermineBonds; bonding comes from SMILES only.
- Keep the design influenced by the provided GraphChargeShell.py. See user's attachment.
"""

import argparse, os
import pandas as pd

# ---------- CLI ----------

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sanity check of SMILES from pd.DataFrame and Orca.")
    p.add_argument("--input-csv", required=True, help="Input CSV with at least a 'smiles' column and 'name' column.")
    p.add_argument("--name-col", default="name", help="Column name for molecule names.")
    p.add_argument("--index-file", required=True, help="Index file for extracting (corresponding to name-col).")
    p.add_argument("--out-csv", required=True, help="Path to write result CSV.")
    return p

def main() -> None:
    args = _build_cli().parse_args()

    with open(args.index_file, 'r') as f:
        name_list = [l.strip() for l in f.readlines()]

    print(f'Processing {args.input_csv}...')
    df = pd.read_csv(args.input_csv)
    print(f'Original length: {df.shape[0]}')
    df = df[df[args.name_col].astype(str).isin(name_list)]
    print(f'Extracted length: {df.shape[0]}')

    df.to_csv(args.out_csv,index=False)
    print(f'Extracted lows saved in {args.out_csv}!')

if __name__ == "__main__":
    main()
