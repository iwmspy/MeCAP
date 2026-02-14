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

import argparse
import pandas as pd

from core_modules.conformer import conformergen_batch

# ---------- CLI ----------

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch xTB optimization job runner from SMILES DataFrame.")
    p.add_argument("--input-csv", required=True, help="Input CSV with at least a 'smiles' column; optional 'name' column.")
    p.add_argument("--out-dir", required=True, help="Directory to write SDF files.")
    p.add_argument("--out-csv", required=True, help="Path to write result CSV with appended columns.")
    p.add_argument("--smiles-col", default="smiles", help="Column name for SMILES in input CSV.")
    p.add_argument("--name-col", default="name", help="Optional column name for molecule names.")
    p.add_argument("--init-mode", choices=["unimol","esnuel"], default="esnuel", help="How to optimize initial state by RDKit.")
    p.add_argument("--final-mode", choices=["xtb","rdkit"], default="xtb", help="Choose 'rdkit' to skip xTB and output RDKit geometry.")
    p.add_argument("--max-workers", type=int, default=2, help="Number of molecules to run in parallel.")
    p.add_argument("--xtb-threads", type=int, default=2, help="Number of threads per molecule for xTB.")
    p.add_argument("--gfn-level", type=int, default=1, help="GFN-xTB level (default 1).")
    p.add_argument("--uhf", type=int, default=0, help="Number of unpaired electrons for UHF (default 0).")
    p.add_argument("--only-2d", action="store_true", help="If True, return 2D-flatten conformations.")
    p.add_argument("--seed", type=int, default=123, help="Base random seed for embeddings.")
    p.add_argument("--save-mode", choices=["archive","keep","none","zip"], default="archive", help="Intermediate files handling.")
    p.add_argument("--archive-format", choices=["tar.xz","tar.gz","zip"], default="tar.xz", help="Archive format when save-mode=archive.")
    p.add_argument("--work-parent", default=None, help="Parent dir for workspaces. Default: system temp.")
    return p

def main() -> None:
    args = _build_cli().parse_args()

    df = pd.read_csv(args.input_csv)
    conformergen_batch(
        df=df,
        out_dir=args.out_dir,
        out_csv=args.out_csv,
        init_mode=args.init_mode,
        final_mode=args.final_mode,   # NEW
        only_2D=args.only_2d,
        smiles_col=args.smiles_col,
        name_col=args.name_col if args.name_col else None,
        max_workers=args.max_workers,
        xtb_threads=args.xtb_threads,
        gfn_level=args.gfn_level,
        uhf=args.uhf,
        seed=args.seed,
        save_mode=args.save_mode,
        archive_format=args.archive_format,
        work_parent=args.work_parent,
    )

if __name__ == "__main__":
    main()

### for debugging ###

# if __name__=='__main__':
#     # Best compression, keep minimal archive size
#     mol = optimize_with_xtb_from_smiles(
#         smiles="c1ccccc1",
#         name="benzene",
#         archive_format="tar.xz",
#         save_mode="archive",
#         xtb_threads=4,
#     )

#     # Faster archiving, decent compression
#     mol = optimize_with_xtb_from_smiles(
#         smiles="CCO",
#         name="ethanol",
#         archive_format="tar.gz",
#         save_mode="archive",
#         xtb_threads=8,
#     )

#     # Maximum portability (Windows-native)
#     write_sdf_with_xtb_from_smiles(
#         smiles="CC(=O)O",
#         sdf_path="acetic_acid.sdf",
#         name="acetic",
#         archive_format="zip",
#         save_mode="archive",
#         xtb_threads=2,
#     )
