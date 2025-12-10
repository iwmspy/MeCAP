# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2025 Yuto Iwasaki
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
from rdkit import Chem
from tqdm import tqdm
tqdm.pandas()

from core_modules.conformer import convert_xyz_to_smiles_from_file

# ---------- CLI ----------

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sanity check of SMILES from pd.DataFrame and Orca.")
    p.add_argument("--input-csv", required=True, help="Input CSV with at least a 'smiles' column and 'name' column.")
    p.add_argument("--input-dirs", required=True, nargs='*', help="Input directory storing Orca-optimized sdf (Need bond determination).")
    p.add_argument("--input-dirs-no-determine", nargs='*', help="Input directory storing Orca-optimized sdf (No need bond determination).")
    p.add_argument("--out-csv", required=True, help="Path to write result CSV.")
    p.add_argument("--smiles-col", default="smiles", help="Column name for SMILES in input CSV.")
    p.add_argument("--name-col", default="name", help="Column name for molecule names.")
    return p

def sanitize_smiles(smi):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)

def sdf_to_smiles(infile):
    try:
        suppl = Chem.SDMolSupplier(infile, removeHs=False, sanitize=True)
        mols = [m for m in suppl if m is not None]
        return [Chem.MolToSmiles(m) for m in mols] if len(mols) > 1 else Chem.MolToSmiles(mols[0])
    except:
        return None

def main() -> None:
    args = _build_cli().parse_args()

    df = pd.read_csv(args.input_csv)[[args.name_col, args.smiles_col]]
    name   = df[args.name_col]
    df['smiles_from_dataframe'] = df[args.smiles_col].progress_apply(sanitize_smiles)

    to_sanity_check = []
    
    for direc in args.input_dirs:
        if direc.endswith('/'):
            direc = direc[:-1]
        df[f'smiles_from_{os.path.basename(direc)}'] = [
            convert_xyz_to_smiles_from_file(os.path.join(direc, f'{n}.sdf')) 
            if os.path.exists(os.path.join(direc, f'{n}.sdf')) else None
            for n in tqdm(name)
            ]
        df[f'match_with_{os.path.basename(direc)}'] = [a == b for a, b in tqdm(zip(df['smiles_from_dataframe'],df[f'smiles_from_{os.path.basename(direc)}']))]
        to_sanity_check.append(f'match_with_{os.path.basename(direc)}')
    
    for direc in args.input_dirs_no_determine:
        if direc.endswith('/'):
            direc = direc[:-1]
        df[f'smiles_from_{os.path.basename(direc)}'] = [
            sdf_to_smiles(os.path.join(direc, f'{n}.sdf')) 
            if os.path.exists(os.path.join(direc, f'{n}.sdf')) else None
            for n in tqdm(name)
            ]
        df[f'match_with_{os.path.basename(direc)}'] = [a == b for a, b in tqdm(zip(df['smiles_from_dataframe'],df[f'smiles_from_{os.path.basename(direc)}']))]
        to_sanity_check.append(f'match_with_{os.path.basename(direc)}')
    
    df['sanity'] = df.apply(lambda row: all(row[to_sanity_check]), axis=1)
    df.to_csv(args.out_csv,index=False)
    df[df['sanity']][args.name_col].to_csv(args.out_csv.replace('.csv','.index'),index=False,header=False)

if __name__ == "__main__":
    main()
