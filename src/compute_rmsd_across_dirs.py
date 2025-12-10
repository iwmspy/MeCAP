#!/usr/bin/env python

import os
import argparse
import csv
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def load_first_mol_from_sdf(sdf_path):
    """Load the first valid RDKit Mol from an SDF file.

    Parameters
    ----------
    sdf_path : str or Path
        Path to the SDF file.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        First valid molecule.

    Raises
    ------
    ValueError
        If no valid molecules or conformers are found in the SDF.
    """
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [m for m in supplier if m is not None]
    if not mols:
        raise ValueError(f"No valid molecules found in {sdf_path}")
    mol = mols[0]
    if mol.GetNumConformers() == 0:
        raise ValueError(f"Molecule in {sdf_path} has no conformers")
    return mol


def compute_rmsd_like_generate_far_conformer(
    name,
    dir_ref,
    path_ref,
    dir_far,
    path_far,
):
    """
    Compute RMSD between reference and far conformer SDFs in the same way as
    generate_far_conformer (AllChem.GetConformerRMS with fixed atom mapping).

    Parameters
    ----------
    name : str
        Compound name (file stem).
    dir_ref : str
        Directory path containing the reference conformer SDF (ref_conf_id).
    path_ref : str
        Path to reference SDF.
    dir_far : str
        Directory path containing the far conformer SDF (best_conf_id).
    path_far : str
        Path to far SDF.

    Returns
    -------
    tuple
        (name, dir_ref, dir_far, rmsd, path_ref, path_far, status)
        rmsd is float or None, status is "ok" or error message.
    """
    try:
        mol_ref = load_first_mol_from_sdf(path_ref)
        mol_far = load_first_mol_from_sdf(path_far)

        if mol_ref.GetNumAtoms() != mol_far.GetNumAtoms():
            return (name, dir_ref, dir_far, None, path_ref, path_far, "atom_count_mismatch")

        # Build a base molecule with two conformers:
        # conformer 0: reference (B, ref_conf_id)
        # conformer 1: far conformer (A, best_conf_id)
        base = Chem.Mol(mol_ref)  # copy to avoid modifying original

        src_conf_far = mol_far.GetConformer(0)
        far_conf_id = base.AddConformer(Chem.Conformer(src_conf_far), assignId=True)

        # This call should match generate_far_conformer:
        # AllChem.GetConformerRMS(work, ref_conf_id, best_conf_id)
        rmsd = AllChem.GetConformerRMS(
            base,
            confId1=0,            # reference conformer
            confId2=far_conf_id,  # far conformer
            atomIds=None,         # use all atoms
            prealigned=False,     # let RDKit align internally
        )

        return (name, dir_ref, dir_far, rmsd, path_ref, path_far, "ok")

    except Exception as e:
        return (name, dir_ref, dir_far, None, path_ref, path_far, f"error: {e}")


def collect_name_to_files(dirs):
    """Collect SDF files from directories and group by file stem.

    Parameters
    ----------
    dirs : list of str
        List of directory paths.

    Returns
    -------
    dict
        Mapping name -> list of (dir_path_str, sdf_path_str).
    """
    name_to_files = {}
    for d in dirs:
        dpath = Path(d)
        if not dpath.is_dir():
            print(f"Warning: {dpath} is not a directory, skipped.")
            continue
        for sdf_path in dpath.glob("*.sdf"):
            name = sdf_path.stem
            name_to_files.setdefault(name, []).append((str(dpath), str(sdf_path)))
    return name_to_files


def build_tasks(name_to_files, heavy_only=False):
    """Build RMSD calculation tasks for all common names across directories.

    Parameters
    ----------
    name_to_files : dict
        Mapping name -> list of (dir_path_str, sdf_path_str).
    heavy_only : bool
        Passed to the worker, kept here for completeness.

    Returns
    -------
    list
        List of task tuples to be passed to the worker.
    """
    tasks = []
    for name, entries in name_to_files.items():
        if len(entries) < 2:
            continue
        for (dir1, path1), (dir2, path2) in combinations(entries, 2):
            tasks.append((name, dir1, path1, dir2, path2))
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute RMSD of conformations stored in SDF files with the same "
            "name across multiple directories."
        )
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        help="Directories containing {name}.sdf files."
    )
    parser.add_argument(
        "--index-file",
        default=None,
        help="Index file to compute RMSD."
    )
    parser.add_argument(
        "-o", "--out-csv",
        default="rmsd_results.csv",
        help="Output CSV filename (default: rmsd_results.csv)."
    )
    parser.add_argument(
        "-j", "--n-jobs",
        type=int,
        default=None,
        help=(
            "Number of parallel processes (default: use all available cores). "
            "If set to 1, computation will be done sequentially."
        )
    )
    parser.add_argument(
        "--heavy-only",
        action="store_true",
        help="Use only heavy atoms (non-hydrogen) for RMSD calculation."
    )

    args = parser.parse_args()

    name_to_files = collect_name_to_files(args.dirs)
    tasks = build_tasks(name_to_files, heavy_only=args.heavy_only)

    if args.index_file:
        with open(args.index_file,'r') as f:
            sn_list = [idx.strip() for idx in f.readlines()]
        tasks = [t for t in tasks if t[0] in sn_list]

    if not tasks:
        print("No common SDF filenames found across the given directories.")
        return

    print(f"Found {len(tasks)} comparison tasks.")

    results = []
    n_jobs = args.n_jobs or os.cpu_count() or 1

    if n_jobs == 1:
        for t in tasks:
            result = compute_rmsd_like_generate_far_conformer(
                t[0], t[1], t[2], t[3], t[4]
            )
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_task = {
                executor.submit(
                    compute_rmsd_like_generate_far_conformer,
                    t[0], t[1], t[2], t[3], t[4]
                ): t
                for t in tasks
            }
            for i, future in enumerate(as_completed(future_to_task), 1):
                result = future.result()
                results.append(result)
                if i % 50 == 0 or i == len(tasks):
                    print(f"Processed {i} / {len(tasks)} tasks.")

    # Write CSV
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name",
            "dir1",
            "dir2",
            "rmsd",
            "path1",
            "path2",
            "status"
        ])
        for row in results:
            writer.writerow(row)

    print(f"Results written to {args.out_csv}")


if __name__ == "__main__":
    main()
