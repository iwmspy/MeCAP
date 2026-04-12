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

import os
import tempfile
import time
import traceback
from typing import List, Optional, Tuple, Dict, Any, Sequence
import shutil
import subprocess
import uuid
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds

# ----------------------------
# Core API
# ----------------------------

def _resolve_xtb_exe() -> str:
    """Resolve xtb executable path, preferring $XTBHOME/bin/xtb if available."""
    xtbhome = os.environ.get("CONDA_PREFIX", "").strip()
    if xtbhome:
        candidate = Path(xtbhome) / "bin" / "xtb"
        if candidate.exists():
            return str(candidate)
    return "xtb"  # fallback to PATH

xtb_exe = _resolve_xtb_exe()
print(f'[Info] xTB path: {xtb_exe}')

def inner_smi2coords_unimol(
    smi,
    seed=42,
    mode='fast',
    remove_hs=True,
    return_mol=False,
    only_2D=False,
    permute_atom_tokens=False,
    permute_heavy_only=True,
):
    '''
    This function is responsible for converting a SMILES (Simplified Molecular Input Line Entry System) string into 3D coordinates for each atom in the molecule. It also allows for the generation of 2D coordinates if 3D conformation generation fails, and optionally removes hydrogen atoms and their coordinates from the resulting data.

    :param smi: (str) The SMILES representation of the molecule.
    :param seed: (int, optional) The random seed for conformation generation. Defaults to 42.
    :param mode: (str, optional) The mode of conformation generation, 'fast' for quick generation, 'heavy' for more attempts. Defaults to 'fast'.
    :param remove_hs: (bool, optional) Whether to remove hydrogen atoms from the final coordinates. Defaults to True.

    :return: A tuple containing the list of atom symbols and their corresponding 3D coordinates.
    :raises AssertionError: If no atoms are present in the molecule or if the coordinates do not align with the atom count.

    --
    Change from original: Conformational energy is added to the mol object.
    '''
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    assert len(atoms) > 0, 'No atoms in molecule: {}'.format(smi)
    try:
        # will random generate conformer with seed equal to -1. else fixed random seed.
        res = AllChem.EmbedMolecule(mol, randomSeed=seed) if not only_2D else 1
        if res == 0:
            try:
                '''Add'''
                # Build MMFF properties (includes atom types, charges, etc.)
                mp = AllChem.MMFFGetMoleculeProperties(mol)
                if mp is None:
                    raise ValueError("MMFF parameters could not be assigned to this molecule.")
                '''/Add'''
                # some conformer can not use MMFF optimize
                AllChem.MMFFOptimizeMolecule(mol)
                '''Add'''
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
                best_energy = float(ff.CalcEnergy())
                '''/Add'''
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                best_energy = float('inf')
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        ## for fast test... ignore this ###
        elif res == -1 and mode == 'heavy':
            AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
            try:
                '''Add'''
                # Build MMFF properties (includes atom types, charges, etc.)
                mp = AllChem.MMFFGetMoleculeProperties(mol)
                if mp is None:
                    raise ValueError("MMFF parameters could not be assigned to this molecule.")
                '''/Add'''
                # some conformer can not use MMFF optimize
                AllChem.MMFFOptimizeMolecule(mol)
                '''Add'''
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
                best_energy = float(ff.CalcEnergy())
                '''/Add'''
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                best_energy = float('inf')
                AllChem.Compute2DCoords(mol)
                coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
                coordinates = coordinates_2d
        else:
            best_energy = float('inf')
            AllChem.Compute2DCoords(mol)
            coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
            coordinates = coordinates_2d
    except:
        print("Failed to generate conformer, replace with zeros.")
        coordinates = np.zeros((len(atoms), 3))
        best_energy = float('inf')
    # Optional deterministic atom-identity permutation (token ablation).
    # Coordinates are kept unchanged.
    if permute_atom_tokens:
        rng = np.random.default_rng(seed)
        try:
            idxs = []
            zs = []
            for a in mol.GetAtoms():
                z = int(a.GetAtomicNum())
                if permute_heavy_only and z == 1:
                    continue
                idxs.append(int(a.GetIdx()))
                zs.append(z)
            rng.shuffle(zs)
            for i, z in zip(idxs, zs):
                mol.GetAtomWithIdx(i).SetAtomicNum(int(z))
            mol.SetIntProp('atom_permutation_seed',int(seed))
            atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        except Exception:
            # If anything fails, keep the original atom types.
            pass
    else:
        mol.SetDoubleProp('mmff_min_energy_kcalmol', best_energy)

    if return_mol:
        return mol  # for unimolv2

    assert len(atoms) == len(
        coordinates
    ), "coordinates shape is not align with {}".format(smi)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(
            coordinates_no_h
        ), "coordinates shape is not align with {}".format(smi)
        return atoms_no_h, coordinates_no_h, mol
    else:
        return atoms, coordinates, mol

def optimize_conf_from_smiles(
    smiles: str,
    name: str = "molecule",
    gfn_level: int = 1,
    uhf: int = 0,
    seed: int = 123,
    init_mode: Optional[str] = "unimol",
    final_mode: str = "rdkit",           # NEW
    only_2D: Optional[bool] = False,
    permute_atom_tokens: Optional[bool] = False,
    permute_heavy_only: Optional[bool] = True,
    save_mode: str = "archive",
    archive_format: str = "tar.xz",
    xtb_threads: Optional[int] = None,
    work_parent: Optional[str] = None,
) -> Chem.Mol:
    if init_mode != "unimol":
        UserWarning(f'Selected init_mode "{init_mode}" is not supported. Force changed to "unimol" mode.')
    if final_mode != "rdkit":
        UserWarning(f'Selected final_mode "{final_mode}" is not supported. Force changed to "rdkit" mode.')
    # RDKit-only finalization
    mol = inner_smi2coords_unimol(smiles, seed, return_mol=True, only_2D=only_2D, permute_atom_tokens=permute_atom_tokens, permute_heavy_only=permute_heavy_only)
    return mol

def write_optimized_sdf_from_smiles(
    smiles: str,
    sdf_path: str,
    name: str = "molecule",
    gfn_level: int = 1,
    uhf: int = 0,
    seed: int = 123,
    init_mode: Optional[str] = "esnuel",
    final_mode: str = "xtb",            # NEW
    only_2D: Optional[bool] = False,
    permute_atom_tokens: Optional[bool] = False,
    permute_heavy_only: Optional[bool] = True,
    save_mode: str = "archive",
    archive_format: str = "tar.xz",
    xtb_threads: Optional[int] = None,
    work_parent: Optional[str] = None,
    return_mol: bool = False
) -> Optional[Chem.Mol]:
    """
    Wrapper that runs optimize_with_xtb_from_smiles and writes the result to an SDF file.
    When final_mode='rdkit', it writes the RDKit-minimized conformer (best-MMFF) directly.
    """
    mol = optimize_conf_from_smiles(
        smiles=smiles,
        name=name,
        gfn_level=gfn_level,
        uhf=uhf,
        seed=seed,
        init_mode=init_mode,
        final_mode=final_mode,       # pass through
        only_2D=only_2D,
        permute_atom_tokens=permute_atom_tokens,
        permute_heavy_only=permute_heavy_only,
        save_mode=save_mode,
        archive_format=archive_format,
        xtb_threads=xtb_threads,
        work_parent=work_parent,
    )

    # Choose which conformer to write:
    conf_id = -1

    writer = Chem.SDWriter(sdf_path)
    try:
        if permute_atom_tokens:
            writer.SetKekulize(False)
            writer.write(mol, confId=conf_id)
        else:
            writer.write(mol, confId=conf_id)
    except Exception as e:
        print(f'[Writing Error!!] {e}')
    finally:
        writer.close()
    if return_mol:
        return mol

# ---------- helpers ----------

def _slugify_name(s: str) -> str:
    """Create a filesystem-friendly slug for a given name."""
    keep = "-_.()[]{}"
    out = []
    for ch in s:
        if ch.isalnum() or ch in keep:
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("._")
    return slug or "molecule"

def _ensure_dir(p: Path) -> None:
    """Create directory if missing."""
    p.mkdir(parents=True, exist_ok=True)

def _single_job(
    idx: int,
    smiles: str,
    name: Optional[str],
    out_dir: str,
    init_mode: Optional[str],
    final_mode: str,
    only_2D: Optional[bool],
    permute_atom_tokens: Optional[bool],
    permute_heavy_only: Optional[bool],
    gfn_level: int,
    uhf: int,
    seed: int,
    save_mode: str,
    archive_format: str,
    xtb_threads: int,
    work_parent: Optional[str],
) -> Tuple[int, Dict[str, Any]]:
    """
    Worker function executed in a separate process.
    Returns (row_index, result_dict).
    """
    t0 = time.time()
    label = name if (name is not None and str(name).strip()) else f"mol_{idx}"
    safe = _slugify_name(str(label))
    sdf_path = Path(out_dir) / f"{safe}.sdf"

    try:
        # Run optimization (only best MMFF conformer goes to xTB per your script)
        mol = write_optimized_sdf_from_smiles(
            smiles=str(smiles),
            sdf_path=sdf_path,
            name=safe,
            gfn_level=gfn_level,
            uhf=uhf,
            seed=seed,
            init_mode=init_mode,
            final_mode=final_mode,          # pass through
            only_2D=only_2D,          # pass through
            permute_atom_tokens=permute_atom_tokens,          # pass through
            permute_heavy_only=permute_heavy_only,          # pass through
            save_mode=save_mode,
            archive_format=archive_format,
            xtb_threads=xtb_threads,
            work_parent=work_parent,
            return_mol=True,
        )
        out: Dict[str, Any] = {    
            "status": "ok",
            "sdf_path": str(sdf_path),
            "name_used": safe,
            "final_mode": final_mode,       # NEW: record mode
            "mmff_min_conf_rdkit_index": mol.GetIntProp("mmff_min_conf_rdkit_index") if mol.HasProp("mmff_min_conf_rdkit_index") else None,
            "elapsed_sec": round(time.time() - t0, 3),
        }
        # ...
        return idx, out

    except Exception as e:
        tb = traceback.format_exc(limit=5)
        out = {
            "status": "fail",
            "sdf_path": None,
            "name_used": safe,
            "error": f"{e.__class__.__name__}: {e}",
            "traceback": tb,
            "elapsed_sec": round(time.time() - t0, 3),
        }
        return idx, out


# ---------- main batch runner ----------

def conformergen_batch(
    df: pd.DataFrame,
    out_dir: str,
    out_csv: str,
    smiles_col: str = "smiles",
    name_col: Optional[str] = "name",
    init_mode: Optional[str] = "esnuel",
    final_mode: str = "xtb",
    only_2D: Optional[bool] = False,
    permute_atom_tokens: Optional[bool] = False,
    permute_heavy_only: Optional[bool] = True,
    max_workers: int = 2,
    xtb_threads: int = 2,
    gfn_level: int = 1,
    uhf: int = 0,
    seed: int = 123,
    save_mode: str = "archive",
    archive_format: str = "tar.xz",
    work_parent: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run batch optimization and SDF writing for each row in DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least a 'smiles' column.
    out_dir : str
        Destination directory for SDF files.
    out_csv : str
        Path to write the resulting CSV with appended 'sdf_path' and status columns.
    smiles_col : str
        Column name for SMILES.
    name_col : Optional[str]
        Optional column name for molecule name. If absent or empty, an auto name is used.
    max_workers : int
        Number of molecules to run concurrently (processes).
    xtb_threads : int
        Threads per molecule (passed to xtb via the optimization function).
    gfn_level, uhf, seed, save_mode, archive_format, work_parent :
        Passed through to 'optimize_with_xtb_from_smiles'.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns: 'sdf_path', 'status', 'error', 'elapsed_sec', etc.
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Input DataFrame must have a '{smiles_col}' column.")

    print(f"[info] Concurrency: {max_workers} molecules in parallel; xtb threads per molecule: {xtb_threads}")
    print(f"[info] Final mode: {final_mode}")  # NEW

    out_dir_path = Path(out_dir)
    _ensure_dir(out_dir_path)

    # Create job list
    jobs = []
    for i, row in df.iterrows():
        smi = row[smiles_col]
        nm = row[name_col] if (name_col and name_col in df.columns) else None
        jobs.append((i, smi, nm))

    total = len(jobs)
    if total == 0:
        raise ValueError("No rows to process.")

    print(f"[info] Total molecules: {total}")
    print(f"[info] Output SDF dir: {out_dir_path.resolve()}")
    print(f"[info] Concurrency: {max_workers} molecules in parallel; xtb threads per molecule: {xtb_threads}")
    print(f"[info] Archive mode: {save_mode} ({archive_format})")

    # Submit to executor
    results: Dict[int, Dict[str, Any]] = {}
    started = 0
    finished = 0
    failed = 0
    t_batch0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut2meta = {}
        for (idx, smi, nm) in jobs:
            fut = ex.submit(
                _single_job,
                idx, smi, nm, str(out_dir_path), init_mode, final_mode, only_2D, permute_atom_tokens, permute_heavy_only, # pass final_mode
                gfn_level, uhf, seed, save_mode, archive_format, xtb_threads, work_parent
            )
            # ...
            fut2meta[fut] = (idx, nm)
            started += 1
            if started % 10 == 1 or started == total:
                print(f"[queue] submitted {started}/{total}")

        for fut in as_completed(fut2meta):
            idx, nm = fut2meta[fut]
            try:
                row_idx, out = fut.result()
                results[row_idx] = out
                finished += 1
                if out.get("status") == "ok":
                    print(f"[ok] idx={row_idx} name={out.get('name_used')} time={out.get('elapsed_sec')}s -> {out.get('sdf_path')}")
                else:
                    failed += 1
                    print(f"[fail] idx={row_idx} name={out.get('name_used')} time={out.get('elapsed_sec')}s err={out.get('error')}")
            except Exception as e:
                finished += 1
                failed += 1
                results[idx] = {"status": "fail", "error": f"ExecutorError: {e}", "sdf_path": None, "elapsed_sec": None}
                print(f"[fail] idx={idx} name={nm} ExecutorError: {e}")

            # simple aggregate progress line
            prog = f"{finished}/{total} done, {failed} failed"
            pct = 100.0 * finished / total
            print(f"[prog] {prog} ({pct:.1f}%)")

    # Merge results back into DataFrame
    df_out = df.copy()
    # Initialize columns
    add_cols = ["sdf_path", "status", "error", "elapsed_sec", "name_used", "final_mode", "mmff_min_conf_rdkit_index"]
    # ...
    for c in add_cols:
        if c not in df_out.columns:
            df_out[c] = None

    for row_idx, info in results.items():
        for k, v in info.items():
            if k in df_out.columns:
                df_out.at[row_idx, k] = v
            else:
                # create new column if not existing
                if k not in df_out.columns:
                    df_out[k] = None
                df_out.at[row_idx, k] = v

    # Save CSV
    out_csv_path = Path(out_csv)
    _ensure_dir(out_csv_path.parent)
    df_out.to_csv(out_csv_path, index=False, encoding="utf-8")

    t_total = round(time.time() - t_batch0, 3)
    print(f"[done] Wrote CSV: {out_csv_path.resolve()}  (elapsed {t_total}s)")
    return df_out

def has_bonds(mol):
    """Return True if molecule has at least one bond."""
    return mol.GetNumBonds() > 0

def convert_xyz_to_sdf(mol, total_charge=None):
    """
    Original script comes from https://github.com/jensengroup/ESNUEL/blob/main/src/esnuel/molecule_formats.py#L33
    """
    mol_copy = Chem.Mol(mol)
    if total_charge is None:
        charge = [a.GetFormalCharge() for a in mol.GetAtoms()]
        total_charge = sum(charge)
    rdDetermineBonds.DetermineBonds(mol_copy, useHueckel=True, charge=total_charge)
    # rdDetermineBonds.DetermineBondOrders(rdkit_mol, charge=chrg)
    # rdDetermineBonds.DetermineBonds(rdkit_mol, charge=chrg, covFactor=1.3, allowChargedFragments=True, useHueckel=False, embedChiral=False, useAtomMap=False)

    if len(Chem.MolToSmiles(mol).split('.')) != 1:
        # print('OBS! Trying to detemine bonds without Hueckel')
        mol_copy = Chem.Mol(mol)
        rdDetermineBonds.DetermineBonds(mol_copy, useHueckel=False, charge=total_charge)

    if len(Chem.MolToSmiles(mol).split('.')) != 1:
        # print('OBS! Trying to detemine bonds without Hueckel and covFactor=1.35')
        mol_copy = Chem.Mol(mol)
        rdDetermineBonds.DetermineBonds(mol_copy, useHueckel=False, covFactor=1.35, charge=total_charge)

    return mol_copy

def convert_xyz_to_smiles(mol, sanitize=True, removeHs=False, total_charge=None):
    try:
        nmol = convert_xyz_to_sdf(mol, total_charge)
        if sanitize:
            Chem.SanitizeMol(nmol)
        if removeHs:
            return (Chem.MolToSmiles(Chem.RemoveAllHs(nmol)), None,)
        return (Chem.MolToSmiles(nmol), None,)
    except Exception as e:
        return (None, e,)

def convert_xyz_to_smiles_from_file(infile, sanitize=True, removeHs=False, total_charge=None):
    suppl = Chem.SDMolSupplier(infile, removeHs=False, sanitize=False)
    mols: List[Chem.Mol] = [m for m in suppl if m is not None]
    return np.array([list(convert_xyz_to_smiles(m,sanitize,removeHs,total_charge)) for m in mols]).T.tolist() if len(mols) > 1 else convert_xyz_to_smiles(mols[0],sanitize,removeHs,total_charge)

def generate_far_conformer(
    mol: Chem.Mol,
    min_rmsd: float = 1.5,
    num_confs: int = 64,
    random_seed: int = 42,
    ) -> Tuple[Chem.Mol, float]:
    """
    Generate a new conformer that is far from the reference (already-optimized) conformer.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule that already contains at least one conformer. The first conformer
        is treated as the reference optimized geometry.
    min_rmsd : float
        Target minimum RMSD (in Angstrom) from the reference geometry. If no conformer
        exceeding this threshold is found, the farthest one is returned.
    num_confs : int
        Number of trial conformers to embed per attempt.
    random_seed : int
        Base random seed for reproducibility. Each attempt uses a different offset.

    Returns
    -------
    Tuple[Chem.Mol, float]
        A tuple of (mol_with_single_far_conformer, rmsd_from_reference).
        The returned molecule contains only the selected far conformer.

    Notes
    -----
    - This function does not modify the input molecule. Internally, a copy is used.
    - Trial conformers are MMFF-minimized before RMSD evaluation.
    - Hydrogens in the input molecule are kept as-is.
    """
    if mol is None:
        raise ValueError("Input mol is None.")
    if mol.GetNumConformers() < 1:
        raise ValueError("Input mol must contain at least one conformer as reference.")
    if mol.HasProp('mmff_min_energy_kcalmol') and mol.GetDoubleProp('mmff_min_energy_kcalmol')==float('inf'):
        raise NotImplementedError("Conformer generation was skipped because this mol has a mmff-errored conformation.")
    
    if not has_bonds(mol):
        mol = convert_xyz_to_sdf(mol)
        mol.SetProp('BondDetermine','RDKit_DetermineBonds')
    else:
        mol.SetProp('BondDetermine','Original')

    work = Chem.Mol(mol, confId=0)
    ref_conf_id = 0

    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = max((min_rmsd, 0.1))
    params.clearConfs = False
    params.useRandomCoords = True
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.enforceChirality = True

    params.randomSeed = int(random_seed)
    new_ids = AllChem.EmbedMultipleConfs(work, numConfs=num_confs, params=params)

    assert work.GetNumConformers()==len(new_ids)+1
    if not new_ids:
        raise RuntimeError("Failed to generate any valid conformer.")
    
    best_conf_id, best_rmsd = max(
        [(cid, AllChem.GetConformerRMS(work, ref_conf_id, cid)) for cid in new_ids], 
        key=lambda t: t[-1]
        )

    if best_rmsd < min_rmsd:
        raise RuntimeError(f"Failed to generate conformer having rmsd_from_referred_conf >= {min_rmsd}.")

    out = Chem.Mol(work)
    chosen_conf = work.GetConformer(best_conf_id)
    out.RemoveAllConformers()
    out.AddConformer(Chem.Conformer(chosen_conf), assignId=True)

    try:
        for k in mol.GetPropNames():
            out.SetProp(k, mol.GetProp(k))
    except Exception:
        pass
    try:
        name = mol.GetProp("_Name")
        out.SetProp("_Name", name)
    except Exception:
        pass

    return out, best_rmsd

def diverse_conf_from_sdf_file(
    infile: str,
    outfile: str,
    min_rmsd: float,
    num_confs: int,
    random_seed: int,
) -> Tuple[bool, Optional[float]]:
    """
    Process one SDF file while keeping hydrogens exactly as in the input:
    - Read all molecules with removeHs=False.
    - For each molecule that has at least one conformer, generate a far conformer.
    - Write only successful far-conformer molecules to the output SDF.
    - Return (success, rmsd_aggregate), where success=True if at least one molecule succeeded.
      rmsd_aggregate is the maximum RMSD among successful molecules, or None if none succeeded.
    """
    suppl = Chem.SDMolSupplier(infile, removeHs=False, sanitize=False)
    mols: List[Chem.Mol] = [m for m in suppl if m is not None]

    if len(mols) == 0:
        return (False, None)

    writer = Chem.SDWriter(outfile)
    # writer.SetKekulize(False)

    any_success = False
    max_rmsd: Optional[float] = None

    error_log = {}

    for i, m in enumerate(mols):
        try:
            if m.GetNumConformers() < 1:
                continue

            far_mol, rmsd = generate_far_conformer(
                m,
                min_rmsd=min_rmsd,
                num_confs=num_confs,
                random_seed=random_seed,
            )

            writer.write(far_mol)
            any_success = True
            if (max_rmsd is None) or (rmsd > max_rmsd):
                max_rmsd = rmsd

        except Exception as e:
            error_log[i] = e

    if not error_log:
        error_log = None
    elif len(mols) < 2:
        error_log = error_log[0]
        
    writer.close()

    if not any_success:
        try:
            if os.path.exists(outfile):
                os.remove(outfile)
        except Exception:
            pass
        return (False, None, error_log)

    return (True, max_rmsd, error_log)

### xTB conformation search (Uni-Mol -> xTB)

def _mol_has_3d(mol: Chem.Mol) -> bool:
    """Check if a molecule has a 3D conformer."""
    if mol is None or mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer(0)
    return conf.Is3D()

def _mol_to_xyz(mol: Chem.Mol, xyz_path: Path) -> None:
    """Write a single-conformer RDKit Mol to an XYZ file with current coordinates."""
    conf = mol.GetConformer(0)
    n = mol.GetNumAtoms()
    lines = [str(n), "generated by rdkit"]
    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        pos = conf.GetAtomPosition(i)
        lines.append(f"{sym} {pos.x:.10f} {pos.y:.10f} {pos.z:.10f}")
    xyz_path.write_text("\n".join(lines), encoding="utf-8")

def _read_xyz_coords(xyz_path: Path) -> List[Tuple[float, float, float]]:
    """Parse coordinates from an XYZ file. Assumes atom order unchanged."""
    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError("Invalid XYZ content.")
    # Skip the first two lines (atom count and comment)
    coords = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        x, y, z = map(float, parts[-3:])
        coords.append((x, y, z))
    return coords

def _apply_coords_to_mol(mol: Chem.Mol, coords: List[Tuple[float, float, float]]) -> None:
    """Overwrite the coordinates of the first conformer with given coordinate list."""
    if mol.GetNumAtoms() != len(coords):
        raise ValueError("Atom count mismatch between RDKit Mol and XYZ.")
    conf = mol.GetConformer(0)
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))

def _xtb_run(
    xyz_in: Path,
    method_flag: str,  # e.g., "ff" for GFN-FF, "1" for GFN1
    charge: int,
    workdir: Path,
    xtb_exe: str,
    xtb_threads: int,
    verbose: bool = True,
) -> Path:
    """
    Run xTB with the requested method (--gfn{method_flag}) and return path to xtbopt.xyz.

    Notes
    -----
    - Matches the requested CLI pattern:
      {XTBHOME}/bin/xtb --gfn{method} {start_structure_xyz} --opt --vfukui --chrg {chrg} --uhf 0 --alpb DMSO
    """
    cmd = [
        xtb_exe,
        f"--gfn{method_flag}",
        str(xyz_in),
        "--opt",
        "--vfukui",
        "--chrg",
        str(int(charge)),
        "--uhf",
        "0",
        "--alpb",
        "DMSO",
    ]
    env = os.environ.copy()
    # Constrain threads to avoid oversubscription in parallel runs
    env["OMP_NUM_THREADS"] = str(int(xtb_threads))
    env["MKL_NUM_THREADS"] = str(int(xtb_threads))

    if verbose:
        print(f"[xtb] cwd={workdir} cmd={' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    # Write logs for debugging
    (workdir / "xtb_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (workdir / "xtb_stderr.txt").write_text(proc.stderr, encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(f"xTB failed with return code {proc.returncode}")

    opt_xyz = workdir / "xtbopt.xyz"
    if not opt_xyz.exists():
        raise FileNotFoundError("xtbopt.xyz not found after xTB optimization.")
    return opt_xyz

def _optimize_one_mol_with_xtb(
    mol: Chem.Mol,
    workdir: Path,
    xtb_exe: str,
    xtb_threads: int,
    verbose: bool = True,
) -> Chem.Mol:
    """
    Optimize a single RDKit Mol by xTB in two stages: GFN-FF then GFN1.
    Coordinates of the first conformer are updated in-place.
    """
    if mol is None:
        raise ValueError("Mol is None.")
    if not _mol_has_3d(mol):
        raise ValueError("Mol has no 3D conformer. Provide 3D coordinates in SDF.")

    _ensure_dir(workdir)

    # Determine total formal charge for the molecule
    charge = Chem.GetFormalCharge(mol)

    # Stage 0: write input geometry
    xyz0 = workdir / "input.xyz"
    _mol_to_xyz(mol, xyz0)

    # Stage 1: GFN-FF optimization
    xyz_ff = _xtb_run(
        xyz_in=xyz0,
        method_flag="ff",
        charge=charge,
        workdir=workdir,
        xtb_exe=xtb_exe,
        xtb_threads=xtb_threads,
        verbose=verbose,
    )

    # Stage 2: GFN1 optimization starting from the previous optimum
    xyz_gfn1_input = workdir / "gfn1_start.xyz"
    shutil.copyfile(xyz_ff, xyz_gfn1_input)

    xyz_gfn1 = _xtb_run(
        xyz_in=xyz_gfn1_input,
        method_flag="1",
        charge=charge,
        workdir=workdir,
        xtb_exe=xtb_exe,
        xtb_threads=xtb_threads,
        verbose=verbose,
    )

    # Update RDKit conformer coordinates with final result
    coords = _read_xyz_coords(xyz_gfn1)
    _apply_coords_to_mol(mol, coords)

    return mol

def _process_single_sdf_file(
    in_sdf: str,
    out_dir: str,
    xtb_exe: str,
    xtb_threads: int,
    keep_work: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Worker that processes one SDF file:
    - Reads all molecules
    - Runs xTB GFN-FF => GFN1
    - Writes an SDF with the same filename into out_dir, preserving SD properties
    """
    t0 = time.time()
    in_path = Path(in_sdf)
    out_dir_path = Path(out_dir)
    _ensure_dir(out_dir_path)
    out_path = out_dir_path / in_path.name

    # Prepare a per-file working directory
    work_root = out_dir_path / f"_xtbwork_{in_path.stem}_{uuid.uuid4().hex[:8]}"
    _ensure_dir(work_root)

    # Read all molecules from the SDF
    suppl = Chem.SDMolSupplier(str(in_path), removeHs=False, sanitize=True)
    mols_original: List[Chem.Mol] = [m for m in suppl if m is not None]

    mols = []

    ignored = None
    for mol in mols_original:
        if mol.GetDoubleProp('mmff_min_energy_kcalmol')==float('inf'):
            ignored = mol.GetProp('_Name') if not ignored else ignored + ', ' + mol.GetProp('_Name')
            continue
        mols.append(mol)
    
    error_msg = "No valid molecules in SDF." if not ignored else f"Partially not implemented because conformers '{ignored}' were not MMFF-optimized."
    if len(mols) == 0:
        if not keep_work:
            shutil.rmtree(work_root, ignore_errors=True)
        if ignored is not None:
            error_msg = "Not implemented because all conformers were not MMFF-optimized."
        return {
            "in_sdf": str(in_path),
            "out_sdf": str(out_path),
            "n_mols": 0,
            "status": "fail",
            "error": error_msg,
            "elapsed_sec": round(time.time() - t0, 3),
        }
    if not ignored:
        error_msg = None

    # Optimize each molecule
    optimized: List[Chem.Mol] = []
    n_ok = 0
    n_fail = 0
    errors: List[str] = []

    for idx, mol in enumerate(mols):
        mol_work = work_root / f"mol_{idx:05d}"
        try:
            _ensure_dir(mol_work)
            _optimize_one_mol_with_xtb(
                mol=mol,
                workdir=mol_work,
                xtb_exe=xtb_exe,
                xtb_threads=xtb_threads,
                verbose=verbose,
            )
            n_ok += 1
            optimized.append(mol)
        except Exception as e:
            n_fail += 1
            errors.append(f"mol[{idx}]: {e}")
            # Keep original for output alignment; coordinates remain as input
            optimized.append(mol)

    # Write output SDF with preserved properties
    writer = Chem.SDWriter(str(out_path))
    # RDKit automatically carries over SD properties stored in mol.GetPropsAsDict
    for mol in optimized:
        writer.write(mol)
    writer.close()

    if not keep_work:
        shutil.rmtree(work_root, ignore_errors=True)

    status = "ok" if n_fail == 0 else ("partial" if n_ok > 0 else "fail")
    err_msg = "; ".join(errors) if errors else None
    if error_msg:
        err_msg = error_msg + '; ' + err_msg if err_msg else error_msg

    return {
        "in_sdf": str(in_path),
        "out_sdf": str(out_path),
        "n_mols": len(mols),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "status": status,
        "error": err_msg,
        "elapsed_sec": round(time.time() - t0, 3),
    }

# ----------------------------
# Public API
# ----------------------------

def xtb_optimize_sdf_dir(
    in_dir: str,
    out_dir: str,
    max_workers: int = 2,
    xtb_threads: int = 2,
    keep_work: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Enumerate all SDF files under 'in_dir', and for each SDF:
      - run xTB optimization with GFN-FF followed by GFN1,
      - write an output SDF with the same filename under 'out_dir',
      - preserve all SDF properties.

    xTB command line matches the requested pattern:
        {XTBHOME}/bin/xtb --gfn{method} {start_structure_xyz} --opt --vfukui --chrg {chrg} --uhf 0 --alpb DMSO

    Parameters
    ----------
    in_dir : str
        Directory containing input SDF files.
    out_dir : str
        Destination directory where optimized SDF files are written.
    max_workers : int
        Number of SDF files to process concurrently (process-based parallelism).
    xtb_threads : int
        Threads per xTB process. Applied via OMP_NUM_THREADS and MKL_NUM_THREADS.
    keep_work : bool
        If True, keep per-molecule working directories for inspection.
    verbose : bool
        If True, print progress messages.

    Returns
    -------
    pd.DataFrame
        One row per input SDF file with columns:
        ['in_sdf', 'out_sdf', 'n_mols', 'n_ok', 'n_fail', 'status', 'error', 'elapsed_sec'].
    """
    t_batch0 = time.time()
    in_dir_path = Path(in_dir)
    out_dir_path = Path(out_dir)
    _ensure_dir(out_dir_path)

    # Collect *.sdf files (case-insensitive)
    sdf_files = sorted([str(p) for p in in_dir_path.glob("**/*.sdf")])
    sdf_files += sorted([str(p) for p in in_dir_path.glob("**/*.SDF")])
    sdf_files = sorted(set(sdf_files))

    if len(sdf_files) == 0:
        raise ValueError(f"No SDF files found under: {in_dir_path.resolve()}")

    if verbose:
        print(f"[info] Input dir: {in_dir_path.resolve()}")
        print(f"[info] Output dir: {out_dir_path.resolve()}")
        print(f"[info] Files to process: {len(sdf_files)}")
        print(f"[info] Concurrency: {max_workers} files; xTB threads per file: {xtb_threads}")

    # Parallel over files
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut2file = {}
        for p in sdf_files:
            fut = ex.submit(
                _process_single_sdf_file,
                p,
                str(out_dir_path),
                xtb_exe,
                xtb_threads,
                keep_work,
                verbose,
            )
            fut2file[fut] = p
            if verbose and (len(fut2file) % 10 == 1 or len(fut2file) == len(sdf_files)):
                print(f"[queue] submitted {len(fut2file)}/{len(sdf_files)}")

        finished = 0
        failed = 0
        for fut in as_completed(fut2file):
            infile = fut2file[fut]
            try:
                info = fut.result()
                rows.append(info)
                finished += 1
                if info.get("status") == "ok":
                    print(f"[ok] {infile} -> {info.get('out_sdf')}  n={info.get('n_mols')}  time={info.get('elapsed_sec')}s")
                else:
                    failed += 1
                    print(f"[warn] {infile} status={info.get('status')} err={info.get('error')}")
            except Exception as e:
                finished += 1
                failed += 1
                rows.append({
                    "in_sdf": infile,
                    "out_sdf": str(Path(out_dir) / Path(infile).name),
                    "n_mols": None,
                    "n_ok": None,
                    "n_fail": None,
                    "status": "fail",
                    "error": f"ExecutorError: {e}",
                    "elapsed_sec": None,
                })
                print(f"[fail] {infile} ExecutorError: {e}")
            pct = 100.0 * finished / len(sdf_files)
            print(f"[prog] {finished}/{len(sdf_files)} done, {failed} failed ({pct:.1f}%)")

    df = pd.DataFrame(rows)
    total_t = round(time.time() - t_batch0, 3)
    if verbose:
        print(f"[done] Processed {len(sdf_files)} files in {total_t}s")
    return df

