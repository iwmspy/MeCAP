#!/usr/bin/env python3
"""
End-to-end scaffold split + scaffold analysis pipeline.

What this script does:
1) Load SMILES from a CSV(.gz) file (e.g., df_ChEMBL50K.csv.gz).
2) Create a Bemis–Murcko scaffold split:
   - First split: 85% trainval / 15% test (scaffold-grouped, weighted by molecule count).
   - Second split: trainval -> 80% train / 20% val (scaffold-grouped, weighted by molecule count).
3) Save split.csv with index=smiles and fold in {train, val, test}.
4) Save scaffold mapping files:
   - scaffold_map.csv: scaffold -> count + JSON list of SMILES
   - scaffold_pairs.csv: long format scaffold, smiles
5) Enrich split.csv with:
   - scaffold
   - scaffold_fold (majority fold for molecules sharing the scaffold)
   - scaffold_fold_nunique (should be 1 for a correct scaffold split)
   - scaffold_max_sim_to_train (max Tanimoto similarity to any TRAIN scaffold)
     computed on unique scaffolds using RDKit BulkTanimotoSimilarity, parallelized.

Notes:
- Scaffold chirality is removed by design (includeChirality=False).
- For acyclic molecules (empty scaffold), we fall back to canonical SMILES to avoid collapsing all acyclics into one group.
- For maximum speed, scaffold fingerprints are computed inside workers. If that becomes slow,
  you can precompute all scaffold FPs in the parent process and pass them differently.
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdMolDescriptors


# ----------------------------
# Configuration defaults
# ----------------------------
DEFAULT_SEED = 42
DEFAULT_TEST_FRAC = 0.15
DEFAULT_VAL_FRAC_IN_TRAINVAL = 0.20

DEFAULT_FP_RADIUS = 2
DEFAULT_FP_NBITS = 2048
DEFAULT_FP_USE_CHIRALITY = False  # scaffold chirality in FP; scaffold extraction is chirality-free by default


# ----------------------------
# Utility: canonicalize and scaffold extraction
# ----------------------------
def canon_smiles(smi: str) -> str:
    """Return canonical SMILES. Raise ValueError if invalid."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def scaffold_smiles_from_smiles(smi: str, include_chirality: bool = False) -> str:
    """
    Return Bemis–Murcko scaffold SMILES.
    If empty (acyclic), fall back to canonical SMILES to avoid a single empty-scaffold bucket.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=bool(include_chirality))
    if scaff is None or scaff == "":
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return scaff


# ----------------------------
# Weighted scaffold splitting (by molecule count)
# ----------------------------
def _stable_group_order(scaffold_to_smis: Dict[str, List[str]], seed: int) -> List[Tuple[str, List[str], int]]:
    """
    Sort scaffold groups by decreasing size with a deterministic random tie-break.
    Returns a list of (scaffold, smis, size).
    """
    rng = np.random.default_rng(seed)
    items = []
    for scaff, smis in scaffold_to_smis.items():
        items.append((scaff, smis, len(smis), rng.random()))
    items.sort(key=lambda x: (-x[2], x[3]))
    return [(scaff, smis, size) for scaff, smis, size, _ in items]


def _two_way_scaffold_split(scaffold_to_smis: Dict[str, List[str]], frac_b: float, seed: int) -> Tuple[List[str], List[str]]:
    """
    Split scaffold groups into A and B with approximate molecule-count ratio:
    B gets frac_b of total molecules, A gets the rest.
    Returns (smis_A, smis_B).
    """
    groups = _stable_group_order(scaffold_to_smis, seed=seed)
    total = sum(size for _, _, size in groups)
    target_b = int(round(total * frac_b))
    target_a = total - target_b

    a_smis: List[str] = []
    b_smis: List[str] = []
    a_cnt = 0
    b_cnt = 0

    for scaff, smis, size in groups:
        if b_cnt + size > target_b and a_cnt + size <= target_a:
            a_smis.extend(smis)
            a_cnt += size
            continue
        if a_cnt + size > target_a and b_cnt + size <= target_b:
            b_smis.extend(smis)
            b_cnt += size
            continue

        fill_a = a_cnt / max(target_a, 1)
        fill_b = b_cnt / max(target_b, 1)
        if fill_b < fill_a:
            b_smis.extend(smis)
            b_cnt += size
        else:
            a_smis.extend(smis)
            a_cnt += size

    return a_smis, b_smis


# ----------------------------
# Scaffold fold assignment for enriched split.csv
# ----------------------------
def assign_scaffold_fold(df: pd.DataFrame, scaff_col: str, fold_col: str) -> pd.DataFrame:
    """
    Determine scaffold_fold from molecule folds.
    If a scaffold spans multiple folds (should not happen for a correct scaffold split),
    assign the majority fold and record the number of unique folds.
    """
    mode_fold = (
        df.groupby(scaff_col)[fold_col]
        .agg(lambda x: x.value_counts().index[0])
        .rename("scaffold_fold")
    )
    nunique = df.groupby(scaff_col)[fold_col].nunique().rename("scaffold_fold_nunique")
    df_scaff = pd.concat([mode_fold, nunique], axis=1)

    n_bad = int((df_scaff["scaffold_fold_nunique"] > 1).sum())
    if n_bad > 0:
        print(f"[WARN] Found {n_bad} scaffolds spanning multiple folds. Using majority fold.")
    return df_scaff


# ----------------------------
# Parallel max-sim computation (BulkTanimotoSimilarity has no thread parameter)
# ----------------------------
_G_TRAIN_FPS = None
_G_FP_RADIUS = None
_G_FP_NBITS = None
_G_FP_USE_CHIRALITY = None


def _init_worker(train_fps, fp_radius: int, fp_nbits: int, use_chirality: bool) -> None:
    """Initialize globals in each worker process."""
    global _G_TRAIN_FPS, _G_FP_RADIUS, _G_FP_NBITS, _G_FP_USE_CHIRALITY
    _G_TRAIN_FPS = train_fps
    _G_FP_RADIUS = int(fp_radius)
    _G_FP_NBITS = int(fp_nbits)
    _G_FP_USE_CHIRALITY = bool(use_chirality)


def _fp_from_scaffold_smiles(scaff_smi: str):
    """Compute Morgan FP bitvector for a scaffold SMILES."""
    mol = Chem.MolFromSmiles(scaff_smi)
    if mol is None:
        return None
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol,
        radius=_G_FP_RADIUS,
        nBits=_G_FP_NBITS,
        useChirality=_G_FP_USE_CHIRALITY,
    )


def _max_sim_one(scaff_smi: str) -> Tuple[str, float]:
    """Return (scaffold_smiles, max_tanimoto_to_train)."""
    fp = _fp_from_scaffold_smiles(scaff_smi)
    if fp is None or _G_TRAIN_FPS is None or len(_G_TRAIN_FPS) == 0:
        return scaff_smi, 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp, _G_TRAIN_FPS)
    return scaff_smi, float(max(sims)) if sims else 0.0


def compute_max_sim_parallel(
    unique_scaffolds: List[str],
    train_fps: List,
    fp_radius: int,
    fp_nbits: int,
    use_chirality: bool,
    n_workers: Optional[int] = None,
    chunksize: int = 2000,
) -> Dict[str, float]:
    """
    Compute max similarity to train scaffolds for each scaffold SMILES, in parallel.

    Returns
    -------
    dict: scaffold_smiles -> max_sim_to_train
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 1)

    max_sim: Dict[str, float] = {}
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(train_fps, fp_radius, fp_nbits, use_chirality),
    ) as ex:
        it = ex.map(_max_sim_one, unique_scaffolds, chunksize=chunksize)
        for scaff, m in tqdm(it, total=len(unique_scaffolds), desc="MaxSim to train scaffolds", unit="scaff"):
            max_sim[scaff] = m
    return max_sim


# ----------------------------
# Main pipeline
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV(.gz) containing a SMILES column.")
    parser.add_argument("--smiles_col", type=str, default="smiles", help="SMILES column name.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test_frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument("--val_frac_in_trainval", type=float, default=DEFAULT_VAL_FRAC_IN_TRAINVAL)

    parser.add_argument("--out_split_csv", type=str, default="split.csv", help="Output split CSV (overwritten).")
    parser.add_argument("--out_scaffold_map_csv", type=str, default="scaffold_map.csv")
    parser.add_argument("--out_scaffold_pairs_csv", type=str, default="scaffold_pairs.csv")

    parser.add_argument("--scaffold_include_chirality", action="store_true",
                        help="Keep chirality when extracting scaffold (default: remove chirality).")
    parser.add_argument("--fp_radius", type=int, default=DEFAULT_FP_RADIUS)
    parser.add_argument("--fp_nbits", type=int, default=DEFAULT_FP_NBITS)
    parser.add_argument("--fp_use_chirality", action="store_true",
                        help="Use chirality in scaffold fingerprints (default: False).")

    parser.add_argument("--n_workers", type=int, default=None, help="Number of worker processes for max-sim.")
    parser.add_argument("--chunksize", type=int, default=2000, help="Chunksize for executor.map.")
    args = parser.parse_args()

    # ----------------------------
    # 1) Load SMILES and canonicalize
    # ----------------------------
    df_origin = pd.read_csv(args.input_csv)
    if args.smiles_col not in df_origin.columns:
        raise ValueError(f"Column '{args.smiles_col}' not found in input.")

    smi_raw = df_origin[args.smiles_col].dropna().astype(str).tolist()

    smi_can: List[str] = []
    invalid = 0
    for smi in tqdm(smi_raw, desc="Canonicalizing SMILES", unit="smi"):
        try:
            smi_can.append(canon_smiles(smi))
        except Exception:
            invalid += 1

    smi_unique = sorted(set(smi_can))
    if len(smi_unique) == 0:
        raise RuntimeError("No valid SMILES found after canonicalization.")
    if invalid > 0:
        print(f"[WARN] Invalid SMILES skipped: {invalid}")

    # ----------------------------
    # 2) Compute scaffold groups (scaffold -> list of SMILES)
    # ----------------------------
    scaffold_to_smis: Dict[str, List[str]] = {}
    scaff_fail = 0
    for smi in tqdm(smi_unique, desc="Computing scaffolds", unit="smi"):
        try:
            scaff = scaffold_smiles_from_smiles(smi, include_chirality=args.scaffold_include_chirality)
            scaffold_to_smis.setdefault(scaff, []).append(smi)
        except Exception:
            scaff_fail += 1

    if scaff_fail > 0:
        print(f"[WARN] Scaffold computation failed for {scaff_fail} SMILES. Those SMILES were skipped.")

    # Ensure uniqueness in groups
    for scaff in list(scaffold_to_smis.keys()):
        scaffold_to_smis[scaff] = sorted(set(scaffold_to_smis[scaff]))

    all_unique_smis = sorted({s for smis in scaffold_to_smis.values() for s in smis})
    if len(all_unique_smis) == 0:
        raise RuntimeError("No SMILES left after scaffold grouping.")

    # ----------------------------
    # 3) Scaffold split: 85/15 then 80/20 within 85
    # ----------------------------
    trainval_smis, test_smis = _two_way_scaffold_split(
        scaffold_to_smis=scaffold_to_smis,
        frac_b=float(args.test_frac),
        seed=int(args.seed),
    )

    trainval_set = set(trainval_smis)
    trainval_scaffold_to_smis: Dict[str, List[str]] = {}
    for scaff, smis in scaffold_to_smis.items():
        kept = [s for s in smis if s in trainval_set]
        if kept:
            trainval_scaffold_to_smis[scaff] = kept

    train_smis, val_smis = _two_way_scaffold_split(
        scaffold_to_smis=trainval_scaffold_to_smis,
        frac_b=float(args.val_frac_in_trainval),
        seed=int(args.seed) + 1,
    )

    # Sanity: no overlap
    s_train = set(train_smis)
    s_val = set(val_smis)
    s_test = set(test_smis)
    if (s_train & s_val) or (s_train & s_test) or (s_val & s_test):
        raise RuntimeError("Split overlap detected. Check split logic.")

    # ----------------------------
    # 4) Save split.csv (smiles index, fold column)
    # ----------------------------
    smi_to_fold: Dict[str, str] = {}
    for s in train_smis:
        smi_to_fold[s] = "train"
    for s in val_smis:
        smi_to_fold[s] = "val"
    for s in test_smis:
        smi_to_fold[s] = "test"

    sf_series = pd.Series(smi_to_fold, name="fold")
    sf_series.index.name = "smiles"
    df_split = sf_series.to_frame()
    df_split.to_csv(args.out_split_csv)

    n_total = len(all_unique_smis)
    print(f"[INFO] Total unique SMILES: {n_total}")
    print(f"[INFO] Train: {len(train_smis)} ({len(train_smis)/n_total:.3f})")
    print(f"[INFO] Val  : {len(val_smis)} ({len(val_smis)/n_total:.3f})")
    print(f"[INFO] Test : {len(test_smis)} ({len(test_smis)/n_total:.3f})")
    print(f"[INFO] Saved split: {args.out_split_csv}")

    # ----------------------------
    # 5) Save scaffold mapping files
    # ----------------------------
    scaffold_rows = []
    pair_rows = []
    for scaff, smis in scaffold_to_smis.items():
        smis_sorted = sorted(set(smis))
        scaffold_rows.append(
            {
                "scaffold": scaff,
                "n_smiles": len(smis_sorted),
                "smiles_list_json": json.dumps(smis_sorted),
            }
        )
        for s in smis_sorted:
            pair_rows.append({"scaffold": scaff, "smiles": s})

    df_scaffold_map = pd.DataFrame(scaffold_rows).sort_values(
        by=["n_smiles", "scaffold"], ascending=[False, True]
    )
    df_scaffold_pairs = pd.DataFrame(pair_rows).sort_values(by=["scaffold", "smiles"])

    df_scaffold_map.to_csv(args.out_scaffold_map_csv, index=False)
    df_scaffold_pairs.to_csv(args.out_scaffold_pairs_csv, index=False)
    print(f"[INFO] Saved: {args.out_scaffold_map_csv}")
    print(f"[INFO] Saved: {args.out_scaffold_pairs_csv}")

    # ----------------------------
    # 6) Enrich split.csv with scaffold columns and max similarity to train scaffolds
    # ----------------------------
    # Build a dataframe with smiles index and fold
    df = df_split.copy()

    # Add scaffold for each SMILES
    scaff_list = []
    for smi in tqdm(df.index.tolist(), desc="Attaching scaffold to split.csv", unit="smi"):
        scaff_list.append(scaffold_smiles_from_smiles(smi, include_chirality=args.scaffold_include_chirality))
    df["scaffold"] = scaff_list

    # Determine scaffold_fold and consistency
    df_scaff = assign_scaffold_fold(df, scaff_col="scaffold", fold_col="fold")
    df = df.join(df_scaff["scaffold_fold"], on="scaffold")
    df = df.join(df_scaff["scaffold_fold_nunique"], on="scaffold")

    unique_scaffolds = df_scaff.index.tolist()
    train_scaffolds = df_scaff.index[df_scaff["scaffold_fold"] == "train"].tolist()
    if len(train_scaffolds) == 0:
        raise RuntimeError("No train scaffolds found. Check fold labels.")

    # Precompute train scaffold FPs in parent process (cheap and avoids recompute per worker init)
    train_fps = []
    for scaff in tqdm(train_scaffolds, desc="Fingerprinting train scaffolds", unit="scaff"):
        mol = Chem.MolFromSmiles(scaff)
        if mol is None:
            continue
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            radius=int(args.fp_radius),
            nBits=int(args.fp_nbits),
            useChirality=bool(args.fp_use_chirality),
        )
        train_fps.append(fp)

    if len(train_fps) == 0:
        raise RuntimeError("All train scaffold fingerprints failed. Check scaffold SMILES validity.")

    # Compute max similarity for each unique scaffold (parallel)
    max_sim = compute_max_sim_parallel(
        unique_scaffolds=unique_scaffolds,
        train_fps=train_fps,
        fp_radius=int(args.fp_radius),
        fp_nbits=int(args.fp_nbits),
        use_chirality=bool(args.fp_use_chirality),
        n_workers=args.n_workers,
        chunksize=int(args.chunksize),
    )

    df["scaffold_max_sim_to_train"] = df["scaffold"].map(max_sim).astype(np.float64)

    # Save enriched split.csv (overwrite)
    df.to_csv(args.out_split_csv)
    print(f"[INFO] Updated split with scaffold columns: {args.out_split_csv}")
    print(f"[INFO] Unique scaffolds: {len(unique_scaffolds)}")
    print(f"[INFO] Train scaffolds : {len(train_scaffolds)}")


if __name__ == "__main__":
    main()