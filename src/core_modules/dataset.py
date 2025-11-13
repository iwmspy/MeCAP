# -*- coding: utf-8 -*-
# Base + V1/V2 DataHub consolidation for Uni-Mol pipelines.

from __future__ import annotations

import os
from pathlib import Path
from hashlib import sha1
from typing import Any, Dict as TDict, List, Optional, Tuple
from multiprocessing import Pool
import traceback

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem

from unimol_tools.config import MODEL_CONFIG
from unimol_tools.data import Dictionary
from unimol_tools.data.datahub import DataHub
from unimol_tools.weights import WEIGHT_DIR, weight_download

# V1/low-level converter
from unimol_tools.data.conformer import coords2unimol  # type: ignore
# V2 feature builder & converter
from unimol_tools.data.conformer import UniMolV2Feature, mol2unimolv2  # type: ignore

BASE_DIR = Path(__file__).parents[2]

# -------------------------
# Helpers (shared)
# -------------------------

# Global dictionary cache for V1 workers
_GLOBAL_V1_DICT = None

def _v1_worker_init(dict_path: str):
    """Pool initializer: load Dictionary once per worker."""
    from unimol_tools.data import Dictionary as _D
    global _GLOBAL_V1_DICT
    _GLOBAL_V1_DICT = _D.load(dict_path)

def _worker_path_to_feature_v1(args) -> Optional[TDict[str, Any]]:
    """
    Worker for V1: re-open SDF and run coords2unimol.
    Uses a process-global dictionary loaded by _v1_worker_init.
    """
    path, max_atoms, remove_hs = args
    try:
        mol = _load_first_mol_from_sdf(path, remove_hs=False)
        if mol is None or mol.GetNumConformers() == 0:
            return None
        conf = mol.GetConformer()
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
            dtype=np.float32,
        )
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        # dictionary is provided by initializer
        feat = coords2unimol(
            atoms, coords, dictionary=_GLOBAL_V1_DICT,
            max_atoms=max_atoms, remove_hs=remove_hs
        )
        return feat
    except Exception:
        return None

def _clone_feat_dict(feat: TDict[str, Any]) -> TDict[str, Any]:
    """Shallow clone feature dict with Tensor copies."""
    out = {}
    for k, v in feat.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        elif isinstance(v, (list, tuple)):
            out[k] = type(v)(v)
        elif isinstance(v, dict):
            out[k] = v.copy()
        else:
            out[k] = v
    return out

def _file_cache_key(path: str) -> str:
    """Cache key based on absolute path and mtime."""
    try:
        st = os.stat(path)
        raw = f"{os.path.abspath(path)}|{st.st_mtime_ns}".encode("utf-8")
    except Exception:
        raw = os.path.abspath(path).encode("utf-8")
    return sha1(raw).hexdigest()

def _resolve_sdf_path(name_or_path: str, base_dir: Optional[str], default_ext: Optional[str]) -> str:
    """Resolve a CSV cell (filename or path) to a file path with optional default extension."""
    p = Path(name_or_path)
    if p.is_absolute():
        cand = p
    else:
        cand = (Path(base_dir) / p) if base_dir else p
    if default_ext and cand.suffix == "":
        ext = default_ext if default_ext.startswith(".") else f".{default_ext}"
        cand = cand.with_suffix(ext)
    return str(cand)

def _load_first_mol_from_sdf(path: str, remove_hs: bool=False) -> Optional[Chem.Mol]:
    """Load the first conformer from an SDF file (keep bonds/charges; sanitize disabled)."""
    try:
        supp = Chem.SDMolSupplier(path, removeHs=remove_hs, sanitize=False)
        if supp is None or len(supp) == 0:
            return None
        mol = supp[0]
        if mol is None or mol.GetNumConformers() == 0:
            return None
        return mol
    except Exception:
        return None

def _resolve_model_dict_key(model_name: str, remove_hs: bool) -> str:
    """Resolve a robust MODEL_CONFIG dict key across versions."""
    candidates = ["molecule_no_h" if remove_hs else "molecule_all_h"]
    for key in candidates:
        if key in MODEL_CONFIG.get("dict", {}):
            return key
    return "molecule_no_h" if remove_hs else "molecule_all_h"

def _orig2heavy_map_from_mol(mol: Chem.Mol) -> np.ndarray:
    """Return array of length n_atoms(original), value = heavy_index or -1 if H."""
    mol_copy = Chem.Mol(mol)
    for atom in mol_copy.GetAtoms():
        atom.SetIntProp('original_order', atom.GetIdx())
    mol_copy_rm_h = AllChem.RemoveAllHs(mol_copy)
    mapping = np.full(mol_copy.GetNumAtoms(), -1, dtype=np.int32)
    for a in mol_copy_rm_h.GetAtoms():
        mapping[a.GetIntProp('original_order')] = a.GetIdx()
    return mapping

def _worker_path_to_feature(args) -> Optional[TDict[str, Any]]:
    """
    Worker for V2: re-open SDF in worker and run mol2unimolv2 (avoid pickling RDKit mols).
    Forced remove_hs=True to match V2 convention.
    """
    path, max_atoms = args
    mol = _load_first_mol_from_sdf(path, remove_hs=False)
    if mol is None:
        return None
    try:
        feat = mol2unimolv2(mol, max_atoms, remove_hs=True)
        return feat
    except Exception:
        return None


# ============================================================
# Base class: common logic shared by V1/V2 hubs
# ============================================================

class _BaseSingleAtomDataHub(DataHub):
    """
    Common scaffolding for V1/V2 Single-Atom DataHubs:
      - param handling, CSV loading, target extraction
      - caching helpers
      - path collection
      - post-build alignment (valid_rows slicing & reset)
      - split override (disable default k-fold)
      - attach_atom_index is delegated to subclass (due to V2 H-drop mapping)
      - building features is delegated to subclass (V1: coords2unimol / V2: mol2unimolv2)
    """
    def __init__(self, data=None, is_train: bool = True, save_path: Optional[str] = None, **params):
        # Store params needed across methods
        self.params = params
        self.conf_cache = bool(params.get("conf_cache", True))
        self.structure_source = params.get("structure_source", "sdf")
        self.sdf_mode = params.get("sdf_mode", "per_row")
        self.sdf_dir = params.get("sdf_dir", None)
        self.sdf_name_col = params.get("sdf_name_col", "SDF_NAME")
        self.sdf_ext = params.get("sdf_ext", None)
        self.sdf_path_col = params.get("sdf_path_col", None)
        self.sdf_file = params.get("sdf_file", None)
        self.sdf_id_prop = params.get("sdf_id_prop", "_Name")

        self.conf_cache_dir = Path(save_path or BASE_DIR / "data" / ".cache") / "conf_cache_unimol"
        if self.conf_cache:
            self.conf_cache_dir.mkdir(parents=True, exist_ok=True)

        # Keep original __init__ body, but add this first for _init_data
        self._input_data_source = data

        # Parent constructor (will call our overrides of _init_data/_init_split)
        super().__init__(data=data, is_train=is_train, save_path=save_path, **params)

        # Subclass-specific atom_index projection/validation
        self._attach_atom_index(**params)

        if self.data.get("unimol_input", None) is None:
            raise ValueError("unimol_input is missing.")
        if self.data.get("atom_index", None) is None:
            raise ValueError("atom_index is missing.")

        self.is_scale = False
        self.mean_ = None
        self.std_  = None

    # ---------- overridable hooks ----------

    def _ensure_dictionary(self, **params):
        """
        V1 requires a dictionary for coords2unimol; V2 does not.
        Subclasses can override. Default: no-op.
        """
        return

    def _build_unimol_input_from_sdf_paths(self, df, remove_hs: bool, max_atoms: int):
        """Subclass must implement."""
        raise NotImplementedError

    def _build_unimol_input_from_sdf_single(self, df, sdf_file: str, id_col: str,
                                            sdf_id_prop: str, remove_hs: bool, max_atoms: int):
        """Subclass must implement."""
        raise NotImplementedError

    def _attach_atom_index(self, **params):
        """Subclass must implement (V1: direct / V2: orig->heavy map)."""
        raise NotImplementedError

    # ---------- shared implementations ----------

    def _init_data(self, **params):
        """
        Common data init:
          - read CSV / DataFrame
          - pick targets
          - ensure dictionary if subclass needs (V1)
          - build features via subclass builders
        """
        src = getattr(self, "_input_data_source", None)
        if src is None:
            raise ValueError("No data source provided.")
        if isinstance(src, str):
            if not os.path.exists(src):
                raise FileNotFoundError(f"CSV not found: {src}")
            df = pd.read_csv(src)
        elif isinstance(src, pd.DataFrame):
            df = src.copy()
        else:
            raise ValueError("Unsupported data type for 'data'; expected CSV path or DataFrame.")

        self.data = {}
        self.data["raw_data"] = df.reset_index(drop=True)

        # target(s) optional
        target_cols = params.get("target_cols", None)
        if target_cols:
            cols = [c.strip() for c in str(target_cols).split(",") if c.strip()]
            for c in cols:
                if c not in df.columns:
                    raise ValueError(f"Target column '{c}' not in CSV.")
            y = df[cols[0]].to_numpy()
            self.data["target"] = [float(v) if pd.notna(v) else np.nan for v in y]
        else:
            self.data["target"] = None

        # Dictionary for V1 (V2 overrides to no-op)
        self._ensure_dictionary(**params)

        # build features
        self._build_unimol_input(self.data["raw_data"])

        self.data["num_data"] = len(self.data.get("unimol_input", []))

    def _init_split(self, **params):
        """Bypass parent's k-fold splitter (requires smiles). We'll split via CSV externally."""
        if not hasattr(self, "data") or self.data is None:
            raise RuntimeError("self.data is not initialized before _init_split")
        self.data.setdefault("smiles", [])
        self.data["split_nfolds"] = 1
        self.data["fold"] = int(params.get("fold", 1))
        self.splitter = None

    def _cache_load(self, key: str) -> Optional[TDict[str, Any]]:
        if not self.conf_cache:
            return None
        p = self.conf_cache_dir / f"{key}.pt"
        if p.exists():
            try:
                return torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                return None
        return None

    def _cache_save(self, key: str, feat: TDict[str, Any]) -> None:
        if not self.conf_cache:
            return
        p = self.conf_cache_dir / f"{key}.pt"
        try:
            torch.save(feat, p)
        except Exception:
            pass

    def _collect_paths_from_df(self, df) -> List[str]:
        """Resolve path list using either sdf_path_col or sdf_name_col + sdf_dir/sdf_ext."""
        paths: List[str] = []
        if self.sdf_path_col and self.sdf_path_col in df.columns:
            raws = df[self.sdf_path_col].astype(str).tolist()
            for p in raws:
                paths.append(_resolve_sdf_path(p, self.sdf_dir, None))
        elif self.sdf_name_col and self.sdf_name_col in df.columns:
            names = df[self.sdf_name_col].astype(str).tolist()
            for n in names:
                paths.append(_resolve_sdf_path(n, self.sdf_dir, self.sdf_ext))
        else:
            raise ValueError("Neither sdf_path_col nor sdf_name_col found in CSV.")
        return paths

    def _build_unimol_input(self, df):
        """Dispatch to subclass per mode; then align raw_data/target by valid_rows and reset indices."""
        remove_hs = bool(self.params.get("remove_hs", True))
        max_atoms = int(self.params.get("max_atoms", 512))

        if self.structure_source != "sdf":
            raise ValueError("Currently supports structure_source='sdf' only.")

        if self.sdf_mode == "per_row":
            self._build_unimol_input_from_sdf_paths(df=df, remove_hs=remove_hs, max_atoms=max_atoms)
        elif self.sdf_mode == "single_file":
            if not self.sdf_file:
                raise ValueError("sdf_mode='single_file' requires 'sdf_file'.")
            id_col = self.params.get("id_col", None)
            if not id_col:
                raise ValueError("sdf_mode='single_file' requires 'id_col' in CSV to match SDF.")
            self._build_unimol_input_from_sdf_single(
                df=df, sdf_file=self.sdf_file, id_col=id_col,
                sdf_id_prop=self.sdf_id_prop, remove_hs=remove_hs, max_atoms=max_atoms
            )
        else:
            raise ValueError("Unknown sdf_mode.")

        # ---- Align by valid_rows exactly once ----
        vr = self.data.get("valid_rows", [])
        self.data["orig_rows_map"] = list(vr) if isinstance(vr, list) else list(vr)

        if vr:
            if "row2uid" in self.data:
                self.data["row2uid"] = [self.data["row2uid"][i] for i in vr]
            if "raw_data" in self.data:
                self.data["raw_data"] = self.data["raw_data"].iloc[vr].reset_index(drop=True)
            if "target" in self.data and self.data["target"] is not None:
                t = self.data["target"]
                if isinstance(t, np.ndarray):
                    self.data["target"] = [t[i] for i in vr]
                elif isinstance(t, list):
                    self.data["target"] = [t[i] for i in vr]

        # Reset valid_rows to contiguous range
        n = len(self.data.get("unimol_input", []))
        self.data["valid_rows"] = list(range(n))
    
    def init_scaler(self, mean: float, std: float):
        self.mean_ = mean
        self.std_  = std

    def do_scaling(self):
        if self.mean_ is None or self.std_ is None:
            raise NotImplementedError
        self.is_scale = True
        self.data["original_target"] = [k for k in self.data["target"]]
        self.data["target"] = [(k - self.mean_) / self.std_ for k in self.data["target"]]
        

# ============================================================
# V1 DataHub (coords2unimol; needs dictionary; H removal optional)
# ============================================================

class SingleAtomDataHubV1(_BaseSingleAtomDataHub):
    """Uni-Mol v1 features from SDF coordinates (atoms+coords -> coords2unimol)."""

    def __init__(self, data=None, is_train: bool = True, save_path: Optional[str] = None, **params):
        # enable parallel feature building like V2
        self.feature_workers = int(params.get("feature_workers", 0))
        super().__init__(data=data, is_train=is_train, save_path=save_path, **params)

    def _ensure_dictionary(self, **params):
        remove_hs = bool(self.params.get("remove_hs", True))
        dict_key = _resolve_model_dict_key(self.params.get("model_name", "unimolv1"), remove_hs)
        dict_path = os.path.join(WEIGHT_DIR, MODEL_CONFIG["dict"][dict_key])
        if not os.path.exists(dict_path):
            weight_download(MODEL_CONFIG["dict"][dict_key], WEIGHT_DIR)
        # keep path for worker initializer
        self._dict_path = dict_path
        # load dictionary if not already present (parent may have done this)
        try:
            _ = self.dictionary
        except AttributeError:
            self.dictionary = Dictionary.load(dict_path)

    def _build_unimol_input_from_sdf_paths(self, df, remove_hs: bool, max_atoms: int):
        paths = self._collect_paths_from_df(df)

        # De-duplicate
        path_to_uid, uniq_paths, uid_of_row = {}, [], []
        for p in paths:
            if p not in path_to_uid:
                path_to_uid[p] = len(uniq_paths)
                uniq_paths.append(p)
            uid_of_row.append(path_to_uid[p])

        # Reuse cache when possible
        uniq_feats: List[Optional[TDict[str, Any]]] = [None] * len(uniq_paths)
        miss_uids: List[int] = []
        for uid, path in enumerate(uniq_paths):
            key = _file_cache_key(path)
            cached = self._cache_load(key)
            if cached is not None:
                uniq_feats[uid] = cached
            else:
                miss_uids.append(uid)

        # Build missing features (sequential or parallel)
        if len(miss_uids) > 0:
            build_paths = [uniq_paths[uid] for uid in miss_uids]

            if self.feature_workers <= 0:
                # Sequential path (original behavior)
                for loc, path in tqdm(enumerate(build_paths), total=len(build_paths), desc="V1: coords->unimol"):
                    uid = miss_uids[loc]
                    feat = None
                    try:
                        supp = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
                        mol = supp[0] if supp is not None and len(supp) > 0 else None
                        if mol is not None and mol.GetNumConformers() > 0:
                            conf = mol.GetConformer()
                            coords = np.array(
                                [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
                                dtype=np.float32,
                            )
                            atoms = [a.GetSymbol() for a in mol.GetAtoms()]
                            feat = coords2unimol(
                                atoms, coords, dictionary=self.dictionary,
                                max_atoms=max_atoms, remove_hs=remove_hs
                            )
                    except Exception:
                        print(f"\n---\nV1 feature build failed for {path}:\n{traceback.format_exc()}\n---\n")
                        feat = None
                    uniq_feats[uid] = feat
            else:
                # Parallel path (like V2): initialize Dictionary in each worker once
                args = [(p, max_atoms, remove_hs) for p in build_paths]
                with Pool(
                    processes=self.feature_workers,
                    initializer=_v1_worker_init,
                    initargs=(getattr(self, "_dict_path", None) or self._dict_path,)
                ) as pool:
                    out_list = list(
                        tqdm(pool.imap(_worker_path_to_feature_v1, args),
                             total=len(args), desc="V1 parallel coords2unimol")
                    )
                for uid, feat in zip(miss_uids, out_list):
                    uniq_feats[uid] = feat

            # Cache newly built
            for uid in miss_uids:
                feat = uniq_feats[uid]
                if feat is not None:
                    self._cache_save(_file_cache_key(uniq_paths[uid]), feat)

        # Expand to rows
        unimol_input, valid_rows, failed_rows = [], [], []
        for i, uid in enumerate(uid_of_row):
            feat = uniq_feats[uid]
            if feat is None:
                failed_rows.append(i)
            else:
                unimol_input.append(_clone_feat_dict(feat))
                valid_rows.append(i)

        self.data["uniq_unimol_input"] = uniq_feats
        self.data["row2uid"] = uid_of_row
        self.data["unimol_input"] = unimol_input
        self.data["valid_rows"] = valid_rows
        self.data["failed_3d_indices"] = failed_rows
        self.data["uniq_paths"] = uniq_paths

    def _build_unimol_input_from_sdf_single(self, df, sdf_file: str, id_col: str,
                                            sdf_id_prop: str, remove_hs: bool, max_atoms: int):
        file_key = _file_cache_key(sdf_file)
        cache_map_path = self.conf_cache_dir / f"{file_key}_map.pt"
        if self.conf_cache and cache_map_path.exists():
            id2feat = torch.load(cache_map_path, map_location="cpu", weights_only=False)
        else:
            id2feat = {}
            try:
                supp = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True)
                for mol in supp:
                    if mol is None or mol.GetNumConformers() == 0:
                        continue
                    key = mol.GetProp(sdf_id_prop) if mol.HasProp(sdf_id_prop) else \
                          (mol.GetProp("_Name") if mol.HasProp("_Name") else None)
                    if key is None:
                        continue
                    conf = mol.GetConformer()
                    coords = np.array(
                        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
                        dtype=np.float32,
                    )
                    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
                    feat = coords2unimol(
                        atoms, coords, dictionary=self.dictionary,
                        max_atoms=max_atoms, remove_hs=remove_hs
                    )
                    id2feat[key] = feat
            except Exception:
                id2feat = {}

            if self.conf_cache:
                try:
                    torch.save(id2feat, cache_map_path)
                except Exception:
                    pass

        uniq_keys, key2uid = [], {}
        for k in df[id_col].astype(str).tolist():
            if k not in key2uid:
                key2uid[k] = len(uniq_keys)
                uniq_keys.append(k)

        uniq_feats = [id2feat.get(k, None) for k in uniq_keys]
        uid_of_row = [key2uid[k] for k in df[id_col].astype(str).tolist()]

        unimol_input, valid_rows, failed_rows = [], [], []
        for i, uid in enumerate(uid_of_row):
            feat = uniq_feats[uid]
            if feat is None:
                failed_rows.append(i)
            else:
                unimol_input.append(_clone_feat_dict(feat))
                valid_rows.append(i)

        self.data["uniq_unimol_input"] = uniq_feats
        self.data["row2uid"] = uid_of_row
        self.data["unimol_input"] = unimol_input
        self.data["valid_rows"] = valid_rows
        self.data["failed_3d_indices"] = failed_rows
        self.data["uniq_ids"] = uniq_keys

    def _attach_atom_index(self, **params):
        raw_df = self.data.get("raw_data", None)
        if raw_df is None:
            raise ValueError("raw_data is missing.")
        atom_index_col = params.get("atom_index_col", "ATOM_INDEX")
        index_base = int(params.get("index_base", 0))

        if atom_index_col not in raw_df.columns:
            raise ValueError(f"Column '{atom_index_col}' not found in raw_data.")

        atom_idx_all = [int(v) for v in raw_df[atom_index_col].tolist()]
        if index_base == 1:
            atom_idx_all = [x - 1 for x in atom_idx_all]
        elif index_base != 0:
            raise ValueError("index_base must be 0 or 1")

        vr = self.data.get("valid_rows", list(range(len(atom_idx_all))))
        atom_idx = [atom_idx_all[i] for i in vr]

        uni_inputs: List[TDict[str, Any]] = self.data["unimol_input"]
        if len(atom_idx) != len(uni_inputs):
            raise ValueError("Length mismatch after reindexing atom_index.")

        row2uid = self.data.get("row2uid", None)
        orig_rows = self.data.get("orig_rows_map", None)
        uniq_paths = self.data.get("uniq_paths", None)
        uniq_ids = self.data.get("uniq_ids", None)

        valid_idx: List[int] = []
        for i, (ai, feat) in enumerate(zip(atom_idx, uni_inputs)):
            n_atoms = int(len(feat["src_tokens"]) - 2)  # Remove BOS/EOS
            if not (0 <= ai < n_atoms):
                uid = None
                if row2uid is not None and i < len(row2uid):
                    uid = row2uid[i]
                uid_info = f" uid={uid}" if uid is not None else ""
                row_info = f" csv_row={orig_rows[i]}" if isinstance(orig_rows, list) and i < len(orig_rows) else ""
                name_info = ""
                if uid is not None:
                    if uniq_paths is not None and uid < len(uniq_paths):
                        name_info = f" path={uniq_paths[uid]}"
                    elif uniq_ids is not None and uid < len(uniq_ids):
                        name_info = f" id={uniq_ids[uid]}"
                raise ValueError(
                    f"Atom index {ai} out of range for sample {i} with {n_atoms} atoms.{uid_info}{row_info}{name_info}"
                )
            valid_idx.append(ai)
        self.data["atom_index"] = valid_idx


# ============================================================
# V2 DataHub (mol2unimolv2; hydrogen removed; needs orig->heavy index map)
# ============================================================

class SingleAtomDataHubV2(_BaseSingleAtomDataHub):
    """Uni-Mol v2 features from SDF via UniMolV2Feature/ mol2unimolv2; forced remove_hs=True."""

    def __init__(self, data=None, is_train: bool=True, save_path: Optional[str]=None, **params):
        self.feature_workers = int(params.get("feature_workers", 0))
        super().__init__(data=data, is_train=is_train, save_path=save_path, **params)

    def _ensure_dictionary(self, **params):
        return

    def _build_unimol_input_from_sdf_paths(self, df, remove_hs: bool, max_atoms: int):
        # remove_hs is forced 'True'
        paths = self._collect_paths_from_df(df)
        path_to_uid, uniq_paths, uid_of_row = {}, [], []
        for p in paths:
            if p not in path_to_uid:
                path_to_uid[p] = len(uniq_paths)
                uniq_paths.append(p)
            uid_of_row.append(path_to_uid[p])

        uniq_mols: List[Optional[Chem.Mol]] = []
        uid_orig2heavy: List[Optional[np.ndarray]] = []
        for path in uniq_paths:
            m = _load_first_mol_from_sdf(path, remove_hs=False)
            uniq_mols.append(m)
            uid_orig2heavy.append(_orig2heavy_map_from_mol(m) if m is not None else None)

        uniq_feats: List[Optional[TDict[str, Any]]] = [None] * len(uniq_paths)
        miss_uids: List[int] = []
        for uid, path in enumerate(uniq_paths):
            key = _file_cache_key(path)
            cached = self._cache_load(key)
            if cached is not None:
                uniq_feats[uid] = cached
            else:
                miss_uids.append(uid)

        if len(miss_uids) > 0:
            build_paths = [uniq_paths[uid] for uid in miss_uids]
            build_mols  = [uniq_mols[uid]  for uid in miss_uids]

            if self.feature_workers <= 0:
                builder = UniMolV2Feature(
                    seed=int(self.params.get("seed", 42)),
                    max_atoms=max_atoms,
                    data_type="molecule",
                    method="rdkit_random",
                    mode=str(self.params.get("mode", "fast")),
                    remove_hs=True,
                    multi_process=False,
                )
                todo_pairs = [(i, m) for i, m in enumerate(build_mols) if m is not None]
                if len(todo_pairs) > 0:
                    order_idx, mols_ok = zip(*todo_pairs)
                    feats = builder.transform_mols(list(mols_ok))
                    for loc, feat in zip(order_idx, feats):
                        uid = miss_uids[loc]
                        uniq_feats[uid] = feat
            else:
                args = [(p, max_atoms) for p in build_paths]
                with Pool(processes=self.feature_workers) as pool:
                    out_list = list(tqdm(pool.imap(_worker_path_to_feature, args),
                                         total=len(args), desc="V2 parallel mol2unimolv2"))
                for uid, feat in zip(miss_uids, out_list):
                    uniq_feats[uid] = feat

            # cache
            for uid in miss_uids:
                feat = uniq_feats[uid]
                if feat is not None:
                    self._cache_save(_file_cache_key(uniq_paths[uid]), feat)

        unimol_input, valid_rows, failed_rows = [], [], []
        for i, uid in enumerate(uid_of_row):
            feat = uniq_feats[uid]
            if feat is None:
                failed_rows.append(i)
            else:
                unimol_input.append(_clone_feat_dict(feat))
                valid_rows.append(i)

        self.data["uniq_unimol_input"] = uniq_feats
        self.data["row2uid"] = uid_of_row
        self.data["unimol_input"] = unimol_input
        self.data["uid_orig2heavy"] = uid_orig2heavy
        self.data["valid_rows"] = valid_rows
        self.data["failed_3d_indices"] = failed_rows
        self.data["uniq_paths"] = uniq_paths

    def _build_unimol_input_from_sdf_single(self, df, sdf_file: str, id_col: str,
                                            sdf_id_prop: str, remove_hs: bool, max_atoms: int):
        # id -> mol
        id2mol: TDict[str, Optional[Chem.Mol]] = {}
        try:
            supp = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False)
            for mol in supp:
                if mol is None or mol.GetNumConformers() == 0:
                    continue
                key = mol.GetProp(sdf_id_prop) if mol.HasProp(sdf_id_prop) else \
                      (mol.GetProp("_Name") if mol.HasProp("_Name") else None)
                if key is None:
                    continue
                id2mol[key] = mol
        except Exception:
            id2mol = {}

        uniq_keys, key2uid = [], {}
        for k in df[id_col].astype(str).tolist():
            if k not in key2uid:
                key2uid[k] = len(uniq_keys)
                uniq_keys.append(k)

        uniq_mols = [id2mol.get(k, None) for k in uniq_keys]
        uid_orig2heavy = [_orig2heavy_map_from_mol(m) if m else None for m in uniq_mols]
        uniq_feats: List[Optional[TDict[str, Any]]] = [None] * len(uniq_keys)

        builder = UniMolV2Feature(
            seed=int(self.params.get("seed", 42)),
            max_atoms=max_atoms,
            data_type="molecule",
            method="rdkit_random",
            mode=str(self.params.get("mode", "fast")),
            remove_hs=True,
            multi_process=False,
        )
        todo_pairs = [(i, m) for i, m in enumerate(uniq_mols) if m is not None]
        if len(todo_pairs) > 0:
            order_idx, mols_ok = zip(*todo_pairs)
            feats = builder.transform_mols(list(mols_ok))
            for loc, feat in zip(order_idx, feats):
                uniq_feats[loc] = feat

        uid_of_row = [key2uid[k] for k in df[id_col].astype(str).tolist()]

        unimol_input, valid_rows, failed_rows = [], [], []
        for i, uid in enumerate(uid_of_row):
            feat = uniq_feats[uid]
            if feat is None:
                failed_rows.append(i)
            else:
                unimol_input.append(_clone_feat_dict(feat))
                valid_rows.append(i)

        self.data["uniq_unimol_input"] = uniq_feats
        self.data["row2uid"] = uid_of_row
        self.data["unimol_input"] = unimol_input
        self.data["uid_orig2heavy"] = uid_orig2heavy
        self.data["valid_rows"] = valid_rows
        self.data["failed_3d_indices"] = failed_rows
        self.data["uniq_ids"] = uniq_keys

    def _attach_atom_index(self, **params):
        raw_df = self.data.get("raw_data", None)
        if raw_df is None:
            raise ValueError("raw_data is missing.")
        atom_index_col = params.get("atom_index_col", "ATOM_INDEX")
        index_base = int(params.get("index_base", 0))
        on_h_index = str(params.get("on_hydrogen_index", "error")).lower()

        if atom_index_col not in raw_df.columns:
            raise ValueError(f"Column '{atom_index_col}' not found in raw_data.")

        atom_idx_all = [int(v) for v in raw_df[atom_index_col].tolist()]
        if index_base == 1:
            atom_idx_all = [x - 1 for x in atom_idx_all]
        elif index_base != 0:
            raise ValueError("index_base must be 0 or 1")

        vr = self.data.get("valid_rows", list(range(len(atom_idx_all))))
        row2uid = self.data.get("row2uid", None)
        atom_idx = [atom_idx_all[i] for i in vr]

        uid_maps = self.data.get("uid_orig2heavy", None)
        if uid_maps is None or row2uid is None:
            raise RuntimeError("orig2heavy maps are not available for conversion.")

        converted, bad_rows = [], []
        for row_pos, ai_full in enumerate(atom_idx):
            uid = row2uid[row_pos]
            m = uid_maps[uid]
            if m is None or ai_full < 0 or ai_full >= len(m):
                bad_rows.append(row_pos)
                converted.append(-1)
            else:
                hai = int(m[ai_full])  # -1 if original atom was H
                if hai < 0:
                    bad_rows.append(row_pos)
                converted.append(hai)

        if bad_rows:
            if on_h_index == "drop":
                keep = sorted(set(range(len(atom_idx))) - set(bad_rows))
                self.data["unimol_input"] = [self.data["unimol_input"][k] for k in keep]
                self.data["raw_data"] = self.data["raw_data"].iloc[keep].reset_index(drop=True)
                if self.data.get("target", None) is not None:
                    self.data["target"] = [self.data["target"][k] for k in keep]
                if row2uid is not None:
                    self.data["row2uid"] = [row2uid[k] for k in keep]
                atom_idx = [converted[k] for k in keep]
                self.data["valid_rows"] = list(range(len(atom_idx)))
                print(f"Dropped {len(bad_rows)} rows whose atom_index pointed to hydrogens.")
            else:
                ex = bad_rows[0]
                raise ValueError(
                    f"CSV atom_index refers to a hydrogen (or out-of-range) after H removal at row {ex}. "
                    f"Set on_hydrogen_index=drop to skip such rows."
                )
        else:
            atom_idx = converted

        uni_inputs = self.data["unimol_input"]
        row2uid = self.data.get("row2uid", None)
        orig_rows = self.data.get("orig_rows_map", None)
        uniq_paths = self.data.get("uniq_paths", None)
        uniq_ids = self.data.get("uniq_ids", None)

        valid_idx = []
        for i, (ai, feat) in enumerate(zip(atom_idx, uni_inputs)):
            n_atoms = int(len(feat["src_tokens"])) if "src_tokens" in feat else int(feat["atom_feat"].shape[0])
            if not (0 <= ai < n_atoms):
                uid = None
                if row2uid is not None and i < len(row2uid):
                    uid = row2uid[i]
                uid_info = f" uid={uid}" if uid is not None else ""
                row_info = f" csv_row={orig_rows[i]}" if isinstance(orig_rows, list) and i < len(orig_rows) else ""
                name_info = ""
                if uid is not None:
                    if uniq_paths is not None and uid < len(uniq_paths):
                        name_info = f" path={uniq_paths[uid]}"
                    elif uniq_ids is not None and uid < len(uniq_ids):
                        name_info = f" id={uniq_ids[uid]}"
                raise ValueError(
                    f"Atom index {ai} out of range for sample {i} with {n_atoms} atoms.{uid_info}{row_info}{name_info}"
                )
            valid_idx.append(ai)
        self.data["atom_index"] = valid_idx


# ============================================================
# Dataset
# ============================================================

class _SingleAtomDataset(Dataset):
    """Dataset exposing Uni-Mol features plus atom_index and target."""
    def __init__(self, hub: SingleAtomDataHubV1 | SingleAtomDataHubV2, indices: List[int], task: str = 'regression'):
        self.hub = hub
        self.hub_type = 'unimolv1' if isinstance(hub, SingleAtomDataHubV1) else 'unimolv2'
        self.indices = indices
        self.task = task

    def __len__(self):
        return len(self.indices)

    def _input_generator(self, feat: TDict[str, Any]) -> TDict[str, Any]:
        if self.hub_type == 'unimolv1':
            return dict(
                src_tokens=feat["src_tokens"],
                src_distance=feat["src_distance"],
                src_coord=feat["src_coord"],
                src_edge_type=feat["src_edge_type"],
            )
        else:
            out = dict(
                src_tokens=feat.get("src_tokens", None),
                src_coord=feat["src_coord"],
            )
            for k in ("atom_feat", "atom_mask", "edge_feat", "shortest_path",
                      "degree", "pair_type", "attn_bias"):
                if k in feat:
                    out[k] = feat[k]
            return out

    def __getitem__(self, i: int):
        j = self.indices[i]
        feat: TDict[str, Any] = self.hub.data['unimol_input'][j]
        sample = self._input_generator(feat)
        sample['atom_index'] = int(self.hub.data['atom_index'][j])
        if self.task.startswith('regression') and ('target' in self.hub.data) and (self.hub.data['target'] is not None):
            t = self.hub.data['target'][j]
            if isinstance(t, (list, tuple, np.ndarray)):
                sample['target'] = float(np.array(t).reshape(-1)[0])
            else:
                sample['target'] = float(t)
        return sample, None


# ============================================================
# Split utility
# ============================================================

def _split_indices_from_column(hub: "SingleAtomDataHubV1 | SingleAtomDataHubV2", split_col: str) -> Tuple[List[int], List[int], List[int]]:
    """Split into train/val/test indices based on a string column with values 'train','val','test'."""
    df = hub.data.get("raw_data", None)
    if df is None:
        raise ValueError("raw_data missing from DataHub.")
    if split_col not in df.columns:
        raise ValueError(f"Split column '{split_col}' not found in CSV.")

    train_idx, val_idx, test_idx = [], [], []
    for i, v in enumerate(df[split_col].tolist()):
        key = str(v).strip().lower()
        if key == "train":
            train_idx.append(i)
        elif key == "val":
            val_idx.append(i)
        elif key == "test":
            test_idx.append(i)
        else:
            raise ValueError(f"Unexpected split value '{v}' at row {i}")
    return train_idx, val_idx, test_idx
