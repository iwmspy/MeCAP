# -*- coding: utf-8 -*-
# End-to-end training script for single-atom regression on Uni-Mol with pre-optimized SDF structures.
# - Reads SDF coordinates directly (no confgen) with filename-only cells in CSV and --sdf_dir
# - Supports per-row SDF filenames or a single SDF with multiple molecules
# - Train/val/test split from a CSV column with values 'train','val','test'
# - Saves best and last checkpoints, evaluates test on best epoch, logs to file
# Comments are in English only.

# -*- coding: utf-8 -*-
# Predict-only utility: load a trained checkpoint, run batched inference,
# and return a DataFrame equal to the original CSV with a prediction column attached.
# Comments are in English only.

from __future__ import annotations

import argparse
import os
import logging
from typing import Optional, Literal

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from core_modules.dataset import SingleAtomDataHubV1, SingleAtomDataHubV2, _SingleAtomDataset
from core_modules.model import build_model_from_checkpoint
from core_modules.evaluate import _predict_loader


def predict_attach_to_csv(
    data: str,
    checkpoint: str,
    *,
    # Split control
    split: Literal["train", "val", "test", "all"] = "all",
    split_col: Optional[str] = None,
    # Atom index, indexing base
    atom_index_col: str = "ATOM_INDEX",
    index_base: int = 0,
    # SDF input configuration (filename-only cells are supported)
    structure_source: str = "sdf",
    sdf_mode: str = "per_row",                 # "per_row" or "single_file"
    sdf_dir: Optional[str] = None,             # used when CSV has filename-only cells
    sdf_name_col: Optional[str] = "SDF_NAME",  # column storing SDF filenames
    sdf_ext: Optional[str] = None,             # default extension when filenames have no suffix
    sdf_path_col: Optional[str] = None,        # alternative: direct path column in CSV
    sdf_file: Optional[str] = None,            # single-file mode only
    sdf_id_prop: str = "_Name",                # single-file mode: molecule ID property to match
    id_col: Optional[str] = None,              # single-file mode: CSV column to match sdf_id_prop
    conf_cache: bool = True,              # single-file mode: CSV column to match sdf_id_prop
    # Uni-Mol feature controls
    remove_hs: bool = True,
    max_atoms: int = 512,
    # Inference/runtime
    batch_size: int = 256,
    num_workers: int = 0,
    feature_workers: int = 0,
    device: Optional[str] = None,
    # Output
    pred_col: str = "pred",
    output_csv: Optional[str] = None,
    # Logging
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a trained Uni-Mol single-atom regression model and attach predictions to the original CSV.

    Args:
        data: Path to input CSV.
        checkpoint: Path to trained checkpoint (state_dict or model wrapper).
        split: Subset to predict ("train", "val", "test", "all").
        split_col: Column in CSV that contains 'train'/'val'/'test'.
        atom_index_col: Column with per-row target atom index.
        index_base: 0-based or 1-based atom indices in CSV.
        structure_source, sdf_mode, ...: SDF input configuration (see training script).
        remove_hs, max_atoms, model_name: Feature and model configuration.
        batch_size, num_workers, device: Inference parameters.
        pred_col: Name of prediction column to attach.
        output_csv: If provided, save the resulting DataFrame to this path.
        save_path: Optional run directory for caches/log; can be None.

    Returns:
        A pandas DataFrame equal to the original CSV with predictions in `pred_col`.
        Rows whose SDF failed or were not selected by `split` remain NaN.
    """
    # Set up a minimal logger
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        log_file = os.path.join(save_path, "predict.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="w")],
        )
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                            handlers=[logging.StreamHandler()])
    log = logging.getLogger(__name__)

    # Read the original CSV (keep full length for scattering predictions back)
    if not os.path.exists(data):
        raise FileNotFoundError(f"CSV not found: {data}")
    df_orig = pd.read_csv(data)
    n_orig = len(df_orig)

    # Build model and load checkpoint
    meta_state, model, device_ = build_model_from_checkpoint(
        checkpoint_path=checkpoint,
        remove_hs=remove_hs,
        device=device,
    )
    model_name = meta_state['model_name']
    mean_ = meta_state['mean_']
    std_  = meta_state['std_']
    log.info(f"Loaded checkpoint: {checkpoint}")

    # Build DataHub in inference mode; this will:
    #  - read CSV internally again,
    #  - build unimol_input from SDF,
    #  - slice raw_data to valid_rows (dropping failed SDF rows).
    DHUB_MOD = SingleAtomDataHubV1 if model_name == 'unimolv1' else SingleAtomDataHubV2
    hub = DHUB_MOD(
        data=data,
        is_train=False,
        save_path=save_path,
        task="regression",
        model_name=model_name,
        target_cols=None,  # prediction-only; do not require target
        atom_index_col=atom_index_col,
        index_base=index_base,
        structure_source=structure_source,
        sdf_mode=sdf_mode,
        sdf_dir=sdf_dir,
        sdf_name_col=sdf_name_col,
        sdf_ext=sdf_ext,
        sdf_path_col=sdf_path_col,
        sdf_file=sdf_file,
        sdf_id_prop=sdf_id_prop,
        id_col=id_col,
        remove_hs=remove_hs,
        max_atoms=max_atoms,
        conf_cache=conf_cache,
        fold=1,
        feature_workers=feature_workers,
    )

    # Sanity logging
    valid_rows = hub.data.get("valid_rows", [])
    failed = hub.data.get("failed_3d_indices", [])
    n_after = len(hub.data.get("raw_data", []))
    n_feats = len(hub.data.get("unimol_input", []))
    log.info(f"[sanity] orig_csv={n_orig}  valid_rows={len(valid_rows)}  "
             f"raw_data_after={n_after}  unimol_input={n_feats}  failed_3d={len(failed)}")
    if n_after != n_feats:
        raise RuntimeError(f"Inconsistent hub state: raw_data_after={n_after} vs unimol_input={n_feats}")

    orig_rows_map = hub.data.get("orig_rows_map", None)
    if orig_rows_map is None:
        raise RuntimeError("orig_rows_map is missing in DataHub; cannot scatter predictions back to original CSV.")

    # Select indices to predict on (indices are w.r.t. hub.data['raw_data'] AFTER slicing)
    if split == "all":
        idxs_reduced = list(range(len(hub.data["raw_data"])))
    else:
        if split_col is None or split_col not in df_orig.columns:
            raise ValueError("split != 'all' requires split_col in CSV.")
        split_vals = df_orig[split_col].astype(str).str.strip().str.lower().tolist()
        orig_sel = {i for i, v in enumerate(split_vals) if v == split}
        idxs_reduced = [i for i, orig_i in enumerate(orig_rows_map) if orig_i in orig_sel]

    # DataLoader on the reduced set
    ds = _SingleAtomDataset(hub, idxs_reduced, task="regression")
    def _collate(samples):
        return model.batch_collate_fn(samples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=_collate)

    # Run prediction (only_predict=True skips metric accumulation)
    n, y_pred_batches, _, _, _ = _predict_loader(model, loader, device_, only_predict=True)
    if len(y_pred_batches) == 0:
        # No rows?
        raise TimeoutError('No prediction results found!!')

    preds = (np.concatenate(y_pred_batches, axis=0) * std_ ) + mean_ \
        if mean_ is not None else np.concatenate(y_pred_batches, axis=0)
    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds[:, 0]

    # Scatter predictions back to the ORIGINAL CSV row indices
    df_out = df_orig.copy()
    df_out[pred_col] = np.nan
    # For each reduced index (position in hub.raw_data), get original row index via valid_rows
    orig_rows = [orig_rows_map[i] for i in idxs_reduced]
    # Safety: clip length in case of partial final batch
    m = min(len(orig_rows), len(preds))
    df_out.loc[orig_rows[:m], pred_col] = preds[:m]

    # Optional saving
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df_out.to_csv(output_csv, index=False)
        log.info(f"Saved predictions to: {output_csv}")

    return df_out

# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run Uni-Mol inference and attach predictions to CSV.")
    p.add_argument("--data", type=str, required=True, help="Input CSV")
    p.add_argument("--checkpoint", type=str, required=True, help="Trained checkpoint path")
    p.add_argument("--split", type=str, default="all", choices=["train","val","test","all"])
    p.add_argument("--split_col", type=str, default=None)

    p.add_argument("--atom_index_col", type=str, default="ATOM_INDEX")
    p.add_argument("--index_base", type=int, default=0, choices=[0,1])

    # SDF input
    p.add_argument("--structure_source", type=str, default="sdf", choices=["sdf"])
    p.add_argument("--sdf_mode", type=str, default="per_row", choices=["per_row","single_file"])
    p.add_argument("--sdf_dir", type=str, default=None)
    p.add_argument("--sdf_name_col", type=str, default="SDF_NAME")
    p.add_argument("--sdf_ext", type=str, default=None)
    p.add_argument("--sdf_path_col", type=str, default=None)
    p.add_argument("--sdf_file", type=str, default=None)
    p.add_argument("--sdf_id_prop", type=str, default="_Name")
    p.add_argument("--id_col", type=str, default=None)
    p.add_argument("--no_cache", action="store_false")

    p.add_argument("--remove_hs", action="store_true")
    p.add_argument("--max_atoms", type=int, default=512)

    # Runtime
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--feature_workers",
        type=int,
        default=0,
        help="Workers for parallel v2 feature conversion (0=off; robust path that re-reads SDFs in workers)",
    )
    p.add_argument("--device", type=str, default=None)

    # Output
    p.add_argument("--pred_col", type=str, default="pred")
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--save_path", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()

    # Kick off training
    predict_attach_to_csv(
        data=args.data,
        checkpoint=args.checkpoint,
        split=args.split,
        split_col=args.split_col,
        atom_index_col=args.atom_index_col,
        index_base=args.index_base,
        structure_source=args.structure_source,
        sdf_mode=args.sdf_mode,
        sdf_dir=args.sdf_dir,
        sdf_name_col=args.sdf_name_col,
        sdf_ext=args.sdf_ext,
        sdf_path_col=args.sdf_path_col,
        sdf_file=args.sdf_file,
        sdf_id_prop=args.sdf_id_prop,
        id_col=args.id_col,
        conf_cache=args.no_cache,
        remove_hs=args.remove_hs,
        max_atoms=args.max_atoms,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        feature_workers=args.feature_workers,
        device=args.device,
        pred_col=args.pred_col,
        output_csv=args.output_csv,
        save_path=args.save_path,
    )
    logging.getLogger(__name__).info("Prediction finished.")

if __name__ == "__main__":
    main()
