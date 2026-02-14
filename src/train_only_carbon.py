# -*- coding: utf-8 -*-
# End-to-end training script (Uni-Mol v2 ready) for single-atom regression with pre-optimized SDF structures.
# - Reads SDF coordinates directly (no confgen) with filename-only cells in CSV and --sdf_dir
# - Supports per-row SDF filenames or a single SDF with multiple molecules
# - Train/val/test split from a CSV column with values 'train','val','test'
# - Saves best and last checkpoints, evaluates test on best epoch, logs to file
# Comments are in English only.

from __future__ import annotations

import os
import argparse
import logging
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core_modules.dataset import SingleAtomDataHubV1, SingleAtomDataHubV2, _SingleAtomDataset, _split_indices_from_column
from core_modules.model import build_single_atom_model
from core_modules.evaluate import _evaluate_loss, _evaluate_metrics


def train_with_splitcol(
    data: str,
    split_col: str,
    target_cols: Optional[str],
    atom_index_col: str,
    index_base: int,
    structure_source: str,
    sdf_mode: str,
    sdf_dir: Optional[str],
    sdf_name_col: Optional[str],
    sdf_ext: Optional[str],
    sdf_path_col: Optional[str],
    sdf_file: Optional[str],
    sdf_id_prop: str,
    remove_hs: bool,
    max_atoms: int,
    model_name: str,
    model_size: Optional[int],
    batch_size: int,
    lr: float,
    epochs: int,
    num_workers: int,
    atom_out_dim: int,
    atom_head_hidden_dim: Optional[int],
    device: Optional[str],
    save_path: Optional[str],
    feature_workers: int = 0,
    seed: int = 42,
    scale: bool = False,
) -> Tuple[nn.Module, dict, int, Optional[dict]]:
    """Build hub, construct model (v1 or v2), train with best/last saving, and evaluate test on best."""
    # Reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Logger
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        log_file = os.path.join(save_path, "train.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="w")],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
    log = logging.getLogger(__name__)

    # DataHub
    hub = SingleAtomDataHubV1(
        data=data,
        is_train=True,
        save_path=save_path,
        task="regression",
        target_cols=target_cols,
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
        remove_hs=remove_hs,
        max_atoms=max_atoms,
        conf_cache=True,
        feature_workers=feature_workers,
        replace_atom_to_carbon=True,
        fold=1,
    )

    # Sanity
    n_rows = len(hub.data.get("raw_data", []))
    n_feat = len(hub.data.get("unimol_input", []))
    failed = hub.data.get("failed_3d_indices", [])
    log.info(f"[sanity] valid_data={n_rows}  unimol_input={n_feat}  failed_3d={len(failed)}")
    if n_rows != n_feat:
        raise RuntimeError(f"valid_data length ({n_rows}) != unimol_input length ({n_feat})")

    # Splits
    tr_idx, va_idx, te_idx = _split_indices_from_column(hub, split_col)
    log.info(f"train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    if scale:
        mean_ = np.mean([hub.data["target"][i] for i in tr_idx])
        std_  = np.std([hub.data["target"][i] for i in tr_idx])
        hub.init_scaler(mean_, std_); hub.do_scaling()

    # Model (v1/v2 is selected by model_name inside builder)
    model, device_ = build_single_atom_model(
        model_name=model_name,
        model_size=model_size,
        atom_out_dim=atom_out_dim,
        atom_head_hidden_dim=atom_head_hidden_dim,
        remove_hs=remove_hs,
        load_original=True,
        device=device,
    )

    # DataLoaders
    def _collate(samples):
        return model.batch_collate_fn(samples)

    train_ds = _SingleAtomDataset(hub, tr_idx, task="regression")
    val_ds = _SingleAtomDataset(hub, va_idx, task="regression") if len(va_idx) > 0 else None
    test_ds = _SingleAtomDataset(hub, te_idx, task="regression") if len(te_idx) > 0 else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=_collate
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate)
        if val_ds
        else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate)
        if test_ds
        else None
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Train loop with best snapshot
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    checkpoint = {
        'model_name': model_name,
        'model_size': model_size if model_name == 'unimolv2' else None,
        'atom_out_dim': atom_out_dim,
        'atom_head_hidden_dim': atom_head_hidden_dim,
        'mean_': hub.mean_,
        'std_' : hub.std_,
        'state_dict': None,
        }

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, tr_loss_raw = _evaluate_loss(model, train_loader, device_, optimizer)
        history["train_loss"].append(tr_loss)

        # Validation
        val_loss, val_loss_raw = None, None
        if val_loader is not None:
            val_loss, val_loss_raw = _evaluate_loss(model, val_loader, device_, optimizer=None)
        history["val_loss"].append(val_loss)

        log.info(
            f"[Epoch {ep}] train={tr_loss:.6f} (raw_scale RMSE: {float(np.sqrt(tr_loss_raw)):.6f})" + (f" val={val_loss:.6f} (raw_scale RMSE: {float(np.sqrt(val_loss_raw)):.6f})" if val_loss is not None else "")
        )

        # Save best
        if val_loader is not None and val_loss is not None and val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            if save_path:
                checkpoint['state_dict'] = best_state
                torch.save(checkpoint, os.path.join(save_path, "best_model.pt"))
                log.info(f"Saved best model at epoch {ep} (val_loss={val_loss:.6f} (raw_scale RMSE: {float(np.sqrt(val_loss_raw)):.6f}))")

    # Save last
    if save_path:
        last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        checkpoint['state_dict'] = last_state
        torch.save(checkpoint, os.path.join(save_path, "last_model.pt"))
        log.info(f"Saved last model at epoch {epochs}")

    # Test eval with best
    test_results_best = None
    if test_loader is not None:
        if best_state is None and save_path and os.path.exists(os.path.join(save_path, "best_model.pt")):
            best_state = torch.load(os.path.join(save_path, "best_model.pt"), map_location="cpu")['state_dict']
        if best_state is not None:
            model.load_state_dict(best_state, strict=True)
            test_results_best_ = _evaluate_metrics(model, test_loader, device_)
            test_results_best = {
                key: item for key, item in test_results_best_.items() if key not in ("gtrue", "pred")
            }
            log.info(
                "Test@best_epoch={}  ".format(best_epoch)
                + " ".join([f"{k}={v:.6f}" for k, v in test_results_best.items()])
            )
        else:
            log.warning("Best weights not available; skipping test@best evaluation.")

    return model, history, best_epoch, test_results_best


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Uni-Mol (v1) single-atom regression using pre-optimized SDF structures."
    )
    p.add_argument("--data", type=str, required=True, help="CSV readable by Uni-Mol DataHub")
    p.add_argument("--split_col", type=str, required=True, help="Column with values 'train','val','test'")
    p.add_argument(
        "--target_cols",
        type=str,
        required=True,
        help="Target column name(s), comma-separated if multiple (first used)",
    )
    p.add_argument("--atom_index_col", type=str, default="ATOM_INDEX")
    p.add_argument("--index_base", type=int, default=0, choices=[0, 1])

    # SDF source: per-row filenames or full paths, or a single SDF with many molecules
    p.add_argument("--structure_source", type=str, default="sdf", choices=["sdf"])
    p.add_argument("--sdf_mode", type=str, default="per_row", choices=["per_row", "single_file"])

    # Per-row filenames mode:
    p.add_argument(
        "--sdf_dir",
        type=str,
        default=None,
        help="Directory containing SDF files when CSV has only filenames",
    )
    p.add_argument("--sdf_name_col", type=str, default="SDF_NAME", help="CSV column with SDF filenames")
    p.add_argument(
        "--sdf_ext",
        type=str,
        default=None,
        help="Default extension to add when filename has no suffix (e.g., .sdf)",
    )
    p.add_argument(
        "--sdf_path_col",
        type=str,
        default=None,
        help="Alternative CSV column with full/relative paths (overrides name+dir)",
    )

    # Single-file mode:
    p.add_argument("--sdf_file", type=str, default=None, help="Single SDF file path (single_file mode)")
    p.add_argument("--sdf_id_prop", type=str, default="_Name", help="SDF property name for ID mapping (single_file mode)")
    p.add_argument("--id_col", type=str, default=None, help="CSV column used to match SDF property (single_file mode)")

    p.add_argument(
        "--remove_hs",
        action="store_true",
        help="Drop hydrogens in features; ensure ATOM_INDEX convention matches",
    )
    p.add_argument("--max_atoms", type=int, default=512)

    # v1/v2 switch
    p.add_argument("--model_name", type=str, default="unimolv1", choices=["unimolv1"])
    
    # Dataloader/optimizer/training
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=0)

    # Single-atom head config (training-time; checkpoint will carry this for inference)
    p.add_argument("--atom_out_dim", type=int, default=1)
    p.add_argument("--atom_head_hidden_dim", type=int, nargs="*", default=[])

    # Device / IO
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_path", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--scale",
        action="store_true",
        help="Whether implement scaling of target value",
    )

    # Feature parallelism for v2 feature building from SDF
    p.add_argument(
        "--feature_workers",
        type=int,
        default=0,
        help="Workers for parallel v2 feature conversion (0=off; robust path that re-reads SDFs in workers)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Kick off training
    train_with_splitcol(
        data=args.data,
        split_col=args.split_col,
        target_cols=args.target_cols,
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
        remove_hs=args.remove_hs,
        max_atoms=args.max_atoms,
        model_name=args.model_name,
        model_size=args.model_size,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        num_workers=args.num_workers,
        atom_out_dim=args.atom_out_dim,
        atom_head_hidden_dim=(args.atom_head_hidden_dim if len(args.atom_head_hidden_dim) else None),
        device=args.device,
        save_path=args.save_path,
        feature_workers=args.feature_workers,
        seed=args.seed,
        scale=args.scale,
    )
    logging.getLogger(__name__).info("Training finished.")


if __name__ == "__main__":
    main()
