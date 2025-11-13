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

# End-to-end training script for single-atom regression on Uni-Mol with pre-optimized SDF structures.
# - Reads SDF coordinates directly (no confgen) with filename-only cells in CSV and --sdf_dir
# - Supports per-row SDF filenames or a single SDF with multiple molecules
# - Train/val/test split from a CSV column with values 'train','val','test'
# - Saves best and last checkpoints, evaluates test on best epoch, logs to file
# Comments are in English only.

from __future__ import annotations

import numpy as np
import torch

from sklearn.metrics import mean_squared_error

def _input_generator(batch, model_name):
    if model_name == 'unimolv1':
        return dict(
            src_tokens=batch["src_tokens"],
            src_distance=batch["src_distance"],
            src_coord=batch["src_coord"],
            src_edge_type=batch["src_edge_type"],
            atom_index=batch["atom_index"],
            target=batch.get("target", None),
        )
    elif model_name == 'unimolv2':
        return dict(
            atom_feat=batch["atom_feat"],
            atom_mask=batch["atom_mask"],
            edge_feat=batch["edge_feat"],
            shortest_path=batch["shortest_path"],
            degree=batch["degree"],
            pair_type=batch["pair_type"],
            attn_bias=batch["attn_bias"],
            src_tokens=batch["src_tokens"],
            src_coord=batch["src_coord"],
            atom_index=batch["atom_index"],
            target=batch.get("target", None),
        )
    else:
        raise ValueError('No existing model selected..')

def _predict_loader(model, loader, device, only_predict=False):
    """Evaluate model on a DataLoader and return a dict of metrics."""
    def _restore_scale(vals: np.array):
        if loader.dataset.hub.is_scale:
            return (vals * loader.dataset.hub.std_) + loader.dataset.hub.mean_
        return vals

    model.eval()
    n, mse_sum, mae_sum = 0, 0.0, 0.0
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for batch, _ in loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            out = model(
                **_input_generator(batch, model.model_name)
            )
            pred = _restore_scale(out["pred"].detach().cpu().numpy().reshape(-1))
            n += pred.shape[0]
            y_pred_all.append(pred)
            # Evaluation for validation dataset
            if not only_predict and batch.get("target", None) is not None:
                tgt = _restore_scale(batch["target"].detach().cpu().numpy().reshape(-1))
                y_true_all.append(tgt)
                mse_sum += np.sum((pred - tgt) ** 2)
                mae_sum += np.sum(np.abs(pred - tgt))
    return n, y_pred_all, y_true_all, mse_sum, mae_sum


def _evaluate_loss(model,loader,device,optimizer=None):
    total, total_raw, n_batches = 0.0, 0.0, 0
    if optimizer is None:
        model.eval()
        with torch.no_grad():
            for batch, _ in loader:
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device)
                out = model(
                    **_input_generator(batch, model.model_name)
                )
                loss = out["loss"]
                total += float(loss.detach().cpu().item())

                if loader.dataset.hub.is_scale:
                    pred_cpu = out["pred"].detach().cpu().numpy()
                    target_cpu = batch["target"].detach().cpu().numpy()
                    pred_unscaled = (pred_cpu * loader.dataset.hub.std_) + loader.dataset.hub.mean_
                    target_unscaled = (target_cpu * loader.dataset.hub.std_) + loader.dataset.hub.mean_
                    if not np.isfinite(pred_unscaled).all(): raise ValueError("pred_unscaled contains NaN/Inf")
                    if not np.isfinite(target_unscaled).all(): raise ValueError("target_unscaled contains NaN/Inf")
                    loss_raw = mean_squared_error(target_unscaled, pred_unscaled)
                else:
                    loss_raw = loss.detach().cpu().item()
                total_raw += float(loss_raw)

                n_batches += 1
    else:
        for batch, _ in loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            out = model(
                **_input_generator(batch, model.model_name)
            )
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu().item())

            if loader.dataset.hub.is_scale:
                pred_cpu = out["pred"].detach().cpu().numpy()
                target_cpu = batch["target"].detach().cpu().numpy()
                pred_unscaled = (pred_cpu * loader.dataset.hub.std_) + loader.dataset.hub.mean_
                target_unscaled = (target_cpu * loader.dataset.hub.std_) + loader.dataset.hub.mean_
                if not np.isfinite(pred_unscaled).all(): raise ValueError("pred_unscaled contains NaN/Inf")
                if not np.isfinite(target_unscaled).all(): raise ValueError("target_unscaled contains NaN/Inf")
                loss_raw = mean_squared_error(target_unscaled, pred_unscaled)
            else:
                loss_raw = loss.detach().cpu().item()
            total_raw += float(loss_raw)
            
            n_batches += 1
    return total / max(1, n_batches), total_raw / max(1, n_batches)

def _evaluate_metrics(model, loader, device, metrics=("mse", "mae", "rmse", "r2")):
    """Evaluate model on a DataLoader and return a dict of metrics."""
    n, y_pred_all, y_true_all, mse_sum, mae_sum = _predict_loader(model, loader, device, only_predict=False)
    results = {
        'gtrue': y_true_all, 
        'pred' : y_pred_all
    }
    if n > 0:
        if "mse" in metrics:
            results["mse"] = mse_sum / n
        if "mae" in metrics:
            results["mae"] = mae_sum / n
        if "rmse" in metrics:
            results["rmse"] = float(np.sqrt(results["mse"]))
        if "r2" in metrics:
            yt = np.concatenate(y_true_all) if len(y_true_all) else np.array([])
            yp = np.concatenate(y_pred_all) if len(y_pred_all) else np.array([])
            denom = float(np.sum((yt - yt.mean()) ** 2)) if yt.size > 0 else 0.0
            results["r2"] = float(1.0 - np.sum((yp - yt) ** 2) / denom) if denom > 0 else float("nan")
    return results
