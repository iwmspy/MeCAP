import os
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from PIL import Image
import io

from rdkit import Chem
from rdkit.Chem import Draw, AllChem

from visualize_attention_map import load_sdf
from core_modules.model import build_model_from_checkpoint
from pathlib import Path

PWD = os.path.realpath(os.path.dirname(__file__))
ROOT_DIR = os.path.realpath(os.path.join(PWD, ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unscale(pred: torch.Tensor, mean_: Optional[float], std_: Optional[float]) -> float:
    """Unscale a model prediction back to real units."""
    val = pred.item()
    if mean_ is not None and std_ is not None:
        val = val * std_ + mean_
    return val


# ---------------------------------------------------------------------------
# Step 1: remove_atom_from_input
# ---------------------------------------------------------------------------

def remove_atom_from_input(
    src_tokens: torch.Tensor,
    src_distance: torch.Tensor,
    src_coord: torch.Tensor,
    src_edge_type: torch.Tensor,
    atom_j: int,
    target_atom_i: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Remove atom *j* from all input tensors and adjust the target atom index.

    Tensor layout: ``(B, L+2, ...)`` where position 0 = CLS, 1..L = atoms,
    L+1 = SEP.  Removing atom *j* means dropping position ``j + 1``.

    Returns:
        (src_tokens_new, src_distance_new, src_coord_new, src_edge_type_new,
         new_target_atom_i)
    """
    pos_j = atom_j + 1  # CLS offset
    seq_len = src_tokens.size(1)
    keep = [k for k in range(seq_len) if k != pos_j]

    src_tokens_new = src_tokens[:, keep]
    src_distance_new = src_distance[:, keep, :][:, :, keep]
    src_coord_new = src_coord[:, keep, :]
    src_edge_type_new = src_edge_type[:, keep, :][:, :, keep]

    new_target = target_atom_i - 1 if target_atom_i > atom_j else target_atom_i

    return src_tokens_new, src_distance_new, src_coord_new, src_edge_type_new, new_target


# ---------------------------------------------------------------------------
# Step 7 (placed early): verify_ablation_consistency
# ---------------------------------------------------------------------------

def verify_ablation_consistency(
    model,
    src_tokens: torch.Tensor,
    src_distance: torch.Tensor,
    src_coord: torch.Tensor,
    src_edge_type: torch.Tensor,
    atom_j: int,
) -> None:
    """Verify that removing atom *j* preserves intermediate representations
    for the remaining atoms (embedding, padding mask, distance bias)."""
    model.eval()
    with torch.no_grad():
        # --- original intermediate values ---
        emb_orig = model.embed_tokens(src_tokens)
        pad_orig = src_tokens.eq(model.padding_idx)
        bias_orig = model._dist_bias(src_distance, src_edge_type)

        # --- remove atom j ---
        st_new, sd_new, sc_new, se_new, _ = remove_atom_from_input(
            src_tokens, src_distance, src_coord, src_edge_type,
            atom_j=atom_j, target_atom_i=0,
        )

        # --- new intermediate values ---
        emb_new = model.embed_tokens(st_new)
        pad_new = st_new.eq(model.padding_idx)
        bias_new = model._dist_bias(sd_new, se_new)

        # --- slice originals to exclude position j+1 ---
        pos_j = atom_j + 1
        seq_len = src_tokens.size(1)
        keep = [k for k in range(seq_len) if k != pos_j]

        emb_sliced = emb_orig[:, keep, :]
        pad_sliced = pad_orig[:, keep]
        bias_sliced = bias_orig[:, keep, :][:, :, keep]

        # --- assertions ---
        assert torch.allclose(emb_new, emb_sliced, atol=1e-6), \
            f"Embedding mismatch after removing atom {atom_j}"
        assert torch.equal(pad_new, pad_sliced), \
            f"Padding mask mismatch after removing atom {atom_j}"
        assert torch.allclose(bias_new, bias_sliced, atol=1e-6), \
            f"Distance bias mismatch after removing atom {atom_j}"

    print(f"  Ablation consistency verified for atom {atom_j}")


# ---------------------------------------------------------------------------
# Step 2: compute_ablation_scores
# ---------------------------------------------------------------------------

def compute_ablation_scores(
    model,
    src_tokens: torch.Tensor,
    src_distance: torch.Tensor,
    src_coord: torch.Tensor,
    src_edge_type: torch.Tensor,
    target_atom_i: int,
    mean_: Optional[float] = None,
    std_: Optional[float] = None,
) -> Tuple[float, np.ndarray]:
    """Run Leave-One-Out ablation for all atoms relative to target atom *i*.

    Returns:
        y_orig:  Original prediction (unscaled).
        delta_y: ``np.ndarray`` of shape ``(seq_len,)`` where
                 ``delta_y[j] = y_orig - y_j_ablated``.
    """
    # CLS, EOSが重要である可能性も考慮して、seq_lenで計算
    seq_len = src_tokens.size(1)   # L + 2

    model.eval()
    with torch.no_grad():
        # --- original prediction ---
        atom_idx_t = torch.tensor([target_atom_i], device=src_tokens.device, dtype=torch.long)
        out_orig = model(src_tokens, src_distance, src_coord, src_edge_type,atom_index=atom_idx_t)
        y_orig = _unscale(out_orig["pred"], mean_, std_)

        # --- LOO loop ---
        delta_y = np.zeros(seq_len, dtype=np.float64)

        for j in range(seq_len):
            if j == target_atom_i + 1: # CLS考慮
                delta_y[j] = 0.0
                continue

            st_new, sd_new, sc_new, se_new, new_target = remove_atom_from_input(
                src_tokens, src_distance, src_coord, src_edge_type,
                atom_j=j, target_atom_i=target_atom_i,
            )
            new_target_t = torch.tensor(
                [new_target], device=src_tokens.device, dtype=torch.long,
            )
            out_j = model(
                st_new, sd_new, sc_new, se_new,
                atom_index=new_target_t,
            )
            y_j = _unscale(out_j["pred"], mean_, std_)
            delta_y[j] = y_orig - y_j

    
    return y_orig, delta_y


# ---------------------------------------------------------------------------
# Step 3: compute_importance
# ---------------------------------------------------------------------------

def compute_importance(delta_y: np.ndarray, method: str = "abs_sum") -> np.ndarray:
    """Convert ``delta_y`` to relative importance scores.

    Methods:
        ``"abs_sum"``: ``delta_y / sum(|delta_y|)``  (sum of |scores| ~ 1.0).
        ``"abs_max"``: ``delta_y / max(|delta_y|)``  (max |score| = 1.0).
        ``"raw"``:     Return raw delta_y values unchanged.
    """
    if method == "raw":
        return delta_y.copy()
    elif method == "abs_sum":
        total = np.abs(delta_y).sum()
        if total == 0:
            return delta_y.copy()
        return delta_y / total
    elif method == "abs_max":
        max_val = np.abs(delta_y).max()
        if max_val == 0:
            return delta_y.copy()
        return delta_y / max_val
    else:
        raise ValueError(f"Unknown importance method: {method}")


# ---------------------------------------------------------------------------
# Step 4: visualize_ablation_on_structure
# ---------------------------------------------------------------------------

def visualize_ablation_on_structure(
    mol: Chem.Mol,
    importance: np.ndarray,
    target_atom_idx: int,
    exp_val: Optional[float] = None,
    pred_val: Optional[float] = None,
    output_path: Optional[str] = None,
    img_size: Tuple[int, int] = (800, 600),
    figsize: Tuple[int, int] = (12, 6),
    vmin: float = 0.0,
    vmax: Optional[float] = 0.3,
) -> plt.Figure:
    """Visualize ablation importance on 2D molecular structure.

    Uses a white-to-red colormap based on ``|importance|``.
    """
    abs_imp = np.abs(importance)
    if vmax is None:
        vmax = float(abs_imp.max())
        
    # --- 2D coordinates ---
    mol_2d = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_2d)

    # --- Colormap (white -> red) ---
    cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"], N=256)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    n_atoms = mol.GetNumAtoms()
    atom_colors = {}
    for i in range(n_atoms):
        val = abs_imp[i] if i < len(abs_imp) else 0.0
        atom_colors[i] = cmap(norm(val))[:3]

    # --- RDKit drawing ---
    w, h = img_size
    drawer = Draw.MolDraw2DCairo(w, h)
    drawer.drawOptions().addAtomIndices = True
    drawer.DrawMolecule(
        mol_2d,
        highlightAtoms=list(range(n_atoms)),
        highlightAtomColors=atom_colors,
        highlightBonds=[],
    )
    drawer.FinishDrawing()
    mol_img = Image.open(io.BytesIO(drawer.GetDrawingText()))

    # --- Figure ---
    fig, (ax_mol, ax_cbar) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [5, 0.3]},
    )

    ax_mol.imshow(mol_img)
    ax_mol.axis("off")

    symbol = mol.GetAtomWithIdx(target_atom_idx).GetSymbol()
    title_lines = f"LOO Ablation — Target Atom: {target_atom_idx} ({symbol})"
    if exp_val is not None and pred_val is not None:
        title_lines += f"\nExperimental: {exp_val:.3f} kcal/mol, Predicted: {pred_val:.3f} kcal/mol"
    elif exp_val is not None:
        title_lines += f"\nExperimental: {exp_val:.3f} kcal/mol"
    elif pred_val is not None:
        title_lines += f"\nPredicted: {pred_val:.3f} kcal/mol"
    ax_mol.set_title(title_lines, fontsize=14, fontweight="bold")

    # --- Colorbar ---
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=ax_cbar).set_label("|Importance|", fontsize=11)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()
    return fig


# ---------------------------------------------------------------------------
# Step 5: visualize_ablation_barchart
# ---------------------------------------------------------------------------

def visualize_ablation_barchart(
    importance: np.ndarray,
    atoms: List[str],
    target_atom_idx: int,
    pred_val: Optional[float] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    top_k: Optional[int] = None,
) -> plt.Figure:
    """Bar chart of ablation importance scores.

    Colors: positive = blue, negative = orange, target = gray.
    """
    n = len(importance)
    labels = [f"{i}:{atoms[i]}" for i in range(n)]

    # --- optional top_k filtering ---
    if top_k is not None and top_k < n:
        abs_imp = np.abs(importance).copy()
        abs_imp[target_atom_idx] = -1  # exclude target from ranking
        top_indices = np.argsort(abs_imp)[::-1][:top_k]
        indices = sorted(set(list(top_indices) + [target_atom_idx]))
        plot_imp = importance[indices]
        plot_labels = [labels[i] for i in indices]
        target_set = {target_atom_idx}
        orig_indices = indices
    else:
        plot_imp = importance
        plot_labels = labels
        target_set = {target_atom_idx}
        orig_indices = list(range(n))

    # --- colors ---
    colors = []
    for pos, orig_idx in enumerate(orig_indices):
        if orig_idx in target_set:
            colors.append("gray")
        elif plot_imp[pos] >= 0:
            colors.append("steelblue")
        else:
            colors.append("darkorange")

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(plot_imp)), plot_imp, color=colors)
    ax.set_xticks(range(len(plot_imp)))
    ax.set_xticklabels(plot_labels, rotation=90, fontsize=8)
    ax.set_ylabel("Importance", fontsize=11)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")

    title = "LOO Ablation Importance"
    if pred_val is not None:
        title += f" (Predicted: {pred_val:.3f} kcal/mol)"
    ax.set_title(title, fontsize=13, fontweight="bold")

    legend_elements = [
        Patch(facecolor="steelblue", label="Positive contribution"),
        Patch(facecolor="darkorange", label="Negative contribution"),
        Patch(facecolor="gray", label="Target atom"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=9)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()
    return fig


# ---------------------------------------------------------------------------
# Step 6: compatibility wrapper (API-parity with visualize_attention_map)
# ---------------------------------------------------------------------------

def compute_and_visualize_ablation_maps(
    sdf_path: str,
    atom_index: int,
    model,
    normalize: bool = False,
    exp_val: Optional[float] = None,
    pred_val: Optional[float] = None,
    output_path: Optional[str] = None,
    device: str = "cuda:0",
) -> plt.Figure:
    """Compute LOO ablation and visualize on molecular structure.

    Interface mirrors ``visualize_attention_map.compute_and_visualize_attention_maps``.
    Notes:
      - ``baseline_method`` and ``n_steps`` are accepted for interface compatibility
        but are not used in LOO ablation.
      - ``normalize=True`` maps importance values to ``abs_max`` scale.
    """
    _, _, atoms, feat = load_sdf(sdf_path, remove_hs=False)
    supp = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = supp[0]
    if mol is None:
        raise ValueError(f"Failed to load molecule from: {sdf_path}")

    src_tokens = torch.tensor(feat["src_tokens"], device=device).unsqueeze(0)
    src_distance = torch.tensor(feat["src_distance"], device=device).unsqueeze(0)
    src_coord = torch.tensor(feat["src_coord"], device=device).unsqueeze(0)
    src_edge_type = torch.tensor(feat["src_edge_type"], device=device).unsqueeze(0)

    y_orig, delta_y = compute_ablation_scores(
        model=model,
        src_tokens=src_tokens,
        src_distance=src_distance,
        src_coord=src_coord,
        src_edge_type=src_edge_type,
        target_atom_i=atom_index,
        mean_=None,
        std_=None,
    )
    method = "abs_max" if normalize else "abs_sum"
    importance = compute_importance(delta_y, method=method)
    importance = importance[1:-1]  # Drop CLS/EOS for atom-wise visualization.

    fig = visualize_ablation_on_structure(
        mol=mol,
        importance=importance,
        target_atom_idx=atom_index,
        exp_val=exp_val,
        pred_val=pred_val if pred_val is not None else y_orig,
        output_path=output_path,
    )
    return fig


# ---------------------------------------------------------------------------
# Step 7: __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Settings (mirrors attention_map.py) ---
    exp_df = pd.read_csv(f"{ROOT_DIR}/data/references/QMdata4ML/df_nuc_x_sample.csv", index_col=0)
    pred_df = pd.read_csv(f"{ROOT_DIR}/data/results/mecap_ref_mca_layer_0/predictions.csv", index_col=0)
    checkpoint = f"{ROOT_DIR}/data/results/mecap_ref_maa_layer_0/best_model.pt"
    device = "cuda:0"

    targets_smiles = ["CN(C)C(=O)CCNC(=O)NCc1ccc(Br)cc1Cl", "NOCc1cccc(I)c1"]
    base_output_dir = f"{ROOT_DIR}/data/results"

    # --- Load model ---
    meta_state, model, device_ = build_model_from_checkpoint(checkpoint_path=checkpoint, device=device)
    mean_ = meta_state.get("mean_", None)
    std_ = meta_state.get("std_", None)

    for target_smiles in targets_smiles:
        target_name = list(set(exp_df.query("smiles == @target_smiles")["name"]))[0]
        sdf_path = f"{ROOT_DIR}/data/results/confs_from_smiles_rdkit/{target_name}.sdf"

        mol, coords, atoms, feat = load_sdf(sdf_path, remove_hs=False)
        src_tokens = torch.tensor(feat["src_tokens"], device=device).unsqueeze(0)
        src_distance = torch.tensor(feat["src_distance"], device=device).unsqueeze(0)
        src_coord = torch.tensor(feat["src_coord"], device=device).unsqueeze(0)
        src_edge_type = torch.tensor(feat["src_edge_type"], device=device).unsqueeze(0)

        n_atoms = mol.GetNumAtoms()

        # --- Consistency verification (once per molecule, atom 0) ---
        print(f"\n=== Verifying ablation consistency for {target_smiles} ===")
        verify_ablation_consistency(model, src_tokens, src_distance, src_coord, src_edge_type, atom_j=0,)

        nuc_sites = list(exp_df.query("smiles == @target_smiles")["nuc_sites"])

        smiles_dir = Path(base_output_dir, "interpretation", target_smiles, "ablation")
        smiles_dir.mkdir(parents=True, exist_ok=True)

        for nuc_site in nuc_sites:
            print(f"\n--- Ablation for target atom {nuc_site} ---")

            # --- Compute ablation scores ---
            y_orig, delta_y = compute_ablation_scores(
                model, src_tokens, src_distance, src_coord, src_edge_type,
                target_atom_i=nuc_site, mean_=mean_, std_=std_,
            )
            importance = compute_importance(delta_y, method="abs_sum")
            importance = importance[1:-1] # CLS, EOSを除外してプロット

            print(f"  y_orig = {y_orig:.4f}")
            print(f"  delta_y[target] = {delta_y[nuc_site]:.6f} (should be 0.0)")
            print(f"  sum(|importance|) = {np.abs(importance).sum():.6f} (should be ~1.0)")

            # --- Get experimental / predicted values for title ---
            pred_val = pred_df.query(
                "smiles == @target_smiles and nuc_sites == @nuc_site"
            )["pred"].values[0]
            exp_val = pred_df.query(
                "smiles == @target_smiles and nuc_sites == @nuc_site"
            )["MCA_values"].values[0]

            # --- Structure colormap ---
            visualize_ablation_on_structure(
                mol, importance, target_atom_idx=nuc_site,
                exp_val=exp_val,
                pred_val=pred_val,
                output_path=os.path.join(smiles_dir, f"{nuc_site}.png"),
            )

            # --- Bar chart ---
            # CLS, EOSを除外してプロット
            visualize_ablation_barchart(
                importance, atoms, target_atom_idx=nuc_site,
                pred_val=pred_val,
                output_path=os.path.join(smiles_dir, f"barchart_{nuc_site}.png"),
            )
            
