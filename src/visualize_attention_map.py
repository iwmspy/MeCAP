import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw
import io

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem import rdDetermineBonds
from unimol_tools.data.conformer import coords2unimol
from unimol_tools.data import Dictionary
from unimol_tools.weights import WEIGHT_DIR
from unimol_tools.config import MODEL_CONFIG

from pathlib import Path
from core_modules.model import build_model_from_checkpoint
from core_modules.dataset import _resolve_model_dict_key

PWD = os.path.realpath(os.path.dirname(__file__))
ROOT_DIR = os.path.realpath(os.path.join(PWD, '..'))


def load_sdf(
    sdf_path: str,
    remove_hs: bool = False
):
    """Load an SDF file and return coordinates, atom symbols, and Uni-Mol features."""
    dict_key = _resolve_model_dict_key(model_name="unimolv1", remove_hs=remove_hs)
    dict_path = os.path.join(WEIGHT_DIR, MODEL_CONFIG["dict"][dict_key])
    supp = Chem.SDMolSupplier(sdf_path, removeHs=remove_hs)
    mol = supp[0]
    if mol is not None and mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
            dtype=np.float32,
        )
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        # CLS, EOSは座標(0, 0, 0)として扱っている
        feat = coords2unimol(atoms, coords, dictionary=Dictionary.load(dict_path), remove_hs=remove_hs)
    return mol, coords, atoms, feat


def extract_attention_weights(
    model,
    src_tokens: torch.Tensor,
    src_distance: torch.Tensor,
    src_coord: torch.Tensor,
    src_edge_type: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Extract post-softmax attention probabilities from all encoder layers.

    Args:
        model: Trained UniMolV1Model.
        src_tokens: (B, L) token indices.
        src_distance: (B, L, L) distance matrix.
        src_coord: (B, L, 3) coordinates.
        src_edge_type: (B, L, L) edge types.

    Returns:
        Dict mapping ``"encoder_layer_{i}"`` to attention probability tensors
        of shape ``(H, N, N)`` (CLS/EOS stripped, post-softmax).
        H = number of attention heads, N = number of atoms.
    """
    model.eval()
    with torch.no_grad():
        x = model.embed_tokens(src_tokens)
        graph_attn_bias = model._dist_bias(src_distance, src_edge_type)

        encoder = model.encoder

        # Replicate encoder pre-processing
        x = encoder.emb_layer_norm(x)
        x = F.dropout(x, p=encoder.emb_dropout, training=False)

        attn_mask = graph_attn_bias

        # Iterate through layers, collecting attention from the target layer
        attn_probs_dict = {}
        for i, encoder_layer in enumerate(encoder.layers):
            x, attn_mask, attn_probs = encoder_layer(
                x, padding_mask=None, attn_bias=attn_mask,
                return_attn=True,
            )

            # Strip CLS (position 0) and EOS (position -1)
            attn_probs = attn_probs[:, 1:-1, 1:-1]

            attn_probs_dict[f'encoder_layer_{i+1}'] = attn_probs.clone().detach().cpu()

    return attn_probs_dict


def visualize_attention_heatmap(
    attn_matrix: np.ndarray,
    atoms: List[str],
    target_atom_idx: Optional[int] = None,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """Visualize atom x atom attention as a heatmap.

    Args:
        attn_matrix: (N, N) attention matrix for a single sample.
        atoms: List of atom symbols (length N).
        target_atom_idx: If provided, highlight the target atom row/column.
        title: Plot title.
        output_path: Save path (None to skip saving).
        figsize: Figure size.
        cmap: Colormap name.
        vmin: Colorscale minimum.
        vmax: Colorscale maximum.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = [f"{i}:{s}" for i, s in enumerate(atoms)]

    im = ax.imshow(attn_matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(atoms)))
    ax.set_yticks(range(len(atoms)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Key (attended to)", fontsize=11)
    ax.set_ylabel("Query (attending from)", fontsize=11)

    if target_atom_idx is not None:
        ax.axhline(y=target_atom_idx, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.axvline(x=target_atom_idx, color="red", linewidth=1.5, linestyle="--", alpha=0.7)

    plt.colorbar(im, ax=ax, label="Attention Weight")

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()
    return fig


def visualize_attention_heatmaps_grid(
    attn_matrices: Dict[str, np.ndarray],
    atoms: List[str],
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    cols: int = 3,
    cmap: str = "Blues",
) -> plt.Figure:
    """Visualize layer-wise attention heatmaps in a single figure."""
    layer_items = list(attn_matrices.items())
    n_layers = len(layer_items)
    if n_layers == 0:
        raise ValueError("attn_matrices is empty.")

    cols = max(1, min(cols, n_layers))
    rows = int(np.ceil(n_layers / cols))

    all_vals = np.concatenate([m.reshape(-1) for _, m in layer_items])
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())

    fig, axes = plt.subplots(rows, cols, figsize=(6.3 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    labels = [f"{i}:{s}" for i, s in enumerate(atoms)]

    last_im = None
    for i, (layer_name, attn_matrix) in enumerate(layer_items):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        last_im = ax.imshow(attn_matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(atoms)))
        ax.set_yticks(range(len(atoms)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Key", fontsize=9)
        ax.set_ylabel("Query", fontsize=9)
        layer_num = int(layer_name.split("_")[-1]) if layer_name.split("_")[-1].isdigit() else i + 1
        ax.text(
            0.5,
            -0.28,
            f"Layer={layer_num}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.95),
        )

    for j in range(n_layers, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    if last_im is not None:
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(last_im, cax=cax, label="Attention Weight")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.subplots_adjust(top=0.90, right=0.90, bottom=0.12)
    else:
        fig.subplots_adjust(top=0.95, right=0.90, bottom=0.12)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close(fig)
    return fig


def _make_mol_overlay_image(
    mol: Chem.Mol,
    attn_from_target: np.ndarray,
    target_atom_idx: int,
    img_size: Tuple[int, int],
    vmin: float,
    vmax: float,
    show_atom_indices: Optional[bool] = None,
    target_aspect: float = 1.1,
) -> Image.Image:
    """Create an RDKit 2D drawing image with attention-based atom coloring."""
    mol_2d = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_2d)

    cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"], N=256)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    n_atoms = mol.GetNumAtoms()
    atom_colors = {}
    for i in range(n_atoms):
        val = attn_from_target[i] if i < len(attn_from_target) else 0.0
        atom_colors[i] = cmap(norm(val))[:3]

    w, h = img_size
    drawer = Draw.MolDraw2DCairo(w, h)
    draw_opts = drawer.drawOptions()
    if show_atom_indices is None:
        show_atom_indices = n_atoms <= 32
    draw_opts.addAtomIndices = show_atom_indices
    if hasattr(draw_opts, "atomHighlightsAreCircles"):
        draw_opts.atomHighlightsAreCircles = True
    if hasattr(draw_opts, "padding"):
        draw_opts.padding = 0.01
    if hasattr(draw_opts, "annotationFontScale"):
        draw_opts.annotationFontScale = 0.85
    if hasattr(draw_opts, "baseFontSize"):
        draw_opts.baseFontSize = 0.50
    if hasattr(draw_opts, "bondLineWidth"):
        draw_opts.bondLineWidth = 1.6
    highlight_radii = {i: 0.28 for i in range(n_atoms)}
    drawer.DrawMolecule(
        mol_2d,
        highlightAtoms=list(range(n_atoms)),
        highlightAtomColors=atom_colors,
        highlightAtomRadii=highlight_radii,
        highlightBonds=[],
    )
    target_xy = drawer.GetDrawCoords(target_atom_idx)
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText()))

    # Emphasize the target atom by outline (not by additional color fill).
    draw = ImageDraw.Draw(img)
    tx, ty = float(target_xy.x), float(target_xy.y)
    r_outer, r_inner = 22, 18
    # Use cyan-blue outline for high contrast against red attention map.
    draw.ellipse((tx - r_outer, ty - r_outer, tx + r_outer, ty + r_outer), outline=(0, 114, 178), width=5)
    draw.ellipse((tx - r_inner, ty - r_inner, tx + r_inner, ty + r_inner), outline=(255, 255, 255), width=3)

    return _trim_white_margins(img, pad=10)


def _trim_white_margins(img: Image.Image, pad: int = 8) -> Image.Image:
    """Trim near-white margins so molecule occupies subplot area better."""
    rgb = img.convert("RGB")
    arr = np.array(rgb)
    bg = (arr > 245).all(axis=2)
    fg = ~bg
    ys, xs = np.where(fg)
    if len(xs) == 0 or len(ys) == 0:
        return img
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, arr.shape[1])
    y1 = min(int(ys.max()) + pad + 1, arr.shape[0])
    return img.crop((x0, y0, x1, y1))


def visualize_attention_on_structure(
    mol: Chem.Mol,
    attn_from_target: np.ndarray,
    target_atom_idx: int,
    pred_val: Optional[float] = None,
    exp_val: Optional[float] = None,
    output_path: Optional[str] = None,
    img_size: Tuple[int, int] = (800, 600),
    figsize: Tuple[int, int] = (12, 6),
    vmin: float = 0.0,
    vmax: Optional[float] = 0.2,
) -> plt.Figure:
    """Visualize attention from a target atom to all others on 2D molecular structure.

    Args:
        mol: RDKit molecule.
        attn_from_target: (N,) attention weights from the target atom to all atoms.
        target_atom_idx: Index of the target (query) atom.
        pred_val: Predicted value (optional, for display).
        exp_val: Experimental value (optional, for display).
        output_path: Save path (None to skip saving).
        img_size: RDKit drawing size (w, h).
        figsize: matplotlib figure size.
        vmin: Colorscale minimum.
        vmax: Colorscale maximum (if None, use max of attn_from_target).

    Returns:
        matplotlib Figure.
    """
    if vmax is None:
        vmax = float(attn_from_target.max())

    # --- 2D coordinates ---
    mol_2d = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_2d)

    # --- Colormap (white -> red) ---
    cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"], N=256)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    n_atoms = mol.GetNumAtoms()
    atom_colors = {}
    for i in range(n_atoms):
        val = attn_from_target[i] if i < len(attn_from_target) else 0.0
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
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [5, 0.3]}
    )

    ax_mol.imshow(mol_img)
    ax_mol.axis("off")

    symbol = mol.GetAtomWithIdx(target_atom_idx).GetSymbol()
    title_lines = f"Attention Map — Target Atom: {target_atom_idx} ({symbol})"
    if pred_val is not None and exp_val is not None:
        title_lines += (
            f"\nExperimental: {exp_val:.3f} kcal/mol, Predicted: {pred_val:.3f} kcal/mol"
        )
    ax_mol.set_title(title_lines, fontsize=14, fontweight="bold")

    # --- Colorbar ---
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=ax_cbar).set_label("Attention Weight", fontsize=11)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()
    return fig


def visualize_attention_on_structure_grid(
    mol: Chem.Mol,
    attn_by_layer: Dict[str, np.ndarray],
    target_atom_idx: int,
    pred_val: Optional[float] = None,
    exp_val: Optional[float] = None,
    output_path: Optional[str] = None,
    img_size: Tuple[int, int] = (1100, 700),
    cols: int = 3,
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    show_atom_indices: Optional[bool] = None,
) -> plt.Figure:
    """Visualize layer-wise structure overlays in a single figure."""
    layer_items = list(attn_by_layer.items())
    n_layers = len(layer_items)
    if n_layers == 0:
        raise ValueError("attn_by_layer is empty.")

    if vmax is None:
        vmax = float(max(np.max(v) for _, v in layer_items))

    cols = max(1, min(cols, n_layers))
    rows = int(np.ceil(n_layers / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.2 * rows), facecolor="white")
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for i, (layer_name, attn_from_target) in enumerate(layer_items):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        mol_img = _make_mol_overlay_image(
            mol=mol,
            attn_from_target=attn_from_target,
            target_atom_idx=target_atom_idx,
            img_size=img_size,
            vmin=vmin,
            vmax=vmax,
            show_atom_indices=show_atom_indices,
        )
        ax.imshow(mol_img)
        ax.axis("off")
        ax.set_facecolor("white")
        layer_num = int(layer_name.split("_")[-1]) if layer_name.split("_")[-1].isdigit() else i + 1
        ax.text(
            0.5,
            -0.08,
            f"Layer={layer_num}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.95),
        )

    for j in range(n_layers, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"], N=256)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(sm, cax=cax, label="Attention Weight")

    symbol = mol.GetAtomWithIdx(target_atom_idx).GetSymbol()
    title_lines = f"Attention Map (All Layers) — Target Atom: {target_atom_idx} ({symbol})"
    if pred_val is not None and exp_val is not None:
        title_lines += f"\nExperimental: {exp_val:.3f} kcal/mol, Predicted: {pred_val:.3f} kcal/mol"
    fig.suptitle(title_lines, fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.95, right=0.90, bottom=0.10)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close(fig)
    return fig


def compute_and_visualize_attention_maps(
    sdf_path: str,
    atom_index: int,
    model,
    normalize: bool = False,
    exp_val: Optional[float] = None,
    pred_val: Optional[float] = None,
    output_path: Optional[str] = None,
    img_size = (700, 700),
    device: str = "cuda:0",
    cols: int = 3,
) -> plt.Figure:
    """Compatibility wrapper with interpretation.compute_and_visualize_attributions.

    This wrapper accepts nearly the same input interface and produces
    an all-layer attention-on-structure figure for a target atom.
    Note:
      - `baseline_method` and `n_steps` are accepted for API compatibility
        but are not used in attention-map extraction.
      - `normalize=True` applies per-layer min-max normalization to the
        target-atom attention vector for display only.
    """
    mol, coords, atoms, feat = load_sdf(sdf_path, remove_hs=False)

    # Add bonds if molecule has no bonds
    if mol is not None and mol.GetNumBonds() == 0:
        mol_copy = Chem.Mol(mol)
        charge = [a.GetFormalCharge() for a in mol.GetAtoms()]
        total_charge = sum(charge)
        rdDetermineBonds.DetermineBonds(mol_copy, useHueckel=True, charge=total_charge)
        mol = mol_copy

    src_tokens = torch.tensor(feat["src_tokens"], device=device).unsqueeze(0)
    src_distance = torch.tensor(feat["src_distance"], device=device).unsqueeze(0)
    src_coord = torch.tensor(feat["src_coord"], device=device).unsqueeze(0)
    src_edge_type = torch.tensor(feat["src_edge_type"], device=device).unsqueeze(0)

    attn_weights_dict = extract_attention_weights(
        model=model,
        src_tokens=src_tokens,
        src_distance=src_distance,
        src_coord=src_coord,
        src_edge_type=src_edge_type,
    )
    attn_agg_by_layer = {
        layer_name: attn_weights.mean(dim=0).cpu().numpy()
        for layer_name, attn_weights in attn_weights_dict.items()
    }
    attn_agg_by_layer = dict(
        sorted(attn_agg_by_layer.items(), key=lambda kv: int(kv[0].split("_")[-1]))
    )

    attn_from_target_by_layer = {}
    for layer_name, attn_agg in attn_agg_by_layer.items():
        vec = attn_agg[atom_index, :].copy()
        if normalize:
            vmin, vmax = float(vec.min()), float(vec.max())
            if vmax > vmin:
                vec = (vec - vmin) / (vmax - vmin)
        attn_from_target_by_layer[layer_name] = vec

    fig = visualize_attention_on_structure_grid(
        mol=mol,
        attn_by_layer=attn_from_target_by_layer,
        target_atom_idx=atom_index,
        pred_val=pred_val,
        exp_val=exp_val,
        output_path=output_path,
        cols=cols,
        img_size=img_size,
    )
    return fig


if __name__ == "__main__":

    # --- Settings ---
    exp_df     = pd.read_csv(f"{ROOT_DIR}/data/references/QMdata4ML/df_nuc_x_sample.csv", index_col=0)
    pred_df    = pd.read_csv(f"{ROOT_DIR}/data/results/mecap_ref_mca_layer_0/predictions.csv", index_col=0)
    checkpoint = f"{ROOT_DIR}/data/results/mecap_ref_maa_layer_0/best_model.pt"
    device     = "cuda:0"

    targets_smiles = ["CN(C)C(=O)CCNC(=O)NCc1ccc(Br)cc1Cl", "NOCc1cccc(I)c1"]
    base_output_dir = f"{ROOT_DIR}/data/results"

    # --- Load model ---
    meta_state, model, _ = build_model_from_checkpoint(checkpoint_path=checkpoint, device=device)

    for target_smiles in targets_smiles:
        target_name = list(set(exp_df.query("smiles == @target_smiles")["name"]))[0]
        sdf_path = f"{ROOT_DIR}/data/results/confs_from_smiles_rdkit/{target_name}.sdf"

        mol, coords, atoms, feat = load_sdf(sdf_path, remove_hs=False)
        src_tokens = torch.tensor(feat["src_tokens"], device=device).unsqueeze(0)
        src_distance = torch.tensor(feat["src_distance"], device=device).unsqueeze(0)
        src_coord = torch.tensor(feat["src_coord"], device=device).unsqueeze(0)
        src_edge_type = torch.tensor(feat["src_edge_type"], device=device).unsqueeze(0)

        nuc_sites = list(exp_df.query("smiles == @target_smiles")["nuc_sites"])

        # --- Extract attention weights for this layer ---
        attn_weights_dict = extract_attention_weights(
            model, src_tokens, src_distance, src_coord, src_edge_type
        )

        attn_agg_by_layer = {
            layer_name: attn_weights.mean(dim=0).cpu().numpy()
            for layer_name, attn_weights in attn_weights_dict.items()
        }

        smiles_dir = Path(
            base_output_dir, "interpretation", target_smiles, "attention"
        )
        smiles_dir.mkdir(parents=True, exist_ok=True)

        # --- Heatmap (all layers in one figure) ---
        visualize_attention_heatmaps_grid(
            attn_agg_by_layer,
            atoms,
            title=f"Attention Heatmaps (All Layers) — {target_smiles}",
            output_path=smiles_dir / "heatmaps_all_layers.png",
        )

        # --- Structure overlay (all layers in one figure per nucleophilic site) ---
        for nuc_site in nuc_sites:
            exp_val = pred_df.query("smiles == @target_smiles and nuc_sites == @nuc_site")["MCA_values"].values[0]
            pred_val = pred_df.query("smiles == @target_smiles and nuc_sites == @nuc_site")["pred"].values[0]

            attn_from_target_by_layer = {
                layer_name: attn_agg[nuc_site, :]
                for layer_name, attn_agg in attn_agg_by_layer.items()
            }

            visualize_attention_on_structure_grid(
                mol,
                attn_from_target_by_layer,
                target_atom_idx=nuc_site,
                pred_val=pred_val,
                exp_val=exp_val,
                output_path=smiles_dir / f"site_{nuc_site}_all_layers.png",
            )
