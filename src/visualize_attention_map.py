import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import io

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
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

        for layer_name, attn_weights in attn_weights_dict.items():

            attn_agg = attn_weights.mean(dim=0).cpu().numpy()

            smiles_dir = Path(
                base_output_dir, "interpretation", target_smiles,
                "attention", layer_name,
            )
            smiles_dir.mkdir(parents=True, exist_ok=True)

            # --- Heatmap (one per molecule per layer) ---
            visualize_attention_heatmap(
                attn_agg, atoms,
                title=f"Attention Heatmap — {layer_name} — {target_smiles}",
                output_path=smiles_dir / "heatmap.png",
            )

            # --- Structure overlay (one per nucleophilic site per layer) ---
            for nuc_site in nuc_sites:
                exp_val = pred_df.query("smiles == @target_smiles and nuc_sites == @nuc_site")["MCA_values"].values[0]
                pred_val = pred_df.query("smiles == @target_smiles and nuc_sites == @nuc_site")["pred"].values[0]

                attn_from_target = attn_agg[nuc_site, :]

                visualize_attention_on_structure(
                    mol,
                    attn_from_target,
                    target_atom_idx=nuc_site,
                    pred_val=pred_val,
                    exp_val=exp_val,
                    output_path=smiles_dir / f"{nuc_site}.png",
                )
