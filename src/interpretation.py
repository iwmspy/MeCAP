import os
import pandas as pd
import numpy as np
from core_modules.model import build_model_from_checkpoint
from core_modules.dataset import  _resolve_model_dict_key
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from unimol_tools.data.conformer import coords2unimol
from unimol_tools.data import Dictionary
from unimol_tools.weights import WEIGHT_DIR, weight_download
from unimol_tools.config import MODEL_CONFIG
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import io
from rdkit.Chem import rdDetermineBonds


PWD = os.path.realpath(os.path.dirname(__file__))
ROOT_DIR = os.path.realpath(os.path.join(PWD, '..'))


def load_sdf(
    sdf_path:str
    ):
    dict_key = _resolve_model_dict_key(model_name="unimolv1", remove_hs=True)
    dict_path = os.path.join(WEIGHT_DIR, MODEL_CONFIG["dict"][dict_key])
    supp = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = supp[0]
    if mol is not None and mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
            dtype=np.float32,
        )
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        feat = coords2unimol(
            atoms, coords, dictionary=Dictionary.load(dict_path)
        )

    return coords, atoms, feat

def integrated_gradients(
    model,
    src_tokens,
    src_distance,
    src_coord,
    src_edge_type,
    atom_index,
    n_steps=200,
    baseline_method='zero'
):
    """
    Compute Integrated Gradients for the embedding x in interpretation_forward.

    Args:
        model: trained model
        src_tokens: token indices
        src_distance: distance matrix
        src_coord: coordinates
        src_edge_type: edge type matrix
        atom_index: target atom index
        n_steps: number of integration steps
        baseline_method: 'zero' or 'mean'

    Returns:
        attributions: attribution scores for each embedding dimension (B, N, D)
        atom_attributions: per-atom attribution scores (B, N) - L2 norm over embedding dim
    """
    model.eval()

    # Get original embedding
    with torch.no_grad():
        x_input = model.embed_tokens(src_tokens)

    # Create baseline
    if baseline_method == 'zero':
        baseline = torch.zeros_like(x_input)
    elif baseline_method == 'mask':
        with torch.no_grad():
            baseline = model.embed_tokens(torch.tensor([model.mask_idx] * src_tokens.shape[1], device=device).unsqueeze(0))
    else:
        raise ValueError(f"Unknown baseline_method: {baseline_method}")

    # Accumulate gradients
    accumulated_grads = torch.zeros_like(x_input)

    padding_mask = src_tokens.eq(model.padding_idx)
    if not padding_mask.any():
        padding_mask = None
    graph_attn_bias = model._dist_bias(src_distance, src_edge_type)

    for step in range(n_steps):
        alpha = (step + 1) / n_steps

        # Interpolate
        x_interp = baseline + alpha * (x_input - baseline)
        x_interp.requires_grad_(True)

        # Forward
        encoder_rep, _, _, _, _ = model.encoder(
            x_interp, padding_mask=padding_mask, attn_mask=graph_attn_bias
        )
        output = model._forward_head(encoder_rep, atom_index, None)
        pred = output['pred']

        # Backward
        model.zero_grad()
        if x_interp.grad is not None:
            x_interp.grad.zero_()

        # We want to maximize/attribute to the predicted value
        pred.sum().backward(retain_graph=True)

        # Accumulate
        if x_interp.grad is not None:
            accumulated_grads += x_interp.grad.detach()

    # Compute attributions
    avg_grads = accumulated_grads / n_steps
    attributions = (x_input - baseline) * avg_grads

    # Per-atom scores (L2 norm over embedding dimension)
    atom_attributions = torch.sqrt((attributions ** 2).sum(dim=-1))
    attributions = attributions.detach().cpu().numpy()
    atom_attributions = atom_attributions.detach().cpu().numpy()

    return attributions, atom_attributions

def visualize_atom_attributions(
    mol: Chem.Mol,
    atom_attributions: np.ndarray,
    nuc_site_idx: int,
    pred_val: float,
    exp_val: float,
    output_path: Optional[str] = None,
    img_size: Tuple[int, int] = (800, 600),
    figsize: Tuple[int, int] = (12, 6),
    vmin: float = 0.0,
    vmax: float = 0.4,
    ):
    """
    Visualize atom attributions as a heatmap on the molecular structure.

    Args:
        mol: Molecule
        atom_attributions: Attribution scores per atom (N,)
        nuc_site_idx: Atom index of the nucleophilic site
        pred_val: Predicted value
        exp_val: Experimental value
        output_path: Output file path (None to skip saving)
        img_size: Image size for RDKit drawing (w, h)
        figsize: matplotlib Figure size
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale

    Returns:
        matplotlib Figure
    """

    # --- Generate 2D coordinates ---
    mol_2d = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_2d)

    # --- Colormap（white to red）---
    cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"], N=256)
    atom_colors = {
        i: cmap(s)[:3] for i, s in enumerate(atom_attributions[0])
    }

    # --- Draw molecule by RDKit ---
    w, h = img_size
    drawer = Draw.MolDraw2DCairo(w, h)
    drawer.drawOptions().addAtomIndices = True
    drawer.DrawMolecule(
        mol_2d,
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightAtomColors=atom_colors,
        highlightBonds=[],
    )
    drawer.FinishDrawing()
    mol_img = Image.open(io.BytesIO(drawer.GetDrawingText()))

    # --- Generate figure  ---
    fig, (ax_mol, ax_cbar) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [5, 0.3]}
    )

    ax_mol.imshow(mol_img)
    ax_mol.axis("off")

    symbol = mol.GetAtomWithIdx(nuc_site_idx).GetSymbol()
    ax_mol.set_title(
        f"Integrated Gradients — Nuc Site: Atom {nuc_site_idx} ({symbol})\n"
        f"Experimental: {exp_val:.3f} kcal/mol, Predicted: {pred_val:.3f} kcal/mol",
        fontsize=14, fontweight="bold",
    )
    
    # --- Colorbar（vmin–vmax scale）---
    sm = cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    fig.colorbar(sm, cax=ax_cbar).set_label("Attribution Score", fontsize=11)

    fig.tight_layout()

    # --- Save ---
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig

def compute_and_visualize_attributions(
    sdf_path: str,
    atom_index: int,
    model,
    baseline_method: str = 'embed',
    normalize: bool = False,
    exp_val: Optional[float] = None,
    pred_val: Optional[float] = None,
    output_path: Optional[str] = None,
    device: str = 'cuda:0',
    n_steps: int = 200,
    ) -> plt.Figure:
    """
    Compute integrated gradients and visualize atom attributions.

    Args:
        sdf_path: Path to the SDF file
        atom_index: Target atom index for attribution
        model: Trained model
        baseline_method: Baseline method ('zero', 'mean', or 'embed')
        normalize: Whether to normalize attributions by L2 norm
        exp_val: Experimental value (optional)
        pred_val: Predicted value (optional)
        output_path: Output file path for saving the figure (optional)
        device: Device to use for computation
        n_steps: Number of integration steps

    Returns:
        matplotlib Figure object
    """
    # Load molecule and features
    coords, atoms, feat = load_sdf(sdf_path)
    
    # Prepare input tensors
    src_tokens = torch.tensor(feat['src_tokens'], device=device).unsqueeze(0)
    src_distance = torch.tensor(feat['src_distance'], device=device).unsqueeze(0)
    src_coord = torch.tensor(feat['src_coord'], device=device).unsqueeze(0)
    src_edge_type = torch.tensor(feat['src_edge_type'], device=device).unsqueeze(0)
    
    # Load molecule for visualization
    supp = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = supp[0]
    # Add bonds if molecule has no bonds
    if mol is not None and mol.GetNumBonds() == 0:
        mol_copy = Chem.Mol(mol)
        charge = [a.GetFormalCharge() for a in mol.GetAtoms()]
        total_charge = sum(charge)
        rdDetermineBonds.DetermineBonds(mol_copy, useHueckel=True, charge=total_charge)
        mol = mol_copy
    
    # Compute integrated gradients
    attributions, atom_attributions = integrated_gradients(
        model=model,
        src_tokens=src_tokens,
        src_distance=src_distance,
        src_coord=src_coord,
        src_edge_type=src_edge_type,
        atom_index=torch.tensor([atom_index], device=device),
        n_steps=n_steps,
        baseline_method=baseline_method,
    )
    
    # Normalize if requested
    if normalize:
        atom_attributions = atom_attributions / np.linalg.norm(atom_attributions)
    
    # Visualize attributions
    fig = visualize_atom_attributions(
        mol=mol,
        atom_attributions=atom_attributions,
        nuc_site_idx=atom_index,
        pred_val=pred_val if pred_val is not None else float('inf'),
        exp_val=exp_val if exp_val is not None else float('inf'),
        output_path=output_path,
    )
    
    return fig

if __name__ == '__main__':

    # Setting
    exp_df     = pd.read_csv(f'{ROOT_DIR}/data/references/QMdata4ML/df_nuc_x_sample.csv', index_col=0)
    pred_df    = pd.read_csv(f'{ROOT_DIR}/data/results/mecap_ref_mca_layer_0/predictions.csv', index_col=0)
    checkpoint = f'{ROOT_DIR}/data/results/mecap_ref_maa_layer_0/best_model.pt'
    # baseline_method = 'embed'
    # baseline_method = 'zero'
    baseline_method = 'mask'
    device     = 'cuda:0'

    targets_smiles = ['CN(C)C(=O)CCNC(=O)NCc1ccc(Br)cc1Cl', 'NOCc1cccc(I)c1']

    # Base output directory
    base_output_dir = f'{ROOT_DIR}/data/results'

    # Load model
    meta_state, model, device_ = build_model_from_checkpoint(checkpoint_path=checkpoint, device=device)

    # Usage
    for target_smiles in targets_smiles:
        target_name = list(set(exp_df.query('smiles == @target_smiles')['name']))[0]
        sdf_path = f'{ROOT_DIR}/data/results/confs_from_smiles_rdkit/{target_name}.sdf'
        
        nuc_sites = list(exp_df.query('smiles == @target_smiles')['nuc_sites'])
        smiles_dir = os.path.join(base_output_dir, 'interpretation', target_name, 'mca')
        os.makedirs(smiles_dir, exist_ok=True)
        
        for nuc_site in nuc_sites:
            exp_val = pred_df.query('smiles == @target_smiles and nuc_sites == @nuc_site')['MCA_values'].values[0]
            pred_val = pred_df.query('smiles == @target_smiles and nuc_sites == @nuc_site')['pred'].values[0]
            
            # Visualize non-normalized attributions
            compute_and_visualize_attributions(
                sdf_path=sdf_path,
                atom_index=nuc_site,
                model=model,
                baseline_method=baseline_method,
                normalize=False,
                exp_val=exp_val,
                pred_val=pred_val,
                output_path=f'{smiles_dir}/{baseline_method}/{nuc_site}.png',
                device=device,
            )
            
            # Visualize normalized attributions
            compute_and_visualize_attributions(
                sdf_path=sdf_path,
                atom_index=nuc_site,
                model=model,
                baseline_method=baseline_method,
                normalize=True,
                exp_val=exp_val,
                pred_val=pred_val,
                output_path=f'{smiles_dir}/{baseline_method}/{nuc_site}_norm.png',
                device=device,
            )

            