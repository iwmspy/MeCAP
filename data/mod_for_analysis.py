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

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))

from typing import Iterable, Optional, Tuple, Union, Mapping, Any, Dict, Set, List
from numbers import Integral
import ast, io, math, os, re
from collections import deque
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D
from PIL import Image, ImageDraw, ImageFont

from core_modules.atom import find_sites, extracted_n_smirks_dict, extracted_e_smirks_dict, added_n_smirks_dict, added_e_smirks_dict

s_params = Chem.SubstructMatchParameters()
s_params.numThreads = min(8, os.cpu_count())
s_params.uniquify = False
thu_aln_smirks_dict = {'thiourea': '[N;X3;H0,H1,H2:2]-[C;X3;v4:1](=[S;X1;v2:3])-[N;X3;H0,H1,H2:4]',
                       'allene': '[C;X2;v4:1](=[!#1:2])=[!#1:3]',
                      }
element_dict        = {'thiourea': 1,
                       'allene': 1,
                      }
substructure_match = lambda mol, submol: mol.GetSubstructMatches(submol,s_params)


def determine_font_size(base_font_size=14, title_scale=1.3, label_scale=1.1, subtitle_scale=1.05):
    ticks_font_size = int(base_font_size)
    title_font_size = int(base_font_size * title_scale)
    label_font_size = int(base_font_size * label_scale)
    subtitle_font_size = int(base_font_size * subtitle_scale)
    return base_font_size, ticks_font_size, title_font_size, label_font_size, subtitle_font_size

# 2D-binning setting
N_BINS_SIM = 10
N_BINS_MDIST = 10
MIN_CELL_N = 200
HEATMAP_CMAP = "viridis"
MASK_COLOR = "lightgray"




def load_csv(path):
    return pd.read_csv(path, index_col=0)

def load_pred_value(path: str):
    l = []
    with open(path, 'r') as f:
        for line in f.readlines():
            l.append(float(line))
    return l

def calculate_stats(x,y):
    r, p = pearsonr(x, y)
    r2 = r2_score(x,y)
    rmse = np.sqrt(np.mean((x - y) ** 2))
    return r, r2, rmse

def stats_and_plot(x, y, hue=None, ax=None, x_label=None, y_label=None, title=None, alpha=0.3, visualize_stats=True):
    """Compute Pearson r, RMSE, and generate a scatter plot."""
    r, r2, rmse = calculate_stats(x, y)

    if ax is None:
        plot, ax = plt.subplots(figsize=(5,5))

    if hue is not None:
        hue_order = sorted(np.unique(hue).tolist())
        sns.scatterplot(x=x, y=y, hue=hue, hue_order=hue_order, alpha=alpha, ax=ax, palette="deep")
    else:
        ax.scatter(x, y, alpha=alpha)

    # Reference y=x line
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "--", alpha=0.7)
    ax.set_xlim(lo-50, hi+50)
    ax.set_ylim(lo-50, hi+50)
    ax.set_xlabel("$x$" if x_label is None else x_label)
    ax.set_ylabel("$y$" if y_label is None else y_label)
    if visualize_stats: title = f"{title}\n$r$={r:.3f}, R2={r2:.3f}, RMSE={rmse:.3f}"
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.4)
    return ax, r, r2, rmse

def _pca_angle_deg_xy(coords):
    """Return angle (deg) of the first principal axis w.r.t. +x, for 2D coords."""
    # coords: list[(x,y)]
    cx = sum(x for x,_ in coords)/len(coords)
    cy = sum(y for _,y in coords)/len(coords)
    xs = [x-cx for x,_ in coords]
    ys = [y-cy for _,y in coords]
    # 2x2 covariance matrix
    sxx = sum(x*x for x in xs); syy = sum(y*y for y in ys); sxy = sum(x*y for x,y in zip(xs,ys))
    # eigenvector for largest eigenvalue of [[sxx,sxy],[sxy,syy]]
    # Closed-form using atan2 of the eigenvector direction
    # tan(2θ) = 2sxy / (sxx - syy)
    theta = 0.5*math.atan2(2*sxy, (sxx - syy))  # radians
    return math.degrees(theta)

def _rotate_mol2d_inplace(mol, angle_deg, about_centroid=True):
    """Rotate 2D conformer positions (z kept) around centroid by angle_deg."""
    if mol.GetNumConformers() == 0:
        return
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    pts = [conf.GetAtomPosition(i) for i in range(n)]
    if about_centroid:
        cx = sum(p.x for p in pts)/n
        cy = sum(p.y for p in pts)/n
    else:
        cx = cy = 0.0
    ang = math.radians(angle_deg)
    c, s = math.cos(ang), math.sin(ang)
    for i, p in enumerate(pts):
        x, y = p.x - cx, p.y - cy
        xr =  x*c - y*s
        yr =  x*s + y*c
        conf.SetAtomPosition(i, Point3D(xr+cx, yr+cy, p.z))

def visualize_smiles_highlight(
    smiles: str,
    atom_indices: Union[int, Iterable[int]],
    qm_value: float,
    pred_value: float,
    image_size: Tuple[int, int] = (500, 350),
    caption_fmt: str = "QM-based MCA: {} [kJ/mol]\nPredicted MCA: {} [kJ/mol]",
    caption_extra: Optional[Union[str, Mapping[str, object], Iterable[Tuple[str, object]]]] = None,
    highlight_color: Tuple[int, int, int, int] = (242, 64, 64, 180),
    highlight_radius_px: int = 16,
    highlight_thickness_px: int = 6,
    save_path: Optional[str] = None,
    return_ignored: bool = False,
    auto_rotate: bool = True,
    rotate_deg: float = 0.0,
    rdkit_padding: float = 0.02,   # smaller = tighter fit; default RDKit ≈ 0.05
    font_size = 18,
):
    """
    Draw a SMILES with highlighted atoms, optional PCA-based auto-rotation, and a caption.

    - auto_rotate=True: aligns the longest principal axis to best match canvas aspect
                        (portrait canvas → make molecule tall; landscape → make it wide).
    - rotate_deg: additional manual rotation (degrees, applied after auto-rotation).
    - rdkit_padding: shrink this to make the molecule larger on the canvas.
    """
    # Parse + 2D coords
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)

    # Normalize indices
    if isinstance(atom_indices, int):
        idx_list = [atom_indices]
    else:
        idx_list = [int(i) for i in atom_indices]
    n_atoms = mol.GetNumAtoms()
    valid = [i for i in idx_list if 0 <= i < n_atoms]
    ignored = [i for i in idx_list if not (0 <= i < n_atoms)]

    # Auto-rotate to “nice” orientation using PCA
    if auto_rotate and n_atoms >= 2 and mol.GetNumConformers():
        conf = mol.GetConformer()
        coords = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y) for i in range(n_atoms)]
        angle_pca = _pca_angle_deg_xy(coords)  # angle vs +x
        W, H = image_size
        # Make the longest axis align with the longer side of the canvas
        # If canvas is portrait, we want the longest axis vertical (≈ +90° from +x).
        target = 90.0 if H >= W else 0.0
        _rotate_mol2d_inplace(mol, target - angle_pca)

    # Optional extra manual rotation
    if abs(rotate_deg) > 1e-6:
        _rotate_mol2d_inplace(mol, rotate_deg)

    # RDKit drawing (tight padding to avoid tiny molecules)
    w, h = image_size
    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    opts.padding = rdkit_padding       # tighter fit -> molecule appears larger
    opts.bondLineWidth = 2
    opts.addStereoAnnotation = False

    atom_colors = {i: (1.0, 0.2, 0.2) for i in valid} if valid else None
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=valid if valid else None,
        highlightAtomColors=atom_colors,
    )
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    base = Image.open(io.BytesIO(png)).convert("RGBA")

    # Pixel-precise highlight circles
    if valid:
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for i in valid:
            x, y = drawer.GetDrawCoords(i)
            l = x - highlight_radius_px; t = y - highlight_radius_px
            r = x + highlight_radius_px; b = y + highlight_radius_px
            d.ellipse([l, t, r, b], outline=highlight_color, width=highlight_thickness_px)
            fill_col = (highlight_color[0], highlight_color[1], highlight_color[2],
                        max(60, highlight_color[3] // 3))
            d.ellipse([l+2, t+2, r-2, b-2], fill=fill_col)
        base = Image.alpha_composite(base, overlay)

    # Caption
    caption = caption_fmt.format(qm_value, pred_value)
    if caption_extra:
        if isinstance(caption_extra, str):
            caption += "\n" + caption_extra
        elif isinstance(caption_extra, Mapping):
            caption += "\n" + "\n".join(f"{k}: {v}" for k, v in caption_extra.items())
        else:
            caption += "\n" + "\n".join(f"{k}: {v}" for k, v in caption_extra)

    pad = 10
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    tmp = ImageDraw.Draw(base)
    try:
        bbox = tmp.multiline_textbbox((0, 0), caption, font=font, spacing=2)
        text_w = bbox[2] - bbox[0]; text_h = bbox[3] - bbox[1]
    except AttributeError:
        lines = caption.split("\n")
        sizes = [tmp.textsize(line, font=font) for line in lines]
        text_w = max(w for w, _ in sizes); text_h = sum(h for _, h in sizes) + 2*(len(lines)-1)

    cap_w = max(base.width, text_w + 2*pad)
    cap_h = text_h + 2*pad
    final_img = Image.new("RGBA", (cap_w, base.height + cap_h), (255, 255, 255, 255))
    final_img.paste(base, ((cap_w - base.width)//2, 0), base)
    draw = ImageDraw.Draw(final_img)
    draw.multiline_text(((cap_w - text_w)//2, base.height + pad), caption, fill=(0,0,0), font=font, align="center", spacing=2)

    if save_path:
        final_img.convert("RGB").save(save_path)
    return (final_img, ignored) if return_ignored else final_img


def _normalize_atom_indices(
    atom_indices: Optional[Iterable[int]],
    n_atoms: int,
) -> Tuple[List[int], List[int]]:
    """Split atom indices into valid and ignored lists."""
    if atom_indices is None:
        return [], []
    valid, ignored = [], []
    for idx in atom_indices:
        idx = int(idx)
        if 0 <= idx < n_atoms:
            valid.append(idx)
        else:
            ignored.append(idx)
    return sorted(set(valid)), sorted(set(ignored))


def _normalize_reactant_atom_indices(
    atom_indices: Optional[Iterable[Any]],
    reactant_templates: List[Chem.Mol],
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Normalize atom indices for reactants.

    Supports two formats:
      1) flat iterable of global indices across all reactants
      2) iterable of iterables, where each sub-iterable contains local indices
         for the corresponding reactant template
    """
    n_reactants = len(reactant_templates)
    valid_by_reactant = [[] for _ in range(n_reactants)]
    ignored_by_reactant = [[] for _ in range(n_reactants)]
    if atom_indices is None:
        return valid_by_reactant, ignored_by_reactant

    raw_items = list(atom_indices)
    if not raw_items:
        return valid_by_reactant, ignored_by_reactant

    is_nested = any(not isinstance(item, Integral) for item in raw_items)
    if is_nested:
        if len(raw_items) > n_reactants:
            raise ValueError(
                f"Received indices for {len(raw_items)} reactants, but the reaction has "
                f"{n_reactants} reactant templates."
            )
        for reactant_idx, local_indices in enumerate(raw_items):
            valid, ignored = _normalize_atom_indices(
                local_indices, reactant_templates[reactant_idx].GetNumAtoms()
            )
            valid_by_reactant[reactant_idx] = valid
            ignored_by_reactant[reactant_idx] = ignored
        return valid_by_reactant, ignored_by_reactant

    atom_offset = 0
    global_valid, global_ignored = _normalize_atom_indices(
        [int(idx) for idx in raw_items],
        sum(mol.GetNumAtoms() for mol in reactant_templates),
    )
    for reactant_idx, reactant_template in enumerate(reactant_templates):
        next_offset = atom_offset + reactant_template.GetNumAtoms()
        valid_by_reactant[reactant_idx] = sorted(
            idx - atom_offset for idx in global_valid if atom_offset <= idx < next_offset
        )
        ignored_by_reactant[reactant_idx] = []
        atom_offset = next_offset
    if global_ignored:
        ignored_by_reactant[0] = global_ignored
    return valid_by_reactant, ignored_by_reactant


def _copy_mol_with_atom_index_notes(mol: Chem.Mol) -> Chem.Mol:
    """Show atom indices as atomNote labels and clear map-number display props."""
    out = Chem.Mol(mol)
    _set_atom_index_notes_inplace(out)
    return out


def _set_atom_index_notes_inplace(
    mol: Chem.Mol,
    annotated_atom_indices: Optional[Iterable[int]] = None,
) -> None:
    """Show selected atom indices as atomNote labels directly on the molecule."""
    annotated_set = None if annotated_atom_indices is None else set(int(idx) for idx in annotated_atom_indices)
    for atom in mol.GetAtoms():
        if atom.HasProp("atomNote"):
            atom.ClearProp("atomNote")
        if annotated_set is None or atom.GetIdx() in annotated_set:
            atom.SetProp("atomNote", str(atom.GetIdx()))
        if atom.GetAtomMapNum():
            atom.SetAtomMapNum(0)
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")


def _prepare_mol_for_panel(
    mol: Chem.Mol,
    annotated_atom_indices: Optional[Iterable[int]] = None,
    auto_rotate: bool = True,
    rotate_deg: float = 0.0,
) -> Chem.Mol:
    """Copy molecule, add selected atom-index notes, and compute 2D coordinates."""
    out = Chem.Mol(mol)
    _set_atom_index_notes_inplace(out, annotated_atom_indices=annotated_atom_indices)
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(out)
    if auto_rotate and out.GetNumAtoms() >= 2 and out.GetNumConformers():
        conf = out.GetConformer()
        coords = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y) for i in range(out.GetNumAtoms())]
        angle_pca = _pca_angle_deg_xy(coords)
        _rotate_mol2d_inplace(out, -angle_pca)
    if abs(rotate_deg) > 1e-6:
        _rotate_mol2d_inplace(out, rotate_deg)
    return out


def _estimate_panel_size(
    mol: Chem.Mol,
    fixed_bond_length: float,
    min_size: Tuple[int, int] = (140, 130),
    max_size: Tuple[int, int] = (420, 300),
    margin_px: int = 70,
) -> Tuple[int, int]:
    """Estimate a natural panel size from the 2D coordinate span."""
    if not mol.GetNumConformers():
        return min_size
    conf = mol.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
    x_span = max(xs) - min(xs) if xs else 0.0
    y_span = max(ys) - min(ys) if ys else 0.0
    width = int(max(min_size[0], min(max_size[0], x_span * fixed_bond_length + 2 * margin_px)))
    height = int(max(min_size[1], min(max_size[1], y_span * fixed_bond_length + 2 * margin_px)))
    return width, height


def _draw_molecule_panel(
    mol: Chem.Mol,
    filled_atoms: Optional[Dict[int, Tuple[float, float, float]]] = None,
    ring_atoms: Optional[Iterable[int]] = None,
    annotated_atom_indices: Optional[Iterable[int]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    fixed_bond_length: float = 32.0,
    auto_rotate: bool = True,
    rotate_deg: float = 0.0,
    atom_note_font_scale: float = 0.72,
    fill_radius: float = 0.42,
    ring_radius_px: int = 18,
    ring_line_width_px: int = 4,
    ring_color: Tuple[int, int, int, int] = (220, 20, 60, 255),
    padding: float = 0.08,
):
    """Draw a single molecule with filled highlights and red outline circles."""
    mol = _prepare_mol_for_panel(
        mol,
        annotated_atom_indices=annotated_atom_indices,
        auto_rotate=auto_rotate,
        rotate_deg=rotate_deg,
    )
    if image_size is None:
        w, h = _estimate_panel_size(mol, fixed_bond_length=fixed_bond_length)
    else:
        w, h = image_size
    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    opts.padding = padding
    opts.bondLineWidth = 2
    opts.addStereoAnnotation = False
    opts.atomHighlightsAreCircles = True
    opts.fillHighlights = True
    opts.annotationFontScale = atom_note_font_scale
    opts.fixedBondLength = fixed_bond_length

    filled_atoms = filled_atoms or {}
    ring_atoms = sorted(set(int(idx) for idx in (ring_atoms or [])))
    highlight_atoms = sorted(filled_atoms)
    highlight_radii = {idx: fill_radius for idx in highlight_atoms}

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms if highlight_atoms else None,
        highlightAtomColors=filled_atoms if filled_atoms else None,
        highlightAtomRadii=highlight_radii if highlight_atoms else None,
    )
    drawer.FinishDrawing()

    image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    if not ring_atoms:
        return image

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for atom_idx in ring_atoms:
        x, y = drawer.GetDrawCoords(atom_idx)
        draw.ellipse(
            [
                x - ring_radius_px,
                y - ring_radius_px,
                x + ring_radius_px,
                y + ring_radius_px,
            ],
            outline=ring_color,
            width=ring_line_width_px,
        )
    return Image.alpha_composite(image, overlay)


def _join_images_horizontally(
    images: List[Image.Image],
    separator: str = "+",
    gap_px: int = 18,
    font_size: int = 30,
    margin_px: int = 10,
) -> Image.Image:
    """Join images with a text separator."""
    if not images:
        return Image.new("RGBA", (1, 1), (255, 255, 255, 0))

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1), (255, 255, 255, 0)))
    bbox = probe.textbbox((0, 0), separator, font=font)
    sep_w = bbox[2] - bbox[0]
    sep_h = bbox[3] - bbox[1]

    total_w = margin_px * 2 + sum(img.width for img in images)
    total_w += gap_px * max(0, len(images) - 1)
    total_w += sep_w * max(0, len(images) - 1)
    total_h = margin_px * 2 + max(max(img.height for img in images), sep_h)

    canvas = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x = margin_px
    for i, img in enumerate(images):
        y = (total_h - img.height) // 2
        canvas.paste(img, (x, y), img)
        x += img.width
        if i < len(images) - 1:
            x += gap_px
            text_y = (total_h - sep_h) // 2
            draw.text((x, text_y), separator, fill=(0, 0, 0), font=font)
            x += sep_w + gap_px
    return canvas


def _resize_to_fixed_height(image: Image.Image, target_height: int) -> Image.Image:
    """Resize an image to the target height while keeping aspect ratio."""
    if image.height == target_height:
        return image
    scale = target_height / image.height
    target_width = max(1, int(round(image.width * scale)))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def visualize_reaction_smiles_highlight(
    reaction_smiles: str,
    reaction_center_atom_indices: Optional[Iterable[Any]] = None,
    mca_atom_indices: Optional[Iterable[Any]] = None,
    maa_atom_indices: Optional[Iterable[Any]] = None,
    overlap_fill_color: Tuple[float, float, float] = (0.45, 0.80, 0.45),
    output_size: Tuple[int, int] = (1000, 260),
    fixed_bond_length: float = 32.0,
    auto_rotate: bool = True,
    rotate_deg: float = 0.0,
    annotate_all_atoms: bool = False,
    atom_note_font_scale: float = 0.72,
    save_path: Optional[str] = None,
    return_ignored: bool = False,
):
    """
    Visualize a reaction SMILES with reactant-side atom highlights.

    Atom indices can be given in either of these formats:
      - per-reactant local indices, e.g. [[0, 2], [1], []]
      - a flat iterable of global indices across the concatenated reactants
        (backward-compatible behavior)

    Highlight rules on reactants:
      - `reaction_center_atom_indices`: red outline circles
      - `mca_atom_indices`: orange filled circles
      - `maa_atom_indices`: light-blue filled circles
      - atoms in both `mca_atom_indices` and `maa_atom_indices`:
        `overlap_fill_color`

    The red outline is added independently when requested.
    `output_size` fixes the final reaction image size, while each molecule panel
    is sized adaptively before the whole layout is scaled into that canvas.
    By default, atom indices are shown only for highlighted reactant atoms.
    `atom_note_font_scale` controls atom-index label size.
    Molecules can be auto-rotated by principal axis, with optional extra `rotate_deg`.
    """
    parts = reaction_smiles.split(">")
    if len(parts) != 3:
        raise ValueError("reaction_smiles must have the form 'reactants>agents>products'.")

    rxn = rdChemReactions.ReactionFromSmarts(reaction_smiles, useSmiles=True)
    if rxn is None:
        raise ValueError("Failed to parse reaction_smiles as a reaction.")
    reactant_templates = [
        rxn.GetReactantTemplate(i) for i in range(rxn.GetNumReactantTemplates())
    ]
    if not reactant_templates:
        raise ValueError("The reaction does not contain reactant templates.")

    reaction_center_valid, reaction_center_ignored = _normalize_reactant_atom_indices(
        reaction_center_atom_indices, reactant_templates
    )
    mca_valid, mca_ignored = _normalize_reactant_atom_indices(
        mca_atom_indices, reactant_templates
    )
    maa_valid, maa_ignored = _normalize_reactant_atom_indices(
        maa_atom_indices, reactant_templates
    )

    reactant_images = []
    for reactant_idx, reactant_template in enumerate(reactant_templates):
        reaction_center_set = set(reaction_center_valid[reactant_idx])
        mca_set = set(mca_valid[reactant_idx])
        maa_set = set(maa_valid[reactant_idx])
        overlap_set = mca_set & maa_set
        local_fill_colors = {
            idx: (1.0, 0.65, 0.0) for idx in (mca_set - overlap_set)
        }
        local_fill_colors.update({
            idx: (0.53, 0.86, 0.98) for idx in (maa_set - overlap_set)
        })
        local_fill_colors.update({idx: overlap_fill_color for idx in overlap_set})
        local_ring_atoms = sorted(reaction_center_set)
        annotated_atom_indices = None if annotate_all_atoms else sorted(
            reaction_center_set | mca_set | maa_set
        )

        reactant_images.append(
            _draw_molecule_panel(
                reactant_template,
                filled_atoms=local_fill_colors,
                ring_atoms=local_ring_atoms,
                annotated_atom_indices=annotated_atom_indices,
                fixed_bond_length=fixed_bond_length,
                auto_rotate=auto_rotate,
                rotate_deg=rotate_deg,
                atom_note_font_scale=atom_note_font_scale,
            )
        )

    reactant_panel = _join_images_horizontally(reactant_images, separator="+")

    product_images = [
        _draw_molecule_panel(
            rxn.GetProductTemplate(i),
            annotated_atom_indices=None if annotate_all_atoms else [],
            fixed_bond_length=fixed_bond_length,
            auto_rotate=auto_rotate,
            rotate_deg=rotate_deg,
            atom_note_font_scale=atom_note_font_scale,
        )
        for i in range(rxn.GetNumProductTemplates())
    ]
    product_panel = _join_images_horizontally(product_images, separator="+")

    agent_images = [
        _draw_molecule_panel(
            rxn.GetAgentTemplate(i),
            annotated_atom_indices=None if annotate_all_atoms else [],
            fixed_bond_length=fixed_bond_length,
            auto_rotate=auto_rotate,
            rotate_deg=rotate_deg,
            atom_note_font_scale=atom_note_font_scale,
        )
        for i in range(rxn.GetNumAgentTemplates())
    ]
    agent_panel = _join_images_horizontally(agent_images, separator="+") if agent_images else None

    target_w, target_h = output_size
    arrow_w = max(90, int(target_w * 0.10))
    top_pad = 10
    agent_h = 0 if agent_panel is None else agent_panel.height + 12
    content_h = max(reactant_panel.height, product_panel.height)
    canvas_w = reactant_panel.width + arrow_w + product_panel.width + 40
    canvas_h = content_h + agent_h + 20
    final_img = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    react_y = agent_h + (content_h - reactant_panel.height) // 2 + top_pad
    prod_y = agent_h + (content_h - product_panel.height) // 2 + top_pad
    react_x = 10
    prod_x = react_x + reactant_panel.width + arrow_w
    final_img.paste(reactant_panel, (react_x, react_y), reactant_panel)
    final_img.paste(product_panel, (prod_x, prod_y), product_panel)

    draw = ImageDraw.Draw(final_img)
    arrow_mid_x0 = react_x + reactant_panel.width + 20
    arrow_mid_x1 = prod_x - 20
    arrow_y = agent_h + content_h // 2 + top_pad
    draw.line((arrow_mid_x0, arrow_y, arrow_mid_x1, arrow_y), fill=(0, 0, 0), width=4)
    draw.polygon(
        [
            (arrow_mid_x1, arrow_y),
            (arrow_mid_x1 - 16, arrow_y - 8),
            (arrow_mid_x1 - 16, arrow_y + 8),
        ],
        fill=(0, 0, 0),
    )

    if agent_panel is not None:
        agent_x = react_x + reactant_panel.width + (arrow_w - agent_panel.width) // 2
        final_img.paste(agent_panel, (agent_x, 0), agent_panel)

    if final_img.height != target_h:
        final_img = _resize_to_fixed_height(final_img, target_h)
    if final_img.width > target_w:
        scale = target_w / final_img.width
        resized_h = max(1, int(round(final_img.height * scale)))
        final_img = final_img.resize((target_w, resized_h), Image.Resampling.LANCZOS)
    if final_img.height > target_h:
        final_img = _resize_to_fixed_height(final_img, target_h)

    canvas = Image.new("RGBA", output_size, (255, 255, 255, 255))
    paste_x = max(0, (target_w - final_img.width) // 2)
    paste_y = max(0, (target_h - final_img.height) // 2)
    canvas.paste(final_img, (paste_x, paste_y), final_img)
    final_img = canvas

    if save_path:
        final_img.convert("RGB").save(save_path)

    if return_ignored:
        ignored = {
            "reaction_center_atom_indices": reaction_center_ignored,
            "mca_atom_indices": mca_ignored,
            "maa_atom_indices": maa_ignored,
        }
        return final_img, ignored
    return final_img

def extract_best_epoch(log_path: str) -> int:
    """
    Parse a training log file and return the best epoch as an integer.

    Priority:
      1) If a line like 'Test@best_epoch=44' exists, return that number.
      2) Otherwise, use the last occurrence of 'Saved best model at epoch N'.
    Raises:
      ValueError: if no best-epoch information can be found.
    """
    # Read the whole file once
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1) Prefer explicit 'Test@best_epoch=...'
    m = re.search(r"Test@best_epoch\s*=\s*(\d+)", text)
    if m:
        return int(m.group(1))

    # 2) Fallback to last 'Saved best model at epoch ...'
    matches = re.findall(r"Saved best model at epoch\s+(\d+)", text)
    if matches:
        return int(matches[-1])

    # If neither pattern is found, raise an informative error
    raise ValueError(
        "Could not determine best epoch. "
        "Expected 'Test@best_epoch=...' or 'Saved best model at epoch ...' in the log."
    )

def calculate_kendall_correlation_by_group(df, group_col, col1, col2):
    """
    Calculate Kendall's tau correlation for each group in the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    group_col : str
        Column name to group by
    col1 : str
        First column for correlation calculation
    col2 : str
        Second column for correlation calculation
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with group names, tau values, and p-values
    """
    results = []
    
    for group_name, group_df in df.groupby(group_col):
        # Remove NaN values
        valid_data = group_df[[col1, col2]].dropna()
        
        if len(valid_data) > 1:  # Need at least 2 points for correlation
            tau, p_value = kendalltau(valid_data[col1], valid_data[col2])
            max_index_match = pd.to_numeric(valid_data[col1], errors='coerce').idxmax() == pd.to_numeric(valid_data[col2], errors='coerce').idxmax()
            results.append({
            group_col: group_name,
            'tau': tau,
            'p_value': p_value,
            'n': len(valid_data),
            'max_index_match': max_index_match,
            })
        else:
            results.append({
            group_col: group_name,
            'tau': None,
            'p_value': None,
            'n': len(valid_data),
            'max_index_match': None,
            })
    res_df = pd.DataFrame(results)
    return res_df.set_index(group_col,drop=True)


def find_additional_nuc_sites(rdkit_mol):
    return find_sites(rdkit_mol, (extracted_n_smirks_dict | added_n_smirks_dict))


def find_additional_elec_sites(rdkit_mol):
    return find_sites(rdkit_mol, (extracted_e_smirks_dict | added_e_smirks_dict))


def _load_first_mol_from_sdf(sdf_path: Path, removeHs: bool = False) -> Optional[Chem.Mol]:
    """Load the first valid molecule from an SDF file. Return None on failure."""
    try:
        if not sdf_path.exists():
            return None
        supp = Chem.SDMolSupplier(str(sdf_path), removeHs=removeHs, sanitize=False)
        if supp is None:
            return None
        mols = [m for m in supp if m is not None]
        if not mols:
            return None
        mol = mols[0]
        if mol.GetNumConformers() == 0:
            return None
        return mol
    except Exception:
        return None


def _get_graph_radius_atom_indices(mol: Chem.Mol, center_idx: int, radius: int) -> list[int]:
    """
    Return atom indices within the given graph radius from center_idx.
    The center atom itself is included.
    """
    n_atoms = mol.GetNumAtoms()
    if center_idx < 0 or center_idx >= n_atoms:
        raise IndexError(f"atom_index {center_idx} is out of range for molecule with {n_atoms} atoms")

    if mol.GetNumBonds()==0:
        rdDetermineBonds.DetermineBonds(mol)

    visited = {center_idx}
    q = deque([(center_idx, 0)])

    while q:
        atom_idx, dist = q.popleft()
        if dist == radius:
            continue

        atom = mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in visited:
                visited.add(nbr_idx)
                q.append((nbr_idx, dist + 1))

    return sorted(visited)


def _pairwise_distance_change_for_local_region(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    atom_indices: list[int],
) -> Optional[float]:
    """
    Compute sqrt(mean((d_uv^A - d_uv^B)^2)) over all atom pairs
    within the local atom set.
    """
    if mol_a.GetNumAtoms() != mol_b.GetNumAtoms():
        return None    

    conf_a = mol_a.GetConformer()
    conf_b = mol_b.GetConformer()

    # Return 0.0 when the local set contains fewer than two atoms.
    if len(atom_indices) < 2:
        return 0.0

    sq_diffs = []
    for u, v in combinations(atom_indices, 2):
        pa_u = conf_a.GetAtomPosition(u)
        pa_v = conf_a.GetAtomPosition(v)
        pb_u = conf_b.GetAtomPosition(u)
        pb_v = conf_b.GetAtomPosition(v)

        d_a = ((pa_u.x - pa_v.x) ** 2 + (pa_u.y - pa_v.y) ** 2 + (pa_u.z - pa_v.z) ** 2) ** 0.5
        d_b = ((pb_u.x - pb_v.x) ** 2 + (pb_u.y - pb_v.y) ** 2 + (pb_u.z - pb_v.z) ** 2) ** 0.5
        sq_diffs.append((d_a - d_b) ** 2)

    return float(np.sqrt(np.mean(sq_diffs)))


def compute_local_pairwise_distance_change(
    df: pd.DataFrame,
    dir_a: str | Path,
    dir_b: str | Path,
    radius: int = 1,
    name_col: str = "name",
    atom_index_col: str = "atom_index",
    result_col: str = "local_pairwise_distance_change",
    removeHs: bool = False,
) -> pd.DataFrame:
    """
    For each (name, atom_index) pair in the DataFrame, load {name}.sdf
    from two directories and compute the local alignment-free
    pairwise-distance change.

    The metric is:
        G_i = sqrt( (1 / |P_i|) * sum_{(u,v) in P_i} (d_uv^(A) - d_uv^(B))^2 )

    where:
        - N_i is the set of atoms within the specified graph radius from atom i
        - P_i is the set of all unordered atom pairs in N_i
        - d_uv^(A), d_uv^(B) are Euclidean distances in structures A and B

    Returns a copy of df with an added result column.
    If a file is missing or any error occurs for a row, the result is None.
    """
    if radius not in (1, 2):
        raise ValueError("radius must be 1 or 2")

    dir_a = Path(dir_a)
    dir_b = Path(dir_b)

    out_df = df.copy()

    # Cache molecules to avoid reloading the same SDF multiple times.
    mol_cache_a: dict[str, Optional[Chem.Mol]] = {}
    mol_cache_b: dict[str, Optional[Chem.Mol]] = {}

    results = []

    for _, row in tqdm(out_df.iterrows(),total=out_df.shape[0]):
        name = str(row[name_col])
        atom_index = int(row[atom_index_col])

        if name not in mol_cache_a:
            mol_cache_a[name] = _load_first_mol_from_sdf(dir_a / f"{name}.sdf", removeHs)
        if name not in mol_cache_b:
            mol_cache_b[name] = _load_first_mol_from_sdf(dir_b / f"{name}.sdf", removeHs)

        mol_a = mol_cache_a[name]
        mol_b = mol_cache_b[name]

        # Return None if either file is missing or cannot be parsed.
        if mol_a is None or mol_b is None:
            results.append(None)
            continue

        try:
            local_atom_indices = _get_graph_radius_atom_indices(mol_a, atom_index, radius)

            # This assumes the atom ordering is consistent between the two SDF files.
            value = _pairwise_distance_change_for_local_region(mol_a, mol_b, local_atom_indices)
            results.append(value)
        except Exception:
            results.append(None)

    out_df[result_col] = results
    return out_df

def _prepare_abs_error(df):
    true_val = 'MCA_values' if 'MCA_values' in df.columns else 'MAA_values'
    df = df.copy()
    df['abs_error_mmff'] = (df['mmff'] - df[true_val]).abs() - (df['esnuel_orca'] - df[true_val]).abs()
    df['abs_error_rmsd'] = (df['rmsd'] - df[true_val]).abs() - (df['esnuel_orca'] - df[true_val]).abs()
    return df

def _interval_label(interval):
    return f'{abs(interval.left):.3f}-{abs(interval.right):.3f}'

def _summarize_heatmap(df, name_col, x_col, y_col, top_names, n_bins=5, percentile=0.50):
    d = df[df[name_col].isin(top_names)][[name_col, x_col, y_col]].dropna().copy()
    if d.empty:
        empty = pd.DataFrame(index=top_names)
        return empty, empty
    d['bin'] = pd.qcut(d[x_col], q=n_bins, duplicates='drop')
    summary = (
        d.groupby([name_col, 'bin'], observed=True)
        .agg(p=(y_col, lambda s: s.quantile(percentile)), count=(y_col, 'size'))
        .reset_index()
    )
    heatmap = summary.pivot(index=name_col, columns='bin', values='p').reindex(top_names)
    counts = summary.pivot(index=name_col, columns='bin', values='count').reindex(top_names)
    labels = [_interval_label(iv) for iv in heatmap.columns]
    heatmap.columns = labels
    counts.columns = labels
    return heatmap, counts

def _plot_heatmap(ax, heatmap, counts, title, row_label='', show_xticks=False, show_yticks=False, ytick_pad=2, vmin=None, vmax=None):
    base_font_size, ticks_font_size, title_font_size, label_font_size, subtitle_font_size = \
        determine_font_size(16, title_scale=1.4)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('lightgray')
    annot = counts.applymap(lambda v: '' if pd.isna(v) else f'{int(v)}')
    sns.heatmap(
        heatmap,
        ax=ax,
        annot=annot,
        fmt='',
        annot_kws={'fontsize': max(1, ticks_font_size)},
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        linewidths=0.4,
        linecolor='white',
        mask=heatmap.isna(),
    )
    ax.set_title(title, fontsize=title_font_size)
    ax.set_xlabel('' )
    ax.set_ylabel(row_label if show_yticks else '', fontsize=label_font_size, fontweight='bold', rotation=90, labelpad=18)
    ax.tick_params(axis='both', labelsize=ticks_font_size, length=0)
    if show_xticks:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        ax.tick_params(axis='x', labeltop=False, labelbottom=True, pad=2)
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.set_xticklabels([])
    if show_yticks:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')
        ax.yaxis.tick_left()
        ax.tick_params(axis='y', labelleft=True, labelright=False, pad=ytick_pad)
    else:
        ax.set_yticklabels([])

def _trimmed_mean(x, trim=0.1):
    if len(x) == 0:
        return np.nan
    lo, hi = x.quantile(trim), x.quantile(1 - trim)
    return x[(x >= lo) & (x <= hi)].mean()


def _format_interval_labels(interval_index, fmt="{:.3f}"):
    """
    Convert pandas IntervalIndex/Categorical interval categories to readable strings.
    Example: (0.123, 0.456] -> "0.123–0.456"
    """
    labels = []
    for iv in interval_index:
        if pd.isna(iv):
            labels.append("")
            continue
        # iv is a pandas Interval
        labels.append(f"{fmt.format(iv.left)}–{fmt.format(iv.right)}")
    return labels


def summarize_2d_error(
    df,
    true_col,
    pred_col="pred",
    sim_col="ScafMaxSim",
    mdist_col="MDist",
    n_bins_sim=N_BINS_SIM,
    n_bins_mdist=N_BINS_MDIST,
    min_cell_n=MIN_CELL_N,
    use_trimmed_mean=False,
    trim=0.1,
):
    d = df[[true_col, pred_col, sim_col, mdist_col]].dropna().copy()
    d["abs_error"] = (d[pred_col] - d[true_col]).abs()

    # qcut returns categorical intervals
    d["sim_bin"] = pd.qcut(d[sim_col], q=n_bins_sim, duplicates="drop")
    d["mdist_bin"] = pd.qcut(d[mdist_col], q=n_bins_mdist, duplicates="drop")

    agg_fn = (lambda s: _trimmed_mean(s, trim=trim)) if use_trimmed_mean else "median"

    g = (
        d.groupby(["sim_bin", "mdist_bin"], observed=True)
        .agg(
            error=("abs_error", agg_fn),
            n=("abs_error", "size"),
        )
        .reset_index()
    )

    heat_error = g.pivot(index="mdist_bin", columns="sim_bin", values="error")
    heat_n = g.pivot(index="mdist_bin", columns="sim_bin", values="n").fillna(0)

    # mask sparse cells
    heat_error = heat_error.mask(heat_n < min_cell_n)
    return g, heat_error, heat_n


def plot_2d_error_heatmap(
    ax,
    heat_error,
    heat_n,
    title,
    cmap=HEATMAP_CMAP,
    vmin=None,
    vmax=None,
    tick_fmt="{:.3f}",
    cell_fontsize=None,
):
    base_font_size, ticks_font_size, title_font_size, label_font_size, subtitle_font_size = \
        determine_font_size()
    m = np.ma.masked_invalid(heat_error.values)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(MASK_COLOR)

    im = ax.imshow(m, aspect="auto", origin="lower", cmap=cmap_obj, vmin=vmin, vmax=vmax)

    # Use interval ranges as tick labels
    x_intervals = list(heat_error.columns)
    y_intervals = list(heat_error.index)
    x_labels = _format_interval_labels(x_intervals, fmt=tick_fmt)
    y_labels = _format_interval_labels(y_intervals, fmt=tick_fmt)

    ax.set_xticks(np.arange(heat_error.shape[1]))
    ax.set_yticks(np.arange(heat_error.shape[0]))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels)

    ax.tick_params(axis="both", labelsize=ticks_font_size)
    ax.yaxis.tick_left()
    ax.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False, pad=4)
    ax.set_xlabel("")
    ax.set_ylabel("Range of Mahalanobis distance", fontsize=label_font_size)
    ax.set_title(title, fontsize=title_font_size)

    # Cell count annotation (no 'n='), bigger font, color adapted to background
    if cell_fontsize is None:
        cell_fontsize = ticks_font_size  # bigger than before

    # For text color decision, normalize by vmin/vmax if available
    if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        # Fallback: always white text
        def _text_color(_val):
            return "white"
    else:
        def _text_color(val):
            # Use white on darker colors, black on brighter colors
            # Threshold at mid-point in normalized space
            t = (val - vmin) / (vmax - vmin)
            return "white" if t < 0.55 else "black"

    for i in range(heat_n.shape[0]):
        for j in range(heat_n.shape[1]):
            n_ij = int(heat_n.iloc[i, j]) if not np.isnan(heat_n.iloc[i, j]) else 0
            val = heat_error.iloc[i, j]
            if np.isnan(val):
                # Masked cell: use dark text on gray background
                color = "black"
            else:
                color = _text_color(float(val))
            ax.text(j, i, f"{n_ij}", ha="center", va="center", fontsize=cell_fontsize, color=color)

    return im

def _build_wide_df(which: str) -> pd.DataFrame:
    """Load base prediction + additional predictions and merge into a wide table."""
    base = load_csv(f'./results/mecap_ref_{which}_layer_0/predictions.csv').copy()
    base = base.rename(columns={"pred": "mmff"})

    xtb = load_csv(f'./results/mecap_ref_{which}_esnuel_orca_layer_0/predictions.csv')[["pred"]].rename(
        columns={"pred": "esnuel_orca"}
    )
    rmsd = load_csv(f'./results/mecap_ref_{which}_rmsd_layer_0/predictions.csv')[["pred"]].rename(
        columns={"pred": "rmsd"}
    )

    wide = pd.concat([base, xtb, rmsd], axis=1)
    return wide

def is_atom_in_pi_system(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetIsAromatic():
        return True
    return any(bond.GetIsConjugated() for bond in atom.GetBonds())
